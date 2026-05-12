#!/usr/bin/env python3

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as exc:
    print(f"Missing dependency: pandas ({exc})")
    sys.exit(2)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

VISIT_ORDER = ["M0", "M12", "M24", "M36", "M48", "M60"]
RUNS_OF_INTEREST = {"T1_01", "T1_02"}

DEFAULT_MANIFEST_CANDIDATES = [
    "/mnt/e/fyassine/ad-early-detection/DATA/_DELCODE/metadata/scan_dates/restingstate_scan_dates_M0_M60.xlsx",
    "/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__v2__/metadata/stratification/data/study_metadata/restingstate_scan_dates_M0_M60.xlsx",
]

DEFAULT_DEST_BASE = "/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__all__"
DEFAULT_LOG_FILE = "/mnt/e/fyassine/ad-early-detection/logs/transfer_resting_state.log"
DEFAULT_LRZ_POSTPROCESSED_BASE = "/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Delcode/data/Converter_newcriteria/postprocessed"
DEFAULT_SELECTION_ROOT = "/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Delcode/all_data/DELCODE_*"

SSH_OPTS = [
    "-o", "Compression=no",
    "-o", "ControlPersist=120",
    "-o", "TCPKeepAlive=yes",
    "-o", "ServerAliveInterval=30",
]


class Colors:
    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    CYAN = "\033[0;36m"
    RESET = "\033[0m"


@dataclass
class ManifestRow:
    pseudonym: str
    visit: str
    run: str
    scan_date: str
    source_zip: str


@dataclass
class RemoteNifti:
    path: str
    size: int
    pseudonym_norm: str
    visit: str
    run: Optional[str]


@dataclass
class TransferMapping:
    row: ManifestRow
    entry: Optional[RemoteNifti]
    status: str
    candidates: int


@dataclass
class CopyResult:
    status: str
    message: str
    bytes_copied: int
    destination: str


def is_resting_state_nifti(path: str) -> bool:
    lower = path.lower()
    if not lower.endswith(".nii.gz"):
        return False
    return "task-rest" in lower or "restingstate" in lower


def resting_state_score(path: str) -> int:
    lower = path.lower()
    score = 0

    if "task-rest" in lower:
        score += 50
    if "restingstate" in lower:
        score += 40
    if lower.endswith("_bold.nii.gz"):
        score += 30
    if "desc-sliced_bold" in lower:
        score += 25
    if "desc-preproc_bold" in lower:
        score += 20
    if "space-mni152nlin2009casym" in lower:
        score += 10

    penalties = [
        "brainmask",
        "mask",
        "dseg",
        "atlas",
        "parcell",
        "timeseries",
        "confound",
        "seg",
        "label",
        "roi",
    ]
    for token in penalties:
        if token in lower:
            score -= 30

    return score


def choose_best_candidate(candidates: List[RemoteNifti]) -> Tuple[RemoteNifti, str, int]:
    ranked = sorted(candidates, key=lambda item: (-resting_state_score(item.path), len(item.path), item.path))
    chosen = ranked[0]
    if len(ranked) == 1:
        return chosen, "unique", 1

    first_score = resting_state_score(ranked[0].path)
    second_score = resting_state_score(ranked[1].path)
    if first_score > second_score:
        return chosen, "resolved", len(ranked)

    return chosen, "ambiguous", len(ranked)


def q(value: str) -> str:
    return shlex.quote(value)


def default_manifest_path() -> str:
    for path in DEFAULT_MANIFEST_CANDIDATES:
        if Path(path).exists():
            return path
    return DEFAULT_MANIFEST_CANDIDATES[0]


def log_message(args: argparse.Namespace, message: str) -> None:
    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def print_status(args: argparse.Namespace, message: str, color: str = Colors.RESET) -> None:
    log_message(args, message)
    formatted = message if args.no_color else f"{color}{message}{Colors.RESET}"
    if HAS_TQDM:
        tqdm.write(formatted)
    else:
        print(formatted)


def normalize_pseudonym(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"^sub-", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    return text.lower()


def parse_remote_tokens(path: str, fallback_visit: str) -> Tuple[str, str, Optional[str]]:
    visit = fallback_visit
    visit_match = re.search(r"Postprocessed_(M\d+)", path, flags=re.IGNORECASE)
    if visit_match:
        visit = visit_match.group(1).upper()
    else:
        visit_match = re.search(r"(?<![A-Za-z0-9])(M(?:0|12|24|36|48|60))(?![A-Za-z0-9])", path, flags=re.IGNORECASE)
        if visit_match:
            visit = visit_match.group(1).upper()

    run = None
    run_match = re.search(r"(T\d+_\d+)", path, flags=re.IGNORECASE)
    if run_match:
        run = run_match.group(1).upper()

    pseudonym = ""
    sub_match = re.search(r"sub-([A-Za-z0-9]+)", path, flags=re.IGNORECASE)
    if sub_match:
        pseudonym = sub_match.group(1)
    else:
        hex_match = re.search(r"(?<![A-Za-z0-9])([0-9a-fA-F]{6,})(?![A-Za-z0-9])", path)
        if hex_match:
            pseudonym = hex_match.group(1)

    return normalize_pseudonym(pseudonym), visit, run


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_on_wunderlich(args: argparse.Namespace, remote_command: str, capture_output: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    cmd = [
        "ssh", "-S", args.local_socket,
        f"{args.wunderlich_user}@{args.wunderlich_host}",
        remote_command,
    ]
    return subprocess.run(cmd, capture_output=capture_output, text=text)


def run_on_lrz(args: argparse.Namespace, remote_command: str, capture_output: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    wrapped = f"ssh -S {q(args.remote_socket)} {q(args.lrz_user + '@' + args.lrz_host)} {q(remote_command)}"
    return run_on_wunderlich(args, wrapped, capture_output=capture_output, text=text)


def setup_local_master_connection(args: argparse.Namespace) -> None:
    print_status(args, "Connecting local machine to Wunderlich...", Colors.BLUE)
    subprocess.run(["rm", "-f", args.local_socket], capture_output=True)
    cmd = [
        "ssh", "-M", "-S", args.local_socket, "-fN",
        *SSH_OPTS,
        f"{args.wunderlich_user}@{args.wunderlich_host}",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Failed to connect to Wunderlich")
    print_status(args, "Local to Wunderlich connection is ready.", Colors.GREEN)


def setup_remote_master_connection(args: argparse.Namespace) -> None:
    print_status(args, "Connecting Wunderlich to LRZ (password and MFA may be requested once)...", Colors.BLUE)
    ssh_opts_str = " ".join(q(opt) for opt in SSH_OPTS)
    command = (
        f"rm -f {q(args.remote_socket)}; "
        f"ssh -M -S {q(args.remote_socket)} -fN {ssh_opts_str} {q(args.lrz_user + '@' + args.lrz_host)}"
    )
    result = subprocess.run(
        ["ssh", "-S", args.local_socket, "-t", f"{args.wunderlich_user}@{args.wunderlich_host}", command]
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to establish Wunderlich to LRZ connection")
    print_status(args, "Wunderlich to LRZ connection is ready.", Colors.GREEN)


def close_remote_master_connection(args: argparse.Namespace) -> None:
    run_on_wunderlich(
        args,
        f"ssh -S {q(args.remote_socket)} -O exit {q(args.lrz_user + '@' + args.lrz_host)}",
        capture_output=True,
        text=True,
    )


def close_local_master_connection(args: argparse.Namespace) -> None:
    subprocess.run(
        ["ssh", "-S", args.local_socket, "-O", "exit", f"{args.wunderlich_user}@{args.wunderlich_host}"],
        capture_output=True,
        text=True,
    )
    subprocess.run(["rm", "-f", args.local_socket], capture_output=True)


def load_manifest_rows(args: argparse.Namespace) -> List[ManifestRow]:
    path = Path(args.manifest_xlsx)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    excel = pd.ExcelFile(path)
    sheet = args.manifest_sheet
    if sheet:
        if sheet not in excel.sheet_names:
            raise ValueError(f"Sheet '{sheet}' is not available in {path}")
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        if "T1_01_T1_02" in excel.sheet_names:
            df = pd.read_excel(path, sheet_name="T1_01_T1_02")
        else:
            frames = []
            for run in ["T1_01", "T1_02"]:
                if run in excel.sheet_names:
                    frames.append(pd.read_excel(path, sheet_name=run))
            if not frames:
                raise ValueError("No usable sheet found in manifest")
            df = pd.concat(frames, ignore_index=True)

    for required in ["pseudonym", "visit", "run"]:
        if required not in df.columns:
            raise ValueError(f"Manifest column '{required}' is missing")

    if "scan_date" in df.columns:
        df = df[df["scan_date"].notna()].copy()

    df["pseudonym"] = df["pseudonym"].astype(str).str.strip()
    df["visit"] = df["visit"].astype(str).str.strip().str.upper()
    df["run"] = df["run"].astype(str).str.strip().str.upper()

    df = df[df["visit"].isin(VISIT_ORDER)]
    df = df[df["run"].isin(RUNS_OF_INTEREST)]

    if "source_zip" not in df.columns:
        df["source_zip"] = ""
    if "scan_date" not in df.columns:
        df["scan_date"] = ""

    df = df.sort_values(["pseudonym", "visit", "run"]).drop_duplicates(
        subset=["pseudonym", "visit", "run"], keep="first"
    )

    rows: List[ManifestRow] = []
    for _, row in df.iterrows():
        rows.append(
            ManifestRow(
                pseudonym=str(row["pseudonym"]).strip(),
                visit=str(row["visit"]).strip().upper(),
                run=str(row["run"]).strip().upper(),
                scan_date=str(row.get("scan_date", "")).strip(),
                source_zip=str(row.get("source_zip", "")).strip(),
            )
        )
    return rows


def fetch_remote_nifti_index(args: argparse.Namespace) -> List[RemoteNifti]:
    all_entries: List[RemoteNifti] = []
    skipped_without_subject = 0
    skipped_not_rest = 0

    for visit in VISIT_ORDER:
        directory = f"Postprocessed_{visit}"
        remote_dir = f"{args.lrz_postprocessed_base.rstrip('/')}/{directory}"
        remote_find = (
            f"if [ -d {q(remote_dir)} ]; then "
            f"find {q(remote_dir)} -type f -name '*.nii.gz' -printf '%p\\t%s\\n'; "
            f"fi"
        )
        result = run_on_lrz(args, remote_find, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to list remote files for {directory}: {result.stderr.strip()}")

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        for line in lines:
            path_part, size_part = (line.split("\t", 1) + ["0"])[:2]
            candidate_path = path_part.strip()
            if (not args.include_all_nifti) and (not is_resting_state_nifti(candidate_path)):
                skipped_not_rest += 1
                continue
            try:
                size_value = int(size_part.strip())
            except ValueError:
                size_value = 0
            pseudonym_norm, parsed_visit, parsed_run = parse_remote_tokens(candidate_path, visit)
            if not pseudonym_norm:
                skipped_without_subject += 1
                continue
            all_entries.append(
                RemoteNifti(
                    path=candidate_path,
                    size=size_value,
                    pseudonym_norm=pseudonym_norm,
                    visit=parsed_visit,
                    run=parsed_run,
                )
            )

    print_status(
        args,
        f"Indexed {len(all_entries)} remote NIfTI files from {args.lrz_postprocessed_base}. "
        f"Skipped {skipped_without_subject} files without pseudonym token and {skipped_not_rest} non-resting files.",
        Colors.CYAN,
    )
    return all_entries


def build_mappings(manifest_rows: List[ManifestRow], remote_entries: List[RemoteNifti]) -> List[TransferMapping]:
    exact: Dict[Tuple[str, str, str], List[RemoteNifti]] = {}
    by_visit: Dict[Tuple[str, str], List[RemoteNifti]] = {}

    for entry in remote_entries:
        by_visit.setdefault((entry.pseudonym_norm, entry.visit), []).append(entry)
        if entry.run:
            exact.setdefault((entry.pseudonym_norm, entry.visit, entry.run), []).append(entry)

    mappings: List[TransferMapping] = []
    for row in manifest_rows:
        key_subject = normalize_pseudonym(row.pseudonym)
        visit_key = row.visit
        run_key = row.run

        exact_candidates = exact.get((key_subject, visit_key, run_key), [])
        if exact_candidates:
            chosen, status, count = choose_best_candidate(exact_candidates)
            mappings.append(TransferMapping(row=row, entry=chosen, status=status, candidates=count))
            continue

        visit_candidates = by_visit.get((key_subject, visit_key), [])
        if not visit_candidates:
            mappings.append(TransferMapping(row=row, entry=None, status="missing", candidates=0))
            continue

        run_aligned = [item for item in visit_candidates if item.run == run_key]
        if len(run_aligned) == 1:
            mappings.append(TransferMapping(row=row, entry=run_aligned[0], status="resolved", candidates=1))
            continue
        if len(run_aligned) > 1:
            chosen, status, count = choose_best_candidate(run_aligned)
            mappings.append(TransferMapping(row=row, entry=chosen, status=status, candidates=count))
            continue

        chosen, status, count = choose_best_candidate(visit_candidates)
        mappings.append(TransferMapping(row=row, entry=chosen, status=status, candidates=count))

    return mappings


def write_mapping_report(args: argparse.Namespace, mappings: List[TransferMapping]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.mapping_report or f"/mnt/e/fyassine/ad-early-detection/logs/resting_state_mapping_{timestamp}.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for mapping in mappings:
        rows.append(
            {
                "pseudonym": mapping.row.pseudonym,
                "visit": mapping.row.visit,
                "run": mapping.row.run,
                "scan_date": mapping.row.scan_date,
                "source_zip": mapping.row.source_zip,
                "mapping_status": mapping.status,
                "candidate_count": mapping.candidates,
                "lrz_path": mapping.entry.path if mapping.entry else "",
                "file_size": mapping.entry.size if mapping.entry else 0,
            }
        )

    pd.DataFrame(rows).to_csv(report_path, index=False)
    return str(report_path)


def summarize_mappings(args: argparse.Namespace, mappings: List[TransferMapping]) -> None:
    status_counts: Dict[str, int] = {}
    for mapping in mappings:
        status_counts[mapping.status] = status_counts.get(mapping.status, 0) + 1
    ordered = ["unique", "resolved", "ambiguous", "missing"]
    parts = [f"{status}={status_counts.get(status, 0)}" for status in ordered]
    print_status(args, f"Mapping summary: {', '.join(parts)}", Colors.CYAN)


def select_mappings_for_copy(args: argparse.Namespace, mappings: List[TransferMapping]) -> List[TransferMapping]:
    allowed = {"unique", "resolved"}
    if args.allow_ambiguous:
        allowed.add("ambiguous")

    selected = [item for item in mappings if item.entry is not None and item.status in allowed]

    if args.pilot_one:
        selected = selected[:1]
    elif args.max_files is not None:
        selected = selected[: max(0, args.max_files)]

    return selected


def estimate_total_bytes(items: List[TransferMapping]) -> int:
    total = 0
    for item in items:
        if item.entry is not None:
            total += max(0, int(item.entry.size))
    return total


def copy_single_item(args: argparse.Namespace, item: TransferMapping) -> CopyResult:
    assert item.entry is not None

    filename = os.path.basename(item.entry.path)
    dest_dir = f"{args.dest_base.rstrip('/')}/{item.row.visit}/{item.row.pseudonym}/{item.row.run}"
    dest_file = f"{dest_dir}/{filename}"
    src_spec = f"{args.lrz_user}@{args.lrz_host}:{item.entry.path}"

    remote_script = (
        "set -e\n"
        f"mkdir -p {q(dest_dir)}\n"
        f"dest_file={q(dest_file)}\n"
        "if [ -s \"$dest_file\" ]; then\n"
        "  echo \"SKIPPED_EXISTS\t$dest_file\"\n"
        "  exit 0\n"
        "fi\n"
        "if command -v rsync >/dev/null 2>&1; then\n"
        f"  rsync -e \"ssh -S {args.remote_socket}\" {q(src_spec)} {q(dest_dir + '/')} >/dev/null 2>&1 || "
        f"scp -o ControlPath={q(args.remote_socket)} {q(src_spec)} {q(dest_dir + '/')}\n"
        "else\n"
        f"  scp -o ControlPath={q(args.remote_socket)} {q(src_spec)} {q(dest_dir + '/')}\n"
        "fi\n"
        "if [ ! -s \"$dest_file\" ]; then\n"
        "  echo \"COPY_FAILED\"\n"
        "  exit 1\n"
        "fi\n"
        "size=$(stat -c%s \"$dest_file\" 2>/dev/null || wc -c < \"$dest_file\")\n"
        "echo \"SUCCESS\t$dest_file\t$size\"\n"
    )

    result = run_on_wunderlich(args, remote_script, capture_output=True, text=True)
    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "copy failed"
        return CopyResult(status="failed", message=message[:400], bytes_copied=0, destination=dest_file)

    for line in stdout_lines:
        if line.startswith("SKIPPED_EXISTS\t"):
            destination = line.split("\t", 1)[1] if "\t" in line else dest_file
            return CopyResult(status="skipped", message="already exists", bytes_copied=0, destination=destination)
        if line.startswith("SUCCESS\t"):
            parts = line.split("\t")
            destination = parts[1] if len(parts) >= 2 else dest_file
            size_value = 0
            if len(parts) >= 3:
                try:
                    size_value = int(parts[2])
                except ValueError:
                    size_value = item.entry.size
            return CopyResult(status="success", message="copied", bytes_copied=size_value, destination=destination)

    return CopyResult(status="success", message="copied", bytes_copied=item.entry.size, destination=dest_file)


def print_dry_run_summary(args: argparse.Namespace, manifest_rows: List[ManifestRow], mappings: List[TransferMapping], selected: List[TransferMapping]) -> None:
    total_bytes = estimate_total_bytes(selected)
    print_status(args, f"Selection root reference: {args.selection_root}", Colors.CYAN)
    print_status(args, f"Manifest rows selected from notebook logic: {len(manifest_rows)}", Colors.CYAN)
    print_status(args, f"Rows eligible for copy with current policy: {len(selected)}", Colors.CYAN)
    print_status(args, f"Approximate transfer size: {total_bytes / (1024 ** 3):.2f} GiB", Colors.CYAN)
    preview_count = min(5, len(selected))
    for idx in range(preview_count):
        item = selected[idx]
        print_status(
            args,
            f"Preview {idx + 1}: {item.row.pseudonym} {item.row.visit} {item.row.run} -> {item.entry.path if item.entry else 'MISSING'}",
            Colors.CYAN,
        )


def run_copy_loop(args: argparse.Namespace, selected: List[TransferMapping]) -> Tuple[int, int, int, int]:
    successful = 0
    skipped = 0
    failed = 0
    bytes_copied = 0

    total = len(selected)
    start_all = time.monotonic()

    progress_bar = None
    if HAS_TQDM:
        progress_bar = tqdm(selected, total=total, unit="file", desc="Copying")
        iterator = progress_bar
    else:
        iterator = selected

    for idx, item in enumerate(iterator, start=1):
        start_one = time.monotonic()
        result = copy_single_item(args, item)
        elapsed_one = time.monotonic() - start_one

        if result.status == "success":
            successful += 1
            bytes_copied += max(0, result.bytes_copied)
            print_status(
                args,
                f"[{idx}/{total}] copied {item.row.pseudonym} {item.row.visit} {item.row.run} in {elapsed_one:.1f}s",
                Colors.GREEN,
            )
        elif result.status == "skipped":
            skipped += 1
            print_status(
                args,
                f"[{idx}/{total}] skipped {item.row.pseudonym} {item.row.visit} {item.row.run} ({result.message})",
                Colors.YELLOW,
            )
        else:
            failed += 1
            print_status(
                args,
                f"[{idx}/{total}] failed {item.row.pseudonym} {item.row.visit} {item.row.run}: {result.message}",
                Colors.RED,
            )

        elapsed_all = time.monotonic() - start_all
        avg = elapsed_all / max(1, idx)
        eta = avg * (total - idx)
        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"ok={successful} skip={skipped} fail={failed} eta={format_seconds(eta)}"
            )

    return successful, skipped, failed, bytes_copied


def spawn_background(args: argparse.Namespace) -> Tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_log = args.background_log or f"/mnt/e/fyassine/ad-early-detection/logs/transfer_resting_state_{timestamp}.log"
    Path(background_log).parent.mkdir(parents=True, exist_ok=True)

    passthrough_args: List[str] = []
    skip_next = False
    for token in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--background":
            continue
        if token == "--background-log":
            skip_next = True
            continue
        passthrough_args.append(token)

    if "--reuse-connections" not in passthrough_args:
        passthrough_args.append("--reuse-connections")
    if "--keep-connections" not in passthrough_args:
        passthrough_args.append("--keep-connections")

    cmd = [sys.executable, str(Path(__file__).resolve()), *passthrough_args]
    nohup_cmd = f"nohup {' '.join(q(part) for part in cmd)} >> {q(background_log)} 2>&1 & echo $!"
    result = subprocess.run(nohup_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to start background process")

    pid = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else "unknown"
    return pid, background_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy resting-state fMRI NIfTI files selected by notebook logic from LRZ using SSH tunnels."
    )

    parser.add_argument("--manifest-xlsx", default=default_manifest_path())
    parser.add_argument("--manifest-sheet", default="")
    parser.add_argument("--selection-root", default=DEFAULT_SELECTION_ROOT)

    parser.add_argument("--lrz-user", default="di54lup")
    parser.add_argument("--lrz-host", default="cool.hpc.lrz.de")
    parser.add_argument("--lrz-postprocessed-base", default=DEFAULT_LRZ_POSTPROCESSED_BASE)

    parser.add_argument("--wunderlich-user", default="wunderlich")
    parser.add_argument("--wunderlich-host", default="138.245.113.9")

    parser.add_argument("--dest-base", default=DEFAULT_DEST_BASE)
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    parser.add_argument("--mapping-report", default="")

    parser.add_argument("--local-socket", default="/tmp/wunderlich_master_socket")
    parser.add_argument("--remote-socket", default="/tmp/lrz_master_socket")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot-one", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--allow-ambiguous", action="store_true")
    parser.add_argument("--include-all-nifti", action="store_true")

    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--reuse-connections", action="store_true")
    parser.add_argument("--keep-connections", action="store_true")

    parser.add_argument("--background", action="store_true")
    parser.add_argument("--background-log", default="")

    parser.add_argument("--no-color", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_status(args, "Starting resting-state LRZ transfer workflow.", Colors.BLUE)
    print_status(args, f"Manifest: {args.manifest_xlsx}", Colors.BLUE)
    print_status(args, f"Destination root: {args.dest_base}", Colors.BLUE)

    try:
        if not args.reuse_connections:
            setup_local_master_connection(args)
            setup_remote_master_connection(args)

        if args.setup_only:
            print_status(args, "Setup complete. Exiting because --setup-only was requested.", Colors.GREEN)
            return 0

        if args.background:
            pid, bg_log = spawn_background(args)
            print_status(args, f"Background run started. PID={pid}", Colors.GREEN)
            print_status(args, f"Background log: {bg_log}", Colors.GREEN)
            print_status(args, "Use --reuse-connections for monitoring reruns while sockets stay active.", Colors.CYAN)
            return 0

        manifest_rows = load_manifest_rows(args)
        if not manifest_rows:
            print_status(args, "No manifest rows were selected after filtering.", Colors.YELLOW)
            return 0

        print_status(args, f"Loaded {len(manifest_rows)} manifest rows.", Colors.CYAN)

        remote_entries = fetch_remote_nifti_index(args)
        mappings = build_mappings(manifest_rows, remote_entries)
        summarize_mappings(args, mappings)

        report_path = write_mapping_report(args, mappings)
        print_status(args, f"Mapping report written: {report_path}", Colors.CYAN)

        selected = select_mappings_for_copy(args, mappings)
        if not selected:
            print_status(args, "No rows are eligible for copy with the current options.", Colors.YELLOW)
            return 0

        print_status(args, f"Rows selected for transfer: {len(selected)}", Colors.CYAN)

        if args.dry_run:
            print_dry_run_summary(args, manifest_rows, mappings, selected)
            print_status(args, "Dry-run completed. No files were copied.", Colors.GREEN)
            return 0

        successful, skipped, failed, bytes_copied = run_copy_loop(args, selected)
        print_status(args, "Transfer finished.", Colors.GREEN)
        print_status(args, f"Successful: {successful}", Colors.GREEN)
        print_status(args, f"Skipped: {skipped}", Colors.YELLOW if skipped else Colors.GREEN)
        print_status(args, f"Failed: {failed}", Colors.RED if failed else Colors.GREEN)
        print_status(args, f"Copied size: {bytes_copied / (1024 ** 3):.2f} GiB", Colors.CYAN)

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        print_status(args, "Interrupted by user.", Colors.RED)
        return 130
    except Exception as exc:
        print_status(args, f"Error: {type(exc).__name__}: {exc}", Colors.RED)
        return 1
    finally:
        if not args.keep_connections and not args.setup_only and not args.background:
            close_remote_master_connection(args)
            close_local_master_connection(args)
            print_status(args, "Closed SSH master connections.", Colors.GREEN)


if __name__ == "__main__":
    raise SystemExit(main())
