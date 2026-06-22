#!/usr/bin/env python3

import argparse
import csv
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
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

DEFAULT_LRZ_NONCONVERTER_BASE = (
    "/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Delcode/data/non-converter/postprocessed"
)
DEFAULT_DEST_BASE = "/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__fmri_wholebrain_sch200_flat__/fmri"
DEFAULT_LOG_FILE = "/mnt/e/fyassine/ad-early-detection/logs/transfer_non_converter_resting_state.log"

SSH_OPTS = [
    "-o",
    "Compression=no",
    "-o",
    "ControlPersist=120",
    "-o",
    "TCPKeepAlive=yes",
    "-o",
    "ServerAliveInterval=30",
]


class Colors:
    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    CYAN = "\033[0;36m"
    RESET = "\033[0m"


@dataclass
class RemoteFile:
    path: str
    size: int
    subject: str
    visit: str
    session: str


@dataclass
class FileGroup:
    subject: str
    visit: str
    session: str
    entries: List[RemoteFile]
    status: str
    chosen: Optional[RemoteFile]


@dataclass
class CopyResult:
    status: str
    message: str
    bytes_copied: int
    destination: str


def q(value: str) -> str:
    return shlex.quote(value)


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


def normalize_subject(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"^sub-", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    return text.lower()


def parse_session(path: str) -> str:
    match = re.search(r"ses-([A-Za-z0-9]+)", path, flags=re.IGNORECASE)
    if match:
        return f"ses-{match.group(1)}"
    return ""


def build_destination_filename(filename: str, visit: str) -> str:
    if not visit:
        return filename
    if f"_{visit}_" in filename:
        return filename
    session_match = re.search(r"(ses-[A-Za-z0-9]+)", filename, flags=re.IGNORECASE)
    if session_match:
        token = session_match.group(1)
        return filename.replace(token, f"{token}_{visit}", 1)
    subject_match = re.search(r"(sub-[A-Za-z0-9]+)", filename, flags=re.IGNORECASE)
    if subject_match:
        token = subject_match.group(1)
        return filename.replace(token, f"{token}_{visit}", 1)
    return f"{visit}_{filename}"


def get_ssh_opts(args: argparse.Namespace) -> List[str]:
    opts = list(SSH_OPTS)
    if args.non_interactive:
        opts.extend(["-o", "BatchMode=yes"])
    if args.ssh_key:
        opts.extend(["-i", args.ssh_key])
    if args.ssh_extra_opts:
        for opt in args.ssh_extra_opts.split(","):
            if opt.strip():
                opts.extend(["-o", opt.strip()])
    return opts


def run_on_wunderlich(
    args: argparse.Namespace, remote_command: str, capture_output: bool = True, text: bool = True
) -> subprocess.CompletedProcess:
    cmd = [
        "ssh",
        "-S",
        args.local_socket,
        f"{args.wunderlich_user}@{args.wunderlich_host}",
        remote_command,
    ]
    return subprocess.run(cmd, capture_output=capture_output, text=text)


def run_on_lrz(
    args: argparse.Namespace, remote_command: str, capture_output: bool = True, text: bool = True
) -> subprocess.CompletedProcess:
    wrapped = (
        f"ssh -S {q(args.remote_socket)} "
        f"{q(args.lrz_user + '@' + args.lrz_host)} {q(remote_command)}"
    )
    return run_on_wunderlich(args, wrapped, capture_output=capture_output, text=text)


def setup_local_master_connection(args: argparse.Namespace) -> None:
    print_status(args, "Connecting local machine to Wunderlich...", Colors.BLUE)
    subprocess.run(["rm", "-f", args.local_socket], capture_output=True)
    cmd = [
        "ssh",
        "-M",
        "-S",
        args.local_socket,
        "-fN",
        *get_ssh_opts(args),
        f"{args.wunderlich_user}@{args.wunderlich_host}",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Failed to connect to Wunderlich. Authentication may have failed or timed out.")
    print_status(args, "Local to Wunderlich connection is ready.", Colors.GREEN)


def setup_remote_master_connection(args: argparse.Namespace) -> None:
    print_status(args, "Connecting Wunderlich to LRZ (password and MFA may be requested once)...", Colors.BLUE)
    ssh_opts_str = " ".join(q(opt) for opt in get_ssh_opts(args))
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


def check_ssh_sockets(args: argparse.Namespace) -> None:
    print_status(args, "Validating SSH master sockets before transfer...", Colors.BLUE)

    # Local check
    local_cmd = [
        "ssh", "-S", args.local_socket, "-O", "check", f"{args.wunderlich_user}@{args.wunderlich_host}"
    ]
    res_local = subprocess.run(local_cmd, capture_output=True, text=True)
    if res_local.returncode != 0:
        raise RuntimeError(f"Local SSH socket missing or inactive: {res_local.stderr.strip()}")

    # Remote check
    remote_check_cmd = f"ssh -S {q(args.remote_socket)} -O check {q(args.lrz_user + '@' + args.lrz_host)}"
    res_remote = run_on_wunderlich(args, remote_check_cmd, capture_output=True, text=True)
    if res_remote.returncode != 0:
        raise RuntimeError(f"Remote SSH socket missing or inactive: {res_remote.stderr.strip()}")

    print_status(args, "SSH master sockets are active and healthy.", Colors.GREEN)


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


def fetch_visit_dirs(args: argparse.Namespace) -> List[str]:
    remote_cmd = (
        f"if [ -d {q(args.lrz_base.rstrip('/'))} ]; then "
        f"find {q(args.lrz_base.rstrip('/'))} -maxdepth 1 -type d -name 'M*' -printf '%f\\n'; "
        f"fi"
    )
    result = run_on_lrz(args, remote_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to list visit directories")
    visits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not visits:
        raise RuntimeError(f"No visit directories found under {args.lrz_base}")
    return sorted(set(visits))


def fetch_remote_files(args: argparse.Namespace, visits: List[str]) -> List[RemoteFile]:
    entries: List[RemoteFile] = []
    skipped_without_subject = 0

    for visit in visits:
        visit_dir = f"{args.lrz_base.rstrip('/')}/{visit}"
        remote_find = (
            f"if [ -d {q(visit_dir)} ]; then "
            f"find {q(visit_dir)} -type f -iname '*bold_reoriented.nii.gz' -printf '%p\\t%s\\n'; "
            f"fi"
        )
        result = run_on_lrz(args, remote_find, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Failed to list files for {visit}")

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        for line in lines:
            path_part, size_part = (line.split("\t", 1) + ["0"])[:2]
            path = path_part.strip()
            try:
                size_value = int(size_part.strip())
            except ValueError:
                size_value = 0

            sub_match = re.search(r"sub-([A-Za-z0-9]+)", path, flags=re.IGNORECASE)
            if not sub_match:
                skipped_without_subject += 1
                continue
            subject = normalize_subject(sub_match.group(1))
            session = parse_session(path)

            entries.append(
                RemoteFile(
                    path=path,
                    size=size_value,
                    subject=subject,
                    visit=visit,
                    session=session,
                )
            )

    print_status(
        args,
        f"Indexed {len(entries)} files from {args.lrz_base}. Skipped {skipped_without_subject} without subject token.",
        Colors.CYAN,
    )
    return entries


def build_groups(entries: List[RemoteFile]) -> List[FileGroup]:
    grouped: Dict[Tuple[str, str, str], List[RemoteFile]] = {}
    for entry in entries:
        key = (entry.subject, entry.visit, entry.session)
        grouped.setdefault(key, []).append(entry)

    groups: List[FileGroup] = []
    for (subject, visit, session), items in grouped.items():
        if len(items) == 1:
            status = "unique"
            chosen = items[0]
        else:
            status = "duplicate"
            chosen = None
        groups.append(
            FileGroup(
                subject=subject,
                visit=visit,
                session=session,
                entries=items,
                status=status,
                chosen=chosen,
            )
        )

    groups.sort(key=lambda g: (g.subject, g.visit, g.session))
    return groups


def write_mapping_report(args: argparse.Namespace, groups: List[FileGroup]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(
        args.mapping_report
        or f"/mnt/e/fyassine/ad-early-detection/logs/non_converter_mapping_{timestamp}.csv"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "visit",
                "session",
                "status",
                "candidate_count",
                "lrz_path",
                "file_size",
                "destination",
            ],
        )
        writer.writeheader()
        for group in groups:
            subject_label = f"sub-{group.subject}"
            chosen_path = group.chosen.path if group.chosen else ""
            chosen_size = group.chosen.size if group.chosen else 0
            dest_filename = (
                build_destination_filename(os.path.basename(chosen_path), group.visit)
                if chosen_path
                else ""
            )
            destination = (
                f"{args.dest_base.rstrip('/')}/{subject_label}/{dest_filename}"
                if dest_filename
                else ""
            )
            writer.writerow(
                {
                    "subject": subject_label,
                    "visit": group.visit,
                    "session": group.session,
                    "status": group.status,
                    "candidate_count": len(group.entries),
                    "lrz_path": chosen_path,
                    "file_size": chosen_size,
                    "destination": destination,
                }
            )

    return str(report_path)


def write_duplicate_report(args: argparse.Namespace, groups: List[FileGroup]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(
        args.duplicate_report
        or f"/mnt/e/fyassine/ad-early-detection/logs/non_converter_duplicates_{timestamp}.csv"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "visit", "session", "lrz_path", "file_size"],
        )
        writer.writeheader()
        for group in groups:
            if group.status != "duplicate":
                continue
            subject_label = f"sub-{group.subject}"
            for entry in group.entries:
                writer.writerow(
                    {
                        "subject": subject_label,
                        "visit": group.visit,
                        "session": group.session,
                        "lrz_path": entry.path,
                        "file_size": entry.size,
                    }
                )

    return str(report_path)


def summarize_groups(args: argparse.Namespace, groups: List[FileGroup]) -> None:
    counts: Dict[str, int] = {"unique": 0, "duplicate": 0}
    for group in groups:
        counts[group.status] = counts.get(group.status, 0) + 1
    print_status(
        args,
        f"Mapping summary: unique={counts.get('unique', 0)}, duplicate={counts.get('duplicate', 0)}",
        Colors.CYAN,
    )


def select_groups_for_copy(args: argparse.Namespace, groups: List[FileGroup]) -> List[FileGroup]:
    selected = [group for group in groups if group.status == "unique" and group.chosen is not None]

    if args.pilot_one:
        selected = selected[:1]
    elif args.max_files is not None:
        selected = selected[: max(0, args.max_files)]

    return selected


def estimate_total_bytes(groups: List[FileGroup]) -> int:
    total = 0
    for group in groups:
        if group.chosen is not None:
            total += max(0, int(group.chosen.size))
    return total


def copy_single_item(args: argparse.Namespace, group: FileGroup) -> CopyResult:
    assert group.chosen is not None

    filename = build_destination_filename(os.path.basename(group.chosen.path), group.visit)
    subject_label = f"sub-{group.subject}"
    dest_dir = f"{args.dest_base.rstrip('/')}/{subject_label}"
    dest_file = f"{dest_dir}/{filename}"
    src_spec = f"{args.lrz_user}@{args.lrz_host}:{group.chosen.path}"

    scp_opts = f"-o ControlPath={q(args.remote_socket)}"
    if args.non_interactive:
        scp_opts += " -o BatchMode=yes"
        ssh_opts_rsync = f"ssh -S {args.remote_socket} -o BatchMode=yes"
    else:
        ssh_opts_rsync = f"ssh -S {args.remote_socket}"

    remote_script = (
        "set -e\n"
        f"mkdir -p {q(dest_dir)}\n"
        f"dest_file={q(dest_file)}\n"
        "if [ -s \"$dest_file\" ]; then\n"
        "  echo \"SKIPPED_EXISTS\t$dest_file\"\n"
        "  exit 0\n"
        "fi\n"
        "if command -v rsync >/dev/null 2>&1; then\n"
        f"  rsync -e {q(ssh_opts_rsync)} {q(src_spec)} \"$dest_file\" >/dev/null 2>&1 || "
        f"scp {scp_opts} {q(src_spec)} \"$dest_file\"\n"
        "else\n"
        f"  scp {scp_opts} {q(src_spec)} \"$dest_file\"\n"
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
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        combined_err = f"{stderr} {stdout}".strip()

        # Extract concise failure reasons
        if "Permission denied" in combined_err:
            reason = "Permission denied"
        elif "No such file" in combined_err:
            reason = "No such file or directory"
        elif "Connection timed out" in combined_err or "Connection refused" in combined_err:
            reason = "Connection failed"
        elif "mux_client_request_session" in combined_err or "master failed" in combined_err:
            reason = "SSH Master socket dropped"
        else:
            reason = combined_err[:100] if combined_err else "Unknown copy error"

        return CopyResult(status="failed", message=reason, bytes_copied=0, destination=dest_file)

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
                    size_value = group.chosen.size
            return CopyResult(status="success", message="copied", bytes_copied=size_value, destination=destination)

    return CopyResult(status="success", message="copied", bytes_copied=group.chosen.size, destination=dest_file)


def print_dry_run_summary(
    args: argparse.Namespace, groups: List[FileGroup], selected: List[FileGroup]
) -> None:
    total_bytes = estimate_total_bytes(selected)
    print_status(args, f"Source base: {args.lrz_base}", Colors.CYAN)
    print_status(args, f"Groups found: {len(groups)}", Colors.CYAN)
    print_status(args, f"Groups eligible for copy: {len(selected)}", Colors.CYAN)
    print_status(args, f"Approximate transfer size: {total_bytes / (1024 ** 3):.2f} GiB", Colors.CYAN)
    preview_count = min(5, len(selected))
    for idx in range(preview_count):
        group = selected[idx]
        print_status(
            args,
            f"Preview {idx + 1}: sub-{group.subject} {group.visit} {group.session} -> {group.chosen.path}",
            Colors.CYAN,
        )


def run_copy_loop(args: argparse.Namespace, selected: List[FileGroup]) -> Tuple[int, int, int, int]:
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

    for idx, group in enumerate(iterator, start=1):
        start_one = time.monotonic()
        result = copy_single_item(args, group)
        elapsed_one = time.monotonic() - start_one

        if result.status == "success":
            successful += 1
            bytes_copied += max(0, result.bytes_copied)
            print_status(
                args,
                f"[{idx}/{total}] copied sub-{group.subject} {group.visit} {group.session} in {elapsed_one:.1f}s",
                Colors.GREEN,
            )
        elif result.status == "skipped":
            skipped += 1
            print_status(
                args,
                f"[{idx}/{total}] skipped sub-{group.subject} {group.visit} {group.session} ({result.message})",
                Colors.YELLOW,
            )
        else:
            failed += 1
            print_status(
                args,
                f"[{idx}/{total}] failed sub-{group.subject} {group.visit} {group.session}: {result.message} (file: {group.chosen.path})",
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


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def spawn_background(args: argparse.Namespace) -> Tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_log = (
        args.background_log
        or f"/mnt/e/fyassine/ad-early-detection/logs/transfer_non_converter_{timestamp}.log"
    )
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
    if "--non-interactive" not in passthrough_args:
        passthrough_args.append("--non-interactive")

    cmd = [sys.executable, str(Path(__file__).resolve()), *passthrough_args]
    nohup_cmd = f"nohup {' '.join(q(part) for part in cmd)} >> {q(background_log)} 2>&1 & echo $!"
    result = subprocess.run(nohup_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Failed to start background process")

    pid = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else "unknown"
    return pid, background_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy non-converter resting-state NIfTI files from LRZ using SSH tunnels."
    )

    parser.add_argument("--lrz-base", default=DEFAULT_LRZ_NONCONVERTER_BASE)
    parser.add_argument("--lrz-user", default="di54lup")
    parser.add_argument("--lrz-host", default="cool.hpc.lrz.de")

    parser.add_argument("--wunderlich-user", default="wunderlich")
    parser.add_argument("--wunderlich-host", default="138.245.113.9")

    parser.add_argument("--dest-base", default=DEFAULT_DEST_BASE)
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    parser.add_argument("--mapping-report", default="")
    parser.add_argument("--duplicate-report", default="")

    parser.add_argument("--local-socket", default="/tmp/wunderlich_master_socket")
    parser.add_argument("--remote-socket", default="/tmp/lrz_master_socket")

    # Auth options
    parser.add_argument("--ssh-key", default="", help="Optional SSH key path for authentication")
    parser.add_argument("--ssh-extra-opts", default="", help="Comma-separated extra SSH options (e.g. ForwardAgent=yes)")
    parser.add_argument("--non-interactive", action="store_true", help="Enable BatchMode=yes to prevent hanging on auth prompts")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot-one", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)

    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--reuse-connections", action="store_true")
    parser.add_argument("--keep-connections", action="store_true")

    parser.add_argument("--background", action="store_true")
    parser.add_argument("--background-log", default="")

    parser.add_argument("--no-color", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print_status(args, "Starting non-converter LRZ transfer workflow.", Colors.BLUE)
    print_status(args, f"Source base: {args.lrz_base}", Colors.BLUE)
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

        # Enforce SSH health check before attempting to fetch or copy files
        check_ssh_sockets(args)

        visits = fetch_visit_dirs(args)
        print_status(args, f"Visit directories: {', '.join(visits)}", Colors.CYAN)

        entries = fetch_remote_files(args, visits)
        if not entries:
            print_status(args, "No matching files were found.", Colors.YELLOW)
            return 0

        groups = build_groups(entries)
        summarize_groups(args, groups)

        mapping_report = write_mapping_report(args, groups)
        print_status(args, f"Mapping report written: {mapping_report}", Colors.CYAN)

        duplicate_report = write_duplicate_report(args, groups)
        print_status(args, f"Duplicate report written: {duplicate_report}", Colors.CYAN)

        selected = select_groups_for_copy(args, groups)
        if not selected:
            print_status(args, "No files are eligible for copy (duplicates are skipped).", Colors.YELLOW)
            return 0

        print_status(args, f"Files selected for transfer: {len(selected)}", Colors.CYAN)

        if args.dry_run:
            print_dry_run_summary(args, groups, selected)
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
