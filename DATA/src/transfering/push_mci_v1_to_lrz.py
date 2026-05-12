#!/usr/bin/env python3

import argparse
import csv
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

SOURCE_DIR_DEFAULT = "/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__v1__/fmri"
COHORT_CSV_DEFAULT = "/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__v1__/metadata/cohorts.csv"
DEST_ROOT_DEFAULT = "/dss/dsshome1/0A/di54lup/DELCODE/__mci_only__"
LRZ_USER_DEFAULT = "di54lup"
LRZ_HOST_DEFAULT = "cool.hpc.lrz.de"
SOCKET_PATH_DEFAULT = "/tmp/lrz_master_socket"
LOG_FILE_DEFAULT = "/mnt/e/fyassine/ad-early-detection/logs/push_mci_v1_to_lrz.log"

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


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp()}] {message}\n")


def print_status(args: argparse.Namespace, message: str, color: str = Colors.RESET) -> None:
    write_log(Path(args.log_file), message)
    if args.no_color:
        print(message)
    else:
        print(f"{color}{message}{Colors.RESET}")


def normalize_subject(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("sub-"):
        text = text[4:]
    return f"sub-{text.lower()}"


def load_mci_subjects(cohort_csv: Path, diagnosis_value: str) -> List[str]:
    if not cohort_csv.exists():
        raise FileNotFoundError(f"Cohort CSV not found: {cohort_csv}")

    target = diagnosis_value.strip().lower()
    subjects: Set[str] = set()

    with open(cohort_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"Pseudonym", "diagnosis"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Missing required columns in cohort CSV: {', '.join(sorted(missing))}")

        for row in reader:
            diagnosis = (row.get("diagnosis") or "").strip().lower()
            if diagnosis != target:
                continue
            subject = normalize_subject(row.get("Pseudonym") or "")
            if subject:
                subjects.add(subject)

    return sorted(subjects)


def split_existing_subjects(source_dir: Path, subject_folders: List[str]) -> Tuple[List[str], List[str]]:
    existing_dir_names = {
        child.name
        for child in source_dir.iterdir()
        if child.is_dir()
    }

    existing = [subject for subject in subject_folders if subject in existing_dir_names]
    missing = [subject for subject in subject_folders if subject not in existing_dir_names]
    return existing, missing


def write_manifest(path: Path, subjects: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for subject in subjects:
            handle.write(subject)
            handle.write("\n")


def run_command(cmd: List[str], capture_output: bool = False, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture_output, text=text)


def setup_master_connection(args: argparse.Namespace) -> None:
    print_status(args, "Establishing persistent LRZ connection (password and MFA required once)...", Colors.BLUE)
    run_command(["rm", "-f", args.socket_path], capture_output=True)
    cmd = [
        "ssh", "-M", "-S", args.socket_path, "-fN",
        *SSH_OPTS,
        f"{args.lrz_user}@{args.lrz_host}",
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError("Failed to establish LRZ master connection")
    print_status(args, "Persistent LRZ connection ready.", Colors.GREEN)


def check_master_connection(args: argparse.Namespace) -> None:
    cmd = [
        "ssh", "-S", args.socket_path, "-O", "check",
        f"{args.lrz_user}@{args.lrz_host}",
    ]
    result = run_command(cmd, capture_output=True)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"LRZ master connection is not available: {details}")


def close_master_connection(args: argparse.Namespace) -> None:
    cmd = [
        "ssh", "-S", args.socket_path, "-O", "exit",
        f"{args.lrz_user}@{args.lrz_host}",
    ]
    run_command(cmd, capture_output=True)
    run_command(["rm", "-f", args.socket_path], capture_output=True)
    print_status(args, "Closed LRZ master connection.", Colors.GREEN)


def ensure_remote_root(args: argparse.Namespace) -> None:
    remote_cmd = f"mkdir -p {shlex.quote(args.dest_root)}"
    cmd = [
        "ssh", "-S", args.socket_path,
        f"{args.lrz_user}@{args.lrz_host}",
        remote_cmd,
    ]
    result = run_command(cmd, capture_output=True)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Could not create destination root: {details}")


def subject_rsync_command(args: argparse.Namespace, subject: str) -> List[str]:
    source_subject = str(Path(args.source_dir) / subject) + "/"
    dest_subject = f"{args.lrz_user}@{args.lrz_host}:{args.dest_root.rstrip('/')}/{subject}/"

    cmd = [
        "rsync",
        "-a",
        "--human-readable",
        "--partial",
        "--append-verify",
        "--info=stats1,progress2",
    ]
    if args.dry_run:
        cmd.append("--dry-run")

    cmd.extend([
        "-e", f"ssh -S {args.socket_path}",
        source_subject,
        dest_subject,
    ])
    return cmd


def copy_subjects(args: argparse.Namespace, subjects: List[str]) -> Tuple[int, int]:
    total = len(subjects)
    success = 0
    failed = 0

    for idx, subject in enumerate(subjects, start=1):
        print_status(args, f"[{idx}/{total}] Syncing {subject}", Colors.CYAN)
        cmd = subject_rsync_command(args, subject)
        result = run_command(cmd, capture_output=True)
        if result.returncode == 0:
            success += 1
            print_status(args, f"[{idx}/{total}] Success {subject}", Colors.GREEN)
            continue

        failed += 1
        details = (result.stderr or result.stdout or "").strip()
        if len(details) > 300:
            details = details[:300] + "..."
        print_status(args, f"[{idx}/{total}] Failed {subject}: {details}", Colors.RED)

    return success, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push strict MCI subject folders from DELCODE v1 source to LRZ with one persistent SSH login."
    )
    parser.add_argument("--source-dir", default=SOURCE_DIR_DEFAULT)
    parser.add_argument("--cohort-csv", default=COHORT_CSV_DEFAULT)
    parser.add_argument("--dest-root", default=DEST_ROOT_DEFAULT)
    parser.add_argument("--diagnosis", default="mci")
    parser.add_argument("--lrz-user", default=LRZ_USER_DEFAULT)
    parser.add_argument("--lrz-host", default=LRZ_HOST_DEFAULT)
    parser.add_argument("--socket-path", default=SOCKET_PATH_DEFAULT)
    parser.add_argument("--log-file", default=LOG_FILE_DEFAULT)
    parser.add_argument("--manifest-out", default="")

    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--max-subjects", type=int, default=None)

    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--reuse-connection", action="store_true")
    parser.add_argument("--keep-connection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-color", action="store_true")

    return parser.parse_args()


def apply_subject_overrides(subjects: List[str], requested: List[str]) -> List[str]:
    if not requested:
        return subjects
    normalized_requested = {normalize_subject(value) for value in requested if normalize_subject(value)}
    return [subject for subject in subjects if subject in normalized_requested]


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    cohort_csv = Path(args.cohort_csv)

    if not source_dir.exists() or not source_dir.is_dir():
        print_status(args, f"Source directory not found: {source_dir}", Colors.RED)
        return 1

    try:
        print_status(args, "Building strict diagnosis subject list from cohort metadata.", Colors.BLUE)
        raw_subjects = load_mci_subjects(cohort_csv, args.diagnosis)
        selected_subjects, missing_subjects = split_existing_subjects(source_dir, raw_subjects)
        selected_subjects = apply_subject_overrides(selected_subjects, args.subject)

        if args.max_subjects is not None:
            selected_subjects = selected_subjects[: max(0, args.max_subjects)]

        manifest_out = Path(args.manifest_out) if args.manifest_out else Path(
            f"/mnt/e/fyassine/ad-early-detection/logs/mci_subject_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        write_manifest(manifest_out, selected_subjects)

        print_status(args, f"Diagnosis target: {args.diagnosis}", Colors.CYAN)
        print_status(args, f"Cohort subjects found: {len(raw_subjects)}", Colors.CYAN)
        print_status(args, f"Subjects found in source dir: {len(selected_subjects)}", Colors.CYAN)
        print_status(args, f"Subjects missing in source dir: {len(missing_subjects)}", Colors.CYAN)
        print_status(args, f"Manifest file: {manifest_out}", Colors.CYAN)

        if not selected_subjects:
            print_status(args, "No subject folders selected for transfer.", Colors.YELLOW)
            return 0

        if args.manifest_only:
            print_status(args, "Manifest-only mode completed.", Colors.GREEN)
            return 0

        if not args.reuse_connection:
            setup_master_connection(args)
        else:
            print_status(args, "Reusing existing LRZ master connection.", Colors.BLUE)

        check_master_connection(args)

        if args.setup_only:
            print_status(args, "Setup-only mode completed.", Colors.GREEN)
            return 0

        ensure_remote_root(args)

        mode_text = "dry-run" if args.dry_run else "copy"
        print_status(args, f"Starting {mode_text} for {len(selected_subjects)} subject folders.", Colors.BLUE)
        success, failed = copy_subjects(args, selected_subjects)

        print_status(args, f"Completed. Success: {success}", Colors.GREEN)
        print_status(args, f"Completed. Failed: {failed}", Colors.RED if failed else Colors.GREEN)

        return 0 if failed == 0 else 1
    except KeyboardInterrupt:
        print_status(args, "Interrupted by user.", Colors.RED)
        return 130
    except Exception as exc:
        print_status(args, f"Error: {type(exc).__name__}: {exc}", Colors.RED)
        return 1
    finally:
        if not args.keep_connection and not args.setup_only and not args.manifest_only and not args.reuse_connection:
            close_master_connection(args)


if __name__ == "__main__":
    raise SystemExit(main())