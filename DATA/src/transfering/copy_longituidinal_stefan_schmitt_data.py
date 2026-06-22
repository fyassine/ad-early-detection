#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install 'tqdm' for progress bars: pip install tqdm")

# --- CONFIGURATION ---
LRZ_USER = "di54lup"
LRZ_HOST = "cool.hpc.lrz.de"
LRZ_BASE = "/dss/dssfs03/pn72zi/pn72zi-dss-0001/di38jor/Projects/Delcode/data/Converter_newcriteria/postprocessed/"

WUNDERLICH_USER = "wunderlich"
WUNDERLICH_HOST = "138.245.113.9"
WUNDERLICH_BASE = "/mnt/e/fyassine/+ad-early-detection/data/DELCODE/fmri/longitudinal/unsorted"

# Log file for background monitoring
LOG_FILE = "/mnt/e/fyassine/+ad-early-detection/logs/copy_longitudinal.log"

# Socket paths
LOCAL_SOCKET = "/tmp/wunderlich_master_socket"
REMOTE_SOCKET = "/tmp/lrz_master_socket"

# SSH optimization options
SSH_OPTS = [
    "-o", "Compression=no",
    "-o", "ControlPersist=120",
    "-o", "TCPKeepAlive=yes",
    "-o", "ServerAliveInterval=30",
]

DIRECTORIES = [
    "Postprocessed_M0",
    "Postprocessed_M12",
    "Postprocessed_M24",
    "Postprocessed_M36",
    "Postprocessed_M48",
    "Postprocessed_M60"
]

class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    RESET = '\033[0m'

def print_colored(message: str, color: str = Colors.RESET):
    formatted_msg = f"{color}{message}{Colors.RESET}"
    log_message(message)
    if HAS_TQDM:
        tqdm.write(formatted_msg)
    else:
        print(formatted_msg)

def log_message(message: str):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def setup_local_master_connection():
    """Establishes persistent SSH connection from local machine to Wunderlich."""
    print_colored("\n========================================", Colors.BLUE)
    print_colored("Step 1a: Connecting to Wunderlich", Colors.BLUE)
    print_colored("========================================", Colors.BLUE)
    log_message("Prompting for Wunderlich password")
    print("Please enter your Wunderlich password:")

    subprocess.run(["rm", "-f", LOCAL_SOCKET], capture_output=True)

    cmd = [
        "ssh", "-M", "-S", LOCAL_SOCKET, "-fN",
        *SSH_OPTS,
        f"{WUNDERLICH_USER}@{WUNDERLICH_HOST}"
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print_colored("\n✗ Failed to connect to Wunderlich. Script aborted.", Colors.RED)
        sys.exit(1)

    print_colored("✓ Connected to Wunderlich!", Colors.GREEN)

def setup_remote_master_connection():
    """Creates Wunderlich -> LRZ connection (password and 2FA entered once)."""
    print_colored("\n========================================", Colors.BLUE)
    print_colored("Step 1b: Establishing Wunderlich -> LRZ Link", Colors.BLUE)
    print_colored("========================================", Colors.BLUE)
    log_message("Prompting for LRZ password and MFA")
    print("You will see the LRZ password prompt below.")
    print("Please enter your LRZ Password and 2FA codes now.")
    print_colored("This will only happen ONCE.", Colors.YELLOW)

    ssh_opts_str = " ".join([f"'{opt}'" for opt in SSH_OPTS])
    setup_cmd = [
        "ssh", "-S", LOCAL_SOCKET, "-t", f"{WUNDERLICH_USER}@{WUNDERLICH_HOST}",
        f"rm -f {REMOTE_SOCKET}; ssh -M -S {REMOTE_SOCKET} -fN {ssh_opts_str} {LRZ_USER}@{LRZ_HOST}"
    ]

    result = subprocess.run(setup_cmd)

    if result.returncode != 0:
        print_colored("\n✗ Failed to establish LRZ link. Script aborted.", Colors.RED)
        close_local_master_connection()
        sys.exit(1)

    print_colored("\n✓ Persistent link to LRZ established!", Colors.GREEN)
    print_colored("\n========================================", Colors.GREEN)
    print_colored("All connections ready! No more passwords needed.", Colors.GREEN)
    print_colored("========================================\n", Colors.GREEN)

def close_remote_master_connection():
    """Tells Wunderlich to close the connection to LRZ."""
    print_colored("\nClosing LRZ connection...", Colors.YELLOW)
    cmd = [
        "ssh", "-S", LOCAL_SOCKET, f"{WUNDERLICH_USER}@{WUNDERLICH_HOST}",
        f"ssh -S {REMOTE_SOCKET} -O exit {LRZ_USER}@{LRZ_HOST}"
    ]
    subprocess.run(cmd, capture_output=True)
    print_colored("✓ LRZ connection closed", Colors.GREEN)

def close_local_master_connection():
    """Closes the local -> Wunderlich connection."""
    print_colored("Closing Wunderlich connection...", Colors.YELLOW)
    cmd = ["ssh", "-S", LOCAL_SOCKET, "-O", "exit", f"{WUNDERLICH_USER}@{WUNDERLICH_HOST}"]
    subprocess.run(cmd, capture_output=True)
    subprocess.run(["rm", "-f", LOCAL_SOCKET], capture_output=True)
    print_colored("✓ All connections closed\n", Colors.GREEN)

def get_lrz_file_list(directory: str) -> List[str]:
    """Gets list of files using the ESTABLISHED connections."""
    # Find all .nii.gz files under the LRZ directory tree for this processing directory
    remote_find = f"find {LRZ_BASE}/{directory} -type f -name '*.nii.gz'"
    cmd = [
        "ssh", "-S", LOCAL_SOCKET, f"{WUNDERLICH_USER}@{WUNDERLICH_HOST}",
        f"ssh -S {REMOTE_SOCKET} {LRZ_USER}@{LRZ_HOST} \"{remote_find}\""
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [f for f in result.stdout.strip().split('\n') if f]

def get_existing_subjects(directory: str) -> set:
    """Get set of already downloaded subject IDs from Wunderlich for a specific directory."""
    cmd = [
        "ssh", "-S", LOCAL_SOCKET, f"{WUNDERLICH_USER}@{WUNDERLICH_HOST}",
        f"ls -1 '{WUNDERLICH_BASE}/{directory}' 2>/dev/null || echo ''"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return set()
    return set(result.stdout.strip().split('\n'))

def check_subject_has_files(directory: str, subject_id: str) -> bool:
    """Check if a subject folder actually contains NIfTI files."""
    cmd = [
        "ssh", "-S", LOCAL_SOCKET, f"{WUNDERLICH_USER}@{WUNDERLICH_HOST}",
        f"ls '{WUNDERLICH_BASE}/{directory}/{subject_id}'/*.nii.gz 2>/dev/null | head -1"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() != ""

def process_file_on_wunderlich(lrz_path: str, directory: str) -> Tuple[bool, str]:
    """
    Copies a single .nii.gz from LRZ to the Wunderlich host preserving session
    structure and placing files under WUNDERLICH_BASE/<directory>/<subject_dir>/.

    The function extracts a subject token like 'sub-XXXX' from the LRZ path when
    possible; if none is found it will use the base filename prefix and ensure
    the destination folder name is prefixed with 'sub-'.
    """
    # Try to extract a 'sub-...' token from the LRZ path
    import re
    m = re.search(r"/(sub-[^/]+)/", lrz_path)
    if m:
        subject_dir = m.group(1)
    else:
        # Fallback: use filename prefix (before first '-') and ensure 'sub-' prefix
        fname = os.path.basename(lrz_path)
        base = fname.split('-')[0]
        if base.startswith('sub'):
            subject_dir = base
        else:
            subject_dir = f"sub-{base}"

    dest_path = f"{WUNDERLICH_BASE}/{directory}/{subject_dir}"

    # Remote script executed on Wunderlich will pull the single file from LRZ
    remote_script = f"""
    set -e

    # Skip if already exists and has .nii.gz files
    if [ -d '{dest_path}' ] && ls '{dest_path}'/*.nii.gz 1>/dev/null 2>&1; then
        echo "ALREADY_EXISTS"
        exit 0
    fi

    mkdir -p '{dest_path}/'

    echo "Copying {lrz_path} to {dest_path}..."
    if command -v rsync &> /dev/null; then
        rsync -e "ssh -S {REMOTE_SOCKET}" '{LRZ_USER}@{LRZ_HOST}:{lrz_path}' '{dest_path}/' 2>/dev/null || \
        scp -o 'ControlPath={REMOTE_SOCKET}' '{LRZ_USER}@{LRZ_HOST}:{lrz_path}' '{dest_path}/'
    else
        scp -o 'ControlPath={REMOTE_SOCKET}' '{LRZ_USER}@{LRZ_HOST}:{lrz_path}' '{dest_path}/'
    fi

    FILE_COUNT=$(ls '{dest_path}'/*.nii.gz 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -eq 0 ]; then
        echo "COPY_FAILED"
        rmdir '{dest_path}' 2>/dev/null || true
        exit 1
    fi

    echo "SUCCESS: Copied $FILE_COUNT NIfTI files"
    """

    result = subprocess.run(
        ["ssh", "-S", LOCAL_SOCKET, f"{WUNDERLICH_USER}@{WUNDERLICH_HOST}", remote_script],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        if "COPY_FAILED" in result.stdout:
            return False, "Failed to copy NIfTI files"
        return False, result.stderr.strip()[:200]

    if "ALREADY_EXISTS" in result.stdout:
        return True, "Skipped (already exists)"

    for line in result.stdout.split('\n'):
        if "SUCCESS:" in line:
            return True, line.replace("SUCCESS:", "").strip()

    return True, "Success"

def process_files_sequential(files: List[str], existing_subjects: set, directory: str) -> Tuple[int, int, int]:
    """Process files one at a time (no parallelism for stability)."""
    # Pre-filter already downloaded subjects
    files_to_process = []
    skipped = 0

    for zip_path in files:
        # files are LRZ paths to .nii.gz; try to extract subject dir token
        import re
        m = re.search(r"/(sub-[^/]+)/", zip_path)
        if m:
            subject_dir = m.group(1)
        else:
            filename = os.path.basename(zip_path)
            subject_dir = filename.split('-')[0]
            if not subject_dir.startswith('sub'):
                subject_dir = f"sub-{subject_dir}"

        if subject_dir in existing_subjects:
            if check_subject_has_files(directory, subject_dir):
                skipped += 1
                continue

        files_to_process.append(zip_path)

    if not files_to_process:
        return 0, 0, skipped

    successful = 0
    failed = 0

    total_files = len(files_to_process)
    print_colored(f"  Starting copy of {total_files} NIfTI files...", Colors.CYAN)

    # Process sequentially with progress bar
    if HAS_TQDM:
        iterator = tqdm(files_to_process, unit="file", desc=f"  {directory}")
    else:
        iterator = files_to_process

    for idx, zip_file in enumerate(iterator, start=1):
        print_colored(
            f"  [{idx}/{total_files}] Copying {os.path.basename(zip_file)}",
            Colors.CYAN
        )
        success, msg = process_file_on_wunderlich(zip_file, directory)

        if success:
            successful += 1
            if "Skipped" in msg:
                skipped += 1
            print_colored(
                f"  ✓ [{idx}/{total_files}] {os.path.basename(zip_file)}: {msg}",
                Colors.GREEN
            )
        else:
            failed += 1
            print_colored(f"  ✗ Failed {os.path.basename(zip_file)}: {msg}", Colors.RED)

    return successful, failed, skipped

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy LRZ NIfTI files to Wunderlich with SSH tunnels."
    )
    parser.add_argument(
        "--reuse-connections",
        action="store_true",
        help="Skip SSH setup; reuse existing master sockets.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only establish SSH connections and exit (no copy).",
    )
    parser.add_argument(
        "--keep-connections",
        action="store_true",
        help="Do not close SSH connections on exit.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        # 1. Setup both persistent connections (passwords entered here only)
        if not args.reuse_connections:
            setup_local_master_connection()
            setup_remote_master_connection()

        if args.setup_only:
            print_colored("Setup complete. Exiting due to --setup-only.", Colors.GREEN)
            return

        total_successful = 0
        total_failed = 0
        total_skipped = 0

        # 2. The Processing Loop (sequential, no parallelism)
        for directory in DIRECTORIES:
            print_colored(f"\n{'='*50}", Colors.BLUE)
            print_colored(f"Processing: {directory}", Colors.BLUE)
            print_colored(f"{'='*50}", Colors.BLUE)

            # Get existing subjects for this specific directory
            print_colored("  Checking existing subjects...", Colors.CYAN)
            existing_subjects = get_existing_subjects(directory)
            print_colored(f"  Found {len(existing_subjects)} existing subject folders", Colors.CYAN)

            # Fetch list via the tunnels
            files = get_lrz_file_list(directory)
            if not files:
                print_colored(f"  No files found in {directory}.", Colors.YELLOW)
                continue

            print_colored(f"  Found {len(files)} NIfTI files to check", Colors.CYAN)

            # Process files SEQUENTIALLY (one at a time for stability)
            successful, failed, skipped = process_files_sequential(files, existing_subjects, directory)

            total_successful += successful
            total_failed += failed
            total_skipped += skipped

            print_colored(f"  ✓ Completed: {successful} successful, {failed} failed, {skipped} skipped",
                         Colors.GREEN if failed == 0 else Colors.YELLOW)

        # Final summary
        print_colored(f"\n{'='*50}", Colors.GREEN)
        print_colored("FINAL SUMMARY", Colors.GREEN)
        print_colored(f"{'='*50}", Colors.GREEN)
        print_colored(f"  Total Successful: {total_successful}", Colors.GREEN)
        print_colored(f"  Total Failed: {total_failed}", Colors.RED if total_failed > 0 else Colors.GREEN)
        print_colored(f"  Total Skipped: {total_skipped}", Colors.CYAN)

    except KeyboardInterrupt:
        print_colored("\nAborted by user.", Colors.RED)
    finally:
        # 3. Cleanup both connections unless user requested otherwise
        if not args.keep_connections and not args.setup_only:
            close_remote_master_connection()
            close_local_master_connection()

if __name__ == "__main__":
    main()
