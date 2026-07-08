#!/usr/bin/env python3
"""
IDA Downloader wrapper
=======================
Thin wrapper around the LONI IDA Downloader jar
(`IdaDownloader_15May2026.jar`), which downloads a server-prepared package
from a one-time IDA portal URL into a local directory.

The jar itself only understands `--directory`, `--chunks`, and a `<URL>`
obtained from the IDA web portal — it has no concept of "image IDs". This
wrapper:
  * validates java/jar/chunks before running anything,
  * accepts the URL via `--url-file` (recommended, since the URL contains
    `&`/`?` and a session token that's awkward to pass on the command line)
    or `--url`,
  * records a small provenance JSON (timestamp, image_ids.txt manifest used,
    chunk count, output directory) for each run, without persisting the
    URL's query string (it embeds a session token).

How to obtain `<URL>`
---------------------
Log in to https://ida.loni.usc.edu, search for / collect the desired images,
open Data Collections -> <collection> -> Not Downloaded, select items, click
1-CLICK DOWNLOAD, and once the portal populates the `#simple-download-link`
anchor, copy that link's href. Paste it into a text file and pass
`--url-file path/to/file.txt`.

Usage
-----
    # Dry run -- validate everything, print the resolved command, no download
    python run_ida_downloader.py --dry-run --url-file url.txt

    # Real download (run inside tmux/screen -- can take a long time)
    python run_ida_downloader.py --url-file url.txt --chunks 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

# ── Paths / constants ────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).resolve().parent
ADNI_DIR = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = ADNI_DIR / "__dicom_zips_flat__"
DEFAULT_IMAGE_IDS_TXT = ADNI_DIR / "__metadata__" / "image_ids.txt"
DEFAULT_JAR = SRC_DIR / "IdaDownloader_15May2026.jar"
# The jar's Main-Class (`launch.Launcher`) is only a gate: it requires
# java.version >= 12 AND java.vendor to contain "oracle", then reflectively
# calls the real worker below. java.vendor is set by the JVM and cannot be
# overridden with -Djava.vendor=, so `java -jar` fails on any OpenJDK build
# branded "Ubuntu"/"Debian"/etc. The worker class itself performs no vendor
# check, so we bypass the Launcher by invoking it directly via -cp.
WORKER_MAIN = "edu.usc.loni.ida.download.resource.ResourceDownloader"
DEFAULT_CHUNKS = 10
MIN_CHUNKS = 1
MAX_CHUNKS = 20
MIN_JAVA_VERSION = 12
RUNS_LOG_DIR_NAME = ".ida_download_runs"


# ── Console output ───────────────────────────────────────────────────────────


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def tprint(message: str, color: str = "") -> None:
    if color:
        print(f"{color}{message}{Colors.RESET}")
    else:
        print(message)


# ── Validation helpers ───────────────────────────────────────────────────────


def check_java_version(min_version: int = MIN_JAVA_VERSION) -> str:
    """Run `java -version` and return the raw output. Raise if missing or too old."""
    try:
        result = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "java is not installed or not on PATH. The IDA Downloader jar "
            f"requires Java {min_version}+."
        ) from exc

    # `java -version` writes to stderr, e.g.:
    #   openjdk version "17.0.2" 2022-01-18
    #   java version "21" 2023-09-19
    output = result.stderr or result.stdout
    match = re.search(r'version "(\d+)', output)
    if not match:
        raise RuntimeError(f"Could not parse java version from output: {output!r}")

    major = int(match.group(1))
    if major < min_version:
        raise RuntimeError(
            f"java {major} found, but the IDA Downloader jar requires "
            f"Java {min_version}+. Output was: {output.strip()!r}"
        )
    return output.strip()


def load_image_ids(image_ids_file: Path) -> list[int]:
    """Parse a comma-separated list of numeric image IDs."""
    if not image_ids_file.exists():
        raise ValueError(f"Image IDs file not found: {image_ids_file}")
    raw = image_ids_file.read_text(encoding="utf-8").strip()
    ids = [tok.strip() for tok in raw.replace("\n", ",").split(",") if tok.strip()]
    if not ids:
        raise ValueError(f"Image IDs file is empty: {image_ids_file}")
    if not all(tok.isdigit() for tok in ids):
        raise ValueError(f"Image IDs file contains non-numeric entries: {image_ids_file}")
    return [int(tok) for tok in ids]


def resolve_url(args: argparse.Namespace) -> str:
    if bool(args.url) == bool(args.url_file):
        raise ValueError(
            "run_ida_downloader requires exactly one of --url or --url-file."
        )
    if args.url_file:
        url_file = Path(args.url_file)
        if not url_file.exists():
            raise ValueError(f"--url-file not found: {url_file}")
        url = url_file.read_text(encoding="utf-8").strip()
    else:
        url = args.url.strip()
    if not url:
        raise ValueError("Resolved IDA download URL is empty.")
    return url


def redact_url(url: str) -> str:
    """Return scheme://host/path with the query string (session token) stripped."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


# ── Provenance ────────────────────────────────────────────────────────────────


def write_run_record(
    output_dir: Path,
    image_ids_file: Path,
    image_id_count: int,
    chunks: int,
    jar: Path,
    url: str,
) -> Path:
    runs_dir = output_dir / RUNS_LOG_DIR_NAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "timestamp": timestamp,
        "image_ids_file": str(image_ids_file),
        "image_id_count": image_id_count,
        "chunks": chunks,
        "directory": str(output_dir),
        "jar": str(jar),
        "url": redact_url(url),
    }
    record_path = runs_dir / f"{timestamp}.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    url_group = parser.add_mutually_exclusive_group()
    url_group.add_argument("--url", default="", help="IDA download URL")
    url_group.add_argument(
        "--url-file",
        default="",
        help="Path to a text file containing only the IDA download URL (recommended)",
    )
    parser.add_argument(
        "--directory",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for downloaded files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=DEFAULT_CHUNKS,
        help=f"Number of parallel download chunks, {MIN_CHUNKS}-{MAX_CHUNKS} (default: {DEFAULT_CHUNKS})",
    )
    parser.add_argument(
        "--jar",
        default=str(DEFAULT_JAR),
        help=f"Path to the IDA Downloader jar (default: {DEFAULT_JAR.name})",
    )
    parser.add_argument(
        "--image-ids-file",
        default=str(DEFAULT_IMAGE_IDS_TXT),
        help=f"Image IDs manifest used for the provenance record (default: {DEFAULT_IMAGE_IDS_TXT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate everything and print the resolved command without downloading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not (MIN_CHUNKS <= args.chunks <= MAX_CHUNKS):
        raise ValueError(
            f"--chunks must be between {MIN_CHUNKS} and {MAX_CHUNKS}, got {args.chunks}"
        )

    jar_path = Path(args.jar)
    if not jar_path.exists():
        raise FileNotFoundError(f"IDA Downloader jar not found: {jar_path}")

    url = resolve_url(args)

    java_version = check_java_version()
    tprint(f"java OK: {java_version.splitlines()[0]}", Colors.GREEN)

    image_ids_file = Path(args.image_ids_file)
    image_ids = load_image_ids(image_ids_file)
    tprint(
        f"Image IDs manifest: {image_ids_file} ({len(image_ids)} IDs, "
        f"e.g. {image_ids[:5]}...)",
        Colors.CYAN,
    )

    if "TMUX" not in os.environ and "STY" not in os.environ:
        tprint(
            "Warning: not running inside tmux/screen. Long downloads will be "
            "interrupted if the SSH session disconnects.",
            Colors.YELLOW,
        )

    output_dir = Path(args.directory)

    record_path = write_run_record(
        output_dir=output_dir,
        image_ids_file=image_ids_file,
        image_id_count=len(image_ids),
        chunks=args.chunks,
        jar=jar_path,
        url=url,
    )
    tprint(f"Run record written: {record_path}", Colors.CYAN)

    cmd = [
        "java", "-cp", str(jar_path), WORKER_MAIN,
        f"--directory={output_dir}",
        f"--chunks={args.chunks}",
        url,
    ]
    redacted_cmd = cmd[:-1] + [redact_url(url)]
    tprint(f"Command: {' '.join(redacted_cmd)}", Colors.BOLD)

    if args.dry_run:
        tprint("[DRY RUN] Not invoking the jar.", Colors.YELLOW)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)

    tprint(f"Download complete -> {output_dir}", Colors.GREEN)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        tprint(f"ERROR: {exc}", Colors.RED)
        raise SystemExit(1)
