#!/usr/bin/env python3
"""
Extract files from a split-ZIP Part 1 that is missing the central directory.

LONI IDA split-ZIP structure:
  Part1.zip  — all the file data (local file headers + compressed data)
  Part2.zip  — central directory + EOCD (often 0 bytes or very small)

Python's zipfile cannot open Part1 alone because there is no EOCD.
This script scans raw local file headers (PK\x03\x04) and streams
the deflate-compressed data out using zlib with data-descriptor support.

Usage:
    python extract_partial_zip.py <part1.zip> <output_dir>

After extraction:
    - If the files are DICOMs, run dcm2niix on the extracted directory.
    - If the files are NIfTI (.nii / .nii.gz), they are ready to use.
"""

from __future__ import annotations

import os
import struct
import sys
import zlib
from pathlib import Path

DCM2NIIX = "/usr/local/fsl/bin/dcm2niix"

LFH_SIG  = b"PK\x03\x04"  # local file header
DD_SIG   = b"PK\x07\x08"  # data descriptor signature (optional)
CD_SIG   = b"PK\x01\x02"  # central directory
EOCD_SIG = b"PK\x05\x06"  # end of central directory

CHUNK = 65536


def read_local_file_header(f: "BinaryIO") -> dict | None:
    """Parse a local file header at the current file position (signature already read)."""
    raw = f.read(26)
    if len(raw) < 26:
        return None
    (version, flags, compression, mod_time, mod_date,
     crc32_val, compressed_size, uncompressed_size,
     fname_len, extra_len) = struct.unpack("<HHHHHIIIHH", raw)
    fname_bytes = f.read(fname_len)
    f.read(extra_len)  # skip extra field
    return {
        "flags":            flags,
        "compression":      compression,
        "crc32":            crc32_val,
        "compressed_size":  compressed_size,
        "uncompressed_size": uncompressed_size,
        "fname":            fname_bytes.decode("utf-8", errors="replace"),
        "has_data_desc":    bool(flags & 8),
    }


def extract_deflate_stream(f: "BinaryIO", out_path: str) -> None:
    """
    Decompress a raw deflate stream (compression=8, unknown compressed size).

    zlib.decompress(chunk) without max_length silently discards bytes after the
    stream end — unconsumed_tail is never populated unless max_length < output.
    To recover the exact stream-end position we CRC-match against the data
    descriptor that immediately follows the stream.

    Algorithm:
      1. Feed CHUNK-sized reads to decompressobj(-15), track CRC32 of output.
      2. When d.eof becomes True, remember (pos_before_read, last_chunk).
      3. Search last_chunk for DD_SIG (PK\x07\x08) + matching CRC32.
         The probability of a false-positive CRC collision is ~1 in 2^32.
      4. Seek to (data_descriptor_start + 16) to land on the next LFH.
      5. If no match in last_chunk, also search prev_chunk (handles span
         of data descriptor across a chunk boundary).
    """
    d = zlib.decompressobj(-15)
    crc32_running = 0
    prev_chunk = b""
    last_chunk = b""
    pos_before_last_read = f.tell()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as fout:
        while not d.eof:
            pos_before_last_read = f.tell()
            chunk = f.read(CHUNK)
            if not chunk:
                break
            prev_chunk = last_chunk
            last_chunk = chunk
            decompressed = d.decompress(chunk)
            fout.write(decompressed)
            crc32_running = zlib.crc32(decompressed, crc32_running) & 0xFFFFFFFF

    # Locate the data descriptor via CRC matching.
    # The descriptor lives in last_chunk (or straddles prev/last).
    search_buf = prev_chunk + last_chunk
    search_base = pos_before_last_read - len(prev_chunk)
    dd_start = _find_data_descriptor(search_buf, crc32_running)

    if dd_start is not None:
        # dd_start is relative to search_buf; map to file offset
        abs_dd = search_base + dd_start
        # Descriptor is 16 bytes (4 sig + 4 crc + 4 csize + 4 usize)
        f.seek(abs_dd + 16)
    else:
        # Fallback: descriptor has no PK\x07\x08 prefix, just 12 bytes
        # Search for raw CRC32 match without signature
        dd_start = _find_dd_no_sig(search_buf, crc32_running)
        if dd_start is not None:
            abs_dd = search_base + dd_start
            f.seek(abs_dd + 12)
        # else: leave f at current position (best-effort)


def _find_data_descriptor(buf: bytes, crc: int) -> int | None:
    """
    Return index of PK\x07\x08 in buf where the following 4 bytes equal crc,
    or None.
    """
    crc_bytes = struct.pack("<I", crc)
    start = 0
    while True:
        idx = buf.find(DD_SIG, start)
        if idx == -1 or idx + 8 > len(buf):
            return None
        if buf[idx + 4 : idx + 8] == crc_bytes:
            return idx
        start = idx + 1


def _find_dd_no_sig(buf: bytes, crc: int) -> int | None:
    """
    When the data descriptor has no PK\x07\x08 prefix, search for the raw
    CRC32 bytes.  Only used as fallback.
    """
    crc_bytes = struct.pack("<I", crc)
    idx = buf.rfind(crc_bytes)
    return idx if idx != -1 else None


def extract_stored_stream(f: "BinaryIO", size: int, out_path: str) -> None:
    """Extract a stored (uncompressed) file of known size."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    remaining = size
    with open(out_path, "wb") as fout:
        while remaining > 0:
            chunk = f.read(min(CHUNK, remaining))
            if not chunk:
                break
            fout.write(chunk)
            remaining -= len(chunk)


def extract_partial_zip(zip_path: str, output_dir: str) -> tuple[int, int]:
    """
    Scan zip_path for local file headers and extract all files to output_dir.
    Returns (files_extracted, errors).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_extracted = 0
    errors = 0

    with open(zip_path, "rb") as f:
        while True:
            pos = f.tell()
            sig = f.read(4)
            if not sig or len(sig) < 4:
                break

            if sig == LFH_SIG:
                header = read_local_file_header(f)
                if header is None:
                    print(f"  [offset {pos}] ✗ truncated header", file=sys.stderr)
                    break

                fname = header["fname"]
                out_path = str(output_dir / fname)

                # Skip directory entries
                if fname.endswith("/"):
                    files_extracted += 1
                    continue

                if header["compression"] == 0:  # Stored
                    if header["has_data_desc"]:
                        print(f"  SKIP (stored+data-descriptor, unknown size): {fname}", file=sys.stderr)
                        errors += 1
                    else:
                        extract_stored_stream(f, header["compressed_size"], out_path)
                        files_extracted += 1

                elif header["compression"] == 8:  # Deflate
                    try:
                        if header["has_data_desc"]:
                            extract_deflate_stream(f, out_path)
                        else:
                            data = f.read(header["compressed_size"])
                            decompressed = zlib.decompress(data, -15)
                            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                            with open(out_path, "wb") as fout:
                                fout.write(decompressed)
                        files_extracted += 1
                        if files_extracted % 200 == 0:
                            print(f"  {files_extracted} files extracted...", flush=True)
                    except Exception as e:
                        print(f"  ✗ Error extracting {fname!r}: {e}", file=sys.stderr)
                        errors += 1

                else:
                    print(f"  SKIP (unsupported compression={header['compression']}): {fname}", file=sys.stderr)
                    errors += 1

            elif sig in (CD_SIG, EOCD_SIG, b"PK\x06\x06"):
                print(f"  Reached central directory at offset {pos} — done scanning data.", flush=True)
                break

            else:
                # Unknown signature — we've lost sync. Stop.
                print(f"  Lost sync at offset {pos}: signature {sig.hex()}", file=sys.stderr)
                break

    return files_extracted, errors


def convert_dicoms(dicom_root: str, nifti_out: str) -> None:
    """Run dcm2niix on the extracted DICOM tree."""
    import subprocess
    nifti_out = Path(nifti_out)
    nifti_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        DCM2NIIX,
        "-z", "y",    # gzip output
        "-f", "%p_%s_%d",  # protocol_series_description
        "-o", str(nifti_out),
        dicom_root,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        nifti_files = list(nifti_out.glob("*.nii.gz")) + list(nifti_out.glob("*.nii"))
        print(f"  ✓ dcm2niix done — {len(nifti_files)} NIfTI files → {nifti_out}")
    else:
        print(f"  ✗ dcm2niix failed:\n{result.stderr}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python extract_partial_zip.py <part1.zip> <output_dir>")
        print()
        print("Options:")
        print("  --convert <nifti_out>   After extraction, run dcm2niix to convert DICOMs")
        sys.exit(1)

    zip_path = sys.argv[1]
    output_dir = sys.argv[2]
    convert_flag = "--convert" in sys.argv
    nifti_out = None
    if convert_flag:
        idx = sys.argv.index("--convert")
        nifti_out = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else output_dir + "_nifti"

    zip_size_mb = Path(zip_path).stat().st_size / (1024 ** 2)
    print(f"  Source : {zip_path}  ({zip_size_mb:.0f} MB)")
    print(f"  Output : {output_dir}")
    print()

    import time
    t0 = time.time()
    files, errs = extract_partial_zip(zip_path, output_dir)
    elapsed = time.time() - t0

    print(f"\n  ✓ Extracted {files} entries in {elapsed:.1f}s  ({errs} errors)")

    # Show what was extracted
    out_path = Path(output_dir)
    all_files = list(out_path.rglob("*"))
    data_files = [f for f in all_files if f.is_file()]
    exts = {}
    for f in data_files:
        ext = "".join(f.suffixes).lower()
        exts[ext] = exts.get(ext, 0) + 1
    print("  File types:")
    for ext, count in sorted(exts.items(), key=lambda x: -x[1]):
        print(f"    {ext or '(no ext)'}: {count}")

    if convert_flag and nifti_out:
        print(f"\n  Running DICOM→NIfTI conversion...")
        convert_dicoms(output_dir, nifti_out)


if __name__ == "__main__":
    main()
