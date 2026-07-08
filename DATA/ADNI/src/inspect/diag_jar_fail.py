#!/usr/bin/env python3
"""Diagnose why the IDA jar sometimes produces no .zip (exit 0).

Mints a fresh, IP-matched download URL for one specific image, runs the jar
with FULL stdout/stderr capture into an empty scratch dir, and lists every
file produced (not just *.zip). Read-only w.r.t. the real output dir."""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

# download_collection lives in the sibling download/ directory
_INSPECT_DIR = Path(__file__).resolve().parent
_DOWNLOAD_DIR = _INSPECT_DIR.parent / "download"
sys.path.insert(0, str(_DOWNLOAD_DIR))

from dotenv import load_dotenv
from playwright.async_api import async_playwright

from download_collection import (
    loni_login, navigate_to_collection, LONI_ADV_SEARCH_URL, NAV_TIMEOUT_MS,
)
from download_collection_jar import (
    DEFAULT_JAR, DEFAULT_JAVA_HOME, WORKER_MAIN, build_java_env,
    read_visible_values, image_values, select_only, get_download_href,
    go_to_next_page, value_to_image_id, wait_for_collection_rows,
)

COLL = "ADNI_Converters_fMRI"


async def main(target_id: int, max_pages: int) -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    u = (os.environ.get("LONI_USERNAME") or os.environ.get("IDA_USERNAME")
         or os.environ.get("ADNI_USERNAME", ""))
    p = (os.environ.get("LONI_PASSWORD") or os.environ.get("IDA_PASSWORD")
         or os.environ.get("ADNI_PASSWORD", ""))
    java_env = build_java_env(DEFAULT_JAVA_HOME)
    scratch = Path(__file__).resolve().parent.parent / ".tmp_diag_jar"
    scratch.mkdir(exist_ok=True)
    for f in scratch.iterdir():
        f.unlink()

    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        pg = await (await b.new_context(accept_downloads=True)).new_page()
        assert await loni_login(pg, u, p), "login failed"
        await pg.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        assert await navigate_to_collection(pg, COLL), "navigate failed"
        await wait_for_collection_rows(pg)

        # Page forward until we find the target image's checkbox value.
        target_val = f"I{target_id}"
        found = False
        for pi in range(max_pages):
            vals = await read_visible_values(pg)
            if target_val in image_values(vals):
                found = True
                print(f"[diag] found {target_val} on page {pi}")
                break
            advanced, _ = await go_to_next_page(pg, vals)
            if not advanced:
                print(f"[diag] hit last page at {pi} without finding {target_val}")
                break
        if not found:
            print(f"[diag] {target_val} not found; aborting")
            await b.close()
            return

        assert await select_only(pg, target_val), "select failed"
        url = await get_download_href(pg)
        print(f"\n[diag] minted URL:\n{url}\n")
        await b.close()

    if not url:
        print("[diag] no URL — 1-CLICK link never populated. That is the failure.")
        return

    cmd = ["java", "-cp", str(DEFAULT_JAR), WORKER_MAIN,
           f"--directory={scratch}", "--chunks=10", url]
    print(f"[diag] running jar (chunks=10)...\n{'-'*60}")
    proc = subprocess.run(cmd, env=java_env, capture_output=True, text=True)
    print(f"[diag] returncode = {proc.returncode}")
    print(f"[diag] --- STDOUT ---\n{proc.stdout}")
    print(f"[diag] --- STDERR ---\n{proc.stderr}")
    print(f"{'-'*60}\n[diag] files left in scratch dir:")
    for f in sorted(scratch.iterdir()):
        print(f"   {f.stat().st_size:>12,} B   {f.name}")
    if not any(scratch.iterdir()):
        print("   (none)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", type=int, default=1611752)
    ap.add_argument("--max-pages", type=int, default=50)
    args = ap.parse_args()
    asyncio.run(main(args.image_id, args.max_pages))
