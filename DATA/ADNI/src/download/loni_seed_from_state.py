#!/usr/bin/env python3
"""
loni_seed_from_state.py
=======================
Seed NeuroLab's persistent Chromium profile (.loni_profile) with LONI session
cookies captured on another machine.

Why this exists
----------------
The manual login (loni_login_manual.py) has to run where a human can see the
browser and solve the reCAPTCHA. When X11 forwarding into this WSL2 box is
broken, that login is done on the user's Mac instead. A raw copy of the Mac
Chromium profile does NOT work: Chromium encrypts cookies at rest with an
OS-specific key, so a Mac-written cookie DB fails to decrypt on Linux.

Instead the Mac login dumps a plaintext `loni_state.json` (Playwright
storage_state). This script reads that JSON and re-injects the cookies via
add_cookies() into the *local* persistent profile, where they get re-encrypted
with THIS machine's key. Every later headless download run then reuses
.loni_profile exactly as if the login had happened here.

Usage (on NeuroLab, after scp'ing loni_state.json to DATA/ADNI/):
    cd /mnt/e/fyassine/ad-early-detection
    .venv/bin/python DATA/ADNI/src/download/loni_seed_from_state.py

Solves/bypasses nothing — it only relocates an already-authenticated session.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

# Same-dir helper, identical import to the other download scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loni_session import is_logged_in, open_context  # noqa: E402

# loni_state.json lives in DATA/ADNI/ (next to .loni_profile).
STATE_PATH = Path(__file__).resolve().parent.parent.parent / "loni_state.json"
LOGIN_URL = "https://ida.loni.usc.edu/login.jsp?project=ADNI"


async def run() -> int:
    if not STATE_PATH.exists():
        print(f"ERROR: {STATE_PATH} not found — scp it here from the Mac first.")
        return 2

    state = json.loads(STATE_PATH.read_text())
    cookies = state.get("cookies", [])
    if not cookies:
        print("ERROR: loni_state.json has no cookies.")
        return 2
    print(f"Loaded {len(cookies)} cookies from {STATE_PATH}")

    # The LONI session cookies (JSESSIONID, IDA_USC) are session cookies
    # (expires == -1). Chromium keeps session cookies in memory only and PURGES
    # them from a persistent profile on the next launch — so a seeded session
    # would authenticate inside this process but be gone by the first real
    # headless run. Give any session cookie a concrete future expiry so it is
    # written to the on-disk Cookies DB and survives across processes. This does
    # not extend the server-side session; it only stops the client from dropping
    # the cookie between launches.
    far_future = time.time() + 7 * 24 * 3600
    promoted = 0
    for c in cookies:
        exp = c.get("expires", -1)
        if exp is None or exp <= 0:
            c["expires"] = far_future
            promoted += 1
    print(f"Promoted {promoted} session cookie(s) to persistent (7-day client expiry)")

    async with async_playwright() as pw:
        context = await open_context(pw, headless=True)  # native, invisible
        await context.add_cookies(cookies)
        page = await context.new_page()
        await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=60_000)
        await asyncio.sleep(4)

        if await is_logged_in(page):
            print(f"✓ Logged in: {page.url}")
            print("✓ Cookies re-encrypted into DATA/ADNI/.loni_profile/ — "
                  "headless resolve/download can now take over.")
            rc = 0
        else:
            print("✗ Cookies did not authenticate (session may have expired). "
                  "Re-run the Mac login to refresh loni_state.json.")
            rc = 1

        await context.close()  # flush profile to disk
        return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
