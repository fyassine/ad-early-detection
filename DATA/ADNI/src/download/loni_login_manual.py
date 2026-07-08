#!/usr/bin/env python3
"""
loni_login_manual.py
====================
One-time, human-in-the-loop LONI IDA login for when the automated submit in
download_adni_smri.py races the reCAPTCHA and fails.

Unlike the automated loni_login() flow, this script does NOT click "Log In"
for you. It only:
  1. opens the shared persistent Chromium profile (.loni_profile) HEADED,
  2. navigates to the ADNI login page,
  3. accepts the cookie banner and reveals the login form,
  4. pre-fills your email/password (purely to save typing),
  5. then STOPS and waits for you to press Enter in this terminal.

That pause gives you unlimited time to solve the reCAPTCHA and click "Log In"
by hand in the visible XQuartz-forwarded window. When you come back and press
Enter, it verifies the session is authenticated and closes the browser
cleanly, so the LONI cookies are flushed into .loni_profile/ for every later
headless run.

This module solves/bypasses nothing — the human solves the captcha.

Usage (over `ssh -Y NeuroLab`, with XQuartz running on your Mac):
    cd /mnt/e/fyassine/ad-early-detection
    export DISPLAY=localhost:10.0            # if `echo $DISPLAY` is empty
    .venv/bin/python DATA/ADNI/src/download/loni_login_manual.py

Credentials are read (in order) from --username/--password, then the process
environment (ADNI_USERNAME / ADNI_PASSWORD), then DATA/ADNI/src/.env, then an
interactive prompt. You can also skip pre-fill entirely and just type
everything into the browser yourself.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: pip install python-dotenv")
    sys.exit(1)

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("ERROR: pip install playwright && playwright install chromium")
    sys.exit(1)

# Same-dir helper, identical to download_adni_smri.py's import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loni_session import is_logged_in, open_context  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent          # .../src/download/
ADNI_SRC_DIR = SRC_DIR.parent                       # .../src/
DEFAULT_ENV_FILE = ADNI_SRC_DIR / ".env"

LONI_BASE_URL = "https://ida.loni.usc.edu"
LONI_LOGIN_URL = f"{LONI_BASE_URL}/login.jsp?project=ADNI"
NAV_TIMEOUT_MS = 60_000


class C:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log(msg: str, color: str = "") -> None:
    print(f"{color}{msg}{C.RESET}" if color else msg, flush=True)


async def prefill(page, username: str, password: str) -> None:
    """Best-effort: reveal the login form and drop the credentials in.

    Every step is wrapped so a DOM change on LONI's side can't abort the
    run — worst case the field is empty and you type it yourself.
    """
    try:
        await page.evaluate(
            "() => { const el = document.querySelector('.ida-cookie-policy-accept');"
            " if (el) el.click(); }"
        )
        await asyncio.sleep(2)
    except Exception:
        pass

    try:
        await page.evaluate(
            """() => {
            const sels = [
                'div.ida-menu-option.login:not(.disabled)',
                'div.ida-menu-option.sub-menu.login',
                'div.ida-menu-option.login',
            ];
            for (const s of sels) {
                const el = document.querySelector(s);
                if (el && el.offsetParent !== null) { el.click(); return; }
            }
        }"""
        )
        await asyncio.sleep(3)
    except Exception:
        pass

    if not (username or password):
        return
    try:
        await page.evaluate(
            """([user, pwd]) => {
            function fill(sels, val) {
                if (!val) return;
                for (const s of sels) {
                    const el = document.querySelector(s);
                    if (el) {
                        el.value = val;
                        el.dispatchEvent(new Event('input',  {bubbles: true}));
                        el.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                }
            }
            fill(["input[name='userEmail']", "input[type='email']"], user);
            fill(["input[name='userPassword']", "input[type='password']"], pwd);
        }""",
            [username, password],
        )
    except Exception:
        pass


async def run(args: argparse.Namespace) -> int:
    if not os.environ.get("DISPLAY"):
        log(
            "WARNING: $DISPLAY is empty — no X server to draw the browser on. "
            "Run `export DISPLAY=localhost:10.0` first (see script header).",
            C.YELLOW,
        )

    env_file = Path(args.env_file)
    if env_file.exists():
        load_dotenv(env_file)
    username = args.username or os.environ.get("ADNI_USERNAME", "")
    password = args.password or os.environ.get("ADNI_PASSWORD", "")
    if not args.no_prefill:
        if not username:
            username = input("LONI IDA username (blank = type it in browser): ").strip()
        if not password and username:
            password = getpass.getpass("LONI IDA password (blank = type it in browser): ")

    async with async_playwright() as pw:
        context = await open_context(pw, headless=False)  # always headed
        page = await context.new_page()

        log("\n── Manual LONI login ───────────────────────────────────────", C.BOLD)
        log(f"  → Navigating to {LONI_LOGIN_URL}", C.CYAN)
        await page.goto(LONI_LOGIN_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)

        if await is_logged_in(page):
            log("  ✓ Already logged in — session already persisted. Nothing to do.", C.GREEN)
            await context.close()
            return 0

        await prefill(page, username, password)

        log("", C.RESET)
        log("  A Chromium window is open on your Mac (via XQuartz).", C.BOLD)
        log("  In THAT window:", C.BOLD)
        log("    1. Solve the reCAPTCHA.", C.YELLOW)
        log("    2. Click 'Log In' (email/password are pre-filled).", C.YELLOW)
        log("    3. Wait until the ADNI/IDA page shows you as logged in.", C.YELLOW)
        log("  Take as long as you need — this script is not on a timer.", C.CYAN)
        log("", C.RESET)

        # Blocking, untimed. Run the input() off the event loop so Playwright
        # keeps the browser responsive while you work in it.
        await asyncio.get_event_loop().run_in_executor(
            None, input, "  ↳ When you are logged in, press Enter here to verify & save… "
        )

        if await is_logged_in(page):
            log(f"  ✓ Logged in: {page.url}", C.GREEN)
            log("  ✓ Session cookies persisted to DATA/ADNI/.loni_profile/", C.GREEN)
            log("    You can now let the headless resolve/download take over.", C.GREEN)
            rc = 0
        else:
            log("  ✗ Still not logged in (LONI header still shows 'Log In').", C.RED)
            log("    Finish the login in the browser and re-run, or press Enter "
                "too early?", C.RED)
            rc = 1

        await context.close()  # flush profile to disk
        return rc


def main() -> int:
    p = argparse.ArgumentParser(description="One-time human-in-the-loop LONI login.")
    p.add_argument("--username", default="")
    p.add_argument("--password", default="")
    p.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    p.add_argument(
        "--no-prefill",
        action="store_true",
        help="Don't read/prompt for credentials; type everything in the browser.",
    )
    args = p.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
