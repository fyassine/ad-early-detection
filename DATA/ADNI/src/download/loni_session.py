#!/usr/bin/env python3
"""
loni_session.py
================
Shared LONI IDA browser-session helper used by download_adni_smri.py,
download_collection.py, and download_collection_jar.py.

Why this exists
----------------
LONI added a reCAPTCHA challenge to login.jsp. reCAPTCHA depends on Chrome's
Storage Access API, which a scripted/incognito-style browser context is
denied regardless of network reachability ("requestStorageAccess: Permission
denied") — there is no cookie or header trick that gets around this; a real
human has to solve the challenge once in a real browser profile.

The fix is a *persistent* Chromium profile on disk (launch_persistent_context
instead of launch() + new_context()): a human runs any of the three download
scripts once with --headless false, solves the captcha by hand in the visible
window, and the resulting LONI session cookies are written to profile_dir.
Every later run — headless or not — reuses that same profile directory, so
the captcha is not re-triggered unless the underlying LONI session actually
expires.

This module does not attempt to solve or bypass reCAPTCHA in any way.
"""

from __future__ import annotations

from pathlib import Path

from playwright.async_api import BrowserContext, Playwright

DEFAULT_PROFILE_DIR = Path(__file__).resolve().parent.parent.parent / ".loni_profile"


async def open_context(
    pw: Playwright, headless: bool, profile_dir: Path = DEFAULT_PROFILE_DIR
) -> BrowserContext:
    """Launch (or reattach to) a persistent Chromium profile at profile_dir.

    The returned BrowserContext also exposes .close(), so it is a drop-in
    replacement everywhere the caller previously did:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(accept_downloads=True)
        ...
        await browser.close()
    by instead doing:
        context = await open_context(pw, headless)
        browser = context
        ...
        await browser.close()
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    return await pw.chromium.launch_persistent_context(
        str(profile_dir), headless=headless, accept_downloads=True
    )


async def is_logged_in(page) -> bool:
    """True iff the current page shows a logged-in LONI header.

    LONI's anonymous header always renders the literal text "Log In"; once
    authenticated that's replaced by the account menu. No dedicated
    "logged in" DOM hook is exposed, so this substring check is the most
    robust signal available.
    """
    text = await page.evaluate("() => document.body.innerText")
    return "Log In" not in text
