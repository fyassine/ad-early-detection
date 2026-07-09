#!/usr/bin/env python3
"""
redownload_and_bids.py
======================
Re-downloads 11 specific ADNI image IDs whose DICOM archives had missing
slices (dcm2niix exit-codes 1 or 8 during convert_to_bids), then converts
the fresh downloads to BIDS NIfTI using the standard convert_to_bids.py.

Root cause: all 11 zips are structurally valid but internally incomplete —
one or more DICOM frames are absent from the series.  This is an upstream
ADNI data quality issue; re-downloading from LONI is the only remedy.

What this script does
---------------------
1. Delete the stale zip for each target from __dicom_zips_flat__.
2. Delete any existing (failed) BIDS NIfTI so convert_to_bids won't skip it.
3. Log in to LONI IDA via the persistent Chromium profile (no re-captcha).
4. For each image_id:
     a. Advanced Search by image_id -> Add to "mci" collection.
     b. Data Collections -> mci -> Not Downloaded -> 1-CLICK DOWNLOAD -> ZIP.
     c. Rename ZIP to {subject_id}_{image_id}.zip in __dicom_zips_flat__.
5. Run convert_to_bids.py --subjects <affected subjects>.

Usage
-----
    python redownload_and_bids.py --dry-run          # safe preview
    python redownload_and_bids.py                    # full headless run
    python redownload_and_bids.py --headless false   # visible browser
    python redownload_and_bids.py --no-bids          # download only
    python redownload_and_bids.py --image-ids 298204 1260257  # subset

Resume safety: existing zips and BIDS outputs are skipped automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make the sibling loni_session importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Dependency checks ──────────────────────────────────────────────────────────

_MISSING: list[str] = []
try:
    from dotenv import load_dotenv
except ImportError:
    _MISSING.append("python-dotenv")

try:
    from playwright.async_api import BrowserContext, Page, async_playwright
    from loni_session import open_context as _loni_open_context
    from loni_session import is_logged_in as _loni_is_logged_in
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    _MISSING.append("playwright")

if _MISSING:
    print(f"ERROR: Missing packages: {', '.join(_MISSING)}")
    print(f"  pip install {' '.join(_MISSING)}")
    if "playwright" in _MISSING:
        print("  playwright install chromium")
    sys.exit(2)

# ── Paths ──────────────────────────────────────────────────────────────────────

_SCRIPT_DIR   = Path(__file__).resolve().parent   # .../src/download/
_SRC_DIR      = _SCRIPT_DIR.parent                # .../src/
_ADNI_DIR     = _SRC_DIR.parent                   # .../ADNI/
_PROJECT_ROOT = _ADNI_DIR.parent.parent           # ad-early-detection/

ZIP_DIR        = _ADNI_DIR / "__dicom_zips_flat__"
BIDS_ROOT      = _ADNI_DIR / "__bold_and_smri__"
MANIFEST_CSV   = _ADNI_DIR / "__metadata__" / "adni_bids_manifest.csv"
ENV_FILE       = _SRC_DIR / ".env"
PROFILE_DIR    = _ADNI_DIR / ".loni_profile"
LOG_DIR        = _PROJECT_ROOT / "logs" / "adni-redownload"
CONVERT_SCRIPT = _SRC_DIR / "unzip" / "convert_to_bids.py"

LONI_BASE_URL       = "https://ida.loni.usc.edu"
LONI_LOGIN_URL      = f"{LONI_BASE_URL}/login.jsp?project=ADNI"
LONI_ADV_SEARCH_URL = (
    f"{LONI_BASE_URL}/pages/access/search.jsp"
    "?project=ADNI&tab=advSearch&page=SEARCH&subPage=NEW_ADV_QUERY"
)
COLLECTION_NAME = "mci"   # pre-existing persistent collection in YAHOO tree
NAV_TIMEOUT_MS  = 60_000
DL_TIMEOUT_MS   = 600_000  # 10 min per zip

# ── The 11 targets ─────────────────────────────────────────────────────────────

TARGETS: list[dict] = [
    {"image_id": 298204,  "subject_id": "002_S_4251", "scan_type": "anat", "note": "T1w MPRAGE 169/170 slices"},
    {"image_id": 1260257, "subject_id": "006_S_6610", "scan_type": "func", "note": "fMRI 9455 not div 48"},
    {"image_id": 1314046, "subject_id": "019_S_6315", "scan_type": "func", "note": "fMRI 9194 not div 48"},
    {"image_id": 1037522, "subject_id": "019_S_6533", "scan_type": "func", "note": "fMRI 9422 not div 48"},
    {"image_id": 1367893, "subject_id": "027_S_6788", "scan_type": "func", "note": "fMRI 9595 not div 48"},
    {"image_id": 1416249, "subject_id": "027_S_6842", "scan_type": "func", "note": "fMRI 9599 not div 48"},
    {"image_id": 272899,  "subject_id": "130_S_2391", "scan_type": "func", "note": "fMRI 1533 not div any"},
    {"image_id": 1162410, "subject_id": "130_S_4817", "scan_type": "func", "note": "fMRI 9449 not div 48"},
    {"image_id": 1441195, "subject_id": "130_S_4817", "scan_type": "func", "note": "fMRI 9455 not div 48"},
    {"image_id": 999077,  "subject_id": "130_S_6329", "scan_type": "func", "note": "fMRI 9448 not div 48"},
    {"image_id": 1129771, "subject_id": "130_S_6688", "scan_type": "func", "note": "fMRI 9455 not div 48"},
]

# ── Logging ────────────────────────────────────────────────────────────────────


class Colors:
    GREEN  = "\033[0;32m"
    RED    = "\033[0;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[0;36m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


_logger: Optional[logging.Logger] = None


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("redownload_bids")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log(msg: str, color: str = "") -> None:
    """Print to console (with optional ANSI color) and write plain text to log file."""
    clean = re.sub(r"\033\[[0-9;]*m", "", msg)
    if _logger:
        # File handler gets plain text; suppress the StreamHandler's duplicate print
        # by writing directly to the file handler and printing to stdout ourselves.
        for handler in _logger.handlers:
            if isinstance(handler, logging.FileHandler):
                record = _logger.makeRecord(
                    _logger.name, logging.INFO, "", 0, clean, (), None
                )
                handler.emit(record)
    if color:
        print(f"{color}{msg}{Colors.RESET}", flush=True)
    else:
        print(msg, flush=True)


# ── Credential loading ─────────────────────────────────────────────────────────


def load_credentials() -> tuple[str, str]:
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=False)
    username = os.environ.get("ADNI_USERNAME", "").strip()
    password = os.environ.get("ADNI_PASSWORD", "").strip()
    if not username:
        username = input("LONI IDA username: ").strip()
    if not password:
        import getpass
        password = getpass.getpass("LONI IDA password: ")
    if not username or not password:
        log("ERROR: Empty credentials.", Colors.RED)
        sys.exit(1)
    return username, password


# ── LONI browser helpers ───────────────────────────────────────────────────────


async def open_browser_context(pw, headless: bool) -> BrowserContext:
    """Use the shared persistent Chromium profile from loni_session.

    This is the same profile used by download_collection.py and
    loni_login_manual.py, so a session established by loni_login_manual.py
    (after solving the reCAPTCHA once) is automatically reused here.
    """
    return await _loni_open_context(pw, headless=headless, profile_dir=PROFILE_DIR)


async def is_logged_in(page: Page) -> bool:
    """Delegate to the shared loni_session helper."""
    return await _loni_is_logged_in(page)


async def loni_login(page: Page, username: str, password: str) -> bool:
    log(f"  -> Navigating to {LONI_LOGIN_URL}", Colors.CYAN)
    await page.goto(LONI_LOGIN_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
    await asyncio.sleep(5)

    if await is_logged_in(page):
        log("  OK Reusing existing LONI session (persistent profile)", Colors.GREEN)
        return True

    await page.evaluate("""() => {
        const el = document.querySelector('.ida-cookie-policy-accept');
        if (el) el.click();
    }""")
    await asyncio.sleep(2)

    await page.evaluate("""() => {
        const sels = [
            'div.ida-menu-option.login:not(.disabled)',
            'div.ida-menu-option.sub-menu.login',
            'div.ida-menu-option.login',
        ];
        for (const s of sels) {
            const el = document.querySelector(s);
            if (el && el.offsetParent !== null) { el.click(); return; }
        }
    }""")
    await asyncio.sleep(3)

    await page.evaluate("""([user, pwd]) => {
        function fill(sels, val) {
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
    }""", [username, password])

    await page.evaluate("""() => {
        const sels = ['.login-btn', 'span.login-btn', 'button[type="submit"]'];
        for (const s of sels) {
            const el = document.querySelector(s);
            if (el) { el.click(); return; }
        }
    }""")

    try:
        await page.wait_for_url(lambda u: "login" not in u.lower(), timeout=25_000)
    except Exception:
        pass
    await asyncio.sleep(2)

    if "login" in page.url.lower():
        log("  FAIL Login failed -- still on login page", Colors.RED)
        return False
    log(f"  OK Logged in: {page.url}", Colors.GREEN)
    return True


async def search_for_image(page: Page, image_id: int) -> bool:
    """Advanced Search by image_id; returns True if at least 1 result row found."""
    await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
    await asyncio.sleep(5)
    if "login" in page.url.lower():
        return False

    await page.evaluate("""([imgId]) => {
        const imgSec = document.getElementById('imageModalityOption');
        if (imgSec && !imgSec.checked) imgSec.click();
        const orig = document.getElementById('originalOption');
        if (orig && !orig.checked) orig.click();
        const idBox = document.getElementById('imageIdText')
                   || document.querySelector('input[name="imgId"]');
        if (idBox) {
            idBox.value = imgId;
            idBox.dispatchEvent(new Event('input',  {bubbles: true}));
            idBox.dispatchEvent(new Event('change', {bubbles: true}));
        }
    }""", [str(image_id)])

    await page.evaluate("""() => {
        const b = document.getElementById('advSearchQuery');
        if (b) b.click();
    }""")

    for _ in range(20):
        await asyncio.sleep(1.5)
        try:
            info = await page.evaluate("""() => ({
                subjectCbs: document.querySelectorAll(
                    'input[type="checkbox"][id^="adv_subject_"][id$="_check"]'
                ).length,
                description: (document.getElementById('advTableDescription') || {}).textContent || '',
            })""")
        except Exception:
            await asyncio.sleep(2)
            continue
        if info["subjectCbs"] > 0 or "Result" in info["description"]:
            return True
    return False


async def add_to_collection(page: Page, collection_name: str) -> None:
    """Select all search results and add to collection_name."""
    await page.evaluate("""() => {
        const cbs = document.querySelectorAll(
            'input[type="checkbox"][id^="adv_subject_"][id$="_check"]'
        );
        for (const cb of cbs) {
            if (!cb.checked) {
                cb.click();
            }
        }
        const sa = document.getElementById('advResultSelectAll');
        if (sa && !sa.checked) {
            sa.click();
        }
    }""")
    await asyncio.sleep(2)

    await page.evaluate("""() => {
        const btn = document.getElementById('advResultAddCollectId');
        if (!btn) return;
        btn.removeAttribute('disabled');
        btn.className = (btn.className || '').replace('buttonDisabled', 'button');
        btn.click();
    }""")
    await asyncio.sleep(4)

    fill_result = await page.evaluate("""([collName]) => {
        const existingSel = document.getElementById('candidateNames');
        const nameInput   = document.getElementById('nameText');
        if (existingSel) {
            const match = [...existingSel.options].find(
                o => o.text.trim() === collName || o.value === collName
            );
            if (match) {
                existingSel.value = match.value;
                existingSel.dispatchEvent(new Event('change', {bubbles: true}));
                if (nameInput) { nameInput.value = ''; nameInput.disabled = true; }
                return 'selected: ' + match.text;
            }
        }
        if (nameInput) {
            nameInput.value = collName;
            nameInput.disabled = false;
            nameInput.dispatchEvent(new Event('input',  {bubbles: true}));
            nameInput.dispatchEvent(new Event('change', {bubbles: true}));
            if (existingSel) existingSel.value = '';
            return 'new: ' + collName;
        }
        return 'dialog not found';
    }""", [collection_name])
    log(f"    -> dialog fill: {fill_result}", Colors.CYAN)

    try:
        if "not found" in fill_result.lower():
            async with page.expect_navigation(wait_until="domcontentloaded", timeout=20000):
                await page.evaluate("""([collName]) => {
                    const form = document.advResultTable || document.forms['advResultTable'];
                    if (!form) return;
                    if (form.userAction)   form.userAction.value = 'add';
                    if (form.newName)      { form.newName.value = collName; form.newName.disabled = false; }
                    if (form.existingName) form.existingName.value = '';
                    form.submit();
                }""", [collection_name])
        else:
            async with page.expect_navigation(wait_until="domcontentloaded", timeout=20000):
                await page.evaluate("""() => {
                    const modal = document.getElementById('regroupDialog_c');
                    if (modal) {
                        for (const btn of modal.querySelectorAll('button')) {
                            if (btn.textContent.trim() === 'OK') { btn.click(); return; }
                        }
                    }
                    for (const btn of document.querySelectorAll('button')) {
                        if (btn.textContent.trim() === 'OK') { btn.click(); return; }
                    }
                    if (typeof _submitParentForm === 'function') _submitParentForm();
                }""")
    except Exception as e:
        log(f"    WARN: Navigation wait after submit timed out or failed: {e}", Colors.YELLOW)
    await asyncio.sleep(4)


async def download_from_collection(
    page: Page,
    context: BrowserContext,
    collection_name: str,
    tmp_zip: Path,
    image_id: int,
) -> bool:
    """
    Data Collections -> collection_name -> Not Downloaded ->
    select our newly-added image -> 1-CLICK DOWNLOAD -> save zip.
    Returns True on success.
    """
    # Force a page reload to refresh the left-hand YUI collections tree
    await page.reload(wait_until="domcontentloaded")
    await asyncio.sleep(4)

    # Click Data Collections tab (SPA navigation keeps YAHOO tree alive)
    tab_clicked = await page.evaluate("""() => {
        const all = [...document.querySelectorAll('a, li a, .yui-nav a')];
        const tab = all.find(el => el.textContent.trim() === 'Data Collections');
        if (tab) { tab.click(); return 'clicked'; }
        return 'not found';
    }""")
    log(f"    -> tab nav: {tab_clicked}", Colors.CYAN)
    await asyncio.sleep(3)

    # Expand My Collections and click the named collection
    coll_clicked = await page.evaluate("""([name]) => {
        try {
            const n = YAHOO.widget.TreeView.getNode('collections', 1);
            if (n && !n.expanded) n.expand();
        } catch(e) {}
        const labels = [...document.querySelectorAll(
            '#collections .ygtvlabel, #collections_tree .ygtvlabel'
        )];
        const myLabel = labels.find(l => l.textContent.trim() === 'My Collections');
        if (myLabel) myLabel.click();
        const found = labels.find(l => {
            const t = l.textContent.trim();
            return t === name || t.startsWith(name + ' (') || t.startsWith(name + '(');
        });
        if (found) { found.click(); return 'ok: ' + found.textContent.trim(); }
        return 'not found: ' + labels.map(l => l.textContent.trim()).filter(Boolean).join(' | ');
    }""", [collection_name])
    log(f"    -> coll click: {coll_clicked}", Colors.CYAN)

    if not coll_clicked.startswith("ok:"):
        log(f"    -> Collection not found on first try. Reloading page and retrying...", Colors.YELLOW)
        await page.reload(wait_until="domcontentloaded")
        await asyncio.sleep(4)
        
        # Click tab again
        await page.evaluate("""() => {
            const all = [...document.querySelectorAll('a, li a, .yui-nav a')];
            const tab = all.find(el => el.textContent.trim() === 'Data Collections');
            if (tab) tab.click();
        }""")
        await asyncio.sleep(3)
        
        # Try finding the collection again
        coll_clicked = await page.evaluate("""([name]) => {
            try {
                const n = YAHOO.widget.TreeView.getNode('collections', 1);
                if (n && !n.expanded) n.expand();
            } catch(e) {}
            const labels = [...document.querySelectorAll(
                '#collections .ygtvlabel, #collections_tree .ygtvlabel'
            )];
            const myLabel = labels.find(l => l.textContent.trim() === 'My Collections');
            if (myLabel) myLabel.click();
            const found = labels.find(l => {
                const t = l.textContent.trim();
                return t === name || t.startsWith(name + ' (') || t.startsWith(name + '(');
            });
            if (found) { found.click(); return 'ok: ' + found.textContent.trim(); }
            return 'not found: ' + labels.map(l => l.textContent.trim()).filter(Boolean).join(' | ');
        }""", [collection_name])
        log(f"    -> coll click retry: {coll_clicked}", Colors.CYAN)

    if not coll_clicked.startswith("ok:"):
        log(f"    FAIL Collection '{collection_name}' not found: {coll_clicked}", Colors.RED)
        return False
    await asyncio.sleep(2)

    # Click Not Downloaded; wait for AJAX
    resp_event: asyncio.Event = asyncio.Event()

    def on_resp(resp):
        if "collectDetail" in resp.url and "NOT_DOWNLOADED" in resp.url:
            resp_event.set()

    page.on("response", on_resp)
    try:
        await page.evaluate("""() => {
            const labels = [...document.querySelectorAll(
                '#collections .ygtvlabel, #collections_tree .ygtvlabel'
            )];
            const notDl = labels.find(l => l.textContent.trim().startsWith('Not Downloaded'));
            if (notDl) notDl.click();
        }""")
        try:
            await asyncio.wait_for(resp_event.wait(), timeout=20)
        except asyncio.TimeoutError:
            pass
    finally:
        page.remove_listener("response", on_resp)
    await asyncio.sleep(2)

    not_dl_cbs = await page.evaluate("""() =>
        [...document.querySelectorAll('input[type="checkbox"][name="checkbox"]')]
            .filter(c => c.offsetParent !== null)
            .length
    """)
    log(f"    -> Not Downloaded view: {not_dl_cbs} checkbox(es)", Colors.CYAN)

    if not_dl_cbs == 0:
        # Fallback: full collection view
        await page.evaluate("""([name]) => {
            const labels = [...document.querySelectorAll(
                '#collections .ygtvlabel, #collections_tree .ygtvlabel'
            )];
            const coll = labels.find(l => {
                const t = l.textContent.trim();
                return t === name || t.startsWith(name + ' (') || t.startsWith(name + '(');
            });
            if (coll) coll.click();
        }""", [collection_name])
        await asyncio.sleep(3)

    # Select the target image_id row (or first visible checkbox as fallback)
    selected = await page.evaluate("""([imgId]) => {
        const id  = String(imgId);
        const idI = 'I' + id;
        const rows = [...document.querySelectorAll('tr')];
        for (const row of rows) {
            if (row.textContent && (row.textContent.includes(id) || row.textContent.includes(idI))) {
                const cb = row.querySelector('input[type="checkbox"]');
                if (cb) { cb.checked = true; cb.click(); return 'row-match: ' + id; }
            }
        }
        const allCbs = [...document.querySelectorAll('input[type="checkbox"][name="checkbox"]')]
            .filter(cb => !cb.closest('#collections') && !cb.closest('#collections_tree'));
        if (allCbs.length > 0) {
            if (!allCbs[0].checked) allCbs[0].click();
            return 'first-cb: ' + allCbs.length + ' visible';
        }
        return 'no checkboxes found';
    }""", [image_id])
    log(f"    -> checkbox select: {selected}", Colors.CYAN)
    await asyncio.sleep(1)

    # Intercept downloadKey AJAX
    dl_key_event: asyncio.Event = asyncio.Event()

    async def on_dl_resp(resp):
        if "downloadKey" in resp.url:
            dl_key_event.set()

    page.on("response", on_dl_resp)

    dl_click = await page.evaluate("""() => {
        const btn = document.getElementById('simple-download-button');
        if (btn) { btn.click(); return 'clicked #simple-download-button'; }
        const all = [...document.querySelectorAll('button, input[type="button"], a')];
        for (const b of all) {
            if ((b.textContent || b.value || '').trim().toUpperCase().includes('1-CLICK')) {
                b.click(); return '1-CLICK: ' + b.textContent.trim();
            }
        }
        return 'not found';
    }""")
    log(f"    -> download click: {dl_click}", Colors.CYAN)

    if "not found" in dl_click:
        page.remove_listener("response", on_dl_resp)
        return False

    try:
        await asyncio.wait_for(dl_key_event.wait(), timeout=60)
    except asyncio.TimeoutError:
        pass
    finally:
        page.remove_listener("response", on_dl_resp)

    link_href: Optional[str] = None
    for _ in range(60):
        await asyncio.sleep(2)
        link_href = await page.evaluate("""() => {
            const link = document.getElementById('simple-download-link');
            if (!link) return null;
            const href = link.href || '';
            return (href && !href.endsWith('#')) ? href : null;
        }""")
        if link_href:
            break

    if not link_href:
        log("    FAIL Download link never populated after 120s", Colors.RED)
        return False
    log(f"    -> link: {link_href[:80]}", Colors.CYAN)

    # Download -- Playwright event first, requests fallback
    try:
        async with page.expect_download(timeout=DL_TIMEOUT_MS) as dl_info:
            await page.evaluate("""() => {
                const link = document.getElementById('simple-download-link');
                if (link) link.click();
            }""")
        dl = await dl_info.value
        await dl.save_as(str(tmp_zip))
        return tmp_zip.exists() and tmp_zip.stat().st_size > 0
    except Exception as exc:
        log(f"    -> Playwright download failed ({exc!s:.80}), trying requests...", Colors.YELLOW)

    try:
        import requests
        cookies_dict = {c["name"]: c["value"] for c in await context.cookies()}
        r = requests.get(
            link_href,
            cookies=cookies_dict,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True,
            timeout=300,
        )
        if r.status_code == 200:
            with open(tmp_zip, "wb") as fh:
                for chunk in r.iter_content(65536):
                    if chunk:
                        fh.write(chunk)
            return tmp_zip.exists() and tmp_zip.stat().st_size > 0
    except Exception as exc2:
        log(f"    FAIL requests fallback failed: {exc2}", Colors.RED)

    return False


def save_dicom_zip(raw_zip: Path, output_dir: Path) -> Optional[tuple[int, Path]]:
    """
    Peek inside the downloaded ZIP for the canonical {subject_id}_{image_id}
    names, then move it into output_dir.  Returns (image_id, dest) or None.
    """
    image_id: Optional[int] = None
    subject_id: Optional[str] = None
    try:
        with zipfile.ZipFile(raw_zip, "r") as zf:
            for name in zf.namelist():
                for part in Path(name).parts:
                    if image_id is None:
                        m = re.match(r"^I(\d+)$", part)
                        if m:
                            image_id = int(m.group(1))
                    if subject_id is None:
                        m = re.match(r"^(\d{3}_S_\d{4})$", part)
                        if m:
                            subject_id = m.group(1)
                if image_id is not None and subject_id is not None:
                    break
    except Exception as e:
        log(f"  FAIL Could not read ZIP {raw_zip.name}: {e}", Colors.RED)
        return None

    if image_id is None or subject_id is None:
        log(
            f"  WARN Could not determine subject/image_id from {raw_zip.name} "
            f"(subject={subject_id!r}, image_id={image_id!r})",
            Colors.YELLOW,
        )
        return None

    dest = output_dir / f"{subject_id}_{image_id}.zip"
    shutil.move(str(raw_zip), str(dest))
    log(f"  OK saved -> {dest.name}", Colors.GREEN)
    return (image_id, dest)


# ── BIDS manifest helpers ──────────────────────────────────────────────────────


def read_manifest_subjects() -> dict[int, dict]:
    """Read adni_bids_manifest.csv -> image_id -> row dict for our 11 targets."""
    result: dict[int, dict] = {}
    if not MANIFEST_CSV.exists():
        return result
    target_ids = {t["image_id"] for t in TARGETS}
    with MANIFEST_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                iid = int(row["image_id"])
            except (KeyError, ValueError):
                continue
            if iid in target_ids:
                result[iid] = row
    return result


def delete_stale_bids_outputs(manifest_rows: dict[int, dict], dry_run: bool) -> None:
    """Delete existing (failed) BIDS NIfTI outputs so convert_to_bids won't skip them."""
    for iid, row in manifest_rows.items():
        dest_nii  = BIDS_ROOT / row["dest_relpath"]
        dest_json = dest_nii.with_suffix("").with_suffix(".json")
        for p in (dest_nii, dest_json):
            if p.exists():
                if dry_run:
                    log(f"  [dry-run] would delete BIDS output: {p}", Colors.YELLOW)
                else:
                    p.unlink()
                    log(f"  deleted stale BIDS output: {p}", Colors.YELLOW)


# ── Main async download loop ───────────────────────────────────────────────────


async def run_downloads(
    targets: list[dict],
    username: str,
    password: str,
    headless: bool,
    dry_run: bool,
) -> tuple[list[int], list[int]]:
    """Re-download each target. Returns (succeeded_ids, failed_ids)."""
    tmp_dir = ZIP_DIR.parent / "_redownload_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    succeeded: list[int] = []
    failed:    list[int] = []

    async with async_playwright() as pw:
        context = await open_browser_context(pw, headless)
        # Always open a fresh page; context.pages[0] from a persistent profile
        # may carry stale navigation state that interferes with login detection.
        page = await context.new_page()

        log("\n-- Logging in to LONI IDA ------------------------------------------", Colors.BOLD)
        logged_in = await loni_login(page, username, password)
        if not logged_in:
            log(
                "\nCould not log in. Check credentials in .env.\n"
                "Run with --headless false to see the browser.",
                Colors.RED,
            )
            await context.close()
            return succeeded, [t["image_id"] for t in targets]

        log(f"\n-- Downloading {len(targets)} image(s) --", Colors.BOLD)

        for idx, target in enumerate(targets, start=1):
            image_id   = target["image_id"]
            subject_id = target["subject_id"]
            note       = target.get("note", "")
            dest_zip   = ZIP_DIR / f"{subject_id}_{image_id}.zip"

            log(
                f"\n[{idx}/{len(targets)}] image_id={image_id}  "
                f"subject={subject_id}  ({note})",
                Colors.BOLD,
            )

            # Skip if already freshly downloaded
            if dest_zip.exists() and dest_zip.stat().st_size > 0:
                log(f"  -> ZIP already present: {dest_zip.name} -- skipping", Colors.YELLOW)
                succeeded.append(image_id)
                continue

            if dry_run:
                log(f"  [dry-run] would re-download image_id={image_id}", Colors.YELLOW)
                succeeded.append(image_id)
                continue

            # Step A: Advanced Search
            log(f"  -> [{image_id}] Searching LONI Advanced Search...", Colors.CYAN)
            found = await search_for_image(page, image_id)
            if not found:
                if "login" in page.url.lower():
                    log("  FAIL Session expired -- stopping.", Colors.RED)
                    failed.extend(t["image_id"] for t in targets[idx - 1:])
                    break
                log(f"  FAIL [{image_id}] No search results -- skipping", Colors.RED)
                failed.append(image_id)
                continue

            # Step B: Add to collection
            log(f"  -> [{image_id}] Adding to '{COLLECTION_NAME}'...", Colors.CYAN)
            await add_to_collection(page, COLLECTION_NAME)

            # Step C: Download from collection
            log(f"  -> [{image_id}] Downloading from '{COLLECTION_NAME}'...", Colors.CYAN)
            tmp_zip = tmp_dir / f"{image_id}_raw.zip"
            tmp_zip.unlink(missing_ok=True)

            ok = await download_from_collection(
                page, context, COLLECTION_NAME, tmp_zip, image_id
            )
            if not ok or not tmp_zip.exists():
                log(f"  FAIL [{image_id}] Download failed", Colors.RED)
                failed.append(image_id)
                continue

            size_mb = tmp_zip.stat().st_size / (1024 * 1024)
            log(f"  -> [{image_id}] ZIP downloaded ({size_mb:.1f} MB). Saving...", Colors.CYAN)

            # Step D: Rename into __dicom_zips_flat__
            result = save_dicom_zip(tmp_zip, ZIP_DIR)
            if result is None:
                log(f"  FAIL [{image_id}] Could not identify subject/image_id in ZIP", Colors.RED)
                failed.append(image_id)
            else:
                succeeded.append(image_id)

            await asyncio.sleep(2)  # polite delay

        await context.close()

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return succeeded, failed


# ── BIDS conversion ────────────────────────────────────────────────────────────


def run_bids_conversion(subjects: list[str], dry_run: bool) -> int:
    """Run convert_to_bids.py --subjects ... Returns exit code."""
    if not CONVERT_SCRIPT.exists():
        log(f"FAIL convert_to_bids.py not found at {CONVERT_SCRIPT}", Colors.RED)
        return 1

    cmd = [sys.executable, str(CONVERT_SCRIPT), "--subjects"] + subjects
    log("\n-- Running BIDS conversion ------------------------------------------", Colors.BOLD)
    log(f"  Command: {' '.join(cmd)}", Colors.CYAN)

    if dry_run:
        log(f"  [dry-run] would convert subjects: {subjects}", Colors.YELLOW)
        return 0

    result = subprocess.run(cmd, cwd=str(CONVERT_SCRIPT.parent))
    return result.returncode


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--image-ids", nargs="*", type=int, default=None, metavar="IMAGE_ID",
        help="Limit to specific image_ids (default: all 11 targets)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without modifying anything",
    )
    p.add_argument(
        "--headless",
        type=lambda v: v.lower() != "false",
        default=True, metavar="true|false",
        help="Run Chromium headlessly (default: true). Use false to debug.",
    )
    p.add_argument(
        "--no-bids", action="store_true",
        help="Skip BIDS conversion (download raw zips only)",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip re-downloading; only run BIDS conversion for the targets",
    )
    return p.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"redownload_and_bids_{timestamp}.log"
    global _logger
    _logger = setup_logging(log_path)

    log("\n" + "=" * 60, Colors.BOLD)
    log("  ADNI Re-download + BIDS Conversion", Colors.BOLD)
    log("=" * 60, Colors.BOLD)
    log(f"  Log file   : {log_path}", Colors.CYAN)
    log(f"  ZIP dir    : {ZIP_DIR}", Colors.CYAN)
    log(f"  BIDS root  : {BIDS_ROOT}", Colors.CYAN)
    log(f"  Headless   : {args.headless}", Colors.CYAN)
    log(f"  Dry-run    : {args.dry_run}", Colors.CYAN)

    # Filter targets
    targets = TARGETS
    if args.image_ids:
        requested = set(args.image_ids)
        targets = [t for t in TARGETS if t["image_id"] in requested]
        if not targets:
            log(f"None of the requested image_ids match known targets: {args.image_ids}", Colors.RED)
            return 1

    log(f"\n  Targets ({len(targets)}):", Colors.BOLD)
    for t in targets:
        log(
            f"    image_id={t['image_id']:>7}  subject={t['subject_id']}  "
            f"[{t['scan_type']}]  {t['note']}",
            Colors.CYAN,
        )

    # Read manifest for BIDS output paths
    manifest_rows = read_manifest_subjects()

    # Step 0: delete stale BIDS outputs
    log("\n-- Step 0: Remove stale BIDS outputs --------------------------------", Colors.BOLD)
    delete_stale_bids_outputs(manifest_rows, args.dry_run)

    # Step 1: delete stale DICOM zips
    log("\n-- Step 1: Remove stale DICOM zips ----------------------------------", Colors.BOLD)
    for t in targets:
        old_zip = ZIP_DIR / f"{t['subject_id']}_{t['image_id']}.zip"
        if old_zip.exists():
            if args.dry_run:
                log(f"  [dry-run] would delete: {old_zip.name}", Colors.YELLOW)
            else:
                old_zip.unlink()
                log(f"  deleted stale zip: {old_zip.name}", Colors.YELLOW)
        else:
            log(f"  (not present): {old_zip.name}", Colors.CYAN)

    # Step 2: re-download
    succeeded_ids: list[int] = []
    failed_ids:    list[int] = []

    if args.skip_download or args.dry_run:
        if args.dry_run:
            log("\n-- Step 2: Re-download from LONI IDA (DRY-RUN -- no browser launched) --", Colors.YELLOW)
            for t in targets:
                log(f"  [dry-run] would re-download image_id={t['image_id']}  subject={t['subject_id']}", Colors.YELLOW)
        else:
            log("\n-- Skipping download (--skip-download set) --", Colors.YELLOW)
        succeeded_ids = [t["image_id"] for t in targets]
    else:
        log("\n-- Step 2: Re-download from LONI IDA --------------------------------", Colors.BOLD)
        username, password = load_credentials()
        succeeded_ids, failed_ids = asyncio.run(
            run_downloads(targets, username, password, args.headless, args.dry_run)
        )

    # Step 3: BIDS conversion
    bids_rc = 0
    if not args.no_bids:
        subjects_to_convert = sorted({
            t["subject_id"]
            for t in targets
            if t["image_id"] in succeeded_ids
        })
        if subjects_to_convert:
            bids_rc = run_bids_conversion(subjects_to_convert, args.dry_run)
        else:
            log("\nNo successful downloads -- nothing to convert.", Colors.YELLOW)
    else:
        log("\n-- Skipping BIDS conversion (--no-bids set) --", Colors.YELLOW)

    # Summary
    log("\n" + "=" * 60, Colors.BOLD)
    log("  Summary", Colors.BOLD)
    log("=" * 60, Colors.BOLD)
    log(f"  Downloads succeeded : {len(succeeded_ids)}  {succeeded_ids}", Colors.GREEN)
    if failed_ids:
        log(f"  Downloads failed    : {len(failed_ids)}  {failed_ids}", Colors.RED)
        log(
            "\n  Tip: If these fail again after re-download, the series was\n"
            "  uploaded incomplete by the scanner site -- exclude from analysis.",
            Colors.YELLOW,
        )
    log(
        f"  BIDS conversion rc  : {bids_rc}",
        Colors.GREEN if bids_rc == 0 else Colors.RED,
    )
    log(f"  Log file            : {log_path}", Colors.CYAN)

    return 0 if (not failed_ids and bids_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
