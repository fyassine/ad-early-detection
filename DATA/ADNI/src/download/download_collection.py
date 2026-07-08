#!/usr/bin/env python3
"""
download_collection.py
======================
Download all images from an existing LONI IDA collection (e.g. mci-all-v2).
Works by navigating directly to Data Collections → named collection → Not Downloaded,
selecting a batch of items, downloading the ZIP, and saving the raw DICOM ZIP
as {subject_id}_{image_id}.zip, repeating until nothing is left in
'Not Downloaded'.

Usage
-----
    python download_collection.py                         # uses mci-all-v2
    python download_collection.py --collection mci-all-v2 --batch-size 5
    python download_collection.py --headless false        # show browser

Resume safety
-------------
  Already-saved {subject_id}_{image_id}.zip files are skipped. Both ids are
  read directly from the ZIP's internal LONI directory structure
  (ADNI/{subject_id}/.../I{image_id}/*.dcm).

NIfTI conversion (dcm2niix) is implemented in extract_niftis() but not
currently used — DICOM ZIPs are kept as-is.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pip install pandas")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: pip install python-dotenv")
    sys.exit(1)

try:
    from playwright.async_api import BrowserContext, Page, async_playwright
except ImportError:
    print("ERROR: pip install playwright && playwright install chromium")
    sys.exit(1)

from loni_session import is_logged_in, open_context

# ── Config ─────────────────────────────────────────────────────────────────────

LONI_BASE_URL = "https://ida.loni.usc.edu"
LONI_LOGIN_URL = f"{LONI_BASE_URL}/login.jsp?project=ADNI"
LONI_ADV_SEARCH_URL = (
    f"{LONI_BASE_URL}/pages/access/search.jsp"
    "?project=ADNI&tab=advSearch&page=SEARCH&subPage=NEW_ADV_QUERY"
)

# src/download/ → parent is src/ → parent is ADNI/ → parent is DATA/ → parent is project root
SRC_DIR  = Path(__file__).resolve().parent          # .../src/download/
DOWNLOAD_DIR = SRC_DIR                             # alias for clarity
ADNI_SRC_DIR = SRC_DIR.parent                      # .../src/
DATA_DIR = ADNI_SRC_DIR.parent                      # .../ADNI/
PROJECT_ROOT = DATA_DIR.parent.parent               # ad-early-detection/

DEFAULT_COLLECTION   = "mci-all-v2"
DEFAULT_OUTPUT_DIR   = DATA_DIR / "__dicom_zips_flat__"
# Converter's output dir (convert_dicom_zips.py) -- consulted only to build the
# local "already have" skip-set below, so a converted-and-cleaned-up image
# isn't re-downloaded.
NIFTI_OUTPUT_DIR     = DATA_DIR / "__fmri_wholebrain_sch200_flat__"

ZIP_NAME_RE = re.compile(r"^\d{3}_S_\d{4}_(\d+)\.zip$")
NII_NAME_RE = re.compile(r"^\d{3}_S_\d{4}_(\d+)(?:_\d+)?\.nii\.gz$")
DEFAULT_METADATA_CSV = DATA_DIR / "__metadata__" / "Extended_rsfMRI_MCI_Longitudinal_14May2026.csv"
DEFAULT_ENV_FILE     = ADNI_SRC_DIR / ".env"


def _default_log_file(script_name: str = "adni_collection_download") -> Path:
    """Return logs/adni-download/<YYYYMMDD_HHMMSS>/<script_name>.log."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / "adni-download" / ts / f"{script_name}.log"


DEFAULT_LOG_FILE = _default_log_file()
DEFAULT_HEADLESS     = True   # run headless by default (no display server needed)
DEFAULT_BATCH_SIZE   = 5    # images per ZIP — increase only if server handles it
NAV_TIMEOUT_MS       = 60_000
DL_TIMEOUT_MS        = 600_000   # 10 min for large zips
MAX_CONSECUTIVE_FAILURES = 3   # fail loudly instead of looping forever on a stuck session
MAX_STALE_PAGE_RETRIES = 6   # reloads to try before giving up on a stale 'Not Downloaded' page

# ── Logging ────────────────────────────────────────────────────────────────────


class Colors:
    GREEN  = "\033[0;32m"
    RED    = "\033[0;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[0;36m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


_logger: logging.Logger | None = None


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("coll_dl")
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
    if _logger:
        clean = re.sub(r"\033\[[0-9;]*m", "", msg)
        _logger.info(clean)
    if color:
        print(f"{color}{msg}{Colors.RESET}")
    else:
        print(msg)


# ── Metadata ───────────────────────────────────────────────────────────────────


def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Load metadata CSV and build a lookup table keyed by (subject_id, acq_date)."""
    df = pd.read_csv(csv_path)
    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Find the image_id column
    for col in ("image_data_id", "image id", "imageid", "image_id"):
        if col in df.columns:
            df = df.rename(columns={col: "image_id"})
            break
    # Find subject/date columns
    for col in ("subject_id", "subject", "subjectid"):
        if col in df.columns:
            df = df.rename(columns={col: "subject_id"})
            break
    for col in ("acq_date", "acqdate", "acquisition_date", "exam_date", "examdate"):
        if col in df.columns:
            df = df.rename(columns={col: "acq_date"})
            break
    df["image_id"] = pd.to_numeric(df["image_id"], errors="coerce")
    df = df.dropna(subset=["image_id"])
    df["image_id"] = df["image_id"].astype(int)
    return df


def build_lookup(df: pd.DataFrame) -> dict:
    """Map (subject_id, acq_date_normalised) → image_id."""
    lookup = {}
    for _, row in df.iterrows():
        subj = str(row.get("subject_id", "")).strip()
        date = str(row.get("acq_date", "")).strip()
        # Normalise date: YYYY-MM-DD or MM/DD/YYYY → YYYYMMDD
        date_norm = re.sub(r"[^0-9]", "", date)
        key = (subj, date_norm)
        lookup[key] = int(row["image_id"])
    return lookup


# ── Login ──────────────────────────────────────────────────────────────────────


async def loni_login(page: Page, username: str, password: str) -> bool:
    log(f"  → Navigating to {LONI_LOGIN_URL}", Colors.CYAN)
    await page.goto(LONI_LOGIN_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
    await asyncio.sleep(5)

    if await is_logged_in(page):
        log("  ✓ Reusing existing LONI session (persistent profile)", Colors.GREEN)
        return True

    # Accept cookie policy
    await page.evaluate("""() => {
        const el = document.querySelector('.ida-cookie-policy-accept');
        if (el) el.click();
    }""")
    await asyncio.sleep(2)

    # Click login nav button
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

    # Fill credentials
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

    # Submit
    await page.evaluate("""() => {
        const sels = ['.login-btn', 'span.login-btn', 'button[type="submit"]', 'input[type="submit"]'];
        for (const s of sels) {
            const el = document.querySelector(s);
            if (el) { el.click(); return; }
        }
    }""")

    try:
        await page.wait_for_url(lambda u: "login" not in u.lower(), timeout=20_000)
    except Exception:
        pass
    await asyncio.sleep(2)

    if "login" in page.url.lower():
        log("  ✗ Login failed", Colors.RED)
        return False
    log(f"  ✓ Logged in: {page.url}", Colors.GREEN)
    return True


# ── Collection navigation ──────────────────────────────────────────────────────


async def navigate_to_collection(page: Page, collection_name: str) -> bool:
    """
    From the current search page, click the Data Collections tab (SPA navigation,
    no page.goto — keeps YAHOO TreeView alive), then find and click the named collection.
    Returns True on success.
    """
    # Click the Data Collections tab in-place (SPA keeps tree cache alive)
    tab = await page.evaluate("""() => {
        const all = [...document.querySelectorAll('a, li a, .yui-nav a')];
        const tab = all.find(el => el.textContent.trim() === 'Data Collections');
        if (tab) { tab.click(); return 'clicked'; }
        return 'not found: ' + all.map(a => a.textContent.trim()).filter(Boolean).slice(0, 8).join(' | ');
    }""")
    log(f"  ↳ tab click: {tab}", Colors.CYAN)
    await asyncio.sleep(3)

    # Expand My Collections and find the named collection
    result = await page.evaluate("""([name]) => {
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
    log(f"  ↳ collection click: {result}", Colors.CYAN)

    if not result.startswith("ok:"):
        log(f"  ✗ Collection '{collection_name}' not found in tree", Colors.RED)
        return False
    await asyncio.sleep(2)
    return True


async def count_not_downloaded(page: Page) -> tuple[int, str | None]:
    """
    Click 'Not Downloaded' subtree and return (number of visible item
    checkboxes, the tree node's label e.g. 'Not Downloaded (781)').

    Clicking the tree node triggers an async
    /pages/ajax/search/collectDetail?...subset=NOT_DOWNLOADED... request
    that *replaces* the table content. We must wait for that response
    before reading cell11_N — otherwise we read the stale table from
    whichever view was active before (e.g. the collection's default 'All'
    view from navigate_to_collection, where row 0 is always the same fixed
    image regardless of download status, causing every batch to re-select
    that same image).
    """
    resp_event = asyncio.Event()

    def on_response(resp):
        if "collectDetail" in resp.url and "NOT_DOWNLOADED" in resp.url:
            resp_event.set()

    page.on("response", on_response)
    try:
        label = await page.evaluate("""() => {
            const labels = [...document.querySelectorAll(
                '#collections .ygtvlabel, #collections_tree .ygtvlabel'
            )];
            const notDl = labels.find(l => l.textContent.trim().startsWith('Not Downloaded'));
            if (notDl) { notDl.click(); return notDl.textContent.trim(); }
            return null;
        }""")
        try:
            await asyncio.wait_for(resp_event.wait(), timeout=20)
        except asyncio.TimeoutError:
            pass
    finally:
        page.remove_listener("response", on_response)

    # Let the DOM finish rendering the new rows after the AJAX response.
    await asyncio.sleep(1)
    count = await page.evaluate("""() =>
        [...document.querySelectorAll('input[type="checkbox"][name="checkbox"]')]
            .filter(c => c.offsetParent !== null)
            .length
    """)
    return count, label


async def visible_not_downloaded_ids(page: Page) -> list[int]:
    """image_ids (as ints) of all visible 'Not Downloaded' row checkboxes."""
    values = await page.evaluate("""() => {
        const out = [];
        let idx = 0;
        let cell;
        while ((cell = document.getElementById('cell11_' + idx))) {
            for (const cb of cell.querySelectorAll('input[type="checkbox"][name="checkbox"]')) {
                out.push(cb.value);
            }
            idx++;
        }
        return out;
    }""")
    return [int(v[1:]) for v in values if v.startswith("I")]


# ── Download one batch ─────────────────────────────────────────────────────────


async def download_batch(
    page: Page,
    context: BrowserContext,
    batch_size: int,
    tmp_zip: Path,
    skip_image_ids: set[int],
) -> bool:
    """Select up to batch_size 'Not Downloaded' items (skipping image_ids we
    already have locally) and download the ZIP."""

    # Select up to batch_size individual row checkboxes (cell11_0, cell11_1, ...
    # each containing <input name="checkbox" value="I{image_id}">), skipping
    # any row whose image_id is in skip_image_ids (already downloaded locally,
    # even if the server's 'Not Downloaded' view still lists it).
    # Deliberately do NOT click #selectAllCheckBox: its handleCheckAll() sets
    # CHECK_BOX_MANAGER._isCheckAll = true, after which getNumberOfDownloadables()
    # returns numberOfAccessibleRows — i.e. every "Not Downloaded" item in the
    # whole collection (100+), not just this batch. That caused the server-side
    # ZIP to never finish within the 120s wait.
    skip_values = [f"I{i}" for i in skip_image_ids]
    selected = await page.evaluate("""([n, skipValues]) => {
        const skip = new Set(skipValues);
        const rowBoxes = [];
        let idx = 0;
        let cell;
        while ((cell = document.getElementById('cell11_' + idx))) {
            for (const cb of cell.querySelectorAll('input[type="checkbox"][name="checkbox"]')) {
                if (!skip.has(cb.value)) rowBoxes.push(cb);
            }
            idx++;
        }
        const toCheck = rowBoxes.slice(0, n);
        let clicked = 0;
        for (const cb of toCheck) {
            if (!cb.checked) { cb.click(); clicked++; }
        }
        return clicked;
    }""", [batch_size, skip_values])
    log(f"  ↳ selected {selected} checkboxes", Colors.CYAN)

    if selected == 0:
        return False

    # Intercept the downloadKey AJAX response (signals server has prepared ZIP)
    dl_key_event = asyncio.Event()
    async def on_response(resp):
        if "downloadKey" in resp.url:
            dl_key_event.set()
    page.on("response", on_response)

    # Click 1-CLICK DOWNLOAD
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
    log(f"  ↳ download click: {dl_click}", Colors.CYAN)

    if "not found" in dl_click:
        return False

    # Wait for downloadKey then for the link href
    try:
        await asyncio.wait_for(dl_key_event.wait(), timeout=60)
    except asyncio.TimeoutError:
        pass

    link_href = None
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
        log("  ✗ Download link never populated after 120s", Colors.RED)
        return False

    log(f"  ↳ download link: {link_href}", Colors.CYAN)

    # Try Playwright expect_download, then fall back to requests
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
        log(f"  ↳ Playwright download failed ({exc!s:.80}), trying requests...", Colors.YELLOW)

    try:
        import requests  # type: ignore
        cookies_dict = {c["name"]: c["value"] for c in await context.cookies()}
        r = requests.get(
            link_href, cookies=cookies_dict,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True, timeout=120,
        )
        if r.status_code == 200:
            with open(tmp_zip, "wb") as fh:
                for chunk in r.iter_content(65536):
                    if chunk:
                        fh.write(chunk)
            return tmp_zip.exists() and tmp_zip.stat().st_size > 0
    except Exception as exc2:
        log(f"  ✗ requests fallback failed: {exc2}", Colors.RED)

    return False


def local_image_ids(zip_dir: Path, nifti_dir: Path) -> set[int]:
    """
    image_ids already present locally, either as a saved
    {subject_id}_{image_id}.zip in zip_dir or a converted
    {subject_id}_{image_id}.nii.gz in nifti_dir.

    Used to skip 'Not Downloaded' rows for images we already have, even if
    LONI's server-side 'Not Downloaded' bookkeeping hasn't caught up (observed:
    a successfully-downloaded image can keep reappearing as the first
    'Not Downloaded' row indefinitely).
    """
    ids: set[int] = set()
    if zip_dir.exists():
        for p in zip_dir.glob("*.zip"):
            m = ZIP_NAME_RE.match(p.name)
            if m:
                ids.add(int(m.group(1)))
    if nifti_dir.exists():
        for p in nifti_dir.glob("*.nii.gz"):
            m = NII_NAME_RE.match(p.name)
            if m:
                ids.add(int(m.group(1)))
    return ids


# ── Save raw DICOM ZIP ─────────────────────────────────────────────────────────


def save_dicom_zip(zip_path: Path, output_dir: Path) -> tuple[int, Path] | None:
    """
    Identify the LONI image_id (I{image_id} dir) and ADNI subject_id
    (NNN_S_NNNN) from zip_path's internal paths, then move the raw ZIP to
    output_dir/{subject_id}_{image_id}.zip (flat, no per-subject
    subdirectories — same naming convention as extract_niftis). Returns
    (image_id, dest), or None if either id couldn't be determined from the
    archive's internal paths.
    """
    image_id = None
    subject_id = None
    with zipfile.ZipFile(zip_path, "r") as zf:
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

    if image_id is None or subject_id is None:
        log(
            f"  ! Could not identify subject/image_id in {zip_path.name} "
            f"(subject={subject_id!r}, image_id={image_id!r})", Colors.YELLOW,
        )
        return None

    dest = output_dir / f"{subject_id}_{image_id}.zip"
    if dest.exists():
        log(f"  ↳ skip {image_id} (already exists)", Colors.YELLOW)
        zip_path.unlink(missing_ok=True)
    else:
        shutil.move(str(zip_path), str(dest))
        log(f"  ✓ {image_id} → {dest}", Colors.GREEN)
    return (image_id, dest)


# ── Extract NIfTIs (defined but unused — kept for possible future re-enabling) ──


def extract_niftis(zip_path: Path, output_dir: Path, lookup: dict) -> list[tuple[int, Path]]:
    """
    Extract DICOMs from zip_path, convert each series with dcm2niix,
    and name output files as {subject_id}_{image_id}.nii.gz, with image_id
    taken from LONI's I{image_id} directory in the ZIP (falling back to the
    metadata lookup table by (subject_id, acq_date) if that's absent).
    Returns list of (image_id, nifti_path) for successfully extracted images.
    """
    tmp_root = zip_path.parent / f"_extract_{zip_path.stem}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_root)

        # Find all DICOM series directories (leaf directories with .dcm files)
        series_dirs = set()
        for f in tmp_root.rglob("*.dcm"):
            series_dirs.add(f.parent)
        if not series_dirs:
            # Some LONI ZIPs have no .dcm extension — find deepest dirs with files
            for f in tmp_root.rglob("*"):
                if f.is_file() and not f.suffix.lower() in (".xml", ".csv", ".json"):
                    series_dirs.add(f.parent)

        for sdir in sorted(series_dirs):
            nii_tmp = sdir / "_nii"
            nii_tmp.mkdir(exist_ok=True)
            try:
                subprocess.run(
                    ["dcm2niix", "-z", "y", "-f", "%i_%s_%d", "-o", str(nii_tmp), str(sdir)],
                    check=True, capture_output=True, timeout=120,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                log(f"  ✗ dcm2niix failed for {sdir.name}: {e}", Colors.RED)
                continue

            # LONI ZIPs nest each series' DICOMs directly under an I{image_id}
            # directory (the ADNI image UID), e.g. .../2022-07-01_.../I1600185/*.dcm.
            # That's the canonical identifier — use it directly instead of
            # guessing PatientID+StudyDate and looking them up in the metadata
            # CSV (which keys on clinical visit "examdate", not the scan's
            # actual acquisition date, and misses real entries).
            dir_match = re.match(r"^I(\d+)$", sdir.name)
            dir_image_id = int(dir_match.group(1)) if dir_match else None

            # Process each generated NIfTI
            for nii in nii_tmp.glob("*.nii.gz"):
                subj = _guess_subject_id(nii.stem)

                if dir_image_id is not None:
                    image_id = dir_image_id
                else:
                    date_norm = _guess_date(sdir)
                    image_id = lookup.get((subj, date_norm))

                if image_id is None:
                    log(f"  ! No image_id for subj={subj!r} dir={sdir.name!r} file={nii.name}", Colors.YELLOW)
                    # Save with original dcm2niix name as fallback
                    dest = output_dir / "unmatched" / nii.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(nii), str(dest))
                    results.append((0, dest))
                    continue

                # Destination: output_dir/{subject_id}_{image_id}.nii.gz (flat)
                dest = output_dir / f"{subj}_{image_id}.nii.gz"
                if dest.exists():
                    log(f"  ↳ skip {image_id} (already exists)", Colors.YELLOW)
                else:
                    shutil.move(str(nii), str(dest))
                    log(f"  ✓ {image_id} → {dest}", Colors.GREEN)
                results.append((image_id, dest))

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return results


def _guess_subject_id(stem: str) -> str:
    """Extract ADNI subject ID (NNN_S_NNNN) from a dcm2niix filename stem."""
    m = re.search(r"(\d{3}_S_\d{4})", stem)
    return m.group(1) if m else stem.split("_")[0]


def _guess_date(sdir: Path) -> str:
    """
    Try to find acquisition date from DICOM headers in sdir.
    Falls back to directory name parsing.
    """
    # Try reading StudyDate from first DICOM file
    for f in sdir.iterdir():
        if f.is_file():
            try:
                import pydicom
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                date = str(getattr(ds, "StudyDate", "")).strip()
                if len(date) == 8:
                    return date
            except Exception:
                break
    # Fallback: parse date from directory path (LONI often has date in path)
    for part in reversed(sdir.parts):
        m = re.search(r"(\d{4})[_\-](\d{2})[_\-](\d{2})", part)
        if m:
            return m.group(1) + m.group(2) + m.group(3)
        m = re.search(r"(\d{8})", part)
        if m:
            return m.group(1)
    return ""


# ── Main loop ──────────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> None:
    global _logger
    log_file = Path(args.log_file)
    _logger = setup_logging(log_file)

    # Load credentials
    env_file = Path(args.env_file)
    if env_file.exists():
        load_dotenv(env_file)
    username = (
        os.environ.get("LONI_USERNAME")
        or os.environ.get("IDA_USERNAME")
        or os.environ.get("ADNI_USERNAME", "")
    )
    password = (
        os.environ.get("LONI_PASSWORD")
        or os.environ.get("IDA_PASSWORD")
        or os.environ.get("ADNI_PASSWORD", "")
    )
    if not username or not password:
        log("ERROR: set LONI_USERNAME / LONI_PASSWORD in .env", Colors.RED)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = DATA_DIR / ".tmp_coll_dl"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n{'═'*60}")
    log(f"  ADNI Collection Downloader")
    log(f"{'═'*60}")
    log(f"  Collection : {args.collection}")
    log(f"  Batch size : {args.batch_size}")
    log(f"  Output dir : {output_dir}")
    log(f"  Log file   : {log_file}")
    log(f"{'═'*60}\n")

    headless = str(args.headless).lower() not in ("false", "0", "no")

    async with async_playwright() as pw:
        context = await open_context(pw, headless)
        browser = context
        page = await context.new_page()

        # ── Login ──────────────────────────────────────────────────────────────
        log("\n── Logging in ──────────────────────────────────────────────")
        ok = await loni_login(page, username, password)
        if not ok:
            log(
                "Could not log in. If this is the first run, LONI now requires a "
                "reCAPTCHA that a scripted browser cannot solve — rerun once with "
                "--headless false and solve it by hand; the session persists in "
                "DATA/ADNI/.loni_profile for later headless runs.",
                Colors.RED,
            )
            await browser.close()
            sys.exit(1)

        # ── Navigate to the Search page (SPA entry point) ──────────────────────
        await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)

        # ── Navigate to collection ─────────────────────────────────────────────
        log(f"\n── Navigating to collection '{args.collection}' ────────────")
        ok = await navigate_to_collection(page, args.collection)
        if not ok:
            log("FATAL: could not find collection in tree. Exiting.", Colors.RED)
            await browser.close()
            sys.exit(1)

        # ── Batch download loop ────────────────────────────────────────────────
        batch_num = 0
        total_ok = 0
        total_fail = 0
        consecutive_failures = 0
        log(f"\n── Starting batch download (batch_size={args.batch_size}) ────")

        while True:
            batch_num += 1
            log(f"\n[Batch {batch_num}]")

            # Count and select Not Downloaded items. The collectDetail AJAX
            # response that count_not_downloaded() waits for is sometimes a
            # stale/cached page whose visible rows are all images we already
            # have locally (observed: a fixed set of already-downloaded ids
            # alternating with the true remaining set across reloads, ~50/50).
            # Retry the reload+count until at least one visible row is
            # genuinely new.
            skip_ids = local_image_ids(output_dir, NIFTI_OUTPUT_DIR)
            nd_count = 0
            nd_label = None
            visible_ids: list[int] = []
            lost_session = False
            for _ in range(MAX_STALE_PAGE_RETRIES):
                nd_count, nd_label = await count_not_downloaded(page)
                if nd_count == 0:
                    break
                visible_ids = await visible_not_downloaded_ids(page)
                if any(i not in skip_ids for i in visible_ids):
                    break
                log(
                    f"  ↳ Not Downloaded page looks stale (all {len(visible_ids)} "
                    f"visible rows already downloaded); reloading...",
                    Colors.YELLOW,
                )
                await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                await asyncio.sleep(5)
                ok = await navigate_to_collection(page, args.collection)
                if not ok:
                    lost_session = True
                    break

            log(f"  ↳ Not Downloaded: {nd_count} items visible ({nd_label})")

            if lost_session:
                log("FATAL: lost session and cannot recover.", Colors.RED)
                break
            if nd_count == 0:
                log("  ✓ No more 'Not Downloaded' items — done!", Colors.GREEN)
                break
            if not any(i not in skip_ids for i in visible_ids):
                log(
                    f"FATAL: all visible 'Not Downloaded' rows already downloaded "
                    f"after {MAX_STALE_PAGE_RETRIES} reloads. Investigate manually.",
                    Colors.RED,
                )
                break

            tmp_zip = tmp_dir / f"batch_{batch_num:04d}.zip"
            tmp_zip.unlink(missing_ok=True)

            downloaded = await download_batch(page, context, args.batch_size, tmp_zip, skip_ids)

            if not downloaded:
                log(f"  ✗ Batch {batch_num} download failed", Colors.RED)
                total_fail += args.batch_size
                consecutive_failures += 1

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log(
                        f"FATAL: {consecutive_failures} consecutive batch failures. "
                        f"Stopping -- investigate manually.",
                        Colors.RED,
                    )
                    break

                # Full page reload to reset state for next attempt (see
                # comment on the success path below for why an in-place
                # tree re-click isn't enough).
                await asyncio.sleep(3)
                await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                await asyncio.sleep(5)
                ok = await navigate_to_collection(page, args.collection)
                if not ok:
                    log("  ↳ Re-logging in...", Colors.YELLOW)
                    await loni_login(page, username, password)
                    await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                    await asyncio.sleep(5)
                    ok = await navigate_to_collection(page, args.collection)
                    if not ok:
                        log("FATAL: lost session and cannot recover.", Colors.RED)
                        break
                continue

            zip_size = tmp_zip.stat().st_size / 1e6
            log(f"  ↳ ZIP downloaded: {zip_size:.1f} MB", Colors.GREEN)
            consecutive_failures = 0

            saved = save_dicom_zip(tmp_zip, output_dir)
            if saved is not None:
                total_ok += 1
            else:
                total_fail += args.batch_size

            tmp_zip.unlink(missing_ok=True)

            # Re-navigate to the collection for next batch. A full page
            # reload (not just the SPA in-place tab/collection click) is
            # required: the YUI TreeView caches the 'Not Downloaded' node's
            # children from the first load, so an in-place re-click keeps
            # returning the same already-downloaded rows forever — this was
            # the cause of every batch re-downloading the same image_id.
            await asyncio.sleep(2)
            await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            await asyncio.sleep(5)
            ok = await navigate_to_collection(page, args.collection)
            if not ok:
                log("  ↳ Re-logging in for next batch...", Colors.YELLOW)
                await loni_login(page, username, password)
                await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                await asyncio.sleep(5)
                ok = await navigate_to_collection(page, args.collection)
                if not ok:
                    log("FATAL: cannot re-navigate to collection.", Colors.RED)
                    break

        await browser.close()

    log(f"\n{'═'*60}")
    log(f"  Done.  Successful: {total_ok}  Failed: {total_fail}")
    log(f"{'═'*60}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection",   default=DEFAULT_COLLECTION)
    p.add_argument("--output-dir",   default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    p.add_argument("--env-file",     default=str(DEFAULT_ENV_FILE))
    p.add_argument("--log-file",     default=str(DEFAULT_LOG_FILE))
    p.add_argument("--batch-size",   type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--headless",     default="true",
                   help="Run browser headless (default: true)")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
