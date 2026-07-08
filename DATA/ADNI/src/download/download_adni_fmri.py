#!/usr/bin/env python3
"""
ADNI fMRI Downloader
====================
Downloads ADNI resting-state fMRI images from the LONI IDA portal
(https://ida.loni.usc.edu) using Playwright browser automation.

Usage
-----
    # First-time setup (run once):
    pip install playwright cryptography
    playwright install chromium

    # Create your credential file:
    cp .env.example .env && nano .env

    # Dry-run (no downloads, just shows what would be fetched):
    python download_adni_fmri.py --dry-run

    # Download 1 file as a smoke-test:
    python download_adni_fmri.py --pilot-one

    # Full download (all 1,069 images, resume-safe):
    python download_adni_fmri.py

    # Run in background overnight:
    nohup python download_adni_fmri.py > logs/adni_download.log 2>&1 &

Output layout
-------------
    __fmri_wholebrain_sch200_flat__/
        002_S_4171/
            249536.nii.gz
            287274.nii.gz
        002_S_4262/
            397604.nii.gz
            ...
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Dependency checks ──────────────────────────────────────────────────────────

_MISSING_DEPS: List[str] = []

try:
    import pandas as pd
except ImportError:
    _MISSING_DEPS.append("pandas")

try:
    from dotenv import load_dotenv
except ImportError:
    _MISSING_DEPS.append("python-dotenv")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from playwright.async_api import (
        BrowserContext,
        Download,
        Page,
        async_playwright,
    )
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    _MISSING_DEPS.append("playwright")

try:
    from cryptography.fernet import Fernet, InvalidToken
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    # cryptography is optional — plain .env still works


# ── Constants ──────────────────────────────────────────────────────────────────

LONI_BASE_URL  = "https://ida.loni.usc.edu"
LONI_LOGIN_URL = f"{LONI_BASE_URL}/login.jsp?project=ADNI"
LONI_ADV_SEARCH_URL = (
    f"{LONI_BASE_URL}/pages/access/search.jsp"
    "?project=ADNI&tab=advSearch&page=SEARCH&subPage=NEW_ADV_QUERY"
)
COLLECTION_NAME = "mci"   # Pre-existing collection — always present in the YAHOO tree cache

# Path hierarchy: download_adni_fmri.py lives in src/download/
_SCRIPT_DIR   = Path(__file__).resolve().parent   # .../src/download/
_ADNI_SRC_DIR = _SCRIPT_DIR.parent               # .../src/
_DATA_DIR     = _ADNI_SRC_DIR.parent             # .../ADNI/
_PROJECT_ROOT = _DATA_DIR.parent.parent           # ad-early-detection/

DEFAULT_METADATA_CSV = (
    _DATA_DIR
    / "__metadata__"
    / "Extended_rsfMRI_MCI_Longitudinal_14May2026.csv"
)
DEFAULT_IMAGE_IDS_TXT = (
    _DATA_DIR / "__metadata__" / "image_ids.txt"
)
DEFAULT_OUTPUT_DIR = _DATA_DIR / "__fmri_wholebrain_sch200_flat__"
DEFAULT_ENV_FILE   = _ADNI_SRC_DIR / ".env"
DEFAULT_LOG_DIR    = _PROJECT_ROOT / "logs" / "adni-download"
DEFAULT_DELAY = 2.0  # seconds between downloads
DOWNLOAD_TIMEOUT_MS = 300_000  # 5 minutes per file
NAV_TIMEOUT_MS = 60_000  # 60 s navigation timeout


# ── Colors ─────────────────────────────────────────────────────────────────────


class Colors:
    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class ImageEntry:
    """One row in the metadata CSV with a valid fMRI image_id."""

    subject_id: str
    image_id: int
    viscode: str
    examdate: str
    fmri_description: str


@dataclass
class DownloadResult:
    image_id: int
    subject_id: str
    status: str  # "success" | "skipped" | "failed"
    message: str
    destination: str = ""
    elapsed_s: float = 0.0


@dataclass
class RunStats:
    successful: int = 0
    skipped: int = 0
    failed: int = 0
    failed_ids: List[int] = field(default_factory=list)


# ── Logging ────────────────────────────────────────────────────────────────────


def setup_logging(log_file: Path, no_color: bool) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("adni_downloader")
    logger.setLevel(logging.DEBUG)
    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    # File handler — plain text only (no colors)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # NOTE: We do NOT add a console StreamHandler here.
    # Console output is handled exclusively by tprint() so we can control
    # colors and tqdm compatibility without double-printing.

    return logger


_logger: Optional[logging.Logger] = None


def tprint(message: str, color: str = Colors.RESET) -> None:
    """Print to console (tqdm-safe) and log to file."""
    # Console output — use tqdm.write when inside a progress bar to avoid
    # corrupting the bar display
    if HAS_TQDM:
        tqdm.write(f"{color}{message}{Colors.RESET}")
    else:
        print(f"{color}{message}{Colors.RESET}", flush=True)
    # File log — plain text (no ANSI escape codes)
    if _logger:
        _logger.info(message)


# ── Credential handling ────────────────────────────────────────────────────────


def load_credentials(env_file: Path, args: argparse.Namespace) -> Tuple[str, str]:
    """
    Load ADNI_USERNAME and ADNI_PASSWORD.

    Priority:
    1. CLI flags --username / --password (not recommended but useful for CI)
    2. .env file loaded via python-dotenv
    3. Interactive prompt (fallback, never stored)
    """
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)
        tprint(f"Loaded credentials from {env_file}", Colors.CYAN)
    else:
        tprint(
            f"⚠  No .env file found at {env_file}. "
            f"Copy .env.example to .env and fill in your credentials.",
            Colors.YELLOW,
        )

    username: str = (
        getattr(args, "username", None)
        or os.environ.get("ADNI_USERNAME", "")
        or ""
    )
    password: str = (
        getattr(args, "password", None)
        or os.environ.get("ADNI_PASSWORD", "")
        or ""
    )

    if not username:
        username = input("LONI IDA username: ").strip()
    if not password:
        password = getpass.getpass("LONI IDA password: ")

    if not username or not password:
        tprint("ERROR: Username or password is empty.", Colors.RED)
        sys.exit(1)

    return username, password


def write_env_file(env_file: Path, username: str, password: str) -> None:
    """Write credentials to .env file (plain text, file-permission protected)."""
    env_file.parent.mkdir(parents=True, exist_ok=True)
    with open(env_file, "w", encoding="utf-8") as fh:
        fh.write(f"ADNI_USERNAME={username}\n")
        fh.write(f"ADNI_PASSWORD={password}\n")
    # Restrict to owner read/write only
    env_file.chmod(0o600)
    tprint(f"Credentials saved to {env_file} (permissions: 600)", Colors.GREEN)


# ── Metadata loading ───────────────────────────────────────────────────────────


def load_image_entries_from_txt(txt_path: Path) -> List[ImageEntry]:
    """
    Load image IDs from a comma-separated text file (no subject metadata).
    Each entry gets subject_id='unknown'.
    """
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)
    raw = txt_path.read_text(encoding="utf-8").strip()
    ids = [tok.strip() for tok in raw.replace("\n", ",").split(",") if tok.strip().isdigit()]
    entries = [
        ImageEntry(
            subject_id="unknown",
            image_id=int(iid),
            viscode="",
            examdate="",
            fmri_description="",
        )
        for iid in ids
    ]
    tprint(f"Loaded {len(entries)} image IDs from {txt_path.name}", Colors.CYAN)
    return entries


def load_image_entries(csv_path: Path, ids_txt_path: Optional[Path] = None) -> List[ImageEntry]:
    """
    Parse the metadata CSV and return unique (image_id, subject_id) pairs
    where has_rsfmri_scan == True and image_id is not NaN.

    Falls back to ids_txt_path (comma-separated image IDs) when csv_path is absent.
    """
    if not csv_path.exists():
        fallback = ids_txt_path or DEFAULT_IMAGE_IDS_TXT
        if fallback.exists():
            tprint(
                f"Metadata CSV not found — falling back to {fallback.name}",
                Colors.YELLOW,
            )
            return load_image_entries_from_txt(fallback)
        tprint(f"ERROR: Metadata CSV not found: {csv_path}", Colors.RED)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Filter to rows that actually have an fMRI image
    if "has_rsfmri_scan" in df.columns:
        df = df[df["has_rsfmri_scan"] == True].copy()  # noqa: E712
    df = df[df["image_id"].notna()].copy()
    df["image_id"] = df["image_id"].astype(int)

    # Deduplicate: each image_id only downloaded once, even if referenced
    # by multiple visit rows. Take the first occurrence.
    df = df.drop_duplicates(subset=["image_id"], keep="first")

    entries: List[ImageEntry] = []
    for _, row in df.iterrows():
        entries.append(
            ImageEntry(
                subject_id=str(row.get("subject_id", "unknown")).strip(),
                image_id=int(row["image_id"]),
                viscode=str(row.get("fmri_visit", row.get("viscode", ""))).strip(),
                examdate=str(row.get("fmri_date", row.get("examdate", ""))).strip(),
                fmri_description=str(row.get("fmri_description", "")).strip(),
            )
        )

    tprint(
        f"Loaded {len(entries)} unique image entries from {csv_path.name}",
        Colors.CYAN,
    )
    return entries


def compute_already_downloaded(output_dir: Path) -> Set[int]:
    """
    Scan the output directory for existing {image_id}.nii.gz files.
    Returns the set of image IDs that are already present.
    """
    if not output_dir.exists():
        return set()

    done: Set[int] = set()
    for nii_file in output_dir.rglob("*.nii.gz"):
        stem = nii_file.stem.replace(".nii", "")
        if stem.isdigit():
            done.add(int(stem))
    return done


# ── NIfTI extraction from ZIP ──────────────────────────────────────────────────


def _find_nifti_in_zip(zf: zipfile.ZipFile) -> List[str]:
    """Return all .nii / .nii.gz paths inside the ZIP."""
    return [
        name
        for name in zf.namelist()
        if name.lower().endswith(".nii.gz") or name.lower().endswith(".nii")
    ]


def _find_dicom_in_zip(zf: zipfile.ZipFile) -> List[str]:
    """Return all DICOM-like files inside the ZIP."""
    dicom_exts = {".dcm", ".ima", ".img"}
    results = []
    for name in zf.namelist():
        suffix = Path(name).suffix.lower()
        if suffix in dicom_exts:
            results.append(name)
    # Also match DICOM files without extension (common in ADNI)
    if not results:
        for name in zf.namelist():
            if not Path(name).suffix and not name.endswith("/"):
                results.append(name)
    return results


def extract_nifti_from_zip(
    zip_path: Path,
    dest_dir: Path,
    image_id: int,
) -> Optional[Path]:
    """
    Extract the primary NIfTI file from the downloaded ZIP.

    Strategy:
    1. Look for .nii.gz or .nii files directly — extract the best one.
    2. If only DICOM files found, extract them to a temp dir and attempt
       dcm2niix conversion.
    3. Returns the final .nii.gz path on success, None on failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / f"{image_id}.nii.gz"

    if not zipfile.is_zipfile(zip_path):
        tprint(f"  ✗ Not a valid ZIP: {zip_path.name}", Colors.RED)
        return None

    with zipfile.ZipFile(zip_path, "r") as zf:
        nifti_names = _find_nifti_in_zip(zf)

        # ── Case 1: NIfTI files present ──────────────────────────────────
        if nifti_names:
            # Prefer .nii.gz over .nii; prefer shorter names (less nesting)
            nifti_names.sort(key=lambda n: (not n.lower().endswith(".nii.gz"), len(n)))
            chosen = nifti_names[0]
            extracted = zf.extract(chosen, path=dest_dir.parent / f"_tmp_{image_id}")
            extracted_path = Path(extracted)

            if chosen.lower().endswith(".nii.gz"):
                shutil.move(str(extracted_path), str(final_path))
            else:
                # .nii → gzip it
                import gzip

                with open(extracted_path, "rb") as f_in:
                    with gzip.open(final_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                extracted_path.unlink(missing_ok=True)

            # Cleanup temp dir
            tmp_dir = dest_dir.parent / f"_tmp_{image_id}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if final_path.exists() and final_path.stat().st_size > 0:
                return final_path
            return None

        # ── Case 2: DICOM files — attempt dcm2niix ────────────────────────
        dicom_names = _find_dicom_in_zip(zf)
        if not dicom_names:
            tprint(
                f"  ✗ ZIP contains neither NIfTI nor recognizable DICOM: {zip_path.name}",
                Colors.RED,
            )
            return None

        dcm2niix_path = shutil.which("dcm2niix")
        if not dcm2niix_path:
            tprint(
                "  ✗ DICOM files found but dcm2niix is not installed. "
                "Install it with: sudo apt install dcm2niix (or conda install -c conda-forge dcm2niix)",
                Colors.RED,
            )
            # Still extract DICOMs as fallback so the user isn't left with nothing
            tmp_dcm_dir = dest_dir.parent / f"_dcm_{image_id}"
            tmp_dcm_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(tmp_dcm_dir)
            tprint(
                f"  ↳ DICOM files extracted to {tmp_dcm_dir} for manual conversion.",
                Colors.YELLOW,
            )
            return None

        # Extract DICOM to temp dir and convert
        with tempfile.TemporaryDirectory(prefix=f"adni_dcm_{image_id}_") as tmp_dcm:
            zf.extractall(tmp_dcm)
            tprint(f"  → Converting DICOM to NIfTI with dcm2niix...", Colors.CYAN)
            import subprocess

            result = subprocess.run(
                [
                    dcm2niix_path,
                    "-z", "y",  # gzip output
                    "-f", str(image_id),  # filename pattern
                    "-o", str(dest_dir),
                    tmp_dcm,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                tprint(
                    f"  ✗ dcm2niix failed: {result.stderr.strip()[:200]}",
                    Colors.RED,
                )
                return None

        if final_path.exists() and final_path.stat().st_size > 0:
            return final_path

        # dcm2niix may have produced multiple files — pick the first .nii.gz
        candidates = sorted(dest_dir.glob(f"{image_id}*.nii.gz"))
        if candidates:
            if candidates[0] != final_path:
                shutil.move(str(candidates[0]), str(final_path))
            return final_path

        return None


# ── Playwright browser automation ──────────────────────────────────────────────


async def loni_login(page: Page, username: str, password: str) -> bool:
    """Log in to the LONI IDA portal using the SPA nav-dropdown login form."""
    tprint(f"→ Navigating to {LONI_LOGIN_URL}", Colors.CYAN)
    try:
        await page.goto(LONI_LOGIN_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
    except Exception as exc:
        tprint(f"  ✗ Failed to reach login page: {exc}", Colors.RED)
        return False
    await asyncio.sleep(5)

    # Accept cookie policy (enables the login button in the nav bar)
    await page.evaluate("""() => {
        const el = document.querySelector('.ida-cookie-policy-accept');
        if (el) el.click();
    }""")
    await asyncio.sleep(2)

    # Wait for the login nav button to become enabled (up to 10 s)
    login_ready = False
    for _ in range(10):
        login_ready = await page.evaluate("""() =>
            !!document.querySelector('div.ida-menu-option.login:not(.disabled)')
        """)
        if login_ready:
            break
        await asyncio.sleep(1)

    # Click the nav login toggle (opens the dropdown form)
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

    # Fill credentials via JS (the form inputs are inside the nav dropdown)
    filled = await page.evaluate("""([user, pass]) => {
        function fill(sels, val) {
            for (const s of sels) {
                const el = document.querySelector(s);
                if (el) {
                    el.value = val;
                    el.dispatchEvent(new Event('input',  {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                    return s;
                }
            }
            return null;
        }
        return {
            u: fill(["input[name='userEmail']", "input[name='username']", "input[type='email']"], user),
            p: fill(["input[name='userPassword']", "input[name='password']", "input[type='password']"], pass),
        };
    }""", [username, password])

    if not filled["u"] or not filled["p"]:
        tprint(f"  ✗ Could not fill login form (u={filled['u']}, p={filled['p']})", Colors.RED)
        return False

    # Submit via JS click on the login-btn span
    await page.evaluate("""() => {
        const sels = ['.login-btn', 'span.login-btn', 'button[type="submit"]', 'input[type="submit"]'];
        for (const s of sels) {
            const el = document.querySelector(s);
            if (el) { el.click(); return; }
        }
    }""")
    # Wait for the URL to leave login.jsp (up to 20 s)
    try:
        await page.wait_for_url(
            lambda u: "login" not in u.lower(),
            timeout=20_000,
        )
    except Exception:
        pass
    await asyncio.sleep(2)

    url = page.url
    if "login" in url.lower():
        tprint(f"  ✗ Still on login page: {url}", Colors.RED)
        return False

    tprint(f"  ✓ Logged in: {url}", Colors.GREEN)
    return True


async def _search_for_image(page: Page, image_id: int) -> bool:
    """
    Navigate to Advanced Search, configure fMRI/Original/imageId filters,
    and wait for the AJAX results to appear in the DataTable.
    Returns True if at least 1 result row was found.
    """
    await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
    await asyncio.sleep(5)
    if "login" in page.url.lower():
        return False  # session expired

    # Set up: Image section + fMRI modality + Original type + image ID
    await page.evaluate("""([imgId]) => {
        const imgSec = document.getElementById('imageModalityOption');
        if (imgSec && !imgSec.checked) imgSec.click();
        const fmri = document.querySelector('input[name="imgModality_checkBox"][value="2"]');
        if (fmri && !fmri.checked) fmri.click();
        const orig = document.getElementById('originalOption');
        if (orig && !orig.checked) orig.click();
        const idBox = document.getElementById('imageIdText') || document.querySelector('input[name="imgId"]');
        if (idBox) {
            idBox.value = imgId;
            idBox.dispatchEvent(new Event('input',  {bubbles: true}));
            idBox.dispatchEvent(new Event('change', {bubbles: true}));
        }
    }""", [str(image_id)])

    # Submit search
    await page.evaluate("""() => {
        const b = document.getElementById('advSearchQuery');
        if (b) b.click();
    }""")

    # Wait for AJAX results (do NOT click the results tab manually — SPA auto-switches)
    for i in range(20):
        await asyncio.sleep(1.5)
        try:
            info = await page.evaluate("""() => ({
                subjectCbs: document.querySelectorAll(
                    'input[type="checkbox"][id^="adv_subject_"][id$="_check"]'
                ).length,
                description: (document.getElementById('advTableDescription') || {}).textContent || '',
            })""")
        except Exception:
            # Page navigated mid-evaluation (SPA form POST) — wait and retry
            await asyncio.sleep(2)
            continue
        if info["subjectCbs"] > 0 or "Result" in info["description"]:
            return True
    return False


async def _add_to_collection(page: Page, collection_name: str) -> bool:
    """
    Select all results in the current search result view and add them
    to the named collection (select existing from dropdown by name).
    Returns True if the form was submitted without error.
    """
    # Click subject-level checkboxes to trigger SPA's _addHiddenTag flow
    await page.evaluate("""() => {
        const cbs = document.querySelectorAll(
            'input[type="checkbox"][id^="adv_subject_"][id$="_check"]'
        );
        for (const cb of cbs) { cb.checked = true; cb.click(); }
        const sa = document.getElementById('advResultSelectAll');
        if (sa) { sa.checked = true; sa.click(); }
    }""")
    await asyncio.sleep(1)

    # Force-enable and click "Add To Collection" (triggers the YAHOO dialog)
    await page.evaluate("""() => {
        const btn = document.getElementById('advResultAddCollectId');
        if (!btn) return;
        btn.removeAttribute('disabled');
        btn.className = (btn.className || '').replace('buttonDisabled', 'button');
        btn.click();
    }""")
    await asyncio.sleep(2)

    # Fill dialog: prefer existing collection by text match, else create new
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
            if (existingSel) { existingSel.value = ''; }
            return 'new: ' + collName;
        }
        return 'dialog not found';
    }""", [collection_name])

    tprint(f"  ↳ dialog fill: {fill_result}", Colors.CYAN)

    if "not found" in fill_result.lower():
        # Dialog never opened — fall back to direct form submission
        submit = await page.evaluate("""([collName]) => {
            const form = document.advResultTable || document.forms['advResultTable'];
            if (!form) return 'no form';
            if (form.userAction)     form.userAction.value = 'add';
            if (form.newName)        { form.newName.value = collName; form.newName.disabled = false; }
            if (form.existingName)   form.existingName.value = '';
            if (form.newDescription) form.newDescription.value = '';
            form.submit();
            return 'direct submit';
        }""", [collection_name])
        tprint(f"  ↳ dialog fallback: {submit}", Colors.YELLOW)
    else:
        # Click OK in the YAHOO dialog
        ok_result = await page.evaluate("""() => {
            for (const btn of document.querySelectorAll('button')) {
                if (btn.textContent.trim() === 'OK') { btn.click(); return 'ok-btn'; }
            }
            if (typeof _submitParentForm === 'function') { _submitParentForm(); return 'submitParentForm'; }
            return 'no ok button found';
        }""")
        tprint(f"  ↳ dialog OK: {ok_result}", Colors.CYAN)

    await asyncio.sleep(4)
    return True


async def _download_from_collection_not_downloaded(
    page: Page,
    collection_name: str,
    context: "BrowserContext",
    tmp_zip_path: Path,
    image_id: int = 0,
) -> bool:
    """
    Navigate to Data Collections → named collection → Not Downloaded,
    select all items, click 1-CLICK DOWNLOAD, and save the ZIP.
    Returns True on successful download.
    """
    # ── Navigate to Data Collections (test-script approach) ──────────────────
    # KEY INSIGHT: click the tab FROM THE CURRENT PAGE without any page.goto().
    # The YAHOO TreeView is pre-built with collection data at SPA load time.
    # page.goto() to the dataColl URL causes the tree to re-initialize, but the
    # XHR for children returns CACHED data that doesn't include newly created
    # collections.  Clicking the tab in-place (SPA navigation) works because
    # pre-existing collections like "mci" ARE in the initial tree cache.
    tab_clicked = await page.evaluate("""() => {
        const all = [...document.querySelectorAll('a, li a, .yui-nav a')];
        const tab = all.find(el => el.textContent.trim() === 'Data Collections');
        if (tab) { tab.click(); return 'clicked tab'; }
        return 'tab not found: ' + all.filter(a => a.textContent.trim())
            .map(a => a.textContent.trim()).slice(0, 10).join(' | ');
    }""")
    tprint(f"  ↳ tab nav: {tab_clicked}", Colors.CYAN)
    await asyncio.sleep(3)

    # Expand "My Collections" using YAHOO API (works in SPA context) + label click
    coll_clicked = await page.evaluate("""([name]) => {
        // Expand 'My Collections' tree node via YAHOO API
        try {
            const myCollNode = YAHOO.widget.TreeView.getNode('collections', 1);
            if (myCollNode && !myCollNode.expanded) myCollNode.expand();
        } catch(e) {}
        // Also click the label directly
        const labels = [...document.querySelectorAll(
            '#collections .ygtvlabel, #collections_tree .ygtvlabel'
        )];
        const myCollLabel = labels.find(l => l.textContent.trim() === 'My Collections');
        if (myCollLabel) myCollLabel.click();
        // Find the named collection node
        const allLabels = [...document.querySelectorAll(
            '#collections .ygtvlabel, #collections_tree .ygtvlabel'
        )];
        const found = allLabels.find(l => {
            const t = l.textContent.trim();
            return t === name || t.startsWith(name + ' (') || t.startsWith(name + '(');
        });
        if (found) { found.click(); return 'ok: ' + found.textContent.trim(); }
        return 'not found: ' + allLabels.map(l => l.textContent.trim()).filter(Boolean).join(' | ');
    }""", [collection_name])
    tprint(f"  ↳ coll click: {coll_clicked}", Colors.CYAN)

    if not coll_clicked.startswith("ok:"):
        tprint(f"  ✗ Collection '{collection_name}' not found in tree: {coll_clicked}", Colors.RED)
        return False
    await asyncio.sleep(2)

    # Click the "Not Downloaded" child node and poll for checkboxes (up to 20 s)
    await page.evaluate("""() => {
        const labels = [...document.querySelectorAll(
            '#collections .ygtvlabel, #collections_tree .ygtvlabel'
        )];
        const notDl = labels.find(l => l.textContent.trim().startsWith('Not Downloaded'));
        if (notDl) notDl.click();
    }""")
    not_dl_cbs = 0
    for _ in range(20):
        await asyncio.sleep(1)
        not_dl_cbs = await page.evaluate("""() => {
            const allCbs = [...document.querySelectorAll('input[type="checkbox"]')]
                .filter(cb => !cb.closest('#collections') && !cb.closest('#collections_tree'));
            return allCbs.length;
        }""")
        if not_dl_cbs > 0:
            break
    tprint(f"  ↳ 'Not Downloaded' view: {not_dl_cbs} checkbox(es)", Colors.CYAN)

    if not_dl_cbs > 0:
        # Select up to 10 items (cap prevents server timeout on large batches)
        MAX_BATCH = 10
        selected = await page.evaluate("""([maxN]) => {
            const cbs = [...document.querySelectorAll('input[type="checkbox"]')]
                .filter(cb => !cb.closest('#collections') && !cb.closest('#collections_tree'));
            const toCheck = cbs.slice(0, maxN);
            for (const cb of toCheck) { if (!cb.checked) cb.click(); }
            return 'not-downloaded: selected ' + toCheck.length + ' of ' + cbs.length;
        }""", [MAX_BATCH])
    else:
        # Strategy 2: navigate to full collection view, find row by image_id
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
        selected = await page.evaluate("""([imgId]) => {
            const id  = String(imgId);
            const idI = 'I' + id;
            const rows = [...document.querySelectorAll('tr, [class*="collItem"], [id*="collItem"]')];
            const row = rows.find(r => r.textContent && (
                r.textContent.includes(id) || r.textContent.includes(idI)
            ));
            if (row) {
                const cb = row.querySelector('input[type="checkbox"]');
                if (cb) { cb.checked = true; cb.click(); return 'row-match: ' + id; }
            }
            // Last resort: select all content-area checkboxes
            const allCbs = [...document.querySelectorAll('input[type="checkbox"]')]
                .filter(cb => !cb.closest('#collections') && !cb.closest('#collections_tree'));
            for (const cb of allCbs) { if (!cb.checked) cb.click(); }
            return 'last-resort all: ' + allCbs.length;
        }""", [image_id])

    tprint(f"  ↳ checkbox select: {selected}", Colors.CYAN)
    await asyncio.sleep(1)

    # Intercept the downloadKey AJAX response
    dl_key_info: dict = {}
    dl_key_event = asyncio.Event()

    async def capture(response: "Response") -> None:
        if "/ajax/download/downloadKey" in response.url:
            try:
                body = await response.text()
                dl_key_info.update(status=response.status, body=body)
                dl_key_event.set()
            except Exception:
                dl_key_event.set()

    page.on("response", capture)

    # Click "1-CLICK DOWNLOAD"
    dl_click = await page.evaluate("""() => {
        const btn = document.getElementById('simple-download-button');
        if (btn) { btn.click(); return 'clicked #simple-download-button'; }
        const all = [...document.querySelectorAll('button, input[type="button"], a')];
        for (const b of all) {
            if ((b.textContent || b.value || '').trim().toUpperCase().includes('1-CLICK')) {
                b.click(); return 'text: ' + b.textContent.trim();
            }
        }
        return 'download button not found';
    }""")

    if "not found" in dl_click:
        tprint(f"  ✗ {dl_click}", Colors.RED)
        return False

    # Wait for AJAX then for the download link href to populate
    try:
        await asyncio.wait_for(dl_key_event.wait(), timeout=60)
    except asyncio.TimeoutError:
        pass

    link_href = None
    for _ in range(45):
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
        tprint("  ✗ Download link never populated after 90s", Colors.RED)
        return False

    # Download the file — try Playwright download event first, fall back to requests
    try:
        async with page.expect_download(timeout=DOWNLOAD_TIMEOUT_MS) as dl_info:
            await page.evaluate("""() => {
                const link = document.getElementById('simple-download-link');
                if (link) link.click();
            }""")
        dl = await dl_info.value
        await dl.save_as(str(tmp_zip_path))
        return tmp_zip_path.exists() and tmp_zip_path.stat().st_size > 0

    except Exception as exc:
        tprint(f"  ↳ Playwright download failed ({exc!s:.100}), trying requests...", Colors.YELLOW)

    try:
        import requests  # type: ignore[import-untyped]
        cookies_list = await context.cookies()
        cookies_dict = {c["name"]: c["value"] for c in cookies_list}
        r = requests.get(
            link_href,
            cookies=cookies_dict,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True,
            timeout=60,
        )
        if r.status_code == 200:
            with open(tmp_zip_path, "wb") as fh:
                for chunk in r.iter_content(65536):
                    if chunk:
                        fh.write(chunk)
            return tmp_zip_path.exists() and tmp_zip_path.stat().st_size > 0
    except Exception as exc2:
        tprint(f"  ✗ requests fallback also failed: {exc2}", Colors.RED)

    return False


async def download_single_image(
    page: Page,
    context: "BrowserContext",
    entry: ImageEntry,
    output_dir: Path,
    tmp_dir: Path,
    delay: float,
) -> DownloadResult:
    """
    Download one fMRI image by image_id using the 3-step LONI IDA SPA flow:
    1. Advanced Search by image_id → wait for AJAX result rows
    2. Select results → "Add To Collection" → create per-image collection "I{image_id}"
    3. Navigate to Data Collections → I{image_id} (always 1 item) → 1-CLICK DOWNLOAD
    4. Extract DICOM ZIP → dcm2niix → {subject_id}/{image_id}.nii.gz

    We use the single persistent COLLECTION_NAME ("mci") which is already in the
    YAHOO TreeView, so it is always found.  We download using the "Not Downloaded"
    subtree filter so we only ever grab the 1 item we just added, regardless of
    how many previously-downloaded items are sitting in the collection.
    """
    image_id = entry.image_id
    subject_id = entry.subject_id
    dest_dir = output_dir / subject_id
    final_path = dest_dir / f"{image_id}.nii.gz"
    start = time.monotonic()
    coll_name = COLLECTION_NAME  # use the existing persistent collection (visible in tree)

    # ── Step 1: search ──────────────────────────────────────────────────────
    tprint(f"  → [{image_id}] Searching LONI IDA...", Colors.CYAN)
    found = await _search_for_image(page, image_id)
    if not found:
        if "login" in page.url.lower():
            return DownloadResult(
                image_id=image_id, subject_id=subject_id,
                status="failed", message="Session expired — redirected to login",
                elapsed_s=time.monotonic() - start,
            )
        return DownloadResult(
            image_id=image_id, subject_id=subject_id,
            status="failed", message="No search results found for this image_id",
            elapsed_s=time.monotonic() - start,
        )

    # ── Step 2: add to collection ────────────────────────────────────────────
    tprint(f"  → [{image_id}] Adding to collection '{coll_name}'...", Colors.CYAN)
    await _add_to_collection(page, coll_name)

    # ── Step 3: download from collection ───────────────────────────────────
    tprint(f"  → [{image_id}] Downloading from collection '{coll_name}'...", Colors.CYAN)
    zip_path = tmp_dir / f"{image_id}.zip"
    downloaded = await _download_from_collection_not_downloaded(
        page, coll_name, context, zip_path, image_id=image_id
    )

    if not downloaded:
        return DownloadResult(
            image_id=image_id, subject_id=subject_id,
            status="failed", message="Collection download step failed",
            elapsed_s=time.monotonic() - start,
        )

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    tprint(f"  → [{image_id}] ZIP downloaded ({zip_size_mb:.1f} MB). Extracting...", Colors.CYAN)

    # ── Step 4: extract NIfTI from ZIP ─────────────────────────────────────
    extracted = extract_nifti_from_zip(zip_path, dest_dir, image_id)
    try:
        zip_path.unlink(missing_ok=True)
    except Exception:
        pass

    elapsed = time.monotonic() - start

    if extracted and extracted.exists() and extracted.stat().st_size > 0:
        size_mb = extracted.stat().st_size / (1024 * 1024)
        tprint(
            f"  ✓ [{image_id}] saved → {extracted.relative_to(output_dir.parent)} "
            f"({size_mb:.1f} MB, {elapsed:.1f}s)",
            Colors.GREEN,
        )
        return DownloadResult(
            image_id=image_id, subject_id=subject_id,
            status="success", message="ok",
            destination=str(extracted), elapsed_s=elapsed,
        )

    return DownloadResult(
        image_id=image_id, subject_id=subject_id,
        status="failed", message="NIfTI extraction failed",
        elapsed_s=elapsed,
    )


# ── Orchestration ──────────────────────────────────────────────────────────────


async def run_downloads(args: argparse.Namespace, entries: List[ImageEntry], username: str, password: str) -> RunStats:
    """Main async download loop using a single Playwright browser session."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir.parent / "_adni_download_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-downloaded images
    already_done = compute_already_downloaded(output_dir)
    if already_done:
        tprint(
            f"Resume mode: {len(already_done)} image(s) already downloaded — skipping them.",
            Colors.YELLOW,
        )

    pending = [e for e in entries if e.image_id not in already_done]

    if args.pilot_one:
        pending = pending[:1]
    elif args.max_files is not None:
        pending = pending[: max(0, args.max_files)]

    tprint(
        f"Images to download: {len(pending)} "
        f"(skipping {len(already_done)} already done, {len(entries) - len(already_done) - len(pending)} capped by --max-files)",
        Colors.CYAN,
    )

    stats = RunStats()
    stats.skipped = len(already_done)

    if args.dry_run:
        tprint("\n[DRY RUN] Would download the following images:", Colors.BOLD)
        for entry in pending[:20]:
            tprint(
                f"  image_id={entry.image_id}  subject={entry.subject_id}  "
                f"viscode={entry.viscode}  desc={entry.fmri_description}",
                Colors.CYAN,
            )
        if len(pending) > 20:
            tprint(f"  ... and {len(pending) - 20} more.", Colors.CYAN)
        tprint(
            f"\nTotal: {len(pending)} to download, {stats.skipped} already present.",
            Colors.GREEN,
        )
        return stats

    if not pending:
        tprint("Nothing to download — all images already present.", Colors.GREEN)
        return stats

    headless = args.headless
    delay = args.delay

    async with async_playwright() as pw:
        # Launch Chromium with settings that reduce detection risk
        browser = await pw.chromium.launch(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--dns-prefetch-disable",
                "--disable-features=DnsOverHttps",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            # Tell Playwright where to save files — we intercept them manually
            accept_downloads=True,
        )
        page = await context.new_page()

        # Suppress automation signals
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        # ── Login ──────────────────────────────────────────────────────────
        tprint("\n── Logging in to LONI IDA ──────────────────────────────────", Colors.BOLD)
        logged_in = await loni_login(page, username, password)
        if not logged_in:
            tprint(
                "\nCould not log in. Please check your credentials in the .env file "
                "and try again. Run with --headless false to see the browser.",
                Colors.RED,
            )
            await browser.close()
            return stats

        # ── Download loop ──────────────────────────────────────────────────
        tprint(f"\n── Starting download of {len(pending)} images ────────────────", Colors.BOLD)

        progress = None
        if HAS_TQDM:
            progress = tqdm(
                total=len(pending),
                unit="file",
                desc="Downloading",
                dynamic_ncols=True,
            )

        for idx, entry in enumerate(pending, start=1):
            tprint(
                f"\n[{idx}/{len(pending)}] image_id={entry.image_id}  subject={entry.subject_id}",
                Colors.BLUE,
            )

            result = await download_single_image(
                page=page,
                context=context,
                entry=entry,
                output_dir=output_dir,
                tmp_dir=tmp_dir,
                delay=delay,
            )

            if result.status == "success":
                stats.successful += 1
            elif result.status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
                stats.failed_ids.append(result.image_id)
                tprint(
                    f"  ✗ FAILED [{result.image_id}]: {result.message}",
                    Colors.RED,
                )

            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(
                    f"ok={stats.successful} skip={stats.skipped} fail={stats.failed}"
                )

            # Polite delay between requests
            if idx < len(pending):
                await asyncio.sleep(delay)

        if progress is not None:
            progress.close()

        await browser.close()

    # Cleanup tmp dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return stats


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ADNI fMRI Downloader — downloads resting-state fMRI images from "
            "the LONI IDA portal using Playwright browser automation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run — see what would be downloaded:
  python download_adni_fmri.py --dry-run

  # Download 1 file as smoke test:
  python download_adni_fmri.py --pilot-one

  # Download 1 file with visible browser (for debugging):
  python download_adni_fmri.py --pilot-one --headless false

  # Download up to 10 files:
  python download_adni_fmri.py --max-files 10

  # Full download (resume-safe — skips already-downloaded):
  python download_adni_fmri.py

  # Save credentials to .env first:
  python download_adni_fmri.py --save-credentials
        """,
    )

    # Paths
    parser.add_argument(
        "--metadata-csv",
        default=str(DEFAULT_METADATA_CSV),
        help=f"Path to metadata CSV (default: {DEFAULT_METADATA_CSV.name})",
    )
    parser.add_argument(
        "--image-ids",
        default=str(DEFAULT_IMAGE_IDS_TXT),
        help=(
            f"Comma-separated image IDs text file used when --metadata-csv is absent "
            f"(default: {DEFAULT_IMAGE_IDS_TXT.name})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for downloaded NIfTI files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help=f"Path to .env credentials file (default: {DEFAULT_ENV_FILE})",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Path to log file (default: auto-generated with timestamp)",
    )

    # Credentials
    parser.add_argument(
        "--username",
        default="",
        help="LONI IDA username (overrides .env; not recommended for security)",
    )
    parser.add_argument(
        "--password",
        default="",
        help="LONI IDA password (overrides .env; not recommended for security)",
    )
    parser.add_argument(
        "--save-credentials",
        action="store_true",
        help="Interactively prompt for credentials and save them to .env file, then exit",
    )

    # Download behavior
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without making any network requests",
    )
    parser.add_argument(
        "--pilot-one",
        action="store_true",
        help="Download only the first pending image (smoke test)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to download in this run",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.environ.get("ADNI_DOWNLOAD_DELAY", DEFAULT_DELAY)),
        help=f"Seconds to wait between downloads (default: {DEFAULT_DELAY})",
    )

    # Browser
    headless_default = os.environ.get("ADNI_HEADLESS", "true").lower() != "false"
    parser.add_argument(
        "--headless",
        type=lambda v: v.lower() != "false",
        default=headless_default,
        metavar="true|false",
        help="Run browser headlessly (default: true). Use false to see the browser window.",
    )

    # Output
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored terminal output",
    )

    return parser.parse_args()


# ── Dependency guard ───────────────────────────────────────────────────────────


def check_dependencies() -> None:
    if _MISSING_DEPS:
        print(f"\n{'=' * 60}")
        print("ERROR: Missing required Python packages:")
        for dep in _MISSING_DEPS:
            print(f"  • {dep}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(_MISSING_DEPS)}")
        if "playwright" in _MISSING_DEPS:
            print("  playwright install chromium")
        print(f"{'=' * 60}\n")
        sys.exit(2)

    if not HAS_PLAYWRIGHT:
        print("\nERROR: Playwright is installed but failed to import.")
        print("Try: playwright install chromium")
        sys.exit(2)

    # Check if Chromium browser binary is actually present on disk
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Quick sanity: can we import the async_api without error?
        from playwright.async_api import async_playwright  # noqa: F401  already imported above
    except Exception:
        print("\nWARNING: Could not verify Playwright installation.")
        print("If you hit browser errors, run: playwright install chromium")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = DEFAULT_LOG_DIR / f"download_adni_fmri_{timestamp}.log"

    global _logger
    _logger = setup_logging(log_path, args.no_color)

    tprint(
        f"\n{'═' * 60}\n  ADNI fMRI Downloader\n{'═' * 60}",
        Colors.BOLD,
    )
    tprint(f"Log file:  {log_path}", Colors.CYAN)
    tprint(f"Output:    {args.output_dir}", Colors.CYAN)
    tprint(f"Metadata:  {args.metadata_csv}", Colors.CYAN)
    tprint(f"Image IDs: {args.image_ids}", Colors.CYAN)
    tprint(f"Delay:    {args.delay}s between downloads", Colors.CYAN)
    tprint(f"Headless: {args.headless}", Colors.CYAN)

    # Dependency check
    check_dependencies()

    # ── Save credentials mode ──────────────────────────────────────────────
    if args.save_credentials:
        tprint("\n── Save Credentials ────────────────────────────────────────", Colors.BOLD)
        tprint(
            "Enter your LONI IDA login credentials. "
            "They will be saved to the .env file with restricted permissions (chmod 600).",
            Colors.CYAN,
        )
        uname = input("LONI IDA username: ").strip()
        pw = getpass.getpass("LONI IDA password: ")
        pw2 = getpass.getpass("Confirm password: ")
        if pw != pw2:
            tprint("ERROR: Passwords do not match.", Colors.RED)
            return 1
        write_env_file(Path(args.env_file), uname, pw)
        tprint(
            "\nCredentials saved. You can now run the downloader without entering them each time.",
            Colors.GREEN,
        )
        tprint(
            "REMINDER: The .env file is gitignored and should never be committed to git.",
            Colors.YELLOW,
        )
        return 0

    # ── Load credentials ───────────────────────────────────────────────────
    if not args.dry_run:
        username, password = load_credentials(Path(args.env_file), args)
    else:
        username, password = "", ""  # Not needed for dry run

    # ── Load metadata ──────────────────────────────────────────────────────
    tprint("\n── Loading metadata ────────────────────────────────────────────", Colors.BOLD)
    entries = load_image_entries(Path(args.metadata_csv), Path(args.image_ids))

    if not entries:
        tprint("No entries found in metadata CSV.", Colors.YELLOW)
        return 0

    # ── Run downloads ──────────────────────────────────────────────────────
    start_all = time.monotonic()
    try:
        stats = asyncio.run(run_downloads(args, entries, username, password))
    except KeyboardInterrupt:
        tprint("\n\nInterrupted by user. Progress is saved — re-run to resume.", Colors.YELLOW)
        return 130

    elapsed_total = time.monotonic() - start_all

    # ── Summary ────────────────────────────────────────────────────────────
    tprint(f"\n{'═' * 60}", Colors.BOLD)
    tprint("  Download Summary", Colors.BOLD)
    tprint(f"{'═' * 60}", Colors.BOLD)
    tprint(f"  Successful : {stats.successful}", Colors.GREEN)
    tprint(
        f"  Skipped    : {stats.skipped}  (already downloaded)",
        Colors.YELLOW if stats.skipped else Colors.GREEN,
    )
    tprint(
        f"  Failed     : {stats.failed}",
        Colors.RED if stats.failed else Colors.GREEN,
    )
    tprint(f"  Total time : {elapsed_total / 60:.1f} min", Colors.CYAN)
    tprint(f"  Log file   : {log_path}", Colors.CYAN)

    if stats.failed_ids:
        tprint(f"\nFailed image IDs ({len(stats.failed_ids)}):", Colors.RED)
        for fid in stats.failed_ids:
            tprint(f"  {fid}", Colors.RED)
        tprint(
            "\nTip: Re-run the script to retry failed downloads (they are not marked as done).",
            Colors.YELLOW,
        )

    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
