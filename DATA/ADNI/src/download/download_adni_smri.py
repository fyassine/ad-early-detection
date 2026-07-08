#!/usr/bin/env python3
"""
download_adni_smri.py
======================
Resolves the T1-weighted / MPRAGE structural (sMRI) image ID that matches
each already-downloaded fMRI scan in __dicom_zips_flat__, and writes the
resolved IDs to a plain text file — same comma-separated format as
__metadata__/image_ids.txt — so they can be pasted into a new LONI Data
Collection by hand and downloaded in bulk with
download_adni_smri_collection.py (or the generic download_collection.py).

This script only resolves image IDs — it never downloads anything or
touches LONI collections itself. That split exists because inline
per-image downloads (an earlier version of this script) blocked the
resolve loop on LONI's single-image "1-CLICK DOWNLOAD" widget, which
routinely stalls for minutes per image; resolving is a cheap search and
should run start-to-finish in one pass, with bulk downloading handled
separately as its own step against a proper collection.

Why a separate resolve step
----------------------------
ADNI/LONI has no "paired scan" field: to find the structural scan that goes
with a given resting-state fMRI acquisition, we search LONI's Advanced
Search for the same subject + the exact same exam date (the "Study Date"
filter, restricted to "Equals") and take every MRI result that matches a
T1w/MPRAGE description. If more than one candidate matches on the same date
(match_type="ambiguous"), all of them are recorded — the resolution CSV can
carry multiple smri_image_id rows for one fmri_image_id. No fallback to
nearby dates is used — if no T1w scan exists for that subject on that exact
date, it is recorded as unresolved.

Usage
-----
    # Preview the target list only (no network calls):
    python download_adni_smri.py --dry-run

    # Resolve 1 target as a smoke test (visible browser):
    python download_adni_smri.py --pilot-one --headless false

    # Full run (resume-safe — already-resolved targets are skipped):
    python download_adni_smri.py

Output
------
    DATA/ADNI/__metadata__/smri_resolution.csv   — full audit trail
    DATA/ADNI/__metadata__/smri_image_ids.txt    — comma-separated resolved
                                                    T1 image IDs (every
                                                    resolved/ambiguous
                                                    candidate), ready to
                                                    paste into LONI's
                                                    "Image ID" search field
                                                    to build a collection

Next step
---------
    Paste the contents of smri_image_ids.txt into LONI's Advanced Search
    "Image ID" field, add all results to a new Data Collection, then run:

        python download_adni_smri_collection.py --collection <name>
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    from playwright.async_api import Page, async_playwright
except ImportError:
    print("ERROR: pip install playwright && playwright install chromium")
    sys.exit(1)

from loni_session import is_logged_in, open_context

# ── Paths ────────────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).resolve().parent  # .../src/download/
ADNI_SRC_DIR = SRC_DIR.parent  # .../src/
DATA_DIR = ADNI_SRC_DIR.parent  # .../ADNI/
PROJECT_ROOT = DATA_DIR.parent.parent  # ad-early-detection/

DEFAULT_ZIP_DIR = DATA_DIR / "__dicom_zips_flat__"
DEFAULT_FMRI_METADATA_CSV = (
    DATA_DIR / "__metadata__" / "All_Subjects_Functional_MRI_Images_12May2026.csv"
)
DEFAULT_RESOLUTION_CSV = DATA_DIR / "__metadata__" / "smri_resolution.csv"
DEFAULT_IDS_TXT = DATA_DIR / "__metadata__" / "smri_image_ids.txt"
DEFAULT_ENV_FILE = ADNI_SRC_DIR / ".env"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "adni_download"

LONI_BASE_URL = "https://ida.loni.usc.edu"
LONI_LOGIN_URL = f"{LONI_BASE_URL}/login.jsp?project=ADNI"
LONI_ADV_SEARCH_URL = (
    f"{LONI_BASE_URL}/pages/access/search.jsp"
    "?project=ADNI&tab=advSearch&page=SEARCH&subPage=NEW_ADV_QUERY"
)
NAV_TIMEOUT_MS = 60_000

ZIP_NAME_RE = re.compile(r"^(\d{3}_S_\d{4})_(\d+)\.zip$")

# ADNI's structural T1-weighted sequence descriptions. "Original" image-type
# filtering (set in the search itself) already excludes ADNI's processed
# derivatives (GradWarp/B1/N3/Scaled variants of the same raw acquisition).
T1W_DESCRIPTION_RE = re.compile(r"MPRAGE|MP-RAGE|MP RAGE|SPGR|IR-SPGR|FSPGR|3D\s*T1", re.IGNORECASE)


# ── Logging ────────────────────────────────────────────────────────────────────


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    CYAN = "\033[0;36m"
    BLUE = "\033[0;34m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


_logger: Optional[logging.Logger] = None


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("smri_dl")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log(msg: str, color: str = "") -> None:
    if _logger:
        clean = re.sub(r"\033\[[0-9;]*m", "", msg)
        _logger.info(clean)
    print(f"{color}{msg}{Colors.RESET}" if color else msg)


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class SmriTarget:
    subject_id: str
    fmri_image_id: int
    fmri_date: str  # YYYY-MM-DD


@dataclass
class ResolutionResult:
    subject_id: str
    fmri_image_id: int
    fmri_date: str
    smri_image_id: Optional[int]
    smri_study_id: Optional[str]
    match_type: str  # resolved | ambiguous | unresolved
    description: str


# ── Target list (pure pandas, no network) ──────────────────────────────────────


def build_targets(zip_dir: Path, fmri_metadata_csv: Path) -> list[SmriTarget]:
    """
    Targets = every (subject_id, fmri_image_id) pair already downloaded as a
    raw DICOM ZIP in zip_dir, joined against fmri_metadata_csv to attach the
    exact exam date each fMRI scan was acquired on.
    """
    if not zip_dir.exists():
        raise FileNotFoundError(zip_dir)
    if not fmri_metadata_csv.exists():
        raise FileNotFoundError(fmri_metadata_csv)

    zip_image_ids: set[int] = set()
    for p in zip_dir.glob("*.zip"):
        m = ZIP_NAME_RE.match(p.name)
        if m:
            zip_image_ids.add(int(m.group(2)))

    df = pd.read_csv(fmri_metadata_csv)
    df["image_id"] = pd.to_numeric(df["image_id"], errors="coerce")
    df = df.dropna(subset=["image_id"])
    df["image_id"] = df["image_id"].astype(int)
    df = df[df["image_id"].isin(zip_image_ids)].drop_duplicates(subset=["image_id"])

    missing = zip_image_ids - set(df["image_id"])
    if missing:
        log(
            f"WARNING: {len(missing)} zip image_id(s) not found in "
            f"{fmri_metadata_csv.name}, skipped: {sorted(missing)[:10]}...",
            Colors.YELLOW,
        )

    targets = [
        SmriTarget(
            subject_id=str(row["subject_id"]).strip(),
            fmri_image_id=int(row["image_id"]),
            fmri_date=str(row["fmri_date"]).strip(),
        )
        for _, row in df.iterrows()
    ]
    return sorted(targets, key=lambda t: (t.subject_id, t.fmri_image_id))


# ── Login (same flow as download_adni_fmri.py / download_collection.py) ────────


async def loni_login(page: Page, username: str, password: str) -> bool:
    log(f"  → Navigating to {LONI_LOGIN_URL}", Colors.CYAN)
    await page.goto(LONI_LOGIN_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
    await asyncio.sleep(5)

    if await is_logged_in(page):
        log("  ✓ Reusing existing LONI session (persistent profile)", Colors.GREEN)
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

    await page.evaluate(
        """([user, pwd]) => {
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
    }""",
        [username, password],
    )

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


# ── Date-filtered subject search ────────────────────────────────────────────────


async def search_subject_on_date(page: Page, subject_id: str, exam_date: str) -> list[dict]:
    """
    Advanced Search: Subject ID = subject_id, Study Date equals exam_date
    (YYYY-MM-DD), MRI modality, Original image type. Returns a list of
    {study_id, image_id, description} dicts for every result row.
    """
    await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
    await asyncio.sleep(5)
    if "login" in page.url.lower():
        return []

    # LONI's date-picker input expects MM/DD/YYYY.
    y, m, d = exam_date.split("-")
    date_mmddyyyy = f"{m}/{d}/{y}"

    await page.evaluate(
        """([subj, dateStr]) => {
        // Subject ID
        const subjOpt = document.getElementById('subjectOption');
        if (subjOpt && !subjOpt.checked) subjOpt.click();
        const subjBox = document.getElementById('subjectIdText');
        if (subjBox) {
            subjBox.value = subj;
            subjBox.dispatchEvent(new Event('input',  {bubbles: true}));
            subjBox.dispatchEvent(new Event('change', {bubbles: true}));
        }

        // Study/Visit -> Study Date equals <dateStr>
        const visitOpt = document.getElementById('visitOption');
        if (visitOpt && !visitOpt.checked) visitOpt.click();
        const studyDateSel = document.getElementById('studyDate');
        if (studyDateSel) {
            studyDateSel.value = 'equals';
            studyDateSel.dispatchEvent(new Event('change', {bubbles: true}));
        }
        const dateBox = document.getElementById('advCalText1');
        if (dateBox) {
            dateBox.value = dateStr;
            dateBox.dispatchEvent(new Event('input',  {bubbles: true}));
            dateBox.dispatchEvent(new Event('change', {bubbles: true}));
        }

        // Image modality -> MRI (value=1), Original type
        const imgSec = document.getElementById('imageModalityOption');
        if (imgSec && !imgSec.checked) imgSec.click();
        const mri = document.querySelector('input[name="imgModality_checkBox"][value="1"]');
        if (mri && !mri.checked) mri.click();
        const orig = document.getElementById('originalOption');
        if (orig && !orig.checked) orig.click();
    }""",
        [subject_id, date_mmddyyyy],
    )

    await page.evaluate("""() => {
        const b = document.getElementById('advSearchQuery');
        if (b) b.click();
    }""")

    # Wait for AJAX results (or an explicit "no results" state).
    for _ in range(20):
        await asyncio.sleep(1.5)
        try:
            info = await page.evaluate("""() => ({
                rowCount: document.querySelectorAll(
                    'input[id^="adv_image_I"][id$="_check"]'
                ).length,
                description: (document.getElementById('advTableDescription') || {}).textContent || '',
            })""")
        except Exception:
            await asyncio.sleep(2)
            continue
        if (
            info["rowCount"] > 0
            or "Result" in info["description"]
            or "No result" in info["description"]
        ):
            break

    # Each result image is its own <tr> carrying an adv_image_I<id>_check
    # checkbox; the scan description is the row's last cell. The
    # adv_study_<id>_check checkbox appears only on the FIRST image row of a
    # study group, so the study id is carried forward to the rows beneath it.
    rows = await page.evaluate("""() => {
        const boxes = [...document.querySelectorAll(
            'input[id^="adv_image_I"][id$="_check"]'
        )];
        const out = [];
        let lastStudyId = '';
        for (const cb of boxes) {
            const row = cb.closest('tr');
            if (!row) continue;
            const studyCb = row.querySelector('input[id^="adv_study_"][id$="_check"]');
            if (studyCb) {
                lastStudyId = studyCb.id.replace('adv_study_', '').replace('_check', '');
            }
            const imageId = (cb.value || cb.id)
                .replace('adv_image_', '')
                .replace('_check', '')
                .replace(/^I/, '');
            const cells = [...row.children].map(td => td.textContent.trim());
            const description = cells.length ? cells[cells.length - 1] : '';
            out.push({ studyId: lastStudyId, imageId, description });
        }
        return out;
    }""")
    return rows


# ── Resolution CSV (resume support + audit trail) ───────────────────────────────


RESOLUTION_COLUMNS = [
    "subject_id",
    "fmri_image_id",
    "fmri_date",
    "smri_image_id",
    "smri_study_id",
    "match_type",
    "description",
]


def load_existing_resolutions(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=RESOLUTION_COLUMNS)


def append_resolution(csv_path: Path, result: ResolutionResult) -> None:
    row = pd.DataFrame([result.__dict__])
    write_header = not csv_path.exists()
    row.to_csv(csv_path, mode="a", header=write_header, index=False)


def write_ids_txt(csv_path: Path, ids_txt_path: Path) -> int:
    """
    Regenerate the plain comma-separated ID list (same format as
    __metadata__/image_ids.txt) from every resolved/ambiguous row in the
    resolution CSV. Returns the number of IDs written.
    """
    df = load_existing_resolutions(csv_path)
    resolved = df[df["match_type"].isin(["resolved", "ambiguous"])]
    ids = sorted(int(i) for i in resolved["smri_image_id"].dropna().unique())
    ids_txt_path.parent.mkdir(parents=True, exist_ok=True)
    ids_txt_path.write_text(",".join(str(i) for i in ids), encoding="utf-8")
    return len(ids)


# ── Resolve loop ─────────────────────────────────────────────────────────────


async def run_resolve(args: argparse.Namespace, targets: list[SmriTarget]) -> None:
    resolution_csv = Path(args.resolution_csv)
    existing = load_existing_resolutions(resolution_csv)
    already_done = (
        set(zip(existing["subject_id"], existing["fmri_image_id"], strict=True))
        if len(existing)
        else set()
    )

    pending = [t for t in targets if (t.subject_id, t.fmri_image_id) not in already_done]
    if args.pilot_one:
        pending = pending[:1]
    elif args.max_targets is not None:
        pending = pending[: max(0, args.max_targets)]

    log(
        f"Targets to resolve: {len(pending)} "
        f"(skipping {len(targets) - len(pending)} already resolved/capped)",
        Colors.CYAN,
    )
    if not pending:
        return

    env_file = Path(args.env_file)
    if env_file.exists():
        load_dotenv(env_file)
    username = args.username or os.environ.get("ADNI_USERNAME", "")
    password = args.password or os.environ.get("ADNI_PASSWORD", "")
    if not username:
        username = input("LONI IDA username: ").strip()
    if not password:
        password = getpass.getpass("LONI IDA password: ")
    if not username or not password:
        log("ERROR: username/password required for resolve phase", Colors.RED)
        sys.exit(1)

    headless = str(args.headless).lower() not in ("false", "0", "no")

    async with async_playwright() as pw:
        context = await open_context(pw, headless)
        browser = context
        page = await context.new_page()

        log("\n── Logging in to LONI IDA ──────────────────────────────────", Colors.BOLD)
        if not await loni_login(page, username, password):
            log(
                "Could not log in. If this is the first run, LONI now requires a "
                "reCAPTCHA that a scripted browser cannot solve — rerun once with "
                "--headless false and solve it by hand; the session persists in "
                "DATA/ADNI/.loni_profile for later headless runs.",
                Colors.RED,
            )
            await browser.close()
            sys.exit(1)

        try:
            for idx, target in enumerate(pending, start=1):
                log(
                    f"\n[{idx}/{len(pending)}] subject={target.subject_id} "
                    f"fmri_image_id={target.fmri_image_id} date={target.fmri_date}",
                    Colors.BLUE,
                )

                rows = await search_subject_on_date(page, target.subject_id, target.fmri_date)
                candidates = [r for r in rows if T1W_DESCRIPTION_RE.search(r["description"])]

                if not candidates:
                    log(f"  ✗ No T1w scan found on {target.fmri_date}", Colors.YELLOW)
                    result = ResolutionResult(
                        subject_id=target.subject_id,
                        fmri_image_id=target.fmri_image_id,
                        fmri_date=target.fmri_date,
                        smri_image_id=None,
                        smri_study_id=None,
                        match_type="unresolved",
                        description="",
                    )
                    append_resolution(resolution_csv, result)
                else:
                    match_type = "resolved" if len(candidates) == 1 else "ambiguous"
                    for chosen in candidates:
                        smri_image_id = int(chosen["imageId"])
                        log(
                            f"  ✓ Matched I{smri_image_id} ({chosen['description']}) "
                            f"[{match_type}, {len(candidates)} candidate(s)]",
                            Colors.GREEN,
                        )
                        result = ResolutionResult(
                            subject_id=target.subject_id,
                            fmri_image_id=target.fmri_image_id,
                            fmri_date=target.fmri_date,
                            smri_image_id=smri_image_id,
                            smri_study_id=chosen["studyId"],
                            match_type=match_type,
                            description=chosen["description"],
                        )

                        append_resolution(resolution_csv, result)

                write_ids_txt(resolution_csv, Path(args.ids_txt))

                if idx < len(pending):
                    await asyncio.sleep(args.delay)
        finally:
            await browser.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--zip-dir", default=str(DEFAULT_ZIP_DIR))
    p.add_argument("--fmri-metadata-csv", default=str(DEFAULT_FMRI_METADATA_CSV))
    p.add_argument("--resolution-csv", default=str(DEFAULT_RESOLUTION_CSV))
    p.add_argument("--ids-txt", default=str(DEFAULT_IDS_TXT))
    p.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    p.add_argument("--username", default="")
    p.add_argument("--password", default="")
    p.add_argument("--delay", type=float, default=2.0, help="Seconds between resolve searches")
    p.add_argument("--headless", default="true")
    p.add_argument(
        "--dry-run", action="store_true", help="Print target list only, no network calls"
    )
    p.add_argument("--pilot-one", action="store_true", help="Resolve only 1 target")
    p.add_argument("--max-targets", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    log_dir = Path(args.log_dir)
    global _logger
    _logger = setup_logging(log_dir / f"download_adni_smri_{_timestamp()}.log")

    log(f"\n{'═' * 60}\n  ADNI sMRI Resolver\n{'═' * 60}", Colors.BOLD)
    log(f"Zip dir:         {args.zip_dir}", Colors.CYAN)
    log(f"fMRI metadata:   {args.fmri_metadata_csv}", Colors.CYAN)
    log(f"Resolution CSV:  {args.resolution_csv}", Colors.CYAN)
    log(f"IDs txt:         {args.ids_txt}", Colors.CYAN)

    targets = build_targets(Path(args.zip_dir), Path(args.fmri_metadata_csv))
    log(f"Total targets: {len(targets)}", Colors.CYAN)

    if args.dry_run:
        log("\n[DRY RUN] Sample targets:", Colors.BOLD)
        for t in targets[:20]:
            log(
                f"  subject={t.subject_id}  fmri_image_id={t.fmri_image_id}  date={t.fmri_date}",
                Colors.CYAN,
            )
        if len(targets) > 20:
            log(f"  ... and {len(targets) - 20} more.", Colors.CYAN)
        return 0

    exit_code = 0
    try:
        asyncio.run(run_resolve(args, targets))
    except Exception:
        _logger.exception("run_resolve crashed")
        log(
            "✗ Resolve loop crashed — see traceback in the log file above. "
            "Already-resolved targets were written; rerun to resume.",
            Colors.RED,
        )
        exit_code = 1

    n_ids = write_ids_txt(Path(args.resolution_csv), Path(args.ids_txt))
    log(f"\n✓ Wrote {n_ids} resolved sMRI image ID(s) to {args.ids_txt}", Colors.GREEN)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
