#!/usr/bin/env python3
"""
download_collection_jar.py
==========================
Download every image of a LONI IDA collection, ONE image at a time, using the
official IDA Downloader jar for the actual transfer.

Why this exists
---------------
Two facts about IDA make the "obvious" approaches fail on a headless box:

  1. IP-lock — every download URL is bound to the public IP of the *browser
     session that minted it*. A link generated in a browser on your laptop
     returns a 287-byte "wrong IP" stub when fetched from this machine.
  2. Throttling — fetching the prepared zip over plain `requests`/Playwright
     gets slow/timed-out; IDA expects its chunked downloader client.

This script sidesteps both: it drives a headless Playwright session *on this
machine* (so every URL is IP-matched to this box), selects a single image,
reads the `#simple-download-link` href — which is exactly
`https://ida.loni.usc.edu/download/image?key=...&count=1&zip=...zip`, the same
endpoint the Advanced Download "Zip File" links use — and hands that URL to the
IDA Downloader jar for a chunked, resumable transfer.

The jar's `Main-Class` (launch.Launcher) refuses non-Oracle JVMs, but the real
worker class `edu.usc.loni.ida.download.resource.ResourceDownloader` has no such
check, so we invoke it directly via `-cp` and it runs on Ubuntu's OpenJDK 21.

Per-image (not whole-collection) downloading is deliberate: each zip is small,
the run is resumable (already-saved {subject}_{image}.zip files are skipped),
and a dropped connection costs one image, not the whole 800-image batch.

Usage
-----
    # Validate the whole pipeline on a single image first:
    python download_collection_jar.py --collection ADNI_Converters_fMRI --limit 1

    # Then the full run (inside tmux/screen):
    python download_collection_jar.py --collection ADNI_Converters_fMRI
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

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

# Reuse the proven login / navigation / save helpers.
import re

import download_collection as dc
from download_collection import (
    LONI_ADV_SEARCH_URL,
    NAV_TIMEOUT_MS,
    Colors,
    local_image_ids,
    loni_login,
    navigate_to_collection,
    save_dicom_zip,
)
from loni_session import open_context

# ── Paths / jar invocation ───────────────────────────────────────────────────

SRC_DIR  = Path(__file__).resolve().parent          # .../src/download/
ADNI_SRC_DIR = SRC_DIR.parent                      # .../src/
DATA_DIR = ADNI_SRC_DIR.parent                      # .../ADNI/
PROJECT_ROOT = DATA_DIR.parent.parent               # ad-early-detection/

DEFAULT_JAR        = ADNI_SRC_DIR / "ida_downloader" / "IdaDownloader_15May2026.jar"
# Bypass launch.Launcher's Oracle-Java vendor gate by calling the worker class
# directly; it performs no vendor check. See run_ida_downloader.py.
WORKER_MAIN        = "edu.usc.loni.ida.download.resource.ResourceDownloader"
DEFAULT_JAVA_HOME  = "/usr/lib/jvm/java-21-openjdk-amd64"
MIN_JAVA_VERSION   = 12

DEFAULT_COLLECTION = "ADNI_Converters_fMRI"
DEFAULT_OUTPUT_DIR = DATA_DIR / "__dicom_zips_flat__"
NIFTI_OUTPUT_DIR   = DATA_DIR / "__fmri_wholebrain_sch200_flat__"
DEFAULT_ENV_FILE   = ADNI_SRC_DIR / ".env"
# IDA's 1-CLICK link now resolves to a *dynamically generated* streaming zip
# (/download/files/ida1/<uuid>/<name>.zip — empty ETag, Last-Modified 1970). The
# server builds it sequentially, so parallel range requests for high offsets
# hang and the transfer stalls after a few MB. A single sequential stream pulls
# the whole file at line rate, so chunks MUST be 1 for this endpoint. (Older
# pre-staged static zips supported ranges; today's do not.)
DEFAULT_CHUNKS     = 1


def _default_log_file(script_name: str = "adni_collection_jar") -> Path:
    """Return logs/adni-download/<YYYYMMDD_HHMMSS>/<script_name>.log."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / "adni-download" / ts / f"{script_name}.log"


def log(msg: str, color: str = "") -> None:
    print(f"{color}{msg}{Colors.RESET}" if color else msg)


# ── Java / jar plumbing ──────────────────────────────────────────────────────


def build_java_env(java_home: str) -> dict[str, str]:
    """Return an environment dict with JAVA_HOME/PATH pointed at java_home.

    Raises loudly if the resolved java is missing or older than the jar's
    minimum, rather than silently falling back to the system java (which is 11
    on this box and is rejected by the jar's worker)."""
    env = dict(os.environ)
    jh = Path(java_home)
    java_bin = jh / "bin" / "java"
    if not java_bin.exists():
        raise FileNotFoundError(
            f"java not found at {java_bin}. Pass --java-home to a JDK {MIN_JAVA_VERSION}+ install."
        )
    env["JAVA_HOME"] = str(jh)
    env["PATH"] = f"{jh / 'bin'}{os.pathsep}{env.get('PATH', '')}"

    out = subprocess.run(
        [str(java_bin), "-version"], capture_output=True, text=True, timeout=10
    )
    banner = (out.stderr or out.stdout).strip()
    import re

    m = re.search(r'version "(\d+)', banner)
    if not m:
        raise RuntimeError(f"Could not parse java version from: {banner!r}")
    major = int(m.group(1))
    if major < MIN_JAVA_VERSION:
        raise RuntimeError(
            f"java {major} at {java_bin} is too old; the IDA worker needs "
            f"Java {MIN_JAVA_VERSION}+. Pass --java-home."
        )
    log(f"  ↳ java OK: {banner.splitlines()[0]}", Colors.GREEN)
    return env


def download_with_jar(
    url: str, jar: Path, out_dir: Path, chunks: int, java_env: dict[str, str],
    stall_s: int = 90, max_s: int = 1200,
) -> Path | None:
    """Run the IDA worker on a single URL into out_dir; return the resulting zip.

    The jar names the file from the URL's &zip= parameter, so we snapshot the
    directory before/after and return the newly-created .zip.

    The jar has no built-in timeout: if the server accepts the connection but
    stops feeding bytes (a stalled prepared zip), it hangs forever and wedges
    the whole run. So we poll the on-disk size and kill the process if it makes
    no progress for `stall_s` seconds (or exceeds an absolute `max_s` cap). A
    killed download returns None and is retried with a freshly-minted URL by the
    caller."""
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in out_dir.glob("*.zip")}

    cmd = [
        "java", "-cp", str(jar), WORKER_MAIN,
        f"--directory={out_dir}",
        f"--chunks={chunks}",
        url,
    ]
    log(f"  ↳ jar: java -cp {jar.name} {WORKER_MAIN} --chunks={chunks} <url>", Colors.CYAN)

    def _cur_bytes() -> int:
        return sum(p.stat().st_size for p in out_dir.glob("*.zip"))

    proc = subprocess.Popen(
        cmd, env=java_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    start = time.monotonic()
    last_bytes = _cur_bytes()
    last_progress = start
    killed_reason = None
    while proc.poll() is None:
        time.sleep(5)
        now = time.monotonic()
        cur = _cur_bytes()
        if cur > last_bytes:
            last_bytes = cur
            last_progress = now
        if now - last_progress > stall_s:
            killed_reason = f"stalled (no progress for {stall_s}s at {cur/1e6:.1f} MB)"
        elif now - start > max_s:
            killed_reason = f"exceeded {max_s}s cap at {cur/1e6:.1f} MB"
        if killed_reason:
            proc.kill()
            break

    try:
        stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    class _P:  # lightweight stand-in so the rest of the function is unchanged
        pass

    proc_result = _P()
    proc_result.returncode = proc.returncode
    proc_result.stdout = stdout
    proc_result.stderr = stderr
    proc = proc_result

    if killed_reason:
        log(f"  ✗ jar killed: {killed_reason}", Colors.RED)
        for p in out_dir.glob("*.zip"):
            if p not in before:
                p.unlink(missing_ok=True)
        return None
    if proc.returncode != 0:
        log(f"  ✗ jar exited {proc.returncode}: {(proc.stderr or '').strip()[:300]}", Colors.RED)
        return None

    # The newest non-empty .zip that wasn't there (or was rewritten) before the
    # run. A 0-byte file is the jar's signature for a download that exited 0 but
    # transferred nothing (e.g. the prepared zip wasn't ready server-side), so
    # treat it as a failure rather than a success.
    candidates = [
        p for p in out_dir.glob("*.zip")
        if (p not in before or p.stat().st_mtime > before[p]) and p.stat().st_size > 0
    ]
    if not candidates:
        log(f"  ✗ jar produced no usable .zip in {out_dir}", Colors.RED)
        # Surface what the jar actually said — otherwise a clean exit with no
        # file is silent. Drop any 0-byte stub so it doesn't poison the next
        # before/after snapshot.
        tail_out = (proc.stdout or "").strip().splitlines()[-6:]
        tail_err = (proc.stderr or "").strip().splitlines()[-6:]
        if tail_out:
            log("    jar stdout: " + " | ".join(tail_out)[:400], Colors.YELLOW)
        if tail_err:
            log("    jar stderr: " + " | ".join(tail_err)[:400], Colors.YELLOW)
        for p in out_dir.glob("*.zip"):
            if p.stat().st_size == 0:
                p.unlink(missing_ok=True)
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── Per-image URL capture (in-session, IP-matched) ───────────────────────────


def value_to_image_id(value: str) -> int | None:
    """Numeric image id from a checkbox value like 'I1310788' (or 'D…')."""
    m = re.search(r"(\d+)", value or "")
    return int(m.group(1)) if m else None


# The collection detail table is server-paged at ~18 rows per page: only those
# 18 cell11_N elements exist in the DOM at a time. The page is advanced by a
# down-arrow control:
#   <input type="image" src=".../Scroll-Arrow-Down.png"
#          onclick="...new SelectHandler().nextPage(); return false;">
# which swaps the table content in place (no full reload). So we process each
# page's images, click that arrow, wait for the rows to actually change, and
# repeat until the arrow no longer changes them (last page).
_NEXT_PAGE_ARROW = 'input[type="image"][onclick*="nextPage"]'

_READ_VISIBLE_JS = """
    () => {
        const out = [];
        let idx = 0, cell;
        while ((cell = document.getElementById('cell11_' + idx))) {
            for (const cb of cell.querySelectorAll('input[type="checkbox"][name="checkbox"]'))
                out.push(cb.value);
            idx++;
        }
        return out;
    }
"""


async def read_visible_values(page: Page) -> list[str]:
    """Checkbox values for the rows currently rendered (the current page)."""
    return await page.evaluate(_READ_VISIBLE_JS)


def image_values(values: list[str]) -> list[str]:
    """Keep only real image rows (value 'I<id>'); drop 'D…'/other non-image
    rows. Selecting a non-image row leaves 1-CLICK disabled and hangs."""
    return [v for v in values if v.startswith("I")]


async def wait_for_collection_rows(page: Page, timeout_s: int = 45) -> int:
    """Poll until the current page's row checkboxes have rendered.

    Clicking the collection node (or the next-page arrow) fires an async
    request that populates the rows; reading the DOM before it lands yields 0."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        values = await read_visible_values(page)
        if values:
            return len(values)
        await asyncio.sleep(1)
    return 0


async def go_to_next_page(
    page: Page, current_values: list[str], timeout_s: int = 25
) -> tuple[bool, list[str]]:
    """Advance to the next page, then wait for the rendered rows to actually
    change. Returns (advanced, new_values); advanced is False on the last page
    (the arrow leaves the rows unchanged).

    The arrow is clicked via JS (firing its onclick → SelectHandler.nextPage()
    directly) rather than a Playwright actionability click: after a page of
    downloads the 1-CLICK modal's overlay mask can still cover the arrow, which
    made an actionability click time out and look like the last page."""
    clicked = await page.evaluate(
        """(sel) => {
            const el = document.querySelector(sel);
            if (!el) return false;
            el.click();
            return true;
        }""",
        _NEXT_PAGE_ARROW,
    )
    if not clicked:
        log("  ✗ next-page arrow not found", Colors.RED)
        return False, current_values

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await asyncio.sleep(1)
        new_values = await read_visible_values(page)
        if new_values and new_values != current_values:
            return True, new_values
    return False, current_values


async def select_only(page: Page, value: str) -> bool:
    """Check exactly the row whose checkbox value == value (it is on the current
    page), unchecking any other checked row first so 1-CLICK prepares a
    single-image zip. Returns True iff the target ended up checked."""
    return await page.evaluate(
        """([val]) => {
            let target = null;
            for (const cb of document.querySelectorAll('input[type="checkbox"][name="checkbox"]')) {
                if (cb.value === val) target = cb;
                else if (cb.checked) cb.click();   // fires CheckBoxHandler.handleCheck
            }
            if (!target) return false;
            if (!target.checked) target.click();
            return target.checked;
        }""",
        [value],
    )


async def close_download_modal(page: Page) -> None:
    """Dismiss the 1-CLICK download modal and clear all checkboxes so the next
    image on this page starts from a clean selection (we stay on the same page
    rather than reloading, to preserve pagination position)."""
    await page.evaluate(
        """() => {
            const sels = ['.simpleDownloadModal .close', '.simpleDownloadModal a.close',
                          '#simpleDownloadModal .close', '.yui-panel .container-close',
                          '.container-close', '[class*="download"] .close'];
            for (const s of sels) { const e = document.querySelector(s); if (e) e.click(); }
            // Clear any leftover modal overlay mask that would intercept the
            // next-page arrow / checkbox clicks.
            for (const m of document.querySelectorAll('.mask, [class*="mask"]')) {
                if (getComputedStyle(m).display !== 'none') m.style.display = 'none';
            }
        }"""
    )
    try:
        await page.keyboard.press("Escape")
    except Exception:  # noqa: BLE001
        pass
    await asyncio.sleep(0.5)
    await page.evaluate(
        """() => {
            for (const cb of document.querySelectorAll('input[type="checkbox"][name="checkbox"]'))
                if (cb.checked) cb.click();
        }"""
    )
    await asyncio.sleep(0.3)


async def get_download_href(page: Page) -> str | None:
    """Click 1-CLICK DOWNLOAD and return the populated #simple-download-link href
    (the /download/image?key=... URL). Assumes a row is already selected."""
    dl_key_event = asyncio.Event()

    def on_response(resp):
        if "downloadKey" in resp.url:
            dl_key_event.set()

    page.on("response", on_response)
    try:
        dl_click = await page.evaluate(
            """() => {
                const btn = document.getElementById('simple-download-button');
                if (btn) { btn.click(); return 'ok'; }
                const all = [...document.querySelectorAll('button, input[type="button"], a')];
                for (const b of all) {
                    if ((b.textContent || b.value || '').trim().toUpperCase().includes('1-CLICK')) {
                        b.click(); return 'ok';
                    }
                }
                return 'not found';
            }"""
        )
        if dl_click != "ok":
            log("  ✗ 1-CLICK button not found", Colors.RED)
            return None

        try:
            await asyncio.wait_for(dl_key_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass

        for _ in range(60):
            await asyncio.sleep(2)
            href = await page.evaluate(
                """() => {
                    const link = document.getElementById('simple-download-link');
                    if (!link) return null;
                    const h = link.href || '';
                    return (h && !h.endsWith('#')) ? h : null;
                }"""
            )
            if href:
                return href
    finally:
        page.remove_listener("response", on_response)

    log("  ✗ download link never populated after 120s", Colors.RED)
    return None


# ── Main loop ────────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> None:
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
        log("ERROR: set ADNI_USERNAME / ADNI_PASSWORD in .env", Colors.RED)
        sys.exit(1)

    jar = Path(args.jar)
    if not jar.exists():
        raise FileNotFoundError(f"IDA Downloader jar not found: {jar}")
    java_env = build_java_env(args.java_home)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = DATA_DIR / ".tmp_coll_jar"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # The jar names every download after the URL's zip param (the collection
    # name), so all images share one tmp filename. A partial left by a killed
    # run pre-exists in download_with_jar's before-snapshot, so its kill-cleanup
    # never removes it AND the stall monitor counts its stale bytes as the
    # baseline — wedging every subsequent image at that size. Clear leftovers up
    # front; tmp_dir holds only in-flight downloads (completed ones are moved
    # out immediately).
    for stale in tmp_dir.glob("*.zip"):
        log(f"  ↳ clearing stale tmp partial: {stale.name} ({stale.stat().st_size/1e6:.1f} MB)", Colors.YELLOW)
        stale.unlink(missing_ok=True)

    if "TMUX" not in os.environ and "STY" not in os.environ:
        log("Warning: not in tmux/screen — a dropped SSH session will kill a long run.", Colors.YELLOW)

    log(f"\n{'═'*60}")
    log(f"  ADNI Collection Downloader (via IDA jar)")
    log(f"  Collection : {args.collection}")
    log(f"  Output dir : {output_dir}")
    log(f"  Limit      : {args.limit if args.limit else 'all'}")
    log(f"{'═'*60}\n")

    headless = str(args.headless).lower() not in ("false", "0", "no")

    async with async_playwright() as pw:
        context: BrowserContext = await open_context(pw, headless)
        browser = context
        page = await context.new_page()

        log("── Logging in ───────────────────────────────────────────────")
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

        await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        if not await navigate_to_collection(page, args.collection):
            log("FATAL: collection not found in tree.", Colors.RED)
            await browser.close()
            sys.exit(1)

        # The table is server-paged at ~18 rows/page. Process every not-yet-local
        # image on the current page, then click the down-arrow to load the next
        # page, until the arrow stops changing the rows (last page). Skip is
        # purely local: an image whose {subject}_{image}.zip already exists is
        # never re-downloaded — IDA's Downloaded/Not-Downloaded views are ignored.
        if await wait_for_collection_rows(page) == 0:
            log("FATAL: collection table never rendered any rows.", Colors.RED)
            await browser.close()
            sys.exit(1)

        done = 0
        fails = 0
        consecutive = 0
        processed: set[str] = set()
        page_idx = 0
        abort = False
        while not abort:
            page_values = await read_visible_values(page)
            imgs = image_values(page_values)
            skip_ids = local_image_ids(output_dir, NIFTI_OUTPUT_DIR)
            already = sum(1 for v in imgs if value_to_image_id(v) in skip_ids)
            log(f"\n── page {page_idx}: {len(imgs)} images, {already} already local ──", Colors.CYAN)

            for v in imgs:
                if args.limit and done >= args.limit:
                    abort = True
                    log(f"\n✓ Reached --limit {args.limit}.", Colors.GREEN)
                    break
                if v in processed:
                    continue
                processed.add(v)
                if value_to_image_id(v) in skip_ids:
                    continue

                log(f"[{done + 1}] image {v} (id={value_to_image_id(v)})  page {page_idx}")
                if not await select_only(page, v):
                    log("  ✗ selection did not register", Colors.RED)
                    fails += 1
                    consecutive += 1
                    if consecutive >= 3:
                        log("FATAL: 3 consecutive selection failures.", Colors.RED)
                        abort = True
                        break
                    continue

                # Most jar "no usable .zip" failures are transient (the prepared
                # zip isn't ready server-side yet). Re-mint the URL and retry a
                # couple of times before giving up; a genuinely bad image still
                # fails fast enough and gets retried on the next full pass.
                zip_path = None
                for attempt in range(1, 4):
                    url = await get_download_href(page)
                    zip_path = (
                        download_with_jar(url, jar, tmp_dir, args.chunks, java_env)
                        if url else None
                    )
                    if zip_path is not None:
                        break
                    if attempt < 3:
                        log(f"  ↻ retry {attempt}/2 for {v} after a short wait", Colors.YELLOW)
                        await close_download_modal(page)
                        await asyncio.sleep(5)
                        if not await select_only(page, v):
                            break
                await close_download_modal(page)
                if zip_path is None:
                    fails += 1
                    consecutive += 1
                    if consecutive >= 3:
                        log("FATAL: 3 consecutive download failures.", Colors.RED)
                        abort = True
                        break
                    continue

                size_mb = zip_path.stat().st_size / 1e6
                log(f"  ↳ downloaded {size_mb:.1f} MB → {zip_path.name}", Colors.GREEN)
                saved = save_dicom_zip(zip_path, output_dir)
                if not saved:
                    fails += 1
                else:
                    done += 1
                    consecutive = 0
                zip_path.unlink(missing_ok=True)

            if abort:
                break

            advanced, _ = await go_to_next_page(page, page_values)
            if not advanced:
                log("\n✓ Reached last page — collection complete.", Colors.GREEN)
                break
            page_idx += 1

        await browser.close()

    log(f"\n{'═'*60}")
    log(f"  Done.  Saved: {done}  Failed: {fails}")
    log(f"{'═'*60}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    p.add_argument("--jar", default=str(DEFAULT_JAR))
    p.add_argument("--java-home", default=DEFAULT_JAVA_HOME)
    p.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    p.add_argument("--limit", type=int, default=0, help="Stop after N images (0 = all). Use --limit 1 to smoke-test.")
    p.add_argument("--headless", default="true")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
