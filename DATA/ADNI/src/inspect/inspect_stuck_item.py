#!/usr/bin/env python3
"""
One-off inspection: log in, navigate to mci-all-v2 -> Not Downloaded,
dump info about the first few rows (image_id, subject_id, full row HTML),
then select row 0, click 1-CLICK DOWNLOAD, and poll for up to 5 minutes
for the download link -- logging elapsed time and scanning the page for
any visible error/dialog/agreement messages.
"""

import asyncio
import os
import sys
from pathlib import Path

INSPECT_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = INSPECT_DIR.parent / "download"
sys.path.insert(0, str(DOWNLOAD_DIR))

from dotenv import load_dotenv
from playwright.async_api import async_playwright

import download_collection as dc


async def main() -> None:
    dc._logger = dc.setup_logging(INSPECT_DIR.parent.parent.parent.parent / "logs" / "adni-download" / "inspect_stuck_item.log")

    load_dotenv(INSPECT_DIR.parent / ".env")
    username = os.environ.get("LONI_USERNAME") or os.environ.get("IDA_USERNAME") or os.environ.get("ADNI_USERNAME", "")
    password = os.environ.get("LONI_PASSWORD") or os.environ.get("IDA_PASSWORD") or os.environ.get("ADNI_PASSWORD", "")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        ok = await dc.loni_login(page, username, password)
        if not ok:
            sys.exit(1)

        await page.goto(dc.LONI_ADV_SEARCH_URL, timeout=dc.NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)

        ok = await dc.navigate_to_collection(page, "mci-all-v2")
        if not ok:
            sys.exit(1)

        count, label = await dc.count_not_downloaded(page)
        print(f"Not Downloaded: {count} items, label={label!r}")

        # Dump info about the first few rows
        rows = await page.evaluate("""() => {
            const out = [];
            for (let idx = 0; idx < 4; idx++) {
                const cell = document.getElementById('cell11_' + idx);
                if (!cell) break;
                const cb = cell.querySelector('input[type="checkbox"][name="checkbox"]');
                const tr = cell.closest('tr');
                const cells = tr ? [...tr.querySelectorAll('td')].map(td => td.textContent.trim()) : [];
                out.push({idx, cbValue: cb ? cb.value : null, rowCells: cells});
            }
            return out;
        }""")
        for r in rows:
            print(f"row {r['idx']}: cbValue={r['cbValue']!r}")
            print(f"  cells: {r['rowCells']}")

        # Select row 0 and click 1-CLICK DOWNLOAD
        selected = await page.evaluate("""() => {
            const cell = document.getElementById('cell11_0');
            const cb = cell ? cell.querySelector('input[type="checkbox"][name="checkbox"]') : null;
            if (cb && !cb.checked) { cb.click(); return cb.value; }
            return null;
        }""")
        print(f"\nSelected checkbox value: {selected}")

        dl_click = await page.evaluate("""() => {
            const btn = document.getElementById('simple-download-button');
            if (btn) { btn.click(); return 'clicked'; }
            return 'not found';
        }""")
        print(f"Download click: {dl_click}")

        # Poll for up to 5 minutes, logging elapsed time + any visible
        # error/dialog/agreement text on the page.
        start = asyncio.get_event_loop().time()
        for i in range(60):  # 60 * 5s = 300s
            await asyncio.sleep(5)
            elapsed = asyncio.get_event_loop().time() - start

            link_href = await page.evaluate("""() => {
                const link = document.getElementById('simple-download-link');
                if (!link) return null;
                const href = link.href || '';
                return (href && !href.endsWith('#')) ? href : null;
            }""")

            messages = await page.evaluate("""() => {
                const sels = [
                    '.ida-message', '.error', '.alert', '.dialog', '.modal',
                    '[class*="error"]', '[class*="Error"]', '[class*="agreement"]',
                    '[class*="Agreement"]', '[role="alert"]', '[role="dialog"]',
                ];
                const seen = new Set();
                const texts = [];
                for (const s of sels) {
                    for (const el of document.querySelectorAll(s)) {
                        if (el.offsetParent === null) continue;
                        const t = el.textContent.trim();
                        if (t && t.length < 300 && !seen.has(t)) { seen.add(t); texts.push(t); }
                    }
                }
                return texts;
            }""")

            print(f"[{elapsed:6.1f}s] link={link_href!r} messages={messages}")

            if link_href:
                print(f"\n*** LINK POPULATED after {elapsed:.1f}s: {link_href}")
                break
        else:
            print(f"\n*** LINK NEVER POPULATED after {elapsed:.1f}s")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
