#!/usr/bin/env python3
"""
One-off inspection: log in once, then repeatedly (full page reload +
navigate_to_collection + count_not_downloaded) dump the list of visible
'Not Downloaded' row image_ids and the '(N)' label, to check whether the
'first page' of 18 rows is stable/cached across reloads or varies, and how
many of those rows are images we already have locally.

Also looks for pagination controls / sort-column links near the table.
"""

import asyncio
import os
import re
import sys
from pathlib import Path

INSPECT_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = INSPECT_DIR.parent / "download"
sys.path.insert(0, str(DOWNLOAD_DIR))

from dotenv import load_dotenv
from playwright.async_api import async_playwright

import download_collection as dc

N_RELOADS = 5


async def main() -> None:
    dc._logger = dc.setup_logging(INSPECT_DIR.parent.parent.parent.parent / "logs" / "adni-download" / "inspect_not_downloaded_pages.log")

    load_dotenv(INSPECT_DIR.parent / ".env")
    username = os.environ.get("LONI_USERNAME") or os.environ.get("IDA_USERNAME") or os.environ.get("ADNI_USERNAME", "")
    password = os.environ.get("LONI_PASSWORD") or os.environ.get("IDA_PASSWORD") or os.environ.get("ADNI_PASSWORD", "")

    skip_ids = dc.local_image_ids(dc.DEFAULT_OUTPUT_DIR, dc.NIFTI_OUTPUT_DIR)
    print(f"Local skip-set ({len(skip_ids)} ids): {sorted(skip_ids)}\n")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        ok = await dc.loni_login(page, username, password)
        if not ok:
            sys.exit(1)

        for i in range(N_RELOADS):
            await page.goto(dc.LONI_ADV_SEARCH_URL, timeout=dc.NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            await asyncio.sleep(5)
            ok = await dc.navigate_to_collection(page, "mci-all-v2")
            if not ok:
                print(f"[{i}] navigate_to_collection failed")
                continue

            count, label = await dc.count_not_downloaded(page)

            rows = await page.evaluate("""() => {
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
            ids = [int(v[1:]) for v in rows if v.startswith("I")]
            new_ids = [i for i in ids if i not in skip_ids]
            print(f"[{i}] count={count} label={label!r}")
            print(f"    rows ({len(ids)}): {ids}")
            print(f"    NEW (not in skip-set): {new_ids}\n")

        # Look for pagination controls / sort links near the result table
        pagination = await page.evaluate("""() => {
            const out = [];
            const sels = [
                '[id*="page" i]', '[class*="page" i]', '[id*="Page"]',
                'a[href*="javascript"]', '[onclick*="sort" i]', '[onclick*="page" i]',
            ];
            for (const s of sels) {
                for (const el of document.querySelectorAll(s)) {
                    if (el.offsetParent === null) continue;
                    const t = (el.textContent || '').trim();
                    if (t && t.length < 40) out.push({sel: s, text: t, id: el.id, cls: el.className});
                }
            }
            return out;
        }""")
        print("Pagination/sort candidates:")
        for p in pagination[:40]:
            print(" ", p)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
