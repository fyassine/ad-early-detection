#!/usr/bin/env python3
"""
One-off inspection of the LONI 'Not Downloaded' table for a collection.

Dumps: the 'Not Downloaded (N)' tree-node label, the list of cell11_N
checkbox values currently rendered, and any pagination controls — to
figure out why every batch in download_collection.py re-selects the
same image_id.
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
    dc._logger = dc.setup_logging(INSPECT_DIR.parent.parent.parent.parent / "logs" / "adni-download" / "inspect_collection.log")

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

        nd_label = await page.evaluate("""() => {
            const labels = [...document.querySelectorAll('#collections .ygtvlabel, #collections_tree .ygtvlabel')];
            const notDl = labels.find(l => l.textContent.trim().startsWith('Not Downloaded'));
            if (notDl) { notDl.click(); return notDl.textContent.trim(); }
            return null;
        }""")
        await asyncio.sleep(5)

        info = await page.evaluate("""() => {
            const rows = [];
            let idx = 0, cell;
            while ((cell = document.getElementById('cell11_' + idx))) {
                const cb = cell.querySelector('input[type="checkbox"][name="checkbox"]');
                const row = cell.closest('tr');
                let subj = null;
                if (row) {
                    const tds = [...row.querySelectorAll('td')];
                    subj = tds.map(td => td.textContent.trim()).filter(Boolean).slice(0, 6);
                }
                rows.push({value: cb ? cb.value : null, checked: cb ? cb.checked : null, row: subj});
                idx++;
            }
            const pagCandidates = [...document.querySelectorAll('a,button,div,span,td')]
                .filter(e => {
                    const t = (e.textContent || '').trim();
                    const c = (e.className || '').toString();
                    return /^(next|prev|previous|»|›|‹|«|\\d+)$/i.test(t) || /pag|paginat/i.test(c);
                })
                .map(e => ({tag: e.tagName, cls: (e.className||'').toString(), text: (e.textContent||'').trim().slice(0,30), id: e.id}))
                .slice(0, 40);
            // Look for a results-summary text e.g. "1-18 of 826"
            const bodyText = document.body.innerText;
            const summaryMatch = bodyText.match(/\\d+\\s*-\\s*\\d+\\s*of\\s*\\d+/i);
            return {rowCount: rows.length, rows, pagCandidates, summary: summaryMatch ? summaryMatch[0] : null};
        }""")

        print("Not Downloaded label:", nd_label)
        print("Results summary:", info["summary"])
        print("Row count (cell11_N):", info["rowCount"])
        print("\nRows:")
        for r in info["rows"]:
            print(" ", r["value"], "checked=", r["checked"], "row=", r["row"])
        print("\nPagination candidates:")
        for p in info["pagCandidates"]:
            print(" ", p)

        screenshots_dir = INSPECT_DIR / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        await page.screenshot(path=str(screenshots_dir / "not_downloaded_table.png"), full_page=True)
        html = await page.content()
        (screenshots_dir / "html" / "not_downloaded_full.html").write_text(html)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
