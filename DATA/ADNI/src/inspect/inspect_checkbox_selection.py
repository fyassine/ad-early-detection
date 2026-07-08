#!/usr/bin/env python3
"""
One-off inspection: log in, navigate to mci-all-v2 -> Not Downloaded,
click ONE row checkbox (cell11_0), then inspect:
  - hidden <input name="id"> tags (what the form thinks is selected)
  - CHECK_BOX_MANAGER state (getNumberOfChecks / getNumberOfDownloadables)
  - the actual outgoing request when clicking #simple-download-button
    (to see which image id(s) the server is asked to zip)
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
    dc._logger = dc.setup_logging(INSPECT_DIR.parent.parent.parent.parent / "logs" / "adni-download" / "inspect_checkbox.log")

    load_dotenv(INSPECT_DIR.parent / ".env")
    username = os.environ.get("LONI_USERNAME") or os.environ.get("IDA_USERNAME") or os.environ.get("ADNI_USERNAME", "")
    password = os.environ.get("LONI_PASSWORD") or os.environ.get("IDA_PASSWORD") or os.environ.get("ADNI_PASSWORD", "")

    requests_seen = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        def on_request(req):
            url = req.url
            if "download" in url.lower() or "Key" in url or "checkbox" in url.lower() or "simpleDownload" in url:
                requests_seen.append((req.method, url, req.post_data))
        page.on("request", on_request)

        ok = await dc.loni_login(page, username, password)
        if not ok:
            sys.exit(1)

        await page.goto(dc.LONI_ADV_SEARCH_URL, timeout=dc.NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)

        ok = await dc.navigate_to_collection(page, "mci-all-v2")
        if not ok:
            sys.exit(1)

        await page.evaluate("""() => {
            const labels = [...document.querySelectorAll('#collections .ygtvlabel, #collections_tree .ygtvlabel')];
            const notDl = labels.find(l => l.textContent.trim().startsWith('Not Downloaded'));
            if (notDl) notDl.click();
        }""")
        await asyncio.sleep(5)

        # Before clicking: dump cell11_0 value and any hidden id inputs / manager state
        before = await page.evaluate("""() => {
            const cell = document.getElementById('cell11_0');
            const cb = cell ? cell.querySelector('input[type=\"checkbox\"][name=\"checkbox\"]') : null;
            const hiddenIds = [...document.querySelectorAll('input[name=\"id\"]')].map(i => i.value);
            let mgr = null;
            try {
                mgr = {
                    numChecks: CHECK_BOX_MANAGER.getNumberOfChecks(),
                    numDownloadables: CHECK_BOX_MANAGER.getNumberOfDownloadables(),
                    isCheckAll: CHECK_BOX_MANAGER._isCheckAll,
                };
            } catch(e) { mgr = 'error: ' + e; }
            return {cbValue: cb ? cb.value : null, cbChecked: cb ? cb.checked : null, hiddenIds, mgr};
        }""")
        print("BEFORE click:", before)

        # Click cell11_0's checkbox
        clicked_value = await page.evaluate("""() => {
            const cell = document.getElementById('cell11_0');
            const cb = cell ? cell.querySelector('input[type=\"checkbox\"][name=\"checkbox\"]') : null;
            if (cb && !cb.checked) cb.click();
            return cb ? cb.value : null;
        }""")
        print("Clicked checkbox value:", clicked_value)
        await asyncio.sleep(2)

        after = await page.evaluate("""() => {
            const cell = document.getElementById('cell11_0');
            const cb = cell ? cell.querySelector('input[type=\"checkbox\"][name=\"checkbox\"]') : null;
            const hiddenIds = [...document.querySelectorAll('input[name=\"id\"]')].map(i => i.value);
            let mgr = null;
            try {
                mgr = {
                    numChecks: CHECK_BOX_MANAGER.getNumberOfChecks(),
                    numDownloadables: CHECK_BOX_MANAGER.getNumberOfDownloadables(),
                    isCheckAll: CHECK_BOX_MANAGER._isCheckAll,
                    checkedIds: (CHECK_BOX_MANAGER.checkedIds || CHECK_BOX_MANAGER.checked || null),
                };
            } catch(e) { mgr = 'error: ' + e; }
            // Also dump any global vars that look like selection state
            const globals = Object.keys(window).filter(k => /check|select|download|cart/i.test(k));
            return {cbValue: cb ? cb.value : null, cbChecked: cb ? cb.checked : null, hiddenIds, mgr, globals};
        }""")
        print("AFTER click:", after)

        # Now click the 1-CLICK DOWNLOAD button and capture the request
        await page.evaluate("""() => {
            const btn = document.getElementById('simple-download-button');
            if (btn) btn.click();
        }""")
        await asyncio.sleep(8)

        print("\nRequests captured (download/key/checkbox related):")
        for method, url, post_data in requests_seen:
            print(f"  {method} {url}")
            if post_data:
                print(f"    post_data: {post_data[:500]}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
