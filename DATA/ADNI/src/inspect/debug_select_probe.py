#!/usr/bin/env python3
"""One-shot diagnostic: log in, open ADNI_Converters_fMRI Not Downloaded, inspect the checkbox
tables, select one item, click 1-CLICK, and report exactly where the flow breaks.
No files are downloaded. Read-only diagnosis."""
from __future__ import annotations
import asyncio, os, sys
from pathlib import Path

# download_collection lives in the sibling download/ directory
_INSPECT_DIR = Path(__file__).resolve().parent
_DOWNLOAD_DIR = _INSPECT_DIR.parent / "download"
sys.path.insert(0, str(_DOWNLOAD_DIR))
from dotenv import load_dotenv
from playwright.async_api import async_playwright

import download_collection as dc
from download_collection import (
    loni_login, navigate_to_collection, count_not_downloaded,
    LONI_ADV_SEARCH_URL, NAV_TIMEOUT_MS,
)

COLL = "ADNI_Converters_fMRI"


async def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    user = os.environ.get("ADNI_USERNAME", "")
    pwd = os.environ.get("ADNI_PASSWORD", "")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(accept_downloads=True)
        page = await ctx.new_page()
        assert await loni_login(page, user, pwd)
        await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        assert await navigate_to_collection(page, COLL)
        cnt, label = await count_not_downloaded(page)
        print(f"\ncount_not_downloaded -> {cnt} visible, label={label!r}")

        # How many cell11_N tables / duplicate ids exist, and all their values?
        info = await page.evaluate("""() => {
            const out = {dupCheck: {}, cells: []};
            // duplicate-id detection
            const all = [...document.querySelectorAll('[id^="cell11_"]')];
            for (const el of all) out.dupCheck[el.id] = (out.dupCheck[el.id]||0)+1;
            // values via getElementById (what our selector actually sees)
            let idx=0, cell;
            while ((cell = document.getElementById('cell11_'+idx))) {
                const vals=[...cell.querySelectorAll('input[type=checkbox][name=checkbox]')].map(c=>c.value);
                out.cells.push({idx, vals});
                idx++;
                if (idx>5) break;
            }
            return out;
        }""")
        dups = {k: v for k, v in info["dupCheck"].items() if v > 1}
        print("duplicate cell11 ids:", dups or "none")
        print("first cells via getElementById:", info["cells"][:4])

        # Selected-count BEFORE
        def sel_text_js():
            return """() => {
                const el=[...document.querySelectorAll('*')].find(e=>/items? selected/i.test(e.childNodes.length===1?e.textContent:''));
                return el ? el.textContent.trim() : 'n/a';
            }"""
        before = await page.evaluate(sel_text_js())
        print("selected-count before:", before)

        # Select first checkbox (mirror our prod selector) and report value
        picked = await page.evaluate("""() => {
            let idx=0, cell;
            while ((cell=document.getElementById('cell11_'+idx))) {
                const cb=cell.querySelector('input[type=checkbox][name=checkbox]');
                if (cb) { if(!cb.checked) cb.click(); return cb.value; }
                idx++;
            }
            return null;
        }""")
        await asyncio.sleep(2)
        after = await page.evaluate(sel_text_js())
        print(f"picked checkbox value: {picked}")
        print("selected-count after :", after)

        # Is the 1-CLICK button enabled now?
        btn = await page.evaluate("""() => {
            const b=document.getElementById('simple-download-button');
            return b ? {found:true, disabled:b.disabled, cls:b.className} : {found:false};
        }""")
        print("1-CLICK button:", btn)

        # Click it and capture downloadKey response
        body = {}
        ev = asyncio.Event()
        async def cap(r):
            if "downloadKey" in r.url:
                try: body.update(url=r.url, status=r.status, text=(await r.text())[:500])
                except Exception as e: body.update(url=r.url, err=str(e))
                ev.set()
        page.on("response", cap)
        await page.evaluate("""() => {const b=document.getElementById('simple-download-button'); if(b) b.click();}""")
        try: await asyncio.wait_for(ev.wait(), timeout=30)
        except asyncio.TimeoutError: print("downloadKey: NO RESPONSE within 30s")
        print("downloadKey response:", body)

        href=None
        for _ in range(10):
            await asyncio.sleep(2)
            href = await page.evaluate("""() => {const l=document.getElementById('simple-download-link'); return l?(l.href||'').slice(0,160):null;}""")
            if href and not href.endswith('#'): break
        print("simple-download-link href:", href)
        await browser.close()

asyncio.run(main())
