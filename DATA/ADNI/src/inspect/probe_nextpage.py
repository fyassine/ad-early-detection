#!/usr/bin/env python3
"""Confirm the collection table paginates via SelectHandler.nextPage():
read page 0 checkbox values, click Next, read page 1, show they differ."""
from __future__ import annotations
import asyncio, os, re, sys

from pathlib import Path

# download_collection lives in the sibling download/ directory
_INSPECT_DIR = Path(__file__).resolve().parent
_DOWNLOAD_DIR = _INSPECT_DIR.parent / "download"
sys.path.insert(0, str(_DOWNLOAD_DIR))
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from download_collection import (
    loni_login, navigate_to_collection, LONI_ADV_SEARCH_URL, NAV_TIMEOUT_MS,
)

COLL = "ADNI_Converters_fMRI"
READ = """() => {
    const out=[]; let i=0,c;
    while((c=document.getElementById('cell11_'+i))){
        for(const cb of c.querySelectorAll('input[type=checkbox][name=checkbox]')) out.push(cb.value);
        i++;
    }
    return out;
}"""
CLICK_NEXT = """() => {
    for (const inp of document.querySelectorAll('input')) {
        const o = inp.getAttribute('onclick') || '';
        if (o.includes('nextPage')) { inp.click(); return true; }
    }
    return false;
}"""


async def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    u = (os.environ.get("LONI_USERNAME") or os.environ.get("IDA_USERNAME")
         or os.environ.get("ADNI_USERNAME", ""))
    p = (os.environ.get("LONI_PASSWORD") or os.environ.get("IDA_PASSWORD")
         or os.environ.get("ADNI_PASSWORD", ""))
    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        pg = await (await b.new_context()).new_page()

        pages_seen = []
        def on_resp(r):
            m = re.search(r"collectPages\?pageNumber=(\d+)", r.url)
            if m:
                pages_seen.append(int(m.group(1)))
        pg.on("response", on_resp)

        assert await loni_login(pg, u, p)
        await pg.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        assert await navigate_to_collection(pg, COLL)
        await asyncio.sleep(3)

        v0 = await pg.evaluate(READ)
        print(f"page0: {len(v0)} rows, first={v0[:3]} last={v0[-3:]}")

        clicked = await pg.evaluate(CLICK_NEXT)
        print(f"clicked next button: {clicked}")
        await asyncio.sleep(4)

        v1 = await pg.evaluate(READ)
        print(f"page1: {len(v1)} rows, first={v1[:3]} last={v1[-3:]}")
        print(f"overlap page0∩page1: {len(set(v0) & set(v1))} (want 0)")
        print(f"collectPages pageNumbers requested: {pages_seen}")

        # click next a few more times, see how high pageNumber goes
        for _ in range(3):
            await pg.evaluate(CLICK_NEXT)
            await asyncio.sleep(3)
        print(f"after 3 more nexts, pageNumbers: {pages_seen}")
        await b.close()


asyncio.run(main())
