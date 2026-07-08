#!/usr/bin/env python3
"""Decisive probe: does a jar download register server-side?

Open the Not-Downloaded subset and report its label count + visible image
ids, then check whether the 18 ids we already pulled via the jar are absent
(i.e. IDA marked them downloaded)."""
from __future__ import annotations
import asyncio, os, sys

from pathlib import Path

# download_collection lives in the sibling download/ directory
_INSPECT_DIR = Path(__file__).resolve().parent
_DOWNLOAD_DIR = _INSPECT_DIR.parent / "download"
sys.path.insert(0, str(_DOWNLOAD_DIR))
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from download_collection import (
    loni_login, navigate_to_collection, count_not_downloaded,
    visible_not_downloaded_ids, LONI_ADV_SEARCH_URL, NAV_TIMEOUT_MS,
)

COLL = "ADNI_Converters_fMRI"
LOCAL = {852750, 876886, 893047, 896824, 996469, 1018170, 1038048, 1043575,
         1061092, 1132982, 1169385, 1187879, 1225860, 1310788, 1481650,
         1486837, 1577480, 1600185}


async def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    u = (os.environ.get("LONI_USERNAME") or os.environ.get("IDA_USERNAME")
         or os.environ.get("ADNI_USERNAME", ""))
    p = (os.environ.get("LONI_PASSWORD") or os.environ.get("IDA_PASSWORD")
         or os.environ.get("ADNI_PASSWORD", ""))
    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        pg = await (await b.new_context()).new_page()
        assert await loni_login(pg, u, p)
        await pg.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        assert await navigate_to_collection(pg, COLL)
        cnt, label = await count_not_downloaded(pg)
        ids = await visible_not_downloaded_ids(pg)
        print(f"\nNot-Downloaded node label : {label!r}")
        print(f"visible row checkboxes    : {cnt}")
        print(f"visible image ids ({len(ids)}) : {sorted(ids)[:25]}")
        overlap = LOCAL & set(ids)
        print(f"\nour 18 downloaded ids still shown as Not-Downloaded: "
              f"{len(overlap)} -> {sorted(overlap)}")
        print("INTERPRETATION:",
              "jar download REGISTERS server-side (they're gone)" if not overlap
              else "jar download does NOT register (they still appear)")
        await b.close()


asyncio.run(main())
