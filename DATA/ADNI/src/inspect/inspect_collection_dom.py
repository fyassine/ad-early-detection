#!/usr/bin/env python3
"""Read-only DOM inspector for the ADNI_Converters_fMRI collection table.

Logs in, opens the collection, and dumps:
  * how many cell11_N rows are rendered,
  * the ancestor chain of cell11_0 with overflow/scroll dimensions
    (to find the real scroll container), and
  * any pagination / "records per page" / total-count controls.

No downloads. Pure diagnosis so we can fix enumeration correctly."""
from __future__ import annotations

import asyncio
import json
import os
import sys
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


async def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    user = (os.environ.get("LONI_USERNAME") or os.environ.get("IDA_USERNAME")
            or os.environ.get("ADNI_USERNAME", ""))
    pwd = (os.environ.get("LONI_PASSWORD") or os.environ.get("IDA_PASSWORD")
           or os.environ.get("ADNI_PASSWORD", ""))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(accept_downloads=True)
        page = await ctx.new_page()

        captured: list[dict] = []

        async def on_response(resp):
            u = resp.url
            if any(k in u for k in ("collectDetail", "collection", "Collection", "search")):
                try:
                    body = await resp.text()
                except Exception:
                    body = "<no body>"
                captured.append({"url": u, "status": resp.status,
                                 "len": len(body), "head": body[:600]})

        page.on("response", on_response)

        assert await loni_login(page, user, pwd), "login failed"
        await page.goto(LONI_ADV_SEARCH_URL, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
        await asyncio.sleep(5)
        assert await navigate_to_collection(page, COLL), "navigate failed"
        await asyncio.sleep(4)

        print("\n=== AJAX responses (collection/collectDetail/search) ===")
        for c in captured:
            # count how many image-id occurrences in the body (rough row count)
            n_ids = len(__import__("re").findall(r'I\d{5,}', c["head"])) if c["len"] < 4000 else "?"
            print(json.dumps({"url": c["url"][:160], "status": c["status"],
                              "len": c["len"]}, indent=2))
        print("=== end AJAX ===\n")

        # Dump every <select> and its options (page-size control?), and any
        # element whose onclick/href mentions page/pageNumber/collectDetail.
        controls = await page.evaluate(r"""() => {
            const out = {selects: [], pagers: []};
            for (const s of document.querySelectorAll('select')) {
                out.selects.push({
                    id: s.id, name: s.name,
                    opts: [...s.options].map(o => o.value + ':' + o.text).slice(0, 12),
                });
            }
            for (const e of document.querySelectorAll('[onclick],[href]')) {
                const a = (e.getAttribute('onclick') || '') + ' ' + (e.getAttribute('href') || '');
                if (/page|pageNumber|collectDetail|offset|start|rows/i.test(a))
                    out.pagers.push({tag: e.tagName, text: (e.textContent||'').trim().slice(0,30),
                                     attr: a.slice(0, 90)});
            }
            out.pagers = out.pagers.slice(0, 20);
            return out;
        }""")
        print("=== selects / pager controls ===")
        print(json.dumps(controls, indent=2))
        print("=== end controls ===\n")

        info = await page.evaluate("""() => {
            const out = {};

            // 1) how many rendered rows
            let n = 0; while (document.getElementById('cell11_' + n)) n++;
            out.renderedRows = n;

            // 2) ancestor chain of cell11_0 (find the scroll container)
            const chain = [];
            let el = document.getElementById('cell11_0');
            for (let i = 0; el && i < 18 && el !== document.body; i++) {
                const s = getComputedStyle(el);
                chain.push({
                    tag: el.tagName,
                    id: el.id || null,
                    cls: (el.className || '').toString().slice(0, 80),
                    overflowY: s.overflowY,
                    scrollH: el.scrollHeight,
                    clientH: el.clientHeight,
                    scrollable: el.scrollHeight > el.clientHeight + 5,
                });
                el = el.parentElement;
            }
            out.ancestorChain = chain;

            // 3) page-level scroll
            const se = document.scrollingElement || document.documentElement;
            out.pageScroll = {scrollH: se.scrollHeight, clientH: se.clientHeight};

            // 4) pagination / paginator / records-per-page controls
            const pag = [];
            for (const sel of ['.yui-pg-container', '[class*="paginat"]', '[class*="pg-"]',
                               'select', 'a']) {
                for (const e of document.querySelectorAll(sel)) {
                    const t = (e.textContent || e.value || '').trim();
                    const c = (e.className || '').toString();
                    if (/page|paginat|records|next|prev|\\b\\d+\\s*-\\s*\\d+\\b/i.test(t + ' ' + c)
                        && t.length < 60) {
                        pag.push({tag: e.tagName, cls: c.slice(0, 60), text: t.slice(0, 60)});
                    }
                }
            }
            // de-dup
            out.paginationCandidates = pag.filter((v, i, a) =>
                a.findIndex(x => x.text === v.text && x.cls === v.cls) === i).slice(0, 25);

            // 5) any "X of 826" / total-count text
            const totals = [];
            for (const e of document.querySelectorAll('*')) {
                if (e.children.length) continue;
                const t = (e.textContent || '').trim();
                if (/\\b826\\b/.test(t) && t.length < 80) totals.push(t);
            }
            out.totalsText = [...new Set(totals)].slice(0, 15);

            return out;
        }""")

        print(json.dumps(info, indent=2))
        await browser.close()


asyncio.run(main())
