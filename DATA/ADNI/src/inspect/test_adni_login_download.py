#!/usr/bin/env python3
"""
ADNI Login + Single Image Download Test
========================================
Implements the full LONI IDA 3-step flow:
  1. Search with Advanced Search (Image section + fMRI + Original + image ID)
  2. Select all results → Add to Collection
  3. Download from collection as NIfTI

DOM findings (from HTML dump analysis):
  - Image section toggle:     #imageModalityOption  (must be checked)
  - fMRI modality:            input[name="imgModality_checkBox"][value="2"]
  - Original image type:      #originalOption
  - Image ID field:           #imageIdText  (name=imgId)
  - Search button:            #advSearchQuery
  - Select-all results:       #advResultSelectAll
  - Add to Collection:        #advResultAddCollectId
  - Collection dialog form:   document.regroup
  - Collection name input:    input[name="newName"] in document.regroup
  - Simple download button:   #simple-download-button  (uses document.forms.collection)
  - Download link href:       https://ida.loni.usc.edu/download/image?key=...

Download note:
  - #simple-download-button  → AJAX for download key → shows #simple-download-modal
  - #simple-download-link    → actual file download (MUST wait for this href to populate)

Run:
    cd DATA/ADNI/src
    python test_adni_login_download.py
"""

from __future__ import annotations

import asyncio
import getpass
import os
import sys
import time
import zipfile
from pathlib import Path

SCRIPT_DIR     = Path(__file__).resolve().parent
ENV_FILE       = SCRIPT_DIR / ".env"
SCREENSHOT_DIR = SCRIPT_DIR / "screenshots"
HTML_DIR       = SCREENSHOT_DIR / "html"
TEST_DL_DIR    = SCRIPT_DIR / "test_download"

TEST_IMAGE_ID  = 249536   # confirmed working in manual test (screenshot)

LONI_BASE       = "https://ida.loni.usc.edu"
LONI_LOGIN_URL  = f"{LONI_BASE}/login.jsp?project=ADNI"
LONI_ADV_SEARCH = (
    f"{LONI_BASE}/pages/access/search.jsp"
    "?project=ADNI&tab=advSearch&page=SEARCH&subPage=NEW_ADV_QUERY"
)

COLLECTION_NAME = "mci-v2"   # Fresh collection, separate from old "mci" test runs

NAV_TIMEOUT = 60_000
DL_TIMEOUT  = 600_000   # 10 min — server zip generation can be slow


# ── helpers ────────────────────────────────────────────────────────────────────

def load_credentials() -> tuple[str, str]:
    username, password = "", ""
    if ENV_FILE.exists():
        with open(ENV_FILE) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("ADNI_USERNAME="):
                    username = line.split("=", 1)[1].strip()
                elif line.startswith("ADNI_PASSWORD="):
                    password = line.split("=", 1)[1].strip()
    username = os.environ.get("ADNI_USERNAME", username)
    password = os.environ.get("ADNI_PASSWORD", password)
    if not username:
        username = input("LONI IDA username: ").strip()
    if not password:
        password = getpass.getpass("LONI IDA password: ")
    return username, password


def step(n: int | str, title: str) -> None:
    print(f"\n{'=' * 60}\n  STEP {n}: {title}\n{'=' * 60}")


async def ss(page, name: str) -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SCREENSHOT_DIR / f"{name}.png"
    await page.screenshot(path=str(out), full_page=True)
    print(f"  📸 screenshot → {out.name}")


async def dump_html(page, name: str) -> None:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    html = await page.content()
    out = HTML_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  📄 HTML dump  → screenshots/html/{name}.html  ({len(html):,} chars)")


async def try_click(page, selectors: list[str], label: str, timeout: int = 3000) -> bool:
    for sel in selectors:
        try:
            elem = page.locator(sel).first
            if await elem.is_visible(timeout=timeout):
                txt = (await elem.text_content() or "").strip()[:40]
                print(f"  → Clicking '{label}' via {sel!r}  text={txt!r}")
                await elem.click()
                return True
        except Exception:
            continue
    return False


# ── login ──────────────────────────────────────────────────────────────────────

async def do_login(page, username: str, password: str) -> bool:
    step("L-A", "Navigate to login.jsp and accept cookie")
    await page.goto(LONI_LOGIN_URL, timeout=NAV_TIMEOUT, wait_until="domcontentloaded")
    await asyncio.sleep(5)
    print(f"  URL: {page.url}")
    await ss(page, "01_login_page")

    # Accept cookie policy (enables the login button in the nav bar)
    cookie_result = await page.evaluate("""() => {
        const el = document.querySelector('.ida-cookie-policy-accept');
        if (!el) return 'NOT FOUND';
        el.click();
        return 'clicked: ' + el.outerHTML.substring(0, 80);
    }""")
    print(f"  Cookie accept: {cookie_result}")
    await asyncio.sleep(2)  # give SPA time to enable the login button

    # Wait for login nav button to become enabled
    login_ready = False
    for _ in range(10):
        login_ready = await page.evaluate("""() => {
            const btn = document.querySelector('div.ida-menu-option.login:not(.disabled)');
            return !!btn;
        }""")
        if login_ready:
            print("  ✓ Login button is now enabled")
            break
        await asyncio.sleep(1)
    if not login_ready:
        print("  ! Login button still disabled — trying nav click anyway")

    step("L-B", "Click 'Log In' nav button and fill form")
    # Show all login-related visible elements for diagnostics
    nav_info = await page.evaluate("""() => {
        const els = [...document.querySelectorAll('div[class*="login"], span[class*="login"], a[class*="login"]')];
        return els.filter(e => e.offsetParent !== null).map(e => ({
            tag: e.tagName, cls: e.className.substring(0, 60), id: e.id, text: e.innerText.trim().substring(0, 40)
        }));
    }""")
    print(f"  Nav login elements ({len(nav_info)}):")
    for el in nav_info[:5]:
        print(f"    {el['tag']} cls={el['cls']!r} id={el['id']!r} text={el['text']!r}")

    # Click the nav login toggle (opens the dropdown form)
    nav_clicked = await page.evaluate("""() => {
        const sels = [
            'div.ida-menu-option.login:not(.disabled)',
            'div.ida-menu-option.sub-menu.login',
            'div.ida-menu-option.login',
            '.ida-menu-option-label',
        ];
        for (const s of sels) {
            const el = document.querySelector(s);
            if (el && el.offsetParent !== null) {
                el.click();
                return 'JS clicked: ' + s;
            }
        }
        return 'nav login element not found';
    }""")
    print(f"  Nav click: {nav_clicked}")
    await asyncio.sleep(3)  # wait for dropdown animation

    filled = await page.evaluate("""([user, pass]) => {
        function fill(sels, val) {
            for (const s of sels) {
                const el = document.querySelector(s);
                if (el) {
                    el.value = val;
                    el.dispatchEvent(new Event('input',  {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                    return s;
                }
            }
            return null;
        }
        return {
            u: fill(["input[name='userEmail']", "input[name='username']", "input[type='email']"], user),
            p: fill(["input[name='userPassword']", "input[name='password']", "input[type='password']"], pass),
        };
    }""", [username, password])
    print(f"  Fill: user={filled['u']}  pass={filled['p']}")
    if not filled["u"] or not filled["p"]:
        print("  ✗ Could not fill form fields")
        return False

    # Try Playwright click first (works when dropdown is fully open/animated),
    # then JS click as fallback — .login-btn is a <span> in a SPA nav dropdown
    # and Playwright's is_visible() can be flaky for it.
    submitted = await try_click(page, [
        ".login-btn", "button[type='submit']", "input[type='submit']",
        "span.login-btn", "[class*='login-btn']",
    ], "Submit")
    if not submitted:
        js_result = await page.evaluate("""() => {
            const sels = ['.login-btn', 'span.login-btn', 'input[type="submit"]', 'button[type="submit"]'];
            for (const s of sels) {
                const el = document.querySelector(s);
                if (el) { el.click(); return 'JS click: ' + s; }
            }
            const pw = document.querySelector("input[type='password']");
            if (pw) {
                pw.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', keyCode:13, bubbles:true}));
                return 'Enter on password';
            }
            return 'no submit found';
        }""")
        print(f"  JS submit fallback: {js_result}")

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=NAV_TIMEOUT)
    except Exception:
        pass
    await asyncio.sleep(4)
    await ss(page, "02_after_login")

    url = page.url
    if "login" in url.lower():
        print(f"  ✗ Still on login: {url}")
        return False
    print(f"  ✓ Logged in: {url}")
    return True


# ── search ─────────────────────────────────────────────────────────────────────

async def run_search(page, image_id: int) -> bool:
    step(2, "Navigate to Advanced Image Search")
    await page.goto(LONI_ADV_SEARCH, timeout=NAV_TIMEOUT, wait_until="domcontentloaded")
    await asyncio.sleep(5)
    await ss(page, "03_adv_search")
    if "login" in page.url.lower():
        print("  ✗ Session lost")
        return False
    print(f"  Loaded: {page.url!r}")

    step(3, f"Set up form and search for image_id={image_id}")
    result = await page.evaluate("""([imageId]) => {
        const log = [];

        // Check Image section (enables image filters)
        const imgSec = document.getElementById('imageModalityOption');
        if (imgSec && !imgSec.checked) { imgSec.click(); log.push('checked imageModalityOption'); }
        else if (imgSec) log.push('imageModalityOption already checked');
        else log.push('imageModalityOption NOT FOUND');

        // Check fMRI modality
        const fmri = document.querySelector('input[name="imgModality_checkBox"][value="2"]');
        if (fmri && !fmri.checked) { fmri.click(); log.push('checked fMRI'); }
        else if (fmri) log.push('fMRI already checked');
        else log.push('fMRI checkbox NOT FOUND');

        // Check Original image type
        const orig = document.getElementById('originalOption');
        if (orig && !orig.checked) { orig.click(); log.push('checked originalOption'); }
        else if (orig) log.push('originalOption already checked');
        else log.push('originalOption NOT FOUND');

        return log.join(', ');
    }""", [str(image_id)])
    print(f"  Form: {result}")
    await asyncio.sleep(1)

    filled = await page.evaluate("""([id]) => {
        const el = document.getElementById('imageIdText') || document.querySelector('input[name="imgId"]');
        if (!el) return 'NOT FOUND';
        el.value = id;
        el.dispatchEvent(new Event('input',  {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        return 'filled imageIdText';
    }""", [str(image_id)])
    print(f"  Image ID: {filled}")

    await ss(page, "04_form_filled")

    await page.evaluate("() => { const b = document.getElementById('advSearchQuery'); if (b) b.click(); }")

    # Wait for the AJAX results to populate advTableData.
    # DO NOT click the results tab manually — the SPA auto-switches to it when
    # results arrive, and a premature click clears the DataTable.
    print("  Waiting for search results (up to 30s)...")
    for i in range(20):
        await asyncio.sleep(1.5)
        info = await page.evaluate("""() => {
            const desc = document.getElementById('advTableDescription');
            const resultCbs = document.querySelectorAll('input[id$="_check"][id^="adv_subject_"]');
            const addBtn = document.getElementById('advResultAddCollectId');
            return {
                description: desc ? desc.textContent.trim() : '',
                subjectCbs:  resultCbs.length,
                addBtnFound: !!addBtn,
            };
        }""")
        if info['subjectCbs'] > 0 or 'Result' in info['description']:
            print(f"  ✓ Results ready: {info['description']!r}  ({info['subjectCbs']} subject checkboxes)")
            break
        if i % 4 == 3:
            print(f"  Waiting... ({(i+1)*1.5:.0f}s)  desc={info['description']!r}")
    else:
        print(f"  ! Search results still not visible after 30s: {info}")

    await ss(page, "05_search_results")
    await dump_html(page, "05_search_results")
    print(f"  URL after search: {page.url}")
    return bool(info.get('subjectCbs', 0) > 0 or 'Result' in info.get('description', ''))


# ── add to collection ──────────────────────────────────────────────────────────

async def add_results_to_collection(page, collection_name: str) -> bool:
    step(4, "Select results and Add to Collection")

    # 1. Click the actual subject-level checkboxes in the result rows.
    #    This triggers the SPA's natural flow: handleSubjectCheck →
    #    _addHiddenTag (adds item to advResultCheckBoxes form div) →
    #    _updateDisplay (enables "Add To Collection" button).
    #    Clicking #advResultSelectAll or calling checkAll() without real
    #    checkbox events doesn't populate advResultCheckBoxes.
    selected = await page.evaluate("""() => {
        const log = [];

        // Click every subject-level checkbox in the results table
        const subjectCbs = [...document.querySelectorAll(
            'input[type="checkbox"][id^="adv_subject_"][id$="_check"]'
        )];
        for (const cb of subjectCbs) {
            cb.checked = true;
            cb.click();   // triggers onclick="...handleSubjectCheck(...)"
        }
        log.push('clicked ' + subjectCbs.length + ' subject checkboxes');

        // Also click the header select-all in case there are study/image level cbs
        const selectAll = document.getElementById('advResultSelectAll');
        if (selectAll) {
            selectAll.checked = true;
            selectAll.click();
            log.push('advResultSelectAll clicked');
        }

        // Wait a tick for the SPA to process, then report button state
        const btn = document.getElementById('advResultAddCollectId');
        log.push('btn.disabled=' + (btn ? btn.disabled : 'N/A'));

        // Report how many hidden tags were added to advResultCheckBoxes
        const hiddenDiv = document.getElementById('advResultCheckBoxes');
        log.push('hiddenTags=' + (hiddenDiv ? hiddenDiv.childElementCount : 'N/A'));

        return log.join('; ');
    }""")
    print(f"  Selection: {selected}")
    await asyncio.sleep(1)

    # 2. Remove the disabled attribute so the YAHOO click listener fires.
    #    (If the button is already enabled this is a no-op.)
    click_result = await page.evaluate("""() => {
        const btn = document.getElementById('advResultAddCollectId');
        if (!btn) return 'advResultAddCollectId NOT FOUND';
        btn.removeAttribute('disabled');
        btn.className = btn.className.replace('buttonDisabled', 'button');
        btn.click();
        return 'clicked (was disabled)';
    }""")
    print(f"  Add btn click: {click_result}")
    await asyncio.sleep(2)
    await ss(page, "06_add_collection_dialog")
    await dump_html(page, "06_add_collection_dialog")

    # 3. Inspect dialog state: is it visible? which existing collections are offered?
    dialog_info = await page.evaluate("""() => {
        const dlg = document.getElementById('regroupDialog');
        if (!dlg) return {visible: false, reason: 'regroupDialog element missing'};
        const vis = window.getComputedStyle(dlg).visibility;
        const disp = window.getComputedStyle(dlg).display;
        const nameInput = document.getElementById('nameText');
        const existingSel = document.getElementById('candidateNames');
        // Capture both name and value so we can match by text
        const collections = existingSel
            ? [...existingSel.options].map(o => ({text: o.text.trim(), value: o.value})).filter(o => o.value)
            : [];
        const buttons = [...document.querySelectorAll('button')].map(b => b.textContent.trim());
        return {
            visible: vis !== 'hidden' && disp !== 'none',
            visibility: vis, display: disp,
            hasNameInput: !!nameInput,
            collections: collections,
            buttons: buttons,
        };
    }""")
    print(f"  Dialog: {dialog_info}")

    if not dialog_info.get("visible"):
        # Dialog didn't render — submit the form directly.
        # document.advResultTable contains hidden fields userAction/newName/existingName.
        print("  Dialog not visible — submitting advResultTable form directly")
        submit_result = await page.evaluate("""([collName]) => {
            const form = document.advResultTable || document.forms['advResultTable'];
            if (!form) return 'ERROR: advResultTable form not found';
            if (form.userAction)     form.userAction.value = 'add';
            if (form.newName)        { form.newName.value = collName; form.newName.disabled = false; }
            if (form.existingName)   form.existingName.value = '';
            if (form.newDescription) form.newDescription.value = '';
            form.submit();
            return 'submitted advResultTable with newName=' + collName;
        }""", [collection_name])
        print(f"  Direct submit: {submit_result}")
        await asyncio.sleep(4)
        await ss(page, "07_after_direct_submit")
        return True

    # 4. Dialog IS visible — fill it: prefer existing "mci" collection in dropdown.
    collections = dialog_info.get("collections", [])
    fill_result = await page.evaluate("""([collName, collections]) => {
        const existingSel = document.getElementById('candidateNames');
        const nameInput   = document.getElementById('nameText');

        // Match by TEXT (display name) — values are internal IDs like '496304'
        if (existingSel) {
            const match = [...existingSel.options].find(
                o => o.text.trim() === collName || o.value === collName
            );
            if (match) {
                existingSel.value = match.value;
                existingSel.dispatchEvent(new Event('change', {bubbles: true}));
                if (nameInput) { nameInput.value = ''; nameInput.disabled = true; }
                return 'selected existing: ' + match.text + ' (id=' + match.value + ')';
            }
        }

        // Create a new collection with this name
        if (nameInput) {
            nameInput.value = collName;
            nameInput.disabled = false;
            nameInput.dispatchEvent(new Event('input',  {bubbles: true}));
            nameInput.dispatchEvent(new Event('change', {bubbles: true}));
            if (existingSel) {
                existingSel.value = '';
                existingSel.dispatchEvent(new Event('change', {bubbles: true}));
            }
            return 'new collection name: ' + collName + ' (no existing match in ' + JSON.stringify(collections) + ')';
        }
        return 'no dialog inputs. collections=' + JSON.stringify(collections);
    }""", [collection_name, collections])
    print(f"  Dialog fill: {fill_result}")
    await asyncio.sleep(0.5)

    # 5. Click "OK" button (YAHOO renders it as <button type="button">OK</button>)
    ok_result = await page.evaluate("""() => {
        for (const btn of document.querySelectorAll('button')) {
            if (btn.textContent.trim() === 'OK') {
                btn.click();
                return 'clicked OK button (id=' + btn.id + ')';
            }
        }
        // Fallback: call _submitParentForm directly
        if (typeof _submitParentForm === 'function') {
            try { _submitParentForm(); return 'called _submitParentForm()'; }
            catch(e) { return 'submitParentForm error: ' + e.message; }
        }
        const vis = [...document.querySelectorAll('button')]
            .filter(b => b.offsetParent !== null)
            .map(b => b.textContent.trim() + '(id=' + b.id + ')');
        return 'No OK found. Visible buttons: ' + vis.join(', ');
    }""")
    print(f"  OK click: {ok_result}")
    await asyncio.sleep(4)
    await ss(page, "07_after_add_collection")

    # 6. Check result
    after = await page.evaluate("""() => ({
        url:  window.location.href,
        text: document.body.innerText.substring(0, 300),
    })""")
    print(f"  After URL: {after['url']}")
    confirm_lines = [
        ln for ln in after["text"].splitlines()
        if "collection" in ln.lower() or "added" in ln.lower() or "mci" in ln.lower()
    ]
    print(f"  Confirm lines: {confirm_lines[:5]}")
    return True


# ── download from collection ───────────────────────────────────────────────────

async def download_from_collection(page, collection_name: str) -> str | None:
    """
    Navigate to the Data Collections tab, open the named collection in the tree,
    select all items, click 1-CLICK DOWNLOAD, and wait for the download link.
    Items download in original format (DCM — no NIfTI conversion step here).
    Returns the download URL or None on failure.
    """
    step(5, "Navigate to Data Collections tab")

    # Click Data Collections tab (YAHOO TabView <a> inside <li>)
    tab_clicked = await page.evaluate("""() => {
        const all = [...document.querySelectorAll('a, li a, .yui-nav a')];
        const tab = all.find(el => el.textContent.trim() === 'Data Collections');
        if (tab) { tab.click(); return 'clicked Data Collections tab'; }
        return 'NOT FOUND. Links: ' + all.filter(a => a.textContent.trim())
            .map(a => a.textContent.trim()).slice(0, 15).join(' | ');
    }""")
    print(f"  Tab: {tab_clicked}")
    await asyncio.sleep(3)
    await ss(page, "08_collections_tab")
    await dump_html(page, "08_collections_tab")

    # Expand the YAHOO TreeView "My Collections" node (collapsed by default),
    # then click the named collection.  The tree is in #collections_tree and
    # uses YAHOO.widget.TreeView with tree name 'collections'.
    coll_clicked = await page.evaluate("""([name]) => {
        // Step 1: Expand "My Collections" node (YAHOO TreeView node 1)
        try {
            const myCollNode = YAHOO.widget.TreeView.getNode('collections', 1);
            if (myCollNode && !myCollNode.expanded) myCollNode.expand();
        } catch(e) {}

        // Step 2: Also try clicking the label directly
        const labels = [...document.querySelectorAll('#collections .ygtvlabel, #collections_tree .ygtvlabel')];
        const myCollLabel = labels.find(l => l.textContent.trim() === 'My Collections');
        if (myCollLabel) myCollLabel.click();

        // Step 3: Find the named collection node
        const allLabels = [...document.querySelectorAll('#collections .ygtvlabel, #collections_tree .ygtvlabel')];
        const found = allLabels.find(l => {
            const t = l.textContent.trim();
            return t === name || t.startsWith(name + ' (') || t.startsWith(name + '(');
        });
        if (found) {
            found.click();
            return 'clicked: ' + found.textContent.trim();
        }
        // Diagnostic: show all visible labels
        const vis = allLabels.map(l => l.textContent.trim()).filter(Boolean);
        return 'NOT FOUND "' + name + '". Tree labels: ' + vis.join(' | ');
    }""", [collection_name])
    print(f"  Collection click: {coll_clicked}")
    await asyncio.sleep(2)

    # If collection not found yet, wait for tree to expand and retry
    if "NOT FOUND" in coll_clicked:
        await asyncio.sleep(2)
        coll_clicked2 = await page.evaluate("""([name]) => {
            const allLabels = [...document.querySelectorAll('#collections .ygtvlabel, #collections_tree .ygtvlabel')];
            const found = allLabels.find(l => {
                const t = l.textContent.trim();
                return t === name || t.startsWith(name + ' (');
            });
            if (found) { found.click(); return 'retry clicked: ' + found.textContent.trim(); }
            return 'still not found. Labels: ' + allLabels.map(l => l.textContent.trim()).filter(Boolean).join(' | ');
        }""", [collection_name])
        print(f"  Retry: {coll_clicked2}")
    await asyncio.sleep(2)
    await ss(page, "09_collection_opened")

    # Select all items: the "All" checkbox is the header checkbox in the item table.
    # It selects everything in the current collection view.
    select_all = await page.evaluate("""() => {
        const visBoxes = [...document.querySelectorAll('input[type="checkbox"]')]
            .filter(c => c.offsetParent !== null);
        let count = 0;
        for (const cb of visBoxes) {
            if (!cb.checked) { cb.click(); count++; }
        }
        return 'checked ' + count + ' of ' + visBoxes.length + ' visible checkboxes';
    }""")
    print(f"  Select all: {select_all}")
    await asyncio.sleep(1)
    await ss(page, "10_items_selected")

    # Intercept the downloadKey AJAX response so we know the key was issued
    download_key_info: dict = {}
    download_key_event = asyncio.Event()

    async def capture_response(response):
        if "/ajax/download/downloadKey" in response.url:
            try:
                body = await response.text()
                download_key_info.update(url=response.url, status=response.status, body=body)
                download_key_event.set()
            except Exception as exc:
                download_key_info["error"] = str(exc)
                download_key_event.set()

    page.on("response", capture_response)

    step("5b", "Click 1-CLICK DOWNLOAD and wait for download key")
    dl_click = await page.evaluate("""() => {
        // Primary ID used on the collections detail panel
        const btn = document.getElementById('simple-download-button');
        if (btn) { btn.click(); return 'clicked #simple-download-button'; }
        // Fallback: find by visible text
        const all = [...document.querySelectorAll('button, input[type="button"], input[type="submit"], a')];
        for (const b of all) {
            const t = (b.textContent || b.value || '').trim().toUpperCase();
            if (t.includes('1-CLICK') || t === '1-CLICK DOWNLOAD') {
                b.click();
                return 'clicked by text: ' + b.textContent.trim() + ' id=' + b.id;
            }
        }
        const ids = all.filter(b => b.offsetParent).map(b => (b.textContent || b.value || '').trim()).filter(Boolean);
        return 'download button not found. Visible: ' + ids.slice(0, 15).join(', ');
    }""")
    print(f"  Download button: {dl_click}")

    # Wait for the AJAX downloadKey response (server may take a while to generate zip)
    try:
        await asyncio.wait_for(download_key_event.wait(), timeout=60)
        print(f"  AJAX {download_key_info.get('status')} body: {download_key_info.get('body', '')[:200]}")
    except asyncio.TimeoutError:
        print("  ✗ No downloadKey AJAX in 60s — proceeding to poll modal")

    # Poll for #simple-download-link href to be populated (up to 90s)
    print("  Waiting for download modal to populate (up to 90s)...")
    link_href = None
    for i in range(45):
        await asyncio.sleep(2)
        link_href = await page.evaluate("""() => {
            const link = document.getElementById('simple-download-link');
            if (!link) return null;
            const href = link.href || '';
            return (href && !href.endsWith('#')) ? href : null;
        }""")
        if link_href:
            break
        if i % 5 == 4:
            spinner_cls = await page.evaluate("""() => {
                const s = document.querySelector('.download-spinner');
                return s ? s.className : 'not found';
            }""")
            print(f"  Still waiting... ({(i+1)*2}s)  spinner={spinner_cls!r}")

    await ss(page, "11_download_modal")
    await dump_html(page, "11_download_modal")

    if not link_href:
        modal_state = await page.evaluate("""() => {
            const modal   = document.getElementById('simple-download-modal');
            const link    = document.getElementById('simple-download-link');
            const spinner = document.querySelector('.download-spinner');
            return {
                modalDisplay: modal   ? window.getComputedStyle(modal).display : 'N/A',
                linkHref:     link    ? link.href : 'NOT FOUND',
                spinnerClass: spinner ? spinner.className : 'N/A',
                modalHTML:    modal   ? modal.outerHTML.substring(0, 500) : 'N/A',
            };
        }""")
        print("  ✗ Download link never populated:")
        for k, v in modal_state.items():
            print(f"    {k}: {v}")
        return None

    print(f"  ✓ Download link: {link_href}")
    return link_href


# ── actual file download ───────────────────────────────────────────────────────

async def do_file_download(page, ctx, link_href: str) -> bool:
    step(6, "Click download link and save file")
    try:
        async with page.expect_download(timeout=DL_TIMEOUT) as dl_info:
            await page.evaluate("""() => {
                const link = document.getElementById('simple-download-link');
                if (link) link.click();
            }""")

        dl = await dl_info.value
        fname = dl.suggested_filename or f"{TEST_IMAGE_ID}.zip"
        save_path = TEST_DL_DIR / fname
        await dl.save_as(str(save_path))
        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded: {fname}  ({size_mb:.2f} MB)")
        await ss(page, "12_downloaded")

        if zipfile.is_zipfile(save_path):
            with zipfile.ZipFile(save_path) as zf:
                names = zf.namelist()
                print(f"  ZIP: {len(names)} files")
                for n in names[:10]:
                    print(f"    {n}  ({zf.getinfo(n).file_size // 1024} KB)")
        else:
            with open(save_path, "rb") as fh:
                print(f"  (Not valid ZIP): {fh.read(200)!r}")

        print(f"\n  ✅ SUCCESS — {save_path}")
        return True

    except Exception as exc:
        print(f"  ✗ Playwright download failed: {exc}")

    # Fallback: direct HTTP download with session cookies
    print("  Attempting direct HTTP download fallback...")
    try:
        import requests  # type: ignore[import-untyped]
        cookies_list = await ctx.cookies()
        cookies_dict = {c["name"]: c["value"] for c in cookies_list}
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
            "Referer": page.url,
        }
        r = requests.get(link_href, cookies=cookies_dict, headers=headers, stream=True, timeout=60)
        print(f"  HTTP {r.status_code}  content-type={r.headers.get('content-type')}  content-disposition={r.headers.get('content-disposition')}")
        if r.status_code == 200:
            save_path = TEST_DL_DIR / f"{TEST_IMAGE_ID}_fallback.zip"
            downloaded = 0
            with open(save_path, "wb") as fh:
                for chunk in r.iter_content(65536):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
            size_mb = downloaded / (1024 * 1024)
            print(f"  ✓ Fallback downloaded: {size_mb:.2f} MB → {save_path}")
            return True
        else:
            print(f"  ✗ HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  ✗ Fallback failed: {e}")

    return False


# ── main ───────────────────────────────────────────────────────────────────────

async def run_test():
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        sys.exit("ERROR: pip install playwright && playwright install chromium")

    username, password = load_credentials()
    print(f"\n  Username : {username}")
    print(f"  Password : {'*' * len(password)}")

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DL_DIR.mkdir(parents=True, exist_ok=True)

    collection_name = COLLECTION_NAME  # "mci" — existing collection

    async with async_playwright() as pw:
        step(1, "Launch headless Chromium")
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        ctx = await browser.new_context(
            viewport={"width": 1400, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            accept_downloads=True,
        )
        page = await ctx.new_page()
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        print("  ✓ Browser launched")

        logged_in = await do_login(page, username, password)
        if not logged_in:
            print("\n  ✗ Login failed.")
            await browser.close()
            return

        search_ok = await run_search(page, TEST_IMAGE_ID)
        if not search_ok:
            await dump_html(page, "ERROR_search")
            await browser.close()
            return

        add_ok = await add_results_to_collection(page, collection_name)
        if not add_ok:
            await dump_html(page, "ERROR_add_collection")
            await browser.close()
            return

        link_href = await download_from_collection(page, collection_name)
        if not link_href:
            await browser.close()
            return

        await do_file_download(page, ctx, link_href)

        print(f"\n  ℹ️  Screenshots → {SCREENSHOT_DIR}")
        print(f"  HTML dumps  → {HTML_DIR}")
        print(f"  Downloads   → {TEST_DL_DIR}")

        print("\n" + "=" * 60)
        print("  TEST COMPLETE")
        print("=" * 60)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_test())
