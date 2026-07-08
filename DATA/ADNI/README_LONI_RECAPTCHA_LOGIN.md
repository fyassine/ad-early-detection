# LONI reCAPTCHA login — how to authenticate the headless downloaders

LONI IDA added a **reCAPTCHA** to `login.jsp`. A scripted browser cannot solve
it (reCAPTCHA denies the Storage Access API to automated contexts), so a human
has to solve the challenge once in a real browser. The download/resolve scripts
(`download_adni_smri.py`, `download_collection*.py`) then reuse the resulting
session from the persistent Chromium profile at `DATA/ADNI/.loni_profile/`.

The normal path is: run any downloader once with `--headless false` on NeuroLab
over `ssh -Y` (XQuartz), solve the captcha in the forwarded window, done.

**This document is the fallback for when that forwarded window can't open** —
which is the case on this box.

---

## Why the forwarded-window path fails here

`NeuroLab` (`FRiTZ`) is a **WSL2** instance (`/mnt/e/...`). X11 forwarding into
WSL2 over SSH is broken: after `ssh -Y NeuroLab`, `echo $DISPLAY` is **empty** —
forwarding was never established. Manually running `export DISPLAY=localhost:10.0`
does **not** help; it points Chromium at a tunnel that doesn't exist, so a headed
launch dies with:

```
ERROR ...ozone_platform_x11.cc: Missing X server or $DISPLAY
The platform failed to initialize. Exiting.
playwright ... TargetClosedError: ... browser has been closed
```

Fixing forwarding (enable `X11Forwarding`, install `xauth` in the WSL sshd) needs
**root**, which we don't have here (`apt` says "ask your administrator").
`xvfb-run` doesn't help either — Xvfb is invisible, and you need to *see* the
captcha.

## The workaround: solve on the Mac, transplant the session

1. Run the manual login **natively on your Mac** (real display, no SSH/X11) — it
   opens a normal Chromium window; you solve the captcha there.
2. It dumps the session cookies as **plaintext JSON** (`loni_state.json`).
3. `scp` that JSON to NeuroLab.
4. A seed script re-injects the cookies into NeuroLab's own
   `.loni_profile/`, where they get re-encrypted with the Linux key.

### Two things that make this non-obvious

- **A raw copy of `.loni_profile/` from Mac → Linux does NOT work.** Chromium
  encrypts cookies at rest with an **OS-specific key** (macOS mock-keychain vs
  Linux basic store), so a Mac-written cookie DB fails to decrypt on Linux. That
  is why we transfer plaintext `storage_state` JSON and re-seed, instead of
  copying the folder.
- **The LONI session cookies (`JSESSIONID`, `IDA_USC`) are *session* cookies.**
  Chromium keeps session cookies in memory only and **purges them from a
  persistent profile on the next launch**. So a naive seed authenticates inside
  the seeding process but is gone by the first real run. The seed script fixes
  this by giving those cookies a concrete future expiry so they are written to
  disk and survive across processes. (This only extends the *client-side*
  lifetime; the LONI server session still expires on its own schedule.)

---

## Scripts involved

| Script | Runs on | Purpose |
|---|---|---|
| `DATA/ADNI/src/download/loni_login_manual.py` | **Mac** | Headed login; you solve the captcha; dumps `DATA/ADNI/loni_state.json` (plaintext cookies via Playwright `storage_state`). |
| `DATA/ADNI/src/download/loni_seed_from_state.py` | **NeuroLab** | Reads `loni_state.json`, promotes session cookies to persistent, injects them into `.loni_profile/` (re-encrypted with the Linux key), verifies login. |
| `DATA/ADNI/src/download/loni_session.py` | both | Shared `open_context()` / `is_logged_in()` helpers. |

> Note: `loni_login_manual.py` on the Mac carries a small addition over the
> NeuroLab copy — after a successful login it writes `loni_state.json`. If you
> re-copy it from NeuroLab, re-add that `storage_state(path=...)` dump.

---

## Procedure

### One-time Mac setup

```bash
mkdir -p ~/loni-login-mac/DATA/ADNI/src/download
cd ~/loni-login-mac
# pull the scripts + credentials from NeuroLab
scp NeuroLab:/mnt/e/fyassine/ad-early-detection/DATA/ADNI/src/download/loni_login_manual.py DATA/ADNI/src/download/
scp NeuroLab:/mnt/e/fyassine/ad-early-detection/DATA/ADNI/src/download/loni_session.py     DATA/ADNI/src/download/
scp NeuroLab:/mnt/e/fyassine/ad-early-detection/DATA/ADNI/src/.env                          DATA/ADNI/src/.env
# python env + browser
python3 -m venv .venv
./.venv/bin/pip install -q playwright python-dotenv
./.venv/bin/playwright install chromium        # installs full Chromium (headed needs it, not just headless-shell)
```

Make sure `loni_login_manual.py` on the Mac dumps `storage_state` on success
(the block that writes `DATA/ADNI/loni_state.json`). Add it if missing:

```python
# after "if await is_logged_in(page):" confirms login, before context.close():
state_path = ADNI_SRC_DIR.parent / "loni_state.json"
await context.storage_state(path=str(state_path))
```

### Step 1 — Solve the login on the Mac (normal Terminal, NOT ssh)

```bash
cd ~/loni-login-mac
./.venv/bin/python DATA/ADNI/src/download/loni_login_manual.py
```

A real Chromium window opens (email/password pre-filled from `.env`).
**Solve the reCAPTCHA → click Log In → wait until you're logged in →** return to
the terminal and press **Enter**. You should see:

```
✓ Logged in: https://ida.loni.usc.edu/home/projectPage.jsp?project=ADNI
✓ storage_state (plaintext cookies) written to .../DATA/ADNI/loni_state.json
```

(The `$DISPLAY is empty` warning is written for the Linux path — ignore it on macOS.)

### Step 2 — Ship the cookies to NeuroLab

```bash
scp ~/loni-login-mac/DATA/ADNI/loni_state.json \
    NeuroLab:/mnt/e/fyassine/ad-early-detection/DATA/ADNI/
```

### Step 3 — Seed NeuroLab's profile (on NeuroLab)

```bash
cd /mnt/e/fyassine/ad-early-detection
rm -rf DATA/ADNI/.loni_profile          # optional: clean seed, avoids stale anon cookies
.venv/bin/python DATA/ADNI/src/download/loni_seed_from_state.py
```

Expected:

```
Loaded 6 cookies from .../DATA/ADNI/loni_state.json
Promoted 2 session cookie(s) to persistent (7-day client expiry)
✓ Logged in: ...
✓ Cookies re-encrypted into DATA/ADNI/.loni_profile/ — headless resolve/download can now take over.
```

### Step 4 — Verify with a headless pilot (on NeuroLab)

```bash
cd /mnt/e/fyassine/ad-early-detection
.venv/bin/python DATA/ADNI/src/download/download_adni_smri.py --pilot-one --headless true
```

Success is confirmed by:

```
✓ Reusing existing LONI session (persistent profile)
```

Once that line appears, every later headless run (the full resolve, the
collection downloaders) authenticates straight from `.loni_profile/`.

---

## When the session expires

The LONI server session eventually times out (independent of the 7-day client
cookie expiry). When headless runs start reporting login failures again, just
**re-run Steps 1–3** (solve the captcha once more on the Mac, re-ship, re-seed).
No code changes needed.

## Troubleshooting

- **`profile in use` / SingletonLock error** — a crashed launch left a lock:
  `rm DATA/ADNI/.loni_profile/Singleton*` then retry.
- **Seed says "Cookies did not authenticate"** — the server session already
  expired between Mac login and seeding; redo Step 1 promptly and seed right away.
- **Pilot logs "Login failed" right after a seed** — the session cookies didn't
  persist. Confirm the seed printed `Promoted 2 session cookie(s) to persistent`;
  if it printed 0, `loni_state.json` was captured without the session cookies
  (you weren't actually logged in when you pressed Enter on the Mac).
