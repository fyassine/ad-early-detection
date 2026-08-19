# Session-scoped Antigravity Login on a Shared Account

## Why

`wunderlich` (and other accounts on this server) can be logged into by more than
one real person over SSH. Google Antigravity (`agy` and Antigravity IDE) stores its OAuth session
in a plain file, `~/.gemini/antigravity-cli/antigravity-oauth-token`. File permissions (`600`) only
stop *other* Linux users (different UIDs) — anyone who connects as the same account is the same OS
principal and already has full read/write access to that file, tmux or not. Authenticating once
left the session usable by anyone else logging into the same account, even outside the terminal
that started it.

The fix: keep credentials encrypted at rest, and only decrypt them for the
lifetime of the session/shell that explicitly unlocked them. In addition, an autonomous
background supervisor (`antigravity-session-guard`) ensures that when the last SSH connection
disconnects, any lingering background servers/processes and plaintext tokens are cleaned up.

**Limitation that this does not solve:** everyone sharing this OS account
shares one Linux UID, so there is no way to stop someone from reading the
decrypted file *during* the window it is unlocked — same UID means same access.
The only setup with complete isolation is a separate OS account per person. This
scheme shrinks the exposure window from "always" to "only while explicitly
logged in," which is the best available mitigation short of individual accounts.

## One-time Setup

Do this once, in your own interactive terminal (not by pasting into anything
non-interactive) — the passphrase you type goes straight into `gpg`'s
prompt and is never visible to anyone else.

1. Create a private directory for the encrypted credential store:

   ```bash
   mkdir -p -m 700 ~/.antigravity-secure
   ```

2. Encrypt your current Antigravity credentials with a passphrase only you know:

   ```bash
   gpg --symmetric --cipher-algo AES256 -o ~/.antigravity-secure/antigravity-oauth-token.gpg ~/.gemini/antigravity-cli/antigravity-oauth-token
   ```

   You will be prompted for the passphrase twice (enter + confirm). Pick
   something you can remember — this is what `agy-login` will ask for from
   now on.

3. Wipe the now-redundant plaintext copy:

   ```bash
   shred -u ~/.gemini/antigravity-cli/antigravity-oauth-token
   ```

4. Reload your shell so the new `agy-login` / `agy-logout` functions
   (added to `~/.bashrc`) are available:

   ```bash
   source ~/.bashrc
   ```

## Day-to-day Use

- **On connecting**, before using `agy`, run:

  ```bash
  agy-login
  ```

  This prompts for your GPG passphrase, decrypts `antigravity-oauth-token` into
  place, and arms cleanup for the current shell so the plaintext file is
  automatically wiped when you log out.

- **Use `agy` normally** — no repeated OAuth browser prompts, since the decrypted
  session is already authenticated.

- **On disconnecting a plain SSH shell** (`exit`, or connection drops), the credentials
  are wiped automatically via shell `EXIT` traps.

- **Inside tmux**, detaching (`Ctrl-b d`) also triggers the wipe via tmux `client-detached`
  hooks, preventing detached background tmux sessions from leaving plaintext credentials around.

- **Stepping away without disconnecting**: run `agy-logout` to re-encrypt and wipe credentials immediately on demand.

- **Autonomous Session Guard (`antigravity-session-guard`)**:
  A detached poller starts on SSH login. If all SSH connections (both interactive ptys and `@notty` remote IDE connections) drop for ~60s, it automatically:
  1. Securely shreds `~/.gemini/antigravity-cli/antigravity-oauth-token`.
  2. Terminates lingering background Antigravity processes (IDE servers, extension hosts, language servers) matching `/proc/<pid>/exe`.

- **Safety net**: Even if an abnormal termination skips traps, a background watchdog wipes
  the plaintext automatically after 8 hours (`AGY_LOGIN_TTL`, in seconds — export a different
  value before calling `agy-login` to change it).

## Quick Reference

| Command / Trigger | What it does |
|---|---|
| `agy-login` | Decrypt credentials for this shell; arms auto-wipe on shell exit / tmux detach / TTL |
| `agy-logout` | Re-encrypt snapshot (prompts for passphrase), then immediately wipe decrypted credentials |
| `agy-guard-status` | Check live status of `antigravity-session-guard` and dry-run scan of token-holding processes |
| `agy-guard-off` | Disarm guard for the current session (e.g. if running an intentional background job) |
| `exit` / SSH disconnect | Auto-wipes credentials via `EXIT` trap; after ~60s guard terminates background processes |
| `Ctrl-b d` (tmux detach) | Auto-wipes credentials via tmux `client-detached` hook |
| Idle for `AGY_LOGIN_TTL` (8h default) | Auto-wipes as a fallback watchdog |

## Known Caveats

### Token Refresh vs. Encrypted Snapshot
Antigravity automatically refreshes OAuth access tokens. If a token refresh occurs
during a long session and the shell exits without re-encrypting, the encrypted snapshot
in `~/.antigravity-secure/antigravity-oauth-token.gpg` will retain the older token.

`agy-logout` handles this by prompting to re-encrypt the live token *before* wiping.
If you notice `agy` prompting for fresh browser OAuth after a long hiatus, re-run the one-time encryption
step with your fresh token.

### Antigravity IDE Remote Connections
VS Code / Antigravity IDE Remote connections maintain an `sshd: wunderlich@notty` session.
The session guard detects this and stays idle as long as your IDE window is connected.
To trigger full logout and teardown, close the remote IDE window as well.

## Verification Checklist

- [ ] `~/.gemini/antigravity-cli/antigravity-oauth-token` does not exist between sessions (`ls` returns no such file).
- [ ] `agy-login` with correct passphrase restores the token and `agy` runs without browser OAuth prompt.
- [ ] `agy-login` with wrong passphrase fails cleanly without writing corrupt files.
- [ ] `agy-logout` re-encrypts the snapshot and removes plaintext immediately.
- [ ] `agy-guard-status` displays armed guard status and reports process targets.
- [ ] Disconnecting SSH drops credentials and terminates background Antigravity IDE servers after grace period.
