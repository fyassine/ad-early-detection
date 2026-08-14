# Session-scoped Claude Code login on a shared account

## Why

`wunderlich` (and other accounts on this box) can be logged into by more than
one real person over SSH. Claude Code stores its OAuth session in a plain file,
`~/.claude/.credentials.json`. File permissions (`600`) only stop *other* Linux
users — anyone who connects as the same account is the same OS principal and
already has full read/write access to that file, tmux or not. Running `claude`
once left the authenticated session usable by anyone else logging into the same
account, even outside the terminal that started it.

The fix: keep the credentials encrypted at rest, and only decrypt them for the
lifetime of the shell that explicitly unlocked them. Logging out (closing the
shell, or detaching a tmux session) wipes the decrypted copy automatically, so
nothing lingers for the next person on the same account.

**Limitation that this does not solve:** everyone sharing this OS account
shares one Linux UID, so there's no way to stop someone from reading the
decrypted file *during* the window it's unlocked — same UID means same access.
The only setup with real isolation is a separate OS account per person. This
scheme shrinks the exposure window from "always" to "only while explicitly
logged in," which is the best available mitigation short of that.

## One-time setup

Do this once, in your own interactive terminal (not by pasting into anything
non-interactive) — the passphrase you type here goes straight into `gpg`'s
prompt and is never visible to anyone else, including any assistant helping you
set this up.

1. Create a private directory for the encrypted credential store:

   ```bash
   mkdir -p -m 700 ~/.claude-secure
   ```

2. Encrypt your current Claude Code credentials with a passphrase only you
   know:

   ```bash
   gpg --symmetric --cipher-algo AES256 -o ~/.claude-secure/credentials.json.gpg ~/.claude/.credentials.json
   ```

   You'll be prompted for the passphrase twice (enter + confirm). Pick
   something you can remember — this is what `claude-login` will ask for from
   now on. If this fails with "No such file or directory", the Claude Code
   background daemon may have been mid-way through its periodic token refresh
   (it atomically rewrites `.credentials.json` roughly every ~8h) — just retry
   the command.

3. Wipe the now-redundant plaintext copy:

   ```bash
   shred -u ~/.claude/.credentials.json
   ```

4. Reload your shell so the new `claude-login` / `claude-logout` functions
   (added to `~/.bashrc`) are available:

   ```bash
   source ~/.bashrc
   ```

## Day-to-day use

- **On connecting**, before using `claude`, run:

  ```bash
  claude-login
  ```

  This prompts for your gpg passphrase, decrypts `.credentials.json` into
  place, and arms cleanup for the current shell so the plaintext file is
  automatically wiped when you log out.

- **Use `claude` normally** — no repeated OAuth prompts, since the decrypted
  session is already authenticated.

- **On disconnecting a plain SSH shell** (`exit`, or the connection just
  drops), the credentials are wiped automatically — no manual step needed.

- **Inside tmux**, detaching (`Ctrl-b d`) also triggers the wipe, since a
  detached tmux session would otherwise keep the plaintext file sitting around
  for anyone else on the account to pick up (this was the exact gap that
  caused the original problem).

- **Stepping away without disconnecting**: run `claude-logout` to wipe the
  credentials immediately on demand, without closing the shell.

- **Safety net**: even if a shell exits abnormally without running the cleanup
  trap, a background watchdog wipes the plaintext automatically after 8 hours
  (`CLAUDE_LOGIN_TTL`, in seconds — export a different value before calling
  `claude-login` to change it).

## Quick reference

| Command | What it does |
|---|---|
| `claude-login` | Decrypt credentials for this shell; arms auto-wipe on shell exit / tmux detach / TTL |
| `claude-logout` | Re-encrypt the snapshot (prompts for passphrase), then immediately wipe the decrypted credentials |
| `exit` / disconnect | Auto-wipes (via the EXIT trap set by `claude-login`) — no re-encrypt |
| `Ctrl-b d` (tmux detach) | Auto-wipes (via the tmux `client-detached` hook) — no re-encrypt |
| (idle for `CLAUDE_LOGIN_TTL` seconds, default 8h) | Auto-wipes as a fallback — no re-encrypt |

## Known caveat: token refresh vs. the encrypted snapshot

Claude Code's background daemon proactively refreshes the OAuth token in
`.credentials.json` roughly every 8 hours while you're logged in. If a refresh
happens mid-session and the plaintext is wiped without being re-encrypted, the
next `claude-login` restores the **older**, pre-refresh token from
`~/.claude-secure/credentials.json.gpg` — which may or may not still be valid
depending on whether the refresh rotated/invalidated it.

`claude-logout` now handles this: it re-encrypts the current `.credentials.json`
over the snapshot (prompting for your passphrase) *before* shredding the
plaintext, so any mid-session token refresh is captured. If the re-encryption
fails or you abort the passphrase prompt, it warns and keeps the previous
snapshot — but still wipes the plaintext, since leaving it behind is the exact
problem this whole scheme exists to prevent.

**The automatic wipes cannot do this.** The EXIT trap, the tmux `client-detached`
hook, and the TTL watchdog all run without a terminal to prompt from, so they
shred the plaintext without re-encrypting. Only an explicit `claude-logout`
refreshes the snapshot.

So: if `claude` unexpectedly asks for a fresh OAuth login after a long session,
the likely cause is that the session ended via one of the automatic wipes rather
than an explicit `claude-logout`. To recover, re-run the one-time encryption step
(step 2 above) against a current `.credentials.json`. To avoid it, prefer running
`claude-logout` by hand when you're ending a long session.

## Verification checklist

- [ ] `~/.claude/.credentials.json` does not exist between sessions
      (`ls -la ~/.claude/.credentials.json` → no such file)
- [ ] `claude-login` with the correct passphrase restores it and `claude`
      works without a fresh OAuth prompt
- [ ] `claude-login` with the wrong passphrase fails cleanly, leaving no
      partial/corrupt file behind
- [ ] `claude-logout` prompts for the passphrase, reports "encrypted snapshot
      refreshed", and removes the plaintext immediately
- [ ] `claude-logout` with an aborted/wrong passphrase warns, leaves the previous
      `~/.claude-secure/credentials.json.gpg` intact and uncorrupted, and still
      wipes the plaintext
- [ ] Exiting a plain SSH shell after `claude-login` wipes the file
      automatically (verify by reconnecting)
- [ ] Detaching a tmux session (`Ctrl-b d`) after `claude-login` wipes the
      file, without exiting the tmux session itself
- [ ] Shortening `CLAUDE_LOGIN_TTL` (e.g. `CLAUDE_LOGIN_TTL=30 claude-login`)
      confirms the background watchdog wipes the file on its own
