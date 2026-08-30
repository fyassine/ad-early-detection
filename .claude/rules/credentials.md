# Credentials

All secrets live in gitignored `.env` files. Never read `.env` file contents (see
`CLAUDE.md` code-search exclusions) — the variable names and their purpose are
documented here so that doesn't need rediscovering each session.

## Root `.env` (`ad-early-detection/.env`)

| Variable | Host | Purpose |
|----------|------|---------|
| `CORE_USER` / `CORE_PASSWORD` / `CORE_HOST` | `core-mgm.med.uni-muenchen.de` | SSH/rsync to CORE HPC (fMRIPrep + postprocessing array jobs). Key-based auth is the default for `DATA/PREPROCESSING/src/fritz/*.sh` — `CORE_PASSWORD` is ignored unless a script is passed `--use-password` (then `sshpass` is used). |
| `LRZ_USER` / `LRZ_PASSWORD` / `LRZ_HOST` | `cool.hpc.lrz.de` | SSH/rsync/scp to LRZ HPC. Read by `DATA/DELCODE/src/transferring/*.py` (some scripts hardcode `di54lup@cool.hpc.lrz.de` as the default instead of reading the env var — check the script). |
| `WANDB_API_KEY` / `WANDB_ENTITY` / `WANDB_PROJECT` | wandb.ai (external) | Experiment tracking, routed through `common/tracking.py` / `SHARED/tracking.py`. |
| `NITRC_USERNAME` / `NITRC_PASSWORD` | nitrc.org (external) | NITRC data access. |
| `IDA_ADNI_USERNAME` / `IDA_ADNI_PASSWORD` | ida.loni.usc.edu (external) | LONI/IDA login for ADNI downloads (root-level pair). |
| `SHARELATEX_GIT_TOKEN` | `sharelatex.tum.de` (external) | Personal Access Token (`olp_*`) for TUM ShareLaTeX / Overleaf Git sync via `.git-thesis`. Automatically read by Git via `~/.netrc` (`chmod 600`) or root `.env`. Never commit this token, print it in logs, or track it in any repo. |

## `DATA/ADNI/src/.env`

| Variable | Host | Purpose |
|----------|------|---------|
| `ADNI_USERNAME` / `ADNI_PASSWORD` | ida.loni.usc.edu (external) | LONI/IDA login used directly by the ADNI downloader scripts (`download/`, `inspect/`, `ida_downloader/`). Some of those scripts also fall back to `LONI_USERNAME`/`LONI_PASSWORD` or `IDA_USERNAME`/`IDA_PASSWORD` if set — same site, alternate var names, not the root `IDA_ADNI_*` pair. |

## ShareLaTeX Git Synchronization

The thesis is synchronized with TUM ShareLaTeX using the isolated `.git-thesis` directory:
- Push: `git --git-dir=.git-thesis --work-tree=THESIS push origin master`
- Pull: `git --git-dir=.git-thesis --work-tree=THESIS pull --rebase origin master`
- Authentication is handled seamlessly via `~/.netrc` (mode 0600) configured with the `olp_*` PAT.
- Never write the PAT into tracked files, commit messages, or PR descriptions.

## Adding a new credential

Add the variable name, host, and purpose to the table above — do not print or commit
the actual value anywhere, including in commit messages or this file.
