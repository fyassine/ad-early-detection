# Rules

- Whenever you execute terminal commands using `run_command` in this workspace, always run `source .venv/bin/activate` first (e.g., by prefixing the command with `source .venv/bin/activate && ...`) to ensure execution is within the project's virtual environment (`tectonic` is available once `.venv` is activated for compiling `THESIS/`).

## Code search scope

Search and index ALL files in the repository **including `DATA/`** and every other folder.
The `.gitignore` excludes large binary/data files from *git tracking* only — it does **not** define
what is relevant code. When searching for files, scripts, or symbols, always include `DATA/`.

**Never read** the following (performance / privacy):

- `.venv/` — Python virtual environment
- `.git/` — git internals
- `.env` — secrets file
- `**/wandb/` — ML tracker artifacts
- `**/checkpoints/` — model checkpoint blobs
- `**/__pycache__/` — bytecode cache
- `**/*.nii.gz`, `**/*.npz`, `**/*.pkl` — large binary arrays
- `**/*.csv`, `**/*.xlsx`, `**/*.xls` — raw data tables
- `DATA/src/processing/subcortex/` — vendored third-party toolbox

Everything else (`.py`, `.ipynb`, `.json`, `.sh`, `.yaml`, `.md`, `.toml`, `.split`, etc.)
is project code and **must** be included in all searches and file discovery.

## Credentials and Thesis Synchronization

- Never commit secrets, API keys, tokens (e.g. Overleaf / ShareLaTeX `olp_*` PAT), or `.env*` files to Git.
- Credentials must be stored in `~/.netrc` (mode 0600) or `.env` (gitignored).
- Refer to `.claude/rules/credentials.md` and `.agents/rules/thesis.md` for full credential contracts and ShareLaTeX sync commands (`git --git-dir=.git-thesis --work-tree=THESIS ...`).

## Agent-Specific Guidelines

- **Gemini / Antigravity CLI**: Refer to `.agents/rules/gemini.md` for the SSH remote execution environment context and mandatory planning artifact storage/path reporting protocol.
