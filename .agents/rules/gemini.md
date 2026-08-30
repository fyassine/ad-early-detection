# Gemini Agent Rules

## 1. Remote SSH Execution Environment

- **Remote Host Context**: You operate directly in a terminal inside a remote Linux server environment over SSH, **not** locally on the user's laptop.
- All filesystem paths (`/mnt/e/fyassine/ad-early-detection`), GPU resources, network endpoints, and execution environments are on the remote server host.
- Always activate the virtual environment (`source .venv/bin/activate`) before running any terminal commands.
- Never assume client-side/laptop local filesystem access, GUI displays, or local laptop tooling unless accessed through remote server capabilities.

## 2. Planning Protocol & Artifact Storage

- **Store Every Plan**: Whenever formulating, revising, or proposing an implementation plan (e.g., in planning mode or before executing multi-step tasks), you **must** persist the plan as a markdown file/artifact on disk (`<appDataDir>/brain/<conversation-id>/<plan_name>.md`).
- **Standardized Path Presentation**: Every planning response must explicitly display the stored plan's location using the following standard format block:
  ```markdown
  ### Implementation Plan
  - **Absolute File Path**: `/path/to/<plan_name>.md`
  - **Clickable Link**: [<plan_name>.md](file:///path/to/<plan_name>.md)
  ```
- **Visible Raw Path**: Always output the full raw absolute path in plain code format so it is directly readable and copy-pasteable in terminal environments where markdown link targets may not be visible.
- **Never Ephemeral**: Never output plans only as ephemeral chat responses without storing them to disk.
