from __future__ import annotations

import shutil
from pathlib import Path


def update_latest_checkpoint(
    checkpoint_root: str | Path,
    model_name: str,
    model_file: str | Path,
) -> Path:
    """
    Point checkpoint_root/latest/<model_name>.pth at model_file.

    Called right after a training run saves its timestamped checkpoint, so
    downstream experiments.yaml entries can reference a stable path instead
    of picking a run directory by hand. Falls back to a copy if the
    filesystem doesn't support symlinks.
    """
    latest_dir = Path(checkpoint_root) / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_path = latest_dir / f"{model_name}.pth"
    if latest_path.is_symlink() or latest_path.exists():
        latest_path.unlink()
    target = Path(model_file).resolve()
    try:
        latest_path.symlink_to(target)
    except OSError:
        shutil.copy2(target, latest_path)
    return latest_path


def select_gaae_checkpoint(
    search_dirs: list[str | Path],
    *,
    checkpoint_path: str | Path | None = None,
) -> tuple[str, Path, Path]:
    """
    List GAAE checkpoints under search_dirs and return (run_name, ckpt_path, run_dir).

    Interactive by default (prompts for an index). For non-interactive / headless
    execution (papermill, run_experiment.py), pass ``checkpoint_path`` to bypass the
    prompt — the matching candidate is resolved by path and returned without any
    ``input()`` call. Passing a checkpoint that is not among the discovered
    candidates raises ``FileNotFoundError`` (fail loudly rather than silently using
    the wrong encoder).

    Raises FileNotFoundError if no checkpoints exist in any search dir.
    """
    candidates: list[tuple[str, Path, Path]] = sorted(
        [
            (run_dir.name, run_dir / f"model_{run_dir.name}.pth", run_dir)
            for ckpt_dir in search_dirs
            for base_path in [Path(ckpt_dir)]
            if base_path.is_dir()
            for run_dir in sorted(base_path.iterdir())
            if run_dir.is_dir()
            if (run_dir / f"model_{run_dir.name}.pth").exists()
        ],
        key=lambda x: x[0],
    )
    if not candidates:
        raise FileNotFoundError(
            f"No GAAE checkpoints found in: {[str(d) for d in search_dirs]}"
        )

    if checkpoint_path is not None:
        target = Path(checkpoint_path).resolve()
        for run_name, ckpt_path, run_dir in candidates:
            if ckpt_path.resolve() == target:
                print(f"Selected (non-interactive): {run_name}")
                return run_name, ckpt_path, run_dir
        available = [str(c[1]) for c in candidates]
        raise FileNotFoundError(
            f"checkpoint_path={checkpoint_path!r} not among discovered GAAE "
            f"checkpoints: {available}"
        )

    print("Available GAAE checkpoints:")
    for i, (name, _, rdir) in enumerate(candidates):
        print(f"  {i}: {name}  ({rdir})")
    idx = int(input("Select checkpoint index: "))
    run_name, ckpt_path, run_dir = candidates[idx]
    print(f"\nSelected: {run_name}")
    return run_name, ckpt_path, run_dir
