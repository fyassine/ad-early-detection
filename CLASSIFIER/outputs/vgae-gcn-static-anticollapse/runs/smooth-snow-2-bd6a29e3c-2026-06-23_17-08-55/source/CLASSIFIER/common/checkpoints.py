from __future__ import annotations

from pathlib import Path


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

def update_latest_checkpoint(
    ckpt_root: str | Path,
    latest_tag: str,
    target_path: str | Path,
) -> Path:
    """
    Update the latest checkpoint pointer to target_path.
    Creates or updates a symlink at {ckpt_root}/{latest_tag}.pth pointing to target_path.
    """
    ckpt_root = Path(ckpt_root)
    target_path = Path(target_path)
    latest_link = ckpt_root / f"{latest_tag}.pth"

    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink(missing_ok=True)

    latest_link.symlink_to(target_path.resolve())
    return latest_link
