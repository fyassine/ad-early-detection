from __future__ import annotations

from pathlib import Path


def select_gaae_checkpoint(
    search_dirs: list[str | Path],
) -> tuple[str, Path, Path]:
    """
    List GAAE checkpoints under search_dirs and prompt for selection.
    Returns (run_name, ckpt_path, run_dir).
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
    print("Available GAAE checkpoints:")
    for i, (name, _, rdir) in enumerate(candidates):
        print(f"  {i}: {name}  ({rdir})")
    idx = int(input("Select checkpoint index: "))
    run_name, ckpt_path, run_dir = candidates[idx]
    print(f"\nSelected: {run_name}")
    return run_name, ckpt_path, run_dir
