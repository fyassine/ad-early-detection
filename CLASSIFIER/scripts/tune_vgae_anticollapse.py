#!/usr/bin/env python3
"""Optuna sweep for the VGAE anti-collapse hyperparameters.

Run from ``CLASSIFIER/``. This is a tuning tool, not part of the experiment
registry (``run_experiment.py``/``experiments/*.yaml``) — it reuses the same
data pipeline and ``VGAEStaticAdapter`` the STATIC_COMMON_DELCODE notebook uses,
just invoked directly (no papermill/provenance snapshotting) so trials are fast.

Why this exists: real training runs of ``vgae-{gcn,gat}-static-anticollapse``
show ``train_kl`` pinned exactly at the free-bits floor (``free_bits *
latent_dim``) for the entire run, with ``train_recon`` flatlining by epoch ~5
(see CLASSIFIER/model/VGAE/losses.py::kl_divergence — ``torch.clamp(min=floor)``
zeroes the gradient for any dimension already at/below the floor). This script
searches ``free_bits`` / ``beta_warmup_epochs`` / ``learning_rate`` /
``feature_loss_weight`` for a combination where the encoder actually escapes
that floor.

Usage
-----
    python scripts/tune_vgae_anticollapse.py --conv-type gcn --n-trials 30
    python scripts/tune_vgae_anticollapse.py --conv-type gat --n-trials 30 --no-wandb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import torch
from torch_geometric.loader import DataLoader

_CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _CLASSIFIER_ROOT.parent
for _p in (str(_REPO_ROOT), str(_CLASSIFIER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters.static import get_static_adapter  # noqa: E402
from common import tracking  # noqa: E402
from common.seeding import make_rng, make_torch_generator, seed_worker, set_seed  # noqa: E402
from model.GAAE.dataset import GraphDatasetInMemoryFiltered  # noqa: E402
from model.GAAE.utils import knn_binary_adjacency_matrix_no_diag  # noqa: E402

from DATA.src.splitting.load_splits import splits_dir  # noqa: E402

SEED = 100  # matches configs/vgae_*_delcode_whole_brain.json

WB_DATA_ROOT = "/mnt/e/fyassine/ad-early-detection/DATA/DELCODE/__fc_wholebrain_sch200_flat__/matrices"

# How many of the final epochs to average when judging where training settled —
# a single epoch's snapshot is noisy; a tail average is a better readout.
_TAIL_EPOCHS = 10
# Hard non-collapse constraint: a trial whose tail-mean fraction of latent
# dimensions still pinned at the free-bits floor exceeds this is pruned. The
# encoder must actually USE a meaningful share of its latent — an encoder pinned
# at the floor produces near-identical embeddings for every subject, which is
# useless to the downstream classifiers regardless of how low its ELBO is.
_MAX_FLOOR_FRAC = 0.5


def build_loaders(cfg: dict, device: torch.device, torch_gen: torch.Generator):
    splits = str(splits_dir("pretrain"))
    adjacency_args = {"k": cfg.get("adjacency_k", 16)}
    file_variant = cfg.get("file_variant", "z_transformed")
    batch_size = cfg.get("batch_size", 64)

    def _make_dataset(csv_name: str):
        csv_path = f"{splits}/{csv_name}"
        return GraphDatasetInMemoryFiltered(
            root=WB_DATA_ROOT,
            adjacency_function=knn_binary_adjacency_matrix_no_diag,
            adjacency_args=adjacency_args,
            filter_csv_path=csv_path,
            patient_info_path=csv_path,
            separator=",",
            file_variant=file_variant,
        )

    train_dataset = _make_dataset("train.csv")
    val_dataset = _make_dataset("val.csv")
    in_features = train_dataset[0].x.size(1)

    # num_workers=0: loaders are built once and reused across every trial in the
    # study, so per-trial worker spawn overhead would dominate at low trial counts.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        worker_init_fn=seed_worker, generator=torch_gen,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, in_features


def post_warmup_score(history: dict, beta_warmup_epochs: int) -> tuple[float, float]:
    """Return ``(score, frac_dims_at_floor_tail_mean)`` from a finished run's history.

    ``score`` is the quantity Optuna minimizes: the best post-warmup **validation
    adjacency-reconstruction loss** (``val_recon``).

    Deliberately NOT the total ELBO (``val_loss = recon + beta*KL + w*feat``): with
    free-bits active and the posterior at the floor, the KL term is
    ``beta * free_bits * latent_dim``, so minimizing total ELBO mechanically
    prefers the smallest ``free_bits`` regardless of representation quality — a
    degenerate ``argmin(free_bits)`` search. ``val_recon`` measures how well
    ``sigmoid(z zᵀ)`` matches the held-out adjacency and is untouched by the
    free-bits floor (which only scales the KL term), so it ranks trials by the
    quality of the learned graph structure. Non-collapse is enforced separately as
    a hard prune (see ``_MAX_FLOOR_FRAC``), not folded into this score.
    """
    val_recon = history["val_recon"]
    frac_at_floor = history["frac_dims_at_floor"]
    n = len(val_recon)
    start = min(beta_warmup_epochs, max(0, n - 1))
    post_warmup_recon = val_recon[start:] or val_recon
    best_post_warmup_recon = min(post_warmup_recon)
    tail = frac_at_floor[-_TAIL_EPOCHS:] or frac_at_floor
    floor_frac_tail_mean = sum(tail) / len(tail)
    return best_post_warmup_recon, floor_frac_tail_mean


def make_early_exit_callback(beta_warmup_epochs: int, max_floor_frac: float):
    """``on_epoch_end`` hook for ``train_vgae_with_val``: bail out once the
    post-warmup tail-mean ``frac_dims_at_floor`` clears ``max_floor_frac``, instead
    of paying the full epoch budget to discover the same collapse at the end.
    Needs ``_TAIL_EPOCHS`` post-warmup epochs of history before it will fire, to
    avoid killing a trial on one noisy early epoch.
    """
    floor_tail: list[float] = []

    def _on_epoch_end(epoch: int, metrics: dict) -> bool:
        if not metrics["warmed_up"]:
            return False
        floor_tail.append(metrics["frac_dims_at_floor"])
        if len(floor_tail) < _TAIL_EPOCHS:
            return False
        tail_mean = sum(floor_tail[-_TAIL_EPOCHS:]) / _TAIL_EPOCHS
        return tail_mean > max_floor_frac

    return _on_epoch_end


def make_objective(
    conv_type: str, base_cfg: dict, train_loader, val_loader, in_features: int,
    device: torch.device, epochs_cap: int, use_wandb: bool, max_floor_frac: float,
):
    def objective(trial: optuna.Trial) -> float:
        cfg = dict(base_cfg)
        cfg["conv_type"] = conv_type
        cfg["beta"] = trial.suggest_float("beta", 0.001, 0.1, log=True)
        cfg["free_bits"] = trial.suggest_float("free_bits", 0.5, 2.0, log=True)
        cfg["beta_warmup_epochs"] = trial.suggest_int("beta_warmup_epochs", 30, 60)
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 5e-4, 3e-3, log=True)
        cfg["feature_loss_weight"] = trial.suggest_float("feature_loss_weight", 1.0, 5.0)
        cfg["epochs"] = epochs_cap
        cfg["feature_decoder"] = True

        rng = make_rng(SEED + trial.number)
        adapter_cls = get_static_adapter("vgae")
        adapter = adapter_cls(cfg=cfg, device=device, rng=rng)
        model, _model_config = adapter.build_model(in_features)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.001)
        )

        wandb_run = None
        if use_wandb:
            # exp["id"] becomes the W&B group and exp["model"] the job_type (see
            # common/tracking.py::_build_init_kwargs), so every trial in this study
            # lands in group="tune-vgae-{conv_type}", job_type="tune" — distinct from
            # the real vgae-{gcn,gat}-static training runs.
            exp_meta = {
                "id": f"tune-vgae-{conv_type}", "mode": "static", "model": "tune",
                "dataset": "DELCODE_WHOLE_BRAIN", "seed": SEED, "wandb": True,
            }
            wandb_run = tracking.init_run(exp_meta, {**cfg, "trial_number": trial.number})

        on_epoch_end = make_early_exit_callback(cfg["beta_warmup_epochs"], max_floor_frac)
        try:
            _best_state, history = adapter.run_training(
                model, optimizer, train_loader, val_loader, wandb_run, on_epoch_end=on_epoch_end,
            )
        finally:
            if wandb_run is not None:
                tracking.finish_run(wandb_run)

        score, floor_frac = post_warmup_score(history, cfg["beta_warmup_epochs"])
        raw_kl_mean_tail = sum(history["raw_kl_mean"][-_TAIL_EPOCHS:]) / len(history["raw_kl_mean"][-_TAIL_EPOCHS:])
        trial.set_user_attr("frac_dims_at_floor_tail_mean", floor_frac)
        trial.set_user_attr("raw_kl_mean_tail", raw_kl_mean_tail)
        trial.set_user_attr("val_recon_best", score)
        trial.set_user_attr("epochs_run", len(history["val_loss"]))

        # Hard non-collapse constraint: a still-collapsed encoder is useless to the
        # downstream classifiers no matter how low its reconstruction loss reads, so
        # prune it out of the search rather than ranking it. The early-exit callback
        # above already stops most collapsing trials well before this point — this
        # is the backstop for trials that never clear the post-warmup tail window
        # (e.g. beta_warmup_epochs close to epochs_cap).
        if floor_frac > max_floor_frac:
            raise optuna.TrialPruned(
                f"{floor_frac:.2f} of latent dims still pinned at the free-bits floor "
                f"(> {max_floor_frac}); encoder collapsed."
            )
        return score

    return objective


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conv-type", choices=["gcn", "gat"], required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs-cap", type=int, default=150)
    parser.add_argument(
        "--max-floor-frac", type=float, default=_MAX_FLOOR_FRAC,
        help="Prune trials whose tail-mean fraction of latent dims pinned at the "
             "free-bits floor exceeds this (default %(default)s). Tighten (e.g. "
             "0.3) if the default lets through trials that still read collapsed.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)

    set_seed(SEED)
    torch_gen = make_torch_generator(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_cfg_path = (
        _CLASSIFIER_ROOT / "configs" / f"vgae_{args.conv_type}_anticollapse_delcode_whole_brain.json"
    )
    base_cfg = json.loads(base_cfg_path.read_text())

    train_loader, val_loader, in_features = build_loaders(base_cfg, device, torch_gen)
    print(f"conv_type={args.conv_type}  train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  in_features={in_features}  device={device}")

    objective = make_objective(
        args.conv_type, base_cfg, train_loader, val_loader, in_features,
        device, args.epochs_cap, use_wandb=not args.no_wandb, max_floor_frac=args.max_floor_frac,
    )
    study = optuna.create_study(direction="minimize", study_name=f"tune-vgae-{args.conv_type}")
    study.optimize(objective, n_trials=args.n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n{len(completed)} completed / {len(pruned)} pruned (collapsed) of {len(study.trials)} trials.")

    best = None
    if completed:
        best = study.best_trial
        print(f"Best trial (#{best.number}): val_recon={best.value:.4f}")
        print(f"  params: {best.params}")
        print(f"  frac_dims_at_floor_tail_mean: {best.user_attrs.get('frac_dims_at_floor_tail_mean')}")
        print(f"  raw_kl_mean_tail: {best.user_attrs.get('raw_kl_mean_tail')}")
        print(f"  epochs_run: {best.user_attrs.get('epochs_run')}")
    else:
        # Every trial tripped the non-collapse prune — that's a real finding: no
        # point in this search space escaped the free-bits floor, so free-bits is
        # not the right remedy here (revisit the clamp / KL-annealing instead).
        print(f"NO trial kept < {args.max_floor_frac} of dims off the free-bits floor — "
              "free-bits did not escape collapse anywhere in this search space.")

    out_path = _CLASSIFIER_ROOT / "scripts" / f"tune_vgae_{args.conv_type}_best.json"
    out_path.write_text(json.dumps({
        "conv_type": args.conv_type,
        "max_floor_frac": args.max_floor_frac,
        "epochs_cap": args.epochs_cap,
        "best_params": best.params if best else None,
        "best_val_recon": best.value if best else None,
        "best_user_attrs": dict(best.user_attrs) if best else None,
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "all_trials": [
            {"number": t.number, "state": str(t.state), "params": t.params,
             "value": t.value, "user_attrs": t.user_attrs}
            for t in study.trials
        ],
    }, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
