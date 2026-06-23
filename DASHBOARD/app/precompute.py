"""
precompute.py — Detached precompute worker.

Runs as a standalone subprocess launched by services/job_manager.py.
Invoke directly for one-off precompute runs:

    cd DASHBOARD/
    .venv/bin/python -m app.precompute \\
        --data-root /mnt/e/fyassine/ad-early-detection/DATA \\
        --csv-path  DELCODE/__fc_dmn_sch200_flat__/metadata/cohorts.csv \\
        --scan-folders DELCODE/__fc_dmn_sch200_flat__/matrices \\
        --job-id manual \\
        --cache-root /mnt/e/fyassine/ad-early-detection/DASHBOARD/.cache

Stages
------
1. CohortStats  (UMAP + EBM + brain-age + time-shift)         ~60 s
2. Graph metrics per cohort (networkx small-worldness, etc.)  ~20 s
3. GELSTM predictions for all subjects (when available)       ~30-90 s
4. QC mean volumes for converter subjects                     varies

Each stage updates the job status JSON and prints timestamped progress.
The process is designed to survive server restarts (uses start_new_session)
but will be killed if the Docker container is stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────── #
# Bootstrap                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _update_status(status_path: Path, **kwargs) -> None:
    try:
        existing: dict = {}
        if status_path.exists():
            existing = json.loads(status_path.read_text())
        existing.update(kwargs)
        # Write atomically
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, default=str, indent=2))
        tmp.replace(status_path)
    except Exception as e:
        _log(f"WARNING: status update failed: {e}")


# ──────────────────────────────────────────────────────────────────────────── #
# Stages                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

def _stage_cohort_stats(
    data_root: str,
    csv_path: str,
    scan_folders: list[str],
    status_path: Path,
) -> object:
    """Stage 1 — Compute and disk-cache CohortStats (UMAP + EBM + brain-age)."""
    import threading

    from app.cohort_stats import get_cohort_stats  # noqa: PLC0415

    _update_status(status_path, stage="cohort_stats", progress=0.05)
    _log("Stage 1: CohortStats — fitting UMAP, EBM, brain-age…")
    _log(f"  csv    : {csv_path}")
    _log(f"  folders: {scan_folders}")

    # Liveness ping only — real per-stage timings come from inside
    # get_cohort_stats (look for "[cohort_stats] <stage>: Xs" lines).
    _done = threading.Event()
    def _heartbeat():
        elapsed = 0
        while not _done.wait(30):
            elapsed += 30
            _log(f"  … still running ({elapsed}s elapsed)")
    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    try:
        stats = get_cohort_stats(data_root, csv_path, scan_folders, force_refresh=False)
    finally:
        _done.set()

    _log(f"Stage 1 done — {len(stats.points)} subjects, n_rois={stats.n_rois}")
    _update_status(status_path, stage="cohort_stats_done", progress=0.40)
    return stats


def _stage_graph_metrics(
    data_root: str,
    scan_folders: list[str],
    stats,
    cache_root: Path,
    csv_path: str,
    density: float,
    status_path: Path,
) -> None:
    """Stage 2 — Compute graph metrics per cohort + write disk cache."""
    import hashlib
    import signal
    import time

    import numpy as np

    from app.biomarkers import find_subject_npz_files, load_correlation_matrix
    from app.cohort_stats import COHORTS
    from app.metadata_parser import _get_baseline, load_metadata
    from app.services.graph_metrics import _HAS_NX, subject_graph_metrics

    if not _HAS_NX:
        _log("Stage 2 SKIPPED — networkx not installed")
        _update_status(status_path, stage="graph_metrics_skipped", progress=0.55)
        return

    _update_status(status_path, stage="graph_metrics", progress=0.41)
    _log(f"Stage 2: Graph metrics (density={density})")

    # Cache key: sha1 of (csv_path, sorted_folders)
    h = hashlib.sha1()
    h.update(csv_path.encode())
    for f in sorted(scan_folders):
        h.update(b"\x00")
        h.update(f.encode())
    cache_key = h.hexdigest()[:20]

    gm_dir = cache_root / "graph_metrics"
    gm_dir.mkdir(parents=True, exist_ok=True)
    cache_path = gm_dir / f"graph_metrics_{cache_key}_density{int(density*100):02d}.json"

    if cache_path.exists():
        _log(f"Stage 2 SKIPPED — graph metrics cache hit ({cache_path.name})")
        _update_status(status_path, stage="graph_metrics_done", progress=0.55)
        return

    abs_csv = os.path.join(data_root, csv_path)
    df = load_metadata(abs_csv)
    if "diagnosis" not in df.columns or "subject_id" not in df.columns:
        _log("Stage 2 SKIPPED — metadata missing required columns")
        _update_status(status_path, stage="graph_metrics_skipped", progress=0.55)
        return

    baseline = _get_baseline(df)
    cohort_subjects: dict[str, list[str]] = {}
    for _, row in baseline.iterrows():
        diag = str(row.get("diagnosis", "")).lower()
        sid = str(row.get("subject_id", "")).strip()
        if sid and sid != "nan":
            cohort_subjects.setdefault(diag, []).append(sid)

    MAX_PER_COHORT = 20
    PER_SUBJECT_TIMEOUT_S = 60

    def _alarm_handler(signum, frame):
        raise TimeoutError("per-subject graph_metrics timeout")

    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    rng = np.random.default_rng(42)
    result: dict = {}
    try:
        for cohort in COHORTS:
            ids = cohort_subjects.get(cohort, [])
            if not ids:
                result[cohort] = {"n": 0, "metrics": {}}
                continue
            if len(ids) > MAX_PER_COHORT:
                ids = list(rng.choice(ids, size=MAX_PER_COHORT, replace=False))
            buckets: dict[str, list[float]] = {
                k: [] for k in ("small_worldness", "clustering", "path_length", "global_efficiency")
            }
            used = 0
            timed_out = 0
            for i, sid in enumerate(ids, start=1):
                recs = find_subject_npz_files(data_root, scan_folders, sid)
                if not recs:
                    continue
                try:
                    m = np.asarray(load_correlation_matrix(recs[0]["abs_path"]))
                except Exception:
                    continue
                t0 = time.monotonic()
                signal.alarm(PER_SUBJECT_TIMEOUT_S)
                try:
                    res = subject_graph_metrics(m, density=density, compute_hubs=False)
                except TimeoutError:
                    signal.alarm(0)
                    timed_out += 1
                    _log(f"  {cohort} [{i}/{len(ids)}] {sid}: TIMEOUT after {PER_SUBJECT_TIMEOUT_S}s — skipped")
                    continue
                finally:
                    signal.alarm(0)
                dt = time.monotonic() - t0
                for k in buckets:
                    v = res.get(k)
                    if v is not None:
                        buckets[k].append(float(v))
                used += 1
                _log(f"  {cohort} [{i}/{len(ids)}] {sid}: {dt:.2f}s")

            metric_summary: dict = {}
            for k, vals in buckets.items():
                if not vals:
                    metric_summary[k] = None
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                metric_summary[k] = {
                    "mean": float(arr.mean()), "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
                    "p5":  float(np.quantile(arr, 0.05)), "p25": float(np.quantile(arr, 0.25)),
                    "p50": float(np.quantile(arr, 0.50)), "p75": float(np.quantile(arr, 0.75)),
                    "p95": float(np.quantile(arr, 0.95)), "n": int(arr.size),
                }
            result[cohort] = {"n": used, "n_sampled": len(ids), "metrics": metric_summary}
            _log(f"  {cohort}: {used}/{len(ids)} subjects computed ({timed_out} timed out)")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

    payload = {
        "density": density,
        "cache_key": cache_key,
        "computed_at": _now(),
        "cohorts": COHORTS,
        "metrics_by_cohort": result,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    _log(f"Stage 2 done — written to {cache_path}")
    _update_status(status_path, stage="graph_metrics_done", progress=0.55)


def _stage_gelstm(
    data_root: str,
    scan_folders: list[str],
    csv_path: str,
    cache_root: Path,
    status_path: Path,
) -> None:
    """Stage 3 — Batch GELSTM predictions for all baseline subjects."""
    import pickle
    import re

    import numpy as np

    from app.biomarkers import find_subject_npz_files, load_correlation_matrix
    from app.metadata_parser import load_metadata
    from app.services.gelstm import get_gelstm_service  # noqa: PLC0415

    _update_status(status_path, stage="gelstm", progress=0.56)
    svc = get_gelstm_service()
    if not svc.is_available():
        _log("Stage 3 SKIPPED — GELSTM checkpoints not present")
        _update_status(status_path, stage="gelstm_skipped", progress=0.70)
        return

    _log("Stage 3: GELSTM batch predictions")
    if not svc.load_ensemble():
        _log(f"Stage 3 SKIPPED — ensemble load failed: {svc._load_error}")
        _update_status(status_path, stage="gelstm_skipped", progress=0.70)
        return

    model_version = svc._ensemble.model_version
    pred_dir = cache_root / "gelstm"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"predictions_{model_version}.pkl"

    # Load existing predictions so we don't re-compute already-done subjects.
    existing: dict = {}
    if pred_path.exists():
        try:
            existing = pickle.loads(pred_path.read_bytes())
        except Exception:
            existing = {}

    abs_csv = os.path.join(data_root, csv_path)
    df = load_metadata(abs_csv)
    if df.empty:
        _log("Stage 3 SKIPPED — metadata empty")
        _update_status(status_path, stage="gelstm_skipped", progress=0.70)
        return

    VISIT_RE = re.compile(r"M(\d+)", re.IGNORECASE)
    def _visit_months(v):
        m = VISIT_RE.match(str(v).strip())
        return int(m.group(1)) if m else None

    subjects = df["subject_id"].astype(str).unique().tolist() if "subject_id" in df.columns else []
    total = len(subjects)
    done = 0
    predictions: dict = dict(existing)

    for sid in subjects:
        if sid in predictions and predictions[sid].get("model_version") == model_version:
            done += 1
            continue  # already cached for this model version

        recs = find_subject_npz_files(data_root, scan_folders, sid)
        ordered = sorted(
            (r for r in recs if _visit_months(r.get("visit")) is not None),
            key=lambda r: _visit_months(r["visit"]),
        )
        if not ordered:
            continue

        matrices, visits, months = [], [], []
        for rec in ordered:
            try:
                matrices.append(np.asarray(load_correlation_matrix(rec["abs_path"])))
                visits.append(rec["visit"])
                months.append(_visit_months(rec["visit"]))
            except Exception:
                continue

        if not matrices:
            continue

        delta_t = [0.0] + [(months[i] - months[i-1]) / 108.0 for i in range(1, len(months))]

        # Get sex/age
        rows = df[df["subject_id"].astype(str) == sid]
        sex_val = age_val = None
        if not rows.empty:
            first = rows.iloc[0]
            if "sex" in rows.columns and first.get("sex") is not None:
                s0 = str(first["sex"]).strip().upper()[:1]
                sex_val = 0.0 if s0 == "F" else (1.0 if s0 == "M" else None)
            if "age" in rows.columns:
                try:
                    age_val = float(first["age"])
                except (TypeError, ValueError):
                    pass

        pred = svc.predict_subject(sid, matrices, delta_t, sex_val, age_val)
        predictions[sid] = pred
        done += 1

        if done % 50 == 0:
            # Save incrementally to avoid losing all work on interrupt.
            tmp = pred_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(predictions))
            tmp.replace(pred_path)
            progress = 0.56 + 0.14 * (done / max(total, 1))
            _update_status(status_path, stage="gelstm", progress=progress)
            _log(f"  GELSTM: {done}/{total} subjects")

    tmp = pred_path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(predictions))
    tmp.replace(pred_path)
    _log(f"Stage 3 done — {done} predictions written to {pred_path}")
    _update_status(status_path, stage="gelstm_done", progress=0.70)


def _stage_qc_volumes(
    data_root: str,
    scan_folders: list[str],
    csv_path: str,
    status_path: Path,
) -> None:
    """Stage 4 — Pre-compute QC temporal-std volumes for converter subjects."""
    from app.biomarkers import find_subject_nifti_files
    from app.metadata_parser import load_metadata  # noqa: PLC0415
    from app.services.qc import _ensure_qc_reduce as _ensure_qc_mean

    _update_status(status_path, stage="qc_volumes", progress=0.71)
    _log("Stage 4: QC mean volumes for converter subjects")

    abs_csv = os.path.join(data_root, csv_path)
    df = load_metadata(abs_csv)
    if "diagnosis" not in df.columns or "subject_id" not in df.columns:
        _update_status(status_path, stage="qc_volumes_skipped", progress=0.95)
        return

    converters = (
        df[df["diagnosis"].astype(str).str.lower() == "converter"]
        ["subject_id"].dropna().astype(str).unique().tolist()
    )
    n_done = 0
    for sid in converters:
        for rec in find_subject_nifti_files(data_root, scan_folders, sid):
            try:
                _ensure_qc_mean(rec["abs_path"])
                n_done += 1
            except Exception:
                continue
    _log(f"Stage 4 done — {n_done} QC volumes for {len(converters)} converters")
    _update_status(status_path, stage="qc_volumes_done", progress=0.95)


def _stage_dfc(
    data_root: str,
    scan_folders: list[str],
    csv_path: str,
    cache_root: Path,
    status_path: Path,
    k: int = 4,
    window: int = 30,
    step: int = 3,
    max_per_cohort: int = 20,
) -> None:
    """Stage 5 — Dynamic FC: parcellate BOLD .nii.gz, fit k-means states, cache."""
    import hashlib
    import json as _json

    import numpy as np

    from app.biomarkers import find_subject_nifti_files
    from app.cohort_stats import COHORTS
    from app.metadata_parser import _get_baseline, load_metadata
    from app.services.dynamic_fc import (
        _HAS_SKLEARN,
        compute_cohort_dfc,
        load_or_extract_timeseries,
    )

    if not _HAS_SKLEARN:
        _log("Stage 5 SKIPPED — scikit-learn not installed")
        _update_status(status_path, stage="dfc_skipped", progress=0.98)
        return

    _update_status(status_path, stage="dfc", progress=0.96)
    _log(f"Stage 5: Dynamic FC (k={k}, window={window}, step={step})")

    # Cache path keyed by (csv, sorted folders, k, window, step)
    h = hashlib.sha1()
    h.update(csv_path.encode())
    for f in sorted(scan_folders):
        h.update(b"\x00")
        h.update(f.encode())
    cache_key = h.hexdigest()[:20]
    dfc_dir = cache_root / "dfc"
    dfc_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dfc_dir / f"dfc_{cache_key}_k{k}_w{window}_s{step}.json"

    if cache_path.exists():
        _log(f"Stage 5 SKIPPED — dFC cache hit ({cache_path.name})")
        _update_status(status_path, stage="dfc_done", progress=0.99)
        return

    abs_csv = os.path.join(data_root, csv_path)
    df = load_metadata(abs_csv)
    if "diagnosis" not in df.columns or "subject_id" not in df.columns:
        _log("Stage 5 SKIPPED — metadata missing required columns")
        _update_status(status_path, stage="dfc_skipped", progress=0.99)
        return

    baseline = _get_baseline(df)
    cohort_subjects: dict[str, list[str]] = {}
    for _, row in baseline.iterrows():
        diag = str(row.get("diagnosis", "")).lower()
        sid = str(row.get("subject_id", "")).strip()
        if sid and sid != "nan":
            cohort_subjects.setdefault(diag, []).append(sid)

    # Plan parcellation budget upfront so progress reports show a real denominator.
    rng = np.random.default_rng(42)
    plan: list[tuple[str, str]] = []  # [(cohort, sid), ...]
    for cohort in COHORTS:
        ids = cohort_subjects.get(cohort, [])
        if not ids:
            continue
        if len(ids) > max_per_cohort:
            ids = list(rng.choice(ids, size=max_per_cohort, replace=False))
        for sid in ids:
            plan.append((cohort, str(sid)))
    plan_total = max(len(plan), 1)

    ts_by_cohort: dict[str, list] = {}
    total_subjects = 0
    subjects_done = 0
    for cohort, sid in plan:
        recs = find_subject_nifti_files(data_root, scan_folders, sid)
        if recs:
            rec = recs[0]
            visit = rec.get("visit") or rec.get("visit_code") or "M0"
            ts = load_or_extract_timeseries(rec["abs_path"], cache_root, sid, str(visit))
            if ts is not None and ts.ndim == 2 and ts.shape[0] >= window + 1:
                ts_by_cohort.setdefault(cohort, []).append(ts)
                total_subjects += 1
        subjects_done += 1
        # 0.96 → 0.99 spans the parcellation loop; update at most every 5 subjects.
        if subjects_done % 5 == 0 or subjects_done == plan_total:
            _update_status(
                status_path,
                stage="dfc",
                progress=0.96 + 0.03 * (subjects_done / plan_total),
            )
    for cohort, bucket in ts_by_cohort.items():
        _log(f"  {cohort}: parcellated {len(bucket)} subjects")

    if not ts_by_cohort:
        payload = {
            "available": False,
            "note": "No BOLD .nii.gz files could be parcellated for any cohort. Confirm the fmri folder is selected.",
        }
    else:
        _log(f"  fitting global k-means on windowed FC across {total_subjects} subjects…")
        payload = compute_cohort_dfc(ts_by_cohort, k=k, window=window, step=step)

    # Only cache successful results — caching `available=False` poisons future runs.
    if payload.get("available"):
        cache_path.write_text(_json.dumps(payload, indent=2))
        _log(f"Stage 5 done — written to {cache_path}")
    else:
        _log(f"Stage 5 — {payload.get('note', 'unavailable')} (not cached; will retry next warmup)")
    _update_status(status_path, stage="dfc_done", progress=0.99)


# ──────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute dashboard caches")
    parser.add_argument("--data-root",    required=True)
    parser.add_argument("--csv-path",     required=True)
    parser.add_argument("--scan-folders", required=True,  help="Comma-separated")
    parser.add_argument("--job-id",       required=True)
    parser.add_argument("--cache-root",   required=True)
    parser.add_argument("--density",      type=float, default=0.20)
    args = parser.parse_args()

    scan_folders = [f.strip() for f in args.scan_folders.split(",") if f.strip()]
    cache_root   = Path(args.cache_root)
    jobs_dir     = cache_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    status_path = jobs_dir / f"{args.job_id}.json"
    pid_path    = jobs_dir / f"{args.job_id}.pid"

    # Write PID immediately (job_manager may have already written it; overwrite is fine).
    pid_path.write_text(str(os.getpid()))

    # Propagate config so that app.* imports use the correct paths.
    os.environ.setdefault("DATA_ROOT",            args.data_root)
    os.environ.setdefault("DASHBOARD_CACHE_ROOT", str(cache_root))

    _update_status(
        status_path,
        status="running",
        stage="starting",
        progress=0.01,
        pid=os.getpid(),
        started_at=_now(),
        csv_path=args.csv_path,
        scan_folders=scan_folders,
        density=args.density,
        error=None,
        finished_at=None,
    )
    _log(f"Precompute job {args.job_id} started (pid={os.getpid()})")
    _log(f"  data_root={args.data_root}  csv={args.csv_path}")
    _log(f"  folders={scan_folders}  density={args.density}")

    try:
        stats = _stage_cohort_stats(args.data_root, args.csv_path, scan_folders, status_path)
        _stage_graph_metrics(args.data_root, scan_folders, stats, cache_root,
                             args.csv_path, args.density, status_path)
        _stage_gelstm(args.data_root, scan_folders, args.csv_path, cache_root, status_path)
        _stage_qc_volumes(args.data_root, scan_folders, args.csv_path, status_path)
        _stage_dfc(args.data_root, scan_folders, args.csv_path, cache_root, status_path)

        _update_status(status_path, status="done", stage="done",
                       progress=1.0, finished_at=_now(), error=None)
        _log(f"Job {args.job_id} completed successfully")
        return 0

    except KeyboardInterrupt:
        _update_status(status_path, status="cancelled", finished_at=_now(),
                       error="Received SIGINT")
        _log(f"Job {args.job_id} cancelled")
        return 1

    except Exception:
        err = traceback.format_exc()
        _update_status(status_path, status="failed", finished_at=_now(), error=err)
        _log(f"Job {args.job_id} FAILED:\n{err}")
        return 1

    finally:
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
