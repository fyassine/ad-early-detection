"""
atn.py — A/T/N classification + biological staging per visit.

Implements the NIA-AA 2024 revised criteria (Jack et al. 2024
*Alzheimer's & Dementia*) using the cohort's available CSF biomarkers
(Aβ42, total-tau, p-tau181). Biological-stage anchors are simplified to
0–6 based on which of A, T, N have crossed their cutoffs:

    Stage 0 — A−, T−, N−   (no AD pathology)
    Stage 1 — A+, T−, N−   (Alzheimer's pathologic change)
    Stage 2 — A+, T+, N−   (Alzheimer's disease, biological)
    Stage 3 — A+, T+, N+   (Alzheimer's disease with neurodegeneration)
    Stage S − amyloid-negative tauopathy / suspected non-AD pathology

If plasma p-Tau217 is provided in a future CSV revision, swap in the
plasma cutoff here without changing the rest of the dashboard.

Cutoffs are population-typical defaults — they should be replaced with
cohort-specific cutoffs (assay-validated) when those become available.
"""

from __future__ import annotations

from typing import Optional


# NIA-AA 2024 / Jack et al. consensus-style defaults. CSF assay typical
# cutoffs (Innogenetics/Lumipulse style); plasma cutoffs differ.
ATN_CUTOFFS = {
    # Aβ42: lower => more abnormal (amyloid plaques sequester Aβ42 from CSF).
    "abeta42_low":   600.0,    # pg/mL — A+ if value <= cutoff
    # p-Tau181: higher => more abnormal.
    "ptau181_high":   27.0,    # pg/mL — T+ if value >= cutoff
    # Total tau: higher => more abnormal (proxy for neurodegeneration).
    "ttau_high":     400.0,    # pg/mL — N+ if value >= cutoff
}


def _coerce(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def classify_atn(
    abeta42: Optional[float],
    p_tau: Optional[float],
    total_tau: Optional[float],
    cutoffs: Optional[dict] = None,
) -> dict:
    """
    Classify a single visit on the A/T/N axis.

    Returns ``{a, t, n, label, stage}`` where:
      - a, t, n     : True / False / None (None if biomarker missing)
      - label       : human-readable string e.g. "A+T+N-"
      - stage       : biological stage 0-3 (Jack 2024); 'S' if non-AD pattern
    """
    cuts = cutoffs or ATN_CUTOFFS
    a_val = _coerce(abeta42)
    t_val = _coerce(p_tau)
    n_val = _coerce(total_tau)

    a = (a_val <= cuts["abeta42_low"]) if a_val is not None else None
    t = (t_val >= cuts["ptau181_high"]) if t_val is not None else None
    n = (n_val >= cuts["ttau_high"]) if n_val is not None else None

    def sym(b):
        if b is None:
            return "?"
        return "+" if b else "-"

    label = f"A{sym(a)}T{sym(t)}N{sym(n)}"

    # Biological stage per simplified NIA-AA 2024 criteria
    if a is None or t is None:
        # Insufficient information — leave stage unknown
        stage: Optional[str] = None
    elif not a and not t:
        stage = "0"
    elif a and not t and not n:
        stage = "1"
    elif a and t and (n is False or n is None):
        stage = "2"
    elif a and t and n:
        stage = "3"
    elif (not a) and (t or n):
        # Amyloid-negative neurodegeneration / suspected non-AD pathology
        stage = "S"
    else:
        stage = None

    return {
        "a": a, "t": t, "n": n,
        "abeta42": a_val, "p_tau": t_val, "total_tau": n_val,
        "label": label, "stage": stage,
    }


def classify_visits(clinical: dict, cutoffs: Optional[dict] = None) -> list[dict]:
    """
    Classify every visit in a `get_patient_clinical_trajectory()` payload.
    Returns a list of A/T/N records aligned with `clinical['visits']`.
    """
    if not clinical or not clinical.get("visits"):
        return []
    csf = clinical.get("csf", {}) or {}
    abeta = csf.get("abeta42") or []
    tau = csf.get("tau") or []
    ptau = csf.get("ptau") or []
    n = len(clinical["visits"])
    out: list[dict] = []
    for i in range(n):
        rec = classify_atn(
            abeta42=abeta[i] if i < len(abeta) else None,
            p_tau=ptau[i] if i < len(ptau) else None,
            total_tau=tau[i] if i < len(tau) else None,
            cutoffs=cutoffs,
        )
        rec["visit"] = clinical["visits"][i]
        out.append(rec)
    return out
