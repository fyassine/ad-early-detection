import os
from typing import Optional

import numpy as np

from ..config import DATA_ROOT


def _safe_round_matrix(m: np.ndarray, decimals: int = 4) -> list:
    """JSON-safe nested list with NaN/inf clipped to 0."""
    arr = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.round(arr.astype(np.float64), decimals)
    return arr.tolist()


def _safe_under_root(abs_path: str) -> bool:
    """Guard against directory traversal — abs_path must live under DATA_ROOT."""
    try:
        root = os.path.realpath(DATA_ROOT)
        target = os.path.realpath(abs_path)
        return target == root or target.startswith(root + os.sep)
    except Exception:
        return False


# ── HTTP caching helpers ──────────────────────────────────────────────────────

def cache_headers(fingerprint: str, max_age: int = 300) -> dict:
    """
    Build HTTP caching headers for heavy, rarely-changing JSON responses.

    max_age=300 (5 min) keeps the response fresh in the browser/tunnel cache
    while still allowing timely updates when data changes. The ETag (first 16
    chars of the .npz fingerprint) enables efficient revalidation with 304.

    stale-while-revalidate=3600 lets the browser serve a stale response while
    fetching a fresh one in the background — critical for tunnel latency.
    """
    etag = f'"{fingerprint[:16]}"' if fingerprint else '"nofingerprint"'
    return {
        "Cache-Control": f"private, max-age={max_age}, stale-while-revalidate=3600",
        "ETag": etag,
        "Vary": "Accept-Encoding",
    }


def check_not_modified(request_headers: dict, etag: str) -> bool:
    """
    Return True if the client's If-None-Match header matches our ETag,
    meaning the cached response is still valid and we should return 304.

    Usage in route handlers:
        from fastapi import Request
        from fastapi.responses import Response

        def my_handler(request: Request, ...):
            stats = get_cohort_stats(...)
            headers = cache_headers(stats.fingerprint)
            etag = headers["ETag"]
            if check_not_modified(dict(request.headers), etag):
                return Response(status_code=304, headers=headers)
            return JSONResponse(data, headers=headers)
    """
    client_etag = request_headers.get("if-none-match", "")
    return bool(client_etag) and (client_etag == etag or client_etag == f'W/{etag}')
