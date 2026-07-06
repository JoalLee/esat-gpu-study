import numpy as np

import esat_rust


def weighted_errors_from_uncertainty(U: np.ndarray) -> np.ndarray:
    """Return LS-NMF weighted-error matrix as float64."""
    return np.float64(1.0) / (U.astype(np.float64) ** np.float64(2.0))


def run_ls_nmf_batched(
        V: np.ndarray,
        W: np.ndarray,
        H: np.ndarray,
        max_iter: int,
        *,
        U: np.ndarray = None,
        We: np.ndarray = None,
        hold_h: bool = False,
        prefer_gpu: bool = True,
) -> dict:
    """Run the Rust batched LS-NMF kernel with consistent dtype/backend handling."""
    if We is None:
        if U is None:
            raise ValueError("Either U or We must be provided")
        We = weighted_errors_from_uncertainty(U)

    result = esat_rust.ls_nmf_batched(
        V.astype(np.float64),
        We.astype(np.float64),
        W.astype(np.float64),
        H.astype(np.float64),
        int(max_iter),
        bool(hold_h),
        bool(prefer_gpu),
    )
    result["backend"] = result.get("backend", "unknown")
    return result
