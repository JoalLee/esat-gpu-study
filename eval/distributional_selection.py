"""Leakage-safe utilities for real-data distributional model selection.

The helpers in this module keep the Static and low-rank distributional fits on
the same observed-cell mask, create one common held-out mask for fair model
comparison, and provide a small static LS-NMF baseline that is safe when a row
or feature has no observed cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class StaticFit:
    """Static weighted LS-NMF fit with the interface used by stability metrics."""

    W: np.ndarray
    H_bar: np.ndarray
    reconstruction: np.ndarray
    q_true: float
    objective: float
    iterations: int
    converged: bool
    profile_rms_variability: np.ndarray
    effective_rank: np.ndarray


@dataclass(frozen=True)
class StabilitySnapshot:
    """Small fitted-state snapshot retained across seed comparisons."""

    W: np.ndarray
    H_bar: np.ndarray
    q_true: float
    objective: float
    profile_rms_variability: np.ndarray
    effective_rank: np.ndarray


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return multiplicative update ratios, preserving entries with no weight."""
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )


def weighted_metrics(
    V: np.ndarray,
    U: np.ndarray,
    reconstruction: np.ndarray,
    evaluation_mask: np.ndarray,
) -> tuple[float, int, float]:
    """Return weighted Q, evaluated-cell count, and Q per evaluated cell."""
    data = np.asarray(V, dtype=np.float64)
    uncertainty = np.asarray(U, dtype=np.float64)
    estimate = np.asarray(reconstruction, dtype=np.float64)
    mask = np.asarray(evaluation_mask, dtype=bool)
    if data.shape != uncertainty.shape or data.shape != estimate.shape:
        raise ValueError("V, U, and reconstruction must have identical shapes")
    if mask.shape != data.shape:
        raise ValueError("evaluation_mask must have the same shape as V")
    if not np.isfinite(data).all() or not np.isfinite(uncertainty).all():
        raise ValueError("V and U must contain only finite values")
    if np.any(uncertainty <= 0.0):
        raise ValueError("U must contain strictly positive values")
    if not np.isfinite(estimate).all():
        raise ValueError("reconstruction must contain only finite values")

    weights = np.where(mask, 1.0 / np.maximum(uncertainty, _EPS) ** 2, 0.0)
    q_true = float(np.sum(weights * (data - estimate) ** 2))
    cell_count = int(mask.sum())
    q_per_cell = q_true / cell_count if cell_count else float("nan")
    return q_true, cell_count, float(q_per_cell)


def make_holdout_mask(
    observation_mask: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Create a deterministic cell-level holdout mask from observed cells.

    The function keeps at least one training cell in every row that has any
    observations. This avoids turning a partially observed sample into an
    all-missing sample solely because of the evaluation split.
    """
    mask = np.asarray(observation_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("observation_mask must be a 2-D array")
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("holdout fraction must be in [0, 1)")

    holdout = np.zeros_like(mask, dtype=bool)
    if fraction <= 0.0 or not np.any(mask):
        return holdout

    rng = np.random.default_rng(int(seed))
    holdout = mask & (rng.random(mask.shape) < fraction)
    fit_mask = mask & ~holdout

    for row in range(mask.shape[0]):
        observed = np.flatnonzero(mask[row])
        if observed.size and not np.any(fit_mask[row]):
            keep = int(observed[rng.integers(observed.size)])
            holdout[row, keep] = False

    # Very small test matrices can otherwise randomly produce no held-out
    # cells. Select one cell only when a row has at least two observations.
    if not np.any(holdout):
        candidate_rows = [
            row for row in range(mask.shape[0]) if int(mask[row].sum()) >= 2
        ]
        if candidate_rows:
            row = int(candidate_rows[rng.integers(len(candidate_rows))])
            observed = np.flatnonzero(mask[row])
            holdout[row, int(observed[rng.integers(observed.size)])] = True

    return holdout


def _validate_fit_inputs(
    V: np.ndarray,
    U: np.ndarray,
    observation_mask: np.ndarray,
    factors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(V, dtype=np.float64)
    uncertainty = np.asarray(U, dtype=np.float64)
    mask = np.asarray(observation_mask, dtype=bool)
    if data.ndim != 2 or uncertainty.shape != data.shape or mask.shape != data.shape:
        raise ValueError("V, U, and observation_mask must have identical 2-D shapes")
    if data.size == 0:
        raise ValueError("V cannot be empty")
    if not np.isfinite(data).all() or not np.isfinite(uncertainty).all():
        raise ValueError("V and U must contain only finite values")
    if np.any(data < 0.0):
        raise ValueError("V must be non-negative")
    if np.any(uncertainty <= 0.0):
        raise ValueError("U must contain strictly positive values")
    if not np.any(mask):
        raise ValueError("observation_mask cannot exclude every cell")
    factors = int(factors)
    if factors < 1 or factors > min(data.shape):
        raise ValueError("factors must be between 1 and the smaller matrix dimension")
    return data, uncertainty, mask


def fit_static_lsnmf(
    V: np.ndarray,
    U: np.ndarray,
    factors: int,
    observation_mask: np.ndarray,
    *,
    seed: int = 42,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> StaticFit:
    """Fit the static ESAT LS-NMF baseline on an explicit cell mask.

    This uses the same multiplicative weighted LS-NMF equations as the repo's
    ``LSNMF`` implementation. Zero denominators preserve the previous value,
    which is the neutral update for a row/feature with no data weight.
    """
    data, uncertainty, mask = _validate_fit_inputs(V, U, observation_mask, factors)
    factors = int(factors)
    max_iter = int(max_iter)
    tol = float(tol)
    if max_iter < 1 or tol <= 0.0:
        raise ValueError("max_iter must be >= 1 and tol must be > 0")

    weights = np.where(mask, 1.0 / np.maximum(uncertainty, _EPS) ** 2, 0.0)
    feature_count = mask.sum(axis=0)
    sample_count = mask.sum(axis=1)
    feature_mean = np.divide(
        (data * mask).sum(axis=0),
        feature_count,
        out=np.full(data.shape[1], 1e-8, dtype=np.float64),
        where=feature_count > 0,
    )
    sample_mean = np.divide(
        (data * mask).sum(axis=1),
        sample_count,
        out=np.full(data.shape[0], 1e-8, dtype=np.float64),
        where=sample_count > 0,
    )

    rng = np.random.default_rng(int(seed))
    H = np.maximum(
        np.sqrt(np.maximum(feature_mean, 1e-8) / factors)[None, :]
        * rng.uniform(0.5, 1.5, size=(factors, data.shape[1])),
        _EPS,
    )
    W = np.maximum(
        np.sqrt(np.maximum(sample_mean, 1e-8) / factors)[:, None]
        * rng.uniform(0.5, 1.5, size=(data.shape[0], factors)),
        _EPS,
    )
    active_rows = np.any(mask, axis=1)
    W[~active_rows] = 0.0

    def current_q(current_W: np.ndarray, current_H: np.ndarray) -> float:
        return weighted_metrics(
            data,
            uncertainty,
            current_W @ current_H,
            mask,
        )[0]

    initial_q = current_q(W, H)
    best_q = initial_q
    best_state = (W.copy(), H.copy())
    objective_history = [initial_q]
    converged = False
    iterations = 0

    for iteration in range(1, max_iter + 1):
        weighted_data = weights * data
        WH = W @ H
        H_num = W.T @ weighted_data
        H_den = W.T @ (weights * WH)
        H = np.maximum(H * _safe_ratio(H_num, H_den), _EPS)

        WH = W @ H
        W_num = weighted_data @ H.T
        W_den = (weights * WH) @ H.T
        W = np.maximum(W * _safe_ratio(W_num, W_den), 0.0)
        W[~active_rows] = 0.0

        objective = current_q(W, H)
        objective_history.append(objective)
        if objective < best_q:
            best_q = objective
            best_state = (W.copy(), H.copy())

        previous = objective_history[-2]
        relative_change = abs(previous - objective) / max(abs(previous), _EPS)
        iterations = iteration
        if relative_change < tol:
            converged = True
            break

    W, H = best_state
    H_scale = np.maximum(H.sum(axis=1), _EPS)
    H_bar = H / H_scale[:, None]
    W = W * H_scale[None, :]
    reconstruction = W @ H_bar
    q_true, _, _ = weighted_metrics(data, uncertainty, reconstruction, mask)

    return StaticFit(
        W=W,
        H_bar=H_bar,
        reconstruction=reconstruction,
        q_true=float(q_true),
        objective=float(q_true),
        iterations=int(iterations),
        converged=bool(converged),
        profile_rms_variability=np.zeros(factors, dtype=np.float64),
        effective_rank=np.zeros(factors, dtype=int),
    )


def snapshot_fit(fit) -> StabilitySnapshot:
    """Copy only the fields needed for cross-seed stability comparisons."""
    required = (
        "W",
        "H_bar",
        "q_true",
        "objective",
        "profile_rms_variability",
        "effective_rank",
    )
    missing = [name for name in required if not hasattr(fit, name)]
    if missing:
        raise ValueError(f"fit is missing stability fields: {missing}")
    return StabilitySnapshot(
        W=np.asarray(fit.W, dtype=np.float64).copy(),
        H_bar=np.asarray(fit.H_bar, dtype=np.float64).copy(),
        q_true=float(fit.q_true),
        objective=float(fit.objective),
        profile_rms_variability=np.asarray(
            fit.profile_rms_variability,
            dtype=np.float64,
        ).copy(),
        effective_rank=np.asarray(fit.effective_rank, dtype=int).copy(),
    )
