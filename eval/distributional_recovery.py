"""Ground-truth recovery metrics for distributional source-type simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


_EPS = 1e-12


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
    return a_norm @ b_norm.T


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) <= _EPS or np.std(y) <= _EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@dataclass(frozen=True)
class DistributionalRecoveryResult:
    """Summary metrics after factor-permutation alignment."""

    factor_mapping: list[int]
    archetype_cosine: list[float]
    archetype_cosine_mean: float
    contribution_correlation: list[float]
    contribution_correlation_mean: float
    local_profile_rmse: float
    variability_true: list[float]
    variability_estimated: list[float]
    variability_rmse: float

    def to_dict(self) -> dict:
        return asdict(self)


def compare_distributional_truth(
    true_archetypes: np.ndarray,
    true_contributions: np.ndarray,
    true_local_profiles: np.ndarray,
    estimated_archetypes: np.ndarray,
    estimated_contributions: np.ndarray,
    estimated_local_profiles: np.ndarray,
) -> DistributionalRecoveryResult:
    """Align estimated factors to truth and calculate recovery metrics.

    Factor alignment maximizes cosine similarity between global archetypes.
    The returned ``factor_mapping[k]`` is the estimated factor index assigned
    to true factor ``k``.
    """

    true_archetypes = np.asarray(true_archetypes, dtype=np.float64)
    estimated_archetypes = np.asarray(estimated_archetypes, dtype=np.float64)
    true_contributions = np.asarray(true_contributions, dtype=np.float64)
    estimated_contributions = np.asarray(estimated_contributions, dtype=np.float64)
    true_local_profiles = np.asarray(true_local_profiles, dtype=np.float64)
    estimated_local_profiles = np.asarray(estimated_local_profiles, dtype=np.float64)

    if true_archetypes.shape != estimated_archetypes.shape:
        raise ValueError("true and estimated archetypes must have identical shapes")
    if true_contributions.shape != estimated_contributions.shape:
        raise ValueError("true and estimated contributions must have identical shapes")
    if true_local_profiles.shape != estimated_local_profiles.shape:
        raise ValueError("true and estimated local profiles must have identical shapes")

    similarities = _cosine_similarity_matrix(true_archetypes, estimated_archetypes)
    true_idx, estimated_idx = linear_sum_assignment(-similarities)
    order = np.empty(len(true_idx), dtype=int)
    order[true_idx] = estimated_idx

    aligned_archetypes = estimated_archetypes[order]
    aligned_contributions = estimated_contributions[:, order]
    aligned_local_profiles = estimated_local_profiles[:, order, :]

    archetype_cosine = [
        float(similarities[k, order[k]]) for k in range(true_archetypes.shape[0])
    ]
    contribution_correlation = [
        _safe_corr(true_contributions[:, k], aligned_contributions[:, k])
        for k in range(true_archetypes.shape[0])
    ]

    local_profile_rmse = float(
        np.sqrt(np.mean((true_local_profiles - aligned_local_profiles) ** 2))
    )

    true_variability = np.sqrt(
        np.mean(
            (true_local_profiles - true_archetypes[None, :, :]) ** 2,
            axis=(0, 2),
        )
    )
    estimated_variability = np.sqrt(
        np.mean(
            (aligned_local_profiles - aligned_archetypes[None, :, :]) ** 2,
            axis=(0, 2),
        )
    )
    variability_rmse = float(
        np.sqrt(np.mean((true_variability - estimated_variability) ** 2))
    )

    finite_corr = [v for v in contribution_correlation if np.isfinite(v)]
    corr_mean = float(np.mean(finite_corr)) if finite_corr else float("nan")

    return DistributionalRecoveryResult(
        factor_mapping=order.tolist(),
        archetype_cosine=archetype_cosine,
        archetype_cosine_mean=float(np.mean(archetype_cosine)),
        contribution_correlation=contribution_correlation,
        contribution_correlation_mean=corr_mean,
        local_profile_rmse=local_profile_rmse,
        variability_true=true_variability.tolist(),
        variability_estimated=estimated_variability.tolist(),
        variability_rmse=variability_rmse,
    )
