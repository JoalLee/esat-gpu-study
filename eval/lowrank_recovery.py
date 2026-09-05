"""Recovery metrics specific to low-rank source-profile families."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


_EPS = 1e-12


def _orthonormal_rows(matrix: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis spanning the row space of matrix."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if not np.any(np.abs(x) > _EPS):
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(s > max(_EPS, 1e-10 * float(s[0])))) if s.size else 0
    return vt[:rank]


def subspace_overlap(true_basis: np.ndarray, estimated_basis: np.ndarray) -> float:
    """Rotation/sign-invariant overlap between two row subspaces in [0, 1].

    The score is the average squared cosine of principal angles over the true
    subspace dimension. Missing estimated dimensions therefore reduce the
    score instead of being silently ignored.
    """
    q_true = _orthonormal_rows(true_basis)
    q_est = _orthonormal_rows(estimated_basis)
    if q_true.shape[0] == 0:
        return 1.0 if q_est.shape[0] == 0 else 0.0
    if q_est.shape[0] == 0:
        return 0.0
    cross = q_true @ q_est.T
    overlap = float(np.sum(cross**2) / q_true.shape[0])
    return float(np.clip(overlap, 0.0, 1.0))


@dataclass(frozen=True)
class LowRankRecoveryResult:
    factor_subspace_overlap: list[float]
    subspace_overlap_mean: float
    true_rank: list[int]
    estimated_rank: list[int]
    rank_absolute_error_mean: float

    def to_dict(self) -> dict:
        return asdict(self)


def compare_lowrank_truth(
    true_loadings: np.ndarray,
    estimated_loadings: np.ndarray,
    factor_mapping: list[int] | np.ndarray,
    estimated_effective_rank: np.ndarray | None = None,
) -> LowRankRecoveryResult:
    """Compare low-rank variability spaces after factor-permutation alignment."""
    true_loadings = np.asarray(true_loadings, dtype=np.float64)
    estimated_loadings = np.asarray(estimated_loadings, dtype=np.float64)
    mapping = np.asarray(factor_mapping, dtype=int)

    if true_loadings.ndim != 3 or estimated_loadings.ndim != 3:
        raise ValueError("true_loadings and estimated_loadings must be K x R x J")
    if true_loadings.shape[0] != estimated_loadings.shape[0]:
        raise ValueError("true and estimated loadings must contain the same number of factors")
    if true_loadings.shape[2] != estimated_loadings.shape[2]:
        raise ValueError("true and estimated loadings must contain the same features")
    if mapping.shape != (true_loadings.shape[0],):
        raise ValueError("factor_mapping length must match number of factors")

    overlaps: list[float] = []
    true_rank: list[int] = []
    estimated_rank: list[int] = []

    for k in range(true_loadings.shape[0]):
        est_k = int(mapping[k])
        true_basis = _orthonormal_rows(true_loadings[k])

        if estimated_effective_rank is None:
            est_basis = _orthonormal_rows(estimated_loadings[est_k])
            est_rank = est_basis.shape[0]
        else:
            est_rank = int(estimated_effective_rank[est_k])
            est_basis = _orthonormal_rows(estimated_loadings[est_k, :est_rank])

        overlaps.append(subspace_overlap(true_basis, est_basis))
        true_rank.append(int(true_basis.shape[0]))
        estimated_rank.append(int(est_rank))

    rank_error = float(
        np.mean(np.abs(np.asarray(true_rank, dtype=float) - np.asarray(estimated_rank, dtype=float)))
    )
    return LowRankRecoveryResult(
        factor_subspace_overlap=overlaps,
        subspace_overlap_mean=float(np.mean(overlaps)),
        true_rank=true_rank,
        estimated_rank=estimated_rank,
        rank_absolute_error_mean=rank_error,
    )
