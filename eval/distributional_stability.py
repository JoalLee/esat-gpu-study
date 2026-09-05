"""Cross-run stability metrics for distributional source-type solutions.

Real receptor data do not provide ground-truth factors. Stability therefore
needs to be evaluated after explicitly matching factor permutations across
independent initializations. This module uses global archetype cosine
similarity as the matching criterion and then reports agreement in archetypes,
contributions, and inferred profile variability.

A stable low-Q solution is not proof of physical correctness, but instability
is direct evidence that the fitted latent decomposition should not be treated
as uniquely identified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


_EPS = 1e-12


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("archetype matrices must have identical K x J shapes")
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
    return an @ bn.T


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) <= _EPS or np.std(b) <= _EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def match_archetypes(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate indices matched to reference factors and matched cosine."""
    similarities = _cosine_matrix(reference, candidate)
    ref_idx, candidate_idx = linear_sum_assignment(-similarities)
    order = np.empty(reference.shape[0], dtype=int)
    order[ref_idx] = candidate_idx
    matched = np.array([similarities[k, order[k]] for k in range(reference.shape[0])])
    return order, matched


@dataclass(frozen=True)
class SolutionStabilityResult:
    candidate_id: str
    factor_mapping: list[int]
    archetype_cosine: list[float]
    archetype_cosine_mean: float
    contribution_correlation: list[float]
    contribution_correlation_mean: float
    variability_absolute_difference: list[float]
    variability_absolute_difference_mean: float
    effective_rank_match_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def compare_fitted_solutions(
    reference_archetypes: np.ndarray,
    reference_contributions: np.ndarray,
    reference_variability: np.ndarray,
    reference_effective_rank: np.ndarray,
    candidate_archetypes: np.ndarray,
    candidate_contributions: np.ndarray,
    candidate_variability: np.ndarray,
    candidate_effective_rank: np.ndarray,
    *,
    candidate_id: str = "candidate",
) -> SolutionStabilityResult:
    """Align a candidate fit to a reference fit and quantify latent stability."""
    ref_h = np.asarray(reference_archetypes, dtype=np.float64)
    ref_w = np.asarray(reference_contributions, dtype=np.float64)
    ref_v = np.asarray(reference_variability, dtype=np.float64)
    ref_rank = np.asarray(reference_effective_rank, dtype=int)
    cand_h = np.asarray(candidate_archetypes, dtype=np.float64)
    cand_w = np.asarray(candidate_contributions, dtype=np.float64)
    cand_v = np.asarray(candidate_variability, dtype=np.float64)
    cand_rank = np.asarray(candidate_effective_rank, dtype=int)

    if ref_h.shape != cand_h.shape:
        raise ValueError("reference/candidate archetypes must have identical shapes")
    if ref_w.shape != cand_w.shape:
        raise ValueError("reference/candidate contributions must have identical shapes")
    if ref_v.shape != cand_v.shape or ref_v.shape != (ref_h.shape[0],):
        raise ValueError("variability vectors must contain one value per factor")
    if ref_rank.shape != cand_rank.shape or ref_rank.shape != (ref_h.shape[0],):
        raise ValueError("rank vectors must contain one value per factor")

    mapping, archetype_cosine = match_archetypes(ref_h, cand_h)
    aligned_w = cand_w[:, mapping]
    aligned_v = cand_v[mapping]
    aligned_rank = cand_rank[mapping]

    contribution_corr = [
        _safe_corr(ref_w[:, k], aligned_w[:, k]) for k in range(ref_h.shape[0])
    ]
    finite_corr = [value for value in contribution_corr if np.isfinite(value)]
    variability_difference = np.abs(ref_v - aligned_v)
    rank_match = float(np.mean(ref_rank == aligned_rank))

    return SolutionStabilityResult(
        candidate_id=str(candidate_id),
        factor_mapping=mapping.tolist(),
        archetype_cosine=archetype_cosine.tolist(),
        archetype_cosine_mean=float(np.mean(archetype_cosine)),
        contribution_correlation=contribution_corr,
        contribution_correlation_mean=(
            float(np.mean(finite_corr)) if finite_corr else float("nan")
        ),
        variability_absolute_difference=variability_difference.tolist(),
        variability_absolute_difference_mean=float(np.mean(variability_difference)),
        effective_rank_match_fraction=rank_match,
    )


def summarize_seed_stability(
    fits: list,
    seeds: list[int],
    *,
    reference_index: int | None = None,
) -> tuple[pd.DataFrame, list[SolutionStabilityResult]]:
    """Compare all fitted low-rank models to one reference solution.

    If ``reference_index`` is omitted, the fit with the lowest ``q_true`` is
    used as the reference. This is a bookkeeping convention, not a claim that
    minimum Q selects the physically correct decomposition.
    """
    if len(fits) != len(seeds) or not fits:
        raise ValueError("fits and seeds must have the same non-zero length")
    if reference_index is None:
        q_values = [float(fit.q_true) for fit in fits]
        reference_index = int(np.argmin(q_values))
    if reference_index < 0 or reference_index >= len(fits):
        raise ValueError("reference_index is outside fit range")

    reference = fits[reference_index]
    results: list[SolutionStabilityResult] = []
    rows: list[dict] = []
    for i, (fit, seed) in enumerate(zip(fits, seeds)):
        comparison = compare_fitted_solutions(
            reference_archetypes=reference.H_bar,
            reference_contributions=reference.W,
            reference_variability=reference.profile_rms_variability,
            reference_effective_rank=reference.effective_rank,
            candidate_archetypes=fit.H_bar,
            candidate_contributions=fit.W,
            candidate_variability=fit.profile_rms_variability,
            candidate_effective_rank=fit.effective_rank,
            candidate_id=f"seed_{seed}",
        )
        results.append(comparison)
        rows.append(
            {
                "seed": int(seed),
                "is_reference": i == reference_index,
                "q_true": float(fit.q_true),
                "objective": float(fit.objective),
                "archetype_cosine_to_reference": comparison.archetype_cosine_mean,
                "contribution_correlation_to_reference": (
                    comparison.contribution_correlation_mean
                ),
                "variability_abs_diff_to_reference": (
                    comparison.variability_absolute_difference_mean
                ),
                "effective_rank_match_fraction": (
                    comparison.effective_rank_match_fraction
                ),
            }
        )

    return pd.DataFrame(rows), results
