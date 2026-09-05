"""Geometric identifiability experiments for W versus profile variability.

This module treats the angle between a true profile-variation direction and the
static factor-difference span as a first-class experimental variable. It fits
three model classes to the exact same synthetic dataset:

* ``static``: ordinary weighted LS-NMF with one fixed profile per factor;
* ``v1``: unrestricted local-profile DistributionalSA;
* ``lowrank``: low-rank distributional profile family.

The resulting table is intended for an identifiability phase diagram, not for
selecting a final real-data model by Q alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from esat.model.distributional_sa import DistributionalSA
from esat.model.lowrank_distributional_sa import LowRankDistributionalSA
from esat.model.ls_nmf import LSNMF
from eval.distributional_recovery import compare_distributional_truth
from eval.lowrank_recovery import compare_lowrank_truth
from eval.span_controlled_simulator import SpanControlledSimulator


_EPS = 1e-12


@dataclass(frozen=True)
class StaticFit:
    W: np.ndarray
    H: np.ndarray
    q_true: float


def fit_static_lsnmf(
    V: np.ndarray,
    U: np.ndarray,
    factors: int,
    *,
    seed: int = 42,
    max_iter: int = 1000,
) -> StaticFit:
    """Fit the repo's weighted LS-NMF and normalize H for fair comparison."""
    V = np.asarray(V, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if V.shape != U.shape:
        raise ValueError("V and U must have the same shape")
    if np.any(U <= 0.0):
        raise ValueError("U must be strictly positive")

    rng = np.random.default_rng(int(seed))
    W = np.maximum(rng.random((V.shape[0], factors)), 1e-6)
    H = np.maximum(rng.random((factors, V.shape[1])), 1e-6)
    We = 1.0 / np.maximum(U**2, _EPS)

    for _ in range(int(max_iter)):
        W, H = LSNMF.update(V=V, We=We, W=W, H=H)
        W = np.maximum(np.nan_to_num(W, nan=1e-12, posinf=1e6), 1e-12)
        H = np.maximum(np.nan_to_num(H, nan=1e-12, posinf=1e6), 1e-12)

    scales = np.maximum(H.sum(axis=1), _EPS)
    H = H / scales[:, None]
    W = W * scales[None, :]
    residual = V - W @ H
    q_true = float(np.sum(We * residual**2))
    return StaticFit(W=W, H=H, q_true=q_true)


def _contribution_relative_error(
    true_contributions: np.ndarray,
    estimated_contributions: np.ndarray,
    factor_mapping: list[int],
) -> float:
    """Scale-aware W distortion after factor permutation alignment."""
    truth = np.asarray(true_contributions, dtype=np.float64)
    estimate = np.asarray(estimated_contributions, dtype=np.float64)[:, factor_mapping]
    numerator = np.linalg.norm(estimate - truth)
    denominator = max(float(np.linalg.norm(truth)), _EPS)
    return float(numerator / denominator)


def run_span_identifiability_grid(
    alignments: Iterable[float],
    variability_levels: Iterable[float],
    noise_levels: Iterable[float],
    seeds: Iterable[int],
    *,
    model_kinds: Iterable[str] = ("static", "v1", "lowrank"),
    factors: int = 3,
    features: int = 10,
    samples: int = 160,
    variable_factor: int = -1,
    v1_profile_penalty: float = 1.0,
    v2_profile_penalty: float = 0.005,
    v2_sv_shrinkage: float = 0.5,
    init_iter: int = 250,
    max_iter: int = 15,
    profile_steps: int = 5,
) -> pd.DataFrame:
    """Sweep confounding geometry, variability magnitude, noise, and seed.

    ``variable_factor`` selects which factor receives profile variability. All
    other factors are generated as static source types. This makes attribution
    error directly observable rather than averaging over multiple simultaneous
    profile families.
    """
    kinds = [str(k).lower() for k in model_kinds]
    invalid = sorted(set(kinds) - {"static", "v1", "lowrank"})
    if invalid:
        raise ValueError(f"unknown model_kinds: {invalid}")

    target = int(variable_factor)
    if target < 0:
        target += factors
    if target < 0 or target >= factors:
        raise ValueError("variable_factor is outside factor range")

    rows: list[dict] = []
    for alignment in alignments:
        for variability in variability_levels:
            variability_vector = np.zeros(factors, dtype=np.float64)
            variability_vector[target] = float(variability)

            for noise in noise_levels:
                for seed in seeds:
                    synthetic = SpanControlledSimulator(
                        seed=int(seed),
                        factors_n=factors,
                        features_n=features,
                        samples_n=samples,
                        alignment=float(alignment),
                        variability=variability_vector,
                        noise_fraction=float(noise),
                    ).generate()

                    for model_kind in kinds:
                        lowrank_metrics = None
                        if model_kind == "static":
                            fitted = fit_static_lsnmf(
                                synthetic.data,
                                synthetic.uncertainty,
                                factors,
                                seed=int(seed),
                                max_iter=init_iter,
                            )
                            est_H = fitted.H
                            est_W = fitted.W
                            est_local = np.broadcast_to(
                                est_H,
                                (samples, factors, features),
                            ).copy()
                            q_true = fitted.q_true
                            estimated_variability = np.zeros(factors)
                            latent_tau = np.full(factors, np.nan)
                            effective_rank = np.zeros(factors)
                        elif model_kind == "v1":
                            model = DistributionalSA(
                                V=synthetic.data,
                                U=synthetic.uncertainty,
                                factors=factors,
                                profile_penalty=v1_profile_penalty,
                                seed=int(seed),
                                init_iter=init_iter,
                                max_iter=max_iter,
                                profile_steps=profile_steps,
                            ).fit()
                            est_H = model.H_bar
                            est_W = model.W
                            est_local = model.H_local
                            q_true = float(model.q_true)
                            estimated_variability = model.profile_rms_variability
                            latent_tau = np.full(factors, np.nan)
                            effective_rank = np.full(factors, np.nan)
                        else:
                            model = LowRankDistributionalSA(
                                V=synthetic.data,
                                U=synthetic.uncertainty,
                                factors=factors,
                                variability_rank=1,
                                sv_shrinkage=v2_sv_shrinkage,
                                profile_penalty=v2_profile_penalty,
                                seed=int(seed),
                                init_iter=init_iter,
                                max_iter=max_iter,
                                profile_steps=profile_steps,
                            ).fit()
                            est_H = model.H_bar
                            est_W = model.W
                            est_local = model.H_local
                            q_true = float(model.q_true)
                            estimated_variability = model.profile_rms_variability
                            latent_tau = model.latent_tau
                            effective_rank = model.effective_rank

                        recovery = compare_distributional_truth(
                            true_archetypes=synthetic.archetypes,
                            true_contributions=synthetic.contributions,
                            true_local_profiles=synthetic.local_profiles,
                            estimated_archetypes=est_H,
                            estimated_contributions=est_W,
                            estimated_local_profiles=est_local,
                        )

                        if model_kind == "lowrank":
                            lowrank_metrics = compare_lowrank_truth(
                                true_loadings=synthetic.loadings,
                                estimated_loadings=model.loadings,
                                factor_mapping=recovery.factor_mapping,
                                estimated_effective_rank=model.effective_rank,
                                true_scores=synthetic.scores,
                            )

                        mapped_target = int(recovery.factor_mapping[target])
                        rows.append(
                            {
                                "model_kind": model_kind,
                                "seed": int(seed),
                                "requested_alignment": float(alignment),
                                "actual_alignment_target": float(
                                    synthetic.actual_alignment[target]
                                ),
                                "requested_variability": float(variability),
                                "true_profile_variability_target": float(
                                    synthetic.actual_profile_rms_variability[target]
                                ),
                                "noise_fraction": float(noise),
                                "variable_factor": target,
                                "mapped_estimated_factor": mapped_target,
                                "q_true": q_true,
                                "archetype_cosine_mean": recovery.archetype_cosine_mean,
                                "contribution_correlation_mean": (
                                    recovery.contribution_correlation_mean
                                ),
                                "contribution_relative_error": _contribution_relative_error(
                                    synthetic.contributions,
                                    est_W,
                                    recovery.factor_mapping,
                                ),
                                "local_profile_rmse": recovery.local_profile_rmse,
                                "variability_rmse": recovery.variability_rmse,
                                "estimated_variability_target": float(
                                    estimated_variability[mapped_target]
                                ),
                                "variability_recovery_ratio": float(
                                    estimated_variability[mapped_target]
                                    / max(
                                        synthetic.actual_profile_rms_variability[target],
                                        _EPS,
                                    )
                                )
                                if synthetic.actual_profile_rms_variability[target] > _EPS
                                else float("nan"),
                                "latent_tau_target": float(latent_tau[mapped_target]),
                                "effective_rank_target": float(
                                    effective_rank[mapped_target]
                                ),
                                "subspace_overlap_target": (
                                    float(
                                        lowrank_metrics.factor_subspace_overlap[target]
                                    )
                                    if lowrank_metrics is not None
                                    else float("nan")
                                ),
                            }
                        )

    return pd.DataFrame(rows)
