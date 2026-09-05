"""Identifiability-grid experiments for distributional source types."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from esat.model.distributional_sa import DistributionalSA
from esat.model.lowrank_distributional_sa import LowRankDistributionalSA
from eval.distributional_recovery import compare_distributional_truth
from eval.distributional_simulator import DistributionalSimulator


def run_identifiability_grid(
    variability_levels: Iterable[float],
    profile_penalties: Iterable[float],
    seeds: Iterable[int],
    *,
    model_kinds: Iterable[str] = ("v1", "lowrank"),
    variability_mode: str = "lowrank",
    variability_rank: int = 2,
    sv_shrinkage: float = 1.0,
    factors: int = 3,
    features: int = 10,
    samples: int = 120,
    noise_fraction: float = 0.05,
    init_iter: int = 300,
    max_iter: int = 20,
    profile_steps: int = 5,
) -> pd.DataFrame:
    """Run a factorial synthetic recovery experiment.

    The same synthetic truth is fitted by each requested model kind so the
    comparison isolates model representation rather than random-data changes.

    ``v1`` is the unrestricted local-profile prototype. ``lowrank`` constrains
    within-type variability to a learned low-dimensional family and estimates a
    factor-specific latent variability scale from singular-value-shrunk CLR
    deviations.
    """

    kinds = [str(k).lower() for k in model_kinds]
    invalid = sorted(set(kinds) - {"v1", "lowrank"})
    if invalid:
        raise ValueError(f"unknown model_kinds: {invalid}")

    rows: list[dict] = []
    for variability in variability_levels:
        variability = float(variability)
        for seed in seeds:
            seed = int(seed)
            synthetic = DistributionalSimulator(
                seed=seed,
                factors_n=factors,
                features_n=features,
                samples_n=samples,
                variability=variability,
                variability_mode=variability_mode,
                variability_rank=variability_rank,
                noise_fraction=noise_fraction,
            ).generate()

            for penalty in profile_penalties:
                penalty = float(penalty)
                for model_kind in kinds:
                    if model_kind == "v1":
                        model = DistributionalSA(
                            V=synthetic.data,
                            U=synthetic.uncertainty,
                            factors=factors,
                            profile_penalty=penalty,
                            seed=seed,
                            init_iter=init_iter,
                            max_iter=max_iter,
                            profile_steps=profile_steps,
                        ).fit()
                        latent_tau_mean = float("nan")
                        effective_rank_mean = float("nan")
                    else:
                        model = LowRankDistributionalSA(
                            V=synthetic.data,
                            U=synthetic.uncertainty,
                            factors=factors,
                            variability_rank=variability_rank,
                            sv_shrinkage=sv_shrinkage,
                            profile_penalty=penalty,
                            seed=seed,
                            init_iter=init_iter,
                            max_iter=max_iter,
                            profile_steps=profile_steps,
                        ).fit()
                        latent_tau_mean = float(model.latent_tau.mean())
                        effective_rank_mean = float(model.effective_rank.mean())

                    recovery = compare_distributional_truth(
                        true_archetypes=synthetic.archetypes,
                        true_contributions=synthetic.contributions,
                        true_local_profiles=synthetic.local_profiles,
                        estimated_archetypes=model.H_bar,
                        estimated_contributions=model.W,
                        estimated_local_profiles=model.H_local,
                    )

                    rows.append(
                        {
                            "model_kind": model_kind,
                            "truth_mode": variability_mode,
                            "truth_rank": int(variability_rank),
                            "fit_rank": (
                                int(variability_rank) if model_kind == "lowrank" else None
                            ),
                            "seed": seed,
                            "true_variability": variability,
                            "profile_penalty": penalty,
                            "sv_shrinkage": (
                                float(sv_shrinkage) if model_kind == "lowrank" else None
                            ),
                            "q_true": float(model.q_true),
                            "objective": float(model.objective),
                            "iterations": int(model.iterations),
                            "converged": bool(model.converged),
                            "estimated_variability_mean": float(
                                model.profile_rms_variability.mean()
                            ),
                            "latent_tau_mean": latent_tau_mean,
                            "effective_rank_mean": effective_rank_mean,
                            "archetype_cosine_mean": recovery.archetype_cosine_mean,
                            "contribution_correlation_mean": (
                                recovery.contribution_correlation_mean
                            ),
                            "local_profile_rmse": recovery.local_profile_rmse,
                            "variability_rmse": recovery.variability_rmse,
                        }
                    )

    return pd.DataFrame(rows)
