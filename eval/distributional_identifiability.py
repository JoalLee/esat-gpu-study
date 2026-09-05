"""Identifiability-grid experiments for distributional source types."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from esat.model.distributional_sa import DistributionalSA
from eval.distributional_recovery import compare_distributional_truth
from eval.distributional_simulator import DistributionalSimulator


def run_identifiability_grid(
    variability_levels: Iterable[float],
    profile_penalties: Iterable[float],
    seeds: Iterable[int],
    *,
    factors: int = 3,
    features: int = 10,
    samples: int = 120,
    noise_fraction: float = 0.05,
    init_iter: int = 300,
    max_iter: int = 20,
    profile_steps: int = 5,
) -> pd.DataFrame:
    """Run a factorial synthetic recovery experiment.

    Each variability level is applied to every factor in that synthetic run.
    This deliberately produces a simple first phase diagram whose axes have a
    clear interpretation: true within-type heterogeneity versus assumed
    shrinkage strength.
    """

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
                noise_fraction=noise_fraction,
            ).generate()

            for penalty in profile_penalties:
                penalty = float(penalty)
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
                        "seed": seed,
                        "true_variability": variability,
                        "profile_penalty": penalty,
                        "q_true": float(model.q_true),
                        "objective": float(model.objective),
                        "iterations": int(model.iterations),
                        "converged": bool(model.converged),
                        "estimated_variability_mean": float(
                            model.profile_rms_variability.mean()
                        ),
                        "archetype_cosine_mean": recovery.archetype_cosine_mean,
                        "contribution_correlation_mean": (
                            recovery.contribution_correlation_mean
                        ),
                        "local_profile_rmse": recovery.local_profile_rmse,
                        "variability_rmse": recovery.variability_rmse,
                    }
                )

    return pd.DataFrame(rows)
