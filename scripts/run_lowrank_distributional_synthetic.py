#!/usr/bin/env python3
"""Run a controlled low-rank distributional source-type recovery experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esat.model.lowrank_distributional_sa import LowRankDistributionalSA
from eval.distributional_recovery import compare_distributional_truth
from eval.lowrank_recovery import compare_lowrank_truth
from eval.distributional_simulator import DistributionalSimulator


def _parse_variability(text: str, factors: int) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(values) == 1:
        values *= factors
    if len(values) != factors:
        raise argparse.ArgumentTypeError(
            f"variability must contain one value or {factors} comma-separated values"
        )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic recovery experiment for low-rank distributional source types."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factors", type=int, default=4)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--variability", default="0.00,0.10,0.20,0.35")
    parser.add_argument("--variability-rank", type=int, default=2)
    parser.add_argument("--fit-rank", type=int, default=2)
    parser.add_argument("--profile-penalty", type=float, default=0.001)
    parser.add_argument("--sv-shrinkage", type=float, default=0.5)
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--init-iter", type=int, default=1000)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    variability = _parse_variability(args.variability, args.factors)
    synthetic = DistributionalSimulator(
        seed=args.seed,
        factors_n=args.factors,
        features_n=args.features,
        samples_n=args.samples,
        variability=variability,
        variability_mode="lowrank",
        variability_rank=args.variability_rank,
        noise_fraction=args.noise_fraction,
    ).generate()

    model = LowRankDistributionalSA(
        V=synthetic.data,
        U=synthetic.uncertainty,
        factors=args.factors,
        variability_rank=args.fit_rank,
        sv_shrinkage=args.sv_shrinkage,
        profile_penalty=args.profile_penalty,
        seed=args.seed,
        init_iter=args.init_iter,
        max_iter=args.max_iter,
        profile_steps=args.profile_steps,
    ).fit()

    recovery = compare_distributional_truth(
        true_archetypes=synthetic.archetypes,
        true_contributions=synthetic.contributions,
        true_local_profiles=synthetic.local_profiles,
        estimated_archetypes=model.H_bar,
        estimated_contributions=model.W,
        estimated_local_profiles=model.H_local,
    )

    lowrank_recovery = None
    if synthetic.loadings is not None:
        lowrank_recovery = compare_lowrank_truth(
            true_loadings=synthetic.loadings,
            estimated_loadings=model.loadings,
            factor_mapping=recovery.factor_mapping,
            estimated_effective_rank=model.effective_rank,
            true_scores=synthetic.scores,
        )

    summary = {
        "seed": args.seed,
        "factors": args.factors,
        "features": args.features,
        "samples": args.samples,
        "truth_variability_rank": args.variability_rank,
        "fit_variability_rank": args.fit_rank,
        "profile_penalty": args.profile_penalty,
        "sv_shrinkage": args.sv_shrinkage,
        "variability_requested": variability,
        "objective_initial": float(model.objective_history[0]),
        "objective_final": float(model.objective),
        "q_true": float(model.q_true),
        "profile_penalty_loss": float(model.profile_penalty_loss),
        "iterations": int(model.iterations),
        "converged": bool(model.converged),
        "latent_tau": model.latent_tau.tolist(),
        "effective_rank": model.effective_rank.tolist(),
        "profile_rms_variability_estimated": model.profile_rms_variability.tolist(),
        "recovery": recovery.to_dict(),
        "lowrank_recovery": (
            lowrank_recovery.to_dict() if lowrank_recovery is not None else None
        ),
    }

    print(json.dumps(summary, indent=2))

    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        arrays_path = output.with_suffix(".npz")
        np.savez_compressed(
            arrays_path,
            data=synthetic.data,
            uncertainty=synthetic.uncertainty,
            true_archetypes=synthetic.archetypes,
            true_local_profiles=synthetic.local_profiles,
            true_contributions=synthetic.contributions,
            true_loadings=synthetic.loadings,
            true_scores=synthetic.scores,
            estimated_archetypes=model.H_bar,
            estimated_local_profiles=model.H_local,
            estimated_contributions=model.W,
            estimated_loadings=model.loadings,
            estimated_scores=model.scores,
            estimated_latent_tau=model.latent_tau,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
