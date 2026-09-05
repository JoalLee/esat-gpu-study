#!/usr/bin/env python3
"""Run the W-versus-profile-variability geometric identifiability experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.span_identifiability import run_span_identifiability_grid


def _floats(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _strings(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep profile-variation alignment with the static factor span to map "
            "when within-type variability can be distinguished from changes in W."
        )
    )
    parser.add_argument("--alignments", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--variability", default="0.15,0.35,0.60")
    parser.add_argument("--noise", default="0.01,0.03,0.07")
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--models", default="static,v1,lowrank")
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--variable-factor", type=int, default=-1)
    parser.add_argument("--v1-profile-penalty", type=float, default=1.0)
    parser.add_argument("--v2-profile-penalty", type=float, default=0.005)
    parser.add_argument("--v2-sv-shrinkage", type=float, default=0.5)
    parser.add_argument("--init-iter", type=int, default=250)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/distributional/span_identifiability.csv"),
    )
    args = parser.parse_args()

    results = run_span_identifiability_grid(
        alignments=_floats(args.alignments),
        variability_levels=_floats(args.variability),
        noise_levels=_floats(args.noise),
        seeds=_ints(args.seeds),
        model_kinds=_strings(args.models),
        factors=args.factors,
        features=args.features,
        samples=args.samples,
        variable_factor=args.variable_factor,
        v1_profile_penalty=args.v1_profile_penalty,
        v2_profile_penalty=args.v2_profile_penalty,
        v2_sv_shrinkage=args.v2_sv_shrinkage,
        init_iter=args.init_iter,
        max_iter=args.max_iter,
        profile_steps=args.profile_steps,
    )

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    summary = (
        results.groupby(
            ["model_kind", "requested_alignment", "requested_variability", "noise_fraction"],
            dropna=False,
        )[
            [
                "contribution_correlation_mean",
                "contribution_relative_error",
                "local_profile_rmse",
                "variability_recovery_ratio",
                "subspace_overlap_target",
            ]
        ]
        .mean()
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"\nSaved full grid: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
