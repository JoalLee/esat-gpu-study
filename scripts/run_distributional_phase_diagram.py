#!/usr/bin/env python3
"""Run the first distributional source-type identifiability phase diagram."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.distributional_identifiability import run_identifiability_grid


def _floats(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep true profile variability and model shrinkage strength."
    )
    parser.add_argument("--variability", default="0,0.05,0.10,0.20,0.35")
    parser.add_argument("--penalties", default="0.1,0.3,1,3,10,100")
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--init-iter", type=int, default=300)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/distributional/identifiability_v1.csv"),
    )
    args = parser.parse_args()

    results = run_identifiability_grid(
        variability_levels=_floats(args.variability),
        profile_penalties=_floats(args.penalties),
        seeds=_ints(args.seeds),
        factors=args.factors,
        features=args.features,
        samples=args.samples,
        noise_fraction=args.noise_fraction,
        init_iter=args.init_iter,
        max_iter=args.max_iter,
        profile_steps=args.profile_steps,
    )

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
