#!/usr/bin/env python3
"""Run leakage-safe Static-versus-V2 selection experiments on FS0610.

Each factor count and seed uses one common held-out mask. Static weighted
LS-NMF and low-rank distributional V2 therefore see the same concentration,
uncertainty, and observed-cell mask. The script reports training Q, held-out Q,
source separation, multi-seed stability, and provisional W/F confounding flags;
it does not select a winning K automatically.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esat.model.lowrank_distributional_sa import LowRankDistributionalSA
from eval.distributional_stability import summarize_seed_stability
from eval.distributional_selection import (
    fit_static_lsnmf,
    make_holdout_mask,
    snapshot_fit,
    weighted_metrics,
)
from eval.identifiability_diagnostics import (
    archetype_separation_summary,
    diagnose_lowrank_model,
)
from scripts.run_distributional_fs0610 import read_and_clean_distributional


DEFAULT_FACTORS = (4, 5, 6, 7, 8)
DEFAULT_SEEDS = (11, 17, 23, 31, 41, 53, 67, 79, 97, 113)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _parse_int_list(text: str, name: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} cannot contain duplicates")
    return values


def _parse_models(text: str) -> list[str]:
    aliases = {"static": "static", "lowrank": "lowrank_v2", "v2": "lowrank_v2"}
    values = [value.strip().lower() for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("models cannot be empty")
    invalid = sorted(set(values) - set(aliases))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown model(s): {invalid}; use static, lowrank, or v2"
        )
    canonical = [aliases[value] for value in values]
    if len(set(canonical)) != len(canonical):
        raise argparse.ArgumentTypeError("models cannot contain duplicates")
    return canonical


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _metric_row(
    V: np.ndarray,
    U: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int, float]:
    return weighted_metrics(V, U, reconstruction, mask)


def _static_diagnostic_rows(
    factors: int,
    seed: int,
    separation: dict[str, float],
) -> list[dict]:
    rows = []
    for factor in range(factors):
        rows.append(
            {
                "model": "static",
                "factors": int(factors),
                "seed": int(seed),
                "factor": int(factor),
                "factor_label": f"factor_{factor + 1}",
                "effective_rank": 0,
                "profile_rms_variability": 0.0,
                "latent_tau": float("nan"),
                "span_alignment": float("nan"),
                "unique_direction_fraction": float("nan"),
                "max_cosine_to_other_archetype": float("nan"),
                "diagnostic_label": "static_model_no_active_family",
                **separation,
            }
        )
    return rows


def _lowrank_diagnostic_rows(
    factors: int,
    seed: int,
    diagnostics,
    separation: dict[str, float],
) -> list[dict]:
    rows = []
    for record in diagnostics.factor_table.to_dict(orient="records"):
        factor = int(record["factor"])
        rows.append(
            {
                "model": "lowrank_v2",
                "factors": int(factors),
                "seed": int(seed),
                "factor": factor,
                "factor_label": f"factor_{factor + 1}",
                **record,
                **separation,
            }
        )
    return rows


def _summary_rows(
    runs: pd.DataFrame,
    stability: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    for (model, factors), group in runs.groupby(["model", "factors"], sort=True):
        stability_group = stability[
            (stability["model"] == model)
            & (stability["factors"] == factors)
            & ~stability["is_reference"].astype(bool)
        ]
        diagnostic_group = diagnostics[
            (diagnostics["model"] == model)
            & (diagnostics["factors"] == factors)
        ]

        row = {
            "model": model,
            "factors": int(factors),
            "seed_count": int(group["seed"].nunique()),
            "q_train_per_cell_median": float(group["q_train_per_cell"].median()),
            "q_heldout_per_cell_median": float(
                group["q_heldout_per_cell"].median()
            ),
            "q_heldout_per_cell_mean": float(group["q_heldout_per_cell"].mean()),
            "q_all_observed_per_cell_median": float(
                group["q_all_observed_per_cell"].median()
            ),
            "converged_fraction": float(group["converged"].mean()),
            "iterations_median": float(group["iterations"].median()),
            "fit_seconds_median": float(group["fit_seconds"].median()),
            "pairwise_archetype_cosine_max_median": float(
                group["pairwise_cosine_max"].median()
            ),
            "profile_variability_mean_median": float(
                group["profile_variability_mean"].median()
            ),
            "active_variability_factor_count_median": float(
                group["active_variability_factor_count"].median()
            ),
            "high_geometric_confounding_factor_count_median": float(
                group["high_geometric_confounding_factor_count"].median()
            ),
            "moderate_geometric_confounding_factor_count_median": float(
                group["moderate_geometric_confounding_factor_count"].median()
            ),
            "diagnostic_rows": int(len(diagnostic_group)),
        }

        if not stability_group.empty:
            row.update(
                {
                    "stability_archetype_cosine_median": float(
                        stability_group["archetype_cosine_to_reference"].median()
                    ),
                    "stability_contribution_correlation_median": float(
                        stability_group[
                            "contribution_correlation_to_reference"
                        ].median()
                    ),
                    "stability_variability_abs_diff_median": float(
                        stability_group[
                            "variability_abs_diff_to_reference"
                        ].median()
                    ),
                    "stability_effective_rank_match_fraction_median": float(
                        stability_group[
                            "effective_rank_match_fraction"
                        ].median()
                    ),
                }
            )
        else:
            row.update(
                {
                    "stability_archetype_cosine_median": float("nan"),
                    "stability_contribution_correlation_median": float("nan"),
                    "stability_variability_abs_diff_median": float("nan"),
                    "stability_effective_rank_match_fraction_median": float("nan"),
                }
            )
        rows.append(row)
    return rows


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; pass --overwrite to reuse it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Static ESAT and low-rank distributional V2 on FS0610."
    )
    parser.add_argument(
        "--models",
        default="static,lowrank",
        help="Comma-separated models: static,lowrank (or v2).",
    )
    parser.add_argument(
        "--factors",
        default=",".join(str(value) for value in DEFAULT_FACTORS),
        help="Comma-separated factor counts; no automatic winner is selected.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--species",
        type=Path,
        default=Path("data/PMF_data/species_FS0610.csv"),
    )
    parser.add_argument(
        "--uncertainty",
        type=Path,
        default=Path("data/PMF_data/unc_FS0610.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/distributional_fs0610_k_sweep"),
    )
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--missing-sentinel", type=float, default=-900.0)
    parser.add_argument("--missing-uncertainty-scale", type=float, default=4.0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--row-step", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--drop-all-missing-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with no observed cells before all models see the data.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--holdout-seed", type=int, default=20260906)
    parser.add_argument("--static-max-iter", type=int, default=500)
    parser.add_argument("--init-iter", type=int, default=250)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--v2-variability-rank", type=int, default=2)
    parser.add_argument("--v2-profile-penalty", type=float, default=0.005)
    parser.add_argument("--v2-family-penalty", type=float, default=0.0)
    parser.add_argument("--v2-sv-shrinkage", type=float, default=0.5)
    parser.add_argument(
        "--variability-threshold",
        type=float,
        default=1e-5,
        help="Threshold for reporting a factor as substantively variable.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    models = _parse_models(args.models)
    factors_list = _parse_int_list(args.factors, "factors")
    seeds = _parse_int_list(args.seeds, "seeds")
    if any(value < 1 for value in factors_list):
        parser.error("factors must all be >= 1")
    if args.static_max_iter < 1 or args.init_iter < 0 or args.max_iter < 1:
        parser.error("iteration counts are invalid")
    if args.profile_steps < 1 or args.tol <= 0.0:
        parser.error("profile_steps must be >= 1 and tol must be > 0")
    if args.variability_threshold < 0.0:
        parser.error("variability-threshold must be >= 0")
    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("holdout-fraction must be in (0, 1)")

    species_path = resolve_path(args.species)
    uncertainty_path = resolve_path(args.uncertainty)
    output_dir = resolve_path(args.output_dir)
    if not species_path.is_file():
        raise FileNotFoundError(f"species file does not exist: {species_path}")
    if not uncertainty_path.is_file():
        raise FileNotFoundError(f"uncertainty file does not exist: {uncertainty_path}")

    V, U, observation_mask, features, times, cleaning = (
        read_and_clean_distributional(
            species_path,
            uncertainty_path,
            time_col=args.time_col,
            missing_sentinel=args.missing_sentinel,
            missing_uncertainty_scale=args.missing_uncertainty_scale,
            start_row=args.start_row,
            row_step=args.row_step,
            max_rows=args.max_rows,
        )
    )

    cleaning = dict(cleaning)
    rows_before_drop = int(V.shape[0])
    if args.drop_all_missing_rows:
        keep = observation_mask.any(axis=1)
        V = V[keep]
        U = U[keep]
        observation_mask = observation_mask[keep]
        times = times[keep]
    dropped_rows = rows_before_drop - int(V.shape[0])
    cleaning["rows_before_all_missing_drop"] = rows_before_drop
    cleaning["dropped_all_missing_rows"] = dropped_rows
    cleaning["selected_rows_after_all_missing_drop"] = int(V.shape[0])
    cleaning["all_missing_rows_after_drop"] = int(
        (~observation_mask.any(axis=1)).sum()
    )
    if V.shape[0] == 0 or not np.any(observation_mask):
        raise ValueError("no observed samples remain after row selection")

    holdout_mask = make_holdout_mask(
        observation_mask,
        fraction=args.holdout_fraction,
        seed=args.holdout_seed,
    )
    fit_mask = observation_mask & ~holdout_mask
    holdout_fraction_realized = float(
        holdout_mask.sum() / max(int(observation_mask.sum()), 1)
    )

    prepare_output_dir(output_dir, args.overwrite)
    np.savez_compressed(
        output_dir / "holdout_masks.npz",
        observation_mask=observation_mask,
        fit_mask=fit_mask,
        holdout_mask=holdout_mask,
        features=np.asarray(features, dtype=str),
        times=np.asarray(times.astype(str), dtype=str),
    )

    run_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    stability_rows: list[dict] = []
    stability_groups: dict[tuple[str, int], tuple[list, list[int]]] = {}
    total_runs = len(factors_list) * len(models) * len(seeds)
    completed_runs = 0

    for factors in factors_list:
        if factors > min(V.shape):
            raise ValueError(
                f"factors={factors} exceeds the selected matrix dimensions {V.shape}"
            )
        for model_kind in models:
            snapshots = []
            for seed in seeds:
                print(
                    f"[{completed_runs + 1}/{total_runs}] fitting "
                    f"model={model_kind} factors={factors} seed={seed}",
                    flush=True,
                )
                started = time.perf_counter()
                if model_kind == "static":
                    fitted = fit_static_lsnmf(
                        V,
                        U,
                        factors,
                        fit_mask,
                        seed=seed,
                        max_iter=args.static_max_iter,
                        tol=args.tol,
                    )
                    H_bar = fitted.H_bar
                    profile_rms = fitted.profile_rms_variability
                    effective_rank = fitted.effective_rank
                    diagnostics = None
                else:
                    fitted = LowRankDistributionalSA(
                        V=V,
                        U=U,
                        factors=factors,
                        observation_mask=fit_mask,
                        variability_rank=args.v2_variability_rank,
                        sv_shrinkage=args.v2_sv_shrinkage,
                        profile_penalty=args.v2_profile_penalty,
                        family_penalty=args.v2_family_penalty,
                        seed=seed,
                        init_iter=args.init_iter,
                        max_iter=args.max_iter,
                        profile_steps=args.profile_steps,
                        tol=args.tol,
                    ).fit()
                    H_bar = fitted.H_bar
                    profile_rms = fitted.profile_rms_variability
                    effective_rank = fitted.effective_rank
                    diagnostics = diagnose_lowrank_model(fitted)

                fit_q, fit_cells, fit_q_per_cell = _metric_row(
                    V,
                    U,
                    fitted.reconstruction,
                    fit_mask,
                )
                holdout_q, holdout_cells, holdout_q_per_cell = _metric_row(
                    V,
                    U,
                    fitted.reconstruction,
                    holdout_mask,
                )
                all_q, all_cells, all_q_per_cell = _metric_row(
                    V,
                    U,
                    fitted.reconstruction,
                    observation_mask,
                )
                separation = {
                    key: _finite_or_nan(value)
                    for key, value in archetype_separation_summary(H_bar).items()
                }

                profile_rms = np.asarray(profile_rms, dtype=np.float64)
                effective_rank = np.asarray(effective_rank, dtype=float)
                if diagnostics is None:
                    labels = []
                    high_count = 0
                    moderate_count = 0
                    diagnostic_rows.extend(
                        _static_diagnostic_rows(factors, seed, separation)
                    )
                else:
                    labels = diagnostics.factor_table["diagnostic_label"].astype(str)
                    high_count = int((labels == "high_geometric_confounding").sum())
                    moderate_count = int(
                        (labels == "moderate_geometric_confounding").sum()
                    )
                    diagnostic_rows.extend(
                        _lowrank_diagnostic_rows(
                            factors,
                            seed,
                            diagnostics,
                            separation,
                        )
                    )

                fit_seconds = time.perf_counter() - started
                run_rows.append(
                    {
                        "model": model_kind,
                        "factors": int(factors),
                        "seed": int(seed),
                        "samples": int(V.shape[0]),
                        "features": int(V.shape[1]),
                        "fit_observed_cells": int(fit_cells),
                        "holdout_cells": int(holdout_cells),
                        "observed_cells": int(all_cells),
                        "holdout_fraction_realized": holdout_fraction_realized,
                        "q_train": float(fit_q),
                        "q_train_per_cell": float(fit_q_per_cell),
                        "q_heldout": float(holdout_q),
                        "q_heldout_per_cell": float(holdout_q_per_cell),
                        "q_all_observed": float(all_q),
                        "q_all_observed_per_cell": float(all_q_per_cell),
                        "objective": float(fitted.objective),
                        "iterations": int(fitted.iterations),
                        "converged": bool(fitted.converged),
                        "fit_seconds": float(fit_seconds),
                        "profile_variability_mean": float(profile_rms.mean()),
                        "profile_variability_max": float(profile_rms.max()),
                        "active_variability_factor_count": int(
                            np.sum(profile_rms > args.variability_threshold)
                        ),
                        "effective_rank_mean": float(effective_rank.mean()),
                        "effective_rank_active_factor_count": int(
                            np.sum(effective_rank > 0)
                        ),
                        "high_geometric_confounding_factor_count": high_count,
                        "moderate_geometric_confounding_factor_count": moderate_count,
                        **separation,
                    }
                )
                snapshots.append(snapshot_fit(fitted))
                completed_runs += 1
                print(
                    f"[{completed_runs}/{total_runs}] finished "
                    f"model={model_kind} factors={factors} seed={seed} "
                    f"q_heldout_per_cell={holdout_q_per_cell:.6g} "
                    f"seconds={fit_seconds:.3f}",
                    flush=True,
                )

            stability_groups[(model_kind, factors)] = (snapshots, seeds)

    for (model_kind, factors), (snapshots, group_seeds) in stability_groups.items():
        group_stability, _ = summarize_seed_stability(snapshots, group_seeds)
        group_stability = group_stability.copy()
        group_stability.insert(0, "model", model_kind)
        group_stability.insert(1, "factors", int(factors))
        stability_rows.extend(group_stability.to_dict(orient="records"))

    runs_df = pd.DataFrame(run_rows).sort_values(
        ["model", "factors", "seed"],
        ignore_index=True,
    )
    stability_df = pd.DataFrame(stability_rows).sort_values(
        ["model", "factors", "seed"],
        ignore_index=True,
    )
    diagnostics_df = pd.DataFrame(diagnostic_rows).sort_values(
        ["model", "factors", "seed", "factor"],
        ignore_index=True,
    )
    summary_df = pd.DataFrame(
        _summary_rows(runs_df, stability_df, diagnostics_df)
    ).sort_values(["model", "factors"], ignore_index=True)

    runs_df.to_csv(output_dir / "k_sweep_runs.csv", index=False)
    stability_df.to_csv(output_dir / "k_sweep_stability.csv", index=False)
    diagnostics_df.to_csv(output_dir / "k_sweep_diagnostics.csv", index=False)
    summary_df.to_csv(output_dir / "k_sweep_summary.csv", index=False)

    metadata = {
        "git_commit": git_commit(),
        "species_path": str(species_path),
        "uncertainty_path": str(uncertainty_path),
        "output_dir": str(output_dir),
        "protocol": {
            "models": models,
            "factors": factors_list,
            "seeds": seeds,
            "same_holdout_mask_for_all_runs": True,
            "holdout_seed": int(args.holdout_seed),
            "holdout_fraction_requested": float(args.holdout_fraction),
            "holdout_fraction_realized": holdout_fraction_realized,
            "drop_all_missing_rows": bool(args.drop_all_missing_rows),
            "automatic_model_or_factor_selection": False,
        },
        "shape": {
            "samples": int(V.shape[0]),
            "features": int(V.shape[1]),
            "observed_cells": int(observation_mask.sum()),
            "fit_cells": int(fit_mask.sum()),
            "holdout_cells": int(holdout_mask.sum()),
        },
        "parameters": {
            "static_max_iter": int(args.static_max_iter),
            "init_iter": int(args.init_iter),
            "max_iter": int(args.max_iter),
            "profile_steps": int(args.profile_steps),
            "tol": float(args.tol),
            "v2_variability_rank": int(args.v2_variability_rank),
            "v2_profile_penalty": float(args.v2_profile_penalty),
            "v2_family_penalty": float(args.v2_family_penalty),
            "v2_sv_shrinkage": float(args.v2_sv_shrinkage),
            "variability_threshold": float(args.variability_threshold),
            "missing_sentinel": float(args.missing_sentinel),
            "missing_uncertainty_scale": float(args.missing_uncertainty_scale),
        },
        "cleaning": cleaning,
        "counts": {
            "runs": int(len(runs_df)),
            "stability_rows": int(len(stability_df)),
            "diagnostic_rows": int(len(diagnostics_df)),
            "summary_rows": int(len(summary_df)),
        },
        "outputs": {
            "runs": "k_sweep_runs.csv",
            "stability": "k_sweep_stability.csv",
            "diagnostics": "k_sweep_diagnostics.csv",
            "summary": "k_sweep_summary.csv",
            "holdout_masks": "holdout_masks.npz",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved sweep outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
