#!/usr/bin/env python3
"""Run distributional source-type models on the real FS0610 PMF inputs.

This runner mirrors the cleaning convention used by ``run_pmf_fs0610.py`` but
also passes an explicit observation mask to the distributional models. Missing
cells are therefore excluded from the fit instead of being treated as zero
concentrations with a large uncertainty.

The default inputs are the tracked FS0610 concentration and uncertainty CSVs.
Use ``--max-rows`` for a quick real-data smoke run before fitting all rows.
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

from esat.model.distributional_sa import DistributionalSA
from esat.model.lowrank_distributional_sa import LowRankDistributionalSA


def git_commit() -> str:
    """Return the short commit used for the fit when Git is available."""
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
    """Resolve a relative CLI path against the repository root."""
    return path if path.is_absolute() else ROOT / path


def _select_rows(
    species_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    *,
    start_row: int,
    row_step: int,
    max_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | None]]:
    if start_row < 0:
        raise ValueError("start_row must be >= 0")
    if row_step < 1:
        raise ValueError("row_step must be >= 1")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be >= 1 when provided")

    positions = np.arange(start_row, len(species_df), row_step, dtype=int)
    if max_rows is not None:
        positions = positions[:max_rows]
    if positions.size == 0:
        raise ValueError("row selection produced no samples")

    selected_species = species_df.iloc[positions].copy()
    selected_uncertainty = uncertainty_df.iloc[positions].copy()
    selection = {
        "input_rows": int(len(species_df)),
        "selected_rows": int(len(positions)),
        "start_row": int(start_row),
        "row_step": int(row_step),
        "max_rows": int(max_rows) if max_rows is not None else None,
    }
    return selected_species, selected_uncertainty, selection


def read_and_clean_distributional(
    species_path: Path,
    uncertainty_path: Path,
    *,
    time_col: str,
    missing_sentinel: float,
    missing_uncertainty_scale: float,
    start_row: int = 0,
    row_step: int = 1,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.Index, dict]:
    """Read FS0610-style CSVs and return ``V``, ``U`` and an observation mask.

    The two files must have the same columns and row-wise timestamps. Negative
    values are considered missing, while observed zero concentrations remain
    valid observations. Invalid uncertainties on observed cells are replaced
    by a feature-wise robust scale; uncertainties on missing cells are finite
    placeholders because their corresponding mask entries are false.
    """
    species_df = pd.read_csv(species_path)
    uncertainty_df = pd.read_csv(uncertainty_path)

    if time_col not in species_df.columns or time_col not in uncertainty_df.columns:
        raise ValueError(f"time column {time_col!r} must exist in both input files")
    if list(species_df.columns) != list(uncertainty_df.columns):
        raise ValueError(
            "species and uncertainty files must have identical columns in the same order"
        )
    if len(species_df) != len(uncertainty_df):
        raise ValueError("species and uncertainty files must have the same row count")

    species_times = species_df[time_col].astype(str)
    uncertainty_times = uncertainty_df[time_col].astype(str)
    if not species_times.equals(uncertainty_times):
        raise ValueError("species and uncertainty timestamps are not row-aligned")

    species_df, uncertainty_df, selection = _select_rows(
        species_df,
        uncertainty_df,
        start_row=start_row,
        row_step=row_step,
        max_rows=max_rows,
    )

    times = pd.to_datetime(species_df[time_col], errors="coerce")
    feature_cols = [column for column in species_df.columns if column != time_col]
    if not feature_cols:
        raise ValueError("input files must contain at least one feature column")

    data = species_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    uncertainty = uncertainty_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    sentinel_missing = data <= missing_sentinel
    negative_non_sentinel = (data < 0) & ~sentinel_missing
    data_missing = data.isna() | sentinel_missing | negative_non_sentinel
    invalid_uncertainty = uncertainty.isna() | (uncertainty <= 0)
    observation_mask = (~data_missing).to_numpy(dtype=bool)

    positive_data = data.where((~data_missing) & (data > 0))
    data_scale = positive_data.median(axis=0, skipna=True)
    data_scale = data_scale.where(data_scale > 0, np.nan).fillna(1.0)

    valid_uncertainty = uncertainty.where(~invalid_uncertainty)
    uncertainty_scale = valid_uncertainty.median(axis=0, skipna=True)
    uncertainty_scale = uncertainty_scale.where(
        uncertainty_scale > 0,
        np.nan,
    ).fillna(data_scale)
    uncertainty_scale = uncertainty_scale.where(uncertainty_scale > 0, 1.0)
    missing_uncertainty = np.maximum(
        missing_uncertainty_scale * data_scale,
        uncertainty_scale,
    )

    clean_data = data.mask(data_missing, 0.0).clip(lower=0.0)
    clean_uncertainty = uncertainty.copy()
    clean_uncertainty = clean_uncertainty.mask(
        data_missing,
        missing_uncertainty,
        axis=1,
    )
    clean_uncertainty = clean_uncertainty.mask(
        invalid_uncertainty & ~data_missing,
        uncertainty_scale,
        axis=1,
    )
    clean_uncertainty = clean_uncertainty.mask(
        clean_uncertainty <= 0,
        uncertainty_scale,
        axis=1,
    )

    clean_data_array = clean_data.to_numpy(dtype=np.float64)
    clean_uncertainty_array = clean_uncertainty.to_numpy(dtype=np.float64)
    if not np.isfinite(clean_data_array).all():
        raise ValueError("cleaned concentration matrix contains non-finite values")
    if not np.isfinite(clean_uncertainty_array).all() or np.any(
        clean_uncertainty_array <= 0
    ):
        raise ValueError(
            "cleaned uncertainty matrix contains non-finite or non-positive values"
        )

    diagnostics = {
        **selection,
        "features": int(len(feature_cols)),
        "missing_sentinel": float(missing_sentinel),
        "missing_uncertainty_scale": float(missing_uncertainty_scale),
        "sentinel_missing_cells": int(sentinel_missing.to_numpy().sum()),
        "negative_non_sentinel_cells": int(negative_non_sentinel.to_numpy().sum()),
        "invalid_uncertainty_cells": int(invalid_uncertainty.to_numpy().sum()),
        "observed_cells": int(observation_mask.sum()),
        "missing_cells": int((~observation_mask).sum()),
        "observed_fraction": float(observation_mask.mean()),
        "all_missing_rows": int((~observation_mask).all(axis=1).sum()),
        "all_missing_features": int((~observation_mask).all(axis=0).sum()),
        "time_parse_failures": int(times.isna().sum()),
        "clean_data_min": float(clean_data_array.min()),
        "clean_data_max": float(clean_data_array.max()),
        "clean_uncertainty_min": float(clean_uncertainty_array.min()),
        "clean_uncertainty_max": float(clean_uncertainty_array.max()),
    }

    return (
        clean_data_array,
        clean_uncertainty_array,
        observation_mask,
        feature_cols,
        pd.Index(times, name=time_col),
        diagnostics,
    )


def validate_model(model, observation_mask: np.ndarray) -> None:
    """Check finite outputs and the profile simplex constraints."""
    arrays = {
        "W": model.W,
        "H_bar": model.H_bar,
        "H_local": model.H_local,
        "reconstruction": model.reconstruction,
        "q_true": np.asarray([model.q_true]),
        "objective": np.asarray([model.objective]),
    }
    for label, value in arrays.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"model output {label} contains non-finite values")

    if np.any(model.W < -1e-10) or np.any(model.H_bar < -1e-10):
        raise ValueError("model produced negative contributions or archetypes")
    if np.any(model.H_local < -1e-10):
        raise ValueError("model produced negative local profiles")
    np.testing.assert_allclose(model.H_bar.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(model.H_local.sum(axis=2), 1.0, atol=1e-8)

    if model.reconstruction.shape != observation_mask.shape:
        raise ValueError("reconstruction shape does not match the input matrix")


def _indexed_frame(
    values: np.ndarray,
    index: pd.Index,
    columns: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=columns)


def save_outputs(
    output_dir: Path,
    model,
    *,
    features: list[str],
    times: pd.Index,
    observation_mask: np.ndarray,
) -> dict[str, str]:
    """Write human-readable CSVs and a complete compressed array snapshot."""
    factor_labels = [f"factor_{i + 1}" for i in range(model.H_bar.shape[0])]
    index_label = times.name or "time"

    pd.DataFrame(model.H_bar, index=factor_labels, columns=features).to_csv(
        output_dir / "archetypes.csv",
        index_label="factor",
    )
    _indexed_frame(model.W, times, factor_labels).to_csv(
        output_dir / "contributions.csv",
        index_label=index_label,
    )
    pd.DataFrame(model.profile_sd, index=factor_labels, columns=features).to_csv(
        output_dir / "profile_sd.csv",
        index_label="factor",
    )
    pd.DataFrame(
        {
            "factor": factor_labels,
            "profile_rms_variability": model.profile_rms_variability,
        }
    ).to_csv(output_dir / "profile_variability.csv", index=False)
    _indexed_frame(model.reconstruction, times, features).to_csv(
        output_dir / "reconstruction.csv",
        index_label=index_label,
    )
    _indexed_frame(
        observation_mask.astype(np.int8),
        times,
        features,
    ).to_csv(output_dir / "observation_mask.csv", index_label=index_label)

    arrays = {
        "archetypes": np.asarray(model.H_bar),
        "contributions": np.asarray(model.W),
        "local_profiles": np.asarray(model.H_local),
        "reconstruction": np.asarray(model.reconstruction),
        "profile_sd": np.asarray(model.profile_sd),
        "profile_rms_variability": np.asarray(model.profile_rms_variability),
        "observation_mask": np.asarray(observation_mask),
        "features": np.asarray(features, dtype=str),
        "times": np.asarray(times.astype(str), dtype=str),
    }
    optional_arrays = (
        "loadings",
        "scores",
        "latent_tau",
        "effective_rank",
        "singular_values_raw",
        "singular_values_shrunk",
    )
    for name in optional_arrays:
        value = getattr(model, name, None)
        if value is not None:
            arrays[name] = np.asarray(value)
    np.savez_compressed(output_dir / "distributional_arrays.npz", **arrays)

    outputs = {
        "archetypes": "archetypes.csv",
        "contributions": "contributions.csv",
        "profile_sd": "profile_sd.csv",
        "profile_variability": "profile_variability.csv",
        "reconstruction": "reconstruction.csv",
        "observation_mask": "observation_mask.csv",
        "arrays": "distributional_arrays.npz",
    }
    if getattr(model, "effective_rank", None) is not None:
        pd.DataFrame(
            {
                "factor": factor_labels,
                "effective_rank": model.effective_rank,
                "latent_tau": model.latent_tau,
            }
        ).to_csv(output_dir / "lowrank_summary.csv", index=False)
        outputs["lowrank_summary"] = "lowrank_summary.csv"
    return outputs


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
        description="Fit distributional source-type models to real FS0610 PMF data."
    )
    parser.add_argument(
        "--model",
        choices=("v1", "lowrank"),
        default="v1",
        help="v1 unrestricted local profiles or lowrank V2 profile families",
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
        default=Path("output/distributional_fs0610"),
    )
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--factors", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-penalty", type=float, default=None)
    parser.add_argument("--family-penalty", type=float, default=0.0)
    parser.add_argument("--variability-rank", type=int, default=2)
    parser.add_argument("--sv-shrinkage", type=float, default=0.5)
    parser.add_argument("--init-iter", type=int, default=250)
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--missing-sentinel", type=float, default=-900.0)
    parser.add_argument("--missing-uncertainty-scale", type=float, default=4.0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--row-step", type=int, default=1)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit selected rows for a smoke run; default uses all input rows.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args()

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

    if args.profile_penalty is None:
        profile_penalty = 1.0 if args.model == "v1" else 0.005
    else:
        profile_penalty = float(args.profile_penalty)

    fit_kwargs = {
        "V": V,
        "U": U,
        "factors": args.factors,
        "observation_mask": observation_mask,
        "profile_penalty": profile_penalty,
        "seed": args.seed,
        "init_iter": args.init_iter,
        "max_iter": args.max_iter,
        "profile_steps": args.profile_steps,
        "tol": args.tol,
    }
    if args.model == "v1":
        model = DistributionalSA(**fit_kwargs)
    else:
        model = LowRankDistributionalSA(
            **fit_kwargs,
            variability_rank=args.variability_rank,
            sv_shrinkage=args.sv_shrinkage,
            family_penalty=args.family_penalty,
        )

    started = time.perf_counter()
    model.fit()
    fit_seconds = time.perf_counter() - started
    validate_model(model, observation_mask)

    prepare_output_dir(output_dir, args.overwrite)
    outputs = save_outputs(
        output_dir,
        model,
        features=features,
        times=times,
        observation_mask=observation_mask,
    )

    metadata = {
        "git_commit": git_commit(),
        "model": args.model,
        "species_path": str(species_path),
        "uncertainty_path": str(uncertainty_path),
        "output_dir": str(output_dir),
        "shape": {
            "samples": int(V.shape[0]),
            "features": int(V.shape[1]),
            "factors": int(args.factors),
        },
        "parameters": {
            "seed": int(args.seed),
            "profile_penalty": float(profile_penalty),
            "family_penalty": float(args.family_penalty),
            "variability_rank": int(args.variability_rank),
            "sv_shrinkage": float(args.sv_shrinkage),
            "init_iter": int(args.init_iter),
            "max_iter": int(args.max_iter),
            "profile_steps": int(args.profile_steps),
            "tol": float(args.tol),
        },
        "cleaning": cleaning,
        "fit": {
            "seconds": float(fit_seconds),
            "objective_initial": float(model.objective_history[0]),
            "objective_final": float(model.objective),
            "q_true": float(model.q_true),
            "profile_penalty_loss": float(model.profile_penalty_loss),
            "iterations": int(model.iterations),
            "converged": bool(model.converged),
            "profile_rms_variability": model.profile_rms_variability.tolist(),
        },
        "outputs": outputs,
    }
    if getattr(model, "latent_tau", None) is not None:
        metadata["fit"]["latent_tau"] = model.latent_tau.tolist()
        metadata["fit"]["effective_rank"] = model.effective_rank.tolist()

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
