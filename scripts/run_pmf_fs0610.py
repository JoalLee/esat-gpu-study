#!/usr/bin/env python3
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

from esat.error.bootstrap import Bootstrap
from esat.metrics import q_loss, qr_loss
from esat.model.batch_sa import BatchSA


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


def validate_model(name: str, model) -> None:
    w64 = np.asarray(model.W, dtype=np.float64)
    h64 = np.asarray(model.H, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        wh64 = np.matmul(w64, h64)
    checks = {
        f"{name}.W": w64,
        f"{name}.H": h64,
        f"{name}.WH": wh64,
        f"{name}.Qtrue": np.array([model.Qtrue]),
        f"{name}.Qrobust": np.array([model.Qrobust]),
    }
    for label, value in checks.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"{label} contains non-finite values")


def normalized_best_index(batch: BatchSA) -> int:
    idx = int(batch.best_model)
    if 0 <= idx < len(batch.results):
        return idx
    if 1 <= idx <= len(batch.results):
        return idx - 1
    raise IndexError(f"best_model={batch.best_model} is outside results length {len(batch.results)}")


def read_and_clean_pmf(
    species_path: Path,
    uncertainty_path: Path,
    *,
    time_col: str,
    missing_sentinel: float,
    missing_uncertainty_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Index, dict]:
    species_df = pd.read_csv(species_path)
    uncertainty_df = pd.read_csv(uncertainty_path)

    if time_col not in species_df.columns or time_col not in uncertainty_df.columns:
        raise ValueError(f"time column {time_col!r} must exist in both input files")

    if list(species_df.columns) != list(uncertainty_df.columns):
        raise ValueError("species and uncertainty files must have identical columns in the same order")

    times = pd.to_datetime(species_df[time_col], errors="coerce")
    feature_cols = [c for c in species_df.columns if c != time_col]

    data = species_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    unc = uncertainty_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    sentinel_missing = data <= missing_sentinel
    negative_non_sentinel = (data < 0) & ~sentinel_missing
    data_missing = data.isna() | sentinel_missing | negative_non_sentinel
    invalid_unc = unc.isna() | (unc <= 0)

    positive_data = data.where((~data_missing) & (data > 0))
    data_scale = positive_data.median(axis=0, skipna=True)
    data_scale = data_scale.where(data_scale > 0, np.nan).fillna(1.0)

    valid_unc = unc.where(~invalid_unc)
    unc_scale = valid_unc.median(axis=0, skipna=True)
    unc_scale = unc_scale.where(unc_scale > 0, np.nan).fillna(data_scale)
    unc_scale = unc_scale.where(unc_scale > 0, 1.0)

    missing_unc = np.maximum(missing_uncertainty_scale * data_scale, unc_scale)

    clean_data = data.mask(data_missing, 0.0).clip(lower=0.0)
    clean_unc = unc.copy()
    clean_unc = clean_unc.mask(data_missing, missing_unc, axis=1)
    clean_unc = clean_unc.mask(invalid_unc & ~data_missing, unc_scale, axis=1)
    clean_unc = clean_unc.mask(clean_unc <= 0, unc_scale, axis=1)

    if not np.isfinite(clean_data.to_numpy()).all():
        raise ValueError("cleaned concentration matrix still contains non-finite values")
    if not np.isfinite(clean_unc.to_numpy()).all() or (clean_unc.to_numpy() <= 0).any():
        raise ValueError("cleaned uncertainty matrix still contains non-finite or non-positive values")

    diagnostics = {
        "input_rows": int(data.shape[0]),
        "features": int(data.shape[1]),
        "missing_sentinel": float(missing_sentinel),
        "missing_uncertainty_scale": float(missing_uncertainty_scale),
        "sentinel_missing_cells": int(sentinel_missing.to_numpy().sum()),
        "negative_non_sentinel_cells": int(negative_non_sentinel.to_numpy().sum()),
        "invalid_uncertainty_cells": int(invalid_unc.to_numpy().sum()),
        "all_missing_rows": int(data_missing.all(axis=1).sum()),
        "time_parse_failures": int(times.isna().sum()),
        "clean_data_min": float(clean_data.to_numpy().min()),
        "clean_data_max": float(clean_data.to_numpy().max()),
        "clean_uncertainty_min": float(clean_unc.to_numpy().min()),
        "clean_uncertainty_max": float(clean_unc.to_numpy().max()),
    }

    return (
        clean_data.to_numpy(dtype=np.float32),
        clean_unc.to_numpy(dtype=np.float32),
        feature_cols,
        pd.Index(times, name=time_col),
        diagnostics,
    )


def save_matrix_outputs(output_dir: Path, model, features: list[str], times: pd.Index) -> None:
    profiles = pd.DataFrame(
        model.H,
        index=[f"factor_{i + 1}" for i in range(model.H.shape[0])],
        columns=features,
    )
    contributions = pd.DataFrame(
        model.W,
        index=times,
        columns=[f"factor_{i + 1}" for i in range(model.W.shape[1])],
    )
    profiles.to_csv(output_dir / "profiles_base.csv", index_label="factor")
    contributions.to_csv(output_dir / "contributions_base.csv", index_label=times.name or "time")


def save_all_matrix_outputs(
    output_dir: Path,
    batch: BatchSA,
    features: list[str],
    times: pd.Index,
) -> pd.DataFrame:
    """Save clean profile and contribution matrices for every base model."""
    rows = []
    for model_i, model in enumerate(batch.results):
        if model is None:
            continue

        model_number = model_i + 1
        profiles = pd.DataFrame(
            model.H,
            index=[f"factor_{i + 1}" for i in range(model.H.shape[0])],
            columns=features,
        )
        contributions = pd.DataFrame(
            model.W,
            index=times,
            columns=[f"factor_{i + 1}" for i in range(model.W.shape[1])],
        )
        profiles.to_csv(output_dir / f"profiles_model_{model_number:02d}.csv", index_label="factor")
        contributions.to_csv(
            output_dir / f"contributions_model_{model_number:02d}.csv",
            index_label=times.name or "time",
        )
        rows.append(
            {
                "model": model_number,
                "is_best": model_i == batch.best_model,
                "q_true": float(model.Qtrue),
                "q_robust": float(model.Qrobust),
                "backend": model.metadata.get("backend", "unknown"),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "base_models_summary.csv", index=False)
    return summary


def save_bootstrap_outputs(output_dir: Path, bs: Bootstrap) -> pd.DataFrame:
    rows = []
    for model_i, result in sorted(bs.bs_results.items()):
        model = result["model"]
        rows.append(
            {
                "bootstrap_run": int(model_i),
                "q_true": float(model.Qtrue),
                "q_robust": float(model.Qrobust),
                "unique_resampled_rows": int(len(set(result["index"]))),
                "total_resampled_rows": int(len(result["index"])),
            }
        )
    q_df = pd.DataFrame(rows)
    q_df.to_csv(output_dir / "bootstrap_q.csv", index=False)
    if bs.mapping_df is not None:
        bs.mapping_df.to_csv(output_dir / "bootstrap_mapping.csv", index=False)
    return q_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PMF FS0610 LS-NMF analysis with explicit data cleaning.")
    parser.add_argument("--species", type=Path, default=Path("data/PMF_data/species_FS0610.csv"))
    parser.add_argument("--uncertainty", type=Path, default=Path("data/PMF_data/unc_FS0610.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/pmf_fs0610_clean"))
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--factors", type=int, default=6)
    parser.add_argument("--models", type=int, default=6)
    parser.add_argument("--bootstrap-runs", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--converge-delta", type=float, default=1.0)
    parser.add_argument("--converge-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--missing-sentinel", type=float, default=-900.0)
    parser.add_argument("--missing-uncertainty-scale", type=float, default=4.0)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-all-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save profile and contribution CSV files for every base model.",
    )
    args = parser.parse_args()

    species_path = resolve_path(args.species)
    uncertainty_path = resolve_path(args.uncertainty)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    V, U, features, times, cleaning = read_and_clean_pmf(
        species_path,
        uncertainty_path,
        time_col=args.time_col,
        missing_sentinel=args.missing_sentinel,
        missing_uncertainty_scale=args.missing_uncertainty_scale,
    )

    t0 = time.perf_counter()
    batch = BatchSA(
        V=V,
        U=U,
        models=args.models,
        factors=args.factors,
        method="ls-nmf",
        max_iter=args.max_iter,
        converge_delta=args.converge_delta,
        converge_n=args.converge_n,
        parallel=False,
        seed=args.seed,
        verbose=False,
        use_gpu=args.use_gpu,
    )
    batch.train()
    base_seconds = time.perf_counter() - t0

    best_model_idx = normalized_best_index(batch)
    base_model = batch.results[best_model_idx]
    validate_model("base_model", base_model)
    save_matrix_outputs(output_dir, base_model, features, times)
    all_models_summary = None
    if args.save_all_models:
        all_models_summary = save_all_matrix_outputs(output_dir, batch, features, times)

    bs = None
    bootstrap_seconds = 0.0
    q_df = pd.DataFrame()
    if args.bootstrap_runs > 0:
        t1 = time.perf_counter()
        bs = Bootstrap(
            sa=base_model,
            feature_labels=features,
            model_selected=best_model_idx,
            bootstrap_n=args.bootstrap_runs,
            block_size=args.block_size,
            threshold=args.threshold,
            parallel=False,
            seed=args.seed,
            use_gpu=args.use_gpu,
        )
        bs.run()
        bootstrap_seconds = time.perf_counter() - t1
        for model_i, result in bs.bs_results.items():
            validate_model(f"bootstrap_{model_i}", result["model"])
        q_df = save_bootstrap_outputs(output_dir, bs)

    metadata = {
        "git_commit": git_commit(),
        "species_path": str(species_path),
        "uncertainty_path": str(uncertainty_path),
        "output_dir": str(output_dir),
        "shape": {"samples": int(V.shape[0]), "features": int(V.shape[1]), "factors": int(args.factors)},
        "parameters": {
            "models": int(args.models),
            "bootstrap_runs": int(args.bootstrap_runs),
            "block_size": int(args.block_size),
            "max_iter": int(args.max_iter),
            "converge_delta": float(args.converge_delta),
            "converge_n": int(args.converge_n),
            "seed": int(args.seed),
            "threshold": float(args.threshold),
            "use_gpu": bool(args.use_gpu),
        },
        "cleaning": cleaning,
        "base": {
            "seconds": float(base_seconds),
            "seconds_per_model": float(base_seconds / max(args.models, 1)),
            "best_model": int(best_model_idx),
            "backend": base_model.metadata.get("backend", "unknown"),
            "q_true": float(base_model.Qtrue),
            "q_robust": float(base_model.Qrobust),
            "all_models_saved": bool(args.save_all_models),
        },
        "bootstrap": {
            "seconds": float(bootstrap_seconds),
            "seconds_per_run": float(bootstrap_seconds / max(args.bootstrap_runs, 1)) if args.bootstrap_runs else 0.0,
            "backend": bs.metadata.get("backend", "unknown") if bs is not None else "skipped",
            "q_true_unique_rounded_6": int(q_df["q_true"].round(6).nunique()) if not q_df.empty else 0,
            "q_robust_unique_rounded_6": int(q_df["q_robust"].round(6).nunique()) if not q_df.empty else 0,
            "q_true_min": float(q_df["q_true"].min()) if not q_df.empty else None,
            "q_true_max": float(q_df["q_true"].max()) if not q_df.empty else None,
            "q_robust_min": float(q_df["q_robust"].min()) if not q_df.empty else None,
            "q_robust_max": float(q_df["q_robust"].max()) if not q_df.empty else None,
        },
        "outputs": {
            "profiles_base": "profiles_base.csv",
            "contributions_base": "contributions_base.csv",
            "all_models_summary": "base_models_summary.csv" if all_models_summary is not None else None,
            "all_model_profiles": "profiles_model_XX.csv" if all_models_summary is not None else None,
            "all_model_contributions": "contributions_model_XX.csv" if all_models_summary is not None else None,
            "bootstrap_q": "bootstrap_q.csv" if args.bootstrap_runs else None,
            "bootstrap_mapping": "bootstrap_mapping.csv" if args.bootstrap_runs else None,
        },
    }

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
