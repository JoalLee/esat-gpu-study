# Real-Data Distributional Model Selection

This document defines the first real-data comparison protocol for FS0610. It
is deliberately a selection and audit workflow, not an automatic claim that a
distributional model is physically correct.

## Implemented protocol

`scripts/run_distributional_k_sweep.py` compares:

- static weighted ESAT LS-NMF;
- low-rank distributional V2 (`LowRankDistributionalSA`);
- factor counts supplied by `--factors`;
- random seeds supplied by `--seeds`.

Both models use the same cleaned concentration matrix, uncertainty matrix,
feature columns, and observed-cell mask. One deterministic cell-level
holdout mask is generated and reused for every factor count, model, and seed.
Rows with no observed cells are dropped by default because they contain no
information for either model; this is recorded in `metadata.json`.

The workflow reports, without selecting a winner automatically:

- training weighted Q and held-out weighted Q;
- Q per evaluated cell;
- archetype pairwise cosine separation and difference-span conditioning;
- factor-permutation-aligned multi-seed stability;
- V2 effective rank and profile variability;
- provisional W/F geometric-confounding labels.

The labels from `eval/identifiability_diagnostics.py` are operational flags,
not calibrated probabilities or significance tests. Continuous metrics must be
reported with them.

## Smoke run

Use a small real-data subset first:

```bash
python scripts/run_distributional_k_sweep.py \
  --models static,lowrank \
  --factors 2,3 \
  --seeds 7,11 \
  --start-row 10 \
  --max-rows 80 \
  --holdout-fraction 0.10 \
  --static-max-iter 50 \
  --init-iter 20 \
  --max-iter 3 \
  --profile-steps 2 \
  --output-dir /private/tmp/esat-gpu-study-k-sweep-smoke
```

## FS0610 selection run

The proposed first study is:

```bash
python scripts/run_distributional_k_sweep.py \
  --models static,lowrank \
  --factors 4,5,6,7,8 \
  --seeds 11,17,23,31,41,53,67,79,97,113 \
  --holdout-fraction 0.10 \
  --output-dir output/distributional_fs0610_k_sweep
```

This uses the full selected FS0610 data by default. The run can be expensive;
`--max-rows` is a smoke-test control, not a scientifically equivalent
substitute for the full dataset.

## Output files

- `k_sweep_runs.csv`: one row per model, factor count, and seed;
- `k_sweep_stability.csv`: each fit aligned to the lowest-training-Q reference
  within its model and factor count;
- `k_sweep_diagnostics.csv`: factor-level separation and W/F geometry fields;
- `k_sweep_summary.csv`: descriptive aggregation by model and factor count;
- `holdout_masks.npz`: the exact observed, training, and held-out masks;
- `metadata.json`: paths, parameters, row filtering, and protocol declaration.

The reference fit used by the stability utility is a bookkeeping convention;
minimum Q is not treated as evidence of physical correctness. A final choice
of K must consider held-out error, stability, separation, confounding, residual
structure, and scientific interpretability together.

## Explicit non-goals

This protocol does not yet implement a serial dynamic-profile competitor, a
novel-source or low-abundance synthetic scenario, context-conditioned
meteorology, Bayesian posterior inference, or GPU acceleration. Those remain
later stages after the real-data selection evidence is established.
