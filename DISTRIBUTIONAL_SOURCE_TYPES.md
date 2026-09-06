# Distributional Source-Type Receptor Model

## Research question

Standard ESAT/PMF represents every latent factor with one fixed profile:

\[
V_{tj} \approx \sum_k W_{tk} H_{kj}.
\]

This branch tests a narrower methodological hypothesis:

> A recurrent aerosol source/process **type** may be better represented as a
> stable, data-driven archetype plus bounded occurrence-specific profile
> variability, rather than as either one fixed profile or one serially
> evolving source object.

The proposed V1 model is

\[
V_{tj} \approx \sum_k W_{tk} H_{tkj},
\]

with

\[
H_{tk} \sim \text{family around } \bar H_k,
\]

where both the global archetype \(\bar H_k\) and local profiles \(H_{tk}\) are
learned from the receptor data. No source names or reference source profiles
are required.

## Why this is different from a time-varying profile trajectory

The latent factor index \(k\) is interpreted as a recurring source/process
**type**, not as one aerosol parcel or one emission episode that must persist
from sample \(t-1\) to \(t\). Two traffic-like occurrences can therefore share
the same archetype without implying that they are the same emitted material.

This branch intentionally does **not** impose

\[
H_{t-1,k} \rightarrow H_{t,k}
\]

as the source identity mechanism. Identity is supplied by partial pooling to a
global latent archetype instead.

## V1 objective

The current implementation is a penalized/MAP-like prototype:

\[
\mathcal L =
\sum_{t,j}
\frac{
\left(V_{tj} - \sum_k W_{tk}H_{tkj}\right)^2
}{U_{tj}^2}
+
\sum_{t,k} \lambda_k
\lVert H_{tk}-\bar H_k\rVert_2^2.
\]

Constraints:

- \(W_{tk}\ge 0\)
- \(H_{tkj}\ge 0\)
- \(\sum_j H_{tkj}=1\)
- \(\sum_j \bar H_{kj}=1\)

The second term preserves source-type identity. Large \(\lambda_k\) forces

\[
H_{tk} \approx \bar H_k,
\]

so the model approaches the static-profile special case. Smaller \(\lambda_k\)
allows the data to support more within-type variability.

This is **not yet full Bayesian posterior inference**. The current goal is to
test identifiability and recovery before introducing SVI/NUTS or a more
complex probabilistic backend.

## Files added in this branch

- `esat/model/distributional_sa.py`
  - global source-type archetypes
  - per-sample local profile realizations
  - weighted NNLS contribution updates
  - projected-gradient profile updates
  - shrinkage-to-static penalty
  - profile variability outputs
- `eval/distributional_simulator.py`
  - synthetic global archetypes
  - logistic-normal local profile variability
  - known contributions and measurement uncertainty
- `eval/distributional_recovery.py`
  - factor permutation alignment
  - archetype cosine recovery
  - contribution correlation
  - local-profile RMSE
  - variability recovery
- `eval/distributional_selection.py`
  - common held-out masks
  - masked static LS-NMF baseline
  - weighted training/held-out metrics
- `eval/distributional_stability.py`
  - multi-seed archetype/contribution/variability stability
- `scripts/run_distributional_synthetic.py`
  - one-command controlled experiment
- `scripts/run_distributional_fs0610.py`
  - real-data single-fit runner for V1 or low-rank V2
- `scripts/run_distributional_k_sweep.py`
  - Static-versus-V2 factor/seed selection protocol
- `tests/model/test_distributional_sa.py`
  - static-limit regression test
  - penalty/flexibility behavior
  - result-shape/output test

## Example

```bash
python scripts/run_distributional_synthetic.py \
  --factors 4 \
  --features 12 \
  --samples 300 \
  --variability 0.05,0.10,0.20,0.35 \
  --profile-penalty 1.0 \
  --output output/distributional/synthetic_v1.json
```

The script writes summary recovery metrics and, when `--output` is given, a
compressed NPZ containing both true and estimated latent quantities.

## Interpretation of W

V1 normalizes each profile across the modeled features. Therefore `W[t, k]` is
the scale of factor `k` in the **modeled feature space**. It can only be called
a total source mass when the selected features and units support that physical
mass-balance interpretation.

This restriction is explicit because otherwise normalizing `H` would silently
change the physical meaning of source contributions.

## Current methodological targets

### H0 — static source type

\[
H_{tk}=H_k.
\]

### H1 — recurrent distributional source type

\[
H_{tk}\mid \bar H_k, \Sigma_k.
\]

The immediate question is not whether flexibility always improves fit. It is
whether the model can:

1. shrink to the static solution when profile variability is absent;
2. recover bounded variability when it is present;
3. improve contribution recovery without creating factor splitting/merging;
4. distinguish genuine new factors from excessive within-type flexibility.

## Next implementation stages

### P0 — current branch MVP

- [x] distributional factorization class
- [x] static initialization using ESAT LS-NMF equations
- [x] explicit shrinkage-to-archetype objective
- [x] distributional synthetic generator
- [x] ground-truth recovery metrics
- [x] basic regression tests
- [x] runnable synthetic experiment

### P1 — identifiability experiment suite

Add controlled scenarios:

1. static profiles;
2. bounded IID within-type variability;
3. context-conditioned variability;
4. serially evolving profiles (Heaton-like competitor);
5. truly novel source appears mid-series;
6. highly overlapping archetypes;
7. variable source abundance / low-contribution factors;
8. missing data and uncertainty perturbation.

Primary outputs should be an identifiability phase diagram and factor
split/merge diagnostics, not only reconstruction Q.

### P2 — empirical-Bayes / probabilistic variability

Replace one manually selected global profile penalty with factor-specific
learned variability parameters, while retaining a strong shrink-to-zero prior
so the static model remains nested inside the flexible model.

Candidate parameterization:

\[
\eta_{tk}=\alpha_k + \tau_k L_k z_{tk},
\qquad
H_{tk}=\operatorname{softmax}(\eta_{tk}),
\]

with shrinkage on \(\tau_k\).

### P3 — context-conditioned profile family

Only after P1/P2 are identifiable, model

\[
P(H_{tk}\mid \text{Type}_k, C_t)
\]

where `C_t` may include RH, temperature, radiation, PBLH, etc. Context should
modify a recurring type's profile distribution; it should not define source
labels in advance.

### P4 — real-data FS0610 study

The initial real-data runner and selection protocol now exist in the feature
worktree. `scripts/run_distributional_k_sweep.py` uses one common cleaned
matrix, uncertainty matrix, observed-cell mask, and held-out mask across the
Static and V2 fits. It exports:

- global archetypes;
- local profiles;
- contributions;
- profile SD / RMS variability;
- static-vs-flexible comparison;
- factor stability across random seeds.

The remaining scientific work is to run the full K/seed grid and interpret it;
the runner does not automatically select a winning factor count.

### P5 — full posterior and acceleration

Only after the model is empirically identifiable:

- evaluate SVI/NUTS or another posterior approximation;
- propagate uncertainty for `W`, archetypes, and within-type variability;
- profile bottlenecks;
- then add a dedicated GPU tensor kernel if justified.

The existing `ls_nmf_batched` kernel should remain an optimized static ESAT
baseline rather than being overloaded with a different tensor factorization.
