# W–F Geometric Identifiability in Distributional Source Types

## Why this note exists

Allowing a recurrent source/process type to have occurrence-specific profiles
creates a new inverse problem beyond standard PMF. The flexible model is

\[
V_t = \sum_{k=1}^K W_{tk} F_{tk} + \epsilon_t,
\]

whereas the static PMF model is

\[
V_t = \sum_{k=1}^K W_{tk} \bar F_k + \epsilon_t.
\]

The central question is not simply whether a flexible model can reduce Q. It
is whether a change in \(F_{tk}\) can be distinguished from a redistribution
of \(W_t\) among the static factor archetypes.

## Local confounding argument

For a variable factor \(k\), write

\[
F_{tk}=\bar F_k + z_{tk}\ell_k,
\]

with

\[
\mathbf 1^T\ell_k=0
\]

so the perturbation remains in the simplex tangent space. The observation
contains the extra term

\[
W_{tk}z_{tk}\ell_k.
\]

Because all normalized archetypes satisfy \(\mathbf 1^T\bar F_k=1\), changing
factor contributions while preserving total scale produces tangent directions
spanned by archetype differences:

\[
\mathcal S
=
\operatorname{span}
\{\bar F_2-\bar F_1,\ldots,\bar F_K-\bar F_1\}.
\]

If

\[
\ell_k \in \mathcal S,
\]

then at least locally the profile-deviation signal can be represented by
changing the mixture of static archetypes. In that regime, the data may not
contain enough information to decide whether \(W\) changed or the profile of
factor \(k\) changed.

If instead

\[
\ell_k \perp \mathcal S
\]

within the simplex tangent space, the profile change creates a direction that
cannot be reproduced by merely redistributing static factor contributions.
This should be a substantially more identifiable regime.

## Proposed identifiability coordinate

Define

\[
a_k
=
\|P_{\mathcal S}\ell_k\|_2^2,
\qquad \|\ell_k\|_2=1.
\]

Then

- \(a_k=0\): profile variation is orthogonal to the static factor span;
- \(a_k=1\): profile variation lies entirely in the static factor span;
- intermediate values continuously interpolate between those regimes.

This is now an explicit simulation axis rather than an informal explanation.

## Implemented experiment

`eval/span_controlled_simulator.py` generates positive simplex-valued source
profiles with controlled \(a_k\). It uses an additive perturbation rather than
a logit-space perturbation because the linear geometry is directly visible:

\[
F_{tk}=\bar F_k+s_{tk}\ell_k.
\]

The score amplitude is bounded by the largest symmetric step that preserves
non-negativity.

`eval/span_identifiability.py` fits the same generated dataset with:

1. static weighted LS-NMF;
2. V1 unrestricted distributional profiles;
3. V2 low-rank distributional profiles.

Primary outputs include:

- archetype recovery;
- contribution correlation and relative error;
- local-profile RMSE;
- variability recovery ratio;
- V2 variability-subspace overlap;
- effective rank;
- Q.

## Phase-diagram hypothesis

The main candidate phase diagram is

\[
\text{span alignment}
\times
\text{profile-variability magnitude}
\times
\text{measurement noise}
\times
\text{source separation}.
\]

Expected qualitative regimes:

### Low alignment, sufficient signal

Profile variability should be identifiable because the observation moves in a
direction unavailable to the static factor mixture. V2 should have a chance to
recover both variability magnitude and the true profile subspace.

### High alignment

Profile variability becomes observationally confounded with changes in W.
Good reconstruction alone is not evidence of correct profile recovery. A
method may preserve excellent Q and archetype recovery while assigning the
variation to the wrong factor or to W.

### Low signal-to-noise

Even orthogonal variation should collapse toward the static model when the
profile-deviation signal is below measurement uncertainty.

## Pilot evidence

A first multi-seed pilot was run through
`.github/workflows/span-identifiability-pilot.yml` using:

- 3 factors and 9 features;
- 100 samples;
- alignments \(0, 0.25, 0.5, 0.75, 1\);
- requested variability fractions 0.4 and 0.7;
- relative noise levels 0.01 and 0.03;
- seeds 7, 11, and 17;
- static LS-NMF, unrestricted V1, and low-rank V2.

The pilot supports the geometric hypothesis. For V2, mean recovery of the true
profile-variation subspace was highest in the orthogonal regime and degraded
strongly as the direction entered the static factor span:

| Span alignment | V2 subspace-overlap range across pilot settings |
| ---: | ---: |
| 0.00 | 0.61–0.72 |
| 0.25 | 0.22–0.26 |
| 0.50 | 0.10–0.13 |
| 0.75 | 0.04–0.05 |
| 1.00 | 0.14–0.21 |

The alignment-1 endpoint shows a modest rebound rather than a perfectly
monotonic continuation. This should be treated as an empirical feature to
investigate, not smoothed away. Even with that rebound, recovery remains far
below the alignment-0 regime.

The recovered profile-variability magnitude showed the same broad loss of
identifiability. At alignment 0, V2 recovered roughly 0.40–0.41 of the true
profile-space RMS variability in this pilot. At alignment 1, recovery ranged
roughly from 0.04 to 0.19 depending on signal magnitude and noise.

The same flexible model retained high contribution correlations throughout the
pilot (approximately 0.94–0.97 for V2), which is itself an important warning:

> good recovery of W or a low reconstruction Q does not imply that the model
> has correctly identified the source-profile variability mechanism.

Static LS-NMF, by construction, recovered zero profile variability and showed
larger contribution distortion than V2 in the pilot. V1 generally improved W
and local reconstruction relative to the static model but, because it has no
explicit low-rank subspace, it cannot directly validate whether a recurring
profile-variation direction was recovered.

These are pilot results, not final performance claims. The next phase should
increase seeds and explicitly vary source-profile separation in addition to
alignment, variability magnitude, and noise.

## Important interpretation

Failure to recover profile variability in a high-alignment regime is not
necessarily a model failure. It can be a property of the inverse problem.
Likewise, a model that always reports flexible profiles is undesirable: it
would be inventing source-type heterogeneity where the receptor data cannot
identify it.

The methodological goal should therefore be stated as:

> infer within-type source-profile variability only inside an empirically
> identifiable operating region, and explicitly diagnose regimes where W and
> F variability are not distinguishable from the receptor data.

This criterion is stronger than simply proposing a more flexible PMF.
