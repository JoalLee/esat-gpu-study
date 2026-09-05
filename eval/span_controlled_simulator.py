"""Synthetic source-type data with controlled W/F identifiability geometry.

The central ambiguity in a flexible receptor model is that a local change in a
factor profile can sometimes be reproduced by changing factor contributions.
For normalized profiles F_k, the contribution-only tangent space is spanned by
profile differences::

    S = span{F_2 - F_1, ..., F_K - F_1}.

A profile-deviation direction lying in S is maximally confounded with changes
in W. A direction orthogonal to S (while remaining in the simplex tangent
space) cannot be reproduced locally by merely redistributing W among the
static archetypes.

The simulator independently controls two geometries:

* ``alignment``: overlap of profile variability with the static factor span;
* ``source_overlap``: commonality among the global source archetypes.

This allows the phase diagram to distinguish W/F confounding from the more
familiar problem of poorly separated source types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_EPS = 1e-12


def _orthonormal_rows(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if x.shape[0] == 0 or not np.any(np.abs(x) > tol):
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    if s.size == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float64)
    rank = int(np.sum(s > max(tol, tol * float(s[0]))))
    return vt[:rank]


def _project_out(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    if basis.shape[0] == 0:
        return v.copy()
    return v - basis.T @ (basis @ v)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        raise ValueError("cannot normalize a near-zero vector")
    return vector / norm


def _static_confounding_basis(archetypes: np.ndarray) -> np.ndarray:
    """Basis for simplex-tangent directions reproducible by changing W."""
    archetypes = np.asarray(archetypes, dtype=np.float64)
    if archetypes.ndim != 2:
        raise ValueError("archetypes must be K x J")
    if archetypes.shape[0] <= 1:
        return np.zeros((0, archetypes.shape[1]), dtype=np.float64)
    differences = archetypes[1:] - archetypes[0]
    return _orthonormal_rows(differences)


def _tangent_orthogonal_direction(
    rng: np.random.Generator,
    confounding_basis: np.ndarray,
    features: int,
) -> np.ndarray:
    """Random unit direction orthogonal to ones and confounding subspace."""
    one_basis = np.ones((1, features), dtype=np.float64) / np.sqrt(features)
    forbidden = np.concatenate([one_basis, confounding_basis], axis=0)
    forbidden = _orthonormal_rows(forbidden)

    for _ in range(100):
        candidate = rng.normal(size=features)
        candidate = _project_out(candidate, forbidden)
        if np.linalg.norm(candidate) > 1e-8:
            return _unit(candidate)
    raise ValueError(
        "no identifiable tangent direction available; increase features or reduce factors"
    )


def _random_in_span(
    rng: np.random.Generator,
    basis: np.ndarray,
) -> np.ndarray:
    if basis.shape[0] == 0:
        raise ValueError("confounding span is empty")
    coefficients = rng.normal(size=basis.shape[0])
    return _unit(coefficients @ basis)


def _direction_with_alignment(
    rng: np.random.Generator,
    confounding_basis: np.ndarray,
    features: int,
    alignment: float,
) -> np.ndarray:
    """Construct unit tangent direction with requested squared span overlap."""
    alignment = float(alignment)
    if not 0.0 <= alignment <= 1.0:
        raise ValueError("alignment must be in [0, 1]")

    orthogonal = _tangent_orthogonal_direction(
        rng,
        confounding_basis=confounding_basis,
        features=features,
    )
    if alignment <= _EPS:
        return orthogonal

    in_span = _random_in_span(rng, confounding_basis)
    direction = np.sqrt(alignment) * in_span + np.sqrt(1.0 - alignment) * orthogonal
    direction -= np.mean(direction)
    return _unit(direction)


def _squared_span_overlap(direction: np.ndarray, basis: np.ndarray) -> float:
    if basis.shape[0] == 0:
        return 0.0
    direction = _unit(direction)
    projection = basis.T @ (basis @ direction)
    return float(np.clip(np.dot(projection, projection), 0.0, 1.0))


def _safe_additive_amplitude(profile: np.ndarray, direction: np.ndarray) -> float:
    """Largest symmetric additive step keeping profile non-negative."""
    profile = np.asarray(profile, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    nonzero = np.abs(direction) > 1e-12
    if not np.any(nonzero):
        return 0.0
    bounds = profile[nonzero] / np.abs(direction[nonzero])
    return 0.80 * float(np.min(bounds))


def _mean_pairwise_cosine(profiles: np.ndarray) -> float:
    profiles = np.asarray(profiles, dtype=np.float64)
    values: list[float] = []
    for i in range(profiles.shape[0]):
        for j in range(i + 1, profiles.shape[0]):
            denom = max(
                float(np.linalg.norm(profiles[i]) * np.linalg.norm(profiles[j])),
                _EPS,
            )
            values.append(float(np.dot(profiles[i], profiles[j]) / denom))
    return float(np.mean(values)) if values else 1.0


@dataclass(frozen=True)
class SpanControlledSyntheticData:
    data: np.ndarray
    uncertainty: np.ndarray
    signal: np.ndarray
    archetypes: np.ndarray
    local_profiles: np.ndarray
    contributions: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    requested_alignment: float
    actual_alignment: np.ndarray
    confounding_basis: np.ndarray
    requested_variability: np.ndarray
    actual_profile_rms_variability: np.ndarray
    requested_source_overlap: float
    pairwise_archetype_cosine_mean: float


class SpanControlledSimulator:
    """Generate receptor data across controlled profile-identifiability axes.

    Parameters
    ----------
    alignment
        Requested squared projection of every variable profile direction onto
        the static contribution-confounding span. ``0`` is orthogonal and
        maximally distinguishable from W redistribution; ``1`` is entirely in
        that span and maximally confounded.
    variability
        Fraction (0..1 recommended) of each factor's positivity-safe additive
        amplitude. A scalar applies to all factors. Zero gives a static factor.
    source_overlap
        Convex mixing fraction of a common profile into every archetype.
        ``0`` leaves independently drawn profiles; values approaching ``1``
        make source types increasingly similar. The realized mean pairwise
        cosine is returned because the mixing fraction itself is not a cosine.
    archetype_concentration
        Dirichlet concentration used for positive archetypes.
    """

    def __init__(
        self,
        seed: int = 42,
        factors_n: int = 3,
        features_n: int = 10,
        samples_n: int = 200,
        alignment: float = 0.0,
        variability: float | Iterable[float] = 0.4,
        source_overlap: float = 0.0,
        contribution_max: float = 10.0,
        noise_fraction: float = 0.03,
        uncertainty_floor: float = 0.01,
        archetype_concentration: float = 3.0,
    ) -> None:
        self.seed = int(seed)
        self.factors_n = int(factors_n)
        self.features_n = int(features_n)
        self.samples_n = int(samples_n)
        self.alignment = float(alignment)
        self.source_overlap = float(source_overlap)
        self.contribution_max = float(contribution_max)
        self.noise_fraction = float(noise_fraction)
        self.uncertainty_floor = float(uncertainty_floor)
        self.archetype_concentration = float(archetype_concentration)

        if self.factors_n < 2:
            raise ValueError("span-controlled experiments require at least 2 factors")
        if self.features_n <= self.factors_n:
            raise ValueError(
                "features_n must exceed factors_n to leave an identifiable tangent complement"
            )
        if self.samples_n < 3:
            raise ValueError("samples_n must be >= 3")
        if not 0.0 <= self.alignment <= 1.0:
            raise ValueError("alignment must be in [0, 1]")
        if not 0.0 <= self.source_overlap < 1.0:
            raise ValueError("source_overlap must be in [0, 1)")
        if self.contribution_max <= 0.0:
            raise ValueError("contribution_max must be > 0")
        if self.noise_fraction < 0.0 or self.uncertainty_floor <= 0.0:
            raise ValueError("invalid noise/uncertainty settings")
        if self.archetype_concentration <= 0.0:
            raise ValueError("archetype_concentration must be > 0")

        raw_variability = np.asarray(variability, dtype=np.float64)
        if raw_variability.ndim == 0:
            raw_variability = np.full(self.factors_n, float(raw_variability))
        if raw_variability.shape != (self.factors_n,):
            raise ValueError("variability must be scalar or length factors_n")
        if np.any(raw_variability < 0.0):
            raise ValueError("variability cannot be negative")
        self.variability = raw_variability
        self.rng = np.random.default_rng(self.seed)

    def _generate_archetypes(self) -> np.ndarray:
        concentration = np.full(self.features_n, self.archetype_concentration)
        unique = self.rng.dirichlet(concentration, size=self.factors_n)
        if self.source_overlap <= 0.0:
            return unique
        common = self.rng.dirichlet(concentration)
        profiles = (
            (1.0 - self.source_overlap) * unique
            + self.source_overlap * common[None, :]
        )
        profiles /= profiles.sum(axis=1, keepdims=True)
        return profiles

    def _generate_contributions(self) -> np.ndarray:
        base = self.rng.gamma(2.0, 1.0, size=(self.samples_n, self.factors_n))
        time = np.linspace(0.0, 4.0 * np.pi, self.samples_n)
        for k in range(self.factors_n):
            phase = 2.0 * np.pi * k / self.factors_n
            base[:, k] *= 0.65 + 0.35 * (np.sin(time + phase) + 1.0) / 2.0
        scale = self.contribution_max / max(float(np.percentile(base, 95)), _EPS)
        return np.maximum(base * scale, _EPS)

    def _generate_profiles(
        self,
        archetypes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        basis = _static_confounding_basis(archetypes)
        if basis.shape[0] >= self.features_n - 1:
            raise ValueError("static factor span fills simplex tangent; no orthogonal axis remains")

        loadings = np.zeros((self.factors_n, 1, self.features_n), dtype=np.float64)
        scores = np.zeros((self.samples_n, self.factors_n, 1), dtype=np.float64)
        local = np.empty(
            (self.samples_n, self.factors_n, self.features_n), dtype=np.float64
        )
        actual_alignment = np.zeros(self.factors_n, dtype=np.float64)

        raw_scores = np.tanh(self.rng.normal(size=(self.samples_n, self.factors_n)))
        raw_scores -= raw_scores.mean(axis=0, keepdims=True)
        max_abs = np.maximum(np.max(np.abs(raw_scores), axis=0), _EPS)
        raw_scores /= max_abs[None, :]

        for k in range(self.factors_n):
            direction = _direction_with_alignment(
                self.rng, basis, self.features_n, self.alignment
            )
            loadings[k, 0] = direction
            actual_alignment[k] = _squared_span_overlap(direction, basis)

            if self.variability[k] <= 0.0:
                local[:, k, :] = archetypes[k]
                continue

            safe = _safe_additive_amplitude(archetypes[k], direction)
            amplitude = min(float(self.variability[k]), 1.0) * safe
            factor_scores = amplitude * raw_scores[:, k]
            candidate = archetypes[k][None, :] + factor_scores[:, None] * direction[None, :]
            if np.min(candidate) < -1e-10:
                raise RuntimeError("positivity-safe profile construction failed")
            candidate = np.maximum(candidate, _EPS)
            candidate /= candidate.sum(axis=1, keepdims=True)

            scores[:, k, 0] = factor_scores
            local[:, k, :] = candidate

        actual_variability = np.sqrt(
            np.mean((local - archetypes[None, :, :]) ** 2, axis=(0, 2))
        )
        return local, loadings, scores, actual_alignment, basis

    def generate(self) -> SpanControlledSyntheticData:
        archetypes = self._generate_archetypes()
        contributions = self._generate_contributions()
        local, loadings, scores, actual_alignment, basis = self._generate_profiles(archetypes)

        signal = np.einsum("tk,tkj->tj", contributions, local)
        uncertainty = np.maximum(
            self.noise_fraction * np.maximum(signal, 0.0),
            self.uncertainty_floor,
        )
        data = np.maximum(signal + self.rng.normal(0.0, uncertainty), 0.0)
        actual_variability = np.sqrt(
            np.mean((local - archetypes[None, :, :]) ** 2, axis=(0, 2))
        )

        return SpanControlledSyntheticData(
            data=data,
            uncertainty=uncertainty,
            signal=signal,
            archetypes=archetypes,
            local_profiles=local,
            contributions=contributions,
            loadings=loadings,
            scores=scores,
            requested_alignment=self.alignment,
            actual_alignment=actual_alignment,
            confounding_basis=basis,
            requested_variability=self.variability.copy(),
            actual_profile_rms_variability=actual_variability,
            requested_source_overlap=self.source_overlap,
            pairwise_archetype_cosine_mean=_mean_pairwise_cosine(archetypes),
        )
