"""Synthetic data generator for distributional source-type experiments.

The existing ESAT simulator assumes one fixed profile per factor.  This module
adds controlled ground-truth scenarios for testing whether a source/process
type is better represented as a fixed profile or as a family of related local
profiles around a shared archetype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_EPS = 1e-12


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


@dataclass(frozen=True)
class DistributionalSyntheticData:
    """Ground-truth synthetic receptor dataset."""

    data: np.ndarray
    uncertainty: np.ndarray
    signal: np.ndarray
    archetypes: np.ndarray
    local_profiles: np.ndarray
    contributions: np.ndarray
    variability: np.ndarray


class DistributionalSimulator:
    """Generate static or distributional latent source-type datasets.

    Parameters
    ----------
    seed
        Random seed.
    factors_n
        Number of latent source/process types.
    features_n
        Number of modeled receptor features.
    samples_n
        Number of observations.
    variability
        Logistic-normal profile variability per factor.  A scalar is applied
        to all factors.  ``0`` produces the static-profile special case.
    contribution_max
        Approximate scale of source contributions.
    noise_fraction
        Relative observation noise standard deviation.
    uncertainty_floor
        Minimum absolute measurement uncertainty.
    """

    def __init__(
        self,
        seed: int = 42,
        factors_n: int = 4,
        features_n: int = 12,
        samples_n: int = 500,
        variability: float | Iterable[float] = 0.15,
        contribution_max: float = 10.0,
        noise_fraction: float = 0.05,
        uncertainty_floor: float = 0.01,
    ) -> None:
        self.seed = int(seed)
        self.factors_n = int(factors_n)
        self.features_n = int(features_n)
        self.samples_n = int(samples_n)
        self.contribution_max = float(contribution_max)
        self.noise_fraction = float(noise_fraction)
        self.uncertainty_floor = float(uncertainty_floor)

        if self.factors_n < 1 or self.features_n < 2 or self.samples_n < 2:
            raise ValueError("factors_n >= 1, features_n >= 2, samples_n >= 2 required")
        if self.contribution_max <= 0.0:
            raise ValueError("contribution_max must be > 0")
        if self.noise_fraction < 0.0 or self.uncertainty_floor <= 0.0:
            raise ValueError("noise_fraction must be >= 0 and uncertainty_floor > 0")

        raw_variability = np.asarray(variability, dtype=np.float64)
        if raw_variability.ndim == 0:
            raw_variability = np.full(self.factors_n, float(raw_variability))
        if raw_variability.shape != (self.factors_n,):
            raise ValueError("variability must be a scalar or one value per factor")
        if np.any(raw_variability < 0.0):
            raise ValueError("variability cannot contain negative values")
        self.variability = raw_variability

        self.rng = np.random.default_rng(self.seed)

    def _generate_archetypes(self) -> np.ndarray:
        # Sparse-ish Dirichlet draws yield source profiles with recognizable
        # marker structure without manually predefining source identities.
        concentration = np.full(self.features_n, 0.5)
        return self.rng.dirichlet(concentration, size=self.factors_n)

    def _generate_contributions(self) -> np.ndarray:
        # Positive contributions with both smooth periodic structure and
        # stochastic occurrence-to-occurrence variation.
        base = self.rng.gamma(shape=2.0, scale=1.0, size=(self.samples_n, self.factors_n))
        time = np.linspace(0.0, 2.0 * np.pi, self.samples_n)
        for k in range(self.factors_n):
            phase = 2.0 * np.pi * k / max(self.factors_n, 1)
            periodic = 0.55 + 0.45 * (np.sin(time + phase) + 1.0) / 2.0
            base[:, k] *= periodic
        scale = self.contribution_max / max(float(np.percentile(base, 95)), _EPS)
        return np.maximum(base * scale, _EPS)

    def _generate_local_profiles(self, archetypes: np.ndarray) -> np.ndarray:
        local = np.empty((self.samples_n, self.factors_n, self.features_n), dtype=np.float64)
        log_archetypes = np.log(np.maximum(archetypes, _EPS))
        for t in range(self.samples_n):
            for k in range(self.factors_n):
                if self.variability[k] <= 0.0:
                    local[t, k] = archetypes[k]
                    continue
                deviation = self.rng.normal(
                    loc=0.0,
                    scale=self.variability[k],
                    size=self.features_n,
                )
                local[t, k] = _softmax(log_archetypes[k] + deviation)
        return local

    def generate(self) -> DistributionalSyntheticData:
        """Generate one complete synthetic dataset and its known truth."""

        archetypes = self._generate_archetypes()
        contributions = self._generate_contributions()
        local_profiles = self._generate_local_profiles(archetypes)

        signal = np.einsum("tk,tkj->tj", contributions, local_profiles)
        uncertainty = np.maximum(
            self.noise_fraction * np.maximum(signal, 0.0),
            self.uncertainty_floor,
        )
        noise = self.rng.normal(loc=0.0, scale=uncertainty)
        data = np.maximum(signal + noise, 0.0)

        return DistributionalSyntheticData(
            data=data,
            uncertainty=uncertainty,
            signal=signal,
            archetypes=archetypes,
            local_profiles=local_profiles,
            contributions=contributions,
            variability=self.variability.copy(),
        )
