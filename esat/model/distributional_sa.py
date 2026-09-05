"""Distributional source-type factorization for receptor modeling.

This module generalizes the standard ESAT factorization ``V ~= W @ H``. Instead
of representing each factor/source type by one fixed profile ``H[k]``, the
model uses a global archetype ``H_bar[k]`` and a local profile realization
``H_local[t, k]`` for each sample::

    V[t, j] ~= sum_k W[t, k] * H_local[t, k, j]

Local profiles are constrained to the probability simplex and are shrunk
toward the global archetype. Archetypes are learned from receptor data; no
source labels or reference profiles are required.

The implementation is deliberately a penalized/MAP-like research prototype,
not a claim of full Bayesian posterior inference. Missing receptor cells can be
excluded explicitly with ``observation_mask`` rather than represented by
artificial zero observations with merely enlarged uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import nnls

from esat.model.ls_nmf import LSNMF


_EPS = 1e-12


def _project_simplex_rows(values: np.ndarray) -> np.ndarray:
    """Project every row of a 2-D array onto the unit simplex."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("simplex projection expects a 2-D array")

    out = np.empty_like(x)
    for i, row in enumerate(x):
        u = np.sort(row)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, row.size + 1, dtype=np.float64)
        positive = u - cssv / ind > 0
        if not np.any(positive):
            out[i] = np.full(row.size, 1.0 / row.size)
            continue
        rho = np.nonzero(positive)[0][-1]
        theta = cssv[rho] / float(rho + 1)
        out[i] = np.maximum(row - theta, 0.0)

    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= _EPS] = 1.0
    return out / row_sum


@dataclass(frozen=True)
class DistributionalSAResult:
    """Immutable snapshot of a fitted :class:`DistributionalSA` model."""

    archetypes: np.ndarray
    local_profiles: np.ndarray
    contributions: np.ndarray
    reconstruction: np.ndarray
    profile_sd: np.ndarray
    profile_rms_variability: np.ndarray
    q_true: float
    profile_penalty_loss: float
    objective: float
    iterations: int
    converged: bool


class DistributionalSA:
    """Hierarchical factorization with recurrent distributional source types.

    Parameters
    ----------
    V
        Non-negative receptor data, shape ``(samples, features)``. Cells marked
        false in ``observation_mask`` may contain placeholder finite values;
        they receive exactly zero reconstruction weight.
    U
        Positive measurement uncertainty with the same shape as ``V``.
    factors
        Number of latent source/process types.
    observation_mask
        Optional boolean matrix with the same shape as ``V``. ``True`` means
        the receptor cell is observed and contributes to the weighted loss;
        ``False`` means the cell is missing/excluded and its weight is exactly
        zero. If omitted, every cell is observed.
    profile_penalty
        Dimensionless shrinkage strength for local profiles. It can be a scalar
        or one value per factor. Larger values approach the static-profile
        special case.
    seed, init_iter, max_iter, profile_steps, tol
        Optimization settings.

    Notes
    -----
    Profiles are normalized across modeled features. Consequently ``W`` is a
    contribution scale in the modeled feature space; it is only total aerosol
    mass when the chosen features support that physical interpretation.
    """

    def __init__(
        self,
        V: np.ndarray,
        U: np.ndarray,
        factors: int,
        observation_mask: np.ndarray | None = None,
        profile_penalty: float | Iterable[float] = 10.0,
        seed: int = 42,
        init_iter: int = 1000,
        max_iter: int = 100,
        profile_steps: int = 10,
        tol: float = 1e-6,
    ) -> None:
        self.V = np.asarray(V, dtype=np.float64)
        self.U = np.asarray(U, dtype=np.float64)
        self.factors = int(factors)
        self.profile_penalty = profile_penalty
        self.seed = int(seed)
        self.init_iter = int(init_iter)
        self.max_iter = int(max_iter)
        self.profile_steps = int(profile_steps)
        self.tol = float(tol)
        self.observation_mask = (
            np.ones(self.V.shape, dtype=bool)
            if observation_mask is None
            else np.asarray(observation_mask, dtype=bool)
        )

        self._validate_inputs()
        base_weight = 1.0 / np.maximum(self.U, _EPS) ** 2
        self.We = np.where(self.observation_mask, base_weight, 0.0)
        self.samples, self.features = self.V.shape

        self.W: np.ndarray | None = None
        self.H_bar: np.ndarray | None = None
        self.H_local: np.ndarray | None = None
        self.profile_penalty_scaled: np.ndarray | None = None

        self.reconstruction: np.ndarray | None = None
        self.profile_sd: np.ndarray | None = None
        self.profile_rms_variability: np.ndarray | None = None
        self.q_true: float | None = None
        self.profile_penalty_loss: float | None = None
        self.objective: float | None = None
        self.objective_history: list[float] = []
        self.converged = False
        self.iterations = 0

    def _validate_inputs(self) -> None:
        if self.V.ndim != 2 or self.U.ndim != 2:
            raise ValueError("V and U must both be 2-D arrays")
        if self.V.shape != self.U.shape:
            raise ValueError("V and U must have identical shapes")
        if self.observation_mask.shape != self.V.shape:
            raise ValueError("observation_mask must have the same shape as V")
        if self.V.size == 0:
            raise ValueError("V and U cannot be empty")
        if not np.isfinite(self.V).all() or not np.isfinite(self.U).all():
            raise ValueError("V and U cannot contain NaN or infinite values")
        if np.any(self.V < 0.0):
            raise ValueError("DistributionalSA requires non-negative V")
        if np.any(self.U <= 0.0):
            raise ValueError("U must contain strictly positive uncertainties")
        if self.factors < 1:
            raise ValueError("factors must be >= 1")
        if self.factors > min(self.V.shape):
            raise ValueError("factors cannot exceed the smaller dimension of V")
        if self.init_iter < 0 or self.max_iter < 1 or self.profile_steps < 1:
            raise ValueError("iteration counts must be non-negative/positive")
        if self.tol <= 0.0:
            raise ValueError("tol must be > 0")
        if not np.any(self.observation_mask):
            raise ValueError("observation_mask cannot exclude every receptor cell")

    def _observed_means(self) -> tuple[np.ndarray, np.ndarray]:
        """Feature- and sample-wise means excluding explicitly masked cells."""
        mask = self.observation_mask.astype(np.float64)
        feature_count = mask.sum(axis=0)
        sample_count = mask.sum(axis=1)
        feature_mean = np.divide(
            (self.V * mask).sum(axis=0),
            feature_count,
            out=np.full(self.V.shape[1], 1e-8, dtype=np.float64),
            where=feature_count > 0,
        )
        sample_mean = np.divide(
            (self.V * mask).sum(axis=1),
            sample_count,
            out=np.full(self.V.shape[0], 1e-8, dtype=np.float64),
            where=sample_count > 0,
        )
        return np.maximum(feature_mean, 1e-8), np.maximum(sample_mean, 1e-8)

    def _static_initialize(self) -> tuple[np.ndarray, np.ndarray]:
        """Initialize W/H with existing ESAT LS-NMF weighted updates."""
        rng = np.random.default_rng(self.seed)
        vh_mean, vw_mean = self._observed_means()

        h_scale = np.sqrt(vh_mean / self.factors)
        w_scale = np.sqrt(vw_mean / self.factors)
        H = h_scale[None, :] * rng.uniform(
            0.5, 1.5, size=(self.factors, self.features)
        )
        W = w_scale[:, None] * rng.uniform(
            0.5, 1.5, size=(self.samples, self.factors)
        )
        H = np.maximum(H, _EPS)
        W = np.maximum(W, _EPS)

        for _ in range(self.init_iter):
            W, H = LSNMF.update(V=self.V, We=self.We, W=W, H=H)
            W = np.nan_to_num(W, nan=_EPS, posinf=1e12, neginf=_EPS)
            H = np.nan_to_num(H, nan=_EPS, posinf=1e12, neginf=_EPS)
            W = np.maximum(W, _EPS)
            H = np.maximum(H, _EPS)

        row_scale = np.maximum(H.sum(axis=1), _EPS)
        H_bar = H / row_scale[:, None]
        W = W * row_scale[None, :]
        return W, H_bar

    def _resolve_penalty(self, W: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.profile_penalty, dtype=np.float64)
        if raw.ndim == 0:
            raw = np.full(self.factors, float(raw))
        if raw.shape != (self.factors,):
            raise ValueError(
                "profile_penalty must be a scalar or contain one value per factor"
            )
        if np.any(raw < 0.0):
            raise ValueError("profile_penalty cannot contain negative values")

        positive_weights = self.We[self.We > 0.0]
        weight_scale = (
            float(np.median(positive_weights)) if positive_weights.size else 1.0
        )
        contribution_scale = np.maximum(np.mean(W**2, axis=0), 1e-8)
        return raw * max(weight_scale, 1e-8) * contribution_scale

    def _update_contributions(self) -> None:
        """Weighted NNLS update of W with local profiles fixed."""
        assert self.W is not None and self.H_local is not None
        for t in range(self.samples):
            sqrt_weight = np.sqrt(self.We[t])
            if not np.any(sqrt_weight > 0.0):
                self.W[t] = 0.0
                continue
            design = self.H_local[t].T * sqrt_weight[:, None]
            target = self.V[t] * sqrt_weight
            solution, _ = nnls(design, target)
            self.W[t] = np.maximum(solution, 0.0)

    def _update_local_profiles(self) -> None:
        """Projected-gradient update of H_local with W/archetypes fixed."""
        assert self.W is not None
        assert self.H_bar is not None
        assert self.H_local is not None
        assert self.profile_penalty_scaled is not None

        penalties = self.profile_penalty_scaled
        for t in range(self.samples):
            wt = self.W[t]
            if np.all(wt <= _EPS) or not np.any(self.We[t] > 0.0):
                self.H_local[t] = self.H_bar
                continue

            local = self.H_local[t].copy()
            lipschitz = 2.0 * (
                float(np.max(self.We[t])) * float(np.dot(wt, wt))
                + float(np.max(penalties))
            )
            step = 1.0 / max(lipschitz, _EPS)

            for _ in range(self.profile_steps):
                prediction = wt @ local
                weighted_residual = self.We[t] * (prediction - self.V[t])
                reconstruction_grad = 2.0 * np.outer(wt, weighted_residual)
                shrinkage_grad = 2.0 * penalties[:, None] * (
                    local - self.H_bar
                )
                local = _project_simplex_rows(
                    local - step * (reconstruction_grad + shrinkage_grad)
                )

            self.H_local[t] = local

    def _update_archetypes(self) -> None:
        assert self.H_local is not None
        H_bar = np.mean(self.H_local, axis=0)
        H_bar = np.maximum(H_bar, _EPS)
        H_bar /= H_bar.sum(axis=1, keepdims=True)
        self.H_bar = H_bar

    def _calculate_losses(self) -> tuple[float, float, float]:
        assert self.W is not None
        assert self.H_bar is not None
        assert self.H_local is not None
        assert self.profile_penalty_scaled is not None

        reconstruction = np.einsum("tk,tkj->tj", self.W, self.H_local)
        residual = self.V - reconstruction
        q_true = float(np.sum(self.We * residual**2))
        profile_penalty_loss = float(
            np.sum(
                self.profile_penalty_scaled[None, :, None]
                * (self.H_local - self.H_bar[None, :, :]) ** 2
            )
        )
        return q_true + profile_penalty_loss, q_true, profile_penalty_loss

    def fit(self) -> "DistributionalSA":
        W, H_bar = self._static_initialize()
        self.W = W
        self.H_bar = H_bar
        self.H_local = np.broadcast_to(
            H_bar, (self.samples, self.factors, self.features)
        ).copy()
        self.profile_penalty_scaled = self._resolve_penalty(W)

        initial_objective, _, _ = self._calculate_losses()
        self.objective_history = [initial_objective]
        best_objective = initial_objective
        best_state = (self.W.copy(), self.H_bar.copy(), self.H_local.copy())

        for iteration in range(1, self.max_iter + 1):
            self._update_contributions()
            self._update_local_profiles()
            self._update_archetypes()

            objective, _, _ = self._calculate_losses()
            self.objective_history.append(objective)
            if objective < best_objective:
                best_objective = objective
                best_state = (
                    self.W.copy(),
                    self.H_bar.copy(),
                    self.H_local.copy(),
                )

            previous = self.objective_history[-2]
            relative_change = abs(previous - objective) / max(abs(previous), _EPS)
            self.iterations = iteration
            if relative_change < self.tol:
                self.converged = True
                break

        self.W, self.H_bar, self.H_local = best_state
        objective, q_true, penalty_loss = self._calculate_losses()
        self.reconstruction = np.einsum("tk,tkj->tj", self.W, self.H_local)
        self.profile_sd = np.std(self.H_local, axis=0)
        self.profile_rms_variability = np.sqrt(
            np.mean(
                (self.H_local - self.H_bar[None, :, :]) ** 2,
                axis=(0, 2),
            )
        )
        self.q_true = q_true
        self.profile_penalty_loss = penalty_loss
        self.objective = objective
        return self

    def result(self) -> DistributionalSAResult:
        if self.W is None or self.H_bar is None or self.H_local is None:
            raise RuntimeError("fit() must be called before result()")
        assert self.reconstruction is not None
        assert self.profile_sd is not None
        assert self.profile_rms_variability is not None
        assert self.q_true is not None
        assert self.profile_penalty_loss is not None
        assert self.objective is not None

        return DistributionalSAResult(
            archetypes=self.H_bar.copy(),
            local_profiles=self.H_local.copy(),
            contributions=self.W.copy(),
            reconstruction=self.reconstruction.copy(),
            profile_sd=self.profile_sd.copy(),
            profile_rms_variability=self.profile_rms_variability.copy(),
            q_true=float(self.q_true),
            profile_penalty_loss=float(self.profile_penalty_loss),
            objective=float(self.objective),
            iterations=int(self.iterations),
            converged=bool(self.converged),
        )
