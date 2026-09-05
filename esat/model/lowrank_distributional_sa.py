"""Low-rank distributional source-type factorization.

V2 constrains occurrence-specific factor profiles to a learned low-dimensional
family around a global archetype. It is an empirical-Bayes / penalized
prototype, not yet a full posterior sampler.

Conceptually, for factor k and sample t::

    eta[t,k] = alpha[k] + D[t,k]
    rank(D[:,k,:]) <= r
    H_local[t,k] = softmax(eta[t,k])
    V[t,j] ~= sum_k W[t,k] * H_local[t,k,j]

The key distinction from V1 is that local profile deviations cannot move
arbitrarily in feature space. A weighted ridge step first asks which profile
deviations are supported by receptor residuals; factor-wise SVD then retains
only recurring low-rank directions and shrinks weak directions toward zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from esat.model.distributional_sa import (
    DistributionalSA,
    _EPS,
    _project_simplex_rows,
)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), _EPS)


def _clr_rows(profiles: np.ndarray) -> np.ndarray:
    log_p = np.log(np.maximum(profiles, _EPS))
    return log_p - np.mean(log_p, axis=1, keepdims=True)


@dataclass(frozen=True)
class LowRankDistributionalSAResult:
    archetypes: np.ndarray
    local_profiles: np.ndarray
    contributions: np.ndarray
    reconstruction: np.ndarray
    loadings: np.ndarray
    scores: np.ndarray
    latent_tau: np.ndarray
    effective_rank: np.ndarray
    profile_sd: np.ndarray
    profile_rms_variability: np.ndarray
    q_true: float
    profile_penalty_loss: float
    objective: float
    iterations: int
    converged: bool


class LowRankDistributionalSA(DistributionalSA):
    """Distributional source-type model with a learned low-rank profile family.

    ``profile_penalty`` is intentionally weak in V2. It stabilizes the local
    ridge proposal, while low-rank compression and singular-value shrinkage
    provide the main protection against arbitrary profile drift.

    ``latent_tau`` is inferred from retained CLR-space deviations. It should be
    interpreted as an empirical factor-specific variability scale, not as a
    posterior draw from an explicit Bayesian prior.
    """

    def __init__(
        self,
        V: np.ndarray,
        U: np.ndarray,
        factors: int,
        variability_rank: int = 2,
        sv_shrinkage: float = 0.5,
        profile_penalty: float | Iterable[float] = 0.001,
        seed: int = 42,
        init_iter: int = 1000,
        max_iter: int = 100,
        profile_steps: int = 10,
        tol: float = 1e-6,
    ) -> None:
        super().__init__(
            V=V,
            U=U,
            factors=factors,
            profile_penalty=profile_penalty,
            seed=seed,
            init_iter=init_iter,
            max_iter=max_iter,
            profile_steps=profile_steps,
            tol=tol,
        )
        self.variability_rank = int(variability_rank)
        self.sv_shrinkage = float(sv_shrinkage)
        if self.variability_rank < 1:
            raise ValueError("variability_rank must be >= 1")
        if self.variability_rank >= self.features:
            raise ValueError("variability_rank must be smaller than number of features")
        if self.sv_shrinkage < 0.0:
            raise ValueError("sv_shrinkage must be >= 0")

        self.loadings: np.ndarray | None = None
        self.scores: np.ndarray | None = None
        self.latent_tau: np.ndarray | None = None
        self.effective_rank: np.ndarray | None = None
        self.singular_values_raw: np.ndarray | None = None
        self.singular_values_shrunk: np.ndarray | None = None

    def _update_local_profile_proposals(self) -> None:
        """Weighted ridge update for sample-specific profile proposals.

        With W and H_bar fixed, every feature j has a small K-dimensional
        quadratic problem::

            min_h  we_j (v_j - w^T h)^2
                   + sum_k lambda_k (h_k - hbar_kj)^2.

        Solving this system directly gives residual-supported profile movement
        before the proposal is projected onto the low-rank family. This is
        substantially less conservative than taking a few generic gradient
        steps from the static solution.
        """
        assert self.W is not None
        assert self.H_bar is not None
        assert self.H_local is not None
        assert self.profile_penalty_scaled is not None

        penalties = np.maximum(self.profile_penalty_scaled, _EPS)
        penalty_matrix = np.diag(penalties + _EPS)

        for t in range(self.samples):
            w = self.W[t]
            if np.all(w <= _EPS):
                self.H_local[t] = self.H_bar
                continue

            proposal = np.empty((self.factors, self.features), dtype=np.float64)
            outer = np.outer(w, w)
            for j in range(self.features):
                weight = float(self.We[t, j])
                a = weight * outer + penalty_matrix
                b = (
                    weight * w * float(self.V[t, j])
                    + penalties * self.H_bar[:, j]
                )
                try:
                    solution = np.linalg.solve(a, b)
                except np.linalg.LinAlgError:
                    solution = np.linalg.lstsq(a, b, rcond=None)[0]
                proposal[:, j] = np.maximum(solution, _EPS)

            self.H_local[t] = _project_simplex_rows(proposal)

    def _compress_profile_family(self) -> None:
        """Project local proposals onto factor-wise low-rank profile families."""
        assert self.H_local is not None

        max_rank = min(self.variability_rank, self.features - 1, self.samples - 1)
        loadings = np.zeros((self.factors, self.variability_rank, self.features))
        scores = np.zeros((self.samples, self.factors, self.variability_rank))
        latent_tau = np.zeros(self.factors)
        effective_rank = np.zeros(self.factors, dtype=int)
        singular_values_raw = np.zeros((self.factors, self.variability_rank))
        singular_values_shrunk = np.zeros((self.factors, self.variability_rank))

        new_archetypes = np.empty((self.factors, self.features), dtype=np.float64)
        new_local = np.empty_like(self.H_local)

        for k in range(self.factors):
            clr = _clr_rows(self.H_local[:, k, :])
            alpha = np.mean(clr, axis=0)
            alpha -= np.mean(alpha)
            centered = clr - alpha[None, :]

            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            rank = min(max_rank, s.size)
            head = s[:rank].copy()
            singular_values_raw[k, :rank] = head

            tail = s[rank:]
            if tail.size == 0:
                start = max(1, s.size // 2)
                tail = s[start:]
            noise_scale = float(np.median(tail)) if tail.size else 0.0
            threshold = self.sv_shrinkage * noise_scale
            shrunk = np.maximum(head - threshold, 0.0)
            singular_values_shrunk[k, :rank] = shrunk

            active_tol = max(
                1e-10,
                1e-8 * max(float(head[0]) if head.size else 0.0, 1.0),
            )
            effective_rank[k] = int(np.sum(shrunk > active_tol))

            if rank > 0:
                loadings[k, :rank] = vt[:rank]
                score_k = u[:, :rank] * shrunk[None, :]
                scores[:, k, :rank] = score_k
                deviation = score_k @ vt[:rank]
            else:
                deviation = np.zeros_like(centered)

            new_local[:, k, :] = _softmax_rows(alpha[None, :] + deviation)
            new_archetypes[k] = _softmax_rows(alpha[None, :])[0]
            latent_tau[k] = float(np.sqrt(np.mean(deviation**2)))

        self.H_local = new_local
        self.H_bar = new_archetypes
        self.loadings = loadings
        self.scores = scores
        self.latent_tau = latent_tau
        self.effective_rank = effective_rank
        self.singular_values_raw = singular_values_raw
        self.singular_values_shrunk = singular_values_shrunk

    def fit(self) -> "LowRankDistributionalSA":
        W, H_bar = self._static_initialize()
        self.W = W
        self.H_bar = H_bar
        self.H_local = np.broadcast_to(
            H_bar, (self.samples, self.factors, self.features)
        ).copy()
        self.profile_penalty_scaled = self._resolve_penalty(W)

        self.loadings = np.zeros(
            (self.factors, self.variability_rank, self.features), dtype=np.float64
        )
        self.scores = np.zeros(
            (self.samples, self.factors, self.variability_rank), dtype=np.float64
        )
        self.latent_tau = np.zeros(self.factors, dtype=np.float64)
        self.effective_rank = np.zeros(self.factors, dtype=int)
        self.singular_values_raw = np.zeros(
            (self.factors, self.variability_rank), dtype=np.float64
        )
        self.singular_values_shrunk = np.zeros_like(self.singular_values_raw)

        initial_objective, _, _ = self._calculate_losses()
        self.objective_history = [initial_objective]
        best_objective = initial_objective
        best_state = self._snapshot_state()

        for iteration in range(1, self.max_iter + 1):
            self._update_contributions()
            self._update_local_profile_proposals()
            self._compress_profile_family()

            objective, _, _ = self._calculate_losses()
            self.objective_history.append(objective)

            if objective < best_objective:
                best_objective = objective
                best_state = self._snapshot_state()

            previous = self.objective_history[-2]
            relative_change = abs(previous - objective) / max(abs(previous), _EPS)
            self.iterations = iteration
            if relative_change < self.tol:
                self.converged = True
                break

        self._restore_state(best_state)
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

    def _snapshot_state(self) -> tuple[np.ndarray, ...]:
        assert self.W is not None
        assert self.H_bar is not None
        assert self.H_local is not None
        assert self.loadings is not None
        assert self.scores is not None
        assert self.latent_tau is not None
        assert self.effective_rank is not None
        assert self.singular_values_raw is not None
        assert self.singular_values_shrunk is not None
        return (
            self.W.copy(),
            self.H_bar.copy(),
            self.H_local.copy(),
            self.loadings.copy(),
            self.scores.copy(),
            self.latent_tau.copy(),
            self.effective_rank.copy(),
            self.singular_values_raw.copy(),
            self.singular_values_shrunk.copy(),
        )

    def _restore_state(self, state: tuple[np.ndarray, ...]) -> None:
        (
            self.W,
            self.H_bar,
            self.H_local,
            self.loadings,
            self.scores,
            self.latent_tau,
            self.effective_rank,
            self.singular_values_raw,
            self.singular_values_shrunk,
        ) = state

    def result(self) -> LowRankDistributionalSAResult:
        if self.W is None or self.H_bar is None or self.H_local is None:
            raise RuntimeError("fit() must be called before result()")
        assert self.reconstruction is not None
        assert self.loadings is not None
        assert self.scores is not None
        assert self.latent_tau is not None
        assert self.effective_rank is not None
        assert self.profile_sd is not None
        assert self.profile_rms_variability is not None
        assert self.q_true is not None
        assert self.profile_penalty_loss is not None
        assert self.objective is not None

        return LowRankDistributionalSAResult(
            archetypes=self.H_bar.copy(),
            local_profiles=self.H_local.copy(),
            contributions=self.W.copy(),
            reconstruction=self.reconstruction.copy(),
            loadings=self.loadings.copy(),
            scores=self.scores.copy(),
            latent_tau=self.latent_tau.copy(),
            effective_rank=self.effective_rank.copy(),
            profile_sd=self.profile_sd.copy(),
            profile_rms_variability=self.profile_rms_variability.copy(),
            q_true=float(self.q_true),
            profile_penalty_loss=float(self.profile_penalty_loss),
            objective=float(self.objective),
            iterations=int(self.iterations),
            converged=bool(self.converged),
        )
