"""Low-rank distributional source-type factorization.

This V2 prototype constrains occurrence-specific factor profiles to a learned
low-dimensional family around a global archetype.  It is intentionally an
empirical-Bayes / penalized prototype rather than a full posterior sampler.

For factor k and sample t, the model is conceptually

    eta[t,k] = alpha[k] + D[t,k]
    rank(D[:,k,:]) <= r
    H_local[t,k] = softmax(eta[t,k])
    V[t,j] ~= sum_k W[t,k] * H_local[t,k,j]

The low-rank deviation D is re-estimated by singular-value shrinkage after a
weighted local-profile update.  This gives the model two distinct mechanisms:

* H_bar / alpha define the recurring source-type identity;
* a small number of learned profile directions define allowable within-type
  heterogeneity.

No source labels or reference source profiles are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from esat.model.distributional_sa import DistributionalSA, _EPS


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(np.clip(shifted, -60.0, 60.0))
    denom = np.maximum(exp_x.sum(axis=1, keepdims=True), _EPS)
    return exp_x / denom


def _clr_rows(profiles: np.ndarray) -> np.ndarray:
    """Centered-log-ratio coordinates for simplex-valued rows."""
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

    Parameters
    ----------
    variability_rank
        Maximum number of profile-variation directions retained per factor.
    sv_shrinkage
        Strength of empirical singular-value shrinkage.  The shrinkage scale
        is estimated separately for each factor from discarded singular
        values, so factors can collapse toward the static special case when
        the data do not support structured variability.
    profile_penalty
        Weak quadratic penalty used only while proposing local profiles before
        projection onto the low-rank family.  In V2 it should usually be much
        weaker than in the unrestricted V1 model because low-rank compression
        supplies the main structural regularization.

    Notes
    -----
    ``latent_tau`` is learned from the retained CLR-space deviations.  It is an
    empirical variability scale, not yet a sampled Bayesian posterior scale.
    A later probabilistic implementation can replace this step with an
    explicit shrinkage prior without changing the model semantics.
    """

    def __init__(
        self,
        V: np.ndarray,
        U: np.ndarray,
        factors: int,
        variability_rank: int = 2,
        sv_shrinkage: float = 1.0,
        profile_penalty: float | Iterable[float] = 0.05,
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

    def _compress_profile_family(self) -> None:
        """Project proposed local profiles onto factor-wise low-rank families."""
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
            alpha = alpha - np.mean(alpha)
            centered = clr - alpha[None, :]

            # SVD learns the recurring directions of within-type profile
            # variation directly from receptor-data-supported local proposals.
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            rank = min(max_rank, s.size)
            head = s[:rank].copy()
            singular_values_raw[k, :rank] = head

            # Estimate a factor-specific noise floor from discarded directions.
            # If no tail exists, use the lower half of the spectrum.  Soft
            # thresholding provides an empirical shrink-to-static mechanism.
            tail = s[rank:]
            if tail.size == 0:
                start = max(1, s.size // 2)
                tail = s[start:]
            noise_scale = float(np.median(tail)) if tail.size else 0.0
            threshold = self.sv_shrinkage * noise_scale
            shrunk = np.maximum(head - threshold, 0.0)
            singular_values_shrunk[k, :rank] = shrunk

            active = shrunk > max(1e-10, 1e-8 * max(float(head[0]) if head.size else 0.0, 1.0))
            effective_rank[k] = int(np.sum(active))

            if rank > 0:
                loadings[k, :rank] = vt[:rank]
                score_k = u[:, :rank] * shrunk[None, :]
                scores[:, k, :rank] = score_k
                deviation = score_k @ vt[:rank]
            else:
                deviation = np.zeros_like(centered)

            logits = alpha[None, :] + deviation
            local_k = _softmax_rows(logits)
            archetype_k = _softmax_rows(alpha[None, :])[0]

            new_local[:, k, :] = local_k
            new_archetypes[k] = archetype_k
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
        """Fit the low-rank distributional source-type factorization."""
        W, H_bar = self._static_initialize()
        self.W = W
        self.H_bar = H_bar
        self.H_local = np.broadcast_to(
            H_bar, (self.samples, self.factors, self.features)
        ).copy()
        self.profile_penalty_scaled = self._resolve_penalty(W)

        # Initialize latent-family outputs at the exact static special case.
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
        best_state = (
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

        for iteration in range(1, self.max_iter + 1):
            self._update_contributions()
            self._update_local_profiles()
            self._compress_profile_family()

            objective, _, _ = self._calculate_losses()
            self.objective_history.append(objective)

            if objective < best_objective:
                best_objective = objective
                best_state = (
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

            previous = self.objective_history[-2]
            relative_change = abs(previous - objective) / max(abs(previous), _EPS)
            self.iterations = iteration
            if relative_change < self.tol:
                self.converged = True
                break

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
        ) = best_state

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
