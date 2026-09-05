import numpy as np

from esat.model.lowrank_distributional_sa import LowRankDistributionalSA
from eval.distributional_simulator import DistributionalSimulator


def test_lowrank_model_shapes_and_simplex_constraints():
    synthetic = DistributionalSimulator(
        seed=31,
        factors_n=2,
        features_n=8,
        samples_n=60,
        variability=[0.05, 0.30],
        variability_mode="lowrank",
        variability_rank=1,
        noise_fraction=0.02,
    ).generate()

    model = LowRankDistributionalSA(
        V=synthetic.data,
        U=synthetic.uncertainty,
        factors=2,
        variability_rank=1,
        sv_shrinkage=0.8,
        profile_penalty=0.02,
        seed=31,
        init_iter=150,
        max_iter=8,
        profile_steps=5,
    ).fit()

    assert model.W.shape == (60, 2)
    assert model.H_bar.shape == (2, 8)
    assert model.H_local.shape == (60, 2, 8)
    assert model.loadings.shape == (2, 1, 8)
    assert model.scores.shape == (60, 2, 1)
    assert model.latent_tau.shape == (2,)
    assert model.effective_rank.shape == (2,)
    assert np.all(model.W >= 0.0)
    assert np.all(model.H_local >= 0.0)
    np.testing.assert_allclose(model.H_bar.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(model.H_local.sum(axis=2), 1.0, atol=1e-8)
    assert np.isfinite(model.objective)


def test_lowrank_model_learns_more_variability_for_variable_factor():
    synthetic = DistributionalSimulator(
        seed=37,
        factors_n=2,
        features_n=8,
        samples_n=100,
        variability=[0.0, 0.40],
        variability_mode="lowrank",
        variability_rank=1,
        contribution_max=10.0,
        noise_fraction=0.01,
    ).generate()

    model = LowRankDistributionalSA(
        V=synthetic.data,
        U=synthetic.uncertainty,
        factors=2,
        variability_rank=1,
        sv_shrinkage=0.8,
        profile_penalty=0.01,
        seed=37,
        init_iter=200,
        max_iter=12,
        profile_steps=6,
    ).fit()

    # Factor order can permute, so compare the sorted learned scales.  A mixed
    # static/variable truth should not collapse both factors to the same scale.
    learned = np.sort(model.profile_rms_variability)
    assert learned[-1] > learned[0] + 1e-4
    assert np.max(model.latent_tau) > 0.0


def test_lowrank_static_truth_remains_close_to_static():
    synthetic = DistributionalSimulator(
        seed=41,
        factors_n=2,
        features_n=8,
        samples_n=80,
        variability=0.0,
        variability_mode="lowrank",
        variability_rank=1,
        noise_fraction=0.01,
    ).generate()

    model = LowRankDistributionalSA(
        V=synthetic.data,
        U=synthetic.uncertainty,
        factors=2,
        variability_rank=1,
        sv_shrinkage=1.5,
        profile_penalty=0.05,
        seed=41,
        init_iter=180,
        max_iter=10,
        profile_steps=5,
    ).fit()

    assert float(np.mean(model.profile_rms_variability)) < 0.02
    assert model.objective <= model.objective_history[0] + 1e-8
