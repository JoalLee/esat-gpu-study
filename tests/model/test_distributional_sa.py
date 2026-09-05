import numpy as np

from esat.model.distributional_sa import DistributionalSA
from eval.distributional_simulator import DistributionalSimulator


def test_distributional_sa_static_limit_is_well_behaved():
    synthetic = DistributionalSimulator(
        seed=11,
        factors_n=2,
        features_n=7,
        samples_n=40,
        variability=0.0,
        contribution_max=8.0,
        noise_fraction=0.02,
    ).generate()

    model = DistributionalSA(
        V=synthetic.data,
        U=synthetic.uncertainty,
        factors=2,
        profile_penalty=1e4,
        seed=11,
        init_iter=150,
        max_iter=10,
        profile_steps=5,
    ).fit()

    assert model.W.shape == (40, 2)
    assert model.H_bar.shape == (2, 7)
    assert model.H_local.shape == (40, 2, 7)
    assert np.all(model.W >= 0.0)
    assert np.all(model.H_bar >= 0.0)
    assert np.all(model.H_local >= 0.0)
    np.testing.assert_allclose(model.H_bar.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(model.H_local.sum(axis=2), 1.0, atol=1e-8)

    assert float(np.max(model.profile_rms_variability)) < 0.01
    assert model.objective <= model.objective_history[0] + 1e-8
    assert np.isfinite(model.q_true)


def test_profile_penalty_controls_within_type_flexibility():
    synthetic = DistributionalSimulator(
        seed=17,
        factors_n=2,
        features_n=7,
        samples_n=40,
        variability=[0.05, 0.35],
        contribution_max=8.0,
        noise_fraction=0.02,
    ).generate()

    flexible = DistributionalSA(
        V=synthetic.data,
        U=synthetic.uncertainty,
        factors=2,
        profile_penalty=0.2,
        seed=17,
        init_iter=150,
        max_iter=10,
        profile_steps=5,
    ).fit()
    static_like = DistributionalSA(
        V=synthetic.data,
        U=synthetic.uncertainty,
        factors=2,
        profile_penalty=100.0,
        seed=17,
        init_iter=150,
        max_iter=10,
        profile_steps=5,
    ).fit()

    assert np.mean(flexible.profile_rms_variability) > np.mean(
        static_like.profile_rms_variability
    )
    assert flexible.q_true <= static_like.q_true + 1e-8


def test_result_snapshot_exposes_distributional_outputs():
    synthetic = DistributionalSimulator(
        seed=23,
        factors_n=2,
        features_n=6,
        samples_n=30,
        variability=0.1,
        noise_fraction=0.03,
    ).generate()

    model = DistributionalSA(
        synthetic.data,
        synthetic.uncertainty,
        factors=2,
        profile_penalty=2.0,
        seed=23,
        init_iter=100,
        max_iter=5,
        profile_steps=3,
    ).fit()
    result = model.result()

    assert result.archetypes.shape == (2, 6)
    assert result.local_profiles.shape == (30, 2, 6)
    assert result.profile_sd.shape == (2, 6)
    assert result.profile_rms_variability.shape == (2,)
    assert result.reconstruction.shape == synthetic.data.shape
    assert np.isfinite(result.objective)


def test_masked_cell_has_exactly_zero_data_influence():
    synthetic = DistributionalSimulator(
        seed=29,
        factors_n=2,
        features_n=6,
        samples_n=35,
        variability=0.1,
        noise_fraction=0.02,
    ).generate()

    mask = np.ones_like(synthetic.data, dtype=bool)
    mask[5, 3] = False

    baseline_data = synthetic.data.copy()
    altered_data = synthetic.data.copy()
    altered_data[5, 3] = 1e6

    kwargs = dict(
        U=synthetic.uncertainty,
        factors=2,
        observation_mask=mask,
        profile_penalty=2.0,
        seed=29,
        init_iter=100,
        max_iter=5,
        profile_steps=3,
    )
    baseline = DistributionalSA(V=baseline_data, **kwargs).fit()
    altered = DistributionalSA(V=altered_data, **kwargs).fit()

    assert baseline.We[5, 3] == 0.0
    assert altered.We[5, 3] == 0.0
    np.testing.assert_allclose(baseline.H_bar, altered.H_bar, atol=1e-10)
    np.testing.assert_allclose(baseline.W, altered.W, atol=1e-10)
    np.testing.assert_allclose(baseline.H_local, altered.H_local, atol=1e-10)
    assert abs(baseline.q_true - altered.q_true) < 1e-10
