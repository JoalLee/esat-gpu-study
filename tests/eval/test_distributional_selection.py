import numpy as np

from eval.distributional_selection import (
    fit_static_lsnmf,
    make_holdout_mask,
    weighted_metrics,
)


def test_holdout_mask_is_observed_only_and_preserves_training_rows():
    observation_mask = np.array(
        [
            [True, True, False, True],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )

    first = make_holdout_mask(observation_mask, fraction=0.9, seed=11)
    second = make_holdout_mask(observation_mask, fraction=0.9, seed=11)

    np.testing.assert_array_equal(first, second)
    assert np.all(~first | observation_mask)
    fit_mask = observation_mask & ~first
    np.testing.assert_array_equal(fit_mask.any(axis=1), np.array([True, False, True]))
    assert first.any()


def test_weighted_metrics_uses_only_evaluation_mask():
    V = np.array([[1.0, 10.0]])
    U = np.ones_like(V)
    reconstruction = np.array([[0.0, 0.0]])

    q_true, cells, q_per_cell = weighted_metrics(
        V,
        U,
        reconstruction,
        np.array([[True, False]]),
    )

    assert q_true == 1.0
    assert cells == 1
    assert q_per_cell == 1.0


def test_static_fit_handles_all_missing_rows_without_nan():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 0.5],
            [1.0, 2.0, 0.25],
        ]
    )
    U = np.ones_like(V)
    observation_mask = np.array(
        [
            [False, False, False],
            [True, True, True],
            [True, True, True],
        ]
    )

    fit = fit_static_lsnmf(
        V,
        U,
        factors=2,
        observation_mask=observation_mask,
        seed=7,
        max_iter=20,
    )

    assert np.isfinite(fit.W).all()
    assert np.isfinite(fit.H_bar).all()
    assert np.isfinite(fit.reconstruction).all()
    np.testing.assert_allclose(fit.W[0], 0.0)
    np.testing.assert_allclose(fit.H_bar.sum(axis=1), 1.0)
