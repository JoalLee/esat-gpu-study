import numpy as np

from eval.span_controlled_simulator import SpanControlledSimulator


def test_requested_span_alignment_is_realized():
    for alignment in (0.0, 0.25, 0.75, 1.0):
        synthetic = SpanControlledSimulator(
            seed=51,
            factors_n=3,
            features_n=9,
            samples_n=80,
            alignment=alignment,
            variability=[0.0, 0.0, 0.6],
            noise_fraction=0.0,
        ).generate()

        assert abs(synthetic.actual_alignment[2] - alignment) < 1e-8
        np.testing.assert_allclose(
            synthetic.local_profiles.sum(axis=2),
            1.0,
            atol=1e-10,
        )
        assert np.min(synthetic.local_profiles) >= 0.0
        assert synthetic.actual_profile_rms_variability[2] > 0.0
        assert synthetic.actual_profile_rms_variability[0] < 1e-12
        assert synthetic.actual_profile_rms_variability[1] < 1e-12


def test_alignment_zero_direction_is_orthogonal_to_static_difference_span():
    synthetic = SpanControlledSimulator(
        seed=53,
        factors_n=3,
        features_n=10,
        samples_n=60,
        alignment=0.0,
        variability=[0.0, 0.0, 0.7],
        noise_fraction=0.0,
    ).generate()

    direction = synthetic.loadings[2, 0]
    basis = synthetic.confounding_basis
    assert abs(direction.sum()) < 1e-10
    if basis.shape[0] > 0:
        assert np.max(np.abs(basis @ direction)) < 1e-8


def test_alignment_one_direction_lies_in_static_difference_span():
    synthetic = SpanControlledSimulator(
        seed=59,
        factors_n=3,
        features_n=10,
        samples_n=60,
        alignment=1.0,
        variability=[0.0, 0.0, 0.7],
        noise_fraction=0.0,
    ).generate()

    direction = synthetic.loadings[2, 0]
    basis = synthetic.confounding_basis
    projection = basis.T @ (basis @ direction)
    assert np.linalg.norm(direction - projection) < 1e-8
