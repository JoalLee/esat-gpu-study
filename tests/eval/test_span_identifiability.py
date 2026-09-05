from eval.span_identifiability import run_span_identifiability_grid


def test_lowrank_recovery_degrades_when_variation_enters_static_factor_span():
    """Scientific regression for the W/F geometric confounding mechanism.

    This is deliberately a controlled synthetic statement, not a universal
    performance threshold. The exact same source profiles/contributions are
    generated at alignment 0 and 1; only the profile-variation direction is
    rotated from the identifiable tangent complement into the static factor
    difference span.
    """
    results = run_span_identifiability_grid(
        alignments=[0.0, 1.0],
        variability_levels=[0.7],
        noise_levels=[0.01],
        seeds=[7],
        model_kinds=["lowrank"],
        factors=3,
        features=9,
        samples=80,
        init_iter=100,
        max_iter=6,
        profile_steps=4,
    ).sort_values("requested_alignment")

    low = results.iloc[0]
    high = results.iloc[1]

    assert low["subspace_overlap_target"] > 0.5
    assert high["subspace_overlap_target"] < 0.2
    assert low["subspace_overlap_target"] > high["subspace_overlap_target"] + 0.4
    assert low["variability_recovery_ratio"] > high["variability_recovery_ratio"]
    assert low["contribution_correlation_mean"] >= high["contribution_correlation_mean"]
