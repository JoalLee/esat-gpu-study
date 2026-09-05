import numpy as np

from eval.identifiability_diagnostics import (
    archetype_separation_summary,
    diagnose_distributional_solution,
    factor_family_alignments,
)
from eval.span_controlled_simulator import SpanControlledSimulator


def test_factor_alignment_matches_controlled_truth_geometry():
    synthetic = SpanControlledSimulator(
        seed=71,
        factors_n=3,
        features_n=9,
        samples_n=80,
        alignment=0.65,
        variability=[0.0, 0.0, 0.7],
        noise_fraction=0.0,
    ).generate()

    effective_rank = np.array([0, 0, 1])
    _, factor_alignment = factor_family_alignments(
        synthetic.archetypes,
        synthetic.loadings,
        effective_rank=effective_rank,
    )

    assert np.isnan(factor_alignment[0])
    assert np.isnan(factor_alignment[1])
    assert abs(factor_alignment[2] - 0.65) < 1e-8


def test_archetype_overlap_diagnostic_tracks_source_commonality():
    separated = SpanControlledSimulator(
        seed=73,
        factors_n=3,
        features_n=10,
        samples_n=50,
        alignment=0.0,
        variability=0.0,
        source_overlap=0.0,
        noise_fraction=0.0,
    ).generate()
    overlapping = SpanControlledSimulator(
        seed=73,
        factors_n=3,
        features_n=10,
        samples_n=50,
        alignment=0.0,
        variability=0.0,
        source_overlap=0.8,
        noise_fraction=0.0,
    ).generate()

    low = archetype_separation_summary(separated.archetypes)
    high = archetype_separation_summary(overlapping.archetypes)
    assert high["pairwise_cosine_mean"] > low["pairwise_cosine_mean"]
    assert high["pairwise_cosine_max"] > 0.97


def test_combined_diagnostic_flags_high_geometric_confounding():
    synthetic = SpanControlledSimulator(
        seed=79,
        factors_n=3,
        features_n=9,
        samples_n=80,
        alignment=0.9,
        variability=[0.0, 0.0, 0.7],
        source_overlap=0.8,
        noise_fraction=0.0,
    ).generate()

    diagnostics = diagnose_distributional_solution(
        archetypes=synthetic.archetypes,
        loadings=synthetic.loadings,
        effective_rank=np.array([0, 0, 1]),
        profile_rms_variability=synthetic.actual_profile_rms_variability,
        latent_tau=np.array([0.0, 0.0, 0.2]),
    )

    table = diagnostics.factor_table
    assert table.loc[0, "diagnostic_label"] in {
        "static_or_no_active_family",
        "high_geometric_confounding",
    }
    assert table.loc[2, "span_alignment"] > 0.85
    assert table.loc[2, "max_cosine_to_other_archetype"] > 0.97
    assert table.loc[2, "diagnostic_label"] == "high_geometric_confounding"
    assert diagnostics.global_summary["pairwise_cosine_mean"] > 0.95
