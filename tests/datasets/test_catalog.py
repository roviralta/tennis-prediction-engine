import pytest

from tennis_prediction.datasets.catalog import candidate_pre_match_feature_columns
from tennis_prediction.datasets.catalog import feature_definitions_for


def test_feature_catalog_defines_current_temporal_feature_families() -> None:
    definitions = feature_definitions_for(
        [
            "surface",
            "player_a_rank",
            "player_a_win_rate_last_10",
            "player_b_surface_win_rate_last_5",
            "head_to_head_matches",
        ]
    )

    by_name = {definition.name: definition for definition in definitions}
    assert by_name["surface"].available_before_match is True
    assert "on or before" in by_name["player_a_rank"].temporal_rule
    assert "strictly before" in by_name["player_a_win_rate_last_10"].temporal_rule
    assert "target surface" in by_name["player_b_surface_win_rate_last_5"].temporal_rule
    assert "strictly before" in by_name["head_to_head_matches"].temporal_rule


def test_feature_catalog_rejects_features_without_reviewed_provenance() -> None:
    with pytest.raises(ValueError, match="No reviewed pre-match feature definition"):
        feature_definitions_for(["player_a_height"])


def test_feature_catalog_rejects_non_positive_form_windows() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        feature_definitions_for(["player_a_win_rate_last_0"])


def test_candidate_feature_set_uses_the_configured_form_window() -> None:
    candidate_columns = candidate_pre_match_feature_columns(window=5)

    assert "player_a_win_rate_last_5" in candidate_columns
    assert "player_a_surface_win_rate_last_5" in candidate_columns
    assert "player_a_win_rate_last_10" not in candidate_columns


def test_candidate_feature_set_requires_positive_window() -> None:
    with pytest.raises(ValueError, match="window must be greater than zero"):
        candidate_pre_match_feature_columns(window=0)
