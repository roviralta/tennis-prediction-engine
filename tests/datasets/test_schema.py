import polars as pl
import pytest

from tennis_prediction.datasets.schema import build_prediction_table
from tennis_prediction.datasets.schema import split_prediction_table


def _matches() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "match_index": [7, 7, 8, 8],
            "tourney_date": [20250101, 20250101, 20250102, 20250102],
            "player_a": [1, 2, 3, 4],
            "player_b": [2, 1, 4, 3],
            "target": [1, 0, 1, 0],
            "surface": ["Hard", "Hard", "Clay", "Clay"],
            "player_a_rank": [10, 20, None, 40],
        }
    )


def test_prediction_table_separates_metadata_features_and_target() -> None:
    result = build_prediction_table(_matches(), ["surface", "player_a_rank"])

    assert result.metadata.columns == [
        "match_index",
        "tourney_date",
        "player_a",
        "player_b",
    ]
    assert result.features.columns == ["surface", "player_a_rank"]
    assert result.features["player_a_rank"].to_list() == [10, 20, None, 40]
    assert result.target.to_list() == [1, 0, 1, 0]
    assert result.metadata["match_index"].to_list() == [7, 7, 8, 8]


@pytest.mark.parametrize("feature", ["target", "match_index", "winner_id", "score", "w_ace"])
def test_prediction_table_rejects_target_metadata_and_post_match_features(
    feature: str,
) -> None:
    with pytest.raises(ValueError, match="cannot be used as pre-match features"):
        build_prediction_table(_matches(), [feature])


def test_prediction_table_requires_each_source_match_to_have_two_perspectives() -> None:
    malformed = _matches().filter(pl.col("match_index") == 7).head(1)

    with pytest.raises(ValueError, match="exactly two player perspectives"):
        build_prediction_table(malformed, ["surface"])


def test_prediction_table_rejects_inconsistent_target_perspectives() -> None:
    malformed = _matches().with_columns(
        pl.when(pl.col("match_index") == 7)
        .then(1)
        .otherwise(pl.col("target"))
        .alias("target")
    )

    with pytest.raises(ValueError, match="one target 0 and one target 1"):
        build_prediction_table(malformed, ["surface"])


def test_prediction_table_rejects_missing_requested_feature() -> None:
    with pytest.raises(ValueError, match="Missing feature columns"):
        build_prediction_table(_matches(), ["unavailable_feature"])


def test_prediction_table_requires_non_empty_explicit_feature_list() -> None:
    with pytest.raises(ValueError, match="at least one pre-match feature"):
        build_prediction_table(_matches(), [])


def test_chronological_split_uses_inclusive_cutoffs_and_keeps_match_views_together() -> None:
    rows = pl.DataFrame(
        {
            "match_index": [7, 7, 8, 8, 9, 9],
            "tourney_date": [20250101, 20250101, 20250102, 20250102, 20250103, 20250103],
            "player_a": [1, 2, 3, 4, 5, 6],
            "player_b": [2, 1, 4, 3, 6, 5],
            "target": [1, 0, 1, 0, 1, 0],
            "surface": ["Hard", "Hard", "Clay", "Clay", "Grass", "Grass"],
            "player_a_rank": [10, 20, 30, 40, 50, 60],
        }
    )
    table = build_prediction_table(rows, ["surface", "player_a_rank"])

    splits = split_prediction_table(
        table,
        train_end_date=20250101,
        validation_end_date=20250102,
    )

    assert splits.train.metadata["match_index"].to_list() == [7, 7]
    assert splits.validation.metadata["match_index"].to_list() == [8, 8]
    assert splits.test.metadata["match_index"].to_list() == [9, 9]
    assert splits.undated.metadata.height == 0
    assert splits.train.target.to_list() == [1, 0]
    assert splits.train.features["player_a_rank"].to_list() == [10, 20]


def test_chronological_split_keeps_undated_rows_out_of_time_periods() -> None:
    rows = pl.DataFrame(
        {
            "match_index": [1, 1, 2, 2],
            "tourney_date": [None, None, 20250102, 20250102],
            "player_a": [1, 2, 3, 4],
            "player_b": [2, 1, 4, 3],
            "target": [1, 0, 1, 0],
            "feature": [None, None, 2.0, 3.0],
        },
        schema_overrides={"tourney_date": pl.Int64},
    )
    table = build_prediction_table(rows, ["feature"])

    splits = split_prediction_table(table, 20250101, 20250102)

    assert splits.train.metadata.height == 0
    assert splits.validation.metadata["match_index"].to_list() == [2, 2]
    assert splits.test.metadata.height == 0
    assert splits.undated.metadata["match_index"].to_list() == [1, 1]
    assert splits.undated.features["feature"].to_list() == [None, None]


def test_chronological_split_requires_ordered_valid_cutoffs() -> None:
    table = build_prediction_table(_matches(), ["surface"])

    with pytest.raises(ValueError, match="must be later"):
        split_prediction_table(table, 20250102, 20250101)

    with pytest.raises(ValueError, match="valid date"):
        split_prediction_table(table, 20250230, 20250301)


def test_chronological_split_rejects_source_match_with_inconsistent_dates() -> None:
    table = build_prediction_table(_matches(), ["surface"])
    malformed = table.metadata.with_columns(
        pl.Series("tourney_date", [20250101, 20250103, 20250102, 20250102])
    )
    malformed_table = type(table)(malformed, table.features, table.target)

    with pytest.raises(ValueError, match="inconsistent tournament dates"):
        split_prediction_table(malformed_table, 20250101, 20250102)
