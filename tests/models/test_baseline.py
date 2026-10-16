import polars as pl
import pytest

from tennis_prediction.datasets.schema import build_prediction_table
from tennis_prediction.datasets.schema import split_prediction_table
from tennis_prediction.models.winner.baseline import train_logistic_baseline
from tennis_prediction.preprocessing.matches import normalize_matches


def _build_splits():
    source_matches = pl.DataFrame(
        {
            "tourney_id": [f"T{i}" for i in range(6)],
            "tourney_name": [f"Tournament {i}" for i in range(6)],
            "surface": ["Hard", "Clay", "Grass", "Hard", "Clay", "Grass"],
            "draw_size": [32] * 6,
            "tourney_level": ["A"] * 6,
            "tourney_date": [20200101, 20200601, 20210101, 20210601, 20220101, 20220601],
            "match_num": [1] * 6,
            "best_of": [3] * 6,
            "round": ["R32"] * 6,
            "winner_id": [1, 2, 3, 4, 5, 6],
            "loser_id": [2, 3, 4, 5, 6, 1],
        }
    )
    normalized = normalize_matches(source_matches).with_columns(
        pl.Series("rank_difference", [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]),
    )
    table = build_prediction_table(normalized, ["surface", "rank_difference"])
    return split_prediction_table(table, 20201231, 20211231)


def test_logistic_baseline_fits_training_preprocessing_and_evaluates_held_out_splits() -> None:
    result = train_logistic_baseline(_build_splits())

    assert result.metrics["split"].to_list() == ["validation", "test"]
    assert result.metrics["rows"].to_list() == [4, 4]
    assert all(
        result.metrics[column].null_count() == 0
        for column in ("accuracy", "log_loss", "brier_score")
    )
    assert result.encoded.encoder.feature_columns == [
        "surface",
        "rank_difference",
        "surface__missing",
        "rank_difference__missing",
    ]


def test_logistic_baseline_rejects_empty_training_split() -> None:
    splits = _build_splits()
    empty = splits.train.__class__(
        metadata=splits.train.metadata.head(0),
        features=splits.train.features.head(0),
        target=splits.train.target.head(0),
    )
    invalid = splits.__class__(empty, splits.validation, splits.test, splits.undated)

    with pytest.raises(ValueError, match="Training split is empty"):
        train_logistic_baseline(invalid)
