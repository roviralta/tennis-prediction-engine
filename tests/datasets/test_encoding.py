import polars as pl
import pytest

from tennis_prediction.datasets.encoding import TrainingFeatureEncoder
from tennis_prediction.datasets.encoding import encode_prediction_splits
from tennis_prediction.datasets.imputation import impute_prediction_splits
from tennis_prediction.datasets.schema import build_prediction_table
from tennis_prediction.datasets.schema import split_prediction_table


def _imputed_splits():
    rows = pl.DataFrame(
        {
            "match_index": [1, 1, 2, 2, 3, 3],
            "tourney_date": [20250101, 20250101, 20250102, 20250102, 20250103, 20250103],
            "player_a": [1, 2, 3, 4, 5, 6],
            "player_b": [2, 1, 4, 3, 6, 5],
            "target": [1, 0, 1, 0, 1, 0],
            "surface": ["Hard", "Hard", "Clay", "Clay", None, None],
            "rank_difference": [10.0, 20.0, None, 40.0, 50.0, 60.0],
        }
    )
    table = build_prediction_table(rows, ["surface", "rank_difference"])
    splits = split_prediction_table(table, 20250101, 20250102)
    return impute_prediction_splits(splits)


def test_encoding_learns_categories_from_training_and_handles_unknown_values() -> None:
    imputed = _imputed_splits()

    result = encode_prediction_splits(imputed)

    assert result.encoder.category_levels["surface"] == ("Hard", "__MISSING__")
    train = result.splits.train.features
    validation = result.splits.validation.features
    test = result.splits.test.features

    assert train["surface__category_0"].to_list() == [1, 1]
    assert train["surface__category_1"].to_list() == [0, 0]
    # Clay was not present in training, so its one-hot values are all zero.
    assert validation["surface__category_0"].to_list() == [0, 0]
    assert validation["surface__category_1"].to_list() == [0, 0]
    # Missing surface uses the predeclared missing category column.
    assert test["surface__category_0"].to_list() == [0, 0]
    assert test["surface__category_1"].to_list() == [1, 1]
    assert validation["rank_difference"].to_list() == [15.0, 40.0]


def test_encoder_requires_imputation_before_fit_and_transform() -> None:
    with pytest.raises(ValueError, match="impute before encoding"):
        TrainingFeatureEncoder().fit(pl.DataFrame({"surface": ["Hard", None]}))

    encoder = TrainingFeatureEncoder().fit(pl.DataFrame({"surface": ["Hard"]}))
    with pytest.raises(ValueError, match="impute before encoding"):
        encoder.transform(pl.DataFrame({"surface": [None]}))


def test_encoder_rejects_new_or_reordered_feature_schema() -> None:
    encoder = TrainingFeatureEncoder().fit(
        pl.DataFrame({"surface": ["Hard"], "rank": [1.0]})
    )

    with pytest.raises(ValueError, match="columns and order must match"):
        encoder.transform(pl.DataFrame({"rank": [1.0], "surface": ["Hard"]}))


def test_encoder_rejects_collision_with_generated_columns() -> None:
    training = pl.DataFrame(
        {"surface": ["Hard"], "surface__category_0": [1.0]}
    )

    with pytest.raises(ValueError, match="conflict with generated category columns"):
        TrainingFeatureEncoder().fit(training)
