from datetime import date

import polars as pl
import pytest

from tennis_prediction.datasets.imputation import MISSING_CATEGORY
from tennis_prediction.datasets.imputation import TrainingFeatureImputer
from tennis_prediction.datasets.imputation import impute_prediction_splits
from tennis_prediction.datasets.schema import build_prediction_table
from tennis_prediction.datasets.schema import split_prediction_table


def test_imputer_learns_numeric_median_and_category_marker_from_training() -> None:
    training = pl.DataFrame(
        {
            "rank_difference": [1.0, None, 5.0],
            "surface": ["Hard", None, "Clay"],
        }
    )
    validation = pl.DataFrame(
        {
            "rank_difference": [None, 100.0],
            "surface": ["Grass", None],
        }
    )

    imputer = TrainingFeatureImputer().fit(training)
    transformed_training = imputer.transform(training)
    transformed_validation = imputer.transform(validation)

    assert imputer.numeric_medians == {"rank_difference": 3.0}
    assert transformed_training["rank_difference"].to_list() == [1.0, 3.0, 5.0]
    assert transformed_training["surface"].to_list() == [
        "Hard",
        MISSING_CATEGORY,
        "Clay",
    ]
    assert transformed_training["rank_difference__missing"].to_list() == [0, 1, 0]
    assert transformed_training["surface__missing"].to_list() == [0, 1, 0]
    assert transformed_validation["rank_difference"].to_list() == [3.0, 100.0]
    assert transformed_validation["surface"].to_list() == ["Grass", MISSING_CATEGORY]
    assert transformed_validation["rank_difference__missing"].to_list() == [1, 0]
    assert transformed_validation["surface__missing"].to_list() == [0, 1]


def test_impute_prediction_splits_fits_on_train_not_validation_or_test() -> None:
    rows = pl.DataFrame(
        {
            "match_index": [1, 1, 2, 2, 3, 3],
            "tourney_date": [20250101, 20250101, 20250102, 20250102, 20250103, 20250103],
            "player_a": [1, 2, 3, 4, 5, 6],
            "player_b": [2, 1, 4, 3, 6, 5],
            "target": [1, 0, 1, 0, 1, 0],
            "rank_difference": [10.0, 20.0, None, 100.0, 200.0, None],
        }
    )
    table = build_prediction_table(rows, ["rank_difference"])
    splits = split_prediction_table(table, 20250101, 20250102)

    result = impute_prediction_splits(splits)

    assert result.imputer.numeric_medians == {"rank_difference": 15.0}
    assert result.splits.train.features["rank_difference"].to_list() == [10.0, 20.0]
    assert result.splits.validation.features["rank_difference"].to_list() == [15.0, 100.0]
    assert result.splits.test.features["rank_difference"].to_list() == [200.0, 15.0]
    assert result.splits.validation.features["rank_difference__missing"].to_list() == [1, 0]
    assert result.splits.test.features["rank_difference__missing"].to_list() == [0, 1]


def test_imputer_rejects_numeric_feature_that_is_null_in_all_training_rows() -> None:
    training = pl.DataFrame(
        {"rank_difference": [None, None]}, schema={"rank_difference": pl.Float64}
    )

    with pytest.raises(ValueError, match="entirely null in training data"):
        TrainingFeatureImputer().fit(training)


def test_imputer_rejects_unsupported_feature_types() -> None:
    training = pl.DataFrame({"match_date": [date(2025, 1, 1)]})

    with pytest.raises(TypeError, match="convert it to a numeric or categorical"):
        TrainingFeatureImputer().fit(training)


def test_imputer_requires_fitted_state_and_matching_schema() -> None:
    imputer = TrainingFeatureImputer()
    with pytest.raises(RuntimeError, match="Fit the imputer"):
        imputer.transform(pl.DataFrame({"rank": [1]}))

    imputer.fit(pl.DataFrame({"rank": [1, 2]}))
    with pytest.raises(ValueError, match="columns and order must match"):
        imputer.transform(pl.DataFrame({"other": [1, 2]}))


def test_imputer_rejects_collision_with_generated_missing_indicators() -> None:
    training = pl.DataFrame(
        {
            "rank": [1, 2],
            "rank__missing": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="conflict with generated missing indicators"):
        TrainingFeatureImputer().fit(training)


def test_imputer_reserves_missing_category_value_in_all_splits() -> None:
    imputer = TrainingFeatureImputer().fit(pl.DataFrame({"surface": ["Hard", "Clay"]}))

    with pytest.raises(ValueError, match="contains reserved value"):
        imputer.transform(pl.DataFrame({"surface": [MISSING_CATEGORY]}))
