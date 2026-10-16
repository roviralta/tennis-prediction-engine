from datetime import date

import polars as pl
import pytest

from tennis_prediction.datasets.audit import FeatureDefinition
from tennis_prediction.datasets.audit import audit_prediction_splits
from tennis_prediction.datasets.schema import PredictionSplits
from tennis_prediction.datasets.schema import PredictionTable
from tennis_prediction.datasets.schema import build_prediction_table
from tennis_prediction.datasets.schema import split_prediction_table


def _splits() -> PredictionSplits:
    rows = pl.DataFrame(
        {
            "match_index": [1, 1, 2, 2, 3, 3],
            "tourney_date": [20250101, 20250101, 20250102, 20250102, 20250103, 20250103],
            "player_a": [1, 2, 3, 4, 5, 6],
            "player_b": [2, 1, 4, 3, 6, 5],
            "target": [1, 0, 1, 0, 1, 0],
            "rank_difference": [10.0, None, None, 40.0, 50.0, 60.0],
            "surface": ["Hard", "Hard", "Clay", None, "Grass", "Grass"],
        }
    )
    table = build_prediction_table(rows, ["rank_difference", "surface"])
    return split_prediction_table(table, 20250101, 20250102)


def _definitions() -> list[FeatureDefinition]:
    return [
        FeatureDefinition(
            name="rank_difference",
            source="ATP rankings archive",
            temporal_rule="Latest ranking dated on or before the tournament date",
            available_before_match=True,
        ),
        FeatureDefinition(
            name="surface",
            source="Tournament metadata",
            temporal_rule="Surface is fixed before the tournament begins",
            available_before_match=True,
        ),
    ]


def test_feature_audit_reports_missingness_and_training_variation_by_split() -> None:
    report = audit_prediction_splits(_splits(), _definitions())

    rank_training = report.distributions.filter(
        (pl.col("feature") == "rank_difference") & (pl.col("split") == "train")
    ).row(0, named=True)
    surface_validation = report.distributions.filter(
        (pl.col("feature") == "surface") & (pl.col("split") == "validation")
    ).row(0, named=True)

    assert rank_training["rows"] == 2
    assert rank_training["missing_count"] == 1
    assert rank_training["missing_fraction"] == 0.5
    assert rank_training["unique_non_missing"] == 1
    assert rank_training["status"] == "constant"
    assert surface_validation["missing_count"] == 1
    assert surface_validation["status"] == "constant"
    rank_test = report.distributions.filter(
        (pl.col("feature") == "rank_difference") & (pl.col("split") == "test")
    ).row(0, named=True)
    assert rank_test["status"] == "variable"
    assert [definition.name for definition in report.definitions] == [
        "rank_difference",
        "surface",
    ]


def test_feature_audit_requires_a_temporal_definition_for_each_feature() -> None:
    definitions = _definitions()[:1]

    with pytest.raises(ValueError, match="definitions must match"):
        audit_prediction_splits(_splits(), definitions)


def test_feature_audit_rejects_features_not_available_before_match() -> None:
    definitions = _definitions()
    definitions[0] = FeatureDefinition(
        name="rank_difference",
        source="ATP rankings archive",
        temporal_rule="Ranking after match completion",
        available_before_match=False,
    )

    with pytest.raises(ValueError, match="not declared available before the match"):
        audit_prediction_splits(_splits(), definitions)


def test_feature_audit_requires_source_and_temporal_rule() -> None:
    definitions = _definitions()
    definitions[0] = FeatureDefinition(
        name="rank_difference",
        source="",
        temporal_rule="Available before match",
        available_before_match=True,
    )

    with pytest.raises(ValueError, match="must declare its source"):
        audit_prediction_splits(_splits(), definitions)


def test_feature_audit_rejects_inconsistent_types_between_splits() -> None:
    splits = _splits()
    validation = PredictionTable(
        metadata=splits.validation.metadata,
        features=splits.validation.features.with_columns(
            pl.col("rank_difference").cast(pl.String)
        ),
        target=splits.validation.target,
    )
    malformed_splits = PredictionSplits(
        train=splits.train,
        validation=validation,
        test=splits.test,
        undated=splits.undated,
    )

    with pytest.raises(TypeError, match="inconsistent dtypes"):
        audit_prediction_splits(malformed_splits, _definitions())


def test_feature_audit_rejects_unsupported_feature_types() -> None:
    rows = pl.DataFrame(
        {
            "match_index": [1, 1, 2, 2, 3, 3],
            "tourney_date": [20250101, 20250101, 20250102, 20250102, 20250103, 20250103],
            "player_a": [1, 2, 3, 4, 5, 6],
            "player_b": [2, 1, 4, 3, 6, 5],
            "target": [1, 0, 1, 0, 1, 0],
            "match_date": [date(2025, 1, 1)] * 6,
        }
    )
    splits = split_prediction_table(
        build_prediction_table(rows, ["match_date"]), 20250101, 20250102
    )
    definitions = [
        FeatureDefinition(
            name="match_date",
            source="Match archive",
            temporal_rule="Tournament date",
            available_before_match=True,
        )
    ]

    with pytest.raises(TypeError, match="Unsupported feature types"):
        audit_prediction_splits(splits, definitions)


def test_feature_audit_marks_all_missing_training_feature() -> None:
    splits = _splits()
    training = PredictionTable(
        metadata=splits.train.metadata,
        features=splits.train.features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("rank_difference")
        ),
        target=splits.train.target,
    )
    all_missing_splits = PredictionSplits(
        train=training,
        validation=splits.validation,
        test=splits.test,
        undated=splits.undated,
    )

    report = audit_prediction_splits(all_missing_splits, _definitions())
    rank_training = report.distributions.filter(
        (pl.col("feature") == "rank_difference") & (pl.col("split") == "train")
    ).row(0, named=True)

    assert rank_training["missing_fraction"] == 1.0
    assert rank_training["status"] == "all_missing"


def test_feature_audit_rejects_post_match_statistic_columns() -> None:
    splits = _splits()

    def with_ace_feature(table: PredictionTable) -> PredictionTable:
        return PredictionTable(
            metadata=table.metadata,
            features=pl.DataFrame({"w_ace": [1] * table.metadata.height}),
            target=table.target,
        )

    post_match_splits = PredictionSplits(
        train=with_ace_feature(splits.train),
        validation=with_ace_feature(splits.validation),
        test=with_ace_feature(splits.test),
        undated=with_ace_feature(splits.undated),
    )
    definitions = [
        FeatureDefinition(
            name="w_ace",
            source="Completed match statistics",
            temporal_rule="Available after the match",
            available_before_match=True,
        )
    ]

    with pytest.raises(ValueError, match="Post-match or metadata columns"):
        audit_prediction_splits(post_match_splits, definitions)
