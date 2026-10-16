from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from tennis_prediction.datasets.imputation import MISSING_CATEGORY
from tennis_prediction.datasets.imputation import ImputedPredictionSplits
from tennis_prediction.datasets.schema import PredictionSplits
from tennis_prediction.datasets.schema import PredictionTable


@dataclass(frozen=True)
class EncodedPredictionSplits:
    """Chronological splits encoded with categories learned from training only."""

    splits: PredictionSplits
    encoder: "TrainingFeatureEncoder"


class TrainingFeatureEncoder:
    """One-hot encode categorical columns using training categories only."""

    def __init__(self) -> None:
        self.feature_columns: list[str] | None = None
        self.numeric_columns: list[str] = []
        self.category_levels: dict[str, tuple[str, ...]] = {}

    def fit(self, training_features: pl.DataFrame) -> "TrainingFeatureEncoder":
        """Learn categorical levels from the training feature frame."""

        if training_features.height == 0:
            raise ValueError("Cannot fit categorical encoding with no training rows")
        if not training_features.columns:
            raise ValueError("Cannot fit categorical encoding with no feature columns")

        numeric_columns: list[str] = []
        category_levels: dict[str, tuple[str, ...]] = {}
        for column, dtype in training_features.schema.items():
            if training_features[column].null_count():
                raise ValueError(
                    f"Feature {column!r} still contains nulls; impute before encoding"
                )
            if dtype.is_numeric():
                numeric_columns.append(column)
            elif dtype in (pl.String, pl.Categorical, pl.Boolean):
                category_values = (
                    training_features[column].cast(pl.String).unique().to_list()
                )
                category_levels[column] = tuple(
                    sorted(set(category_values) | {MISSING_CATEGORY})
                )
            else:
                raise TypeError(
                    f"Unsupported feature type for {column!r}: {dtype}; "
                    "convert it to a numeric or categorical feature first"
                )

        generated_category_columns = {
            f"{column}__category_{category_index}"
            for column, categories in category_levels.items()
            for category_index, _ in enumerate(categories)
        }
        collisions = sorted(generated_category_columns & set(training_features.columns))
        if collisions:
            raise ValueError(
                f"Feature names conflict with generated category columns: {collisions}"
            )

        self.feature_columns = training_features.columns.copy()
        self.numeric_columns = numeric_columns
        self.category_levels = category_levels
        return self

    def transform(self, features: pl.DataFrame) -> pl.DataFrame:
        """Return numeric columns and one-hot columns in the fitted stable schema."""

        if self.feature_columns is None:
            raise RuntimeError("Fit the encoder on training features before transform")
        if features.columns != self.feature_columns:
            raise ValueError(
                "Feature columns and order must match the training schema; "
                f"expected {self.feature_columns}, found {features.columns}"
            )

        numeric_set = set(self.numeric_columns)
        expressions = []
        for column in self.feature_columns:
            if features[column].null_count():
                raise ValueError(
                    f"Feature {column!r} still contains nulls; impute before encoding"
                )
            if column in numeric_set:
                if not features.schema[column].is_numeric():
                    raise TypeError(f"Feature {column!r} must remain numeric after fitting")
                expressions.append(pl.col(column).cast(pl.Float64).alias(column))
            else:
                if features.schema[column] not in (pl.String, pl.Categorical, pl.Boolean):
                    raise TypeError(f"Feature {column!r} must remain categorical after fitting")
                for category_index, category in enumerate(self.category_levels[column]):
                    encoded_name = f"{column}__category_{category_index}"
                    expressions.append(
                        (pl.col(column).cast(pl.String) == category)
                        .cast(pl.UInt8)
                        .alias(encoded_name)
                    )

        return features.select(expressions)

    def fit_transform(self, training_features: pl.DataFrame) -> pl.DataFrame:
        """Fit category levels on training data, then encode training rows."""

        return self.fit(training_features).transform(training_features)


def _replace_features(table: PredictionTable, features: pl.DataFrame) -> PredictionTable:
    return PredictionTable(
        metadata=table.metadata,
        features=features,
        target=table.target,
    )


def encode_prediction_splits(
    imputed: ImputedPredictionSplits,
) -> EncodedPredictionSplits:
    """Fit category levels on imputed training data and encode every split."""

    encoder = TrainingFeatureEncoder().fit(imputed.splits.train.features)
    encoded_splits = PredictionSplits(
        train=_replace_features(
            imputed.splits.train,
            encoder.transform(imputed.splits.train.features),
        ),
        validation=_replace_features(
            imputed.splits.validation,
            encoder.transform(imputed.splits.validation.features),
        ),
        test=_replace_features(
            imputed.splits.test,
            encoder.transform(imputed.splits.test.features),
        ),
        undated=_replace_features(
            imputed.splits.undated,
            encoder.transform(imputed.splits.undated.features),
        ),
    )
    return EncodedPredictionSplits(splits=encoded_splits, encoder=encoder)
