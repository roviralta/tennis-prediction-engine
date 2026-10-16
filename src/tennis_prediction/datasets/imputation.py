from dataclasses import dataclass

import polars as pl

from tennis_prediction.datasets.schema import PredictionSplits
from tennis_prediction.datasets.schema import PredictionTable


MISSING_CATEGORY = "__MISSING__"


class TrainingFeatureImputer:
    """Impute numeric and categorical features using fitted training statistics."""

    def __init__(self, missing_category: str = MISSING_CATEGORY) -> None:
        if not missing_category:
            raise ValueError("missing_category must not be empty")
        self.missing_category = missing_category
        self.feature_columns: list[str] | None = None
        self.numeric_medians: dict[str, float] = {}
        self.categorical_columns: list[str] = []

    def fit(self, training_features: pl.DataFrame) -> "TrainingFeatureImputer":
        """Learn medians and feature types from training rows only."""

        if training_features.height == 0:
            raise ValueError("Cannot fit feature imputation with no training rows")
        if not training_features.columns:
            raise ValueError("Cannot fit feature imputation with no feature columns")

        numeric_medians: dict[str, float] = {}
        categorical_columns: list[str] = []
        for column, dtype in training_features.schema.items():
            if dtype.is_numeric():
                median = training_features[column].median()
                if median is None:
                    raise ValueError(
                        f"Numeric feature {column!r} is entirely null in training data; "
                        "remove it explicitly or provide training observations"
                    )
                numeric_medians[column] = float(median)
            elif dtype in (pl.String, pl.Categorical, pl.Boolean):
                non_null_values = (
                    training_features[column]
                    .cast(pl.String)
                    .drop_nulls()
                    .unique()
                    .to_list()
                )
                if self.missing_category in non_null_values:
                    raise ValueError(
                        f"Categorical feature {column!r} contains reserved value "
                        f"{self.missing_category!r}"
                    )
                categorical_columns.append(column)
            else:
                raise TypeError(
                    f"Unsupported feature type for {column!r}: {dtype}; "
                    "convert it to a numeric or categorical feature first"
                )

        indicator_names = {f"{column}__missing" for column in training_features.columns}
        collisions = sorted(indicator_names & set(training_features.columns))
        if collisions:
            raise ValueError(
                f"Feature names conflict with generated missing indicators: {collisions}"
            )

        self.feature_columns = training_features.columns.copy()
        self.numeric_medians = numeric_medians
        self.categorical_columns = categorical_columns
        return self

    def transform(self, features: pl.DataFrame) -> pl.DataFrame:
        """Apply fitted imputation and append one missingness indicator per feature."""

        if self.feature_columns is None:
            raise RuntimeError("Fit the imputer on training features before transform")
        if features.columns != self.feature_columns:
            raise ValueError(
                "Feature columns and order must match the training schema; "
                f"expected {self.feature_columns}, found {features.columns}"
            )

        categorical_set = set(self.categorical_columns)
        expressions = []
        for column in self.feature_columns:
            if column in categorical_set:
                observed_values = (
                    features[column].cast(pl.String).drop_nulls().unique().to_list()
                )
                if self.missing_category in observed_values:
                    raise ValueError(
                        f"Categorical feature {column!r} contains reserved value "
                        f"{self.missing_category!r}"
                    )
                expressions.append(
                    pl.col(column)
                    .cast(pl.String)
                    .fill_null(self.missing_category)
                    .alias(column)
                )
            else:
                if not features.schema[column].is_numeric():
                    raise TypeError(
                        f"Feature {column!r} must remain numeric after fitting"
                    )
                expressions.append(
                    pl.col(column)
                    .cast(pl.Float64)
                    .fill_null(self.numeric_medians[column])
                    .alias(column)
                )
            expressions.append(
                pl.col(column).is_null().cast(pl.UInt8).alias(f"{column}__missing")
            )

        return features.with_columns(expressions)

    def fit_transform(self, training_features: pl.DataFrame) -> pl.DataFrame:
        """Fit on training data, then transform those same training rows."""

        return self.fit(training_features).transform(training_features)


@dataclass(frozen=True)
class ImputedPredictionSplits:
    """Chronological splits transformed with an imputer fitted on training only."""

    splits: PredictionSplits
    imputer: TrainingFeatureImputer


def _replace_features(
    table: PredictionTable,
    features: pl.DataFrame,
) -> PredictionTable:
    return PredictionTable(
        metadata=table.metadata,
        features=features,
        target=table.target,
    )


def impute_prediction_splits(
    splits: PredictionSplits,
) -> ImputedPredictionSplits:
    """Fit preprocessing on training only and apply it to every split."""

    imputer = TrainingFeatureImputer().fit(splits.train.features)
    transformed = PredictionSplits(
        train=_replace_features(splits.train, imputer.transform(splits.train.features)),
        validation=_replace_features(
            splits.validation, imputer.transform(splits.validation.features)
        ),
        test=_replace_features(splits.test, imputer.transform(splits.test.features)),
        undated=_replace_features(splits.undated, imputer.transform(splits.undated.features)),
    )
    return ImputedPredictionSplits(splits=transformed, imputer=imputer)
