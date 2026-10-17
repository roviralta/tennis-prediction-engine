from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from tennis_prediction.datasets.schema import FORBIDDEN_FEATURE_COLUMNS
from tennis_prediction.datasets.schema import PredictionSplits


@dataclass(frozen=True)
class FeatureDefinition:
    """Reviewable source and timing declaration for one model feature."""

    name: str
    source: str
    temporal_rule: str
    available_before_match: bool


@dataclass(frozen=True)
class FeatureAuditReport:
    """Feature distribution summary and its reviewed temporal declarations."""

    distributions: pl.DataFrame
    definitions: tuple[FeatureDefinition, ...]


def _feature_kind(dtype: pl.DataType) -> str:
    if dtype.is_numeric():
        return "numeric"
    if dtype in (pl.String, pl.Categorical, pl.Boolean):
        return "categorical"
    return "unsupported"


def _validate_feature_definitions(
    feature_columns: list[str],
    definitions: Sequence[FeatureDefinition],
) -> tuple[FeatureDefinition, ...]:
    definitions_by_name: dict[str, FeatureDefinition] = {}
    for definition in definitions:
        if definition.name in definitions_by_name:
            raise ValueError(f"Duplicate feature definition: {definition.name!r}")
        definitions_by_name[definition.name] = definition

    missing_definitions = [
        column for column in feature_columns if column not in definitions_by_name
    ]
    extra_definitions = [
        name for name in definitions_by_name if name not in feature_columns
    ]
    if missing_definitions or extra_definitions:
        raise ValueError(
            "Feature definitions must match the selected feature columns; "
            f"missing={missing_definitions}, extra={extra_definitions}"
        )

    forbidden = sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS)
    forbidden.extend(
        sorted(
            column
            for column in feature_columns
            if column.startswith(("w_", "l_", "winner_", "loser_"))
        )
    )
    if forbidden:
        raise ValueError(
            "Post-match or metadata columns cannot be audited as features: "
            f"{forbidden}"
        )

    ordered_definitions = tuple(definitions_by_name[column] for column in feature_columns)
    for definition in ordered_definitions:
        if not definition.source.strip():
            raise ValueError(f"Feature {definition.name!r} must declare its source")
        if not definition.temporal_rule.strip():
            raise ValueError(
                f"Feature {definition.name!r} must declare its temporal availability rule"
            )
        if not definition.available_before_match:
            raise ValueError(
                f"Feature {definition.name!r} is not declared available before the match"
            )

    return ordered_definitions


def _split_distribution(
    name: str,
    frame: pl.DataFrame,
    split_name: str,
) -> dict[str, object]:
    series = frame[name]
    non_null_values = series.drop_nulls()
    unique_values = non_null_values.n_unique()
    if non_null_values.len() == 0:
        distribution_status = "all_missing"
    elif unique_values == 1:
        distribution_status = "constant"
    else:
        distribution_status = "variable"

    missing_count = series.null_count()
    return {
        "feature": name,
        "split": split_name,
        "dtype": str(series.dtype),
        "rows": len(series),
        "missing_count": missing_count,
        "missing_fraction": missing_count / len(series) if len(series) else None,
        "unique_non_missing": unique_values,
        "status": distribution_status,
    }


def audit_prediction_splits(
    splits: PredictionSplits,
    feature_definitions: Sequence[FeatureDefinition],
) -> FeatureAuditReport:
    """Summarize feature distributions and require pre-match provenance declarations.

    This function checks the declared source and temporal rule; it cannot prove
    that an arbitrary external data source actually honored that rule. Call it
    before imputation to inspect original missing-value rates.
    """

    split_frames = {
        "train": splits.train.features,
        "validation": splits.validation.features,
        "test": splits.test.features,
    }
    feature_columns = split_frames["train"].columns
    if not feature_columns:
        raise ValueError("Cannot audit prediction splits without feature columns")

    for split_name, frame in split_frames.items():
        if frame.columns != feature_columns:
            raise ValueError(
                f"Feature columns in {split_name} must match training columns and order"
            )
        for column in feature_columns:
            if frame.schema[column] != split_frames["train"].schema[column]:
                raise TypeError(
                    f"Feature {column!r} has inconsistent dtypes between train and "
                    f"{split_name}: {split_frames['train'].schema[column]} vs "
                    f"{frame.schema[column]}"
                )

    ordered_definitions = _validate_feature_definitions(
        feature_columns, feature_definitions
    )

    unsupported = [
        (column, str(split_frames["train"].schema[column]))
        for column in feature_columns
        if _feature_kind(split_frames["train"].schema[column]) == "unsupported"
    ]
    if unsupported:
        raise TypeError(f"Unsupported feature types: {unsupported}")

    distributions = [
        _split_distribution(column, frame, split_name)
        for column in feature_columns
        for split_name, frame in split_frames.items()
    ]
    return FeatureAuditReport(
        distributions=pl.DataFrame(distributions),
        definitions=ordered_definitions,
    )
