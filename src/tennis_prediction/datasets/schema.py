from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl


METADATA_COLUMNS = ["match_index", "tourney_date", "player_a", "player_b"]
TARGET_COLUMN = "target"
REQUIRED_COLUMNS = METADATA_COLUMNS + [TARGET_COLUMN]
FORBIDDEN_FEATURE_COLUMNS = {
    *METADATA_COLUMNS,
    TARGET_COLUMN,
    "winner_id",
    "loser_id",
    "score",
    "minutes",
}


@dataclass(frozen=True)
class PredictionTable:
    """Prediction rows split into metadata, model inputs, and labels."""

    metadata: pl.DataFrame
    features: pl.DataFrame
    target: pl.Series

    def __post_init__(self) -> None:
        row_counts = {
            self.metadata.height,
            self.features.height,
            len(self.target),
        }
        if len(row_counts) != 1:
            raise ValueError("metadata, features, and target must have equal row counts")


@dataclass(frozen=True)
class PredictionSplits:
    """Chronological partitions, with rows lacking dates kept separately."""

    train: PredictionTable
    validation: PredictionTable
    test: PredictionTable
    undated: PredictionTable


def _validate_match_perspectives(matches: pl.DataFrame) -> None:
    """Require two complementary, reversed perspectives for every source match."""

    for group in matches.partition_by("match_index", maintain_order=True):
        match_index = group["match_index"][0]
        if match_index is None:
            raise ValueError("match_index cannot contain null values")
        if group.height != 2:
            raise ValueError(
                f"match_index {match_index!r} must have exactly two player perspectives"
            )

        rows = group.select(METADATA_COLUMNS[1:] + [TARGET_COLUMN]).to_dicts()
        first, second = rows
        if {first[TARGET_COLUMN], second[TARGET_COLUMN]} != {0, 1}:
            raise ValueError(
                f"match_index {match_index!r} must have one target 0 and one target 1"
            )
        if first["tourney_date"] != second["tourney_date"]:
            raise ValueError(
                f"match_index {match_index!r} has inconsistent tournament dates"
            )
        if (
            first["player_a"] != second["player_b"]
            or first["player_b"] != second["player_a"]
        ):
            raise ValueError(
                f"match_index {match_index!r} does not contain reversed player perspectives"
            )


def build_prediction_table(
    matches: pl.DataFrame,
    feature_columns: Sequence[str],
) -> PredictionTable:
    """Select a leakage-aware prediction schema without changing missing values.

    The caller must explicitly list pre-match features. Source-match metadata and
    the target are returned separately so they cannot enter the feature frame by
    accident. No rows are dropped and no values are imputed.
    """

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in matches.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    if isinstance(feature_columns, str):
        raise TypeError("feature_columns must be a sequence of column names, not a string")

    selected_features = list(feature_columns)
    if not selected_features:
        raise ValueError("feature_columns must contain at least one pre-match feature")
    if len(selected_features) != len(set(selected_features)):
        raise ValueError("feature_columns cannot contain duplicates")

    forbidden = sorted(set(selected_features) & FORBIDDEN_FEATURE_COLUMNS)
    forbidden.extend(
        sorted(
            column
            for column in selected_features
            if column.startswith(("w_", "l_"))
        )
    )
    if forbidden:
        raise ValueError(f"Columns cannot be used as pre-match features: {forbidden}")

    missing_features = [
        column for column in selected_features if column not in matches.columns
    ]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if matches[TARGET_COLUMN].null_count():
        raise ValueError("target cannot contain null values")
    unexpected_targets = sorted(set(matches[TARGET_COLUMN].to_list()) - {0, 1})
    if unexpected_targets:
        raise ValueError(f"target must contain only 0 and 1; found: {unexpected_targets}")

    _validate_match_perspectives(matches)

    return PredictionTable(
        metadata=matches.select(METADATA_COLUMNS),
        features=matches.select(selected_features),
        target=matches[TARGET_COLUMN].clone(),
    )


def _validate_yyyymmdd(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer in YYYYMMDD format")
    try:
        datetime.strptime(str(value), "%Y%m%d")
    except ValueError as error:
        raise ValueError(f"{name} must be a valid date in YYYYMMDD format") from error


def _subset_prediction_table(
    table: PredictionTable,
    mask: list[bool],
) -> PredictionTable:
    row_mask = pl.Series("_split_mask", mask, dtype=pl.Boolean)
    return PredictionTable(
        metadata=table.metadata.filter(row_mask),
        features=table.features.filter(row_mask),
        target=table.target.filter(row_mask),
    )


def split_prediction_table(
    table: PredictionTable,
    train_end_date: int,
    validation_end_date: int,
) -> PredictionSplits:
    """Split rows by inclusive YYYYMMDD cutoffs without separating match views.

    Training includes dates through ``train_end_date``. Validation includes
    later dates through ``validation_end_date``; test includes all dates after
    that. Rows with unknown dates are returned in ``undated`` and are not
    silently assigned to a chronological partition.
    """

    _validate_yyyymmdd(train_end_date, "train_end_date")
    _validate_yyyymmdd(validation_end_date, "validation_end_date")
    if validation_end_date <= train_end_date:
        raise ValueError("validation_end_date must be later than train_end_date")

    match_dates: dict[int, int | None] = {}
    dates = table.metadata["tourney_date"].to_list()
    match_ids = table.metadata["match_index"].to_list()
    for match_id, match_date in zip(match_ids, dates, strict=True):
        if match_id is None:
            raise ValueError("match_index cannot contain null values")
        if match_date is not None:
            if not isinstance(match_date, int):
                raise TypeError("tourney_date must contain YYYYMMDD integers or nulls")
            _validate_yyyymmdd(match_date, "tourney_date")
        if match_id in match_dates and match_dates[match_id] != match_date:
            raise ValueError(f"match_index {match_id!r} has inconsistent tournament dates")
        match_dates[match_id] = match_date

    train_mask = [date is not None and date <= train_end_date for date in dates]
    validation_mask = [
        date is not None and train_end_date < date <= validation_end_date
        for date in dates
    ]
    test_mask = [date is not None and date > validation_end_date for date in dates]
    undated_mask = [date is None for date in dates]

    return PredictionSplits(
        train=_subset_prediction_table(table, train_mask),
        validation=_subset_prediction_table(table, validation_mask),
        test=_subset_prediction_table(table, test_mask),
        undated=_subset_prediction_table(table, undated_mask),
    )
