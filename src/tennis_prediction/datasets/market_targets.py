from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from tennis_prediction.datasets.markets import CandidateMarket
from tennis_prediction.datasets.markets import candidate_markets
from tennis_prediction.preprocessing.scores import ParsedMatchScore
from tennis_prediction.preprocessing.scores import parse_match_score


@dataclass(frozen=True)
class MarketTargetTable:
    """Post-match outcomes kept separate from pre-match model features."""

    metadata: pl.DataFrame
    targets: pl.DataFrame
    definitions: tuple[CandidateMarket, ...]

    def __post_init__(self) -> None:
        if self.metadata.height != self.targets.height:
            raise ValueError("market target metadata and values must have equal row counts")
        if "match_index" not in self.metadata.columns:
            raise ValueError("market target metadata must contain match_index")
        if "match_index" in self.targets.columns:
            raise ValueError("match_index belongs in metadata, not the target values")


@dataclass(frozen=True)
class MarketTargetAudit:
    """Descriptive range and consistency checks for post-match targets."""

    ranges: pl.DataFrame
    consistency: pl.DataFrame
    violations: pl.DataFrame = field(default_factory=pl.DataFrame)


@dataclass(frozen=True)
class MarketTargetSplits:
    """Chronological market-target partitions with undated rows kept separate."""

    train: MarketTargetTable
    validation: MarketTargetTable
    test: MarketTargetTable
    undated: MarketTargetTable


_DIRECT_PLAYER_TARGETS = {
    "aces": ("w_ace", "l_ace"),
    "double_faults": ("w_df", "l_df"),
    "serve_points_played": ("w_svpt", "l_svpt"),
    "first_serves_in": ("w_1stIn", "l_1stIn"),
    "first_serve_points_won": ("w_1stWon", "l_1stWon"),
    "second_serve_points_won": ("w_2ndWon", "l_2ndWon"),
    "serve_games_recorded": ("w_SvGms", "l_SvGms"),
    "break_points_saved": ("w_bpSaved", "l_bpSaved"),
    "break_points_faced": ("w_bpFaced", "l_bpFaced"),
}

_SCORE_TARGET_COLUMNS = (
    "winner_sets",
    "loser_sets",
    "winner_games",
    "loser_games",
    "total_games",
    "correct_set_score",
)


def _integer_source(column: str) -> pl.Expr:
    numeric = pl.col(column).cast(pl.Float64, strict=False)
    is_integer = (numeric.is_finite() & (numeric == numeric.floor())).fill_null(False)
    return pl.when(is_integer).then(numeric.cast(pl.Int64)).otherwise(None)


def _completed_score_targets(score: ParsedMatchScore) -> dict[str, object]:
    complete = score.status in {"complete", "complete_with_match_tiebreak"}
    return {
        "winner_sets": score.winner_sets if complete else None,
        "loser_sets": score.loser_sets if complete else None,
        "winner_games": score.winner_games if complete else None,
        "loser_games": score.loser_games if complete else None,
        "total_games": score.total_games if complete else None,
        "correct_set_score": score.set_score if complete else None,
    }


def build_market_target_table(matches: pl.DataFrame) -> MarketTargetTable:
    """Build one post-match target row per source match.

    Match statistics stay in a dedicated target frame and never enter the
    pre-match feature schema. Player outcomes use the archive's winner/loser
    orientation. Match-tiebreak scores retain set outcomes but leave game totals
    null because game-settlement conventions vary by market. Retirements,
    defaults, walkovers, and invalid scores have null score-derived targets.
    """

    required_columns = ["tourney_date", "winner_id", "loser_id"]
    missing = [column for column in required_columns if column not in matches.columns]
    if missing:
        raise ValueError(f"Missing required outcome metadata columns: {missing}")
    if matches["winner_id"].null_count() or matches["loser_id"].null_count():
        raise ValueError("winner_id and loser_id cannot be null in outcome rows")

    indexed = matches.with_row_index("match_index")
    metadata_columns = ["match_index", "tourney_date", "winner_id", "loser_id"]
    if "best_of" in indexed.columns:
        metadata_columns.append("best_of")
    metadata = indexed.select(metadata_columns)
    parsed_scores: list[ParsedMatchScore] | None = None
    if "score" in indexed.columns:
        best_of_values = (
            indexed["best_of"].to_list()
            if "best_of" in indexed.columns
            else [None] * indexed.height
        )
        parsed_scores = [
            parse_match_score(score, best_of)
            for score, best_of in zip(indexed["score"].to_list(), best_of_values, strict=True)
        ]
        completed_matches = pl.Series(
            "_completed_match",
            [
                score.status in {"complete", "complete_with_match_tiebreak"}
                for score in parsed_scores
            ],
            dtype=pl.Boolean,
        )
        indexed = indexed.with_columns(completed_matches)

    def completed_target(expression: pl.Expr) -> pl.Expr:
        if "_completed_match" not in indexed.columns:
            return expression
        return pl.when(pl.col("_completed_match")).then(expression).otherwise(None)

    definitions_by_name = {market.name: market for market in candidate_markets()}
    target_expressions: list[pl.Expr] = []
    included_definitions: list[CandidateMarket] = []

    for market_name, (winner_column, loser_column) in _DIRECT_PLAYER_TARGETS.items():
        if winner_column not in indexed.columns or loser_column not in indexed.columns:
            continue
        target_expressions.extend(
            [
                completed_target(_integer_source(winner_column)).alias(
                    f"winner_{market_name}"
                ),
                completed_target(_integer_source(loser_column)).alias(
                    f"loser_{market_name}"
                ),
            ]
        )
        included_definitions.append(definitions_by_name[market_name])

    if "w_bpFaced" in indexed.columns and "w_bpSaved" in indexed.columns:
        if "l_bpFaced" in indexed.columns and "l_bpSaved" in indexed.columns:
            winner_converted = _integer_source("l_bpFaced") - _integer_source("l_bpSaved")
            loser_converted = _integer_source("w_bpFaced") - _integer_source("w_bpSaved")
            target_expressions.extend(
                [
                    completed_target(winner_converted).alias("winner_break_points_converted"),
                    completed_target(loser_converted).alias("loser_break_points_converted"),
                ]
            )
            included_definitions.append(definitions_by_name["break_points_converted"])

    return_sources = (
        "w_svpt",
        "w_1stWon",
        "w_2ndWon",
        "l_svpt",
        "l_1stWon",
        "l_2ndWon",
    )
    if all(column in indexed.columns for column in return_sources):
        winner_return_points = (
            _integer_source("l_svpt")
            - _integer_source("l_1stWon")
            - _integer_source("l_2ndWon")
        )
        loser_return_points = (
            _integer_source("w_svpt")
            - _integer_source("w_1stWon")
            - _integer_source("w_2ndWon")
        )
        target_expressions.extend(
            [
                completed_target(winner_return_points).alias("winner_return_points_won"),
                completed_target(loser_return_points).alias("loser_return_points_won"),
            ]
        )
        included_definitions.append(definitions_by_name["return_points_won"])

    if "minutes" in indexed.columns:
        target_expressions.append(
            completed_target(_integer_source("minutes")).alias("match_duration")
        )
        included_definitions.append(definitions_by_name["match_duration"])

    if parsed_scores is not None:
        score_values = [_completed_score_targets(score) for score in parsed_scores]
        for column in _SCORE_TARGET_COLUMNS:
            dtype = pl.String if column == "correct_set_score" else pl.Int64
            target_expressions.append(
                pl.Series(column, [row[column] for row in score_values], dtype=dtype)
            )
        target_expressions.append(
            pl.Series("score_status", [score.status for score in parsed_scores], dtype=pl.String)
        )
        target_expressions.append(
            pl.Series(
                "score_has_match_tiebreak",
                [score.has_match_tiebreak for score in parsed_scores],
                dtype=pl.Boolean,
            )
        )
        included_definitions.extend(
            definitions_by_name[name]
            for name in (
                "set_score",
                "set_winner",
                "correct_set_score",
                "total_games",
                "player_games",
                "games_handicap",
                "sets_handicap",
            )
        )

    if not target_expressions:
        raise ValueError("No candidate market target sources are available in the match data")

    targets = indexed.select(target_expressions)
    return MarketTargetTable(
        metadata=metadata,
        targets=targets,
        definitions=tuple(included_definitions),
    )


def _consistency_checks(columns: set[str]) -> list[tuple[str, set[str], pl.Expr]]:
    checks: list[tuple[str, set[str], pl.Expr]] = []

    def add_check(name: str, inputs: set[str], expression: pl.Expr) -> None:
        if inputs <= columns:
            checks.append((name, inputs, expression))

    for side in ("winner", "loser"):
        svpt = f"{side}_serve_points_played"
        first_in = f"{side}_first_serves_in"
        first_won = f"{side}_first_serve_points_won"
        second_won = f"{side}_second_serve_points_won"
        ace = f"{side}_aces"
        df = f"{side}_double_faults"
        bp_saved = f"{side}_break_points_saved"
        bp_faced = f"{side}_break_points_faced"
        add_check(
            f"{side}_first_serves_exceed_points_played",
            {svpt, first_in},
            pl.col(first_in) > pl.col(svpt),
        )
        add_check(
            f"{side}_first_serve_wins_exceed_first_serves_in",
            {first_in, first_won},
            pl.col(first_won) > pl.col(first_in),
        )
        add_check(
            f"{side}_second_serve_wins_exceed_second_serve_attempts",
            {svpt, first_in, second_won},
            pl.col(second_won) > (pl.col(svpt) - pl.col(first_in)),
        )
        add_check(
            f"{side}_second_serve_wins_and_double_faults_exceed_attempts",
            {svpt, first_in, second_won, df},
            pl.col(second_won) + pl.col(df) > pl.col(svpt) - pl.col(first_in),
        )
        add_check(f"{side}_aces_exceed_serve_points_played", {svpt, ace}, pl.col(ace) > pl.col(svpt))
        add_check(
            f"{side}_double_faults_exceed_serve_points_played",
            {svpt, df},
            pl.col(df) > pl.col(svpt),
        )
        add_check(
            f"{side}_break_points_saved_exceed_faced",
            {bp_saved, bp_faced},
            pl.col(bp_saved) > pl.col(bp_faced),
        )

    add_check(
        "total_games_differs_from_player_games",
        {"winner_games", "loser_games", "total_games"},
        pl.col("winner_games") + pl.col("loser_games") != pl.col("total_games"),
    )
    add_check(
        "score_winner_did_not_win_more_sets",
        {"winner_sets", "loser_sets"},
        pl.col("winner_sets") <= pl.col("loser_sets"),
    )
    return checks


def audit_market_targets(table: MarketTargetTable) -> MarketTargetAudit:
    """Report impossible ranges and match-stat relationships without changing data."""

    numeric_columns = [
        column
        for column, dtype in table.targets.schema.items()
        if dtype.is_numeric() and dtype != pl.Boolean
    ]
    range_rows: list[dict[str, object]] = []
    for column in numeric_columns:
        values = table.targets[column]
        range_rows.append(
            {
                "target": column,
                "rows": len(values),
                "observed": values.count(),
                "missing": values.null_count(),
                "negative": int((values < 0).fill_null(False).sum()),
                "nonpositive": (
                    int((values <= 0).fill_null(False).sum())
                    if column == "match_duration"
                    else int((values < 0).fill_null(False).sum())
                ),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )

    consistency_rows = []
    violation_frames = []
    for name, _, expression in _consistency_checks(set(table.targets.columns)):
        mask = table.targets.select(expression).to_series()
        violations = mask.fill_null(False)
        count = int(violations.sum() or 0)
        consistency_rows.append(
            {"check": name, "rows_checked": mask.count(), "violations": count}
        )
        if count:
            violation_frames.append(
                table.metadata.filter(violations).with_columns(pl.lit(name).alias("check"))
            )

    empty_violations = table.metadata.head(0).with_columns(
        pl.Series("check", [], dtype=pl.String)
    )
    return MarketTargetAudit(
        ranges=pl.DataFrame(range_rows),
        consistency=pl.DataFrame(
            consistency_rows,
            schema={"check": pl.String, "rows_checked": pl.UInt32, "violations": pl.UInt32},
        ),
        violations=pl.concat(violation_frames) if violation_frames else empty_violations,
    )


def market_target_eligibility(table: MarketTargetTable) -> pl.DataFrame:
    """Return one boolean per available market and source match for label selection.

    A paired market requires both players' outcomes. If a consistency check
    fails, none of its inputs can be trusted in isolation, so markets using any
    of those inputs are ineligible for that match. An unevaluable check does
    not invalidate an otherwise observed target. Use these flags only to select
    post-match labels, never as pre-match features.
    """

    sides = ("winner", "loser")
    sets = ("winner_sets", "loser_sets")
    games = ("winner_games", "loser_games")
    outcome_columns = {
        market: tuple(f"{side}_{market}" for side in sides)
        for market in (*_DIRECT_PLAYER_TARGETS, "break_points_converted", "return_points_won")
    }
    outcome_columns.update(
        {
            "set_score": ("correct_set_score", *sets),
            "set_winner": sets,
            "correct_set_score": ("correct_set_score",),
            "sets_handicap": sets,
            "total_games": ("total_games",),
            "player_games": games,
            "games_handicap": games,
            "match_duration": ("match_duration",),
        }
    )
    dependencies = {
        "correct_set_score": sets,
        "total_games": games,
        "return_points_won": tuple(
            f"{side}_{stat}"
            for side in sides
            for stat in ("serve_points_played", "first_serve_points_won", "second_serve_points_won")
        ),
        "break_points_converted": tuple(
            f"{side}_{stat}"
            for side in sides
            for stat in ("break_points_faced", "break_points_saved")
        ),
    }
    columns = set(table.targets.columns)
    checks = _consistency_checks(columns)
    flags = []
    for market in table.definitions:
        outcomes = outcome_columns[market.name]
        inputs = set(outcomes).union(dependencies.get(market.name, ())) & columns
        if not set(outcomes) <= columns:
            flags.append(pl.lit(False).alias(market.name))
            continue
        valid = [pl.col(column).is_not_null() for column in outcomes]
        for column in sorted(inputs):
            if table.targets.schema[column].is_numeric():
                bound = pl.col(column) > 0 if column == "match_duration" else pl.col(column) >= 0
                valid.append(bound.fill_null(False))
        valid.extend(~violation.fill_null(False) for _, fields, violation in checks if fields & inputs)
        flags.append(pl.all_horizontal(valid).alias(market.name))

    return table.targets.with_columns(table.metadata["match_index"]).select("match_index", *flags)


def _validate_split_date(value: int, name: str) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer in YYYYMMDD format")
    try:
        datetime.strptime(str(value), "%Y%m%d")
    except ValueError as error:
        raise ValueError(f"{name} must be a valid date in YYYYMMDD format") from error


def split_market_target_table(
    table: MarketTargetTable,
    train_end_date: int,
    validation_end_date: int,
) -> MarketTargetSplits:
    """Split complete source matches chronologically using inclusive cutoffs.

    ``tourney_date`` is a tournament-level date, not an exact match timestamp.
    Every source match is represented once, so its outcomes cannot cross splits.
    """

    _validate_split_date(train_end_date, "train_end_date")
    _validate_split_date(validation_end_date, "validation_end_date")
    if validation_end_date <= train_end_date:
        raise ValueError("validation_end_date must be later than train_end_date")
    if table.metadata["match_index"].null_count():
        raise ValueError("match_index cannot contain null values")
    if table.metadata["match_index"].n_unique() != table.metadata.height:
        raise ValueError("market target table must contain one row per source match")
    if "tourney_date" not in table.metadata.columns:
        raise ValueError("market target metadata must contain tourney_date")

    dates = table.metadata["tourney_date"].to_list()
    for date in dates:
        if date is not None:
            _validate_split_date(date, "tourney_date")

    masks = {
        "train": [date is not None and date <= train_end_date for date in dates],
        "validation": [date is not None and train_end_date < date <= validation_end_date for date in dates],
        "test": [date is not None and date > validation_end_date for date in dates],
        "undated": [date is None for date in dates],
    }

    def subset(mask: list[bool]) -> MarketTargetTable:
        selection = pl.Series("_split_mask", mask, dtype=pl.Boolean)
        return MarketTargetTable(
            metadata=table.metadata.filter(selection),
            targets=table.targets.filter(selection),
            definitions=table.definitions,
        )

    return MarketTargetSplits(**{name: subset(mask) for name, mask in masks.items()})
