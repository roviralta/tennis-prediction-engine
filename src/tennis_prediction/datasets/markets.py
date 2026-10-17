from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class CandidateMarket:
    """A candidate outcome with its raw archive sources and construction limits."""

    name: str
    target_level: str
    unit: str
    definition: str
    source_columns: tuple[str, ...]
    requires_parser: bool = False


@dataclass(frozen=True)
class MarketCoverageReport:
    """Overall and tournament-year source coverage for candidate outcomes."""

    summary: pl.DataFrame
    by_year: pl.DataFrame
    chronology: pl.DataFrame


_PLAYER_STAT_MARKETS = (
    CandidateMarket(
        name="aces",
        target_level="player_match",
        unit="count",
        definition="Aces recorded for the winner and loser; orient to player perspective.",
        source_columns=("w_ace", "l_ace"),
    ),
    CandidateMarket(
        name="double_faults",
        target_level="player_match",
        unit="count",
        definition="Double faults recorded for the winner and loser; orient to player perspective.",
        source_columns=("w_df", "l_df"),
    ),
    CandidateMarket(
        name="serve_points_played",
        target_level="player_match",
        unit="count",
        definition="Serve points recorded for the winner and loser.",
        source_columns=("w_svpt", "l_svpt"),
    ),
    CandidateMarket(
        name="first_serves_in",
        target_level="player_match",
        unit="count",
        definition="First serves made by the winner and loser.",
        source_columns=("w_1stIn", "l_1stIn"),
    ),
    CandidateMarket(
        name="first_serve_points_won",
        target_level="player_match",
        unit="count",
        definition="First-serve points won by the winner and loser.",
        source_columns=("w_1stWon", "l_1stWon"),
    ),
    CandidateMarket(
        name="second_serve_points_won",
        target_level="player_match",
        unit="count",
        definition="Second-serve points won by the winner and loser.",
        source_columns=("w_2ndWon", "l_2ndWon"),
    ),
    CandidateMarket(
        name="serve_games_recorded",
        target_level="player_match",
        unit="count",
        definition=(
            "Serve games recorded by the archive; confirm exact target interpretation "
            "before modeling."
        ),
        source_columns=("w_SvGms", "l_SvGms"),
    ),
    CandidateMarket(
        name="break_points_saved",
        target_level="player_match",
        unit="count",
        definition="Break points saved by the winner and loser.",
        source_columns=("w_bpSaved", "l_bpSaved"),
    ),
    CandidateMarket(
        name="break_points_faced",
        target_level="player_match",
        unit="count",
        definition="Break points faced by the winner and loser.",
        source_columns=("w_bpFaced", "l_bpFaced"),
    ),
    CandidateMarket(
        name="break_points_converted",
        target_level="player_match",
        unit="count",
        definition="Opponent break points faced minus saved, oriented to the returning player.",
        source_columns=("w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"),
    ),
    CandidateMarket(
        name="return_points_won",
        target_level="player_match",
        unit="count",
        definition=(
            "Opponent serve points minus opponent first- and second-serve points won, "
            "oriented to the returning player."
        ),
        source_columns=("w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"),
    ),
)

CANDIDATE_MARKETS = (
    *_PLAYER_STAT_MARKETS,
    CandidateMarket(
        name="match_duration",
        target_level="match",
        unit="minutes",
        definition="Recorded match duration in minutes.",
        source_columns=("minutes",),
    ),
    CandidateMarket(
        name="set_score",
        target_level="match",
        unit="score",
        definition="Set-by-set score parsed from the archive score string.",
        source_columns=("score",),
        requires_parser=True,
    ),
    CandidateMarket(
        name="set_winner",
        target_level="player_match",
        unit="sets",
        definition="Sets won by each player derived from a validated set-by-set score parse.",
        source_columns=("score",),
        requires_parser=True,
    ),
    CandidateMarket(
        name="correct_set_score",
        target_level="match",
        unit="score",
        definition="Correct set score derived from a validated set-by-set score parse.",
        source_columns=("score",),
        requires_parser=True,
    ),
    CandidateMarket(
        name="total_games",
        target_level="match",
        unit="games",
        definition="Total games derived from a validated set-by-set score parse.",
        source_columns=("score",),
        requires_parser=True,
    ),
    CandidateMarket(
        name="player_games",
        target_level="player_match",
        unit="games",
        definition="Games won by each player derived from a validated score parse.",
        source_columns=("score",),
        requires_parser=True,
    ),
    CandidateMarket(
        name="games_handicap",
        target_level="player_match",
        unit="games",
        definition="Per-player game margin derived from a validated score parse.",
        source_columns=("score",),
        requires_parser=True,
    ),
    CandidateMarket(
        name="sets_handicap",
        target_level="player_match",
        unit="sets",
        definition="Per-player set margin derived from a validated set-by-set score parse.",
        source_columns=("score",),
        requires_parser=True,
    ),
)


def candidate_markets() -> tuple[CandidateMarket, ...]:
    """Return candidate markets in a stable, reviewable order."""

    return CANDIDATE_MARKETS


def _coverage_row(
    matches: pl.DataFrame,
    market: CandidateMarket,
) -> dict[str, object]:
    missing_columns = [
        column for column in market.source_columns if column not in matches.columns
    ]
    row: dict[str, object] = {
        "market": market.name,
        "target_level": market.target_level,
        "unit": market.unit,
        "definition": market.definition,
        "source_columns": ",".join(market.source_columns),
        "matches_total": matches.height,
        "matches_with_complete_sources": 0,
        "matches_with_invalid_source_values": 0,
        "source_coverage": None,
        "missing_columns": ",".join(missing_columns),
        "source_dtypes": "",
        "requires_parser": market.requires_parser,
        "status": "missing_source_columns" if missing_columns else "available_for_audit",
    }
    if missing_columns:
        return row

    numeric_source_columns = [
        column
        for column in market.source_columns
        if column != "score" and not matches.schema[column].is_numeric()
    ]
    unsupported_source_columns = [
        column
        for column in numeric_source_columns
        if matches.schema[column] != pl.String
    ]
    if unsupported_source_columns:
        row["status"] = "unsupported_source_dtype"
        row["source_dtypes"] = ",".join(
            f"{column}:{matches.schema[column]}" for column in unsupported_source_columns
        )
        return row

    if matches.height == 0:
        row["source_coverage"] = 0.0
        return row

    numeric_expressions = {
        column: pl.col(column).cast(pl.Float64, strict=False)
        for column in market.source_columns
        if column != "score"
    }
    valid_numeric_expressions = {
        column: (value.is_finite() & (value == value.floor())).fill_null(False)
        for column, value in numeric_expressions.items()
    }
    complete_expressions = [
        valid_numeric_expressions[column]
        if column in numeric_expressions
        else pl.col(column).is_not_null()
        for column in market.source_columns
    ]
    complete_sources = pl.all_horizontal(complete_expressions)
    invalid_expressions = [
        pl.col(column).is_not_null() & ~valid_numeric_expressions[column]
        for column in numeric_expressions
    ]
    usable = int(matches.select(complete_sources.sum()).item() or 0)
    invalid = (
        int(matches.select(pl.any_horizontal(invalid_expressions).sum()).item() or 0)
        if invalid_expressions
        else 0
    )
    row["matches_with_complete_sources"] = usable
    row["matches_with_invalid_source_values"] = invalid
    row["source_coverage"] = usable / matches.height
    row["source_dtypes"] = ",".join(
        f"{column}:{matches.schema[column]}" for column in numeric_source_columns
    )
    if market.requires_parser:
        row["status"] = "parser_required"
    elif numeric_source_columns:
        row["status"] = (
            "numeric_strings_parsed" if invalid == 0 else "invalid_numeric_strings_present"
        )
    else:
        row["status"] = "source_coverage_measured"
    return row


def audit_market_coverage(matches: pl.DataFrame) -> MarketCoverageReport:
    """Measure candidate-target source completeness overall and by tournament year.

    Coverage means every required numeric source parses to a value and every
    required text source is non-null. It does not validate score parsing,
    numerical ranges, statistical consistency, or whether target values are
    available before a match; these outcomes must only be labels or prior history.
    Tournament years come from ``tourney_date``, not archive filenames.
    """

    if "tourney_date" not in matches.columns:
        raise ValueError("Missing required match date column: tourney_date")

    summary_rows = [_coverage_row(matches, market) for market in CANDIDATE_MARKETS]
    summary = pl.DataFrame(summary_rows, infer_schema_length=None)

    dated = matches.filter(pl.col("tourney_date").is_not_null()).with_columns(
        pl.col("tourney_date")
        .cast(pl.String)
        .str.slice(0, 4)
        .cast(pl.Int32, strict=False)
        .alias("tournament_year")
    )
    date_numbers = (
        matches["tourney_date"]
        .cast(pl.String)
        .str.replace_all(r"\D", "")
        .str.slice(0, 8)
        .cast(pl.Int64, strict=False)
    )
    valid_dates = date_numbers.drop_nulls()
    chronology = pl.DataFrame(
        {
            "matches_total": [matches.height],
            "matches_with_date": [valid_dates.len()],
            "matches_without_valid_date": [matches.height - valid_dates.len()],
            "earliest_tournament_date": [valid_dates.min()],
            "latest_tournament_date": [valid_dates.max()],
            "date_limitation": [
                "tourney_date is tournament-level; within-day match order is unknown"
            ],
        }
    )
    year_rows: list[dict[str, object]] = []
    for year_group in dated.partition_by("tournament_year", maintain_order=True):
        year = year_group["tournament_year"][0]
        for market in CANDIDATE_MARKETS:
            row = _coverage_row(year_group, market)
            year_rows.append(
                {
                    "tournament_year": year,
                    "market": row["market"],
                    "matches_total": row["matches_total"],
                    "matches_with_complete_sources": row["matches_with_complete_sources"],
                    "source_coverage": row["source_coverage"],
                    "status": row["status"],
                }
            )

    by_year = pl.DataFrame(year_rows) if year_rows else pl.DataFrame(
        schema={
            "tournament_year": pl.Int32,
            "market": pl.String,
            "matches_total": pl.Int64,
            "matches_with_complete_sources": pl.Int64,
            "source_coverage": pl.Float64,
            "status": pl.String,
        }
    )
    return MarketCoverageReport(summary=summary, by_year=by_year, chronology=chronology)
