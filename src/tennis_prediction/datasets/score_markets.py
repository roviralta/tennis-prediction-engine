from dataclasses import dataclass

import polars as pl

from tennis_prediction.datasets.build import build_historical_prediction_table
from tennis_prediction.datasets.market_targets import MarketTargetTable
from tennis_prediction.datasets.market_targets import build_market_target_table
from tennis_prediction.datasets.market_targets import market_target_eligibility
from tennis_prediction.datasets.market_targets import split_market_target_table
from tennis_prediction.datasets.schema import METADATA_COLUMNS


@dataclass(frozen=True)
class ScoreMarketTable:
    """One row per match, with player A ordered by ID and outcomes stored separately."""

    metadata: pl.DataFrame
    features: pl.DataFrame
    targets: pl.DataFrame

    def __post_init__(self) -> None:
        if len({self.metadata.height, self.features.height, self.targets.height}) != 1:
            raise ValueError("score market metadata, features, and targets must align")


@dataclass(frozen=True)
class ScoreMarketSplits:
    train: ScoreMarketTable
    validation: ScoreMarketTable
    test: ScoreMarketTable
    undated: ScoreMarketTable


def _assemble_score_market_table(
    pre_match_rows: pl.DataFrame,
    market: MarketTargetTable,
    feature_columns: list[str],
    source_indices: pl.Series,
) -> ScoreMarketTable:
    eligible = market_target_eligibility(market)
    rows = (
        market.metadata.select("match_index", "winner_id")
        .hstack(market.targets.select("winner_sets", "loser_sets", "total_games"))
        .hstack(
            eligible.select(
                pl.col("correct_set_score").alias("_eligible_set_score"),
                pl.col("total_games").alias("_eligible_total_games"),
            )
        )
        .join(pre_match_rows, on="match_index", how="inner", validate="1:1", maintain_order="left")
    )
    if rows.height != market.metadata.height:
        raise ValueError("each score market match must have one distinct-player pre-match row")

    player_a_won = pl.col("player_a") == pl.col("winner_id")
    player_a_sets = pl.when(player_a_won).then(pl.col("winner_sets")).otherwise(pl.col("loser_sets"))
    player_b_sets = pl.when(player_a_won).then(pl.col("loser_sets")).otherwise(pl.col("winner_sets"))
    targets = rows.select(
        pl.when(pl.col("_eligible_set_score"))
        .then(pl.concat_str(player_a_sets, player_b_sets, separator="-"))
        .alias("correct_set_score"),
        pl.when(pl.col("_eligible_total_games"))
        .then(pl.col("total_games"))
        .alias("total_games"),
    )
    return ScoreMarketTable(
        metadata=rows.select(METADATA_COLUMNS).with_columns(
            source_indices.gather(rows["match_index"])
        ),
        features=rows.select(feature_columns),
        targets=targets,
    )


def build_score_market_splits(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    rankings: pl.DataFrame,
    train_end_date: int,
    validation_end_date: int,
    window: int = 10,
) -> ScoreMarketSplits:
    """Build chronological score-market datasets from the same source match order.

    Keep rows with unavailable labels so either market can select its own
    eligible matches. Build history features from matches with distinct players,
    including those without labels, under the historical pipeline's strict date rules.
    """

    required = ("score", "winner_id", "loser_id")
    missing = [column for column in required if column not in matches.columns]
    if missing:
        raise ValueError(f"Missing required score market columns: {missing}")

    # Preserve archive row identity when dropping rows without distinct players.
    indexed = matches.with_row_index("match_index").filter(
        (pl.col("winner_id") != pl.col("loser_id")).fill_null(True)
    )
    source_indices = indexed["match_index"]
    distinct_matches = indexed.drop("match_index")
    market_splits = split_market_target_table(
        build_market_target_table(distinct_matches), train_end_date, validation_end_date
    )
    historical = build_historical_prediction_table(distinct_matches, players, rankings, window)
    pre_match_rows = historical.metadata.hstack(historical.features).filter(
        pl.col("player_a") < pl.col("player_b")
    )

    def assemble(market: MarketTargetTable) -> ScoreMarketTable:
        return _assemble_score_market_table(
            pre_match_rows, market, historical.features.columns, source_indices
        )

    return ScoreMarketSplits(
        train=assemble(market_splits.train),
        validation=assemble(market_splits.validation),
        test=assemble(market_splits.test),
        undated=assemble(market_splits.undated),
    )
