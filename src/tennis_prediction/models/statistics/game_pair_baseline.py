import numpy as np
import polars as pl

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import ScoreMarketTable
from tennis_prediction.models.statistics.score_model import _game_count_losses


_GAME_COLUMNS = ("player_a_games", "player_b_games")
_OUTCOMES = (*_GAME_COLUMNS, "total_games", "game_margin")


def _eligible_games(table: ScoreMarketTable) -> pl.DataFrame:
    return table.features.select("best_of").hstack(
        table.targets.select(_OUTCOMES)
    ).drop_nulls(_GAME_COLUMNS)


def fit_game_pair_reference(training: ScoreMarketTable) -> pl.DataFrame:
    """Estimate the empirical joint game-count distribution within each format."""
    pairs = _eligible_games(training)
    if pairs["best_of"].null_count():
        raise ValueError("Eligible training games require best_of")
    return (
        pairs.group_by("best_of", *_GAME_COLUMNS)
        .len(name="matches")
        .with_columns(
            (pl.col("matches") / pl.col("matches").sum().over("best_of")).alias("probability"),
            (pl.col("player_a_games") + pl.col("player_b_games")).alias("total_games"),
            (pl.col("player_a_games") - pl.col("player_b_games")).alias("game_margin"),
        )
        .sort("best_of", *_GAME_COLUMNS)
    )


def evaluate_game_pair_reference(splits: ScoreMarketSplits) -> pl.DataFrame:
    """Evaluate each marginal of the same joint reference on validation only."""
    reference = fit_game_pair_reference(splits.train)
    validation = _eligible_games(splits.validation)
    if validation.is_empty() or validation["best_of"].null_count():
        raise ValueError("Validation requires eligible game pairs with best_of")

    rows = []
    for cohort in validation.partition_by("best_of"):
        best_of = cohort["best_of"][0]
        pairs = reference.filter(pl.col("best_of") == best_of)
        if pairs.is_empty():
            raise ValueError(f"Missing training game pairs for best_of={best_of}")
        weights = pairs["matches"].to_numpy()
        training_matches = int(weights.sum())
        for market in _OUTCOMES:
            counts = pairs[market].to_numpy()
            support = np.arange(counts.min(), counts.max() + 1)
            cdf = np.bincount(counts - support[0], weights=weights).cumsum() / training_matches
            observed = cohort[market].to_numpy()
            crps, mae = _game_count_losses(
                np.broadcast_to(cdf, (observed.size, support.size)), support, observed
            )
            rows.append({
                "best_of": best_of,
                "market": market,
                "training_matches": training_matches,
                "validation_matches": observed.size,
                "crps": float(crps.mean()),
                "median_mae": float(mae.mean()),
            })
    return pl.DataFrame(rows).sort("best_of", "market")
