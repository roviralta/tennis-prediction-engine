from datetime import datetime

import polars as pl

from tennis_prediction.datasets.catalog import candidate_pre_match_feature_columns
from tennis_prediction.features.form import add_recent_form_features
from tennis_prediction.features.form import add_surface_recent_form_features
from tennis_prediction.features.head_to_head import add_head_to_head_features
from tennis_prediction.preprocessing.players import enrich_player_information
from tennis_prediction.preprocessing.rankings import add_match_rankings
from tennis_prediction.preprocessing.rankings import prepare_rankings


def build_match_features(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    rankings: pl.DataFrame,
    player_a_id: int,
    player_b_id: int,
    match_date: int,
    *,
    surface: str | None = None,
    window: int = 10,
) -> pl.DataFrame:
    """Build one pre-match feature row using only history before ``match_date``."""

    if player_a_id == player_b_id:
        raise ValueError("player_a_id and player_b_id must identify different players")
    if window < 1:
        raise ValueError("window must be greater than zero")
    try:
        datetime.strptime(str(match_date), "%Y%m%d")
    except (TypeError, ValueError) as error:
        raise ValueError("match_date must be a valid YYYYMMDD date") from error

    required_history = {"tourney_date", "winner_id", "loser_id", "surface"}
    missing_history = sorted(required_history - set(matches.columns))
    if missing_history:
        raise ValueError(f"Missing match history columns: {missing_history}")

    upcoming = pl.DataFrame(
        {
            "player_a": [player_a_id],
            "player_b": [player_b_id],
            "tourney_date": [match_date],
            "surface": pl.Series("surface", [surface], dtype=pl.String),
            "draw_size": pl.Series("draw_size", [None], dtype=pl.Int64),
            "tourney_level": pl.Series("tourney_level", [None], dtype=pl.String),
            "best_of": pl.Series("best_of", [None], dtype=pl.Int64),
            "round": pl.Series("round", [None], dtype=pl.String),
        }
    )
    enriched = enrich_player_information(upcoming, players)
    ranked = add_match_rankings(enriched, prepare_rankings(rankings))
    history = matches.select("tourney_date", "winner_id", "loser_id", "surface")
    formed = add_recent_form_features(ranked, history, window=window)
    surface_formed = add_surface_recent_form_features(formed, history, window=window)
    with_head_to_head = add_head_to_head_features(surface_formed, history)
    return with_head_to_head.select(candidate_pre_match_feature_columns(window=window))
