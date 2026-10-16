import polars as pl

from tennis_prediction.datasets.catalog import candidate_pre_match_feature_columns
from tennis_prediction.datasets.schema import PredictionTable
from tennis_prediction.datasets.schema import build_prediction_table
from tennis_prediction.features.form import add_recent_form_features
from tennis_prediction.features.form import add_surface_recent_form_features
from tennis_prediction.features.head_to_head import add_head_to_head_features
from tennis_prediction.preprocessing.matches import normalize_matches
from tennis_prediction.preprocessing.players import enrich_player_information
from tennis_prediction.preprocessing.rankings import add_match_rankings
from tennis_prediction.preprocessing.rankings import prepare_rankings


def build_historical_prediction_table(
    matches: pl.DataFrame,
    players: pl.DataFrame,
    rankings: pl.DataFrame,
    window: int = 10,
) -> PredictionTable:
    """Build candidate pre-match features from complete one-row-per-match history.

    Supply the full available source history so form and head-to-head features can
    look backwards across years. Their feature functions exclude same-date and
    future results. Ranking joins use rankings dated on or before each match.
    """

    if window < 1:
        raise ValueError("window must be greater than zero")

    history_columns = ["tourney_date", "winner_id", "loser_id", "surface"]
    missing_history_columns = [
        column for column in history_columns if column not in matches.columns
    ]
    if missing_history_columns:
        raise ValueError(f"Missing required match history columns: {missing_history_columns}")

    history = matches.select(history_columns)
    normalized = normalize_matches(matches)
    enriched = enrich_player_information(normalized, players)
    ranked = add_match_rankings(enriched, prepare_rankings(rankings))
    formed = add_recent_form_features(ranked, history, window=window)
    surface_formed = add_surface_recent_form_features(formed, history, window=window)
    featured = add_head_to_head_features(surface_formed, history)

    return build_prediction_table(
        featured,
        candidate_pre_match_feature_columns(window=window),
    )
