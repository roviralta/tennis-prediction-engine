from bisect import bisect_left
from collections import defaultdict

import polars as pl


HISTORY_COLUMNS = ["tourney_date", "winner_id", "loser_id"]


def _missing_columns(frame: pl.DataFrame, required_columns: list[str]) -> list[str]:
    return [column for column in required_columns if column not in frame.columns]


def _build_head_to_head_index(
    history: pl.DataFrame,
) -> dict[tuple[int, int], tuple[list[int], list[int]]]:
    """Index each unordered pair by date and cumulative wins by lower ID."""

    pair_matches: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)

    for source_order, (match_date, winner_id, loser_id) in enumerate(
        history.select(HISTORY_COLUMNS).iter_rows()
    ):
        if match_date is None or winner_id is None or loser_id is None:
            continue
        if winner_id == loser_id:
            continue

        pair = tuple(sorted((winner_id, loser_id)))
        winner_is_lower_id = int(winner_id == pair[0])
        pair_matches[pair].append((match_date, source_order, winner_is_lower_id))

    index = {}
    for pair, matches in pair_matches.items():
        # Same-date ordering is unknown; source order provides only a stable
        # deterministic tie-break for keeping all matches in the cumulative count.
        matches.sort(key=lambda item: (item[0], item[1]))
        dates = [match_date for match_date, _, _ in matches]
        lower_id_win_prefix = [0]
        for _, _, winner_is_lower_id in matches:
            lower_id_win_prefix.append(
                lower_id_win_prefix[-1] + winner_is_lower_id
            )
        index[pair] = (dates, lower_id_win_prefix)

    return index


def add_head_to_head_features(
    matches: pl.DataFrame,
    history: pl.DataFrame,
) -> pl.DataFrame:
    """Add prior head-to-head counts and rates using strictly earlier dates.

    The history contains one row per source match. For each target row, all
    meetings with the same unordered player pair before its tournament date are
    counted. Same-date and future meetings are excluded because exact match
    timestamps are unavailable. A missing target date or player ID yields null
    statistics; a first meeting has zero prior matches and a null win rate.
    """

    required_match_columns = ["player_a", "player_b", "tourney_date"]
    missing_match_columns = _missing_columns(matches, required_match_columns)
    if missing_match_columns:
        raise ValueError(f"Missing match columns: {missing_match_columns}")

    missing_history_columns = _missing_columns(history, HISTORY_COLUMNS)
    if missing_history_columns:
        raise ValueError(f"Missing required columns: {missing_history_columns}")

    pair_index = _build_head_to_head_index(history)
    cache: dict[
        tuple[int | None, int | None, int | None],
        tuple[int | None, int | None, int | None, float | None, float | None],
    ] = {}
    prior_match_counts: list[int | None] = []
    player_a_wins: list[int | None] = []
    player_b_wins: list[int | None] = []
    player_a_rates: list[float | None] = []
    player_b_rates: list[float | None] = []

    for player_a, player_b, match_date in matches.select(
        required_match_columns
    ).iter_rows():
        key = (player_a, player_b, match_date)
        if key not in cache:
            if (
                player_a is None
                or player_b is None
                or match_date is None
                or player_a == player_b
            ):
                values = (None, None, None, None, None)
            else:
                pair = tuple(sorted((player_a, player_b)))
                dates, lower_id_win_prefix = pair_index.get(pair, ([], [0]))
                eligible_count = bisect_left(dates, match_date)
                lower_id_wins = lower_id_win_prefix[eligible_count]
                if player_a == pair[0]:
                    wins_a = lower_id_wins
                else:
                    wins_a = eligible_count - lower_id_wins
                wins_b = eligible_count - wins_a
                rate_a = wins_a / eligible_count if eligible_count else None
                rate_b = wins_b / eligible_count if eligible_count else None
                values = (eligible_count, wins_a, wins_b, rate_a, rate_b)
            cache[key] = values

        count, wins_a, wins_b, rate_a, rate_b = cache[key]
        prior_match_counts.append(count)
        player_a_wins.append(wins_a)
        player_b_wins.append(wins_b)
        player_a_rates.append(rate_a)
        player_b_rates.append(rate_b)

    return matches.with_columns(
        pl.Series("head_to_head_matches", prior_match_counts, dtype=pl.Int64),
        pl.Series("player_a_head_to_head_wins", player_a_wins, dtype=pl.Int64),
        pl.Series("player_b_head_to_head_wins", player_b_wins, dtype=pl.Int64),
        pl.Series("player_a_head_to_head_win_rate", player_a_rates, dtype=pl.Float64),
        pl.Series("player_b_head_to_head_win_rate", player_b_rates, dtype=pl.Float64),
    ).with_columns(
        (
            pl.col("player_a_head_to_head_win_rate")
            - pl.col("player_b_head_to_head_win_rate")
        ).alias("head_to_head_win_rate_difference")
    )
