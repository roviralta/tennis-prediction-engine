from bisect import bisect_left
from collections import defaultdict

import polars as pl


HISTORY_COLUMNS = ["tourney_date", "winner_id", "loser_id"]


def _missing_columns(frame: pl.DataFrame, required_columns: list[str]) -> list[str]:
    return [column for column in required_columns if column not in frame.columns]


def _build_player_history(
    history: pl.DataFrame,
) -> dict[int, tuple[list[int], list[int]]]:
    """Build date-sorted match dates and win-count prefixes for each player."""

    player_matches: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for source_order, (match_date, winner_id, loser_id) in enumerate(
        history.select(HISTORY_COLUMNS).iter_rows()
    ):
        if match_date is None:
            continue
        if winner_id is not None:
            player_matches[winner_id].append((match_date, source_order, 1))
        if loser_id is not None and loser_id != winner_id:
            player_matches[loser_id].append((match_date, source_order, 0))

    indexed_history = {}
    for player_id, matches in player_matches.items():
        # The archive has no match timestamps. Reverse source order within each
        # date so a window cut is deterministic and matches stable descending
        # date sorting; this tie-break does not claim within-date chronology.
        matches.sort(key=lambda item: (item[0], -item[1]))
        dates = [match_date for match_date, _, _ in matches]
        win_prefix = [0]
        for _, _, won in matches:
            win_prefix.append(win_prefix[-1] + won)
        indexed_history[player_id] = (dates, win_prefix)

    return indexed_history


def _build_player_surface_history(
    history: pl.DataFrame,
) -> dict[tuple[int, str], tuple[list[int], list[int]]]:
    """Build date-sorted histories for each player and known surface."""

    player_matches: dict[tuple[int, str], list[tuple[int, int, int]]] = defaultdict(list)

    for source_order, (match_date, winner_id, loser_id, surface) in enumerate(
        history.select(HISTORY_COLUMNS + ["surface"]).iter_rows()
    ):
        if match_date is None or surface is None:
            continue
        if winner_id is not None:
            player_matches[(winner_id, surface)].append((match_date, source_order, 1))
        if loser_id is not None and loser_id != winner_id:
            player_matches[(loser_id, surface)].append((match_date, source_order, 0))

    indexed_history = {}
    for player_surface, matches in player_matches.items():
        matches.sort(key=lambda item: (item[0], -item[1]))
        dates = [match_date for match_date, _, _ in matches]
        win_prefix = [0]
        for _, _, won in matches:
            win_prefix.append(win_prefix[-1] + won)
        indexed_history[player_surface] = (dates, win_prefix)

    return indexed_history


def _recent_form_from_index(
    player_history: dict[object, tuple[list[int], list[int]]],
    player_id: object,
    match_date: int,
    window: int,
) -> dict[str, int | float | None]:
    if match_date is None:
        return {"matches_played": 0, "wins": 0, "win_rate": None}

    dates, win_prefix = player_history.get(player_id, ([], [0]))
    end = bisect_left(dates, match_date)
    start = max(0, end - window)
    matches_played = end - start
    wins = win_prefix[end] - win_prefix[start]

    return {
        "matches_played": matches_played,
        "wins": wins,
        "win_rate": wins / matches_played if matches_played else None,
    }


def calculate_recent_form(
    history: pl.DataFrame,
    player_id: int,
    match_date: int,
    window: int = 10,
) -> dict[str, int | float | None]:
    """Calculate a player's win rate using only previous tournament dates."""

    if window < 1:
        raise ValueError("window must be greater than zero")

    missing_columns = _missing_columns(history, HISTORY_COLUMNS)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if match_date is None:
        return {"matches_played": 0, "wins": 0, "win_rate": None}

    previous_matches = (
        history
        .filter(
            (pl.col("tourney_date") < match_date)
            & (
                (pl.col("winner_id") == player_id)
                | (pl.col("loser_id") == player_id)
            )
        )
        .sort("tourney_date", descending=True)
        .head(window)
    )

    matches_played = previous_matches.height
    wins = previous_matches.filter(pl.col("winner_id") == player_id).height
    win_rate = wins / matches_played if matches_played else None

    return {
        "matches_played": matches_played,
        "wins": wins,
        "win_rate": win_rate,
    }


def add_recent_form_features(
    matches: pl.DataFrame,
    history: pl.DataFrame,
    window: int = 10,
) -> pl.DataFrame:
    """Add recent win-rate features using only strictly earlier tournament dates.

    History is indexed once, then each distinct player/date query uses binary
    search over that player's ordered matches. Same-date and future history is
    excluded, so tournament-level dates do not imply match ordering.
    """

    if window < 1:
        raise ValueError("window must be greater than zero")

    required_match_columns = ["player_a", "player_b", "tourney_date"]
    missing_match_columns = _missing_columns(matches, required_match_columns)
    if missing_match_columns:
        raise ValueError(f"Missing match columns: {missing_match_columns}")

    missing_history_columns = _missing_columns(history, HISTORY_COLUMNS)
    if missing_history_columns:
        raise ValueError(f"Missing required columns: {missing_history_columns}")

    player_history = _build_player_history(history)
    form_cache: dict[tuple[int | None, int | None], float | None] = {}
    player_a_rates: list[float | None] = []
    player_b_rates: list[float | None] = []

    for player_a, player_b, match_date in matches.select(required_match_columns).iter_rows():
        for player_id, rates in (
            (player_a, player_a_rates),
            (player_b, player_b_rates),
        ):
            key = (player_id, match_date)
            if key not in form_cache:
                form_cache[key] = _recent_form_from_index(
                    player_history, player_id, match_date, window
                )["win_rate"]
            rates.append(form_cache[key])

    suffix = f"last_{window}"
    player_a_column = f"player_a_win_rate_{suffix}"
    player_b_column = f"player_b_win_rate_{suffix}"

    return matches.with_columns(
        pl.Series(player_a_column, player_a_rates, dtype=pl.Float64),
        pl.Series(player_b_column, player_b_rates, dtype=pl.Float64),
    ).with_columns(
        (pl.col(player_a_column) - pl.col(player_b_column)).alias(
            f"win_rate_difference_{suffix}"
        )
    )


def add_surface_recent_form_features(
    matches: pl.DataFrame,
    history: pl.DataFrame,
    window: int = 10,
) -> pl.DataFrame:
    """Add recent win rates on each match's surface using earlier dates only.

    Rows with an unknown surface or no prior matches on that surface receive a
    null rate. The feature uses the same deterministic source-order tie-break
    as general recent form when a window cuts through one tournament date.
    """

    if window < 1:
        raise ValueError("window must be greater than zero")

    required_match_columns = ["player_a", "player_b", "tourney_date", "surface"]
    missing_match_columns = _missing_columns(matches, required_match_columns)
    if missing_match_columns:
        raise ValueError(f"Missing match columns: {missing_match_columns}")

    missing_history_columns = _missing_columns(
        history, HISTORY_COLUMNS + ["surface"]
    )
    if missing_history_columns:
        raise ValueError(f"Missing required columns: {missing_history_columns}")

    player_surface_history = _build_player_surface_history(history)
    form_cache: dict[tuple[int | None, str | None, int | None], float | None] = {}
    player_a_rates: list[float | None] = []
    player_b_rates: list[float | None] = []

    for player_a, player_b, match_date, surface in matches.select(
        required_match_columns
    ).iter_rows():
        for player_id, rates in ((player_a, player_a_rates), (player_b, player_b_rates)):
            key = (player_id, surface, match_date)
            if key not in form_cache:
                if surface is None or match_date is None:
                    rate = None
                else:
                    rate = _recent_form_from_index(
                        player_surface_history,
                        (player_id, surface),
                        match_date,
                        window,
                    )["win_rate"]
                form_cache[key] = rate
            rates.append(form_cache[key])

    suffix = f"last_{window}"
    player_a_column = f"player_a_surface_win_rate_{suffix}"
    player_b_column = f"player_b_surface_win_rate_{suffix}"

    return matches.with_columns(
        pl.Series(player_a_column, player_a_rates, dtype=pl.Float64),
        pl.Series(player_b_column, player_b_rates, dtype=pl.Float64),
    ).with_columns(
        (pl.col(player_a_column) - pl.col(player_b_column)).alias(
            f"surface_win_rate_difference_{suffix}"
        )
    )
