import polars as pl

from tennis_prediction.features.form import calculate_recent_form
from tennis_prediction.features.form import add_recent_form_features


def test_calculate_recent_win_rate() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [
                20250101,
                20250102,
                20250103,
                20250104,
                20250105,
            ],
            "winner_id": [1, 1, 2, 1, 2],
            "loser_id": [2, 3, 1, 4, 1],
        }
    )

    result = calculate_recent_form(
        history=history,
        player_id=1,
        match_date=20250106,
        window=5,
    )

    assert result["matches_played"] == 5
    assert result["wins"] == 3
    assert result["win_rate"] == 0.6


def test_recent_form_uses_available_matches() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [20250101, 20250102, 20250103],
            "winner_id": [1, 2, 1],
            "loser_id": [2, 1, 3],
        }
    )

    result = calculate_recent_form(
        history=history,
        player_id=1,
        match_date=20250104,
        window=10,
    )

    assert result["matches_played"] == 3
    assert result["wins"] == 2
    assert result["win_rate"] == 2 / 3


def test_recent_form_excludes_same_date() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [20250101, 20250102, 20250103],
            "winner_id": [1, 2, 1],
            "loser_id": [2, 1, 3],
        }
    )

    result = calculate_recent_form(
        history=history,
        player_id=1,
        match_date=20250103,
        window=10,
    )

    assert result["matches_played"] == 2
    assert result["wins"] == 1
    assert result["win_rate"] == 0.5
    
def test_add_recent_form_features() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [
                20250101,
                20250102,
                20250103,
                20250104,
            ],
            "winner_id": [1, 1, 2, 2],
            "loser_id": [2, 3, 1, 1],
        }
    )

    matches = pl.DataFrame(
        {
            "player_a": [1],
            "player_b": [2],
            "tourney_date": [20250105],
        }
    )

    result = add_recent_form_features(
        matches=matches,
        history=history,
        window=10,
    )
    assert (
    result["player_b_win_rate_last_10"].to_list()
    == [2 / 3]
	)
    assert (
		result["win_rate_difference_last_10"].to_list()
		== [0.5 - 2 / 3]
	)

    
def test_recent_form_features_with_real_matches() -> None:
    from pathlib import Path

    from tennis_prediction.ingestion.atp.loader import ATPDataLoader
    from tennis_prediction.preprocessing.matches import normalize_matches

    project_root = Path(__file__).resolve().parents[2]

    loader = ATPDataLoader(
        project_root / "data/external/tennis-sackmann-archive/atp"
    )

    history = loader.load_matches(
        start_year=2024,
        end_year=2025,
    )

    matches_2025 = loader.load_matches(
        start_year=2025,
        end_year=2025,
    )

    normalized = normalize_matches(matches_2025)

    from tennis_prediction.features.form import add_recent_form_features

    result = add_recent_form_features(
        matches=normalized,
        history=history,
        window=10,
    )

    assert result.height == normalized.height
    assert "player_a_win_rate_last_10" in result.columns
    assert "player_b_win_rate_last_10" in result.columns
    assert "win_rate_difference_last_10" in result.columns

    assert result["player_a_win_rate_last_10"].drop_nulls().is_between(0.0, 1.0).all()
    assert result["player_b_win_rate_last_10"].drop_nulls().is_between(0.0, 1.0).all()


def test_configurable_window_names_and_excludes_same_date_and_future() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [20250101, 20250102, 20250103, 20250104],
            "winner_id": [1, 2, 1, 2],
            "loser_id": [2, 1, 3, 1],
        }
    )
    matches = pl.DataFrame(
        {
            "player_a": [1],
            "player_b": [2],
            "tourney_date": [20250103],
        }
    )

    result = add_recent_form_features(matches, history, window=2)

    assert result["player_a_win_rate_last_2"].to_list() == [0.5]
    assert result["player_b_win_rate_last_2"].to_list() == [0.5]
    assert result["win_rate_difference_last_2"].to_list() == [0.0]
    assert "player_a_win_rate_last_10" not in result.columns


def test_bulk_recent_form_reuses_history_safely_for_repeated_player_dates() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [20250101, 20250102, 20250103],
            "winner_id": [1, 2, 1],
            "loser_id": [2, 1, 3],
        }
    )
    matches = pl.DataFrame(
        {
            "player_a": [1, 1],
            "player_b": [2, 3],
            "tourney_date": [20250104, 20250104],
        }
    )

    result = add_recent_form_features(matches, history, window=10)

    assert result.height == matches.height
    assert result["player_a_win_rate_last_10"].to_list() == [2 / 3, 2 / 3]
    assert result["player_b_win_rate_last_10"].to_list() == [1 / 2, 0.0]


def test_recent_form_with_null_match_date_has_no_available_history() -> None:
    history = pl.DataFrame(
        {"tourney_date": [20250101], "winner_id": [1], "loser_id": [2]}
    )
    matches = pl.DataFrame(
        {"player_a": [1], "player_b": [2], "tourney_date": [None]},
        schema={"player_a": pl.Int64, "player_b": pl.Int64, "tourney_date": pl.Int64},
    )

    result = add_recent_form_features(matches, history)

    assert result["player_a_win_rate_last_10"].to_list() == [None]
    assert result["player_b_win_rate_last_10"].to_list() == [None]


def test_no_history_is_null_while_a_real_zero_win_rate_stays_zero() -> None:
    history = pl.DataFrame(
        {"tourney_date": [20250101], "winner_id": [3], "loser_id": [2]}
    )
    matches = pl.DataFrame(
        {"player_a": [1], "player_b": [2], "tourney_date": [20250102]}
    )

    result = add_recent_form_features(matches, history)

    assert result["player_a_win_rate_last_10"].to_list() == [None]
    assert result["player_b_win_rate_last_10"].to_list() == [0.0]
    assert result["win_rate_difference_last_10"].to_list() == [None]


def test_calculate_recent_form_returns_null_rate_without_prior_matches() -> None:
    history = pl.DataFrame(
        {"tourney_date": [20250101], "winner_id": [1], "loser_id": [2]}
    )

    result = calculate_recent_form(history, player_id=3, match_date=20250102)

    assert result == {"matches_played": 0, "wins": 0, "win_rate": None}


def test_window_cut_with_same_date_matches_uses_stable_source_order() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [20250101, 20250102, 20250102],
            "winner_id": [2, 1, 2],
            "loser_id": [1, 2, 1],
        }
    )
    matches = pl.DataFrame(
        {"player_a": [1], "player_b": [2], "tourney_date": [20250103]}
    )

    result = add_recent_form_features(matches, history, window=1)

    # Source order is only a deterministic tie-break; it does not imply
    # that the archive provides exact ordering within the tournament date.
    assert result["player_a_win_rate_last_1"].to_list() == [1.0]
