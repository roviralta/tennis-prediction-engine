import polars as pl

from tennis_prediction.features.head_to_head import add_head_to_head_features


def test_head_to_head_counts_prior_meetings_and_orients_wins_to_players() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [20250101, 20250102, 20250103, 20250104, 20250105],
            "winner_id": [1, 2, 1, 3, 2],
            "loser_id": [2, 1, 2, 1, 1],
        }
    )
    matches = pl.DataFrame(
        {
            "player_a": [1, 2],
            "player_b": [2, 1],
            "tourney_date": [20250104, 20250104],
        }
    )

    result = add_head_to_head_features(matches, history)

    assert result["head_to_head_matches"].to_list() == [3, 3]
    assert result["player_a_head_to_head_wins"].to_list() == [2, 1]
    assert result["player_b_head_to_head_wins"].to_list() == [1, 2]
    assert result["player_a_head_to_head_win_rate"].to_list() == [2 / 3, 1 / 3]
    assert result["player_b_head_to_head_win_rate"].to_list() == [1 / 3, 2 / 3]
    assert result["head_to_head_win_rate_difference"].to_list() == [1 / 3, -1 / 3]


def test_head_to_head_excludes_same_date_and_future_matches() -> None:
    history = pl.DataFrame(
        {
            "tourney_date": [20250101, 20250102, 20250103],
            "winner_id": [1, 2, 1],
            "loser_id": [2, 1, 2],
        }
    )
    matches = pl.DataFrame(
        {"player_a": [1], "player_b": [2], "tourney_date": [20250102]}
    )

    result = add_head_to_head_features(matches, history)

    assert result["head_to_head_matches"].to_list() == [1]
    assert result["player_a_head_to_head_wins"].to_list() == [1]
    assert result["player_a_head_to_head_win_rate"].to_list() == [1.0]


def test_first_meeting_has_zero_count_and_null_rates() -> None:
    history = pl.DataFrame(
        {"tourney_date": [20250101], "winner_id": [3], "loser_id": [4]}
    )
    matches = pl.DataFrame(
        {"player_a": [1], "player_b": [2], "tourney_date": [20250102]}
    )

    result = add_head_to_head_features(matches, history)

    assert result["head_to_head_matches"].to_list() == [0]
    assert result["player_a_head_to_head_wins"].to_list() == [0]
    assert result["player_b_head_to_head_wins"].to_list() == [0]
    assert result["player_a_head_to_head_win_rate"].to_list() == [None]
    assert result["player_b_head_to_head_win_rate"].to_list() == [None]
    assert result["head_to_head_win_rate_difference"].to_list() == [None]


def test_missing_dates_and_player_ids_produce_null_statistics() -> None:
    history = pl.DataFrame(
        {"tourney_date": [20250101], "winner_id": [1], "loser_id": [2]}
    )
    matches = pl.DataFrame(
        {
            "player_a": [1, None],
            "player_b": [2, 3],
            "tourney_date": [None, 20250102],
        },
        schema={
            "player_a": pl.Int64,
            "player_b": pl.Int64,
            "tourney_date": pl.Int64,
        },
    )

    result = add_head_to_head_features(matches, history)

    assert result["head_to_head_matches"].to_list() == [None, None]
    assert result["player_a_head_to_head_win_rate"].to_list() == [None, None]

