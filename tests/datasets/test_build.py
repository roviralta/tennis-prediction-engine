import polars as pl
import pytest

from tennis_prediction.datasets.build import build_historical_prediction_table


def test_build_historical_prediction_table_assembles_pre_match_features() -> None:
    matches = pl.DataFrame(
        {
            "tourney_id": ["A", "B"],
            "tourney_name": ["A", "B"],
            "surface": ["Hard", "Hard"],
            "draw_size": [32, 32],
            "tourney_level": ["A", "A"],
            "tourney_date": [20240101, 20240103],
            "match_num": [1, 1],
            "best_of": [3, 3],
            "round": ["R32", "R32"],
            "winner_id": [1, 2],
            "loser_id": [2, 1],
        }
    )
    players = pl.DataFrame(
        {
            "player_id": [1, 2],
            "hand": ["R", "L"],
            "height": [180, 185],
            "dob": [19900101, 19920101],
        }
    )
    rankings = pl.DataFrame(
        {
            "ranking_date": [20231225, 20231225],
            "rank": [10, 20],
            "player": [1, 2],
            "points": [2000, 1000],
        }
    )

    table = build_historical_prediction_table(matches, players, rankings, window=1)

    assert table.metadata.height == 4
    assert table.features.columns[0:5] == [
        "surface",
        "draw_size",
        "tourney_level",
        "best_of",
        "round",
    ]
    assert table.features["player_a_age"].to_list() == [34, 32, 32, 34]
    assert table.features["player_a_rank"].to_list() == [10, 20, 20, 10]
    # Both views of the second match see the first match as prior history.
    assert table.features["head_to_head_matches"].to_list() == [0, 0, 1, 1]
    assert table.features["player_a_win_rate_last_1"].to_list() == [None, None, 0.0, 1.0]
    assert table.target.to_list() == [1, 0, 1, 0]


def test_build_historical_prediction_table_requires_full_history_schema() -> None:
    with pytest.raises(ValueError, match="winner_id"):
        build_historical_prediction_table(
            pl.DataFrame({"tourney_date": [20240101]}),
            pl.DataFrame(),
            pl.DataFrame(),
        )
