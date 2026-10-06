import polars as pl

from tennis_prediction.preprocessing.matches import normalize_matches


def test_normalize_matches() -> None:
    matches = pl.DataFrame(
        {
            "tourney_id": ["2025-001"],
            "tourney_name": ["Test Open"],
            "surface": ["Hard"],
            "draw_size": [32],
            "tourney_level": ["A"],
            "tourney_date": [20250101],
            "match_num": [1],
            "winner_id": [100001],
            "loser_id": [100002],
            "best_of": [3],
            "round": ["F"],
        }
    )

    normalized = normalize_matches(matches)

    assert normalized.height == 2

    assert normalized["match_index"].to_list() == [0, 0]
    assert normalized["player_a"].to_list() == [100001, 100002]
    assert normalized["player_b"].to_list() == [100002, 100001]
    assert normalized["target"].to_list() == [1, 0]