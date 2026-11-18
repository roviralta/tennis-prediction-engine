import polars as pl
import pytest

from tennis_prediction.datasets.score_markets import build_score_market_splits
from tennis_prediction.models.statistics.baseline import evaluate_score_market_baselines


def test_score_market_splits_and_baselines_use_neutral_rows_and_training_only() -> None:
    matches = pl.DataFrame(
        {
            "tourney_id": ["A", "B", "C", "D", "E", "F"],
            "tourney_name": ["A", "B", "C", "D", "E", "F"],
            "surface": ["Hard", "Clay", "Clay", "Grass", "Hard", "Hard"],
            "draw_size": [32] * 6,
            "tourney_level": ["A"] * 6,
            "tourney_date": [20180101, 20181231, 20190101, 20211231, 20220101, None],
            "match_num": [1] * 6,
            "best_of": [3] * 6,
            "round": ["R32"] * 6,
            "winner_id": [1, 2, 2, 1, 2, 2],
            "loser_id": [2, 1, 1, 2, 1, 1],
            "score": ["6-0 6-0", "6-4 6-4", "6-4 6-4", "6-4 4-6 [10-8]", "6-2 1-0 RET", "6-4 6-4"],
        }
    )
    self_match = matches.head(1).with_columns(
        pl.lit(1, dtype=pl.Int64).alias("loser_id"),
        pl.lit(20180102, dtype=pl.Int64).alias("tourney_date"),
    )
    matches = pl.concat([matches.head(1), self_match, matches.slice(1)])
    players = pl.DataFrame(
        {"player_id": [1, 2], "hand": ["R", "L"], "height": [180, 185], "dob": [19900101, 19920101]}
    )
    rankings = pl.DataFrame(
        {"ranking_date": [20171225, 20171225], "rank": [10, 20], "player": [1, 2], "points": [2000, 1000]}
    )

    splits = build_score_market_splits(matches, players, rankings, 20181231, 20211231)
    assert splits.train.metadata["match_index"].to_list() == [0, 2]
    assert splits.validation.metadata["match_index"].to_list() == [3, 4]
    assert splits.test.metadata["match_index"].to_list() == [5]
    assert splits.undated.metadata["match_index"].to_list() == [6]
    assert splits.train.metadata["player_a"].to_list() == [1, 1]
    assert splits.train.features["player_a_rank"].to_list() == [10, 10]
    assert splits.train.features["player_a_win_rate_last_10"].to_list() == [None, 1.0]
    assert set(splits.train.features.columns).isdisjoint({"target", "score", "correct_set_score", "total_games"})
    assert splits.train.targets.rows() == [("2-0", 12), ("0-2", 20)]
    assert splits.validation.targets.rows() == [("0-2", 20), ("2-1", None)]
    assert splits.test.targets.rows() == [(None, None)]

    metrics = evaluate_score_market_baselines(splits)
    assert metrics["validation_matches"].to_list() == [2, 2, 1, 1]
    assert metrics["value"].to_list() == pytest.approx([1.0, 0.75, 4.0, 0.0])
