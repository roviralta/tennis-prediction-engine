import polars as pl
import pytest

from tennis_prediction.datasets.score_markets import build_score_market_splits
from tennis_prediction.models.statistics.baseline import evaluate_score_market_baselines
from tennis_prediction.models.statistics.game_pair_baseline import evaluate_game_pair_reference
from tennis_prediction.models.statistics.game_pair_baseline import fit_game_pair_reference


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
            "w_ace": [2, 4, 6, 8, 10, 12],
            "l_ace": [1, 3, 5, 7, 9, 11],
            "w_df": [1, 2, 3, 4, 5, 6],
            "l_df": [0, 1, 2, 3, 4, 5],
            "minutes": [55, 60, 65, 80, 40, 0],
            "w_bpFaced": [3, 5, 3, 3, 3, 3],
            "w_bpSaved": [3, 3, 2, 2, 2, 2],
            "l_bpFaced": [5, 6, 5, 5, 5, 5],
            "l_bpSaved": [2, 2, 2, 2, 2, 2],
            "w_SvGms": [6, 10, 10, 10, 10, 10],
            "l_SvGms": [6, 10, 1, 10, 10, 10],
            "score": ["6-0 6-0", "6-4 6-4", "6-4 6-4", "6-4 4-6 [10-8]", "6-2 1-0 RET", "7-6 0-6 7-6"],
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
    score_columns = ("correct_set_score", "total_games", "player_a_games", "player_b_games", "game_margin")
    assert splits.train.targets.select(score_columns).rows() == [("2-0", 12, 12, 0, 12), ("0-2", 20, 8, 12, -4)]
    assert splits.validation.targets.select(score_columns).rows() == [("0-2", 20, 8, 12, -4), ("2-1", None, None, None, None)]
    assert splits.test.targets.select(score_columns).rows() == [(None, None, None, None, None)]
    # Player B wins the match but loses the game-count comparison.
    assert splits.undated.targets.select(score_columns).rows() == [("1-2", 32, 18, 14, 4)]
    assert set(splits.train.targets.columns).isdisjoint(splits.train.features.columns)

    stat_columns = (
        "player_a_aces", "player_b_aces", "player_a_double_faults",
        "player_b_double_faults", "match_duration",
    )
    assert splits.train.targets.select(stat_columns).rows() == [
        (2, 1, 1, 0, 55), (3, 4, 1, 2, 60),
    ]
    assert splits.test.targets.select(stat_columns).rows() == [(None,) * 5]
    assert splits.undated.targets["match_duration"].item() is None
    break_columns = ("player_a_breaks", "player_b_breaks")
    assert splits.train.targets.select(break_columns).rows() == [(3, 0), (2, 4)]
    assert splits.validation.targets.select(break_columns).rows() == [(None, None)] * 2
    assert splits.test.targets.select(break_columns).rows() == [(None, None)]

    metrics = evaluate_score_market_baselines(splits)
    assert metrics["validation_matches"].to_list() == [2, 2, 1, 1]
    assert metrics["value"].to_list() == pytest.approx([1.0, 0.75, 4.0, 0.0])


    reference = fit_game_pair_reference(splits.train)
    assert reference.select("player_a_games", "player_b_games", "probability").rows() == [
        (8, 12, 0.5), (12, 0, 0.5),
    ]
    game_metrics = evaluate_game_pair_reference(splits)
    assert game_metrics["validation_matches"].to_list() == [1] * 4
    assert game_metrics["crps"].to_list() == pytest.approx([4, 1, 3, 2])
    assert game_metrics["median_mae"].to_list() == pytest.approx([8, 2, 6, 4])
