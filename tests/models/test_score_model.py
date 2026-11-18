import numpy as np
import polars as pl
import pytest

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import ScoreMarketTable
from tennis_prediction.models.statistics.score_model import _game_count_losses
from tennis_prediction.models.statistics.score_model import _game_line_probability
from tennis_prediction.models.statistics.score_model import diagnose_player_score_models
from tennis_prediction.models.statistics.score_model import evaluate_player_score_models


def test_game_count_losses_include_outcomes_outside_training_support() -> None:
    cdf = np.array([[0.5, 0.5, 1.0]] * 3)
    crps, absolute_errors = _game_count_losses(cdf, np.array([12, 13, 14]), np.array([10, 13, 17]))
    assert crps == pytest.approx([2.5, 0.5, 3.5])
    assert absolute_errors.tolist() == [3, 0, 4]


def test_score_model_evaluates_available_labels_and_uses_only_training_and_validation() -> None:
    def table(formats, scores, games, surfaces, ranks):
        return ScoreMarketTable(
            metadata=pl.DataFrame({
                "match_index": range(len(formats)),
                "tourney_date": [20190101 + index * 10000 for index in range(len(formats))],
            }),
            features=pl.DataFrame(
                {"best_of": formats, "surface": surfaces, "rank_difference": ranks},
                schema={"best_of": pl.Int64, "surface": pl.String, "rank_difference": pl.Int64},
            ),
            targets=pl.DataFrame(
                {"correct_set_score": scores, "total_games": games},
                schema={"correct_set_score": pl.String, "total_games": pl.Int64},
            ),
        )

    train = table(
        [3, 3, 3, 3, 5], ["2-0", "2-0", "2-1", "2-1", "3-0"], [12, 14, 30, 32, 18],
        ["Hard", None, "Hard", "Clay", "Hard"], [None, -20, 14, 24, None],
    )
    validation = table(
        [3, 3, 5, 3], ["2-0", "0-2", "0-3", None], [13, None, 20, None],
        ["Grass", "Hard", None, "Hard"], [100, None, 4, 200],
    )
    unused = table([7], ["4-0"], [49], ["Carpet"], [None])
    splits = ScoreMarketSplits(train, validation, unused, unused)
    report = evaluate_player_score_models(splits)

    assert report.select("market", "metric", "training_matches", "validation_matches").rows() == [
        ("correct_set_score", "brier", 5, 3),
        ("total_games", "crps", 5, 2),
        ("total_games", "mae", 5, 2),
    ]
    # At best_of=3 the empirical games distribution has median 22 and
    # CRPS 4.75 for 13 games. At best_of=5 both losses are 2 for 20 games.
    assert report["format_only"].to_list() == pytest.approx([4 / 3, 3.375, 5.5])
    assert report["player_model"].to_list() == pytest.approx(report["format_only"].to_list())


    diagnostics = diagnose_player_score_models(splits)
    score_losses = diagnostics.losses.filter(
        (pl.col("best_of") == 3) & (pl.col("metric") == "brier")
    )
    assert score_losses.select("year", "validation_matches").rows() == [
        (None, 2), (2019, 1), (2020, 1),
    ]
    unseen_score = diagnostics.calibration.filter(
        (pl.col("best_of") == 3) & (pl.col("year") == 2020) & (pl.col("event") == "0-2")
    )
    assert unseen_score.select("observed_rate", "model_probability", "model_ece").row(0) == (
        1.0, 0.0, 1.0,
    )
    game_line = diagnostics.calibration.filter(
        pl.col("year").is_null() & (pl.col("event") == "over_22.5")
    )
    assert game_line.select("matches", "observed_rate", "model_probability", "model_ece").row(0) == (
        1, 0.0, 0.5, 0.5,
    )


def test_game_line_probability_uses_strictly_more_games() -> None:
    counts = np.array([20, 21, 22])
    cdf = np.array([[0.25, 0.5, 1.0]])
    assert _game_line_probability(cdf, counts, 19.5).tolist() == [1.0]
    assert _game_line_probability(cdf, counts, 21.5).tolist() == [0.5]
    assert _game_line_probability(cdf, counts, 22.5).tolist() == [0.0]
