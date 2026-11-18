from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import ScoreMarketTable
from tennis_prediction.models.statistics.recalibration import _scale_score_probabilities
from tennis_prediction.models.statistics.recalibration import _shift_game_cdf
from tennis_prediction.models.statistics.recalibration import evaluate_score_recalibration


def test_adjustments_preserve_probability_structure() -> None:
    scores = np.array([[0.7, 0.3, 0.0], [0.0, 1.0, 0.0]])
    adjusted = _scale_score_probabilities(scores, 0.5)
    assert adjusted.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert 0.5 < adjusted[0, 0] < 0.7
    assert adjusted[:, 2].tolist() == [0.0, 0.0]

    cdf = np.array([[0.0, 0.3, 0.3, 0.9, 1.0]])
    shifted = _shift_game_cdf(cdf, -0.7)
    assert shifted[:, [0, -1]].tolist() == [[0.0, 1.0]]
    assert np.all(np.diff(shifted, axis=1) >= 0)
    assert np.all(shifted[:, 1:-1] < cdf[:, 1:-1])


def test_recalibration_uses_only_the_designated_year_for_fitting() -> None:
    def table(dates, scores, games):
        return ScoreMarketTable(
            metadata=pl.DataFrame({"tourney_date": dates}),
            features=pl.DataFrame({"best_of": [3] * len(dates)}),
            targets=pl.DataFrame({"correct_set_score": scores, "total_games": games}),
        )

    training = table([20180101] * 4, ["2-0", "2-0", "2-0", "2-1"], [18, 18, 20, 24])
    validation = table(
        [20190101, 20190102, 20190103, 20200101, 20210101],
        ["2-0", "2-1", "2-1", "2-1", "2-1"],
        [18, 24, 24, 24, 24],
    )
    splits = ScoreMarketSplits(training, validation, training, training)
    report = evaluate_score_recalibration(splits)
    changed_later_labels = table(
        validation.metadata["tourney_date"].to_list(),
        ["2-0", "2-1", "2-1", "2-0", "2-0"],
        [18, 24, 24, 18, 18],
    )
    changed = evaluate_score_recalibration(replace(splits, validation=changed_later_labels))
    assert report.parameters.equals(changed.parameters)
    assert report.parameters["matches"].to_list() == [3, 3]
    assert report.losses["year"].unique().sort().to_list() == [2020, 2021]
    assert not report.losses.equals(changed.losses)
    assert report.game_lines.height == 6
    assert report.parameters.filter(pl.col("market") == "total_games")["value"].item() < 0
    with pytest.raises(ValueError, match="Training must precede validation"):
        evaluate_score_recalibration(replace(splits, train=validation))
