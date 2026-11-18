from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import ScoreMarketTable
from tennis_prediction.models.statistics.workflow import evaluate_score_holdout
from tennis_prediction.models.statistics.workflow import fit_daily_score_models
from tennis_prediction.models.statistics.workflow import load_score_artifact
from tennis_prediction.models.statistics.workflow import score_line_probabilities
from tennis_prediction.models.winner.artifact import save_model_artifact


def _table(date: int) -> ScoreMarketTable:
    return ScoreMarketTable(
        metadata=pl.DataFrame({"tourney_date": [date] * 4}),
        features=pl.DataFrame({"best_of": [3, 3, 5, 5]}),
        targets=pl.DataFrame({
            "correct_set_score": ["2-0", "2-1", "3-0", "3-1"],
            "total_games": [20, 30, 30, 38],
        }),
    )


def test_fixed_score_workflow_persists_and_evaluates_reserved_rows(tmp_path) -> None:
    training, calibration, holdout = map(_table, (20180101, 20190101, 20220101))
    splits = ScoreMarketSplits(training, calibration, holdout, holdout)
    artifact = fit_daily_score_models(splits)
    path = save_model_artifact(artifact, tmp_path / "score.joblib")
    with pytest.raises(FileExistsError):
        save_model_artifact(artifact, path)
    restored = load_score_artifact(path)
    assert restored.training_cutoff_date == 20191231
    scores, games = restored.formats[3].forecast(holdout.features.head(1))
    assert scores.sum() == pytest.approx(1.0)
    assert np.all(np.diff(games[0]) >= -1e-12)
    assert restored.formats[3].log_temperature == 0
    assert restored.formats[5].log_odds_shift == 0
    assert evaluate_score_holdout(restored, holdout)["matches"].to_list() == [2] * 6
    with pytest.raises(ValueError, match="after 2021"):
        evaluate_score_holdout(restored, calibration)
    with pytest.raises(ValueError, match="through 2018"):
        fit_daily_score_models(replace(splits, train=calibration))


def test_score_lines_include_pushes_and_preserve_tail_probabilities() -> None:
    counts = np.array([20, 21, 22])
    cdf = np.array([0.25, 0.5, 1.0])
    assert score_line_probabilities(counts, cdf, 21.0) == (0.25, 0.25, 0.5)
    assert score_line_probabilities(counts, cdf, 21.5) == (0.5, 0.0, 0.5)
    assert score_line_probabilities(counts, cdf, 19.5) == (0.0, 0.0, 1.0)
    assert score_line_probabilities(counts, cdf, 22.5) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="half integers"):
        score_line_probabilities(counts, cdf, 21.25)
