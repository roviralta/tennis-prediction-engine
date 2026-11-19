import numpy as np
import polars as pl
import pytest

from tennis_prediction.datasets.score_markets import ScoreMarketSplits, ScoreMarketTable
from tennis_prediction.models.statistics.stat_markets import (
    automatic_stat_thresholds,
    evaluate_stat_markets,
    fit_stat_markets,
    load_stat_artifact,
    reverse_player_features,
    select_stat_markets,
)
from tennis_prediction.models.winner.artifact import save_model_artifact


def _table(date: int, size: int) -> ScoreMarketTable:
    features = pl.DataFrame({
        "best_of": [3] * size,
        "surface": ["Hard"] * size,
        "player_a_rank": [5] * size,
        "player_b_rank": [90] * size,
        "rank_difference": [-85] * size,
    })
    targets = pl.DataFrame({
        "player_a_aces": [9] * size, "player_b_aces": [1] * size,
        "player_a_double_faults": [2] * size, "player_b_double_faults": [7] * size,
        "player_a_breaks": [5] * size, "player_b_breaks": [1] * size,
    })
    return ScoreMarketTable(pl.DataFrame({"tourney_date": [date] * size}), features, targets)


def test_stat_models_use_both_player_orientations_and_gate_by_market(tmp_path) -> None:
    train, validation, holdout = (
        _table(20180101, 60), _table(20190101, 200), _table(20220101, 200)
    )
    assert reverse_player_features(reverse_player_features(train.features)).equals(train.features)
    artifact = fit_stat_markets(train)
    report = evaluate_stat_markets(
        artifact, ScoreMarketSplits(train, validation, holdout, holdout)
    )
    assert report["matches"].to_list() == [200] * 6
    aces = report.filter(pl.col("market") == "aces")
    assert aces["observations"].to_list() == [400, 400]
    assert (aces["model_crps"] < aces["reference_crps"]).all()
    chosen = select_stat_markets(artifact, report)
    assert set(chosen.models) == {("aces", 3), ("double_faults", 3), ("breaks", 3)}
    restored = load_stat_artifact(save_model_artifact(chosen, tmp_path / "stats.joblib"))
    cdf = restored.models["aces", 3].forecast(train.features.head(1))
    assert cdf[0, -1] == pytest.approx(1)
    assert np.all(np.diff(cdf, axis=1) >= -1e-12)

    with pytest.raises(ValueError, match="holdout"):
        evaluate_stat_markets(artifact, ScoreMarketSplits(train, validation, validation, holdout))
    assert not select_stat_markets(artifact, report.with_columns(pl.lit(199).alias("matches"))).models


def test_automatic_stat_thresholds_include_common_counts_and_forecast_tail() -> None:
    assert list(automatic_stat_thresholds(np.arange(3), np.array([0.5, 0.9, 1]))) == [
        1, 2, 3, 4,
    ]
    counts = np.arange(8)
    cdf = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.94, 0.98, 1.0])
    assert list(automatic_stat_thresholds(counts, cdf)) == list(range(1, 7))
