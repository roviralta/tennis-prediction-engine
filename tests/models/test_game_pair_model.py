from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import ScoreMarketTable
from tennis_prediction.models.statistics.game_pair_baseline import evaluate_game_pair_reference
from tennis_prediction.models.statistics.game_pair_model import diagnose_game_pair_model
from tennis_prediction.models.statistics.game_pair_model import evaluate_game_pair_model
from tennis_prediction.models.statistics.game_pair_model import fit_game_pair_model
from tennis_prediction.models.statistics.game_pair_recalibration import GamePairAdjustment
from tennis_prediction.models.statistics.game_pair_recalibration import evaluate_game_pair_recalibration
from tennis_prediction.models.statistics.game_pair_workflow import (
    evaluate_game_pair_holdout,
    fit_daily_game_pairs,
    game_pair_line,
    load_game_pair_artifact,
)
from tennis_prediction.models.statistics.workflow import ScoreForecastModel
from tennis_prediction.models.winner.artifact import save_model_artifact
from tennis_prediction.models.statistics.workflow import ScoreModelArtifact


def _table(date: int, scores: list[str], a: list[int], b: list[int]) -> ScoreMarketTable:
    targets = pl.DataFrame({
        "correct_set_score": scores, "player_a_games": a, "player_b_games": b,
    }).with_columns(
        (pl.col("player_a_games") + pl.col("player_b_games")).alias("total_games"),
        (pl.col("player_a_games") - pl.col("player_b_games")).alias("game_margin"),
    )
    return ScoreMarketTable(
        metadata=pl.DataFrame({"tourney_date": [date] * len(scores)}),
        features=pl.DataFrame({
            "best_of": [3] * len(scores),
            "rank_difference": [-10 if score == "2-0" else 10 for score in scores],
        }),
        targets=targets,
    )


def test_joint_model_weights_pairs_with_prematch_features() -> None:
    train = _table(
        20180101, ["2-0"] * 60 + ["0-2"] * 60,
        [12] * 60 + [8] * 60, [8] * 60 + [12] * 60,
    )
    model = fit_game_pair_model(train, 3)
    assert model.pairs.tolist() == [[8, 12], [12, 8]]
    assert model.conditional_probabilities.tolist() == [[1, 0], [0, 1]]
    probabilities = model.predict(train.features[[0, 60]])
    assert probabilities.sum(axis=1) == pytest.approx([1, 1])
    assert (probabilities > 0.5).tolist() == [[False, True], [True, False]]

    validation = _table(20190101, ["2-0", "0-2"], [12, 8], [8, 12])
    unused = _table(20220101, ["2-0"], [12], [0])
    splits = ScoreMarketSplits(train, validation, unused, unused)
    metrics = evaluate_game_pair_model(splits)
    reference = evaluate_game_pair_reference(splits)
    np.testing.assert_allclose(metrics["reference_crps"], reference["crps"])
    assert metrics["validation_matches"].to_list() == [2] * 4
    margin = metrics.filter(pl.col("market") == "game_margin")
    assert margin["model_crps"].item() < margin["reference_crps"].item()
    with pytest.raises(ValueError, match="precede validation"):
        evaluate_game_pair_model(replace(splits, train=validation))

    # All training game pairs total 20. Give the daily comparator a
    # different support to verify that CRPS uses the same observed rows.
    daily = ScoreForecastModel(
        model.classifier, np.array([19]), np.array([1.0]), np.ones((2, 1)), 0.0, 0.0
    )
    dated = validation.metadata.with_columns(pl.Series("tourney_date", [20190101, 20200101]))
    diagnostics = diagnose_game_pair_model(
        replace(splits, validation=replace(validation, metadata=dated)),
        ScoreModelArtifact({3: daily}),
    )
    totals = diagnostics.yearly_losses.filter(pl.col("market") == "total_games")
    assert totals.select("year", "matches", "model_crps", "daily_crps").rows() == [
        (None, 2, 0.0, 1.0), (2019, 1, 0.0, 1.0), (2020, 1, 0.0, 1.0),
    ]

    margin_line = diagnostics.line_calibration.filter(
        (pl.col("market") == "game_margin") & (pl.col("event") == "over_3.5")
    )
    assert margin_line["observed_rate"].to_list() == [0.5, 1.0, 0.0]
    assert margin_line["format_probability"].to_list() == [0.5] * 3
    assert (margin_line["model_ece"] < margin_line["format_ece"]).to_list()[1:] == [True, True]


def test_pair_adjustment_preserves_joint_support_and_ordered_cumulative_probabilities() -> None:
    pairs = np.array([[8, 12], [10, 12], [13, 12]])
    probabilities = np.array([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])
    adjusted = GamePairAdjustment(0.2, 0.3).apply(probabilities, pairs)
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)
    assert np.all(adjusted >= 0)
    assert np.array_equal(adjusted == 0, probabilities == 0)
    assert adjusted[0, 1] > probabilities[0, 1]
    for counts in (pairs[:, 0], pairs[:, 1], pairs.sum(axis=1), pairs[:, 0] - pairs[:, 1]):
        cdf = adjusted @ (counts[:, None] <= np.unique(counts))
        assert np.all(np.diff(cdf, axis=1) >= -1e-12)


def test_pair_recalibration_uses_only_2019_labels() -> None:
    train = _table(20180101, ["2-0", "0-2"], [12, 8], [8, 12])
    early = _table(20190101, ["2-0", "2-0"], [12, 12], [8, 8])
    late = _table(20200101, ["2-0", "0-2"], [12, 8], [8, 12])
    validation = ScoreMarketTable(
        pl.concat([early.metadata, late.metadata]),
        pl.concat([early.features, late.features]),
        pl.concat([early.targets, late.targets]),
    )
    splits = ScoreMarketSplits(train, validation, late, late)
    report = evaluate_game_pair_recalibration(splits)
    changed = _table(20200101, ["0-2", "0-2"], [8, 8], [12, 12])
    alternative = replace(validation, targets=pl.concat([early.targets, changed.targets]))
    alternative_report = evaluate_game_pair_recalibration(replace(splits, validation=alternative))

    assert report.parameters.equals(alternative_report.parameters)
    assert report.parameters["margin_coefficient"].item() > 0
    assert report.yearly_losses["year"].unique().to_list() == [2020]
    assert not report.yearly_losses.equals(alternative_report.yearly_losses)
    assert report.line_calibration.height == 8
    with pytest.raises(ValueError, match="later validation year"):
        evaluate_game_pair_recalibration(replace(splits, validation=early))


def test_joint_workflow_serializes_selected_formats_and_scores_only_holdout(tmp_path) -> None:
    def table(date: int) -> ScoreMarketTable:
        first = _table(date, ["2-0", "0-2"], [12, 8], [8, 12])
        second = _table(date, ["3-0", "0-3"], [18, 9], [9, 18])
        return ScoreMarketTable(
            pl.concat([first.metadata, second.metadata]),
            pl.concat([
                first.features,
                second.features.with_columns(pl.lit(5, dtype=pl.Int64).alias("best_of")),
            ]),
            pl.concat([first.targets, second.targets]),
        )

    train, validation, holdout = map(table, (20180101, 20190101, 20220101))
    artifact = fit_daily_game_pairs(ScoreMarketSplits(train, validation, holdout, holdout))
    assert artifact.formats[3].adjustment is not None
    assert artifact.formats[5].adjustment is None
    path = save_model_artifact(artifact, tmp_path / "game-pairs.joblib")
    restored = load_game_pair_artifact(path)
    assert evaluate_game_pair_holdout(restored, holdout)["matches"].to_list() == [2] * 8
    with pytest.raises(ValueError, match="after 2021"):
        evaluate_game_pair_holdout(restored, validation)

    outcomes = np.array([-3, 0, 3])
    probabilities = np.array([0.2, 0.3, 0.5])
    # A -3 covers only when A's raw margin exceeds +3.
    assert game_pair_line(outcomes, probabilities, 3.0) == (0.5, 0.5, 0.0)
    assert game_pair_line(-outcomes, probabilities, -3.0) == (0.0, 0.5, 0.5)
    with pytest.raises(ValueError, match="half integers"):
        game_pair_line(outcomes, probabilities, -2.25)
