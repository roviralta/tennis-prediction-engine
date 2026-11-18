from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import ScoreMarketTable
from tennis_prediction.models.statistics.recalibration import _fit_adjustment
from tennis_prediction.models.statistics.recalibration import _scale_score_probabilities
from tennis_prediction.models.statistics.recalibration import _shift_game_cdf
from tennis_prediction.models.statistics.score_model import FittedScoreFormat
from tennis_prediction.models.statistics.score_model import _game_count_losses
from tennis_prediction.models.statistics.score_model import _multiclass_brier
from tennis_prediction.models.statistics.score_model import fit_game_distributions
from tennis_prediction.models.statistics.score_model import fit_score_format

TRAIN_END = 20181231
VALIDATION_END = 20211231
CALIBRATION_YEAR = 2019
CALIBRATION_END = 20191231
ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class ScoreForecastModel:
    classifier: FittedScoreFormat
    counts: np.ndarray
    format_cdf: np.ndarray
    conditional_cdfs: np.ndarray
    log_temperature: float
    log_odds_shift: float

    def forecast(self, features: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scores = self.classifier.predict(features)
        games = _shift_game_cdf(scores @ self.conditional_cdfs, self.log_odds_shift)
        return _scale_score_probabilities(scores, self.log_temperature), games


@dataclass(frozen=True)
class ScoreModelArtifact:
    formats: dict[int, ScoreForecastModel]
    training_cutoff_date: int = CALIBRATION_END
    format_version: int = ARTIFACT_VERSION


def _eligible(table: ScoreMarketTable, best_of: int, *, calibration: bool = False) -> pl.Series:
    mask = (table.features["best_of"] == best_of) & table.targets["correct_set_score"].is_not_null()
    if calibration:
        mask = mask & (table.metadata["tourney_date"] // 10000 == CALIBRATION_YEAR)
    return mask.fill_null(False)


def fit_daily_score_models(splits: ScoreMarketSplits) -> ScoreModelArtifact:
    """Fit the fixed, historically evaluated specification for local predictions."""
    dates = splits.train.metadata["tourney_date"]
    if dates.is_empty() or dates.null_count() or dates.max() > TRAIN_END:
        raise ValueError("Score model training rows must be dated through 2018-12-31")

    formats = {}
    for best_of in (3, 5):
        rows = _eligible(splits.train, best_of)
        targets = splits.train.targets.filter(rows)
        if targets.is_empty():
            raise ValueError(f"No score training labels for best_of={best_of}")
        classifier = fit_score_format(
            splits.train.features.filter(rows), targets["correct_set_score"]
        )
        counts, format_cdf, conditional_cdfs, _ = fit_game_distributions(targets, classifier.classes)

        calibration_rows = _eligible(splits.validation, best_of, calibration=True)
        calibration_targets = splits.validation.targets.filter(calibration_rows)
        if calibration_targets.is_empty():
            raise ValueError(f"No 2019 calibration labels for best_of={best_of}")
        predictions = classifier.predict(splits.validation.features.filter(calibration_rows))
        temperature = shift = 0.0
        if best_of == 5:
            observed = calibration_targets["correct_set_score"].to_numpy()
            temperature = _fit_adjustment(
                lambda value: _multiclass_brier(
                    _scale_score_probabilities(predictions, value), classifier.classes, observed
                ).mean()
            )
        else:
            games = calibration_targets["total_games"]
            if games.drop_nulls().is_empty():
                raise ValueError("No 2019 best-of-three game calibration labels")
            cdf = predictions[games.is_not_null().to_numpy()] @ conditional_cdfs
            shift = _fit_adjustment(
                lambda value: _game_count_losses(
                    _shift_game_cdf(cdf, value), counts, games.drop_nulls().to_numpy()
                )[0].mean()
            )
        formats[best_of] = ScoreForecastModel(
            classifier, counts, format_cdf, conditional_cdfs, temperature, shift
        )
    return ScoreModelArtifact(formats)


def evaluate_score_holdout(artifact: ScoreModelArtifact, holdout: ScoreMarketTable) -> pl.DataFrame:
    """Score the frozen artifact on the reserved 2022+ score-market outcomes."""
    dates = holdout.metadata["tourney_date"]
    if dates.is_empty() or dates.null_count() or dates.min() <= VALIDATION_END:
        raise ValueError("Score holdout rows must be dated after 2021-12-31")
    report = []
    for best_of, model in artifact.formats.items():
        rows = _eligible(holdout, best_of)
        targets = holdout.targets.filter(rows)
        features = holdout.features.filter(rows)
        observed = targets["correct_set_score"].to_numpy()
        observed_games = targets["total_games"].drop_nulls().to_numpy()
        if not observed.size or not observed_games.size:
            raise ValueError(f"Missing holdout labels for best_of={best_of}")

        raw = model.classifier.predict(features)
        scores, cdf = model.forecast(features)
        classes = model.classifier.classes
        score_reference = np.broadcast_to(model.classifier.reference, raw.shape)
        game_rows = targets["total_games"].is_not_null().to_numpy()
        game_reference = np.broadcast_to(model.format_cdf, cdf[game_rows].shape)
        raw_cdf = raw[game_rows] @ model.conditional_cdfs

        game_losses = [
            _game_count_losses(distribution, model.counts, observed_games)
            for distribution in (game_reference, raw_cdf, cdf[game_rows])
        ]
        for market, metric, losses in (
            ("correct_set_score", "brier", [
                _multiclass_brier(p, classes, observed) for p in (score_reference, raw, scores)
            ]),
            ("total_games", "crps", [pair[0] for pair in game_losses]),
            ("total_games", "mae", [pair[1] for pair in game_losses]),
        ):
            reference, raw_loss, selected = map(lambda values: float(values.mean()), losses)
            report.append({
                "best_of": best_of, "market": market, "metric": metric,
                "matches": len(losses[0]), "format_only": reference,
                "raw": raw_loss, "selected": selected,
            })
    return pl.DataFrame(report)


def load_score_artifact(path: Path) -> ScoreModelArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, ScoreModelArtifact) or artifact.format_version != ARTIFACT_VERSION:
        raise ValueError("Unsupported score model artifact")
    return artifact


def score_line_probabilities(
    counts: np.ndarray, cdf: np.ndarray, line: float
) -> tuple[float, float, float]:
    """Return under, push and over probabilities for either integer or half lines."""
    if not np.isfinite(line) or line < 0 or not float(2 * line).is_integer():
        raise ValueError("Game lines must be nonnegative integers or half integers")
    cumulative = np.concatenate(([0.0], np.clip(cdf, 0.0, 1.0)[:-1], [1.0]))
    under = float(cumulative[counts.searchsorted(line, side="left")])
    through = float(cumulative[counts.searchsorted(line, side="right")])
    return under, through - under, 1.0 - through
