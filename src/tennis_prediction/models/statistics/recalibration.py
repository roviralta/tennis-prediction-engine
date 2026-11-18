from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.models.statistics.score_model import _calibration_row
from tennis_prediction.models.statistics.score_model import _evaluate_formats
from tennis_prediction.models.statistics.score_model import _GAME_LINES
from tennis_prediction.models.statistics.score_model import _game_count_losses
from tennis_prediction.models.statistics.score_model import _game_line_probability
from tennis_prediction.models.statistics.score_model import _multiclass_brier


@dataclass(frozen=True)
class ScoreRecalibrationReport:
    parameters: pl.DataFrame
    losses: pl.DataFrame
    game_lines: pl.DataFrame


def _fit_adjustment(objective: Callable[[float], float]) -> float:
    result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Score recalibration failed: {result.message}")
    return float(result.x) if result.fun < objective(0.0) else 0.0


def _scale_score_probabilities(probabilities: np.ndarray, log_temperature: float) -> np.ndarray:
    scaled = probabilities ** np.exp(-log_temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _shift_game_cdf(cdf: np.ndarray, log_odds_shift: float) -> np.ndarray:
    """Apply the same increasing odds transform at every game threshold."""

    bounded = np.clip(cdf, 0.0, 1.0)
    numerator = np.exp(log_odds_shift) * bounded
    return numerator / (1 - bounded + numerator)


def evaluate_score_recalibration(
    splits: ScoreMarketSplits, calibration_year: int = 2019
) -> ScoreRecalibrationReport:
    """Fit by format on one validation year and score subsequent validation years."""

    train_dates = splits.train.metadata["tourney_date"]
    dates = splits.validation.metadata["tourney_date"]
    if train_dates.null_count() or dates.null_count():
        raise ValueError("Score recalibration requires dated training and validation rows")
    if train_dates.is_empty() or dates.is_empty() or train_dates.max() >= dates.min():
        raise ValueError("Training must precede validation for score recalibration")

    parameters = []
    losses = []
    game_lines = []
    for result in _evaluate_formats(splits):
        years = dates.filter(result.rows).to_numpy() // 10000
        calibration = years == calibration_year
        later_years = np.unique(years[years > calibration_year])
        if not calibration.any() or not later_years.size:
            raise ValueError(
                f"best_of={result.best_of} requires score labels in {calibration_year} "
                "and a later validation year"
            )

        probabilities = result.probabilities[calibration]
        observed = result.observed[calibration]
        log_temperature = _fit_adjustment(
            lambda value: _multiclass_brier(
                _scale_score_probabilities(probabilities, value), result.classes, observed
            ).mean()
        )
        scores_adjusted = _scale_score_probabilities(result.probabilities, log_temperature)
        score_before = _multiclass_brier(result.probabilities, result.classes, result.observed)
        score_after = _multiclass_brier(scores_adjusted, result.classes, result.observed)
        cohort_losses = [("correct_set_score", "brier", years, score_before, score_after)]
        parameters.append({
            "best_of": result.best_of, "market": "correct_set_score",
            "calibration_year": calibration_year, "matches": int(calibration.sum()),
            "adjustment": "log_temperature", "value": log_temperature,
        })

        games = result.games
        if games is not None:
            game_years = years[games.rows]
            calibration_games = game_years == calibration_year
            if not calibration_games.any():
                raise ValueError(
                    f"Missing games labels for best_of={result.best_of} in {calibration_year}"
                )
            cdf = games.model_cdf[calibration_games]
            observed_games = games.observed[calibration_games]
            shift = _fit_adjustment(
                lambda value: _game_count_losses(
                    _shift_game_cdf(cdf, value), games.counts, observed_games
                )[0].mean()
            )
            adjusted_cdf = _shift_game_cdf(games.model_cdf, shift)
            if np.any(np.diff(adjusted_cdf, axis=1) < -1e-12):
                raise ValueError("Recalibrated game distributions must be nondecreasing")
            crps_before, mae_before = _game_count_losses(
                games.model_cdf, games.counts, games.observed
            )
            crps_after, mae_after = _game_count_losses(adjusted_cdf, games.counts, games.observed)
            cohort_losses.extend([
                ("total_games", "crps", game_years, crps_before, crps_after),
                ("total_games", "mae", game_years, mae_before, mae_after),
            ])
            parameters.append({
                "best_of": result.best_of, "market": "total_games",
                "calibration_year": calibration_year, "matches": int(calibration_games.sum()),
                "adjustment": "log_odds_shift", "value": shift,
            })
            for year in later_years:
                mask = game_years == year
                if not mask.any():
                    continue
                for line in _GAME_LINES.get(result.best_of, ()):
                    game_lines.append({
                        "best_of": result.best_of, "year": int(year),
                        **_calibration_row(
                            "total_games", f"over_{line}", games.observed[mask] > line,
                            _game_line_probability(games.model_cdf[mask], games.counts, line),
                            _game_line_probability(adjusted_cdf[mask], games.counts, line), 10,
                        ),
                    })

        for market, metric, outcome_years, before, after in cohort_losses:
            for year in later_years:
                mask = outcome_years == year
                if mask.any():
                    losses.append({
                        "best_of": result.best_of, "year": int(year), "market": market,
                        "metric": metric, "matches": int(mask.sum()),
                        "original": float(before[mask].mean()),
                        "recalibrated": float(after[mask].mean()),
                    })

    lines = pl.DataFrame(game_lines)
    if game_lines:
        lines = lines.rename({
            "format_probability": "original_probability",
            "model_probability": "recalibrated_probability",
            "format_ece": "original_ece",
            "model_ece": "recalibrated_ece",
        })
    return ScoreRecalibrationReport(pl.DataFrame(parameters), pl.DataFrame(losses), lines)
