from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.models.statistics.game_pair_model import _validation_forecasts
from tennis_prediction.models.statistics.game_pair_model import game_pair_lines
from tennis_prediction.models.statistics.score_model import _calibration_row
from tennis_prediction.models.statistics.score_model import _game_count_losses
from tennis_prediction.models.statistics.score_model import _game_line_probability


@dataclass(frozen=True)
class GamePairRecalibrationReport:
    parameters: pl.DataFrame
    yearly_losses: pl.DataFrame
    line_calibration: pl.DataFrame


def _pair_statistics(pairs: np.ndarray) -> np.ndarray:
    player_a, player_b = pairs.T
    return np.column_stack((player_a + player_b, player_a - player_b)) / 10


@dataclass(frozen=True)
class GamePairAdjustment:
    """Shared reweighting of the original pair probabilities for one format."""

    total: float
    margin: float

    def apply(self, probabilities: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        log_weights = _pair_statistics(pairs) @ np.array([self.total, self.margin])
        weights = np.exp(log_weights - log_weights.max())
        weighted = probabilities * weights
        return weighted / weighted.sum(axis=1, keepdims=True)


def fit_game_pair_adjustment(
    probabilities: np.ndarray, pairs: np.ndarray, observed_pairs: np.ndarray
) -> GamePairAdjustment:
    """Match the calibration cohort's mean total and margin with a regularized tilt."""
    statistics = _pair_statistics(pairs)
    observed = _pair_statistics(observed_pairs).mean(axis=0)
    penalty = 0.01

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        log_weights = statistics @ coefficients
        center = log_weights.max()
        weighted = probabilities * np.exp(log_weights - center)
        mass = weighted.sum(axis=1)
        expected = (weighted / mass[:, None]).mean(axis=0) @ statistics
        loss = (np.log(mass).mean() + center - observed @ coefficients
                + penalty * (coefficients @ coefficients) / 2)
        return float(loss), expected - observed + penalty * coefficients

    result = minimize(
        objective, np.zeros(2), method="L-BFGS-B", jac=True, bounds=[(-2, 2), (-2, 2)]
    )
    if not result.success:
        raise RuntimeError(f"Game-pair recalibration failed: {result.message}")
    return GamePairAdjustment(*map(float, result.x))


def evaluate_game_pair_recalibration(
    splits: ScoreMarketSplits, calibration_year: int = 2019
) -> GamePairRecalibrationReport:
    """Fit on one validation year and compare original and adjusted forecasts later."""
    parameters = []
    losses = []
    lines = []
    for forecast in _validation_forecasts(splits):
        calibration = forecast.years == calibration_year
        later_years = np.unique(forecast.years[forecast.years > calibration_year])
        if not calibration.any() or not later_years.size:
            raise ValueError(
                f"best_of={forecast.best_of} requires eligible games in {calibration_year} "
                "and a later validation year"
            )

        observed_pairs = forecast.targets.select("player_a_games", "player_b_games").to_numpy()
        adjustment = fit_game_pair_adjustment(
            forecast.probabilities[calibration], forecast.pairs, observed_pairs[calibration]
        )
        adjusted = adjustment.apply(forecast.probabilities, forecast.pairs)
        parameters.append({
            "best_of": forecast.best_of,
            "calibration_year": calibration_year,
            "matches": int(calibration.sum()),
            "total_coefficient": adjustment.total,
            "margin_coefficient": adjustment.margin,
        })

        for original, recalibrated in zip(
            forecast.marginals(), forecast.marginals(adjusted), strict=True
        ):
            market, support, _, original_cdf = original
            _, _, _, adjusted_cdf = recalibrated
            observed = forecast.targets[market].to_numpy()
            original_crps, original_mae = _game_count_losses(original_cdf, support, observed)
            adjusted_crps, adjusted_mae = _game_count_losses(adjusted_cdf, support, observed)
            for year in later_years:
                mask = forecast.years == year
                cohort = {"best_of": forecast.best_of, "year": int(year)}
                losses.append({
                    **cohort,
                    "market": market,
                    "matches": int(mask.sum()),
                    "original_crps": float(original_crps[mask].mean()),
                    "adjusted_crps": float(adjusted_crps[mask].mean()),
                    "original_median_mae": float(original_mae[mask].mean()),
                    "adjusted_median_mae": float(adjusted_mae[mask].mean()),
                })
                for line in game_pair_lines(forecast.best_of, market):
                    lines.append({
                        **cohort,
                        **_calibration_row(
                            market, f"over_{line}", observed[mask] > line,
                            _game_line_probability(original_cdf[mask], support, line),
                            _game_line_probability(adjusted_cdf[mask], support, line), 10,
                        ),
                    })
    calibration = pl.DataFrame(lines)
    if lines:
        calibration = calibration.rename({
            "format_probability": "original_probability",
            "model_probability": "adjusted_probability",
            "format_ece": "original_ece",
            "model_ece": "adjusted_ece",
        })
    return GamePairRecalibrationReport(pl.DataFrame(parameters), pl.DataFrame(losses), calibration)
