from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from tennis_prediction.datasets.score_markets import ScoreMarketSplits, ScoreMarketTable
from tennis_prediction.models.statistics.game_pair_model import (
    GamePairModel,
    _ValidationForecast,
    _game_pair_mask,
    fit_game_pair_model,
)
from tennis_prediction.models.statistics.game_pair_recalibration import (
    GamePairAdjustment,
    fit_game_pair_adjustment,
)
from tennis_prediction.models.statistics.score_model import _game_count_losses
from tennis_prediction.models.statistics.workflow import CALIBRATION_END, TRAIN_END, VALIDATION_END


@dataclass(frozen=True)
class FittedGamePairs:
    model: GamePairModel
    adjustment: GamePairAdjustment | None

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        probabilities = self.model.predict(features)
        if self.adjustment is not None:
            return self.adjustment.apply(probabilities, self.model.pairs)
        return probabilities


@dataclass(frozen=True)
class GamePairArtifact:
    formats: dict[int, FittedGamePairs]
    training_cutoff_date: int = CALIBRATION_END
    format_version: int = 1


def fit_daily_game_pairs(splits: ScoreMarketSplits) -> GamePairArtifact:
    dates = splits.train.metadata["tourney_date"]
    if dates.is_empty() or dates.null_count() or dates.max() > TRAIN_END:
        raise ValueError("Joint game models require dated training rows through 2018")
    formats = {}
    for best_of in (3, 5):
        model = fit_game_pair_model(splits.train, best_of)
        adjustment = None
        if best_of == 3:
            table = splits.validation
            mask = _game_pair_mask(table, best_of) & (table.metadata["tourney_date"] // 10000 == 2019)
            features = table.features.filter(mask)
            if features.is_empty():
                raise ValueError("No eligible 2019 best-of-three calibration games")
            observed = table.targets.filter(mask).select("player_a_games", "player_b_games")
            adjustment = fit_game_pair_adjustment(
                model.predict(features), model.pairs, observed.to_numpy()
            )
        formats[best_of] = FittedGamePairs(model, adjustment)
    return GamePairArtifact(formats)


def evaluate_game_pair_holdout(artifact: GamePairArtifact, holdout: ScoreMarketTable) -> pl.DataFrame:
    dates = holdout.metadata["tourney_date"]
    if dates.is_empty() or dates.null_count() or dates.min() <= VALIDATION_END:
        raise ValueError("Joint game holdout must be dated after 2021-12-31")

    rows = []
    for best_of, fitted in artifact.formats.items():
        mask = _game_pair_mask(holdout, best_of)
        features = holdout.features.filter(mask)
        if features.is_empty():
            raise ValueError(f"No eligible holdout games for best_of={best_of}")
        model = fitted.model
        forecast = _ValidationForecast(
            best_of, dates.filter(mask).to_numpy() // 10000, features,
            holdout.targets.filter(mask), model.pairs,
            model.classifier.reference @ model.conditional_probabilities,
            fitted.predict(features),
        )
        for market, support, reference, selected in forecast.marginals():
            observed = forecast.targets[market].to_numpy()
            reference_crps, reference_mae = _game_count_losses(reference, support, observed)
            selected_crps, selected_mae = _game_count_losses(selected, support, observed)
            rows.append({
                "best_of": best_of, "market": market, "matches": observed.size,
                "reference_crps": float(reference_crps.mean()),
                "selected_crps": float(selected_crps.mean()),
                "reference_median_mae": float(reference_mae.mean()),
                "selected_median_mae": float(selected_mae.mean()),
            })
    return pl.DataFrame(rows)


def load_game_pair_artifact(path: Path) -> GamePairArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, GamePairArtifact) or artifact.format_version != 1:
        raise ValueError("Unsupported joint game model artifact")
    return artifact


def game_pair_line(
    values: np.ndarray, probabilities: np.ndarray, line: float
) -> tuple[float, float, float]:
    """Return below, exact, and above probabilities for a signed half-step line."""
    if not np.isfinite(line) or not float(2 * line).is_integer():
        raise ValueError("Lines must be integers or half integers")
    return (
        float(probabilities[values < line].sum()),
        float(probabilities[values == line].sum()),
        float(probabilities[values > line].sum()),
    )
