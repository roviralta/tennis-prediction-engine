from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import ScoreMarketTable
from tennis_prediction.models.statistics.game_pair_baseline import _GAME_COLUMNS
from tennis_prediction.models.statistics.game_pair_baseline import _OUTCOMES
from tennis_prediction.models.statistics.score_model import FittedScoreFormat
from tennis_prediction.models.statistics.score_model import _calibration_row
from tennis_prediction.models.statistics.score_model import _game_line_probability
from tennis_prediction.models.statistics.score_model import _game_count_losses
from tennis_prediction.models.statistics.score_model import fit_score_format
from tennis_prediction.models.statistics.workflow import ScoreModelArtifact


@dataclass(frozen=True)
class GamePairModel:
    """Joint probabilities on observed game pairs in one match format."""

    classifier: FittedScoreFormat
    pairs: np.ndarray
    conditional_probabilities: np.ndarray

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        return self.classifier.predict(features) @ self.conditional_probabilities


def _game_pair_mask(table: ScoreMarketTable, best_of: int) -> pl.Series:
    return (table.features["best_of"] == best_of) & table.targets.select(
        pl.all_horizontal(pl.col(*_GAME_COLUMNS, "correct_set_score").is_not_null())
    ).to_series()


def fit_game_pair_model(training: ScoreMarketTable, best_of: int) -> GamePairModel:
    """Fit a set-score mixture using only game-eligible training matches."""
    rows = _game_pair_mask(training, best_of)
    features, targets = training.features.filter(rows), training.targets.filter(rows)
    if features.is_empty():
        raise ValueError(f"No eligible training game pairs for best_of={best_of}")

    score = targets["correct_set_score"]
    classifier = fit_score_format(features, score)
    pairs, pair_indexes = np.unique(
        targets.select(_GAME_COLUMNS).to_numpy(), axis=0, return_inverse=True
    )
    class_indexes = np.searchsorted(classifier.classes, score.to_numpy())
    frequencies = np.zeros((classifier.classes.size, pairs.shape[0]), dtype=np.int64)
    np.add.at(frequencies, (class_indexes, pair_indexes), 1)
    conditional_probabilities = frequencies / frequencies.sum(axis=1, keepdims=True)
    return GamePairModel(classifier, pairs, conditional_probabilities)


@dataclass(frozen=True)
class _ValidationForecast:
    best_of: int
    years: np.ndarray
    features: pl.DataFrame
    targets: pl.DataFrame
    pairs: np.ndarray
    reference: np.ndarray
    probabilities: np.ndarray

    def marginals(
        self, probabilities: np.ndarray | None = None
    ) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        if probabilities is None:
            probabilities = self.probabilities
        player_a, player_b = self.pairs.T
        values_by_market = (player_a, player_b, player_a + player_b, player_a - player_b)
        for market, values in zip(_OUTCOMES, values_by_market, strict=True):
            support = np.arange(values.min(), values.max() + 1)
            membership = values[:, None] <= support
            reference = np.broadcast_to(
                self.reference @ membership, (self.features.height, support.size)
            )
            yield market, support, reference, probabilities @ membership


def _validation_forecasts(splits: ScoreMarketSplits) -> Iterator[_ValidationForecast]:
    train_dates = splits.train.metadata["tourney_date"]
    validation_dates = splits.validation.metadata["tourney_date"]
    if (
        train_dates.is_empty() or validation_dates.is_empty()
        or train_dates.null_count() or validation_dates.null_count()
        or train_dates.max() >= validation_dates.min()
    ):
        raise ValueError("Dated game-pair training rows must precede validation")

    eligible_formats = splits.validation.features.filter(
        splits.validation.targets["game_margin"].is_not_null()
    )["best_of"].unique().sort()
    if eligible_formats.is_empty() or eligible_formats.null_count():
        raise ValueError("Validation requires eligible game pairs with best_of")

    for best_of in eligible_formats:
        model = fit_game_pair_model(splits.train, best_of)
        mask = _game_pair_mask(splits.validation, best_of)
        features = splits.validation.features.filter(mask)
        if features.is_empty():
            raise ValueError(f"No joint validation outcomes for best_of={best_of}")
        yield _ValidationForecast(
            best_of=best_of,
            years=validation_dates.filter(mask).to_numpy() // 10000,
            features=features,
            targets=splits.validation.targets.filter(mask),
            pairs=model.pairs,
            reference=model.classifier.reference @ model.conditional_probabilities,
            probabilities=model.predict(features),
        )


def evaluate_game_pair_model(splits: ScoreMarketSplits) -> pl.DataFrame:
    """Compare candidate and empirical joint forecasts on chronological validation."""
    rows = []
    for forecast in _validation_forecasts(splits):
        for market, support, reference, candidate in forecast.marginals():
            observed = forecast.targets[market].to_numpy()
            reference_crps, reference_mae = _game_count_losses(reference, support, observed)
            crps, mae = _game_count_losses(candidate, support, observed)
            rows.append({
                "best_of": forecast.best_of,
                "market": market,
                "validation_matches": observed.size,
                "reference_crps": float(reference_crps.mean()),
                "model_crps": float(crps.mean()),
                "reference_median_mae": float(reference_mae.mean()),
                "model_median_mae": float(mae.mean()),
            })
    return pl.DataFrame(rows).sort("best_of", "market")


@dataclass(frozen=True)
class GamePairDiagnostics:
    yearly_losses: pl.DataFrame
    line_calibration: pl.DataFrame


_PLAYER_LINES = {3: (10.5, 12.5), 5: (17.5, 19.5)}
_MARGIN_LINES = {3: (-3.5, -0.5, 0.5, 3.5), 5: (-5.5, -0.5, 0.5, 5.5)}


def game_pair_lines(best_of: int, market: str) -> tuple[float, ...]:
    if market == "game_margin":
        return _MARGIN_LINES.get(best_of, ())
    if market in _GAME_COLUMNS:
        return _PLAYER_LINES.get(best_of, ())
    return ()


def diagnose_game_pair_model(
    splits: ScoreMarketSplits, daily_score: ScoreModelArtifact
) -> GamePairDiagnostics:
    """Diagnose yearly validation errors and fixed half-point lines by format."""
    loss_rows = []
    calibration_rows = []
    for forecast in _validation_forecasts(splits):
        daily_model = daily_score.formats.get(forecast.best_of)
        if daily_model is None:
            raise ValueError(f"Missing daily total-games model for best_of={forecast.best_of}")
        _, daily_cdf = daily_model.forecast(forecast.features)
        years = np.unique(forecast.years).tolist()
        for market, support, reference, candidate in forecast.marginals():
            observed = forecast.targets[market].to_numpy()
            reference_crps, reference_mae = _game_count_losses(reference, support, observed)
            model_crps, model_mae = _game_count_losses(candidate, support, observed)
            if market == "total_games":
                daily_crps, daily_mae = _game_count_losses(
                    daily_cdf, daily_model.counts, observed
                )
            lines = game_pair_lines(forecast.best_of, market)
            for year in [None, *years]:
                mask = (
                    np.ones(observed.size, dtype=bool)
                    if year is None else forecast.years == year
                )
                cohort = {"best_of": forecast.best_of, "year": year}
                loss_rows.append({
                    **cohort,
                    "market": market,
                    "matches": int(mask.sum()),
                    "reference_crps": float(reference_crps[mask].mean()),
                    "model_crps": float(model_crps[mask].mean()),
                    "daily_crps": float(daily_crps[mask].mean()) if market == "total_games" else None,
                    "reference_median_mae": float(reference_mae[mask].mean()),
                    "model_median_mae": float(model_mae[mask].mean()),
                    "daily_median_mae": (
                        float(daily_mae[mask].mean()) if market == "total_games" else None
                    ),
                })
                for line in lines:
                    calibration_rows.append({
                        **cohort,
                        **_calibration_row(
                            market, f"over_{line}", observed[mask] > line,
                            _game_line_probability(reference[mask], support, line),
                            _game_line_probability(candidate[mask], support, line), 10,
                        ),
                    })
    return GamePairDiagnostics(
        pl.DataFrame(loss_rows, schema_overrides={"year": pl.Int32}),
        pl.DataFrame(calibration_rows, schema_overrides={"year": pl.Int32}),
    )
