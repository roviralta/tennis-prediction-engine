from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from tennis_prediction.datasets.encoding import TrainingFeatureEncoder
from tennis_prediction.datasets.imputation import TrainingFeatureImputer
from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import ScoreMarketTable


@dataclass(frozen=True)
class ScoreModelDiagnostics:
    """Validation errors and probability calibration by format and year."""

    losses: pl.DataFrame
    calibration: pl.DataFrame


@dataclass(frozen=True)
class _GameForecasts:
    rows: np.ndarray
    counts: np.ndarray
    observed: np.ndarray
    reference_cdf: np.ndarray
    model_cdf: np.ndarray


@dataclass(frozen=True)
class _FormatEvaluation:
    best_of: int
    rows: pl.Series
    classes: np.ndarray
    observed: np.ndarray
    reference: np.ndarray
    probabilities: np.ndarray
    losses: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]]
    games: _GameForecasts | None


def _multiclass_brier(
    probabilities: np.ndarray, classes: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    # An unseen observed class contributes one through the final term.
    return 1 + (probabilities ** 2).sum(axis=1) - 2 * (
        probabilities * (observed[:, None] == classes)
    ).sum(axis=1)


def _game_count_losses(
    cdf: np.ndarray, game_counts: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a discrete CDF, including observations outside training support."""

    lower_median = game_counts[(cdf >= 0.5).argmax(axis=1)]
    upper_median = game_counts[(cdf > 0.5).argmax(axis=1)]
    median = (lower_median + upper_median) / 2
    crps = ((cdf[:, :-1] - (observed[:, None] <= game_counts[:-1])) ** 2).sum(axis=1)
    crps += np.maximum(game_counts[0] - observed, 0)
    crps += np.maximum(observed - game_counts[-1], 0)
    return crps, np.abs(median - observed)


def _empirical_cdf(observed: np.ndarray, game_counts: np.ndarray) -> np.ndarray:
    frequencies = np.bincount(observed - game_counts[0], minlength=game_counts.size)
    return frequencies.cumsum() / observed.size


@dataclass(frozen=True)
class FittedScoreFormat:
    columns: tuple[str, ...]
    imputer: TrainingFeatureImputer
    encoder: TrainingFeatureEncoder
    model: HistGradientBoostingClassifier | None
    classes: np.ndarray
    reference: np.ndarray

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.broadcast_to(self.reference, (features.height, self.classes.size))
        encoded = self.encoder.transform(self.imputer.transform(features.select(self.columns)))
        return self.model.predict_proba(encoded.to_numpy())


def fit_score_format(features: pl.DataFrame, target: pl.Series) -> FittedScoreFormat:
    """Fit the fixed score classifier and its training-only preprocessing."""
    available = tuple(
        column for column in features.columns
        if features[column].null_count() < features.height
    )
    imputer = TrainingFeatureImputer()
    encoder = TrainingFeatureEncoder()
    encoded = encoder.fit_transform(imputer.fit_transform(features.select(available)))
    labels = target.to_numpy()
    classes, frequencies = np.unique(labels, return_counts=True)
    model = None
    if classes.size > 1:
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )
        model.fit(encoded.to_numpy(), labels)
    return FittedScoreFormat(available, imputer, encoder, model, classes, frequencies / labels.size)


def fit_game_distributions(
    training_targets: pl.DataFrame, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    training_games = training_targets.drop_nulls("total_games")
    games = training_games["total_games"].to_numpy()
    if not games.size:
        raise ValueError("Missing total-games training labels")

    game_counts = np.arange(games.min(), games.max() + 1)
    format_cdf = _empirical_cdf(games, game_counts)
    played_sets = np.array([sum(map(int, score.split("-"))) for score in classes])
    training_sets = np.array([
        sum(map(int, score.split("-")))
        for score in training_games["correct_set_score"]
    ])
    cdf_by_set_count = {}
    for set_count in np.unique(played_sets):
        sample = games[training_sets == set_count]
        cdf_by_set_count[set_count] = (
            _empirical_cdf(sample, game_counts) if sample.size else format_cdf
        )
    conditional_cdfs = np.stack([cdf_by_set_count[count] for count in played_sets])
    return game_counts, format_cdf, conditional_cdfs, games.size


def _evaluate_format(
    train: ScoreMarketTable, validation: ScoreMarketTable, best_of: int
) -> _FormatEvaluation:
    train_mask = (train.features["best_of"] == best_of) & train.targets[
        "correct_set_score"
    ].is_not_null()
    validation_mask = (validation.features["best_of"] == best_of) & validation.targets[
        "correct_set_score"
    ].is_not_null()
    training_features = train.features.filter(train_mask)
    training_targets = train.targets.filter(train_mask)
    validation_targets = validation.targets.filter(validation_mask)
    if not training_targets.height:
        raise ValueError(f"Missing score training labels for best_of={best_of}")

    fitted = fit_score_format(training_features, training_targets["correct_set_score"])
    classes = fitted.classes
    labels = training_targets["correct_set_score"].to_numpy()
    outcomes = validation_targets["correct_set_score"].to_numpy()
    reference = np.broadcast_to(fitted.reference, (outcomes.size, classes.size))
    probabilities = fitted.predict(validation.features.filter(validation_mask))

    losses = {
        ("correct_set_score", "brier"): (
            _multiclass_brier(reference, classes, outcomes),
            _multiclass_brier(probabilities, classes, outcomes),
            labels.size,
        )
    }
    game_rows = validation_targets["total_games"].is_not_null().to_numpy()
    game_forecasts = None
    if game_rows.any():
        game_forecasts, game_losses = _evaluate_games(
            training_targets, validation_targets, game_rows, classes, probabilities
        )
        losses.update(game_losses)
    return _FormatEvaluation(
        best_of, validation_mask, classes, outcomes, reference, probabilities, losses, game_forecasts
    )


def _evaluate_games(
    training_targets: pl.DataFrame,
    validation_targets: pl.DataFrame,
    game_rows: np.ndarray,
    classes: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[_GameForecasts, dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]]]:
    game_counts, format_cdf, conditional_cdfs, training_count = fit_game_distributions(
        training_targets, classes
    )
    observed_games = validation_targets["total_games"].drop_nulls().to_numpy()
    reference_cdf = np.broadcast_to(format_cdf, (observed_games.size, game_counts.size))
    model_cdf = probabilities[game_rows] @ conditional_cdfs
    reference_crps, reference_mae = _game_count_losses(reference_cdf, game_counts, observed_games)
    model_crps, model_mae = _game_count_losses(model_cdf, game_counts, observed_games)
    return _GameForecasts(game_rows, game_counts, observed_games, reference_cdf, model_cdf), {
        ("total_games", "crps"): (reference_crps, model_crps, training_count),
        ("total_games", "mae"): (reference_mae, model_mae, training_count),
    }


def _evaluate_formats(splits: ScoreMarketSplits) -> list[_FormatEvaluation]:
    validation_formats = splits.validation.features.filter(
        splits.validation.targets["correct_set_score"].is_not_null()
    )["best_of"].unique().sort()
    if validation_formats.is_empty():
        raise ValueError("No eligible validation score labels")
    if validation_formats.null_count():
        raise ValueError("Eligible validation score labels require best_of")

    return [
        _evaluate_format(splits.train, splits.validation, best_of)
        for best_of in validation_formats
    ]


def evaluate_player_score_models(splits: ScoreMarketSplits) -> pl.DataFrame:
    """Compare player-aware and format-only forecasts on validation matches only."""

    losses_by_metric = {}
    for evaluation in _evaluate_formats(splits):
        for key, losses in evaluation.losses.items():
            losses_by_metric.setdefault(key, []).append(losses)
    return pl.DataFrame([
        {
            "market": market,
            "metric": metric,
            "training_matches": sum(size for _, _, size in parts),
            "validation_matches": sum(reference.size for reference, _, _ in parts),
            "format_only": float(np.concatenate([reference for reference, _, _ in parts]).mean()),
            "player_model": float(np.concatenate([model for _, model, _ in parts]).mean()),
        }
        for (market, metric), parts in losses_by_metric.items()
    ])


_GAME_LINES = {3: (20.5, 22.5, 24.5), 5: (36.5, 38.5, 40.5)}


def _calibration_row(
    market: str,
    event: str,
    observed: np.ndarray,
    reference: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int,
) -> dict[str, object]:
    def calibration_error(predicted: np.ndarray) -> float:
        bins = np.minimum((predicted * bin_count).astype(int), bin_count - 1)
        # Summed gaps divided by sample size are equivalent to weighted bin mean gaps.
        gaps = np.bincount(bins, weights=predicted - observed, minlength=bin_count)
        return float(np.abs(gaps).sum() / observed.size)

    return {
        "market": market,
        "event": event,
        "matches": observed.size,
        "observed_rate": float(observed.mean()),
        "format_probability": float(reference.mean()),
        "model_probability": float(probabilities.mean()),
        "format_ece": calibration_error(reference),
        "model_ece": calibration_error(probabilities),
    }


def _game_line_probability(cdf: np.ndarray, counts: np.ndarray, line: float) -> np.ndarray:
    index = counts.searchsorted(line, side="right") - 1
    if index < 0:
        return np.ones(cdf.shape[0])
    if index >= counts.size - 1:
        return np.zeros(cdf.shape[0])
    return np.clip(1 - cdf[:, index], 0, 1)


def diagnose_player_score_models(
    splits: ScoreMarketSplits, bin_count: int = 10
) -> ScoreModelDiagnostics:
    """Check fixed-bin calibration and errors on validation cohorts.

    Each model is fitted once per format. A null year is the pooled cohort
    for that format. Game-line events use half-point thresholds without pushes.
    """

    if bin_count < 1:
        raise ValueError("bin_count must be greater than zero")
    dates = splits.validation.metadata["tourney_date"]
    if dates.null_count():
        raise ValueError("Validation diagnostics require dated rows")

    loss_rows = []
    calibration_rows = []
    for result in _evaluate_formats(splits):
        years = dates.filter(result.rows).to_numpy() // 10000
        classes = sorted(set(result.classes) | set(result.observed))
        class_columns = {score: index for index, score in enumerate(result.classes)}
        for year in [None, *np.unique(years).tolist()]:
            cohort = {"best_of": result.best_of, "year": year}
            scores = np.ones(years.size, dtype=bool) if year is None else years == year
            games = result.games
            game_rows = scores[games.rows] if games is not None else None

            for (market, metric), (reference, model, training_matches) in result.losses.items():
                mask = scores if market == "correct_set_score" else game_rows
                if not mask.any():
                    continue
                loss_rows.append({
                    **cohort,
                    "market": market,
                    "metric": metric,
                    "training_matches": training_matches,
                    "validation_matches": int(mask.sum()),
                    "format_only": float(reference[mask].mean()),
                    "player_model": float(model[mask].mean()),
                })

            for score in classes:
                index = class_columns.get(score)
                if index is None:
                    reference = probabilities = np.zeros(int(scores.sum()))
                else:
                    reference = result.reference[scores, index]
                    probabilities = result.probabilities[scores, index]
                calibration_rows.append({
                    **cohort,
                    **_calibration_row(
                        "correct_set_score", score, result.observed[scores] == score,
                        reference, probabilities, bin_count,
                    ),
                })

            if games is not None and game_rows.any():
                for line in _GAME_LINES.get(result.best_of, ()):
                    calibration_rows.append({
                        **cohort,
                        **_calibration_row(
                            "total_games", f"over_{line}", games.observed[game_rows] > line,
                            _game_line_probability(games.reference_cdf[game_rows], games.counts, line),
                            _game_line_probability(games.model_cdf[game_rows], games.counts, line),
                            bin_count,
                        ),
                    })

    return ScoreModelDiagnostics(
        losses=pl.DataFrame(loss_rows, schema_overrides={"year": pl.Int32}),
        calibration=pl.DataFrame(calibration_rows, schema_overrides={"year": pl.Int32}),
    )
