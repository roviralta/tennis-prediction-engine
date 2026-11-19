from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from tennis_prediction.datasets.score_markets import ScoreMarketSplits, ScoreMarketTable
from tennis_prediction.models.statistics.score_model import (
    FittedScoreFormat,
    _empirical_cdf,
    _game_count_losses,
    fit_score_format,
)
from tennis_prediction.models.statistics.workflow import TRAIN_END, VALIDATION_END


STAT_BINS = {"aces": (3, 18), "double_faults": (2, 12), "breaks": (2, 10)}
def automatic_stat_thresholds(counts: np.ndarray, cdf: np.ndarray) -> range:
    """Show count thresholds through at least four or the forecast's 95th percentile."""

    upper_index = min(cdf.searchsorted(0.95), counts.size - 1)
    return range(1, max(4, int(counts[upper_index])) + 1)


def reverse_player_features(features: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    for column in features.columns:
        if column.startswith("player_a_"):
            expressions.append(pl.col(column.replace("player_a_", "player_b_", 1)).alias(column))
        elif column.startswith("player_b_"):
            expressions.append(pl.col(column.replace("player_b_", "player_a_", 1)).alias(column))
        elif "_difference" in column:
            expressions.append((-pl.col(column)).alias(column))
        else:
            expressions.append(pl.col(column))
    return features.select(expressions)


def _market_data(
    table: ScoreMarketTable, market: str, best_of: int
) -> tuple[pl.DataFrame, pl.Series, int]:
    columns = (f"player_a_{market}", f"player_b_{market}")
    mask = (table.features["best_of"] == best_of) & table.targets.select(
        pl.all_horizontal(pl.col(*columns).is_not_null())
    ).to_series()
    features = table.features.filter(mask)
    targets = table.targets.select(columns).filter(mask)
    observed = pl.concat([targets[columns[0]], targets[columns[1]]])
    return pl.concat([features, reverse_player_features(features)]), observed, features.height


@dataclass(frozen=True)
class StatDistribution:
    classifier: FittedScoreFormat
    counts: np.ndarray
    conditional_cdfs: np.ndarray
    reference_cdf: np.ndarray

    def forecast(self, features: pl.DataFrame) -> np.ndarray:
        return self.classifier.predict(features) @ self.conditional_cdfs


@dataclass(frozen=True)
class StatMarketArtifact:
    models: dict[tuple[str, int], StatDistribution]
    training_cutoff_date: int = TRAIN_END
    format_version: int = 1


def _fit_distribution(
    features: pl.DataFrame, observed: pl.Series, width: int, tail: int
) -> StatDistribution:
    counts = observed.to_numpy()
    bins = np.minimum(counts, tail) // width
    classifier = fit_score_format(features, pl.Series("bin", bins))
    support = np.arange(counts.min(), counts.max() + 1)
    conditional = np.stack([
        _empirical_cdf(counts[bins == label], support) for label in classifier.classes
    ])
    return StatDistribution(classifier, support, conditional, _empirical_cdf(counts, support))


def fit_stat_markets(training: ScoreMarketTable) -> StatMarketArtifact:
    dates = training.metadata["tourney_date"]
    if dates.is_empty() or dates.null_count() or dates.max() > TRAIN_END:
        raise ValueError("Statistical market training requires dated rows through 2018")
    models = {}
    for market, (width, tail) in STAT_BINS.items():
        for best_of in (3, 5):
            features, observed, matches = _market_data(training, market, best_of)
            if matches:
                models[market, best_of] = _fit_distribution(features, observed, width, tail)
    if not models:
        raise ValueError("No eligible statistical market training labels")
    return StatMarketArtifact(models)


def evaluate_stat_markets(artifact: StatMarketArtifact, splits: ScoreMarketSplits) -> pl.DataFrame:
    rows = []
    for period, table, lower, upper in (
        ("validation", splits.validation, TRAIN_END, VALIDATION_END),
        ("holdout", splits.test, VALIDATION_END, None),
    ):
        dates = table.metadata["tourney_date"]
        if (
            dates.is_empty() or dates.null_count() or dates.min() <= lower
            or (upper is not None and dates.max() > upper)
        ):
            raise ValueError(f"Invalid chronological dates in {period}")
        for (market, best_of), distribution in artifact.models.items():
            features, observed, matches = _market_data(table, market, best_of)
            if not matches:
                continue
            predicted = distribution.forecast(features)
            reference = np.broadcast_to(distribution.reference_cdf, predicted.shape)
            reference_crps, reference_mae = _game_count_losses(
                reference, distribution.counts, observed.to_numpy()
            )
            crps, mae = _game_count_losses(predicted, distribution.counts, observed.to_numpy())
            rows.append({
                "period": period, "market": market, "best_of": best_of,
                "matches": matches, "observations": len(observed),
                "reference_crps": float(reference_crps.mean()), "model_crps": float(crps.mean()),
                "reference_median_mae": float(reference_mae.mean()), "model_median_mae": float(mae.mean()),
            })
    if not rows:
        raise ValueError("No eligible labels for statistical market evaluation")
    return pl.DataFrame(rows)


def select_stat_markets(artifact: StatMarketArtifact, report: pl.DataFrame) -> StatMarketArtifact:
    qualified = report.group_by("market", "best_of").agg(
        pl.col("period").n_unique().alias("periods"),
        ((pl.col("matches") >= 200) & (pl.col("model_crps") < pl.col("reference_crps")))
        .all().alias("qualified"),
    ).filter((pl.col("periods") == 2) & pl.col("qualified"))
    identities = set(qualified.select("market", "best_of").iter_rows())
    return StatMarketArtifact({key: model for key, model in artifact.models.items() if key in identities})


def load_stat_artifact(path: Path) -> StatMarketArtifact:
    artifact = joblib.load(path)
    if not isinstance(artifact, StatMarketArtifact) or artifact.format_version != 1:
        raise ValueError("Unsupported statistics model artifact")
    return artifact
