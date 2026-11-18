from collections import Counter, defaultdict
from statistics import median

import polars as pl

from tennis_prediction.datasets.score_markets import ScoreMarketSplits


def evaluate_score_market_baselines(splits: ScoreMarketSplits) -> pl.DataFrame:
    """Score training-period empirical references on chronological validation rows.

    Compare format-only groups with format/surface groups. An unseen or missing
    surface uses its format group. An unseen format raises an error rather
    than assigning probabilities to incompatible match outcomes.
    """

    train = splits.train.features.select("best_of", "surface").hstack(splits.train.targets)
    validation = splits.validation.features.select("best_of", "surface").hstack(
        splits.validation.targets
    )
    results = []
    for market, metric in (("correct_set_score", "brier"), ("total_games", "mae")):
        by_format = defaultdict(list)
        by_surface = defaultdict(list)
        for best_of, surface, outcome in train.select("best_of", "surface", market).drop_nulls(
            market
        ).iter_rows():
            by_format[best_of].append(outcome)
            if surface:
                by_surface[best_of, surface].append(outcome)

        if market == "correct_set_score":
            format_predictions = {
                key: _score_distribution(values) for key, values in by_format.items()
            }
            surface_predictions = {
                key: _score_distribution(values) for key, values in by_surface.items()
            }
        else:
            format_predictions = {key: median(values) for key, values in by_format.items()}
            surface_predictions = {key: median(values) for key, values in by_surface.items()}

        format_losses = []
        surface_losses = []
        for best_of, surface, outcome in validation.select(
            "best_of", "surface", market
        ).drop_nulls(market).iter_rows():
            if best_of not in format_predictions:
                raise ValueError(f"Missing training labels for {market} at best_of={best_of!r}")
            format_prediction = format_predictions[best_of]
            surface_prediction = surface_predictions.get((best_of, surface), format_prediction)
            if market == "correct_set_score":
                format_losses.append(_brier_loss(format_prediction, outcome))
                surface_losses.append(_brier_loss(surface_prediction, outcome))
            else:
                format_losses.append(abs(format_prediction - outcome))
                surface_losses.append(abs(surface_prediction - outcome))

        for baseline, losses in (
            ("format_only", format_losses),
            ("format_and_surface", surface_losses),
        ):
            results.append(
                {
                    "market": market,
                    "baseline": baseline,
                    "metric": metric,
                    "training_matches": sum(map(len, by_format.values())),
                    "validation_matches": len(losses),
                    "value": sum(losses) / len(losses) if losses else None,
                }
            )

    return pl.DataFrame(results, schema_overrides={"value": pl.Float64})


def _score_distribution(outcomes: list[str]) -> dict[str, float]:
    return {score: count / len(outcomes) for score, count in Counter(outcomes).items()}


def _brier_loss(probabilities: dict[str, float], observed: str) -> float:
    # The observed class contributes one even when it was absent from training.
    return sum(
        (probability - (score == observed)) ** 2
        for score, probability in probabilities.items()
    ) + (observed not in probabilities)
