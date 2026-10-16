from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tennis_prediction.datasets.encoding import EncodedPredictionSplits
from tennis_prediction.datasets.encoding import encode_prediction_splits
from tennis_prediction.datasets.imputation import ImputedPredictionSplits
from tennis_prediction.datasets.imputation import impute_prediction_splits
from tennis_prediction.datasets.schema import PredictionSplits
from tennis_prediction.datasets.schema import PredictionTable


RANKING_FEATURE_COLUMNS = (
    "player_a_rank",
    "player_b_rank",
    "rank_difference",
    "player_a_rank_points",
    "player_b_rank_points",
    "rank_points_difference",
)


@dataclass(frozen=True)
class BaselineResult:
    """Fitted logistic baseline and held-out metrics for reproducible review."""

    model: BaseEstimator
    imputed: ImputedPredictionSplits
    encoded: EncodedPredictionSplits
    metrics: pl.DataFrame


@dataclass(frozen=True)
class BaselineComparisonResult:
    """Full-feature and ranking-only results evaluated on identical splits."""

    full_features: BaselineResult
    ranking_only: BaselineResult
    metrics: pl.DataFrame


def _select_split_features(
    splits: PredictionSplits,
    feature_columns: tuple[str, ...] | None,
) -> PredictionSplits:
    if feature_columns is None:
        return splits
    if not feature_columns:
        raise ValueError("feature_columns must contain at least one feature")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns cannot contain duplicates")

    selected_splits = {}
    for split_name in ("train", "validation", "test", "undated"):
        table = getattr(splits, split_name)
        missing = [
            column for column in feature_columns if column not in table.features.columns
        ]
        if missing:
            raise ValueError(f"Missing requested features in {split_name}: {missing}")
        selected_splits[split_name] = PredictionTable(
            metadata=table.metadata,
            features=table.features.select(feature_columns),
            target=table.target,
        )
    return PredictionSplits(**selected_splits)


def _metrics_for_split(
    split_name: str,
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int | str]:
    predictions = (probabilities >= 0.5).astype(np.uint8)
    return {
        "split": split_name,
        "rows": int(target.size),
        "accuracy": float(accuracy_score(target, predictions)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probabilities)),
    }


def _train_baseline_with_estimator(
    splits: PredictionSplits,
    model: BaseEstimator,
    feature_columns: tuple[str, ...] | None = None,
    evaluate_validation: bool = True,
    evaluate_test: bool = True,
) -> BaselineResult:
    splits = _select_split_features(splits, feature_columns)
    if splits.train.metadata.height == 0:
        raise ValueError("Training split is empty")
    if evaluate_validation and splits.validation.metadata.height == 0:
        raise ValueError("Validation split is empty")
    if evaluate_test and splits.test.metadata.height == 0:
        raise ValueError("Test split is empty")

    imputed = impute_prediction_splits(splits)
    encoded = encode_prediction_splits(imputed)
    training = encoded.splits.train
    training_target = training.target.to_numpy().astype(np.uint8)
    if np.unique(training_target).size != 2:
        raise ValueError("Training split must contain both target classes")

    model.fit(training.features.to_numpy(), training_target)

    metric_rows = []
    evaluation_splits = tuple(
        split_name
        for split_name, should_evaluate in (
            ("validation", evaluate_validation),
            ("test", evaluate_test),
        )
        if should_evaluate
    )
    if not evaluation_splits:
        raise ValueError("At least one evaluation split must be selected")
    for split_name in evaluation_splits:
        split = getattr(encoded.splits, split_name)
        target = split.target.to_numpy().astype(np.uint8)
        if target.size == 0:
            raise ValueError(f"{split_name.capitalize()} split is empty")
        probabilities = model.predict_proba(split.features.to_numpy())[:, 1]
        metric_rows.append(_metrics_for_split(split_name, target, probabilities))

    return BaselineResult(
        model=model,
        imputed=imputed,
        encoded=encoded,
        metrics=pl.DataFrame(metric_rows),
    )


def train_logistic_baseline(
    splits: PredictionSplits,
    feature_columns: tuple[str, ...] | None = None,
    evaluate_test: bool = True,
) -> BaselineResult:
    """Fit a transparent logistic-regression baseline on training data only.

    Missing-value statistics and categorical levels are learned from training
    rows only. Validation and, optionally, test are evaluated but never used to
    fit the model. The output metrics treat each prediction perspective as a
    row; the two views of each source match remain together in the same split.
    """

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=0),
    )
    return _train_baseline_with_estimator(
        splits,
        model,
        feature_columns=feature_columns,
        evaluate_test=evaluate_test,
    )


def train_hist_gradient_boosting_baseline(
    splits: PredictionSplits,
    feature_columns: tuple[str, ...] | None = None,
    evaluate_test: bool = True,
) -> BaselineResult:
    """Fit the selected nonlinear gradient-boosting candidate on training data.

    The fixed settings were selected using the 2019 validation year and checked
    on 2020-2021. Internal random early stopping is disabled to keep the
    chronological evaluation protocol explicit.
    """

    model = _hist_gradient_boosting_estimator()
    return _train_baseline_with_estimator(
        splits,
        model,
        feature_columns=feature_columns,
        evaluate_test=evaluate_test,
    )


def _hist_gradient_boosting_estimator() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def refit_hist_gradient_boosting_on_train_and_validation(
    splits: PredictionSplits,
) -> BaselineResult:
    """Refit selected settings on train+validation, then evaluate test once.

    Model selection must be finished before calling this function. This uses
    every labeled match through the validation cutoff for fitting and does not
    use test outcomes until the final metric calculation.
    """

    if splits.train.metadata.height == 0 or splits.validation.metadata.height == 0:
        raise ValueError("Training and validation splits must both be non-empty")

    combined_training = PredictionTable(
        metadata=pl.concat([splits.train.metadata, splits.validation.metadata]),
        features=pl.concat([splits.train.features, splits.validation.features]),
        target=pl.concat([splits.train.target, splits.validation.target]),
    )
    empty_validation = PredictionTable(
        metadata=splits.validation.metadata.head(0),
        features=splits.validation.features.head(0),
        target=splits.validation.target.head(0),
    )
    refit_splits = PredictionSplits(
        train=combined_training,
        validation=empty_validation,
        test=splits.test,
        undated=splits.undated,
    )
    return _train_baseline_with_estimator(
        refit_splits,
        _hist_gradient_boosting_estimator(),
        evaluate_validation=False,
        evaluate_test=True,
    )


def compare_full_and_ranking_baselines(
    splits: PredictionSplits,
) -> BaselineComparisonResult:
    """Compare the reviewed feature set with rankings alone on the same splits."""

    full_features = train_logistic_baseline(splits)
    ranking_only = train_logistic_baseline(
        splits,
        feature_columns=RANKING_FEATURE_COLUMNS,
    )
    comparison_metrics = pl.concat(
        [
            full_features.metrics.with_columns(pl.lit("full_features").alias("model")),
            ranking_only.metrics.with_columns(pl.lit("ranking_only").alias("model")),
        ]
    ).select("model", "split", "rows", "accuracy", "log_loss", "brier_score")
    return BaselineComparisonResult(
        full_features=full_features,
        ranking_only=ranking_only,
        metrics=comparison_metrics,
    )
