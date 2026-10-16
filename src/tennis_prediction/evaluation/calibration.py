from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss

from tennis_prediction.models.winner.baseline import BaselineResult


@dataclass(frozen=True)
class CalibrationResult:
    """Validation calibration table and its weighted absolute probability gap."""

    table: pl.DataFrame
    expected_calibration_error: float
    total_rows: int
    bin_count: int


@dataclass(frozen=True)
class ValidationSubgroupResult:
    """Validation metrics grouped by tournament surface and best player rank."""

    by_surface: pl.DataFrame
    by_best_rank_band: pl.DataFrame
    probability_bin_count: int


def evaluate_validation_calibration(
    result: BaselineResult,
    bin_count: int = 10,
) -> CalibrationResult:
    """Compare validation probabilities with observed win rates in fixed bins.

    The function is descriptive: it does not adjust model probabilities. The
    final test partition is deliberately not read by this validation diagnostic.
    """

    if bin_count < 1:
        raise ValueError("bin_count must be greater than zero")

    validation = result.encoded.splits.validation
    if validation.metadata.height == 0:
        raise ValueError("Validation split is empty")

    target = validation.target.to_numpy().astype(np.uint8)
    probabilities = result.model.predict_proba(validation.features.to_numpy())[:, 1]
    bin_indexes = np.minimum((probabilities * bin_count).astype(np.int64), bin_count - 1)

    rows: list[dict[str, int | float | str]] = []
    weighted_absolute_gap = _expected_calibration_error(
        target,
        probabilities,
        bin_count,
    )
    for bin_index in range(bin_count):
        in_bin = bin_indexes == bin_index
        count = int(in_bin.sum())
        if count == 0:
            continue

        lower_bound = bin_index / bin_count
        upper_bound = (bin_index + 1) / bin_count
        mean_probability = float(probabilities[in_bin].mean())
        observed_win_rate = float(target[in_bin].mean())
        absolute_gap = abs(mean_probability - observed_win_rate)
        upper_bracket = "]" if bin_index == bin_count - 1 else ")"
        rows.append(
            {
                "interval": f"[{lower_bound:.1f}, {upper_bound:.1f}{upper_bracket}",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "rows": count,
                "mean_predicted_probability": mean_probability,
                "observed_win_rate": observed_win_rate,
                "absolute_gap": absolute_gap,
            }
        )

    return CalibrationResult(
        table=pl.DataFrame(rows),
        expected_calibration_error=weighted_absolute_gap,
        total_rows=int(target.size),
        bin_count=bin_count,
    )


def _expected_calibration_error(
    target: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int,
) -> float:
    bin_indexes = np.minimum((probabilities * bin_count).astype(np.int64), bin_count - 1)
    weighted_gap = 0.0
    for bin_index in range(bin_count):
        in_bin = bin_indexes == bin_index
        if in_bin.any():
            gap = abs(float(probabilities[in_bin].mean()) - float(target[in_bin].mean()))
            weighted_gap += float(in_bin.sum()) / target.size * gap
    return weighted_gap


def _best_rank_band_labels(features: pl.DataFrame) -> list[str]:
    player_a_ranks = features["player_a_rank"].to_list()
    player_b_ranks = features["player_b_rank"].to_list()
    player_a_missing = features["player_a_rank__missing"].to_list()
    player_b_missing = features["player_b_rank__missing"].to_list()

    labels = []
    for rank_a, rank_b, missing_a, missing_b in zip(
        player_a_ranks,
        player_b_ranks,
        player_a_missing,
        player_b_missing,
        strict=True,
    ):
        known_ranks = []
        if missing_a == 0:
            known_ranks.append(rank_a)
        if missing_b == 0:
            known_ranks.append(rank_b)

        if not known_ranks:
            labels.append("unranked")
            continue

        best_rank = min(known_ranks)
        if best_rank <= 10:
            labels.append("1-10")
        elif best_rank <= 50:
            labels.append("11-50")
        elif best_rank <= 100:
            labels.append("51-100")
        else:
            labels.append("101+")
    return labels


def _subgroup_metrics(
    result: BaselineResult,
    group_name: str,
    group_labels: list[str],
    probabilities: np.ndarray,
    target: np.ndarray,
    bin_count: int,
) -> pl.DataFrame:
    validation = result.encoded.splits.validation
    categories = sorted(set(group_labels))
    rows = []
    for category in categories:
        mask = np.asarray([label == category for label in group_labels])
        subgroup_target = target[mask]
        subgroup_probabilities = probabilities[mask]
        source_matches = (
            validation.metadata
            .filter(pl.Series("_subgroup_mask", mask.tolist()))["match_index"]
            .n_unique()
        )
        rows.append(
            {
                group_name: category,
                "rows": int(mask.sum()),
                "source_matches": int(source_matches),
                "accuracy": float(
                    accuracy_score(subgroup_target, subgroup_probabilities >= 0.5)
                ),
                "log_loss": float(
                    log_loss(subgroup_target, subgroup_probabilities, labels=[0, 1])
                ),
                "brier_score": float(
                    brier_score_loss(subgroup_target, subgroup_probabilities)
                ),
                "ece": _expected_calibration_error(
                    subgroup_target,
                    subgroup_probabilities,
                    bin_count,
                ),
            }
        )
    return pl.DataFrame(rows)


def evaluate_validation_subgroups(
    result: BaselineResult,
    probability_bin_count: int = 10,
) -> ValidationSubgroupResult:
    """Report calibration and predictive metrics by surface and ranking band.

    Ranking groups use the best available rank among the two players. If only
    one player has a ranking, that rank determines the group; if neither does,
    the match is labeled ``unranked``. Both perspectives of a source match are
    kept in the same group. The final test partition is not read.
    """

    if probability_bin_count < 1:
        raise ValueError("probability_bin_count must be greater than zero")
    validation = result.encoded.splits.validation
    if validation.metadata.height == 0:
        raise ValueError("Validation split is empty")

    target = validation.target.to_numpy().astype(np.uint8)
    probabilities = result.model.predict_proba(validation.features.to_numpy())[:, 1]
    imputed_features = result.imputed.splits.validation.features
    surfaces = [
        "unknown" if surface == "__MISSING__" else str(surface)
        for surface in imputed_features["surface"].cast(pl.String).to_list()
    ]
    rank_bands = _best_rank_band_labels(imputed_features)

    return ValidationSubgroupResult(
        by_surface=_subgroup_metrics(
            result,
            "surface",
            surfaces,
            probabilities,
            target,
            probability_bin_count,
        ),
        by_best_rank_band=_subgroup_metrics(
            result,
            "best_player_rank_band",
            rank_bands,
            probabilities,
            target,
            probability_bin_count,
        ),
        probability_bin_count=probability_bin_count,
    )


def evaluate_validation_by_year(result: BaselineResult) -> pl.DataFrame:
    """Summarize validation metrics by calendar year without reading test rows."""

    validation = result.encoded.splits.validation
    if validation.metadata.height == 0:
        raise ValueError("Validation split is empty")

    target = validation.target.to_numpy().astype(np.uint8)
    probabilities = result.model.predict_proba(validation.features.to_numpy())[:, 1]
    dates = validation.metadata["tourney_date"].to_list()
    years = [str(match_date // 10000) for match_date in dates]
    return _subgroup_metrics(
        result,
        "year",
        years,
        probabilities,
        target,
        bin_count=10,
    )
