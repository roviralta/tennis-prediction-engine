from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import joblib
import polars as pl
from sklearn.base import BaseEstimator

from tennis_prediction.datasets.encoding import TrainingFeatureEncoder
from tennis_prediction.datasets.imputation import TrainingFeatureImputer
from tennis_prediction.models.winner.baseline import BaselineResult


ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class WinnerModelArtifact:
    """A fitted winner model and the preprocessing needed to reuse it."""

    model: BaseEstimator
    imputer: TrainingFeatureImputer
    encoder: TrainingFeatureEncoder
    feature_columns: tuple[str, ...]
    training_cutoff_date: int | None = None
    format_version: int = ARTIFACT_FORMAT_VERSION


def create_winner_model_artifact(result: BaselineResult) -> WinnerModelArtifact:
    """Collect a fitted training result into a self-contained prediction artifact."""

    feature_columns = result.imputed.imputer.feature_columns
    if feature_columns is None:
        raise ValueError("The result does not contain a fitted feature imputer")
    training_dates = result.imputed.splits.train.metadata["tourney_date"].drop_nulls()
    training_cutoff_date = training_dates.max() if len(training_dates) else None
    return WinnerModelArtifact(
        model=result.model,
        imputer=result.imputed.imputer,
        encoder=result.encoded.encoder,
        feature_columns=tuple(feature_columns),
        training_cutoff_date=training_cutoff_date,
    )


def save_winner_model_artifact(
    artifact: WinnerModelArtifact,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a model artifact atomically; refuse to replace files by default."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Model artifact already exists: {destination}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(artifact, temporary_path)
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            # Linking is atomic and fails if another process created the target.
            os.link(temporary_path, destination)
            temporary_path.unlink()
        return destination
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_winner_model_artifact(path: str | Path) -> WinnerModelArtifact:
    """Load a trusted artifact created by this package and validate its format."""

    artifact = joblib.load(Path(path))
    if not isinstance(artifact, WinnerModelArtifact):
        raise ValueError("File does not contain a winner model artifact")
    if artifact.format_version != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported model artifact version: {artifact.format_version}"
        )
    if not artifact.feature_columns:
        raise ValueError("Model artifact has no input feature columns")
    return artifact


def predict_winner_probabilities(
    artifact: WinnerModelArtifact,
    features: pl.DataFrame,
) -> pl.DataFrame:
    """Predict player-A win probabilities for rows in the fitted feature schema."""

    if features.columns != list(artifact.feature_columns):
        raise ValueError(
            "Feature columns and order must match the model artifact; "
            f"expected {list(artifact.feature_columns)}, found {features.columns}"
        )
    imputed = artifact.imputer.transform(features)
    encoded = artifact.encoder.transform(imputed)
    probabilities = artifact.model.predict_proba(encoded.to_numpy())[:, 1]
    return pl.DataFrame(
        {
            "probability_player_a_wins": probabilities,
            "predicted_player_a_wins": probabilities >= 0.5,
        }
    )
