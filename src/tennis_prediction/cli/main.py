from datetime import datetime
from pathlib import Path

import polars as pl
import typer

from tennis_prediction.datasets.build import build_historical_prediction_table
from tennis_prediction.datasets.markets import audit_market_coverage
from tennis_prediction.datasets.market_targets import build_market_target_table
from tennis_prediction.datasets.market_targets import audit_market_targets
from tennis_prediction.ingestion.atp.loader import ATPDataLoader
from tennis_prediction.models.winner import create_winner_model_artifact
from tennis_prediction.models.winner import fit_production_hist_gradient_boosting
from tennis_prediction.models.winner import build_match_features
from tennis_prediction.models.winner import load_winner_model_artifact
from tennis_prediction.models.winner import predict_winner_probabilities
from tennis_prediction.models.winner import save_winner_model_artifact

app = typer.Typer(
    name="tennis",
    help="ATP Tennis Prediction & Analytics Engine",
)


@app.command()
def version() -> None:
    """Show the current application version."""
    typer.echo("Tennis Prediction Engine v0.1.0")


@app.command()
def status() -> None:
    """Show the application status."""
    typer.echo("Tennis Prediction Engine is running.")


@app.command("train-winner")
def train_winner(
    data_dir: Path = typer.Option(
        Path("data/external/tennis-sackmann-archive/atp"),
        help="Directory containing ATP matches, rankings, and player CSV files.",
        file_okay=False,
        dir_okay=True,
        exists=True,
    ),
    artifact_path: Path = typer.Option(
        ...,
        help="Destination path for the fitted winner model artifact.",
        dir_okay=False,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an artifact already present at the destination.",
    ),
) -> None:
    """Build historical pre-match features and save the production winner model."""

    loader = ATPDataLoader(data_dir)
    table = build_historical_prediction_table(
        matches=loader.load_matches(),
        players=loader.load_players(),
        rankings=loader.load_rankings(),
    )
    result = fit_production_hist_gradient_boosting(table)
    artifact = create_winner_model_artifact(result)
    saved_path = save_winner_model_artifact(
        artifact,
        artifact_path,
        overwrite=overwrite,
    )
    latest_date = table.metadata.filter(
        table.metadata["tourney_date"].is_not_null()
    )["tourney_date"].max()
    typer.echo(
        f"Saved production winner artifact to {saved_path} "
        f"(training rows: {result.encoded.splits.train.metadata.height}, "
        f"latest tournament date: {latest_date})."
    )


@app.command("predict-winner")
def predict_winner(
    player_a: str = typer.Option(
        ..., "--player-a", help="Exact ATP player name or player ID."
    ),
    player_b: str = typer.Option(
        ..., "--player-b", help="Exact ATP player name or player ID."
    ),
    match_date: str = typer.Option(
        ..., "--date", help="Tournament date in YYYYMMDD format."
    ),
    artifact_path: Path = typer.Option(
        Path("models/winner.joblib"),
        "--artifact-path",
        help="Path to a winner artifact created by `tennis train-winner`.",
        exists=True,
        dir_okay=False,
    ),
    data_dir: Path = typer.Option(
        Path("data/external/tennis-sackmann-archive/atp"),
        "--data-dir",
        help="ATP archive used for the model's training data.",
        file_okay=False,
        dir_okay=True,
        exists=True,
    ),
    surface: str | None = typer.Option(
        None,
        "--surface",
        help="Optional tournament surface: Hard, Clay, Grass, or Carpet.",
    ),
) -> None:
    """Estimate both players' win probabilities for a future ATP match."""

    try:
        parsed_date = int(match_date)
        datetime.strptime(match_date, "%Y%m%d")
    except ValueError as error:
        raise typer.BadParameter(
            "Use a valid date in YYYYMMDD format", param_hint="--date"
        ) from error

    loader = ATPDataLoader(data_dir)
    matches = loader.load_matches()
    players = loader.load_players()
    rankings = loader.load_rankings()
    artifact = load_winner_model_artifact(artifact_path)
    cutoff_date = getattr(artifact, "training_cutoff_date", None)
    if cutoff_date is None:
        raise typer.BadParameter(
            "This artifact has no recorded training cutoff. Recreate it with `tennis train-winner`.",
            param_hint="--artifact-path",
        )
    if parsed_date <= cutoff_date:
        raise typer.BadParameter(
            f"Prediction date must be after the artifact's training cutoff ({cutoff_date}); "
            "otherwise later match results could leak into the prediction.",
            param_hint="--date",
        )

    def resolve_player(value: str) -> tuple[int, str]:
        if value.isdigit():
            player_id = int(value)
            found = players.filter(pl.col("player_id") == player_id)
        else:
            if not {"name_first", "name_last"} <= set(players.columns):
                raise typer.BadParameter(
                    "Player names are unavailable; provide a numeric ATP player ID"
                )
            full_name = pl.concat_str(
                [pl.col("name_first").fill_null(""), pl.col("name_last").fill_null("")],
                separator=" ",
            ).str.strip_chars()
            found = players.filter(full_name.str.to_lowercase() == value.strip().lower())
        if found.height == 0:
            raise typer.BadParameter(f"No ATP player matches {value!r}")
        if found.height > 1:
            ids = found["player_id"].to_list()
            raise typer.BadParameter(
                f"Player name {value!r} is ambiguous; use one of these IDs: {ids}"
            )
        player_id = int(found["player_id"][0])
        row = found.row(0, named=True)
        name = " ".join(
            str(row[column]).strip()
            for column in ("name_first", "name_last")
            if row.get(column) is not None
        )
        return player_id, name or str(player_id)

    player_a_id, player_a_name = resolve_player(player_a)
    player_b_id, player_b_name = resolve_player(player_b)
    if player_a_id == player_b_id:
        raise typer.BadParameter("Choose two different players")
    features = build_match_features(
        matches,
        players,
        rankings,
        player_a_id,
        player_b_id,
        parsed_date,
        surface=surface,
    )
    probabilities = predict_winner_probabilities(artifact, features)
    probability_a = probabilities["probability_player_a_wins"][0]
    probability_b = 1.0 - probability_a
    typer.echo(f"{player_a_name}: {probability_a:.1%}")
    typer.echo(f"{player_b_name}: {probability_b:.1%}")
    favorite = player_a_name if probability_a >= 0.5 else player_b_name
    typer.echo(f"Higher estimated probability: {favorite}")
    typer.echo(f"Prediction date: {parsed_date} | surface: {surface or 'unknown'}")
    typer.echo(f"Artifact training cutoff: {cutoff_date}")
    if surface is None:
        typer.echo(
            "Note: surface not supplied; surface-specific inputs use missing-value handling."
        )


@app.command("audit-markets")
def audit_markets(
    data_dir: Path = typer.Option(
        Path("data/external/tennis-sackmann-archive/atp"),
        help="Directory containing yearly ATP match CSV files.",
        file_okay=False,
        dir_okay=True,
        exists=True,
    ),
    by_year: bool = typer.Option(
        False,
        "--by-year",
        help="Also show source coverage by tournament year.",
    ),
) -> None:
    """Audit archive coverage for candidate match-statistics outcomes."""

    matches = ATPDataLoader(data_dir).load_matches()
    report = audit_market_coverage(matches)
    target_table = build_market_target_table(matches)
    target_audit = audit_market_targets(target_table)
    columns = [
        "market",
        "target_level",
        "unit",
        "matches_total",
        "matches_with_complete_sources",
        "matches_with_invalid_source_values",
        "source_coverage",
        "source_dtypes",
        "requires_parser",
        "status",
    ]
    typer.echo("Candidate market source coverage (parseable numeric inputs):")
    typer.echo(str(report.summary.select(columns)))
    typer.echo("Chronology and date coverage:")
    typer.echo(str(report.chronology))
    typer.echo("Parsed match score outcomes:")
    typer.echo(
        str(
            target_table.targets.group_by("score_status")
            .len()
            .sort("score_status")
        )
    )
    typer.echo("Post-match target ranges (observed values are retained):")
    typer.echo(str(target_audit.ranges))
    typer.echo("Internal-consistency violations (per applicable target pair):")
    typer.echo(str(target_audit.consistency))
    if by_year:
        typer.echo("Source coverage by tournament year:")
        typer.echo(str(report.by_year))


if __name__ == "__main__":
    app()
