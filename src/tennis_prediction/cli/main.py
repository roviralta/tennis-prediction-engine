from datetime import datetime
from pathlib import Path
from subprocess import TimeoutExpired

import polars as pl
import typer

from tennis_prediction.datasets.build import build_historical_prediction_table
from tennis_prediction.datasets.markets import audit_market_coverage
from tennis_prediction.datasets.market_targets import build_market_target_table
from tennis_prediction.datasets.market_targets import audit_market_targets
from tennis_prediction.datasets.score_markets import ScoreMarketSplits
from tennis_prediction.datasets.score_markets import build_score_market_splits
from tennis_prediction.ingestion.atp.loader import ATPDataLoader
from tennis_prediction.ingestion.atp.refresh import DEFAULT_SOURCE
from tennis_prediction.ingestion.atp.refresh import default_atp_data_dir
from tennis_prediction.ingestion.atp.refresh import refresh_atp_data
from tennis_prediction.models.statistics.game_pair_model import game_pair_lines
from tennis_prediction.models.statistics.game_pair_workflow import (
    evaluate_game_pair_holdout,
    fit_daily_game_pairs,
    game_pair_line,
    load_game_pair_artifact,
)
from tennis_prediction.models.statistics.score_model import _GAME_LINES
from tennis_prediction.models.statistics.stat_markets import (
    automatic_stat_thresholds,
    evaluate_stat_markets,
    fit_stat_markets,
    load_stat_artifact,
    reverse_player_features,
    select_stat_markets,
)
from tennis_prediction.models.statistics.workflow import TRAIN_END, VALIDATION_END
from tennis_prediction.models.statistics.workflow import evaluate_score_holdout
from tennis_prediction.models.statistics.workflow import fit_daily_score_models
from tennis_prediction.models.statistics.workflow import load_score_artifact
from tennis_prediction.models.statistics.workflow import score_line_probabilities
from tennis_prediction.models.winner.artifact import save_model_artifact
from tennis_prediction.models.winner import create_winner_model_artifact
from tennis_prediction.models.winner import fit_production_hist_gradient_boosting
from tennis_prediction.models.winner import build_match_features
from tennis_prediction.models.winner import load_winner_model_artifact
from tennis_prediction.models.winner import predict_winner_probabilities
from tennis_prediction.models.winner import save_winner_model_artifact

_ATP_DATA_DIR = default_atp_data_dir()
_SCORE_ARTIFACT_PATH = Path("models/score.joblib")
_WINNER_ARTIFACT_PATH = Path("models/winner.joblib")
_GAME_PAIR_ARTIFACT_PATH = Path("models/game-pairs.joblib")
_STAT_ARTIFACT_PATH = Path("models/statistics.joblib")

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


@app.command("refresh-data")
def refresh_data(
    source: str = typer.Option(DEFAULT_SOURCE, help="HTTPS Git archive or local archive directory."),
) -> None:
    """Validate and activate an ATP data snapshot."""
    try:
        result = refresh_atp_data(source)
    except (OSError, TimeoutExpired, ValueError, pl.exceptions.PolarsError) as error:
        raise typer.BadParameter(str(error), param_hint="--source") from error

    summary = result.summary
    action = "Activated ATP snapshot" if result.changed else "ATP archive already matches the source"
    typer.echo(f"{action}: {result.path}")
    typer.echo(f"Matches: {summary.matches:,} | latest tournament: {summary.latest_match}")
    typer.echo(f"Rankings: {summary.rankings:,} | latest ranking: {summary.latest_ranking}")
    typer.echo(f"Players: {summary.players:,}")


@app.command("train-winner")
def train_winner(
    data_dir: Path = typer.Option(
        _ATP_DATA_DIR,
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


def _resolve_player(players: pl.DataFrame, value: str) -> tuple[int, str]:
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
        _ATP_DATA_DIR,
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

    player_a_id, player_a_name = _resolve_player(players, player_a)
    player_b_id, player_b_name = _resolve_player(players, player_b)
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


def _load_score_splits(data_dir: Path) -> ScoreMarketSplits:
    loader = ATPDataLoader(data_dir)
    return build_score_market_splits(
        loader.load_matches(), loader.load_players(), loader.load_rankings(),
        TRAIN_END, VALIDATION_END,
    )


@app.command("validate-score")
def validate_score(
    data_dir: Path = typer.Option(_ATP_DATA_DIR, exists=True, file_okay=False),
) -> None:
    """Evaluate the fixed score and games models on the reserved 2022+ period."""
    splits = _load_score_splits(data_dir)
    report = evaluate_score_holdout(fit_daily_score_models(splits), splits.test)
    typer.echo(report.write_csv(), nl=False)


@app.command("train-score")
def train_score(
    artifact_path: Path = typer.Option(_SCORE_ARTIFACT_PATH, dir_okay=False),
    data_dir: Path = typer.Option(_ATP_DATA_DIR, exists=True, file_okay=False),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Fit and save the fixed local score and total-games models."""
    if artifact_path.exists() and not overwrite:
        raise typer.BadParameter("Artifact already exists. Use --overwrite to replace it.")
    artifact = fit_daily_score_models(_load_score_splits(data_dir))
    destination = save_model_artifact(artifact, artifact_path, overwrite=overwrite)
    typer.echo(f"Saved score and games artifact to {destination} (cutoff: {artifact.training_cutoff_date}).")


@app.command("validate-game-pairs")
def validate_game_pairs(
    data_dir: Path = typer.Option(_ATP_DATA_DIR, exists=True, file_okay=False),
) -> None:
    """Evaluate player games and handicap on reserved 2022+ outcomes."""
    splits = _load_score_splits(data_dir)
    report = evaluate_game_pair_holdout(fit_daily_game_pairs(splits), splits.test)
    typer.echo(report.write_csv(), nl=False)


@app.command("train-game-pairs")
def train_game_pairs(
    artifact_path: Path = typer.Option(_GAME_PAIR_ARTIFACT_PATH, dir_okay=False),
    data_dir: Path = typer.Option(_ATP_DATA_DIR, exists=True, file_okay=False),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Fit and save the fixed joint games model."""
    if artifact_path.exists() and not overwrite:
        raise typer.BadParameter("Artifact already exists. Use --overwrite to replace it.")
    artifact = fit_daily_game_pairs(_load_score_splits(data_dir))
    destination = save_model_artifact(artifact, artifact_path, overwrite=overwrite)
    typer.echo(f"Saved joint game models to {destination}.")


@app.command("train-stats")
def train_stats(
    artifact_path: Path = typer.Option(_STAT_ARTIFACT_PATH, dir_okay=False),
    data_dir: Path = typer.Option(_ATP_DATA_DIR, exists=True, file_okay=False),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Evaluate and save the statistical markets that pass the fixed criteria."""
    if artifact_path.exists() and not overwrite:
        raise typer.BadParameter("Artifact already exists. Use --overwrite to replace it.")
    splits = _load_score_splits(data_dir)
    candidates = fit_stat_markets(splits.train)
    report = evaluate_stat_markets(candidates, splits)
    typer.echo(report.write_csv(), nl=False)
    artifact = select_stat_markets(candidates, report)
    if not artifact.models:
        raise typer.BadParameter("No market and format met the validation criteria")
    destination = save_model_artifact(artifact, artifact_path, overwrite=overwrite)
    typer.echo(f"Saved validated statistical markets to {destination}.")


def _echo_forecast_context(
    features: pl.DataFrame,
    player_names: tuple[str, str],
    reversed_order: bool,
    match_date: int,
    match_history_date: int | None,
    ranking_history_date: int | None,
    winner_path: Path,
) -> None:
    """Print the observed pre-match context and the dedicated winner forecast."""

    if winner_path.is_file():
        winner = load_winner_model_artifact(winner_path)
        cutoff = getattr(winner, "training_cutoff_date", None)
        if cutoff is None or match_date <= cutoff:
            typer.echo(f"Match winner unavailable for this date (model cutoff: {cutoff or 'unknown'}).")
        else:
            # Use one canonical forecast so reversing the requested order swaps probabilities.
            probability = predict_winner_probabilities(
                winner, features.select(winner.feature_columns)
            )["probability_player_a_wins"].item()
            if reversed_order:
                probability = 1 - probability
            typer.echo(
                f"Match winner (winner model) | {player_names[0]} {probability:.2%} | "
                f"{player_names[1]} {1 - probability:.2%} | fit through {cutoff}"
            )
    else:
        typer.echo("Match winner unavailable. Run tennis train-winner --artifact-path models/winner.joblib.")

    context = reverse_player_features(features) if reversed_order else features
    observed = context.row(0, named=True)

    def rate(column: str) -> str:
        value = observed[column]
        return f"{value:.1%}" if value is not None else "unavailable"

    typer.echo("Pre-match context:")
    for player, name in zip(("a", "b"), player_names, strict=True):
        rank = observed[f"player_{player}_rank"]
        rank_label = f"#{int(rank)}" if rank is not None else "unavailable"
        typer.echo(
            f"  {name} | ATP rank {rank_label} | "
            f"recent form {rate(f'player_{player}_win_rate_last_10')} | "
            f"surface form {rate(f'player_{player}_surface_win_rate_last_10')}"
        )
    meetings = observed["head_to_head_matches"]
    if meetings:
        typer.echo(
            f"  Head-to-head | {player_names[0]} {observed['player_a_head_to_head_wins']}–"
            f"{observed['player_b_head_to_head_wins']} {player_names[1]} ({meetings} matches)"
        )
    else:
        typer.echo("  Head-to-head | no eligible prior meetings")
    typer.echo(
        f"  Latest eligible archive dates | matches {match_history_date or 'unavailable'} | "
        f"rankings {ranking_history_date or 'unavailable'}"
    )
    typer.echo("  Form is the win rate over up to 10 prior matches.")


@app.command("forecast", help="Show a complete pre-match ATP forecast and historical context.")
@app.command("predict-score")
def predict_score(
    ctx: typer.Context,
    player_a: str = typer.Option(..., "--player-a", help="Exact ATP name or player ID."),
    player_b: str = typer.Option(..., "--player-b", help="Exact ATP name or player ID."),
    match_date: str = typer.Option(..., "--date", help="Tournament date in YYYYMMDD format."),
    best_of: int = typer.Option(..., "--best-of", help="Match format: 3 or 5."),
    surface: str | None = typer.Option(None, help="Tournament surface."),
    draw_size: int | None = typer.Option(None, min=1, help="Tournament draw size."),
    tourney_level: str | None = typer.Option(None, help="ATP archive tournament level code."),
    round_code: str | None = typer.Option(None, "--round", help="ATP round code, such as QF."),
    line: list[float] | None = typer.Option(None, "--line", help="Repeat to add game lines."),
    game_pairs: bool = typer.Option(True, "--game-pairs/--no-game-pairs", help="Show player games and handicap when available."),
    player_line: list[float] | None = typer.Option(None, "--player-line", help="Line for each player."),
    handicap: list[float] | None = typer.Option(None, "--handicap", help="Games added to player A."),
    game_pair_artifact: Path = typer.Option(_GAME_PAIR_ARTIFACT_PATH, dir_okay=False),
    stats: bool = typer.Option(True, "--stats/--no-stats", help="Show available aces, double faults and breaks."),
    aces_line: list[float] | None = typer.Option(None, "--aces-line"),
    double_faults_line: list[float] | None = typer.Option(None, "--double-faults-line"),
    breaks_line: list[float] | None = typer.Option(None, "--breaks-line"),
    stat_artifact: Path = typer.Option(_STAT_ARTIFACT_PATH, dir_okay=False),
    artifact_path: Path = typer.Option(_SCORE_ARTIFACT_PATH, exists=True, dir_okay=False),
    winner_artifact: Path = typer.Option(_WINNER_ARTIFACT_PATH, dir_okay=False),
    data_dir: Path = typer.Option(_ATP_DATA_DIR, exists=True, file_okay=False),
) -> None:
    """Predict available score and statistical markets for an ATP match."""
    try:
        if len(match_date) != 8 or not match_date.isascii() or not match_date.isdigit():
            raise ValueError
        parsed_date = int(datetime.strptime(match_date, "%Y%m%d").strftime("%Y%m%d"))
    except ValueError as error:
        raise typer.BadParameter("Use a valid date in YYYYMMDD format", param_hint="--date") from error
    if best_of not in (3, 5):
        raise typer.BadParameter("Choose 3 or 5", param_hint="--best-of")

    artifact = load_score_artifact(artifact_path)
    if parsed_date <= artifact.training_cutoff_date:
        raise typer.BadParameter(
            f"Prediction date must be after the artifact cutoff ({artifact.training_cutoff_date})",
            param_hint="--date",
        )
    loader = ATPDataLoader(data_dir)
    players = loader.load_players()
    a_id, a_name = _resolve_player(players, player_a)
    b_id, b_name = _resolve_player(players, player_b)
    if a_id == b_id:
        raise typer.BadParameter("Choose two different players")
    canonical_a, canonical_b = sorted((a_id, b_id))
    if surface is not None:
        surface = surface.strip().title()
        if surface not in {"Hard", "Clay", "Grass", "Carpet"}:
            raise typer.BadParameter("Choose Hard, Clay, Grass, or Carpet", param_hint="--surface")
    history = loader.load_matches().filter(pl.col("winner_id") != pl.col("loser_id"))
    rankings = loader.load_rankings()
    features = build_match_features(
        history, players, rankings, canonical_a, canonical_b,
        parsed_date, surface=surface,
    ).with_columns(
        pl.lit(best_of, dtype=pl.Int64).alias("best_of"),
        pl.lit(draw_size, dtype=pl.Int64).alias("draw_size"),
        pl.lit(tourney_level, dtype=pl.String).alias("tourney_level"),
        pl.lit(round_code, dtype=pl.String).alias("round"),
    )
    model = artifact.formats[best_of]
    scores, games = model.forecast(features)
    lines = line or list(_GAME_LINES[best_of])
    try:
        line_probabilities = [
            (value, *score_line_probabilities(model.counts, games[0], value))
            for value in sorted(set(lines))
        ]
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--line") from error

    joint_outputs = []
    if (player_line or handicap) and not game_pair_artifact.is_file():
        raise typer.BadParameter("Create the joint model with tennis train-game-pairs")
    if (game_pairs or player_line or handicap) and game_pair_artifact.is_file():
        joint_artifact = load_game_pair_artifact(game_pair_artifact)
        if parsed_date <= joint_artifact.training_cutoff_date:
            raise typer.BadParameter("Date must be after the joint model cutoff", param_hint="--date")
        joint = joint_artifact.formats.get(best_of)
        if joint is None:
            raise typer.BadParameter("Joint forecasts are unavailable for this format")
        probabilities = joint.predict(features)[0]
        a_games, b_games = joint.model.pairs.T
        if a_id != canonical_a:
            a_games, b_games = b_games, a_games
        player_lines = player_line or game_pair_lines(best_of, "player_a_games")
        handicaps = handicap or [-value for value in game_pair_lines(best_of, "game_margin")]
        if any(value < 0 for value in player_lines):
            raise typer.BadParameter("Player-game lines must be nonnegative", param_hint="--player-line")
        try:
            for label, values in ((a_name, a_games), (b_name, b_games)):
                for value in sorted(set(player_lines)):
                    under, push, over = game_pair_line(values, probabilities, value)
                    joint_outputs.append(
                        f"  {label} {value:g} | under {under:.2%} | push {push:.2%} | over {over:.2%}"
                    )
            for value in sorted(set(handicaps)):
                loss, push, cover = game_pair_line(a_games - b_games, probabilities, -value)
                joint_outputs.append(
                    f"  {a_name} {value:+g} | loss {loss:.2%} | push {push:.2%} | cover {cover:.2%}"
                )
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--player-line / --handicap") from error

    stat_outputs = []
    stat_lines_requested = aces_line or double_faults_line or breaks_line
    if stat_lines_requested and not stat_artifact.is_file():
        raise typer.BadParameter("Create the statistics model with tennis train-stats")
    if (stats or stat_lines_requested) and stat_artifact.is_file():
        statistics = load_stat_artifact(stat_artifact)
        if parsed_date <= statistics.training_cutoff_date:
            raise typer.BadParameter("Date must be after the statistics model cutoff", param_hint="--date")
        requested = {
            "aces": aces_line, "double_faults": double_faults_line, "breaks": breaks_line,
        }
        ordered = [(a_name, features), (b_name, reverse_player_features(features))]
        if a_id != canonical_a:
            ordered = [(a_name, ordered[1][1]), (b_name, ordered[0][1])]
        try:
            for market, requested_lines in requested.items():
                if not stats and not requested_lines:
                    continue
                label = market.replace("_", " ").capitalize()
                distribution = statistics.models.get((market, best_of))
                if distribution is None:
                    stat_outputs.append(f"  {label}: unavailable for best of {best_of}")
                    continue
                stat_outputs.append(f"  {label}:")
                for subject, inputs in ordered:
                    cdf = distribution.forecast(inputs)[0]
                    if not requested_lines:
                        thresholds = automatic_stat_thresholds(distribution.counts, cdf)
                        entries = []
                        for threshold in thresholds:
                            _, _, probability = score_line_probabilities(
                                distribution.counts, cdf, threshold - 0.5
                            )
                            entries.append(f"{threshold}+: {probability:.2%}")
                        stat_outputs.append(f"    {subject} | " + " | ".join(entries))
                        continue
                    for value in sorted(set(requested_lines)):
                        under, push, over = score_line_probabilities(distribution.counts, cdf, value)
                        stat_outputs.append(
                            f"    {subject} {value:g} | under {under:.2%} | "
                            f"push {push:.2%} | over {over:.2%}"
                        )
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="statistics lines") from error

    typer.echo(f"Match: {a_name} vs {b_name} | best of {best_of} | {parsed_date}")
    if ctx.info_name == "forecast":
        typer.echo(f"Surface: {surface or 'unavailable'}")
        latest_match = history.filter(pl.col("tourney_date") < parsed_date)["tourney_date"].max()
        ranking_dates = rankings["ranking_date"].cast(pl.Int64, strict=False)
        latest_ranking = ranking_dates.filter(ranking_dates <= parsed_date).max()
        _echo_forecast_context(
            features, (a_name, b_name), a_id != canonical_a, parsed_date,
            latest_match, latest_ranking, winner_artifact,
        )
    typer.echo(f"Final set score ({a_name} first):")
    score_probabilities = zip(model.classifier.classes, scores[0], strict=True)
    for label, probability in sorted(score_probabilities, key=lambda item: -item[1]):
        shown = label if a_id == canonical_a else "-".join(reversed(label.split("-")))
        typer.echo(f"  {shown}: {probability:.2%}")
    typer.echo("Total games (standard completed matches):")
    for value, under, push, over in line_probabilities:
        typer.echo(f"  {value:g} | under {under:.2%} | push {push:.2%} | over {over:.2%}")
    if joint_outputs:
        typer.echo("Player games and handicap (joint model, standard completed matches):")
        for output in joint_outputs:
            typer.echo(output)
    if stat_outputs:
        typer.echo("Match statistics (completed matches):")
        for output in stat_outputs:
            typer.echo(output)
        typer.echo(f"Statistics fitting cutoff: {statistics.training_cutoff_date}")
    if game_pairs and not game_pair_artifact.is_file():
        typer.echo("Player games and handicap unavailable. Run tennis train-game-pairs.")
    if stats and not stat_artifact.is_file():
        typer.echo("Match statistics unavailable. Run tennis train-stats.")
    typer.echo(f"Model fitting and calibration cutoff: {artifact.training_cutoff_date}")


@app.command("audit-markets")
def audit_markets(
    data_dir: Path = typer.Option(
        _ATP_DATA_DIR,
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
    if target_audit.violations.height:
        typer.echo("Matches with consistency violations:")
        typer.echo(str(target_audit.violations))
    if by_year:
        typer.echo("Source coverage by tournament year:")
        typer.echo(str(report.by_year))


if __name__ == "__main__":
    app()
