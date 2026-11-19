from typer.testing import CliRunner

from tennis_prediction.cli.main import app


runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "Tennis Prediction Engine v0.1.0" in result.stdout

def test_forecast_context_orients_history_and_marks_missing_values(tmp_path, capsys) -> None:
    import polars as pl

    from tennis_prediction.cli.main import _echo_forecast_context

    features = pl.DataFrame({
        "player_a_rank": [None], "player_b_rank": [2],
        "player_a_win_rate_last_10": [None], "player_b_win_rate_last_10": [0.7],
        "player_a_surface_win_rate_last_10": [0.0],
        "player_b_surface_win_rate_last_10": [None],
        "head_to_head_matches": [3],
        "player_a_head_to_head_wins": [1], "player_b_head_to_head_wins": [2],
    })
    _echo_forecast_context(
        features, ("B", "A"), True, 20261201, 20260525, 20260608,
        tmp_path / "missing.joblib",
    )
    output = capsys.readouterr().out
    assert "Match winner unavailable" in output
    assert "B | ATP rank #2 | recent form 70.0% | surface form unavailable" in output
    assert "A | ATP rank unavailable | recent form unavailable | surface form 0.0%" in output
    assert "B 2–1 A (3 matches)" in output
    assert "matches 20260525 | rankings 20260608" in output


def test_forecast_has_public_command(tmp_path) -> None:
    artifact = tmp_path / "score.joblib"
    artifact.touch()
    result = runner.invoke(app, [
        "forecast", "--player-a", "A", "--player-b", "B", "--date", "bad",
        "--best-of", "3", "--artifact-path", str(artifact),
        "--winner-artifact", str(tmp_path / "winner.joblib"),
        "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 2
    assert "Use a valid date in YYYYMMDD format" in result.output
