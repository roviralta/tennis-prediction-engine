from typer.testing import CliRunner

from tennis_prediction.cli.main import app


runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "Tennis Prediction Engine v0.1.0" in result.stdout