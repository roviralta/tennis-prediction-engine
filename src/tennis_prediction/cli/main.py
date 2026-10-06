import typer

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


if __name__ == "__main__":
    app()