# Tennis Prediction Engine

An ATP tennis prediction and analytics engine built with Python and machine learning.

## Goals

The system aims to:

- Predict ATP match winners.
- Produce calibrated win probabilities.
- Predict match statistics.
- Analyze player form and performance.
- Model surface-specific performance.
- Incorporate Elo and player-strength metrics.
- Explain predictions using SHAP.
- Backtest predictions without data leakage.
- Simulate complete tennis tournaments using Monte Carlo methods.

## Project Status

🚧 Early development — historical winner modeling, production training, and future-match
winner probabilities are available through the CLI. Statistical market models are next.

## Technology

- Python
- Pandas
- Polars
- NumPy
- DuckDB
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SciPy
- Statsmodels
- SHAP
- MLflow
- Typer
- Jupyter
- Pytest

## Development

Install dependencies:

```bash
uv sync
```

Train and save the production winner model from the local ATP archive:

```bash
uv run tennis train-winner --artifact-path models/winner.joblib
```

The command uses all dated, labeled matches in the archive and reports the training row count and
latest tournament date. It does not calculate evaluation metrics on those training rows. To replace
an existing artifact, add `--overwrite`. Pass `--data-dir PATH` to use a different archive location.

Predict a future match after creating the artifact:

```bash
uv run tennis predict-winner \
  --player-a "Carlos Alcaraz" \
  --player-b "Jannik Sinner" \
  --date 20261201 \
  --surface Hard \
  --artifact-path models/winner.joblib
```

Names must match the ATP player file (case-insensitive); numeric player IDs can be used when names
are ambiguous. `--surface` is optional. The command prints both probabilities and the player with
the higher estimated probability. The requested date must be later than the artifact's recorded
training cutoff, and the artifact and `--data-dir` should come from the same local archive. The
archive currently ends on 2026-05-25, so forecasts after that date do not include later results
until the deferred data-refresh step is implemented.

Inspect candidate statistics-market coverage:

```bash
uv run tennis audit-markets --by-year
```

This command also reports observed target ranges and counts of internal
match-stat inconsistencies. The audit is diagnostic: it preserves every row
and does not silently repair or discard questionable values. Market outcomes
can be partitioned with explicit chronological cutoffs through
`split_market_target_table`; rows without a tournament date remain separate.

The audit reports parseable source coverage overall and by tournament year, classifies score lines,
and displays target range and consistency checks. Only completed, structurally valid scores produce
set and game targets. Retirements and ambiguous scores remain visible in the status counts but do
not become ordinary completed-match outcomes. Matches decided by a bracketed match tiebreak retain
set targets while game totals remain missing.
