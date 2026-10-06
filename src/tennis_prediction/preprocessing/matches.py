import polars as pl


MATCH_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "best_of",
    "round",
]


def normalize_matches(matches: pl.DataFrame) -> pl.DataFrame:
    """Convert winner/loser data into two neutral player perspectives."""

    required_columns = MATCH_COLUMNS + ["winner_id", "loser_id"]

    missing_columns = [
        column
        for column in required_columns
        if column not in matches.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    base = (
        matches
        .with_row_index("match_index")
        .select(
            ["match_index"]
            + MATCH_COLUMNS
            + [
                pl.col("winner_id").alias("player_a"),
                pl.col("loser_id").alias("player_b"),
            ]
        )
        .with_columns(
            pl.lit(1).alias("target")
        )
    )

    reversed_matches = (
        matches
        .with_row_index("match_index")
        .select(
            ["match_index"]
            + MATCH_COLUMNS
            + [
                pl.col("loser_id").alias("player_a"),
                pl.col("winner_id").alias("player_b"),
            ]
        )
        .with_columns(
            pl.lit(0).alias("target")
        )
    )

    return (
        pl.concat(
            [base, reversed_matches],
            how="vertical",
        )
        .sort("match_index")
    )