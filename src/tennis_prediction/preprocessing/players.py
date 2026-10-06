import polars as pl


PLAYER_COLUMNS = [
    "player_id",
    "hand",
    "height",
]


def enrich_player_information(
    matches: pl.DataFrame,
    players: pl.DataFrame,
) -> pl.DataFrame:
    """Add basic player information to normalized matches."""

    missing_match_columns = [
        column
        for column in ["player_a", "player_b"]
        if column not in matches.columns
    ]

    if missing_match_columns:
        raise ValueError(
            f"Missing match columns: {missing_match_columns}"
        )

    missing_player_columns = [
        column
        for column in PLAYER_COLUMNS
        if column not in players.columns
    ]

    if missing_player_columns:
        raise ValueError(
            f"Missing player columns: {missing_player_columns}"
        )

    player_a = players.select(
    [
        pl.col("player_id").alias("player_a"),
        pl.col("hand").alias("player_a_hand"),
        pl.col("height").alias("player_a_height"),
        pl.col("dob").alias("player_a_dob"),
    ]
)

    player_b = players.select(
    [
        pl.col("player_id").alias("player_b"),
        pl.col("hand").alias("player_b_hand"),
        pl.col("height").alias("player_b_height"),
        pl.col("dob").alias("player_b_dob"),
    ]
)

    enriched = (
    matches
    .join(player_a, on="player_a", how="left")
    .join(player_b, on="player_b", how="left")
    .with_columns(
        calculate_age(
            pl.col("player_a_dob"),
            pl.col("tourney_date"),
        ).alias("player_a_age"),
        calculate_age(
            pl.col("player_b_dob"),
            pl.col("tourney_date"),
        ).alias("player_b_age"),
    )
)

    return enriched

def calculate_age(
    birth_date: pl.Expr,
    match_date: pl.Expr,
) -> pl.Expr:
    """Calculate a player's age in years at the time of a match."""

    birth = (
        birth_date
        .cast(pl.String)
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
    )

    match = (
        match_date
        .cast(pl.String)
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
    )

    return (
        (
            match.dt.year() - birth.dt.year()
        )
        - (
            (
                match.dt.month() < birth.dt.month()
            )
            | (
                (match.dt.month() == birth.dt.month())
                & (match.dt.day() < birth.dt.day())
            )
        ).cast(pl.Int32)
    )