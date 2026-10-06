import polars as pl


def prepare_rankings(rankings: pl.DataFrame) -> pl.DataFrame:
    """Prepare ranking data for temporal joins."""

    required_columns = [
        "ranking_date",
        "rank",
        "player",
        "points",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in rankings.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing ranking columns: {missing_columns}"
        )

    return rankings.with_columns(
        pl.col("ranking_date")
        .cast(pl.String)
        .str.strptime(
            pl.Date,
            "%Y%m%d",
            strict=False,
        )
        .alias("ranking_date")
    )
    
def get_latest_ranking_before_match(
    rankings: pl.DataFrame,
    player_id: int,
    match_date: pl.date,
) -> pl.DataFrame:
    """Get the latest ranking available for a player before a match."""

    return (
        rankings
        .filter(
            (pl.col("player") == player_id)
            & (pl.col("ranking_date") <= match_date)
        )
        .sort("ranking_date", descending=True)
        .head(1)
    )
    
def add_player_ranking(
    matches: pl.DataFrame,
    rankings: pl.DataFrame,
    player_column: str,
    prefix: str,
) -> pl.DataFrame:
    """Add the latest known ranking to each match row."""

    ranking_data = (
        rankings
        .select(
            [
                pl.col("player").alias(player_column),
                pl.col("ranking_date"),
                pl.col("rank").alias(f"{prefix}_rank"),
                pl.col("points").alias(f"{prefix}_rank_points"),
            ]
        )
        .sort("ranking_date")
    )

    match_data = matches.with_columns(
        pl.col("tourney_date")
        .cast(pl.String)
        .str.strptime(
            pl.Date,
            "%Y%m%d",
            strict=False,
        )
        .alias("_match_date")
    )

    return match_data.join_asof(
        ranking_data,
        left_on="_match_date",
        right_on="ranking_date",
        by=player_column,
        strategy="backward",
        check_sortedness=False,
    ).drop("_match_date")


def add_match_rankings(
    matches: pl.DataFrame,
    rankings: pl.DataFrame,
) -> pl.DataFrame:
    """Add pre-match rankings for both players."""

    result = add_player_ranking(
        matches,
        rankings,
        player_column="player_a",
        prefix="player_a",
    )

    result = add_player_ranking(
        result,
        rankings,
        player_column="player_b",
        prefix="player_b",
    )

    return result


def add_ranking_differences(
    matches: pl.DataFrame,
) -> pl.DataFrame:
    """Add relative ranking features between both players."""

    return matches.with_columns(
        (
            pl.col("player_a_rank")
            - pl.col("player_b_rank")
        ).alias("rank_difference"),
        (
            pl.col("player_a_rank_points")
            - pl.col("player_b_rank_points")
        ).alias("rank_points_difference"),
    )