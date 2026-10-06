from pathlib import Path

import polars as pl

from tennis_prediction.ingestion.atp.loader import ATPDataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ATP_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "tennis-sackmann-archive"
    / "atp"
)


def test_load_matches_for_single_year() -> None:
    loader = ATPDataLoader(ATP_DATA_PATH)

    matches = loader.load_matches(2025, 2025)

    assert matches.height > 0
    assert matches.width >= 49
    assert "winner_id" in matches.columns
    assert "loser_id" in matches.columns


def test_load_rankings_for_single_year() -> None:
    loader = ATPDataLoader(ATP_DATA_PATH)

    rankings = loader.load_rankings(2025, 2025)

    assert rankings.height > 0
    assert rankings.width == 4
    assert rankings.columns == [
        "ranking_date",
        "rank",
        "player",
        "points",
    ]
    assert rankings["points"].dtype == pl.Int64


def test_load_players() -> None:
    loader = ATPDataLoader(ATP_DATA_PATH)

    players = loader.load_players()

    assert players.height > 0
    assert players.width == 8
    assert players.columns == [
        "player_id",
        "name_first",
        "name_last",
        "hand",
        "dob",
        "ioc",
        "height",
        "wikidata_id",
    ]
    assert players["player_id"].dtype == pl.Int64
    
def test_match_players_exist_in_player_dataset() -> None:
    loader = ATPDataLoader(ATP_DATA_PATH)

    matches = loader.load_matches(2025, 2025)
    players = loader.load_players()

    player_ids = set(players["player_id"].to_list())

    match_player_ids = set(
        matches["winner_id"].drop_nulls().to_list()
        + matches["loser_id"].drop_nulls().to_list()
    )

    missing_player_ids = match_player_ids - player_ids

    assert not missing_player_ids, (
        f"Found {len(missing_player_ids)} match player IDs "
        f"missing from the player dataset: {sorted(missing_player_ids)[:10]}"
    )