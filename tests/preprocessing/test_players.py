from pathlib import Path

import polars as pl

from tennis_prediction.ingestion.atp.loader import ATPDataLoader
from tennis_prediction.preprocessing.matches import normalize_matches
from tennis_prediction.preprocessing.players import enrich_player_information
from tennis_prediction.preprocessing.players import (
    calculate_age,
    enrich_player_information,
)
from tennis_prediction.ingestion.atp.loader import ATPDataLoader
from tennis_prediction.preprocessing.matches import normalize_matches


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ATP_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "tennis-sackmann-archive"
    / "atp"
)


def test_enrich_player_information() -> None:
    matches = pl.DataFrame(
        {
            "player_a": [100001],
            "player_b": [100002],
            "tourney_date": [20250310],
        }
    )

    players = pl.DataFrame(
        {
            "player_id": [100001, 100002],
            "hand": ["R", "L"],
            "height": [185, 168],
            "dob": [19900101, 19950101],
        }
    )

    enriched = enrich_player_information(matches, players)

    assert enriched["player_a_hand"].to_list() == ["R"]
    assert enriched["player_b_hand"].to_list() == ["L"]

    assert enriched["player_a_height"].to_list() == [185]
    assert enriched["player_b_height"].to_list() == [168]
    
    assert enriched["player_a_age"].to_list() == [35]
    assert enriched["player_b_age"].to_list() == [30]


def test_all_match_players_have_player_information() -> None:
    loader = ATPDataLoader(ATP_DATA_PATH)

    matches = loader.load_matches(2025, 2025)
    players = loader.load_players()

    normalized = normalize_matches(matches)

    enriched = enrich_player_information(
        normalized,
        players,
    )

    assert enriched["player_a_hand"].null_count() == 0
    assert enriched["player_b_hand"].null_count() == 0
    
def test_calculate_age() -> None:
    matches = pl.DataFrame(
        {
            "dob": [19970522, 19900101, None],
            "match_date": [20250310, 20250310, 20250310],
        }
    )

    result = matches.select(
        calculate_age(
            pl.col("dob"),
            pl.col("match_date"),
        ).alias("age")
    )

    assert result["age"].to_list() == [27, 35, None]
    

def test_real_matches_have_player_information() -> None:

    project_root = Path(__file__).resolve().parents[2]

    loader = ATPDataLoader(
        project_root / "data/external/tennis-sackmann-archive/atp"
    )

    matches = loader.load_matches(start_year=2025, end_year=2025)
    players = loader.load_players()

    normalized = normalize_matches(matches)

    enriched = enrich_player_information(
        normalized,
        players,
    )

    assert enriched.height == 5888

    assert enriched["player_a_hand"].null_count() == 0
    assert enriched["player_b_hand"].null_count() == 0
    
def test_real_matches_have_valid_ages() -> None:
    
    project_root = Path(__file__).resolve().parents[2]

    loader = ATPDataLoader(
        project_root / "data/external/tennis-sackmann-archive/atp"
    )

    matches = loader.load_matches(start_year=2025, end_year=2025)
    players = loader.load_players()

    normalized = normalize_matches(matches)

    enriched = enrich_player_information(
        normalized,
        players,
    )

    ages = enriched.select(
        [
            "player_a_age",
            "player_b_age",
        ]
    )

    print("\nAge statistics:")
    print(ages.describe())

    assert ages.filter(
        (pl.col("player_a_age") < 14)
        | (pl.col("player_a_age") > 50)
    ).height == 0

    assert ages.filter(
        (pl.col("player_b_age") < 14)
        | (pl.col("player_b_age") > 50)
    ).height == 0