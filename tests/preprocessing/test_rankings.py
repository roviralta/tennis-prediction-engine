from datetime import date
from pathlib import Path
import polars as pl

from tennis_prediction.preprocessing.rankings import (
    add_player_ranking,
    add_match_rankings,
    add_ranking_differences,
    get_latest_ranking_before_match,
    prepare_rankings,
)

from tennis_prediction.ingestion.atp.loader import ATPDataLoader
from tennis_prediction.preprocessing.matches import normalize_matches

def test_prepare_rankings_converts_ranking_date() -> None:
    rankings = pl.DataFrame(
        {
            "ranking_date": [20250303, 20250310],
            "rank": [1, 2],
            "player": [100001, 100002],
            "points": [10000, 9000],
        }
    )

    result = prepare_rankings(rankings)

    assert result["ranking_date"].dtype == pl.Date
    assert result["ranking_date"].to_list() == [
        date(2025, 3, 3),
        date(2025, 3, 10),
    ]
    
def test_get_latest_ranking_before_match() -> None:
    from datetime import date

    rankings = pl.DataFrame(
        {
            "ranking_date": [
                20250303,
                20250310,
                20250317,
            ],
            "rank": [5, 7, 6],
            "player": [100001, 100001, 100001],
            "points": [8000, 7500, 7800],
        }
    )

    rankings = prepare_rankings(rankings)

    result = get_latest_ranking_before_match(
        rankings,
        player_id=100001,
        match_date=date(2025, 3, 12),
    )

    assert result["rank"].to_list() == [7]
    assert result["points"].to_list() == [7500]
    

def test_add_player_ranking_uses_latest_ranking_before_match() -> None:
    matches = pl.DataFrame(
        {
            "player_a": [100001],
            "tourney_date": [20250312],
        }
    )

    rankings = pl.DataFrame(
        {
            "ranking_date": [20250303, 20250310, 20250317],
            "rank": [5, 7, 6],
            "player": [100001, 100001, 100001],
            "points": [8000, 7500, 7800],
        }
    )

    rankings = prepare_rankings(rankings)

    result = add_player_ranking(
        matches,
        rankings,
        player_column="player_a",
        prefix="player_a",
    )

    assert result["player_a_rank"].to_list() == [7]
    assert result["player_a_rank_points"].to_list() == [7500]
    

def test_add_match_rankings() -> None:
    matches = pl.DataFrame(
        {
            "player_a": [100001],
            "player_b": [100002],
            "tourney_date": [20250312],
        }
    )

    rankings = pl.DataFrame(
        {
            "ranking_date": [
                20250303,
                20250310,
                20250310,
            ],
            "rank": [5, 7, 12],
            "player": [100001, 100001, 100002],
            "points": [8000, 7500, 6000],
        }
    )

    rankings = prepare_rankings(rankings)

    result = add_match_rankings(
        matches,
        rankings,
    )

    assert result["player_a_rank"].to_list() == [7]
    assert result["player_a_rank_points"].to_list() == [7500]

    assert result["player_b_rank"].to_list() == [12]
    assert result["player_b_rank_points"].to_list() == [6000]
    

def test_real_matches_have_rankings() -> None:

    project_root = Path(__file__).resolve().parents[2]

    loader = ATPDataLoader(
        project_root / "data/external/tennis-sackmann-archive/atp"
    )

    matches = loader.load_matches(start_year=2025, end_year=2025)
    rankings = loader.load_rankings(end_year=2025)
    normalized = normalize_matches(matches)
    rankings = prepare_rankings(rankings)

    enriched = add_match_rankings(
        normalized,
        rankings,
    )

    assert enriched.height == 5888


def test_add_ranking_differences() -> None:
    matches = pl.DataFrame(
        {
            "player_a_rank": [5, 23],
            "player_b_rank": [23, 5],
            "player_a_rank_points": [8200, 3100],
            "player_b_rank_points": [3100, 8200],
        }
    )

    result = add_ranking_differences(matches)

    assert result["rank_difference"].to_list() == [-18, 18]
    assert result["rank_points_difference"].to_list() == [5100, -5100]