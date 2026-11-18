import polars as pl

from tennis_prediction.datasets.market_targets import audit_market_targets
from tennis_prediction.datasets.market_targets import build_market_target_table
from tennis_prediction.datasets.market_targets import market_target_eligibility
from tennis_prediction.datasets.market_targets import split_market_target_table


def test_audit_second_serve_counts_comparable_rows_and_identifies_matches() -> None:
    matches = pl.DataFrame(
        {
            "tourney_date": [20240101, 20240102, 20240103, 20240104],
            "winner_id": [1, 1, 1, 1],
            "loser_id": [2, 2, 2, 2],
            "w_svpt": [20, 20, 20, 20],
            "w_1stIn": [12, 12, 12, 12],
            "w_2ndWon": [9, 7, 6, None],
            "w_df": [0, 2, 2, 2],
            "l_svpt": [20, 20, 20, 20],
            "l_1stIn": [12, 12, 12, 12],
            "l_2ndWon": [6, 6, 7, 6],
            "l_df": [2, 2, 2, 2],
        }
    )
    table = build_market_target_table(matches)
    audit = audit_market_targets(table)
    wins_check = "winner_second_serve_wins_exceed_second_serve_attempts"
    winner_check = "winner_second_serve_wins_and_double_faults_exceed_attempts"
    loser_check = "loser_second_serve_wins_and_double_faults_exceed_attempts"

    checks = audit.consistency.filter(pl.col("check").is_in([wins_check, winner_check, loser_check]))
    assert {name: (checked, violations) for name, checked, violations in checks.iter_rows()} == {
        wins_check: (3, 1),
        winner_check: (3, 2),
        loser_check: (4, 1),
    }
    assert set(audit.violations.select("check", "match_index").iter_rows()) == {
        (wins_check, 0),
        (winner_check, 0),
        (winner_check, 1),
        (loser_check, 2),
    }
    assert table.targets["winner_second_serve_points_won"].to_list() == [9, 7, 6, None]
    assert audit_market_targets(build_market_target_table(matches.slice(3))).violations.is_empty()


def test_eligibility_excludes_only_affected_markets_and_keeps_match_alignment() -> None:
    matches = pl.DataFrame(
        {
            "tourney_date": [20201231, 20210101, 20211231, 20220101, None],
            "winner_id": [1] * 5,
            "loser_id": [2] * 5,
            "best_of": [3] * 5,
            "score": ["6-2 6-2", "6-2 6-2", "6-2 6-2", "6-2 1-0 RET", "6-2 6-2"],
            "minutes": [60, 0, None, 30, 65],
            "w_ace": [0, 0, -1, 0, 0],
            "l_ace": [1] * 5,
            "w_svpt": [20] * 5,
            "l_svpt": [20] * 5,
            "w_1stIn": [12] * 5,
            "l_1stIn": [12] * 5,
            "w_1stWon": [10] * 5,
            "l_1stWon": [10] * 5,
            "w_2ndWon": [6, 7, 6, 6, 6],
            "l_2ndWon": [6, 6, None, 6, 6],
            "w_df": [2] * 5,
            "l_df": [2] * 5,
        }
    )
    table = build_market_target_table(matches)
    flags = market_target_eligibility(table)
    assert flags["match_index"].to_list() == [0, 1, 2, 3, 4]
    assert flags["return_points_won"].to_list() == [True, False, False, False, True]
    assert flags["double_faults"].to_list() == [True, False, True, False, True]
    assert flags["aces"].to_list() == [True, True, False, False, True]
    assert flags["total_games"].to_list() == [True, True, True, False, True]
    assert flags["match_duration"].to_list() == [True, False, False, False, True]
    assert table.targets["loser_return_points_won"].to_list() == [4, 3, 4, None, 4]

    splits = split_market_target_table(table, 20201231, 20211231)
    assert market_target_eligibility(splits.validation).equals(flags.slice(1, 2))
    assert market_target_eligibility(splits.undated).equals(flags.tail(1))


def test_match_tiebreak_is_eligible_for_sets_only_among_score_markets() -> None:
    matches = pl.DataFrame(
        {
            "tourney_date": [20240101],
            "winner_id": [1],
            "loser_id": [2],
            "best_of": [3],
            "score": ["6-4 4-6 [10-8]"],
        }
    )
    flags = market_target_eligibility(build_market_target_table(matches))
    assert flags["correct_set_score"].item()
    assert flags["sets_handicap"].item()
    assert not flags["games_handicap"].item()
    assert not flags["total_games"].item()
