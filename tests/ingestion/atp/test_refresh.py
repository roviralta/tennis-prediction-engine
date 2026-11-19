from pathlib import Path

import polars as pl
import pytest

from tennis_prediction.ingestion.atp.refresh import refresh_atp_data


def _archive(path: Path, latest: int) -> Path:
    path.mkdir()
    pl.DataFrame({
        "tourney_date": [latest], "tourney_id": ["event"], "tourney_name": ["Test"],
        "draw_size": [32], "tourney_level": ["A"], "match_num": [1], "best_of": [3],
        "round": ["F"], "winner_id": [1], "loser_id": [2], "surface": ["Hard"],
        "score": ["6-4 6-4"],
    }).write_csv(path / "atp_matches_2026.csv")
    pl.DataFrame({
        "ranking_date": [latest], "rank": [1], "player": [1], "points": [1000],
    }).write_csv(path / "atp_rankings_current.csv")
    pl.DataFrame({
        "player_id": [1], "name_first": ["Test"], "name_last": ["Player"],
        "dob": [19900101], "hand": ["R"], "height": [180],
    }).write_csv(path / "atp_players.csv")
    return path


def test_refresh_activates_valid_snapshot_and_keeps_original(tmp_path) -> None:
    old = _archive(tmp_path / "old", 20260101)
    new = _archive(tmp_path / "new", 20260108)
    (new / "atp_matches_qual_chall_2026.csv").write_text("unrelated\n1\n")
    root = tmp_path / "managed"

    result = refresh_atp_data(str(new), snapshot_root=root, fallback=old)
    assert result.changed
    assert result.path.is_symlink()
    assert result.summary.latest_match == 20260108
    assert result.summary.latest_ranking == 20260108
    assert not (result.path / "atp_matches_qual_chall_2026.csv").exists()
    assert "20260101" in (old / "atp_matches_2026.csv").read_text()
    assert not refresh_atp_data(str(new), snapshot_root=root, fallback=old).changed

    previous = result.path.resolve()
    with pytest.raises(ValueError, match="older than"):
        refresh_atp_data(str(old), snapshot_root=root, fallback=old)
    assert result.path.resolve() == previous

    invalid = _archive(tmp_path / "invalid", 20260115)
    (invalid / "atp_rankings_current.csv").write_text("wrong\n1\n")
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        refresh_atp_data(str(invalid), snapshot_root=root, fallback=old)
    assert result.path.resolve() == previous


def test_refresh_bootstraps_an_empty_installation(tmp_path) -> None:
    source = _archive(tmp_path / "source", 20260101)
    result = refresh_atp_data(
        str(source), snapshot_root=tmp_path / "managed", fallback=tmp_path / "missing"
    )
    assert result.summary.matches == 1
    assert result.changed
