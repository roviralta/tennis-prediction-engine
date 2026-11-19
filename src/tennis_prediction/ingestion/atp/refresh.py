from dataclasses import dataclass
from filecmp import cmp
from fcntl import LOCK_EX, flock
from pathlib import Path
from shutil import copy2
from subprocess import run
from tempfile import TemporaryDirectory
from uuid import uuid4

import polars as pl

from tennis_prediction.ingestion.atp.loader import ATPDataLoader

LEGACY_ATP_DIR = Path("data/external/tennis-sackmann-archive/atp")
SNAPSHOT_ROOT = Path("data/external/atp-snapshots")
DEFAULT_SOURCE = "https://github.com/Aneeshers/tennis-sackmann-archive.git"


@dataclass(frozen=True)
class ArchiveSummary:
    matches: int
    latest_match: int
    rankings: int
    latest_ranking: int
    players: int


@dataclass(frozen=True)
class RefreshResult:
    path: Path
    summary: ArchiveSummary
    changed: bool


def default_atp_data_dir() -> Path:
    active = SNAPSHOT_ROOT / "current"
    return active.resolve() if active.is_dir() else LEGACY_ATP_DIR


def _archive_files(path: Path) -> dict[str, Path]:
    return {
        file.name: file
        for pattern in ("atp_matches_[0-9][0-9][0-9][0-9].csv", "atp_rankings_*.csv", "atp_players.csv")
        for file in path.glob(pattern)
        if file.is_file() and not file.is_symlink()
    }


def _inventory(path: Path) -> ArchiveSummary:
    loader = ATPDataLoader(path)
    matches, rankings, players = loader.load_matches(), loader.load_rankings(), loader.load_players()
    required = (
        (matches, {"tourney_date", "tourney_id", "tourney_name", "draw_size", "tourney_level",
                   "match_num", "best_of", "round", "winner_id", "loser_id", "surface", "score"}),
        (rankings, {"ranking_date", "rank", "player", "points"}),
        (players, {"player_id", "name_first", "name_last", "dob", "hand", "height"}),
    )
    for frame, columns in required:
        missing = columns - set(frame.columns)
        if missing or frame.is_empty():
            raise ValueError(f"Invalid ATP snapshot (missing columns: {sorted(missing)})")

    def latest(frame: pl.DataFrame, column: str) -> int:
        source = frame[column].cast(pl.String)
        parsed = source.str.strptime(pl.Date, "%Y%m%d", strict=False)
        if parsed.count() != source.count() or not parsed.count():
            raise ValueError(f"Invalid or missing dates in {column}")
        return int(parsed.max().strftime("%Y%m%d"))

    return ArchiveSummary(
        matches.height, latest(matches, "tourney_date"),
        rankings.height, latest(rankings, "ranking_date"), players.height,
    )


def refresh_atp_data(
    source: str = DEFAULT_SOURCE,
    *,
    snapshot_root: Path = SNAPSHOT_ROOT,
    fallback: Path = LEGACY_ATP_DIR,
) -> RefreshResult:
    """Validate a local or HTTPS archive before switching the active snapshot."""
    snapshot_root.mkdir(parents=True, exist_ok=True)
    active = snapshot_root / "current"

    with TemporaryDirectory(prefix=".refresh-", dir=snapshot_root) as work:
        workspace = Path(work)
        local = Path(source)
        if source.startswith("https://"):
            local = workspace / "source"
            result = run(
                ["git", "clone", "--depth", "1", "--quiet", "--", source, str(local)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode:
                raise ValueError(f"Could not fetch ATP archive: {result.stderr.strip()}")
        elif not local.is_dir():
            raise ValueError("ATP source must be a local directory or HTTPS Git repository")

        source_data = local / "atp" if (local / "atp").is_dir() else local
        files = _archive_files(source_data)
        staged = workspace / "validated"
        staged.mkdir()
        for name, file in files.items():
            copy2(file, staged / name)
        for name, file in (("LICENSE", local / "LICENSE"), ("UPSTREAM_README.md", source_data / "UPSTREAM_README.md")):
            if file.is_file() and not file.is_symlink():
                copy2(file, staged / name)
        (staged / "SOURCE.txt").write_text(source + "\n", encoding="utf-8")
        summary = _inventory(staged)
        with (snapshot_root / ".refresh.lock").open("a") as lock:
            flock(lock, LOCK_EX)
            if active.exists() and not active.is_symlink():
                raise ValueError(f"Managed snapshot pointer is not a symlink: {active}")
            previous = active.resolve() if active.is_dir() else fallback
            previous_files = _archive_files(previous)
            missing = previous_files.keys() - files.keys()
            if missing:
                raise ValueError(f"Snapshot would lose archive files: {', '.join(sorted(missing))}")
            if previous_files:
                old = _inventory(previous)
                if summary.latest_match < old.latest_match or summary.latest_ranking < old.latest_ranking:
                    raise ValueError("Snapshot dates are older than the active archive")
                if summary.matches < old.matches or summary.rankings < old.rankings or summary.players < old.players:
                    raise ValueError("Snapshot would reduce match, ranking, or player coverage")
                if files.keys() == previous_files.keys() and all(
                    cmp(staged / name, previous_files[name], shallow=False) for name in files
                ):
                    return RefreshResult(previous, old, False)

            version = snapshot_root / f"atp-{summary.latest_match}-{uuid4().hex[:8]}"
            staged.rename(version)
            pointer = workspace / "current"
            pointer.symlink_to(version.name, target_is_directory=True)
            pointer.replace(active)
            return RefreshResult(active, summary, True)
