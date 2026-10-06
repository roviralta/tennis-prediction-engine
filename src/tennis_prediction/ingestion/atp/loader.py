from pathlib import Path

import polars as pl


class ATPDataLoader:
    """Loads ATP data from the external Sackmann dataset."""

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)

    def load_matches(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pl.DataFrame:
        """Load ATP match data for the requested year range."""

        match_files = sorted(self.data_path.glob("atp_matches_[0-9][0-9][0-9][0-9].csv"))

        if start_year is not None:
            match_files = [
                path
                for path in match_files
                if self._extract_year(path) >= start_year
            ]

        if end_year is not None:
            match_files = [
                path
                for path in match_files
                if self._extract_year(path) <= end_year
            ]

        if not match_files:
            raise FileNotFoundError("No ATP match files found.")

        return pl.concat(
            [pl.read_csv(path) for path in match_files],
            how="diagonal_relaxed",
        )
    
    def load_rankings(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pl.DataFrame:
        """Load ATP ranking data for the requested year range."""

        ranking_files = sorted(
            self.data_path.glob("atp_rankings_*.csv")
        )

        if not ranking_files:
            raise FileNotFoundError("No ATP ranking files found.")

        rankings = pl.concat(
            [pl.read_csv(path) for path in ranking_files],
            how="diagonal_relaxed",
        )
        
        rankings = rankings.with_columns(pl.col("points").cast(pl.Int64))

        if start_year is not None or end_year is not None:
            rankings = rankings.with_columns(
                pl.col("ranking_date")
                .cast(pl.String)
                .str.slice(0, 4)
                .cast(pl.Int32)
                .alias("ranking_year")
            )

            if start_year is not None:
                rankings = rankings.filter(
                    pl.col("ranking_year") >= start_year
                )

            if end_year is not None:
                rankings = rankings.filter(
                    pl.col("ranking_year") <= end_year
                )

            rankings = rankings.drop("ranking_year")

        return rankings
    
    def load_players(self) -> pl.DataFrame:
        """Load ATP player data."""
        players_file = self.data_path / "atp_players.csv"
        if not players_file.exists():
            raise FileNotFoundError(
				f"ATP players file not found: {players_file}"
			)
        players = pl.read_csv(players_file)
        
        return players.with_columns(
			pl.col("player_id").cast(pl.Int64)
		)

    @staticmethod
    def _extract_year(path: Path) -> int:
        """Extract the year from an ATP match filename."""

        year = path.stem.replace("atp_matches_", "")

        if not year.isdigit():
            raise ValueError(f"Invalid ATP match filename: {path.name}")

        return int(year)