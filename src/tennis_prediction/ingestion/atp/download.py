from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
DATA_DIR = Path("data/raw/atp")

START_YEAR = 2000
END_YEAR = 2026


def download_file(url: str, destination: Path) -> None:
    """Download a file if it does not already exist."""
    if destination.exists():
        print(f"Already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading: {destination.name}")
    urlretrieve(url, destination)

    print(f"Saved: {destination}")


def download_match_data() -> None:
    """Download ATP match data for the configured year range."""
    for year in range(START_YEAR, END_YEAR + 1):
        filename = f"atp_matches_{year}.csv"
        url = f"{BASE_URL}/{filename}"
        destination = DATA_DIR / filename

        download_file(url, destination)


if __name__ == "__main__":
    download_match_data()