"""Download and combine historical football-data.co.uk league CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import BASE_URL, configured_football_data_leagues, download_url, season_codes
from src.preprocessing import clean_match_data


def update_historical_data(start_year: int = 2018, raw_dir: Path = Path("data/raw"), output: Path = Path("data/processed/historical_matches.csv")) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for comp in configured_football_data_leagues():
        code = comp["football_data_code"]
        for season in season_codes(start_year):
            url = f"{BASE_URL}/mmz4281/{season}/{code}.csv"
            raw_path = raw_dir / f"{season}_{code}.csv"
            try:
                content = download_url(url)
            except Exception as exc:  # noqa: BLE001 - continue across unavailable seasons/leagues
                print(f"skip {url}: {exc}")
                continue
            raw_path.write_bytes(content)
            frame = pd.read_csv(raw_path, encoding_errors="ignore")
            frame["Competition"] = comp["name"]
            frame["SourceFile"] = raw_path.name
            frames.append(frame)
            print(f"downloaded {url}")
    if not frames:
        raise SystemExit("No historical CSVs were downloaded.")
    cleaned = clean_match_data(pd.concat(frames, ignore_index=True))
    cleaned = cleaned.drop_duplicates(["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Competition"], keep="last")
    cleaned.to_csv(output, index=False)
    print(f"wrote {len(cleaned):,} rows to {output}")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2018)
    args = parser.parse_args()
    update_historical_data(args.start_year)


if __name__ == "__main__":
    main()
