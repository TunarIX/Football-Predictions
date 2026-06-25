"""Download football-data.co.uk upcoming fixtures when available, with manual fallback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import (
    BASE_URL,
    UPCOMING_COLUMNS,
    configured_football_data_leagues,
    normalize_upcoming_frame,
    read_csv_url,
)

FIXTURE_URLS = [
    f"{BASE_URL}/fixtures.csv",
    f"{BASE_URL}/fixtures.xls",  # pandas may parse if available in some mirrors/environments
]


def _football_data_fixtures() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    code_to_name = {comp["football_data_code"]: comp["name"] for comp in configured_football_data_leagues()}
    for url in FIXTURE_URLS:
        frame = read_csv_url(url)
        if frame.empty:
            print(f"fixtures unavailable: {url}")
            continue
        if "Div" in frame.columns:
            frame = frame[frame["Div"].isin(code_to_name)].copy()
            frame["Competition"] = frame["Div"].map(code_to_name)
        frames.append(frame)
        print(f"downloaded fixtures from {url}")
    if not frames:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    return normalize_upcoming_frame(pd.concat(frames, ignore_index=True))


def update_upcoming_fixtures(manual_csv: str | None = None, output: Path = Path("data/upcoming/upcoming_fixtures.csv")) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)
    automatic = _football_data_fixtures()
    frames = [automatic] if not automatic.empty else []
    if manual_csv:
        manual = pd.read_csv(manual_csv, encoding_errors="ignore")
        frames.append(normalize_upcoming_frame(manual))
        print(f"loaded manual fallback {manual_csv}")
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=UPCOMING_COLUMNS)
    result = result.drop_duplicates(["Date", "Competition", "HomeTeam", "AwayTeam"], keep="first")
    result.to_csv(output, index=False)
    print(f"wrote {len(result):,} rows to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-csv", help="Optional manual upcoming fixtures CSV fallback")
    args = parser.parse_args()
    update_upcoming_fixtures(args.manual_csv)


if __name__ == "__main__":
    main()
