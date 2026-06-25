"""Download API-based upcoming club fixtures and odds, with manual fallback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import safe_read_csv
from scripts.data_sources import (
    BASE_URL,
    UPCOMING_COLUMNS,
    api_football_events,
    configured_football_data_leagues,
    normalize_upcoming_frame,
    odds_api_events,
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
    automatic, messages = odds_api_events(international=False)
    for msg in messages:
        print(msg)
    frames = [automatic] if not automatic.empty else []
    if automatic.empty:
        fallback, fallback_messages = api_football_events(international=False)
        for msg in fallback_messages:
            print(msg)
        if not fallback.empty:
            frames.append(fallback)
    if manual_csv:
        manual = safe_read_csv(manual_csv, UPCOMING_COLUMNS)
        frames.append(normalize_upcoming_frame(manual))
        print(f"loaded manual fallback {manual_csv}")
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=UPCOMING_COLUMNS)
    result = result.drop_duplicates(["Date", "Competition", "HomeTeam", "AwayTeam"], keep="first")
    result = result.reindex(columns=UPCOMING_COLUMNS)
    result.to_csv(output, index=False)
    if result.empty:
        print(
            "No API or manual upcoming fixtures were available; "
            f"created {output} with valid headers for manual fallback."
        )
    print(f"wrote {len(result):,} rows to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-csv", help="Optional manual upcoming fixtures CSV fallback")
    args = parser.parse_args()
    update_upcoming_fixtures(args.manual_csv)


if __name__ == "__main__":
    main()
