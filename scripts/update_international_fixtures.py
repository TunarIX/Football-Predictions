"""Download and safely update shared international upcoming fixtures."""

from __future__ import annotations

import argparse
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import UPCOMING_COLUMNS, USER_AGENT, api_football_events, normalize_upcoming_frame, odds_api_events
from src.data_loader import safe_read_csv
from src.fixtures import INTERNATIONAL_UPCOMING, write_international_fixtures
from src.match_context import is_international_competition_name, tournament_category

DEFAULT_SOURCES = [
    "https://fixturedownload.com/feed/json/fifa-world-cup-2026",
    "https://fixturedownload.com/feed/json/uefa-nations-league-2024",
]
FALLBACK_SOURCES = [
    "https://fixturedownload.com/feed/csv/fifa-world-cup-2026",
    "https://fixturedownload.com/feed/csv/uefa-nations-league-2024",
]


def _download(url: str, timeout: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:  # nosec - public fixture feeds only
        return response.read()


def _source_urls() -> list[str]:
    configured = os.getenv("INTERNATIONAL_FIXTURES_URLS", "").strip()
    if configured:
        return [url.strip() for url in configured.split(",") if url.strip()]
    return [*DEFAULT_SOURCES, *FALLBACK_SOURCES]


def _read_payload(payload: bytes, url: str) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    suffix = url.lower()
    try:
        if suffix.endswith(".json") or b"{" in payload[:20] or b"[" in payload[:20]:
            raw = json.loads(payload.decode("utf-8-sig"))
            if isinstance(raw, dict):
                for key in ("fixtures", "matches", "events", "data"):
                    if isinstance(raw.get(key), list):
                        raw = raw[key]
                        break
            return pd.json_normalize(raw)
        if suffix.endswith((".xls", ".xlsx")):
            return pd.read_excel(BytesIO(payload))
        return pd.read_csv(StringIO(payload.decode("utf-8-sig")))
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame(columns=UPCOMING_COLUMNS)


def _normalise_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    data = frame.copy()
    data.columns = [str(c).strip() for c in data.columns]
    aliases = {
        "matchdate": "Date", "dateutc": "Date", "utcdate": "Date", "kickoff": "Date", "datetime": "Date",
        "home_team": "HomeTeam", "hometeam": "HomeTeam", "home": "HomeTeam", "team1": "HomeTeam",
        "away_team": "AwayTeam", "awayteam": "AwayTeam", "away": "AwayTeam", "team2": "AwayTeam",
        "competition": "Competition", "league": "Competition", "tournament": "Competition", "round": "Competition", "group": "Competition",
    }
    data = data.rename(columns={c: aliases.get(c.lower().replace(" ", "").replace("_", ""), c) for c in data.columns})
    if "Competition" not in data.columns:
        data["Competition"] = data.get("Round", "International matches")
    if "Time" not in data.columns:
        parsed = pd.to_datetime(data.get("Date"), errors="coerce", utc=True)
        data["Time"] = parsed.dt.strftime("%H:%M").fillna("") if not parsed.empty else ""
    normalized = normalize_upcoming_frame(data)
    if normalized.empty:
        return normalized
    international = normalized["Competition"].map(is_international_competition_name)
    normalized = normalized[international | normalized["Competition"].astype(str).str.contains("world cup|euro|copa|gold cup|asian cup|africa cup|friendly|nations|qual", case=False, na=False)].copy()
    normalized["Competition"] = normalized["Competition"].replace("", "International matches")
    normalized["TournamentCategory"] = normalized["Competition"].map(tournament_category)
    return normalized.reindex(columns=[*UPCOMING_COLUMNS, "TournamentCategory"])


def download_international_fixtures() -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    messages: list[str] = []
    api_frame, api_messages = odds_api_events(international=True)
    messages.extend(api_messages)
    if not api_frame.empty:
        frames.append(api_frame)
    if api_frame.empty:
        fallback_frame, fallback_messages = api_football_events(international=True)
        messages.extend(fallback_messages)
        if not fallback_frame.empty:
            frames.append(fallback_frame)

    for url in _source_urls():
        try:
            frame = _normalise_source_frame(_read_payload(_download(url), url))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            messages.append(f"failed {url}: {exc}")
            continue
        if frame.empty:
            messages.append(f"no valid fixtures from {url}")
            continue
        messages.append(f"downloaded {len(frame):,} fixtures from {url}")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*UPCOMING_COLUMNS, "TournamentCategory"]), messages
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "Competition", "HomeTeam", "AwayTeam"], keep="first")
    return result.sort_values(["Date", "Time", "Competition"]).reset_index(drop=True), messages


def read_source(source_csv: str | None = None) -> pd.DataFrame:
    if not source_csv:
        fixtures, _ = download_international_fixtures()
        return fixtures
    return safe_read_csv(source_csv, UPCOMING_COLUMNS)


def _valid(fixtures: pd.DataFrame) -> bool:
    return not fixtures.empty and {"Date", "Competition", "HomeTeam", "AwayTeam"}.issubset(fixtures.columns)


def update_international_fixtures(source_csv: str | None = None, output: Path = INTERNATIONAL_UPCOMING) -> pd.DataFrame:
    existing = safe_read_csv(output, UPCOMING_COLUMNS) if output.exists() else pd.DataFrame(columns=UPCOMING_COLUMNS)
    messages: list[str] = []
    if source_csv:
        fixtures = safe_read_csv(source_csv, UPCOMING_COLUMNS)
        messages.append(f"loaded manual fallback {source_csv}")
    else:
        fixtures, messages = download_international_fixtures()
    normalized = normalize_upcoming_frame(fixtures)
    if not _valid(normalized):
        for msg in messages:
            print(msg)
        if _valid(existing):
            print(f"International fixtures update failed validation; kept existing valid file at {output}.")
            return existing
        print(f"No valid international fixtures found; kept/created no replacement for {output}.")
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    written = write_international_fixtures(normalized, output)
    for msg in messages:
        print(msg)
    print(f"wrote {len(written):,} international fixtures to {output}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", help="Manual international fixtures CSV")
    parser.add_argument("--output", type=Path, default=INTERNATIONAL_UPCOMING)
    args = parser.parse_args()
    update_international_fixtures(args.source_csv, args.output)


if __name__ == "__main__":
    main()
