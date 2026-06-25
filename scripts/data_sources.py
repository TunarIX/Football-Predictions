"""Shared football-data.co.uk download and odds-source helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.competitions import load_competitions

BASE_URL = "https://www.football-data.co.uk"
USER_AGENT = "Football-Predictions/1.0 (+https://www.football-data.co.uk/)"
ODDS_SOURCE_NOTES = (
    "football-data.co.uk family (Betbrain, Oddsportal, individual bookmakers)"
)
UPCOMING_COLUMNS = [
    "Date",
    "Time",
    "Competition",
    "HomeTeam",
    "AwayTeam",
    "HomeOdds",
    "DrawOdds",
    "AwayOdds",
    "OddsSource",
]


def configured_football_data_leagues(config_path: str | Path = "config/competitions.yml") -> list[dict]:
    """Return configured football-data.co.uk competitions with a league code."""
    return [
        comp
        for comp in load_competitions(config_path)
        if comp.get("data_source") == "football-data.co.uk" and comp.get("football_data_code")
    ]


def season_codes(start_year: int = 2018, end_year: int | None = None) -> list[str]:
    """Build football-data season folder codes, e.g. 2324."""
    if end_year is None:
        today = pd.Timestamp.utcnow()
        end_year = today.year if today.month >= 7 else today.year - 1
    return [f"{year % 100:02d}{(year + 1) % 100:02d}" for year in range(start_year, end_year + 1)]


def download_url(url: str, timeout: int = 30) -> bytes:
    """Download one URL politely with a transparent user agent."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:  # nosec - configured public CSV URLs
        return response.read()


def read_csv_url(url: str) -> pd.DataFrame:
    """Read a CSV URL and return an empty frame when it is unavailable."""
    try:
        return pd.read_csv(url, encoding_errors="ignore")
    except (HTTPError, URLError, TimeoutError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def choose_odds(row: pd.Series) -> tuple[float | None, float | None, float | None, str]:
    """Prefer market-average odds, otherwise Bet365, keeping the selected source visible."""
    for source, cols in (("Market Avg", ("AvgH", "AvgD", "AvgA")), ("Bet365", ("B365H", "B365D", "B365A"))):
        values = pd.to_numeric(row.reindex(cols), errors="coerce")
        if values.notna().all() and (values > 1).all():
            return float(values.iloc[0]), float(values.iloc[1]), float(values.iloc[2]), source
    return None, None, None, "Unavailable"


def normalize_upcoming_frame(df: pd.DataFrame, competition: str | None = None) -> pd.DataFrame:
    """Normalize football-data/manual fixture CSVs to the app's upcoming schema."""
    if df.empty:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    aliases = {
        "home": "HomeTeam",
        "hometeam": "HomeTeam",
        "home_team": "HomeTeam",
        "away": "AwayTeam",
        "awayteam": "AwayTeam",
        "away_team": "AwayTeam",
        "match date": "Date",
        "fixture date": "Date",
        "time": "Time",
        "competition": "Competition",
        "league": "Competition",
        "homeodds": "HomeOdds",
        "home_odds": "HomeOdds",
        "drawodds": "DrawOdds",
        "draw_odds": "DrawOdds",
        "awayodds": "AwayOdds",
        "away_odds": "AwayOdds",
        "oddssource": "OddsSource",
        "odds_source": "OddsSource",
    }
    data = data.rename(columns={c: aliases.get(c.lower().replace(" ", ""), c) for c in data.columns})
    if "Competition" not in data.columns:
        data["Competition"] = competition or data.get("Div", "")
    if competition:
        data["Competition"] = data["Competition"].fillna(competition).replace("", competition)
    if "Time" not in data.columns:
        data["Time"] = ""
    odds = data.apply(choose_odds, axis=1, result_type="expand")
    odds.columns = ["_HomeOdds", "_DrawOdds", "_AwayOdds", "_OddsSource"]
    for col, fallback in [("HomeOdds", "_HomeOdds"), ("DrawOdds", "_DrawOdds"), ("AwayOdds", "_AwayOdds"), ("OddsSource", "_OddsSource")]:
        if col not in data.columns:
            data[col] = odds[fallback]
        else:
            data[col] = data[col].fillna(odds[fallback])
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", dayfirst=False)
    missing_dates = data["Date"].isna()
    if missing_dates.any():
        data.loc[missing_dates, "Date"] = pd.to_datetime(
            data.loc[missing_dates, "Date"], errors="coerce", dayfirst=True
        )
    for col in ["HomeOdds", "DrawOdds", "AwayOdds"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["OddsSource"] = data["OddsSource"].fillna("Unavailable")
    keep = data.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    if keep.empty:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    keep["Date"] = keep["Date"].dt.strftime("%Y-%m-%d")
    return keep[UPCOMING_COLUMNS].drop_duplicates(["Date", "Competition", "HomeTeam", "AwayTeam"]).sort_values(["Date", "Time", "Competition"]).reset_index(drop=True)
