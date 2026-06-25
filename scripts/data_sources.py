"""Shared football-data.co.uk download and odds-source helpers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from urllib.parse import urlencode

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
    "Over25Odds",
    "Under25Odds",
    "OddsSource",
]



def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env without overriding the shell."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _decimal(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number) or float(number) <= 1:
        return None
    return float(number)


def _outcome_price(outcomes: list[dict], name: str | None = None, point: float | None = None) -> float | None:
    for outcome in outcomes or []:
        if name is not None and str(outcome.get("name", "")).casefold() != name.casefold():
            continue
        if point is not None:
            outcome_point = pd.to_numeric(outcome.get("point"), errors="coerce")
            if pd.isna(outcome_point) or float(outcome_point) != point:
                continue
        price = _decimal(outcome.get("price"))
        if price is not None:
            return price
    return None


def _split_time(value: object) -> tuple[str | None, str]:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None, ""
    return parsed.strftime("%Y-%m-%d"), parsed.strftime("%H:%M")


def odds_api_missing_key_message() -> str:
    return "Set ODDS_API_KEY in .env or use manual CSV fallback."


def odds_api_events(international: bool = False, api_key: str | None = None, regions: str = "us,uk,eu", timeout: int = 30) -> tuple[pd.DataFrame, list[str]]:
    """Fetch upcoming soccer fixtures and h2h/totals odds from The Odds API."""
    load_dotenv()
    key = api_key or os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        return pd.DataFrame(columns=UPCOMING_COLUMNS), [odds_api_missing_key_message()]
    params = urlencode({"apiKey": key, "regions": regions, "markets": "h2h,totals", "oddsFormat": "decimal"})
    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?{params}"
    try:
        payload = download_url(url, timeout=timeout)
        events = json.loads(payload.decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return pd.DataFrame(columns=UPCOMING_COLUMNS), [f"The Odds API request failed: {exc}"]
    rows = []
    for event in events if isinstance(events, list) else []:
        sport_title = str(event.get("sport_title") or event.get("sport_key") or "Soccer")
        sport_key = str(event.get("sport_key") or "")
        looks_international = any(token in f"{sport_title} {sport_key}".lower() for token in ["international", "world cup", "uefa nations", "friendlies"])
        if international != looks_international:
            continue
        date, time = _split_time(event.get("commence_time"))
        if not date:
            continue
        row = {"Date": date, "Time": time, "Competition": sport_title, "HomeTeam": event.get("home_team"), "AwayTeam": event.get("away_team"), "HomeOdds": None, "DrawOdds": None, "AwayOdds": None, "Over25Odds": None, "Under25Odds": None, "OddsSource": "Unavailable"}
        for bookmaker in event.get("bookmakers") or []:
            source = bookmaker.get("title") or bookmaker.get("key") or "The Odds API"
            for market in bookmaker.get("markets") or []:
                key_name = market.get("key")
                outcomes = market.get("outcomes") or []
                if key_name == "h2h" and row["HomeOdds"] is None:
                    row["HomeOdds"] = _outcome_price(outcomes, event.get("home_team"))
                    row["AwayOdds"] = _outcome_price(outcomes, event.get("away_team"))
                    row["DrawOdds"] = _outcome_price(outcomes, "Draw")
                    if row["HomeOdds"] and row["AwayOdds"]:
                        row["OddsSource"] = f"The Odds API: {source}"
                elif key_name == "totals" and row["Over25Odds"] is None:
                    row["Over25Odds"] = _outcome_price(outcomes, "Over", 2.5)
                    row["Under25Odds"] = _outcome_price(outcomes, "Under", 2.5)
            if row["HomeOdds"] is not None and row["Over25Odds"] is not None:
                break
        rows.append(row)
    return normalize_upcoming_frame(pd.DataFrame(rows)), [f"downloaded {len(rows):,} fixtures from The Odds API"]


def api_football_events(international: bool = False, api_key: str | None = None, timeout: int = 30) -> tuple[pd.DataFrame, list[str]]:
    """Optional API-Football future fallback; fixtures-only when configured."""
    load_dotenv()
    key = api_key or os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        return pd.DataFrame(columns=UPCOMING_COLUMNS), []
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    params = urlencode({"from": today, "to": (pd.Timestamp.utcnow() + pd.Timedelta(days=14)).strftime("%Y-%m-%d")})
    request = Request(f"https://v3.football.api-sports.io/fixtures?{params}", headers={"x-apisports-key": key, "User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec - configured public API
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return pd.DataFrame(columns=UPCOMING_COLUMNS), [f"API-Football request failed: {exc}"]
    rows = []
    for item in data.get("response", []) if isinstance(data, dict) else []:
        league = item.get("league", {})
        country = str(league.get("country", ""))
        league_name = str(league.get("name", "Soccer"))
        is_intl = country.casefold() == "world" or "international" in league_name.casefold()
        if international != is_intl:
            continue
        date, time = _split_time(item.get("fixture", {}).get("date"))
        teams = item.get("teams", {})
        rows.append({"Date": date, "Time": time, "Competition": league_name, "HomeTeam": teams.get("home", {}).get("name"), "AwayTeam": teams.get("away", {}).get("name"), "OddsSource": "API-Football fixtures (odds unavailable)"})
    return normalize_upcoming_frame(pd.DataFrame(rows)), [f"downloaded {len(rows):,} fixtures from API-Football"]


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
    except (HTTPError, URLError, TimeoutError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
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
        "hometeamname": "HomeTeam",
        "away": "AwayTeam",
        "awayteam": "AwayTeam",
        "awayteamname": "AwayTeam",
        "date": "Date",
        "matchdate": "Date",
        "fixturedate": "Date",
        "time": "Time",
        "competition": "Competition",
        "league": "Competition",
        "tournament": "Competition",
        "homeodds": "HomeOdds",
        "drawodds": "DrawOdds",
        "awayodds": "AwayOdds",
        "over25odds": "Over25Odds",
        "over25": "Over25Odds",
        "over250dds": "Over25Odds",
        "under25odds": "Under25Odds",
        "under25": "Under25Odds",
        "under250dds": "Under25Odds",
        "oddssource": "OddsSource",
    }

    def canonical_column_name(column: object) -> str:
        key = re.sub(r"[^a-z0-9]", "", str(column).strip().lower())
        return aliases.get(key, str(column).strip())

    data = data.rename(columns={c: canonical_column_name(c) for c in data.columns})
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
    for col in ["Over25Odds", "Under25Odds"]:
        if col not in data.columns:
            data[col] = pd.NA
    for col in ["HomeOdds", "DrawOdds", "AwayOdds", "Over25Odds", "Under25Odds"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if "OddsSource" not in data.columns:
        data["OddsSource"] = "Unavailable"
    data["OddsSource"] = data["OddsSource"].fillna("Unavailable")
    keep = data.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    if keep.empty:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    keep["Date"] = keep["Date"].dt.strftime("%Y-%m-%d")
    return keep[UPCOMING_COLUMNS].drop_duplicates(["Date", "Competition", "HomeTeam", "AwayTeam"]).sort_values(["Date", "Time", "Competition"]).reset_index(drop=True)
