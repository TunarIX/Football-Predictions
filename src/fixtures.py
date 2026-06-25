"""Fixture and shared international-data helpers.

This module intentionally keeps international competitions on one national-team
history/fixtures source so a real fixture API adapter can be added later without
changing the prediction pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.data_sources import UPCOMING_COLUMNS, normalize_upcoming_frame
from src.data_loader import safe_read_csv
from src.preprocessing import EXPECTED_COLUMNS, clean_international_match_data

INTERNATIONAL_HISTORICAL = Path("data/processed/international_matches.csv")
INTERNATIONAL_UPCOMING = Path("data/upcoming/international_fixtures.csv")
CLUB_HISTORICAL = Path("data/processed/historical_matches.csv")
CLUB_UPCOMING = Path("data/upcoming/upcoming_fixtures.csv")
MISSING_INTERNATIONAL_HISTORY_MESSAGE = (
    "Missing international historical data. Add data/processed/international_matches.csv "
    "with national-team match rows, or update it from your international data provider."
)
NO_INTERNATIONAL_FIXTURES_MESSAGE = (
    "No international fixtures available. Add data/upcoming/international_fixtures.csv "
    "or connect a fixture API."
)


def is_international_competition(competition: str | None) -> bool:
    text = (competition or "").strip().lower()
    return text in {"fifa world cup", "international matches"}


def competition_filter_name(competition: str | None) -> str | None:
    """Return the shared-data tournament filter for a selectable competition."""
    if (competition or "").strip().lower() == "fifa world cup":
        return "FIFA World Cup"
    return None


def _filter_competition(df: pd.DataFrame, competition: str | None) -> pd.DataFrame:
    filter_name = competition_filter_name(competition)
    if not filter_name or df.empty:
        return df
    comp_col = "Competition" if "Competition" in df.columns else "Tournament" if "Tournament" in df.columns else None
    if comp_col is None:
        return df.iloc[0:0].copy()
    return df[df[comp_col].astype(str).str.strip().str.casefold() == filter_name.casefold()].copy()


def load_historical_matches_for_competition(competition: str | None, bookmaker: str | None = None) -> tuple[pd.DataFrame, str, str | None]:
    """Load historical matches for club or shared international competitions.

    Returns ``(dataframe, source_note, warning_message)``.
    """
    if is_international_competition(competition):
        if not INTERNATIONAL_HISTORICAL.exists():
            return pd.DataFrame(columns=EXPECTED_COLUMNS), str(INTERNATIONAL_HISTORICAL), MISSING_INTERNATIONAL_HISTORY_MESSAGE
        raw = safe_read_csv(INTERNATIONAL_HISTORICAL, EXPECTED_COLUMNS, parse_dates=["Date"])
        cleaned = clean_international_match_data(raw) if not raw.empty else pd.DataFrame(columns=EXPECTED_COLUMNS)
        filtered = _filter_competition(cleaned, competition)
        return filtered, str(INTERNATIONAL_HISTORICAL), None
    data = safe_read_csv(CLUB_HISTORICAL, EXPECTED_COLUMNS, parse_dates=["Date"])
    if competition and not data.empty and "Competition" in data.columns:
        data = data[data["Competition"].astype(str).str.casefold() == competition.casefold()].copy()
    return data, str(CLUB_HISTORICAL), None


def load_upcoming_fixtures_for_competition(competition: str | None) -> tuple[pd.DataFrame, Path, str | None]:
    """Load upcoming fixtures using shared international fixtures when applicable."""
    path = INTERNATIONAL_UPCOMING if is_international_competition(competition) else CLUB_UPCOMING
    warning = NO_INTERNATIONAL_FIXTURES_MESSAGE if is_international_competition(competition) and not path.exists() else None
    fixtures = normalize_upcoming_frame(safe_read_csv(path, UPCOMING_COLUMNS))
    fixtures = _filter_competition(fixtures, competition)
    if fixtures.empty and is_international_competition(competition):
        warning = NO_INTERNATIONAL_FIXTURES_MESSAGE
    return fixtures, path, warning


def write_international_fixtures(fixtures: pd.DataFrame, output: Path = INTERNATIONAL_UPCOMING) -> pd.DataFrame:
    """Safely normalize and write shared international upcoming fixtures CSV."""
    normalized = normalize_upcoming_frame(fixtures)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return normalized
