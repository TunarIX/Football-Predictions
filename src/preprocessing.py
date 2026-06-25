"""Data cleaning utilities for football match CSV files."""

from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "B365H",
    "B365D",
    "B365A",
    "BWH",
    "BWD",
    "BWA",
    "IWH",
    "IWD",
    "IWA",
    "PSH",
    "PSD",
    "PSA",
    "MaxH",
    "MaxD",
    "MaxA",
    "AvgH",
    "AvgD",
    "AvgA",
    "OddsSource",
]

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLUMNS = [
    c
    for c in EXPECTED_COLUMNS
    if c
    not in {
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HTHG",
        "HTAG",
        "HTR",
    }
]
NUMERIC_COLUMNS = ["FTHG", "FTAG", "HTHG", "HTAG", *[c for c in ODDS_COLUMNS if c != "OddsSource"]]
RESULTS = {"H", "D", "A"}

INTERNATIONAL_ALIASES = {
    "date": "Date",
    "home_team": "HomeTeam",
    "hometeam": "HomeTeam",
    "home": "HomeTeam",
    "away_team": "AwayTeam",
    "awayteam": "AwayTeam",
    "away": "AwayTeam",
    "home_score": "FTHG",
    "homegoals": "FTHG",
    "fthg": "FTHG",
    "away_score": "FTAG",
    "awaygoals": "FTAG",
    "ftag": "FTAG",
    "tournament": "Competition",
    "competition": "Competition",
    "neutral": "Neutral",
    "country": "Country",
    "home_odds": "AvgH",
    "draw_odds": "AvgD",
    "away_odds": "AvgA",
    "odds_source": "OddsSource",
    "avgh": "AvgH",
    "avgd": "AvgD",
    "avga": "AvgA",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and map case-insensitive football-data column names to canonical names."""
    canonical = {col.lower(): col for col in EXPECTED_COLUMNS}
    renamed = {
        col: canonical.get(str(col).strip().lower(), str(col).strip())
        for col in df.columns
    }
    return df.rename(columns=renamed)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse common football date formats without assuming one fixed locale."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing], errors="coerce", dayfirst=False
        )
    return parsed


def _finish_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    for col in EXPECTED_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA
    data = data[
        [
            *EXPECTED_COLUMNS,
            *[
                c
                for c in ["Competition", "Neutral", "Country", "SourceFile"]
                if c in data.columns
            ],
        ]
    ].copy()
    data["Date"] = parse_dates(data["Date"])
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["HomeTeam", "AwayTeam", "FTR", "HTR", "OddsSource"]:
        data[col] = data[col].astype("string").str.strip()
    avg_available = data[["AvgH", "AvgD", "AvgA"]].notna().all(axis=1)
    bet365_available = data[["B365H", "B365D", "B365A"]].notna().all(axis=1)
    missing_source = data["OddsSource"].isna() | (data["OddsSource"] == "")
    data.loc[missing_source & avg_available, "OddsSource"] = "Market Avg"
    data.loc[missing_source & ~avg_available & bet365_available, "OddsSource"] = "Bet365"
    data.loc[missing_source & ~avg_available & ~bet365_available, "OddsSource"] = "Unavailable"
    data["FTR"] = data["FTR"].str.upper()
    data["HTR"] = data["HTR"].str.upper()
    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[data["FTR"].isin(RESULTS)]
    data = data[(data["FTHG"] >= 0) & (data["FTAG"] >= 0)]
    return data.sort_values("Date").reset_index(drop=True)


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize uploaded football-data.co.uk historical match rows."""
    return _finish_cleaning(normalize_columns(df).copy())


def clean_international_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean national-team CSVs with dates, teams, scores, tournaments, and optional odds."""
    data = df.copy()
    renamed = {
        col: INTERNATIONAL_ALIASES.get(str(col).strip().lower(), str(col).strip())
        for col in data.columns
    }
    data = data.rename(columns=renamed)
    if "FTR" not in data.columns and {"FTHG", "FTAG"}.issubset(data.columns):
        home_goals = pd.to_numeric(data["FTHG"], errors="coerce")
        away_goals = pd.to_numeric(data["FTAG"], errors="coerce")
        data["FTR"] = "D"
        data.loc[home_goals > away_goals, "FTR"] = "H"
        data.loc[home_goals < away_goals, "FTR"] = "A"
    return _finish_cleaning(data)
