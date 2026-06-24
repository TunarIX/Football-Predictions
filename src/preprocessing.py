"""Data cleaning utilities for football match CSV files."""
from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "Tournament", "Neutral", "B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA",
    "PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
]
FOOTBALL_DATA_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in {"Tournament", "Neutral"}]
REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Tournament", "Neutral"}]
NUMERIC_COLUMNS = ["FTHG", "FTAG", "HTHG", "HTAG", *ODDS_COLUMNS]
RESULTS = {"H", "D", "A"}
INTERNATIONAL_ALIASES = {
    "date": "Date", "home_team": "HomeTeam", "away_team": "AwayTeam", "home": "HomeTeam", "away": "AwayTeam",
    "home_score": "FTHG", "away_score": "FTAG", "home_goals": "FTHG", "away_goals": "FTAG",
    "tournament": "Tournament", "neutral": "Neutral", "result": "FTR",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and map common football CSV variants to canonical names."""
    canonical = {col.lower(): col for col in EXPECTED_COLUMNS}
    canonical.update(INTERNATIONAL_ALIASES)
    renamed = {col: canonical.get(str(col).strip().lower(), str(col).strip()) for col in df.columns}
    return df.rename(columns=renamed)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse common football date formats without assuming one fixed locale."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce", dayfirst=False)
    return parsed


def infer_result(data: pd.DataFrame) -> pd.Series:
    """Infer H/D/A result labels from scores when FTR is absent or non-standard."""
    existing = data.get("FTR", pd.Series(pd.NA, index=data.index)).astype("string").str.upper().str.strip()
    inferred = pd.Series(pd.NA, index=data.index, dtype="string")
    inferred.loc[data["FTHG"] > data["FTAG"]] = "H"
    inferred.loc[data["FTHG"] == data["FTAG"]] = "D"
    inferred.loc[data["FTHG"] < data["FTAG"]] = "A"
    return existing.where(existing.isin(RESULTS), inferred)


def clean_match_data(df: pd.DataFrame, match_type: str = "club") -> pd.DataFrame:
    """Clean and standardize uploaded historical match rows."""
    data = normalize_columns(df).copy()
    for col in EXPECTED_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA

    data = data[EXPECTED_COLUMNS].copy()
    data["Date"] = parse_dates(data["Date"])
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ["HomeTeam", "AwayTeam", "FTR", "HTR", "Tournament"]:
        data[col] = data[col].astype("string").str.strip()
    data["FTR"] = infer_result(data)
    data["HTR"] = data["HTR"].str.upper()
    data["MatchType"] = match_type

    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[data["FTR"].isin(RESULTS)]
    data = data[(data["FTHG"] >= 0) & (data["FTAG"] >= 0)]
    data = data.sort_values("Date").reset_index(drop=True)
    return data
