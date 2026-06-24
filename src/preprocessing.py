"""Data cleaning utilities for football-data.co.uk CSV files."""
from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA",
    "PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
]

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"}]
NUMERIC_COLUMNS = ["FTHG", "FTAG", "HTHG", "HTAG", *ODDS_COLUMNS]
RESULTS = {"H", "D", "A"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names and map case-insensitive football-data column names to canonical names."""
    canonical = {col.lower(): col for col in EXPECTED_COLUMNS}
    renamed = {col: canonical.get(str(col).strip().lower(), str(col).strip()) for col in df.columns}
    return df.rename(columns=renamed)


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse common football-data date formats without assuming one fixed locale."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series.loc[missing], errors="coerce", dayfirst=False)
    return parsed


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize uploaded historical match rows."""
    data = normalize_columns(df).copy()
    for col in EXPECTED_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA

    data = data[EXPECTED_COLUMNS].copy()
    data["Date"] = parse_dates(data["Date"])
    for col in NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ["HomeTeam", "AwayTeam", "FTR", "HTR"]:
        data[col] = data[col].astype("string").str.strip()
    data["FTR"] = data["FTR"].str.upper()
    data["HTR"] = data["HTR"].str.upper()

    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[data["FTR"].isin(RESULTS)]
    data = data[(data["FTHG"] >= 0) & (data["FTAG"] >= 0)]
    data = data.sort_values("Date").reset_index(drop=True)
    return data
