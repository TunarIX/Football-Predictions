"""Odds conversion and historical odds-similarity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

BOOKMAKERS = {
    "Bet365": ("B365H", "B365D", "B365A"),
    "Bet&Win": ("BWH", "BWD", "BWA"),
    "Interwetten": ("IWH", "IWD", "IWA"),
    "Pinnacle": ("PSH", "PSD", "PSA"),
    "Market Max": ("MaxH", "MaxD", "MaxA"),
    "Market Avg": ("AvgH", "AvgD", "AvgA"),
}
PROB_COLS = ["ImpHome", "ImpDraw", "ImpAway"]


def odds_to_probabilities(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Convert decimal odds to normalized implied probabilities."""
    odds = np.array([home_odds, draw_odds, away_odds], dtype=float)
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1):
        return (np.nan, np.nan, np.nan)
    raw = 1 / odds
    probs = raw / raw.sum()
    return tuple(probs.round(4))


def add_implied_probabilities(
    df: pd.DataFrame, bookmaker: str = "Market Avg"
) -> pd.DataFrame:
    """Add normalized implied probability columns for the selected bookmaker odds."""
    data = df.copy()
    h_col, d_col, a_col = BOOKMAKERS.get(bookmaker, BOOKMAKERS["Market Avg"])
    for col in (h_col, d_col, a_col):
        if col not in data.columns:
            data[col] = pd.NA
    probs = data[[h_col, d_col, a_col]].apply(
        lambda row: odds_to_probabilities(*row), axis=1, result_type="expand"
    )
    probs.columns = PROB_COLS
    return pd.concat([data, probs], axis=1)


def odds_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Compare average implied probabilities with actual H/D/A outcome frequencies."""
    valid = df.dropna(subset=PROB_COLS + ["FTR"])
    if valid.empty:
        return pd.DataFrame()
    rows = []
    for result, label, col in [
        ("H", "Home win", "ImpHome"),
        ("D", "Draw", "ImpDraw"),
        ("A", "Away win", "ImpAway"),
    ]:
        rows.append(
            {
                "Outcome": label,
                "Average implied probability": valid[col].mean(),
                "Actual frequency": (valid["FTR"] == result).mean(),
                "Matches": len(valid),
            }
        )
    return pd.DataFrame(rows)


def similar_odds_matches(
    df: pd.DataFrame,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    tolerance: float = 0.04,
) -> pd.DataFrame:
    """Find historical rows with similar implied-probability profiles."""
    valid = df.dropna(subset=PROB_COLS + ["FTR"]).copy()
    if valid.empty:
        return valid
    target = np.array([home_prob, draw_prob, away_prob], dtype=float)
    valid["OddsDistance"] = np.linalg.norm(valid[PROB_COLS].to_numpy() - target, axis=1)
    return valid[valid["OddsDistance"] <= tolerance].sort_values("OddsDistance")
