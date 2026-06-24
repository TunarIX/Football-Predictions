"""Elo rating utilities for chronological football match data."""
from __future__ import annotations

import pandas as pd

HOME_ADVANTAGE = 60.0
DEFAULT_ELO = 1500.0
K_FACTOR = 28.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def result_score(ftr: str, perspective: str) -> float:
    if ftr == "D":
        return 0.5
    if perspective == "home":
        return 1.0 if ftr == "H" else 0.0
    return 1.0 if ftr == "A" else 0.0


def add_pre_match_elo(df: pd.DataFrame, default_elo: float = DEFAULT_ELO, k_factor: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE) -> pd.DataFrame:
    """Attach pre-match Elo ratings and update ratings after each match."""
    ratings: dict[str, float] = {}
    rows = []
    for _, match in df.sort_values("Date").iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]
        home_elo = ratings.get(home_team, default_elo)
        away_elo = ratings.get(away_team, default_elo)
        home_expected = expected_score(home_elo + home_advantage, away_elo)
        home_actual = result_score(match["FTR"], "home")
        change = k_factor * (home_actual - home_expected)

        enriched = match.to_dict()
        enriched["HomeElo"] = home_elo
        enriched["AwayElo"] = away_elo
        enriched["EloDiff"] = (home_elo + home_advantage) - away_elo
        enriched["HomeEloExpected"] = home_expected
        rows.append(enriched)

        ratings[home_team] = home_elo + change
        ratings[away_team] = away_elo - change
    return pd.DataFrame(rows)


def current_elos(df: pd.DataFrame, default_elo: float = DEFAULT_ELO, k_factor: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for _, match in df.sort_values("Date").iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]
        home_elo = ratings.get(home_team, default_elo)
        away_elo = ratings.get(away_team, default_elo)
        change = k_factor * (result_score(match["FTR"], "home") - expected_score(home_elo + home_advantage, away_elo))
        ratings[home_team] = home_elo + change
        ratings[away_team] = away_elo - change
    return ratings
