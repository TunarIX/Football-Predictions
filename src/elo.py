"""Chronological Elo ratings for football teams."""

from __future__ import annotations

import pandas as pd


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def actual_scores(ftr: str) -> tuple[float, float]:
    if ftr == "H":
        return 1.0, 0.0
    if ftr == "A":
        return 0.0, 1.0
    return 0.5, 0.5


def elo_before_matches(
    df: pd.DataFrame, base: float = 1500, k: float = 24, home_advantage: float = 55
) -> pd.DataFrame:
    """Attach pre-match Elo ratings and update ratings after each result."""
    ratings: dict[str, float] = {}
    rows = []
    for _, match in df.sort_values("Date").iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        home_elo = ratings.get(home, base)
        away_elo = ratings.get(away, base)
        rows.append(
            {
                "HomeElo": home_elo,
                "AwayElo": away_elo,
                "EloDiff": home_elo + home_advantage - away_elo,
            }
        )
        exp_home = expected_score(home_elo + home_advantage, away_elo)
        score_home, score_away = actual_scores(match["FTR"])
        ratings[home] = home_elo + k * (score_home - exp_home)
        ratings[away] = away_elo + k * (score_away - (1 - exp_home))
    enriched = df.sort_values("Date").copy().reset_index(drop=True)
    return pd.concat([enriched, pd.DataFrame(rows)], axis=1)


def current_elo_ratings(
    df: pd.DataFrame, base: float = 1500, k: float = 24, home_advantage: float = 55
) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for _, match in df.sort_values("Date").iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        home_elo = ratings.get(home, base)
        away_elo = ratings.get(away, base)
        exp_home = expected_score(home_elo + home_advantage, away_elo)
        score_home, score_away = actual_scores(match["FTR"])
        ratings[home] = home_elo + k * (score_home - exp_home)
        ratings[away] = away_elo + k * (score_away - (1 - exp_home))
    return ratings
