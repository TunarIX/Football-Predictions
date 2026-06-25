"""Chronological Elo ratings for football teams.

The helpers in this module deliberately update ratings one match at a time so
features built for a fixture only see matches that were already played.
"""

from __future__ import annotations

import re

import pandas as pd

DEFAULT_HOME_ADVANTAGE = 55.0
DEFAULT_BASE_K = 24.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def actual_scores(ftr: str) -> tuple[float, float]:
    if ftr == "H":
        return 1.0, 0.0
    if ftr == "A":
        return 0.0, 1.0
    return 0.5, 0.5


def competition_weight(competition: object = "") -> float:
    """Return an Elo multiplier based on competition importance.

    Top domestic leagues and World Cup matches move ratings more than routine
    fixtures. Friendlies are intentionally dampened because teams often rotate
    and competitive incentives are weaker.
    """
    text = "" if pd.isna(competition) else str(competition).lower()
    if "friendly" in text or "friendlies" in text:
        return 0.45
    if "world cup" in text or "fifa world" in text:
        return 1.65
    top_terms = (
        "premier league",
        "la liga",
        "serie a",
        "bundesliga",
        "ligue 1",
        "champions league",
        "europa league",
        "copa libertadores",
    )
    if any(term in text for term in top_terms):
        return 1.25
    if re.search(r"\b(final|semi|quarter|knockout|play-?off)\b", text):
        return 1.15
    return 1.0


def dynamic_k_factor(
    home_matches: int,
    away_matches: int,
    competition: object = "",
    base_k: float = DEFAULT_BASE_K,
) -> float:
    """Compute a match-specific K-factor.

    Ratings are more responsive when either team has little history and are
    then gradually stabilized. The result is multiplied by competition weight.
    """
    experience = min(home_matches, away_matches)
    if experience < 10:
        experience_multiplier = 1.35
    elif experience < 30:
        experience_multiplier = 1.15
    elif experience > 120:
        experience_multiplier = 0.85
    else:
        experience_multiplier = 1.0
    return float(base_k * experience_multiplier * competition_weight(competition))


def elo_before_matches(
    df: pd.DataFrame,
    base: float = 1500,
    k: float = DEFAULT_BASE_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> pd.DataFrame:
    """Attach pre-match Elo ratings and update ratings after each result."""
    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}
    rows = []
    for _, match in df.sort_values("Date").iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        home_elo = ratings.get(home, base)
        away_elo = ratings.get(away, base)
        match_k = dynamic_k_factor(
            match_counts.get(home, 0), match_counts.get(away, 0), match.get("Competition", ""), k
        )
        rows.append(
            {
                "HomeElo": home_elo,
                "AwayElo": away_elo,
                "EloDiff": home_elo + home_advantage - away_elo,
                "EloKFactor": match_k,
                "CompetitionWeight": competition_weight(match.get("Competition", "")),
            }
        )
        exp_home = expected_score(home_elo + home_advantage, away_elo)
        score_home, score_away = actual_scores(match["FTR"])
        ratings[home] = home_elo + match_k * (score_home - exp_home)
        ratings[away] = away_elo + match_k * (score_away - (1 - exp_home))
        match_counts[home] = match_counts.get(home, 0) + 1
        match_counts[away] = match_counts.get(away, 0) + 1
    enriched = df.sort_values("Date").copy().reset_index(drop=True)
    return pd.concat([enriched, pd.DataFrame(rows)], axis=1)


def current_elo_ratings(
    df: pd.DataFrame,
    base: float = 1500,
    k: float = DEFAULT_BASE_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> dict[str, float]:
    ratings: dict[str, float] = {}
    match_counts: dict[str, int] = {}
    for _, match in df.sort_values("Date").iterrows():
        home, away = match["HomeTeam"], match["AwayTeam"]
        home_elo = ratings.get(home, base)
        away_elo = ratings.get(away, base)
        match_k = dynamic_k_factor(
            match_counts.get(home, 0), match_counts.get(away, 0), match.get("Competition", ""), k
        )
        exp_home = expected_score(home_elo + home_advantage, away_elo)
        score_home, score_away = actual_scores(match["FTR"])
        ratings[home] = home_elo + match_k * (score_home - exp_home)
        ratings[away] = away_elo + match_k * (score_away - (1 - exp_home))
        match_counts[home] = match_counts.get(home, 0) + 1
        match_counts[away] = match_counts.get(away, 0) + 1
    return ratings
