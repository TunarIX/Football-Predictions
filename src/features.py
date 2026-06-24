"""Feature engineering, team statistics, form, home/away and head-to-head metrics."""
from __future__ import annotations

import pandas as pd

from .elo import DEFAULT_ELO, HOME_ADVANTAGE, current_elos, expected_score

RESULT_POINTS = {"W": 3, "D": 1, "L": 0}
MODEL_FEATURE_COLUMNS = [
    "HomeForm5PPG", "AwayForm5PPG", "HomeForm10PPG", "AwayForm10PPG",
    "HomeVenuePPG", "AwayVenuePPG", "HomeGFTrend", "HomeGATrend", "AwayGFTrend", "AwayGATrend",
    "H2HHomeWinRate", "H2HDrawRate", "H2HAwayWinRate", "H2HGoalDiff",
    "HomeElo", "AwayElo", "EloDiff", "HomeEloExpected", "ImpHome", "ImpDraw", "ImpAway",
]


def _team_match_rows(df: pd.DataFrame, team: str) -> pd.DataFrame:
    home = df[df["HomeTeam"] == team].copy()
    home["Venue"] = "Home"
    home["GF"] = home["FTHG"]
    home["GA"] = home["FTAG"]
    home["TeamResult"] = home["FTR"].map({"H": "W", "D": "D", "A": "L"})
    away = df[df["AwayTeam"] == team].copy()
    away["Venue"] = "Away"
    away["GF"] = away["FTAG"]
    away["GA"] = away["FTHG"]
    away["TeamResult"] = away["FTR"].map({"H": "L", "D": "D", "A": "W"})
    return pd.concat([home, away], ignore_index=True).sort_values("Date")


def recent_form(df: pd.DataFrame, team: str, n: int = 5) -> str:
    matches = _team_match_rows(df, team).tail(n)
    return "".join(matches["TeamResult"].tolist()) or "No matches"


def _ppg(rows: pd.DataFrame) -> float:
    return float(rows["TeamResult"].map(RESULT_POINTS).mean()) if len(rows) else 0.0


def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    value = series.mean()
    return float(default if pd.isna(value) else value)


def team_statistics(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Build a team-level statistical summary."""
    teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    ratings = current_elos(df)
    rows = []
    for team in teams:
        matches = _team_match_rows(df, team)
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]
        venue_home = matches[matches["Venue"] == "Home"]
        venue_away = matches[matches["Venue"] == "Away"]
        total = len(matches)
        if total == 0:
            continue
        rows.append({
            "Team": team,
            "Matches": total,
            "Home win rate": (home["FTR"] == "H").mean() if len(home) else 0,
            "Away win rate": (away["FTR"] == "A").mean() if len(away) else 0,
            "Draw rate": (matches["TeamResult"] == "D").mean(),
            "Avg goals scored": matches["GF"].mean(),
            "Avg goals conceded": matches["GA"].mean(),
            "Home PPG": _ppg(venue_home),
            "Away PPG": _ppg(venue_away),
            "Form 5": recent_form(df, team, 5),
            "Form 10": recent_form(df, team, 10),
            "Recent form": recent_form(df, team, form_window),
            "Recent points/game": _ppg(matches.tail(form_window)),
            "Elo": ratings.get(team, DEFAULT_ELO),
        })
    return pd.DataFrame(rows).sort_values(["Elo", "Recent points/game", "Avg goals scored"], ascending=False)


def team_feature_snapshot(history: pd.DataFrame, team: str, venue: str) -> dict[str, float]:
    matches = _team_match_rows(history, team)
    venue_matches = matches[matches["Venue"] == venue]
    return {
        "Form5PPG": _ppg(matches.tail(5)),
        "Form10PPG": _ppg(matches.tail(10)),
        "VenuePPG": _ppg(venue_matches.tail(10)),
        "GFTrend": _safe_mean(matches.tail(5)["GF"], _safe_mean(matches["GF"])),
        "GATrend": _safe_mean(matches.tail(5)["GA"], _safe_mean(matches["GA"])),
    }


def head_to_head_features(history: pd.DataFrame, home_team: str, away_team: str, window: int = 10) -> dict[str, float]:
    h2h = history[
        ((history["HomeTeam"] == home_team) & (history["AwayTeam"] == away_team))
        | ((history["HomeTeam"] == away_team) & (history["AwayTeam"] == home_team))
    ].tail(window)
    if h2h.empty:
        return {"H2HHomeWinRate": 0.0, "H2HDrawRate": 0.0, "H2HAwayWinRate": 0.0, "H2HGoalDiff": 0.0}
    home_wins = ((h2h["HomeTeam"] == home_team) & (h2h["FTR"] == "H")) | ((h2h["AwayTeam"] == home_team) & (h2h["FTR"] == "A"))
    away_wins = ((h2h["HomeTeam"] == away_team) & (h2h["FTR"] == "H")) | ((h2h["AwayTeam"] == away_team) & (h2h["FTR"] == "A"))
    goal_diff = h2h.apply(lambda r: (r["FTHG"] - r["FTAG"]) if r["HomeTeam"] == home_team else (r["FTAG"] - r["FTHG"]), axis=1)
    return {
        "H2HHomeWinRate": float(home_wins.mean()),
        "H2HDrawRate": float((h2h["FTR"] == "D").mean()),
        "H2HAwayWinRate": float(away_wins.mean()),
        "H2HGoalDiff": _safe_mean(goal_diff),
    }


def feature_row(history: pd.DataFrame, home_team: str, away_team: str, implied_probs: tuple[float, float, float]) -> dict[str, float]:
    home = team_feature_snapshot(history, home_team, "Home")
    away = team_feature_snapshot(history, away_team, "Away")
    ratings = current_elos(history)
    home_elo = ratings.get(home_team, DEFAULT_ELO)
    away_elo = ratings.get(away_team, DEFAULT_ELO)
    row = {
        "HomeForm5PPG": home["Form5PPG"], "AwayForm5PPG": away["Form5PPG"],
        "HomeForm10PPG": home["Form10PPG"], "AwayForm10PPG": away["Form10PPG"],
        "HomeVenuePPG": home["VenuePPG"], "AwayVenuePPG": away["VenuePPG"],
        "HomeGFTrend": home["GFTrend"], "HomeGATrend": home["GATrend"],
        "AwayGFTrend": away["GFTrend"], "AwayGATrend": away["GATrend"],
        "HomeElo": home_elo, "AwayElo": away_elo,
        "EloDiff": (home_elo + HOME_ADVANTAGE) - away_elo,
        "HomeEloExpected": expected_score(home_elo + HOME_ADVANTAGE, away_elo),
        "ImpHome": implied_probs[0], "ImpDraw": implied_probs[1], "ImpAway": implied_probs[2],
    }
    row.update(head_to_head_features(history, home_team, away_team))
    return row


def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create chronological model features from information available before each match."""
    rows = []
    history = []
    for _, match in df.sort_values("Date").iterrows():
        hist = pd.DataFrame(history)
        if len(hist) >= 20:
            implied = (match.get("ImpHome"), match.get("ImpDraw"), match.get("ImpAway"))
            row = feature_row(hist, match["HomeTeam"], match["AwayTeam"], implied)
            row.update({"Date": match["Date"], "HomeTeam": match["HomeTeam"], "AwayTeam": match["AwayTeam"], "FTR": match["FTR"], "FTHG": match["FTHG"], "FTAG": match["FTAG"]})
            rows.append(row)
        history.append(match.to_dict())
    return pd.DataFrame(rows).dropna(subset=MODEL_FEATURE_COLUMNS + ["FTR"])


def upcoming_features(df: pd.DataFrame, home_team: str, away_team: str, implied_probs: tuple[float, float, float]) -> pd.DataFrame:
    return pd.DataFrame([feature_row(df, home_team, away_team, implied_probs)])
