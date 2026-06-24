"""Feature engineering and team statistics."""
from __future__ import annotations

import pandas as pd

RESULT_POINTS = {"W": 3, "D": 1, "L": 0}


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


def team_statistics(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Build a team-level statistical summary."""
    teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    rows = []
    for team in teams:
        matches = _team_match_rows(df, team)
        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]
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
            "Recent form": recent_form(df, team, form_window),
            "Recent points/game": matches.tail(form_window)["TeamResult"].map(RESULT_POINTS).mean() if total else 0,
        })
    return pd.DataFrame(rows).sort_values(["Recent points/game", "Avg goals scored"], ascending=False)


def build_match_features(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Create chronological model features from form before each match."""
    rows = []
    history = []
    for _, match in df.sort_values("Date").iterrows():
        hist = pd.DataFrame(history)
        if len(hist) >= 10:
            stats = team_statistics(hist, form_window).set_index("Team")
            home = stats.reindex([match["HomeTeam"]]).fillna(0).iloc[0]
            away = stats.reindex([match["AwayTeam"]]).fillna(0).iloc[0]
            rows.append({
                "Date": match["Date"], "HomeTeam": match["HomeTeam"], "AwayTeam": match["AwayTeam"], "FTR": match["FTR"],
                "HomeRecentPPG": home.get("Recent points/game", 0),
                "AwayRecentPPG": away.get("Recent points/game", 0),
                "HomeGF": home.get("Avg goals scored", 0), "HomeGA": home.get("Avg goals conceded", 0),
                "AwayGF": away.get("Avg goals scored", 0), "AwayGA": away.get("Avg goals conceded", 0),
                "ImpHome": match.get("ImpHome"), "ImpDraw": match.get("ImpDraw"), "ImpAway": match.get("ImpAway"),
            })
        history.append(match.to_dict())
    return pd.DataFrame(rows).dropna()


def upcoming_features(df: pd.DataFrame, home_team: str, away_team: str, implied_probs: tuple[float, float, float], form_window: int = 5) -> pd.DataFrame:
    stats = team_statistics(df, form_window).set_index("Team")
    home = stats.reindex([home_team]).fillna(0).iloc[0]
    away = stats.reindex([away_team]).fillna(0).iloc[0]
    return pd.DataFrame([{
        "HomeRecentPPG": home.get("Recent points/game", 0), "AwayRecentPPG": away.get("Recent points/game", 0),
        "HomeGF": home.get("Avg goals scored", 0), "HomeGA": home.get("Avg goals conceded", 0),
        "AwayGF": away.get("Avg goals scored", 0), "AwayGA": away.get("Avg goals conceded", 0),
        "ImpHome": implied_probs[0], "ImpDraw": implied_probs[1], "ImpAway": implied_probs[2],
    }])
