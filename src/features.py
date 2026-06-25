"""Feature engineering and team statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .elo import current_elo_ratings

RESULT_POINTS = {"W": 3, "D": 1, "L": 0}


def _team_match_rows(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return one team's historical matches with team-centric result and goal columns."""
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


def _window_stats(rows: pd.DataFrame, n: int) -> dict[str, float]:
    """Summarize points, scoring, concession, and reliability over the latest n rows."""
    tail = rows.tail(n)
    if tail.empty:
        return {
            f"PPG{n}": 0.0,
            f"GF{n}": 0.0,
            f"GA{n}": 0.0,
            f"GD{n}": 0.0,
            f"ScoredRate{n}": 0.0,
            f"CleanSheetRate{n}": 0.0,
            f"DrawRate{n}": 0.0,
        }

    points = tail["TeamResult"].map(RESULT_POINTS)
    return {
        f"PPG{n}": float(points.mean()),
        f"GF{n}": float(tail["GF"].mean()),
        f"GA{n}": float(tail["GA"].mean()),
        f"GD{n}": float((tail["GF"] - tail["GA"]).mean()),
        f"ScoredRate{n}": _rate(tail["GF"] > 0, True),
        f"CleanSheetRate{n}": _rate(tail["GA"] == 0, True),
        f"DrawRate{n}": _rate(tail["TeamResult"], "D"),
    }


def recent_form(df: pd.DataFrame, team: str, n: int = 5) -> str:
    """Return a compact W/D/L form string for a team."""
    matches = _team_match_rows(df, team).tail(n)
    return "".join(matches["TeamResult"].tolist()) or "No matches"


def team_statistics(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Build a team-level statistical summary for display pages."""
    teams = sorted(set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna()))
    rows = []

    for team in teams:
        matches = _team_match_rows(df, team)
        if matches.empty:
            continue

        home = df[df["HomeTeam"] == team]
        away = df[df["AwayTeam"] == team]
        rows.append(
            {
                "Team": team,
                "Matches": len(matches),
                "Home win rate": _rate(home["FTR"], "H"),
                "Away win rate": _rate(away["FTR"], "A"),
                "Draw rate": _rate(matches["TeamResult"], "D"),
                "Avg goals scored": float(matches["GF"].mean()),
                "Avg goals conceded": float(matches["GA"].mean()),
                "Recent form": recent_form(df, team, form_window),
                "Recent points/game": float(
                    matches.tail(form_window)["TeamResult"].map(RESULT_POINTS).mean()
                ),
                "Last 5 PPG": _window_stats(matches, 5)["PPG5"],
                "Last 10 PPG": _window_stats(matches, 10)["PPG10"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Recent points/game", "Avg goals scored"], ascending=False
    )


def _rate(series: pd.Series, result: str) -> float:
    """Return the share of a series equal to a result, or zero for empty inputs."""
    return float((series == result).mean()) if len(series) else 0.0


def _as_bool(value: object) -> bool:
    """Parse common CSV boolean values without treating missing values as true."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "neutral"}


def _match_type_flags(competition: object) -> tuple[float, float]:
    """Return World Cup and knockout/high-pressure flags from competition text."""
    text = "" if pd.isna(competition) else str(competition).lower()
    world_cup = float("world cup" in text or "fifa world" in text)
    knockout_terms = (
        "final",
        "semi",
        "quarter",
        "round of",
        "play-off",
        "playoff",
        "knockout",
    )
    knockout = float(any(term in text for term in knockout_terms))
    return world_cup, knockout


def _rest_days(rows: pd.DataFrame, match_date: object) -> float:
    """Days since a team's previous match, capped to limit outlier off-season effects."""
    if rows.empty or pd.isna(match_date):
        return 14.0
    days = (pd.to_datetime(match_date) - rows["Date"].max()).days
    return float(np.clip(days if pd.notna(days) else 14, 0, 30))


def _features_from_history(
    hist: pd.DataFrame,
    home_team: str,
    away_team: str,
    implied: tuple[float, float, float],
    match_date: object | None = None,
    neutral: object = False,
    competition: object = "",
) -> dict[str, float]:
    """Generate the full multi-feature set from matches before the target fixture."""
    home_rows = _team_match_rows(hist, home_team)
    away_rows = _team_match_rows(hist, away_team)
    home_5 = _window_stats(home_rows, 5)
    home_10 = _window_stats(home_rows, 10)
    away_5 = _window_stats(away_rows, 5)
    away_10 = _window_stats(away_rows, 10)

    home_long = _window_stats(home_rows, 25)
    away_long = _window_stats(away_rows, 25)
    home_at_home = hist[hist["HomeTeam"] == home_team].tail(10)
    away_on_road = hist[hist["AwayTeam"] == away_team].tail(10)
    home_home_rows = _team_match_rows(home_at_home, home_team)
    away_away_rows = _team_match_rows(away_on_road, away_team)
    h2h = hist[
        ((hist["HomeTeam"] == home_team) & (hist["AwayTeam"] == away_team))
        | ((hist["HomeTeam"] == away_team) & (hist["AwayTeam"] == home_team))
    ].tail(8)

    home_h2h_points = []
    for _, match in h2h.iterrows():
        if match["FTR"] == "D":
            home_h2h_points.append(1)
        elif (match["HomeTeam"] == home_team and match["FTR"] == "H") or (
            match["AwayTeam"] == home_team and match["FTR"] == "A"
        ):
            home_h2h_points.append(3)
        else:
            home_h2h_points.append(0)

    ratings = current_elo_ratings(hist)
    home_elo = ratings.get(home_team, 1500.0)
    away_elo = ratings.get(away_team, 1500.0)
    neutral_flag = float(_as_bool(neutral))
    home_advantage = 0.0 if neutral_flag else 55.0
    world_cup, knockout = _match_type_flags(competition)
    market = np.array(implied, dtype=float)
    market_entropy = float(
        -(market * np.log(np.clip(market, 1e-9, 1))).sum() / np.log(3)
    )

    return {
        "HomePPG5": home_5["PPG5"],
        "HomePPG10": home_10["PPG10"],
        "AwayPPG5": away_5["PPG5"],
        "AwayPPG10": away_10["PPG10"],
        "HomeGF5": home_5["GF5"],
        "HomeGA5": home_5["GA5"],
        "HomeGF10": home_10["GF10"],
        "HomeGA10": home_10["GA10"],
        "AwayGF5": away_5["GF5"],
        "AwayGA5": away_5["GA5"],
        "AwayGF10": away_10["GF10"],
        "AwayGA10": away_10["GA10"],
        "HomeGD5": home_5["GD5"],
        "HomeGD10": home_10["GD10"],
        "AwayGD5": away_5["GD5"],
        "AwayGD10": away_10["GD10"],
        "HomeLongPPG": home_long["PPG25"],
        "AwayLongPPG": away_long["PPG25"],
        "HomeLongGD": home_long["GD25"],
        "AwayLongGD": away_long["GD25"],
        "HomeVenueWinRate": _rate(home_at_home["FTR"], "H"),
        "AwayVenueWinRate": _rate(away_on_road["FTR"], "A"),
        "HomeVenuePPG": _window_stats(home_home_rows, 10)["PPG10"],
        "AwayVenuePPG": _window_stats(away_away_rows, 10)["PPG10"],
        "HomeScoredRate5": home_5["ScoredRate5"],
        "AwayScoredRate5": away_5["ScoredRate5"],
        "HomeCleanSheetRate5": home_5["CleanSheetRate5"],
        "AwayCleanSheetRate5": away_5["CleanSheetRate5"],
        "HomeDrawRate10": home_10["DrawRate10"],
        "AwayDrawRate10": away_10["DrawRate10"],
        "HomeH2HPPG": float(np.mean(home_h2h_points)) if home_h2h_points else 1.0,
        "H2HMatches": float(len(h2h)),
        "HomeElo": float(home_elo),
        "AwayElo": float(away_elo),
        "EloDiff": float(home_elo + home_advantage - away_elo),
        "HomeMatches": float(len(home_rows)),
        "AwayMatches": float(len(away_rows)),
        "HomeRestDays": _rest_days(home_rows, match_date),
        "AwayRestDays": _rest_days(away_rows, match_date),
        "RestDiff": _rest_days(home_rows, match_date)
        - _rest_days(away_rows, match_date),
        "NeutralVenue": neutral_flag,
        "WorldCupMatch": world_cup,
        "KnockoutMatch": knockout,
        "ImpHome": implied[0],
        "ImpDraw": implied[1],
        "ImpAway": implied[2],
        "MarketEntropy": market_entropy,
    }


def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create chronological model features using only matches before each fixture."""
    feature_columns = [
        "HomePPG5",
        "HomePPG10",
        "AwayPPG5",
        "AwayPPG10",
        "HomeGF5",
        "HomeGA5",
        "HomeGF10",
        "HomeGA10",
        "AwayGF5",
        "AwayGA5",
        "AwayGF10",
        "AwayGA10",
        "HomeGD5",
        "HomeGD10",
        "AwayGD5",
        "AwayGD10",
        "HomeLongPPG",
        "AwayLongPPG",
        "HomeLongGD",
        "AwayLongGD",
        "HomeVenueWinRate",
        "AwayVenueWinRate",
        "HomeVenuePPG",
        "AwayVenuePPG",
        "HomeScoredRate5",
        "AwayScoredRate5",
        "HomeCleanSheetRate5",
        "AwayCleanSheetRate5",
        "HomeDrawRate10",
        "AwayDrawRate10",
        "HomeH2HPPG",
        "H2HMatches",
        "HomeElo",
        "AwayElo",
        "EloDiff",
        "HomeMatches",
        "AwayMatches",
        "HomeRestDays",
        "AwayRestDays",
        "RestDiff",
        "NeutralVenue",
        "WorldCupMatch",
        "KnockoutMatch",
        "ImpHome",
        "ImpDraw",
        "ImpAway",
        "MarketEntropy",
    ]
    rows = []
    history = []

    for _, match in df.sort_values("Date").iterrows():
        hist = pd.DataFrame(history)
        if len(hist) >= 10:
            rows.append(
                {
                    "Date": match["Date"],
                    "HomeTeam": match["HomeTeam"],
                    "AwayTeam": match["AwayTeam"],
                    "FTR": match["FTR"],
                    **_features_from_history(
                        hist,
                        match["HomeTeam"],
                        match["AwayTeam"],
                        (
                            match.get("ImpHome"),
                            match.get("ImpDraw"),
                            match.get("ImpAway"),
                        ),
                        match.get("Date"),
                        match.get("Neutral", False),
                        match.get("Competition", ""),
                    ),
                }
            )
        history.append(match.to_dict())

    if not rows:
        return pd.DataFrame(
            columns=["Date", "HomeTeam", "AwayTeam", "FTR", *feature_columns]
        )

    return pd.DataFrame(rows).dropna(subset=feature_columns).reset_index(drop=True)


def upcoming_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    implied_probs: tuple[float, float, float],
) -> pd.DataFrame:
    """Generate prediction features from the full historical dataframe without an FTR column."""
    return pd.DataFrame(
        [
            _features_from_history(
                df.sort_values("Date"),
                home_team,
                away_team,
                implied_probs,
                pd.Timestamp.today(),
            )
        ]
    )
