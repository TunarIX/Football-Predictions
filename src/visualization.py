"""Plotly chart builders for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def goals_distribution(df: pd.DataFrame):
    totals = (df["FTHG"] + df["FTAG"]).rename("Total goals")
    return px.histogram(
        totals, x="Total goals", nbins=12, title="Total goals distribution"
    )


def result_distribution(df: pd.DataFrame):
    labels = {"H": "Home win", "D": "Draw", "A": "Away win"}
    counts = df["FTR"].map(labels).value_counts().reset_index()
    counts.columns = ["Result", "Matches"]
    return px.bar(
        counts, x="Result", y="Matches", title="Full-time result distribution"
    )


def team_form_chart(df: pd.DataFrame, team: str, window: int = 10):
    from .features import _team_match_rows, RESULT_POINTS

    rows = _team_match_rows(df, team).tail(window).copy()
    rows["Points"] = rows["TeamResult"].map(RESULT_POINTS)
    rows["Opponent"] = rows.apply(
        lambda r: r["AwayTeam"] if r["Venue"] == "Home" else r["HomeTeam"], axis=1
    )
    return px.line(
        rows,
        x="Date",
        y="Points",
        markers=True,
        hover_data=["Opponent", "Venue", "GF", "GA"],
        title=f"Recent form: {team}",
    )


def odds_vs_actual_chart(calibration: pd.DataFrame):
    melted = calibration.melt(
        id_vars="Outcome",
        value_vars=["Average implied probability", "Actual frequency"],
        var_name="Metric",
        value_name="Probability",
    )
    return px.bar(
        melted,
        x="Outcome",
        y="Probability",
        color="Metric",
        barmode="group",
        range_y=[0, 1],
        title="Bookmaker implied probabilities vs actual outcomes",
    )
