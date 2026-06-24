"""Streamlit app for Football Predictions analytics."""
from __future__ import annotations

import numpy as np
import streamlit as st

from src.data_loader import load_uploaded_files
from src.features import team_statistics
from src.odds import BOOKMAKERS, add_implied_probabilities, odds_calibration, odds_to_probabilities, similar_odds_matches
from src.predictor import predict_match, train_baseline_model
from src.visualization import goals_distribution, odds_vs_actual_chart, result_distribution, team_form_chart

st.set_page_config(page_title="Football Predictions", page_icon="⚽", layout="wide")
st.title("⚽ Football Predictions")
st.caption("A football data analytics and probability estimation dashboard — not a betting app.")

with st.sidebar:
    st.header("Data")
    uploads = st.file_uploader("Upload football-data.co.uk CSV files", type="csv", accept_multiple_files=True)
    bookmaker = st.selectbox("Odds source", list(BOOKMAKERS.keys()), index=list(BOOKMAKERS.keys()).index("Market Avg"))
    form_window = st.slider("Recent form window", 3, 10, 5)
    page = st.radio("Navigation", ["Overview", "Team statistics", "Odds analysis", "Predict match"], index=0)

if not uploads:
    st.info("Upload one or more football-data.co.uk CSV files to begin.")
    st.markdown(
        """
        **Expected columns include:** Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, half-time result columns,
        and common bookmaker odds columns such as B365H/B365D/B365A, PSH/PSD/PSA, MaxH/MaxD/MaxA, AvgH/AvgD/AvgA.
        """
    )
    st.stop()

try:
    data = load_uploaded_files(uploads)
    data = add_implied_probabilities(data, bookmaker)
except Exception as exc:  # Streamlit boundary for user-uploaded data errors.
    st.error(f"Could not load the uploaded CSV files: {exc}")
    st.stop()

if data.empty:
    st.warning("No valid historical match rows were found after cleaning.")
    st.stop()

teams = sorted(set(data["HomeTeam"].dropna()) | set(data["AwayTeam"].dropna()))
st.sidebar.success(f"Loaded {len(data):,} matches and {len(teams):,} teams")

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(data):,}")
    c2.metric("Teams", f"{len(teams):,}")
    c3.metric("Seasons span", f"{data['Date'].min().date()} → {data['Date'].max().date()}")
    c4.metric("Rows with selected odds", f"{data[['ImpHome','ImpDraw','ImpAway']].dropna().shape[0]:,}")
    st.plotly_chart(result_distribution(data), use_container_width=True)
    st.plotly_chart(goals_distribution(data), use_container_width=True)
    st.subheader("Cleaned data preview")
    st.dataframe(data.head(100), use_container_width=True)

elif page == "Team statistics":
    stats = team_statistics(data, form_window)
    selected_team = st.selectbox("Team", teams)
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.plotly_chart(team_form_chart(data, selected_team, max(form_window, 10)), use_container_width=True)

elif page == "Odds analysis":
    calibration = odds_calibration(data)
    if calibration.empty:
        st.warning("The selected odds source does not have enough valid odds data.")
    else:
        st.plotly_chart(odds_vs_actual_chart(calibration), use_container_width=True)
        st.dataframe(calibration.style.format({"Average implied probability": "{:.1%}", "Actual frequency": "{:.1%}"}), use_container_width=True)

    st.subheader("Similar historical odds profiles")
    col1, col2, col3, col4 = st.columns(4)
    h_odds = col1.number_input("Home odds", min_value=1.01, value=2.10, step=0.05)
    d_odds = col2.number_input("Draw odds", min_value=1.01, value=3.30, step=0.05)
    a_odds = col3.number_input("Away odds", min_value=1.01, value=3.50, step=0.05)
    tolerance = col4.slider("Similarity tolerance", 0.01, 0.20, 0.05, 0.01)
    probs = odds_to_probabilities(h_odds, d_odds, a_odds)
    matches = similar_odds_matches(data, *probs, tolerance=tolerance)
    st.write(f"Normalized implied probabilities: home {probs[0]:.1%}, draw {probs[1]:.1%}, away {probs[2]:.1%}")
    if matches.empty:
        st.info("No historical matches found in that odds range.")
    else:
        st.write(matches["FTR"].value_counts(normalize=True).rename(index={"H": "Home", "D": "Draw", "A": "Away"}))
        st.dataframe(matches[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "ImpHome", "ImpDraw", "ImpAway", "OddsDistance"]].head(50), use_container_width=True)

else:
    st.subheader("Baseline probability estimator")
    st.write("The first version uses recent team form, home advantage proxies, goals scored/conceded, and bookmaker implied probabilities.")
    model, training_data = train_baseline_model(data)
    if model is None:
        st.warning("At least 30 feature-ready historical matches with valid odds are needed to train the baseline model.")
        st.dataframe(training_data, use_container_width=True)
        st.stop()

    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home team", teams)
    away_team = c2.selectbox("Away team", [team for team in teams if team != home_team])
    c3, c4, c5 = st.columns(3)
    home_odds = c3.number_input("Current home odds", min_value=1.01, value=2.10, step=0.05)
    draw_odds = c4.number_input("Current draw odds", min_value=1.01, value=3.30, step=0.05)
    away_odds = c5.number_input("Current away odds", min_value=1.01, value=3.50, step=0.05)
    implied = odds_to_probabilities(home_odds, draw_odds, away_odds)
    prediction = predict_match(model, data, home_team, away_team, implied)
    likely = prediction.loc[prediction["Model probability"].idxmax(), "Outcome"]

    st.metric("Most likely result", likely)
    st.dataframe(prediction.style.format({"Model probability": "{:.1%}", "Bookmaker implied": "{:.1%}"}), use_container_width=True, hide_index=True)
    if prediction["Value signal"].any():
        st.info("Value signal means the model probability is higher than the selected bookmaker implied probability. Treat it as an analytical flag, not financial advice.")
    st.caption(f"Training rows used by baseline model: {len(training_data):,}")
