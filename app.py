"""Streamlit app for Football Predictions analytics."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from src.competitions import competitions_table, load_competitions
from src.data_loader import load_uploaded_files
from scripts.data_sources import normalize_upcoming_frame
from src.features import team_statistics
from src.odds import (
    BOOKMAKERS,
    add_implied_probabilities,
    odds_calibration,
    odds_to_probabilities,
    similar_odds_matches,
)
from src.predictor import (
    feature_importance,
    predict_match,
    similar_historical_matches,
    train_baseline_model,
)
from src.visualization import (
    goals_distribution,
    odds_vs_actual_chart,
    result_distribution,
    team_form_chart,
)

st.set_page_config(page_title="Football Predictions", page_icon="⚽", layout="wide")
st.title("⚽ Football Predictions")
st.caption(
    "A football data analytics and probability estimation dashboard — not a betting app."
)
competitions = load_competitions()
competition_names = [c["name"] for c in competitions]

with st.sidebar:
    st.header("Data")
    selected_competition = st.selectbox(
        "Competition", competition_names, index=0 if competition_names else None
    )
    comp = next(
        (c for c in competitions if c["name"] == selected_competition),
        {"data_source": "football-data.co.uk"},
    )
    if st.button("Update historical data"):
        result = subprocess.run(["python", "scripts/update_historical_data.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Historical data updated.")
            st.code(result.stdout[-2000:])
        else:
            st.error("Historical update failed.")
            st.code(result.stderr or result.stdout)
    if st.button("Update upcoming fixtures"):
        result = subprocess.run(["python", "scripts/update_upcoming_fixtures.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Upcoming fixtures updated.")
            st.code(result.stdout[-2000:])
        else:
            st.error("Upcoming fixtures update failed.")
            st.code(result.stderr or result.stdout)
    uploads = st.file_uploader(
        "Upload match CSV files (manual fallback)", type="csv", accept_multiple_files=True
    )
    bookmaker = st.selectbox(
        "Odds source",
        list(BOOKMAKERS.keys()),
        index=list(BOOKMAKERS.keys()).index("Market Avg"),
    )
    form_window = st.slider("Recent form window", 3, 10, 5)
    page = st.radio(
        "Navigation",
        ["Overview", "Team statistics", "Odds analysis", "Upcoming prediction", "Next 48 Hours Predictions"],
        index=0,
    )

historical_path = Path("data/processed/historical_matches.csv")

try:
    if uploads:
        data = load_uploaded_files(uploads, comp.get("data_source", "football-data.co.uk"))
        data_source_note = "manual upload"
    elif historical_path.exists():
        data = pd.read_csv(historical_path, parse_dates=["Date"])
        data_source_note = str(historical_path)
    else:
        st.info(
            "Upload one or more CSV files or click 'Update historical data' to download configured football-data.co.uk league CSVs."
        )
        st.subheader("Configured competitions")
        st.dataframe(competitions_table(), use_container_width=True, hide_index=True)
        st.stop()
    data = add_implied_probabilities(data, bookmaker)
except Exception as exc:
    st.error(f"Could not load match data: {exc}")
    st.stop()

if data.empty:
    st.warning("No valid historical match rows were found after cleaning.")
    st.stop()

teams = sorted(set(data["HomeTeam"].dropna()) | set(data["AwayTeam"].dropna()))
st.sidebar.success(f"Loaded {len(data):,} matches and {len(teams):,} teams")
st.sidebar.caption(f"Historical data: {data_source_note}")
st.sidebar.caption(
    f"Source: {comp.get('data_source')} · type: {comp.get('match_type')}"
)

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(data):,}")
    c2.metric("Teams", f"{len(teams):,}")
    c3.metric(
        "Seasons span", f"{data['Date'].min().date()} → {data['Date'].max().date()}"
    )
    c4.metric(
        "Rows with selected odds",
        f"{data[['ImpHome','ImpDraw','ImpAway']].dropna().shape[0]:,}",
    )
    st.plotly_chart(result_distribution(data), use_container_width=True)
    st.plotly_chart(goals_distribution(data), use_container_width=True)
    st.subheader("Competition configuration")
    st.dataframe(competitions_table(), use_container_width=True, hide_index=True)
    st.subheader("Cleaned data preview")
    st.dataframe(data.head(100), use_container_width=True)

elif page == "Team statistics":
    stats = team_statistics(data, form_window)
    selected_team = st.selectbox("Team", teams)
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.plotly_chart(
        team_form_chart(data, selected_team, max(form_window, 10)),
        use_container_width=True,
    )

elif page == "Odds analysis":
    calibration = odds_calibration(data)
    if calibration.empty:
        st.warning("The selected odds source does not have enough valid odds data.")
    else:
        st.plotly_chart(odds_vs_actual_chart(calibration), use_container_width=True)
        st.dataframe(
            calibration.style.format(
                {"Average implied probability": "{:.1%}", "Actual frequency": "{:.1%}"}
            ),
            use_container_width=True,
        )
    st.subheader("Odds-only historical context")
    col1, col2, col3, col4 = st.columns(4)
    h_odds = col1.number_input("Home odds", min_value=1.01, value=2.10, step=0.05)
    d_odds = col2.number_input("Draw odds", min_value=1.01, value=3.30, step=0.05)
    a_odds = col3.number_input("Away odds", min_value=1.01, value=3.50, step=0.05)
    tolerance = col4.slider("Similarity tolerance", 0.01, 0.20, 0.05, 0.01)
    probs = odds_to_probabilities(h_odds, d_odds, a_odds)
    matches = similar_odds_matches(data, *probs, tolerance=tolerance)
    st.write(
        f"Normalized implied probabilities: home {probs[0]:.1%}, draw {probs[1]:.1%}, away {probs[2]:.1%}"
    )
    st.dataframe(
        matches[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
                "ImpHome",
                "ImpDraw",
                "ImpAway",
                "OddsDistance",
            ]
        ].head(50),
        use_container_width=True,
    )

elif page == "Upcoming prediction":
    st.subheader("Upcoming match probability estimator")
    st.write(
        "Predictions combine leak-free recent form, goal difference, scoring reliability, venue/rest effects, neutral-site and World Cup context, Elo ratings, calibrated bookmaker context, and similar historical matches."
    )
    model, training_data = train_baseline_model(data)
    if model is None:
        st.warning(
            "At least 30 feature-ready historical matches with valid odds are needed to train the multi-feature baseline model."
        )
        st.dataframe(training_data, use_container_width=True)
        st.stop()
    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home team", teams)
    away_team = c2.selectbox("Away team", [team for team in teams if team != home_team])
    c3, c4, c5 = st.columns(3)
    home_odds = c3.number_input(
        "Current home odds", min_value=1.01, value=2.10, step=0.05
    )
    draw_odds = c4.number_input(
        "Current draw odds", min_value=1.01, value=3.30, step=0.05
    )
    away_odds = c5.number_input(
        "Current away odds", min_value=1.01, value=3.50, step=0.05
    )
    implied = odds_to_probabilities(home_odds, draw_odds, away_odds)
    prediction, feature_row, explanation = predict_match(
        model, data, home_team, away_team, implied
    )
    likely = prediction.loc[prediction["Model probability"].idxmax(), "Outcome"]
    st.metric("Most likely result", likely)
    st.metric("Confidence score", f"{prediction['Confidence score'].iloc[0]:.0%}")
    st.metric("Predicted score estimate", prediction["Estimated score"].iloc[0])
    st.dataframe(
        prediction.style.format(
            {
                "Model probability": "{:.1%}",
                "Bookmaker implied": "{:.1%}",
                "Confidence score": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Why the model made this prediction")
    for note in explanation:
        st.write(f"- {note}")
    st.subheader("Feature importance")
    importances = feature_importance(model, training_data)
    if importances.empty:
        st.info(
            "Feature importance needs more feature-ready matches before it is reliable."
        )
    else:
        st.dataframe(
            importances.style.format({"Importance": "{:.4f}", "Stability": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
    st.subheader("Similar historical matches (multi-feature context)")
    similar = similar_historical_matches(training_data, feature_row)
    st.dataframe(
        similar[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTR",
                "HomePPG5",
                "AwayPPG5",
                "HomeGD5",
                "AwayGD5",
                "NeutralVenue",
                "WorldCupMatch",
                "EloDiff",
                "ImpHome",
                "ImpDraw",
                "ImpAway",
                "SimilarityDistance",
            ]
        ],
        use_container_width=True,
    )
    st.caption(
        f"Training rows used by calibrated model: {len(training_data):,}. This is an analytical probability estimate, not financial advice."
    )

else:
    st.subheader("Next 48 Hours Predictions")
    st.write("Automatically uses `data/processed/historical_matches.csv` and `data/upcoming/upcoming_fixtures.csv`. Fixture odds keep their visible source; football-data.co.uk market averages are preferred, then Bet365.")
    manual_upcoming = st.file_uploader("Manual upcoming fixtures CSV fallback", type="csv", key="manual_upcoming")
    if manual_upcoming is not None:
        manual = normalize_upcoming_frame(pd.read_csv(manual_upcoming, encoding_errors="ignore"))
        Path("data/upcoming").mkdir(parents=True, exist_ok=True)
        manual.to_csv("data/upcoming/upcoming_fixtures.csv", index=False)
        st.success(f"Saved {len(manual):,} manual upcoming fixtures.")
    if st.button("Generate next 48h predictions"):
        result = subprocess.run(["python", "scripts/predict_next_48h.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Predictions generated.")
            st.code(result.stdout[-2000:])
        else:
            st.error("Prediction generation failed.")
            st.code(result.stderr or result.stdout)
    predictions_path = Path("data/predictions/next_48h_predictions.csv")
    if not predictions_path.exists():
        st.info("No generated predictions yet. Update fixtures, then generate next 48h predictions.")
    else:
        predictions = pd.read_csv(predictions_path)
        if predictions.empty:
            st.warning("No fixtures found in the next 48 hours.")
        else:
            st.dataframe(
                predictions.style.format(
                    {
                        "HomeWinProbability": "{:.1%}",
                        "DrawProbability": "{:.1%}",
                        "AwayWinProbability": "{:.1%}",
                        "ConfidenceScore": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            selected = st.selectbox("Prediction detail", predictions.index, format_func=lambda i: f"{predictions.loc[i, 'HomeTeam']} vs {predictions.loc[i, 'AwayTeam']}")
            row = predictions.loc[selected]
            st.metric("Predicted score", row["PredictedScore"])
            st.metric("Confidence", f"{row['ConfidenceScore']:.0%}")
            st.metric("Value signal", row["ValueSignal"])
            st.caption(f"Odds source: {row['OddsSource']}")
            st.write("Model explanation")
            for note in str(row["ModelExplanation"]).split(" | "):
                st.write(f"- {note}")
            st.write("Similar historical matches")
            st.write(row["SimilarHistoricalMatches"])
