"""Streamlit app for Football Predictions analytics."""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from src.competitions import competitions_table, load_competitions
from src.data_loader import load_uploaded_files, safe_read_csv
from scripts.data_sources import UPCOMING_COLUMNS, normalize_upcoming_frame
from scripts.predict_next_48h import PREDICTION_COLUMNS
from src.preprocessing import EXPECTED_COLUMNS
from src.features import team_statistics
from src.fixtures import (
    INTERNATIONAL_UPCOMING,
    NO_INTERNATIONAL_FIXTURES_MESSAGE,
    is_international_competition,
    load_historical_matches_for_competition,
    load_upcoming_fixtures_for_competition,
)
from src.odds import (
    BOOKMAKERS,
    add_implied_probabilities,
    odds_calibration,
    odds_to_probabilities,
    similar_odds_matches,
)
from src.predictor import (
    evaluate_model_by_date,
    feature_importance,
    feature_influence_summary,
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
        update_script = "scripts/update_international_data.py" if is_international_competition(selected_competition) else "scripts/update_historical_data.py"
        result = subprocess.run(["python", update_script], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Historical data updated.")
            st.code(result.stdout[-2000:])
        else:
            st.error("Historical update failed.")
            st.code(result.stderr or result.stdout)
    if st.button("Update upcoming fixtures"):
        command = ["python", "scripts/update_international_fixtures.py"] if is_international_competition(selected_competition) else ["python", "scripts/update_upcoming_fixtures.py"]
        result = subprocess.run(command, capture_output=True, text=True)
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
        ["Overview", "Team statistics", "Odds analysis", "Upcoming prediction", "Backtesting", "Next 48 Hours Predictions"],
        index=0,
    )

historical_path = Path("data/processed/historical_matches.csv")

try:
    if uploads:
        data = load_uploaded_files(uploads, comp.get("data_source", "football-data.co.uk"))
        data_source_note = "manual upload"
        data_warning = None
    else:
        data, data_source_note, data_warning = load_historical_matches_for_competition(selected_competition, bookmaker)
    if data_warning:
        st.warning(data_warning)
    data = add_implied_probabilities(data, bookmaker) if not data.empty else data
except Exception as exc:
    st.error(f"Could not load match data: {exc}")
    data = pd.DataFrame(columns=EXPECTED_COLUMNS)
    data_source_note = "unavailable historical data"

if data.empty and page != "Next 48 Hours Predictions":
    st.warning("Please click Update historical data first or upload CSV files manually.")
    st.subheader("Configured competitions")
    st.dataframe(competitions_table(), use_container_width=True, hide_index=True)
    st.stop()

teams = sorted(set(data.get("HomeTeam", pd.Series(dtype=object)).dropna()) | set(data.get("AwayTeam", pd.Series(dtype=object)).dropna()))
st.sidebar.success(f"Loaded {len(data):,} matches and {len(teams):,} teams")
st.sidebar.caption(f"Historical data: {data_source_note}")
if historical_path.exists():
    st.sidebar.caption(f"Last historical update: {pd.Timestamp(historical_path.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
odds_rows = data[["ImpHome", "ImpDraw", "ImpAway"]].dropna().shape[0] if {"ImpHome", "ImpDraw", "ImpAway"}.issubset(data.columns) else 0
st.sidebar.caption(f"Selected odds source availability: {odds_rows:,} rows")
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
    model, training_data = train_baseline_model(data, selected_competition)
    if not teams:
        st.warning("No teams are available in the loaded dataset. Please update historical data or upload CSV files manually.")
        st.stop()
    if model is None:
        st.warning(
            "At least 30 feature-ready historical matches are needed to train the multi-feature baseline model. Odds are optional; Elo/form/goals/H2H/tournament context still run without them."
        )
        st.dataframe(training_data, use_container_width=True)
        st.stop()
    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home team", teams)
    away_options = [team for team in teams if team != home_team]
    if not away_options:
        st.warning("Selected teams are missing from the dataset or there are not enough teams to compare.")
        st.stop()
    away_team = c2.selectbox("Away team", away_options)
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
        model, data, home_team, away_team, implied, competition=selected_competition
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
    local_influence = feature_influence_summary(model, feature_row)
    if not local_influence.empty:
        st.write("Feature signals for this fixture")
        st.dataframe(local_influence.style.format({"Signal strength": "{:.3f}"}), use_container_width=True, hide_index=True)
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
    st.subheader("Model evaluation")
    metrics, calibration_table = evaluate_model_by_date(data)
    if metrics.empty:
        st.info("Evaluation needs more chronological feature-ready matches.")
    else:
        st.dataframe(metrics.style.format({"Value": "{:.3f}"}), use_container_width=True, hide_index=True)
        st.write("Calibration table")
        st.dataframe(calibration_table.style.format({"AveragePredictedProbability": "{:.1%}", "ActualFrequency": "{:.1%}"}), use_container_width=True, hide_index=True)
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
        f"Training data scope: {'international/national-team only' if is_international_competition(selected_competition) else 'club competition only'} · historical feature rows used: {len(training_data):,} · odds available: {bool(pd.notna([home_odds, draw_odds, away_odds]).all())}. This is an analytical probability estimate, not financial advice."
    )

elif page == "Backtesting":
    st.subheader("Backtesting")
    st.write("Run a chronological historical backtest on `data/processed/historical_matches.csv`. The split is date-based only because football data is time-dependent.")
    c1, c2 = st.columns(2)
    train_until = c1.date_input("Train until", value=pd.Timestamp("2024-06-30").date())
    test_from = c2.date_input("Test from", value=pd.Timestamp("2024-07-01").date())
    if st.button("Run backtest on saved historical data"):
        result = subprocess.run(
            [sys.executable, "scripts/backtest_model.py", "--train-until", str(train_until), "--test-from", str(test_from)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            st.success("Backtest completed. Reports saved in data/reports/.")
            st.code(result.stdout[-4000:])
        else:
            st.error("Backtest failed.")
            st.code(result.stderr or result.stdout)
    reports_dir = Path("data/reports")
    metrics_path = reports_dir / "backtest_metrics.json"
    predictions_path = reports_dir / "backtest_predictions.csv"
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        m2.metric("Log loss", f"{metrics.get('log_loss', 0):.3f}")
        m3.metric("Brier score", f"{metrics.get('brier_score', 0):.3f}")
        m4.metric("Bookmaker favorite accuracy", f"{metrics.get('bookmaker_favorite_accuracy', 0):.1%}")
        st.caption("Lower log loss and Brier score indicate better probabilistic estimates. Accuracy only checks the top pick and does not guarantee future outcomes.")
    calibration_path = reports_dir / "backtest_calibration.csv"
    if calibration_path.exists():
        st.write("Calibration table")
        calibration = pd.read_csv(calibration_path)
        st.dataframe(calibration.style.format({"AveragePredictedProbability": "{:.1%}", "ActualFrequency": "{:.1%}"}), use_container_width=True, hide_index=True)
    matrix_path = reports_dir / "backtest_confusion_matrix.csv"
    if matrix_path.exists():
        st.write("Confusion matrix")
        st.dataframe(pd.read_csv(matrix_path, index_col=0), use_container_width=True)
    comparison_path = reports_dir / "backtest_probability_comparison.csv"
    if comparison_path.exists():
        st.write("Model vs bookmaker probability comparison")
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison.style.format({"AverageModelProbability": "{:.1%}", "AverageBookmakerProbability": "{:.1%}"}), use_container_width=True, hide_index=True)
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path, parse_dates=["Date"])
        failed = predictions[~predictions["CorrectModel"]].sort_values("Date", ascending=False)
        st.write("Recent failed model predictions")
        st.dataframe(failed.head(25), use_container_width=True, hide_index=True)

else:
    st.subheader("Next 48 Hours Predictions")
    if is_international_competition(selected_competition):
        st.write("Uses `data/processed/international_matches.csv` and `data/upcoming/international_fixtures.csv`. FIFA World Cup is filtered from the shared international rows.")
    else:
        st.write("Automatically uses `data/processed/historical_matches.csv` and `data/upcoming/upcoming_fixtures.csv`. Fixture odds are API-first: The Odds API is primary, optional API-Football can provide fallback fixtures, and manual CSV upload remains available.")
    st.info("Manual upcoming fixtures CSV columns: Date, Time, Competition, HomeTeam, AwayTeam, HomeOdds, DrawOdds, AwayOdds, Over25Odds, Under25Odds, OddsSource")
    upcoming_preview, upcoming_path, fixture_warning = load_upcoming_fixtures_for_competition(selected_competition)
    st.caption(f"Upcoming fixtures loaded: {len(upcoming_preview):,}")
    if upcoming_path.exists():
        st.caption(f"Last upcoming update: {pd.Timestamp(upcoming_path.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')}")
    if upcoming_preview.empty:
        st.warning(fixture_warning or "Set ODDS_API_KEY in .env or use manual CSV fallback. Manual columns: Date, Time, Competition, HomeTeam, AwayTeam, HomeOdds, DrawOdds, AwayOdds, Over25Odds, Under25Odds, OddsSource.")
    else:
        odds_available = upcoming_preview[["HomeOdds", "DrawOdds", "AwayOdds"]].notna().all(axis=1).sum()
        st.caption(f"Upcoming odds source availability: {odds_available:,}/{len(upcoming_preview):,} fixtures with full odds")
    manual_upcoming = st.file_uploader("Manual upcoming fixtures CSV fallback", type="csv", key="manual_upcoming")
    if manual_upcoming is not None:
        manual = normalize_upcoming_frame(safe_read_csv(manual_upcoming, UPCOMING_COLUMNS))
        Path("data/upcoming").mkdir(parents=True, exist_ok=True)
        target_path = INTERNATIONAL_UPCOMING if is_international_competition(selected_competition) else Path("data/upcoming/upcoming_fixtures.csv")
        manual.to_csv(target_path, index=False)
        st.success(f"Saved {len(manual):,} manual upcoming fixtures.")
    if st.button("Generate next 48h predictions"):
        command = ["python", "scripts/predict_next_48h.py"]
        if selected_competition:
            command.extend(["--competition", selected_competition])
        result = subprocess.run(command, capture_output=True, text=True)
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
        predictions = safe_read_csv(predictions_path, PREDICTION_COLUMNS)
        if predictions.empty:
            if is_international_competition(selected_competition):
                st.warning(NO_INTERNATIONAL_FIXTURES_MESSAGE)
            else:
                st.warning("No next-48h predictions available. Upcoming fixtures may be missing or no matches are scheduled in the next 48 hours.")
            st.info("Manual fallback: upload a CSV with columns Date, Time, Competition, HomeTeam, AwayTeam, HomeOdds, DrawOdds, AwayOdds, Over25Odds, Under25Odds, OddsSource, then click Generate next 48h predictions.")
        else:
            st.dataframe(
                predictions.style.format(
                    {
                        "HomeWinProbability": "{:.1%}",
                        "DrawProbability": "{:.1%}",
                        "AwayWinProbability": "{:.1%}",
                        "ConfidenceScore": "{:.1%}",
                        "ExpectedHomeGoals": "{:.2f}",
                        "ExpectedAwayGoals": "{:.2f}",
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
            st.write(f"Confidence reason: {row.get('ConfidenceReason', 'Not available')}")
            st.write(f"Top 5 scorelines: {row.get('Top5Scorelines', 'Not available')}")
            st.write(f"Feature importance summary: {row.get('FeatureImportanceSummary', 'Not available')}")
            st.caption(f"Odds source: {row['OddsSource']}")
            st.write("Model explanation")
            for note in str(row["ModelExplanation"]).split(" | "):
                st.write(f"- {note}")
            st.write("Similar historical matches")
            st.write(row["SimilarHistoricalMatches"])
