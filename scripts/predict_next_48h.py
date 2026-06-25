"""Generate detailed predictions for fixtures in the next 48 hours."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import UPCOMING_COLUMNS, normalize_upcoming_frame
from src.data_loader import safe_read_csv
from src.fixtures import (
    load_historical_matches_for_competition,
    load_upcoming_fixtures_for_competition,
)
from src.odds import odds_to_probabilities
from src.preprocessing import EXPECTED_COLUMNS
from src.predictor import feature_influence_summary, predict_match, similar_historical_matches, train_baseline_model

HISTORICAL = Path("data/processed/historical_matches.csv")
UPCOMING = Path("data/upcoming/upcoming_fixtures.csv")
OUTPUT = Path("data/predictions/next_48h_predictions.csv")
PREDICTION_COLUMNS = [
    "FixtureDateTime", "Date", "Time", "Competition", "HomeTeam", "AwayTeam",
    "HomeOdds", "DrawOdds", "AwayOdds", "OddsSource", "HomeWinProbability",
    "DrawProbability", "AwayWinProbability", "PredictedScore", "ExpectedHomeGoals",
    "ExpectedAwayGoals", "Top5Scorelines", "ConfidenceScore", "ConfidenceReason",
    "ValueSignal", "ModelExplanation", "FeatureImportanceSummary", "SimilarHistoricalMatches",
]


def _write_empty_predictions(message: str) -> pd.DataFrame:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame(columns=PREDICTION_COLUMNS)
    empty.to_csv(OUTPUT, index=False)
    print(f"{message} Wrote headers only to {OUTPUT}.")
    return empty


def _fixture_datetime(fixtures: pd.DataFrame) -> pd.Series:
    time = fixtures.get("Time", "").fillna("").astype(str)
    return pd.to_datetime(fixtures["Date"].astype(str) + " " + time, errors="coerce")


def value_signal(model_prob: float, implied: float | None) -> str:
    if pd.isna(implied):
        return "No odds available"
    edge = model_prob - float(implied)
    if edge >= 0.05:
        return "Positive analytical edge"
    if edge <= -0.05:
        return "Below market implied probability"
    return "Close to market"


def generate_next_48h_predictions(now: pd.Timestamp | None = None, competition: str | None = None) -> pd.DataFrame:
    if competition:
        historical, historical_source, history_warning = load_historical_matches_for_competition(competition)
        upcoming, upcoming_source, fixture_warning = load_upcoming_fixtures_for_competition(competition)
    else:
        historical = safe_read_csv(HISTORICAL, EXPECTED_COLUMNS, parse_dates=["Date"])
        historical_source = str(HISTORICAL)
        history_warning = None
        upcoming = normalize_upcoming_frame(safe_read_csv(UPCOMING, UPCOMING_COLUMNS))
        upcoming_source = UPCOMING
        fixture_warning = None
    if historical.empty:
        if history_warning:
            return _write_empty_predictions(history_warning)
        return _write_empty_predictions(
            f"Missing or empty {historical_source}; please update historical data before generating predictions."
        )
    if upcoming.empty:
        if fixture_warning:
            return _write_empty_predictions(fixture_warning)
        return _write_empty_predictions(
            "No upcoming fixtures available; run scripts/update_upcoming_fixtures.py or provide a manual CSV."
        )
    upcoming["FixtureDateTime"] = _fixture_datetime(upcoming)
    now = now or pd.Timestamp.now(tz=None)
    horizon = now + pd.Timedelta(hours=48)
    upcoming = upcoming[(upcoming["FixtureDateTime"] >= now) & (upcoming["FixtureDateTime"] <= horizon)].copy()
    if upcoming.empty:
        return _write_empty_predictions("No matches are scheduled in the next 48 hours.")
    model, training = train_baseline_model(historical, competition)
    if model is None:
        return _write_empty_predictions("Not enough historical rows to train the prediction model.")
    rows: list[dict] = []
    for _, fixture in upcoming.iterrows():
        implied = odds_to_probabilities(fixture.HomeOdds, fixture.DrawOdds, fixture.AwayOdds)
        if any(pd.isna(x) for x in implied):
            implied = (1 / 3, 1 / 3, 1 / 3)
        table, feature_row, explanation = predict_match(
            model,
            historical,
            fixture.HomeTeam,
            fixture.AwayTeam,
            implied,
            competition=fixture.get("Competition", ""),
        )
        similar = similar_historical_matches(training, feature_row, limit=5)
        influence = feature_influence_summary(model, feature_row, limit=5)
        influence_text = "; ".join(f"{r['Feature']}: {r['Signal strength']:.2f}" for _, r in influence.iterrows())
        similar_text = "; ".join(
            f"{r.Date.date() if hasattr(r.Date, 'date') else r.Date} {r.HomeTeam} {r.FTR} {r.AwayTeam}"
            for r in similar.itertuples()
        )
        home_p = float(table.loc[table["Outcome"] == "Home win", "Model probability"].iloc[0])
        draw_p = float(table.loc[table["Outcome"] == "Draw", "Model probability"].iloc[0])
        away_p = float(table.loc[table["Outcome"] == "Away win", "Model probability"].iloc[0])
        best = max((("Home win", home_p, implied[0]), ("Draw", draw_p, implied[1]), ("Away win", away_p, implied[2])), key=lambda x: x[1])
        rows.append(
            {
                "FixtureDateTime": fixture.FixtureDateTime,
                "Date": fixture.Date,
                "Time": fixture.Time,
                "Competition": fixture.Competition,
                "HomeTeam": fixture.HomeTeam,
                "AwayTeam": fixture.AwayTeam,
                "HomeOdds": fixture.HomeOdds,
                "DrawOdds": fixture.DrawOdds,
                "AwayOdds": fixture.AwayOdds,
                "OddsSource": fixture.OddsSource,
                "HomeWinProbability": home_p,
                "DrawProbability": draw_p,
                "AwayWinProbability": away_p,
                "PredictedScore": table["Estimated score"].iloc[0],
                "ExpectedHomeGoals": float(table["Expected home goals"].iloc[0]),
                "ExpectedAwayGoals": float(table["Expected away goals"].iloc[0]),
                "Top5Scorelines": table["Top 5 scorelines"].iloc[0],
                "ConfidenceScore": float(table["Confidence score"].iloc[0]),
                "ConfidenceReason": table["Confidence reason"].iloc[0],
                "ValueSignal": value_signal(best[1], best[2]),
                "ModelExplanation": " | ".join(explanation),
                "FeatureImportanceSummary": influence_text,
                "SimilarHistoricalMatches": similar_text,
            }
        )
    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {len(result):,} predictions to {OUTPUT}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", help="Optional selected competition filter, e.g. FIFA World Cup")
    args = parser.parse_args()
    generate_next_48h_predictions(competition=args.competition)
