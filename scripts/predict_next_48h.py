"""Generate detailed predictions for fixtures in the next 48 hours."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import normalize_upcoming_frame
from src.odds import odds_to_probabilities
from src.predictor import predict_match, similar_historical_matches, train_baseline_model

HISTORICAL = Path("data/processed/historical_matches.csv")
UPCOMING = Path("data/upcoming/upcoming_fixtures.csv")
OUTPUT = Path("data/predictions/next_48h_predictions.csv")


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


def generate_next_48h_predictions(now: pd.Timestamp | None = None) -> pd.DataFrame:
    if not HISTORICAL.exists():
        raise SystemExit(f"Missing {HISTORICAL}; run scripts/update_historical_data.py first.")
    if not UPCOMING.exists():
        raise SystemExit(f"Missing {UPCOMING}; run scripts/update_upcoming_fixtures.py first.")
    historical = pd.read_csv(HISTORICAL, parse_dates=["Date"])
    upcoming = normalize_upcoming_frame(pd.read_csv(UPCOMING, encoding_errors="ignore"))
    if upcoming.empty:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame()
        empty.to_csv(OUTPUT, index=False)
        return empty
    upcoming["FixtureDateTime"] = _fixture_datetime(upcoming)
    now = now or pd.Timestamp.now(tz=None)
    horizon = now + pd.Timedelta(hours=48)
    upcoming = upcoming[(upcoming["FixtureDateTime"] >= now) & (upcoming["FixtureDateTime"] <= horizon)].copy()
    model, training = train_baseline_model(historical)
    if model is None:
        raise SystemExit("Not enough historical rows with odds to train the prediction model.")
    rows: list[dict] = []
    for _, fixture in upcoming.iterrows():
        implied = odds_to_probabilities(fixture.HomeOdds, fixture.DrawOdds, fixture.AwayOdds)
        if any(pd.isna(x) for x in implied):
            implied = (1 / 3, 1 / 3, 1 / 3)
        table, feature_row, explanation = predict_match(model, historical, fixture.HomeTeam, fixture.AwayTeam, implied)
        similar = similar_historical_matches(training, feature_row, limit=5)
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
                "ConfidenceScore": float(table["Confidence score"].iloc[0]),
                "ValueSignal": value_signal(best[1], best[2]),
                "ModelExplanation": " | ".join(explanation),
                "SimilarHistoricalMatches": similar_text,
            }
        )
    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {len(result):,} predictions to {OUTPUT}")
    return result


if __name__ == "__main__":
    generate_next_48h_predictions()
