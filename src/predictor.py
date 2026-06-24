"""Smarter modular baseline prediction model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURE_COLUMNS, build_match_features, upcoming_features
from .odds import similar_odds_matches, similar_odds_summary

CLASS_LABELS = ["H", "D", "A"]
OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}


def train_baseline_model(df: pd.DataFrame) -> tuple[Pipeline | None, dict[str, RandomForestRegressor] | None, pd.DataFrame]:
    """Train outcome and score estimators from engineered football features."""
    dataset = build_match_features(df)
    if len(dataset) < 50 or dataset["FTR"].nunique() < 2:
        return None, None, dataset
    classifier = Pipeline([
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=0.05, random_state=42)),
    ])
    classifier.fit(dataset[MODEL_FEATURE_COLUMNS], dataset["FTR"])
    regressors = {
        "home_goals": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42),
        "away_goals": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=43),
    }
    regressors["home_goals"].fit(dataset[MODEL_FEATURE_COLUMNS], dataset["FTHG"])
    regressors["away_goals"].fit(dataset[MODEL_FEATURE_COLUMNS], dataset["FTAG"])
    return classifier, regressors, dataset


def probability_table(model: Pipeline, feature_frame: pd.DataFrame, implied_probs: tuple[float, float, float]) -> pd.DataFrame:
    probabilities = dict.fromkeys(CLASS_LABELS, 0.0)
    for klass, prob in zip(model.classes_, model.predict_proba(feature_frame[MODEL_FEATURE_COLUMNS])[0], strict=False):
        probabilities[klass] = float(prob)
    return pd.DataFrame([
        {"Outcome": OUTCOME_LABELS["H"], "Model probability": probabilities["H"], "Bookmaker implied": implied_probs[0], "Value signal": probabilities["H"] > implied_probs[0]},
        {"Outcome": OUTCOME_LABELS["D"], "Model probability": probabilities["D"], "Bookmaker implied": implied_probs[1], "Value signal": probabilities["D"] > implied_probs[1]},
        {"Outcome": OUTCOME_LABELS["A"], "Model probability": probabilities["A"], "Bookmaker implied": implied_probs[2], "Value signal": probabilities["A"] > implied_probs[2]},
    ])


def estimate_confidence(probabilities: pd.DataFrame, feature_frame: pd.DataFrame, similar_summary: dict[str, float | int]) -> float:
    """Estimate confidence from model separation, Elo signal, odds support and similar-match sample size."""
    sorted_probs = probabilities["Model probability"].sort_values(ascending=False).to_numpy()
    separation = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    elo_strength = min(abs(float(feature_frame.iloc[0]["EloDiff"])) / 400, 1.0)
    similar_strength = min(float(similar_summary["matches"]) / 100, 1.0)
    confidence = 0.35 + (0.35 * separation) + (0.2 * elo_strength) + (0.1 * similar_strength)
    return float(np.clip(confidence, 0.05, 0.95))


def predicted_score(regressors: dict[str, RandomForestRegressor], feature_frame: pd.DataFrame) -> tuple[float, float]:
    home_goals = float(regressors["home_goals"].predict(feature_frame[MODEL_FEATURE_COLUMNS])[0])
    away_goals = float(regressors["away_goals"].predict(feature_frame[MODEL_FEATURE_COLUMNS])[0])
    return max(home_goals, 0.0), max(away_goals, 0.0)


def build_explanation(feature_frame: pd.DataFrame, probabilities: pd.DataFrame, similar_summary: dict[str, float | int]) -> list[str]:
    row = feature_frame.iloc[0]
    explanations = [
        f"Recent form signal: home last-5 PPG {row['HomeForm5PPG']:.2f} vs away last-5 PPG {row['AwayForm5PPG']:.2f}; last-10 PPG {row['HomeForm10PPG']:.2f} vs {row['AwayForm10PPG']:.2f}.",
        f"Venue signal: home-team home PPG {row['HomeVenuePPG']:.2f} vs away-team away PPG {row['AwayVenuePPG']:.2f}.",
        f"Goals trend: home GF/GA {row['HomeGFTrend']:.2f}/{row['HomeGATrend']:.2f}; away GF/GA {row['AwayGFTrend']:.2f}/{row['AwayGATrend']:.2f}.",
        f"Head-to-head signal: home win {row['H2HHomeWinRate']:.0%}, draw {row['H2HDrawRate']:.0%}, away win {row['H2HAwayWinRate']:.0%}, goal diff {row['H2HGoalDiff']:.2f}.",
        f"Elo signal: home Elo {row['HomeElo']:.0f}, away Elo {row['AwayElo']:.0f}, adjusted Elo difference {row['EloDiff']:.0f}.",
        f"Market signal is included as one feature group only: implied probabilities are H {row['ImpHome']:.0%}, D {row['ImpDraw']:.0%}, A {row['ImpAway']:.0%}.",
    ]
    if similar_summary["matches"]:
        explanations.append(
            f"Similar historical odds context: {similar_summary['matches']} matches, with outcomes H {similar_summary['home_rate']:.0%}, D {similar_summary['draw_rate']:.0%}, A {similar_summary['away_rate']:.0%}."
        )
    top = probabilities.loc[probabilities["Model probability"].idxmax()]
    explanations.append(f"Most likely result is {top['Outcome']} because it has the highest combined model probability ({top['Model probability']:.1%}).")
    return explanations


def predict_match(model: Pipeline, regressors: dict[str, RandomForestRegressor], df: pd.DataFrame, home_team: str, away_team: str, implied_probs: tuple[float, float, float]) -> dict[str, object]:
    """Predict an upcoming match with probabilities, score estimate, confidence and explanations."""
    features = upcoming_features(df, home_team, away_team, implied_probs)
    probabilities = probability_table(model, features, implied_probs)
    similar = similar_odds_matches(df, *implied_probs, tolerance=0.05)
    summary = similar_odds_summary(similar)
    home_goals, away_goals = predicted_score(regressors, features)
    return {
        "probabilities": probabilities,
        "most_likely": probabilities.loc[probabilities["Model probability"].idxmax(), "Outcome"],
        "predicted_score": (home_goals, away_goals),
        "confidence": estimate_confidence(probabilities, features, summary),
        "explanations": build_explanation(features, probabilities, summary),
        "similar_matches": similar,
        "similar_summary": summary,
        "features": features,
    }
