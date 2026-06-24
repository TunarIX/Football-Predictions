"""Baseline scikit-learn prediction model."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_match_features, upcoming_features

FEATURE_COLUMNS = ["HomeRecentPPG", "AwayRecentPPG", "HomeGF", "HomeGA", "AwayGF", "AwayGA", "ImpHome", "ImpDraw", "ImpAway"]
CLASS_LABELS = ["H", "D", "A"]


def train_baseline_model(df: pd.DataFrame) -> tuple[Pipeline | None, pd.DataFrame]:
    dataset = build_match_features(df)
    if len(dataset) < 30 or dataset["FTR"].nunique() < 2:
        return None, dataset
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=4, random_state=42, class_weight="balanced")),
    ])
    model.fit(dataset[FEATURE_COLUMNS], dataset["FTR"])
    return model, dataset


def predict_match(model: Pipeline, df: pd.DataFrame, home_team: str, away_team: str, implied_probs: tuple[float, float, float]) -> pd.DataFrame:
    features = upcoming_features(df, home_team, away_team, implied_probs)
    probabilities = dict.fromkeys(CLASS_LABELS, 0.0)
    for klass, prob in zip(model.classes_, model.predict_proba(features[FEATURE_COLUMNS])[0], strict=False):
        probabilities[klass] = float(prob)
    return pd.DataFrame([{
        "Outcome": "Home win", "Model probability": probabilities["H"], "Bookmaker implied": implied_probs[0], "Value signal": probabilities["H"] > implied_probs[0],
    }, {
        "Outcome": "Draw", "Model probability": probabilities["D"], "Bookmaker implied": implied_probs[1], "Value signal": probabilities["D"] > implied_probs[1],
    }, {
        "Outcome": "Away win", "Model probability": probabilities["A"], "Bookmaker implied": implied_probs[2], "Value signal": probabilities["A"] > implied_probs[2],
    }])
