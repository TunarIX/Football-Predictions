"""Multi-feature baseline football prediction model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_match_features, upcoming_features

FEATURE_COLUMNS = [
    "HomePPG5",
    "HomePPG10",
    "AwayPPG5",
    "AwayPPG10",
    "HomeWPPG3",
    "HomeWPPG5",
    "HomeWPPG10",
    "AwayWPPG3",
    "AwayWPPG5",
    "AwayWPPG10",
    "HomeWGF3",
    "HomeWGF5",
    "HomeWGF10",
    "HomeWGA3",
    "HomeWGA5",
    "HomeWGA10",
    "HomeWGD3",
    "HomeWGD5",
    "HomeWGD10",
    "AwayWGF3",
    "AwayWGF5",
    "AwayWGF10",
    "AwayWGA3",
    "AwayWGA5",
    "AwayWGA10",
    "AwayWGD3",
    "AwayWGD5",
    "AwayWGD10",
    "HomeFormConsistency",
    "AwayFormConsistency",
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
CLASS_LABELS = ["H", "D", "A"]
OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}


def _calibration_cv(y: pd.Series) -> int:
    """Choose a safe CV fold count so every class appears in each calibration fold."""
    min_class_count = int(y.value_counts().min())
    return max(2, min(5, min_class_count))


def _new_base_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=220,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        min_samples_leaf=12,
        random_state=42,
    )


def _new_calibrated_model(y: pd.Series) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    _new_base_model(),
                    cv=_calibration_cv(y),
                    method="sigmoid",
                    ensemble=True,
                ),
            ),
        ]
    )


def train_baseline_model(df: pd.DataFrame) -> tuple[Pipeline | None, pd.DataFrame]:
    """Train a calibrated football model on chronological, pre-match features."""
    dataset = build_match_features(df)
    if len(dataset) < 30 or dataset["FTR"].nunique() < 2 or dataset["FTR"].value_counts().min() < 2:
        return None, dataset

    model = _new_calibrated_model(dataset["FTR"])
    model.fit(dataset[FEATURE_COLUMNS], dataset["FTR"])
    return model, dataset



def feature_importance(
    model: Pipeline | None, training_data: pd.DataFrame, limit: int = 20
) -> pd.DataFrame:
    """Estimate model feature importance with lightweight permutation importance."""
    if model is None or training_data.empty:
        return pd.DataFrame()
    valid = training_data.dropna(subset=FEATURE_COLUMNS + ["FTR"])
    if len(valid) < 30:
        return pd.DataFrame()
    sample = valid.tail(min(len(valid), 700))
    result = permutation_importance(
        model,
        sample[FEATURE_COLUMNS],
        sample["FTR"],
        n_repeats=6,
        random_state=42,
        scoring="neg_log_loss",
    )
    importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": result.importances_mean,
            "Stability": result.importances_std,
        }
    )
    return (
        importance.sort_values("Importance", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def _poisson_pmf(lam: float, max_goals: int = 7) -> np.ndarray:
    values = [np.exp(-lam) * lam**goals / math.factorial(goals) for goals in range(max_goals + 1)]
    return np.array(values, dtype=float)


def poisson_goal_model(features: pd.DataFrame, max_goals: int = 7) -> tuple[float, float, pd.DataFrame]:
    """Estimate expected goals and top scoreline probabilities with a Poisson model."""
    row = features.iloc[0]
    elo_home_edge = np.clip(row["EloDiff"] / 400, -0.8, 0.8)
    home_xg = np.clip(
        0.35
        + 0.36 * row["HomeWGF5"]
        + 0.21 * row["HomeWGF10"]
        + 0.27 * row["AwayWGA5"]
        + 0.11 * row["HomeVenuePPG"]
        + 0.20 * row["ImpHome"]
        + 0.18 * elo_home_edge,
        0.15,
        4.5,
    )
    away_xg = np.clip(
        0.30
        + 0.36 * row["AwayWGF5"]
        + 0.21 * row["AwayWGF10"]
        + 0.27 * row["HomeWGA5"]
        + 0.11 * row["AwayVenuePPG"]
        + 0.20 * row["ImpAway"]
        - 0.18 * elo_home_edge,
        0.15,
        4.5,
    )
    home_pmf = _poisson_pmf(float(home_xg), max_goals)
    away_pmf = _poisson_pmf(float(away_xg), max_goals)
    rows = []
    for hg, hp in enumerate(home_pmf):
        for ag, ap in enumerate(away_pmf):
            rows.append({"Scoreline": f"{hg}-{ag}", "HomeGoals": hg, "AwayGoals": ag, "Probability": float(hp * ap)})
    scores = pd.DataFrame(rows).sort_values("Probability", ascending=False).reset_index(drop=True)
    return round(float(home_xg), 2), round(float(away_xg), 2), scores.head(5)


def feature_influence_summary(model: Pipeline | None, features: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """Lightweight local feature contribution proxy for Streamlit display."""
    if model is None or features.empty:
        return pd.DataFrame()
    row = features.iloc[0][FEATURE_COLUMNS]
    baselines = pd.Series(0.0, index=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        if col.startswith("Imp"):
            baselines[col] = 1 / 3
        elif "Elo" in col:
            baselines[col] = 1500 if col != "EloDiff" else 0
    deltas = (row - baselines).abs().sort_values(ascending=False).head(limit)
    return pd.DataFrame({"Feature": deltas.index, "Signal strength": deltas.values})


def prediction_explanation(
    features: pd.DataFrame, home_team: str, away_team: str
) -> list[str]:
    """Build human-readable notes from the same feature row used by the model."""
    row = features.iloc[0]
    venue = "neutral venue" if row["NeutralVenue"] else "home/away venue"
    return [
        f"Recent form: {home_team} weighted last-3/5/10 PPG {row['HomeWPPG3']:.2f}/{row['HomeWPPG5']:.2f}/{row['HomeWPPG10']:.2f} vs {away_team} {row['AwayWPPG3']:.2f}/{row['AwayWPPG5']:.2f}/{row['AwayWPPG10']:.2f}.",
        f"Goal reliability: scored-rate last 5 is {row['HomeScoredRate5']:.0%} vs {row['AwayScoredRate5']:.0%}; clean-sheet rate {row['HomeCleanSheetRate5']:.0%} vs {row['AwayCleanSheetRate5']:.0%}.",
        f"Venue/rest: {venue}; venue PPG {row['HomeVenuePPG']:.2f} vs {row['AwayVenuePPG']:.2f}; rest days {row['HomeRestDays']:.0f} vs {row['AwayRestDays']:.0f}.",
        f"Head-to-head context: {row['H2HMatches']:.0f} recent meetings, {home_team} averaged {row['HomeH2HPPG']:.2f} points.",
        f"Elo: {home_team} {row['HomeElo']:.0f}, {away_team} {row['AwayElo']:.0f}, venue-adjusted difference {row['EloDiff']:.0f}.",
        f"Match type: World Cup flag {bool(row['WorldCupMatch'])}, knockout/high-pressure flag {bool(row['KnockoutMatch'])}.",
        f"Market context: normalized implied probabilities H/D/A {row['ImpHome']:.1%}/{row['ImpDraw']:.1%}/{row['ImpAway']:.1%}; entropy {row['MarketEntropy']:.2f}.",
    ]


def _confidence_score(probabilities: dict[str, float], row: pd.Series) -> tuple[float, str]:
    probs = np.array([probabilities[label] for label in CLASS_LABELS], dtype=float)
    margin = np.sort(probs)[-1] - np.sort(probs)[-2]
    entropy = -(probs * np.log(np.clip(probs, 1e-9, 1))).sum() / np.log(3)
    experience = min((row["HomeMatches"] + row["AwayMatches"]) / 100, 1.0)
    elo_signal = min(abs(row["EloDiff"]) / 250, 1.0)
    form_consistency = float(np.clip((row["HomeFormConsistency"] + row["AwayFormConsistency"]) / 2, 0, 1))
    market_agreement = float(
        np.clip(1 - np.abs(probs - np.array([row["ImpHome"], row["ImpDraw"], row["ImpAway"]])).mean() * 1.5, 0, 1)
    )
    weak_data_penalty = 0.55 + 0.45 * experience
    score = (0.40 * margin + 0.15 * (1 - entropy) + 0.18 * experience + 0.10 * elo_signal + 0.09 * form_consistency + 0.08 * market_agreement) * weak_data_penalty
    reasons = [
        f"probability gap {margin:.1%}",
        f"data availability {experience:.0%}",
        f"Elo signal {elo_signal:.0%}",
        f"form consistency {form_consistency:.0%}",
        f"odds/model agreement {market_agreement:.0%}",
    ]
    if experience < 0.45:
        reasons.append("capped by limited team history")
    return float(np.clip(score, 0, 0.92)), "; ".join(reasons)


def predict_match(
    model: Pipeline,
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    implied_probs: tuple[float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return prediction table, feature row, and explanation notes for an upcoming match."""
    features = upcoming_features(df, home_team, away_team, implied_probs)
    probabilities = dict.fromkeys(CLASS_LABELS, 0.0)
    for klass, probability in zip(
        model.classes_, model.predict_proba(features[FEATURE_COLUMNS])[0], strict=False
    ):
        probabilities[klass] = float(probability)

    home_xg, away_xg, scorelines = poisson_goal_model(features)
    confidence, confidence_reason = _confidence_score(probabilities, features.iloc[0])
    table = pd.DataFrame(
        [
            {
                "Outcome": OUTCOME_LABELS[label],
                "Model probability": probabilities[label],
                "Bookmaker implied": implied_probs[index],
                "Value signal": probabilities[label] > implied_probs[index],
                "Confidence contribution": (
                    confidence
                    if probabilities[label] == max(probabilities.values())
                    else 0.0
                ),
            }
            for index, label in enumerate(CLASS_LABELS)
        ]
    )
    table["Estimated score"] = f"{home_team} {int(scorelines.iloc[0].HomeGoals)} - {int(scorelines.iloc[0].AwayGoals)} {away_team} (xG {home_xg:.2f}-{away_xg:.2f})"
    table["Expected home goals"] = home_xg
    table["Expected away goals"] = away_xg
    table["Top 5 scorelines"] = "; ".join(f"{r.Scoreline} ({r.Probability:.1%})" for r in scorelines.itertuples())
    table["Confidence score"] = confidence
    table["Confidence reason"] = confidence_reason
    notes = prediction_explanation(features, home_team, away_team)
    notes.append(f"Poisson score model top lines: {table['Top 5 scorelines'].iloc[0]}.")
    notes.append(f"Confidence reason: {confidence_reason}.")
    return table, features, notes



def evaluate_model_by_date(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test evaluation with leakage-safe pre-match features."""
    dataset = build_match_features(df).sort_values("Date").reset_index(drop=True)
    if len(dataset) < 50 or dataset["FTR"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    split = max(1, int(len(dataset) * (1 - test_size)))
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    if test.empty or train["FTR"].value_counts().min() < 2:
        return pd.DataFrame(), pd.DataFrame()
    model = _new_calibrated_model(train["FTR"])
    model.fit(train[FEATURE_COLUMNS], train["FTR"])
    proba = model.predict_proba(test[FEATURE_COLUMNS])
    class_order = list(model.classes_)
    pred = [class_order[i] for i in np.argmax(proba, axis=1)]
    metrics = {"Metric": ["Accuracy", "Log loss", "Brier score"], "Value": [accuracy_score(test["FTR"], pred), log_loss(test["FTR"], proba, labels=class_order), np.mean([brier_score_loss((test["FTR"] == c).astype(int), proba[:, i]) for i, c in enumerate(class_order)])]}
    confidence = proba.max(axis=1)
    correct = (np.array(pred) == test["FTR"].to_numpy()).astype(int)
    calib = pd.DataFrame({"PredictedProbability": confidence, "Correct": correct})
    calib["Bucket"] = pd.cut(calib["PredictedProbability"], bins=np.linspace(0, 1, 6), include_lowest=True)
    table = calib.groupby("Bucket", observed=False).agg(Matches=("Correct", "size"), AveragePredictedProbability=("PredictedProbability", "mean"), ActualFrequency=("Correct", "mean")).reset_index()
    table["Bucket"] = table["Bucket"].astype(str)
    return pd.DataFrame(metrics), table


def similar_historical_matches(
    training_data: pd.DataFrame, features: pd.DataFrame, limit: int = 12
) -> pd.DataFrame:
    """Find historical matches with similar engineered features, not only similar odds."""
    if training_data.empty:
        return pd.DataFrame()
    weights = np.array(
        [
            (
                1.4
                if column
                in {
                    "HomePPG5",
                    "AwayPPG5",
                    "EloDiff",
                    "ImpHome",
                    "ImpDraw",
                    "ImpAway",
                    "NeutralVenue",
                    "WorldCupMatch",
                }
                else 1.0
            )
            for column in FEATURE_COLUMNS
        ]
    )
    hist = training_data.dropna(subset=FEATURE_COLUMNS).copy()
    scale = hist[FEATURE_COLUMNS].std().replace(0, 1)
    target = features.iloc[0][FEATURE_COLUMNS]
    hist["SimilarityDistance"] = (
        (((hist[FEATURE_COLUMNS] - target) / scale) * weights) ** 2
    ).sum(axis=1) ** 0.5
    return hist.sort_values("SimilarityDistance").head(limit)
