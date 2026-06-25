"""Multi-feature baseline football prediction model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_match_features, upcoming_features

FEATURE_COLUMNS = [
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
CLASS_LABELS = ["H", "D", "A"]
OUTCOME_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}


def _time_series_cv_splits(n_rows: int) -> TimeSeriesSplit | int:
    """Return a chronological calibration strategy that works for small datasets."""
    if n_rows >= 240:
        return TimeSeriesSplit(n_splits=5)
    if n_rows >= 90:
        return TimeSeriesSplit(n_splits=3)
    return 3


def train_baseline_model(df: pd.DataFrame) -> tuple[Pipeline | None, pd.DataFrame]:
    """Train a calibrated football model on chronological, pre-match features."""
    dataset = build_match_features(df)
    if len(dataset) < 30 or dataset["FTR"].nunique() < 2:
        return None, dataset

    base_model = HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=220,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        min_samples_leaf=12,
        random_state=42,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    base_model,
                    cv=_time_series_cv_splits(len(dataset)),
                    method="sigmoid",
                ),
            ),
        ]
    )
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


def _score_estimate(features: pd.DataFrame) -> tuple[float, float]:
    """Estimate expected goals from recent scoring, defending, venue, and Elo context."""
    row = features.iloc[0]
    elo_home_edge = np.clip(row["EloDiff"] / 400, -0.8, 0.8)
    home_goals = max(
        0.1,
        0.42 * row["HomeGF5"]
        + 0.22 * row["HomeGF10"]
        + 0.24 * row["AwayGA5"]
        + 0.12 * row["HomeVenuePPG"]
        + 0.18 * row["ImpHome"]
        + 0.18 * elo_home_edge,
    )
    away_goals = max(
        0.1,
        0.42 * row["AwayGF5"]
        + 0.22 * row["AwayGF10"]
        + 0.24 * row["HomeGA5"]
        + 0.12 * row["AwayVenuePPG"]
        + 0.18 * row["ImpAway"]
        - 0.18 * elo_home_edge,
    )
    return round(float(home_goals), 2), round(float(away_goals), 2)


def prediction_explanation(
    features: pd.DataFrame, home_team: str, away_team: str
) -> list[str]:
    """Build human-readable notes from the same feature row used by the model."""
    row = features.iloc[0]
    venue = "neutral venue" if row["NeutralVenue"] else "home/away venue"
    return [
        f"Recent form: {home_team} last-5 PPG {row['HomePPG5']:.2f} vs {away_team} {row['AwayPPG5']:.2f}; goal difference last 5 {row['HomeGD5']:.2f} vs {row['AwayGD5']:.2f}.",
        f"Goal reliability: scored-rate last 5 is {row['HomeScoredRate5']:.0%} vs {row['AwayScoredRate5']:.0%}; clean-sheet rate {row['HomeCleanSheetRate5']:.0%} vs {row['AwayCleanSheetRate5']:.0%}.",
        f"Venue/rest: {venue}; venue PPG {row['HomeVenuePPG']:.2f} vs {row['AwayVenuePPG']:.2f}; rest days {row['HomeRestDays']:.0f} vs {row['AwayRestDays']:.0f}.",
        f"Head-to-head context: {row['H2HMatches']:.0f} recent meetings, {home_team} averaged {row['HomeH2HPPG']:.2f} points.",
        f"Elo: {home_team} {row['HomeElo']:.0f}, {away_team} {row['AwayElo']:.0f}, venue-adjusted difference {row['EloDiff']:.0f}.",
        f"Match type: World Cup flag {bool(row['WorldCupMatch'])}, knockout/high-pressure flag {bool(row['KnockoutMatch'])}.",
        f"Market context: normalized implied probabilities H/D/A {row['ImpHome']:.1%}/{row['ImpDraw']:.1%}/{row['ImpAway']:.1%}; entropy {row['MarketEntropy']:.2f}.",
    ]


def _confidence_score(probabilities: dict[str, float], row: pd.Series) -> float:
    probs = np.array([probabilities[label] for label in CLASS_LABELS], dtype=float)
    margin = np.sort(probs)[-1] - np.sort(probs)[-2]
    entropy = -(probs * np.log(np.clip(probs, 1e-9, 1))).sum() / np.log(3)
    experience = min((row["HomeMatches"] + row["AwayMatches"]) / 80, 1.0)
    market_agreement = (
        1
        - np.abs(
            probs - np.array([row["ImpHome"], row["ImpDraw"], row["ImpAway"]])
        ).mean()
    )
    return float(
        np.clip(
            0.20
            + 0.45 * margin
            + 0.20 * (1 - entropy)
            + 0.20 * experience
            + 0.15 * market_agreement,
            0,
            1,
        )
    )


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

    home_xg, away_xg = _score_estimate(features)
    confidence = _confidence_score(probabilities, features.iloc[0])
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
    table["Estimated score"] = f"{home_team} {home_xg:.2f} - {away_xg:.2f} {away_team}"
    table["Confidence score"] = confidence
    return table, features, prediction_explanation(features, home_team, away_team)


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
