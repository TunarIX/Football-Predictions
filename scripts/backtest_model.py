"""Chronological backtest for the football prediction model on historical data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import build_match_features
from src.odds import BOOKMAKERS, odds_to_probabilities
from src.predictor import CLASS_LABELS, FEATURE_COLUMNS, _new_calibrated_model

DEFAULT_INPUT = ROOT / "data/processed/historical_matches.csv"
DEFAULT_REPORT_DIR = ROOT / "data/reports"
PREDICTION_COLUMNS = [
    "Date",
    "Competition",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "ModelHome",
    "ModelDraw",
    "ModelAway",
    "BookmakerHome",
    "BookmakerDraw",
    "BookmakerAway",
    "ModelPick",
    "BookmakerPick",
    "CorrectModel",
    "CorrectBookmaker",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a chronological model backtest.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Historical CSV path.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Output folder.")
    parser.add_argument("--train-until", required=True, help="Last training date, e.g. 2024-06-30.")
    parser.add_argument("--test-from", required=True, help="First test date, e.g. 2024-07-01.")
    return parser.parse_args()


def _bookmaker_probabilities(row: pd.Series) -> tuple[float, float, float, str]:
    """Return the first valid normalized implied probability tuple by market priority."""
    for name in ("Market Avg", "Bet365", "Pinnacle", "Market Max", "Bet&Win", "Interwetten"):
        h_col, d_col, a_col = BOOKMAKERS[name]
        if {h_col, d_col, a_col}.issubset(row.index):
            probs = odds_to_probabilities(row.get(h_col), row.get(d_col), row.get(a_col))
            if np.all(np.isfinite(probs)):
                return (*probs, name)
    return (np.nan, np.nan, np.nan, "Unavailable")


def _load_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Historical data not found: {path}")
    data = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "HomeTeam", "AwayTeam", "FTR"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Historical data is missing required columns: {sorted(missing)}")
    data = data[data["FTR"].isin(CLASS_LABELS)].copy()
    data = data.sort_values("Date").reset_index(drop=True)
    bookmaker = data.apply(_bookmaker_probabilities, axis=1, result_type="expand")
    bookmaker.columns = ["BookmakerHome", "BookmakerDraw", "BookmakerAway", "BookmakerSource"]
    return pd.concat([data, bookmaker], axis=1)


def _features_with_metadata(data: pd.DataFrame) -> pd.DataFrame:
    feature_input = data.copy()
    feature_input["ImpHome"] = feature_input["BookmakerHome"]
    feature_input["ImpDraw"] = feature_input["BookmakerDraw"]
    feature_input["ImpAway"] = feature_input["BookmakerAway"]
    dataset = build_match_features(feature_input)
    metadata_cols = [
        "Date",
        "Competition",
        "HomeTeam",
        "AwayTeam",
        "FTR",
        "BookmakerHome",
        "BookmakerDraw",
        "BookmakerAway",
        "BookmakerSource",
    ]
    available = [col for col in metadata_cols if col in data.columns]
    metadata = data[available].copy()
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    metadata["Date"] = pd.to_datetime(metadata["Date"])
    return dataset.merge(metadata, on=["Date", "HomeTeam", "AwayTeam", "FTR"], how="left")


def _aligned_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features[FEATURE_COLUMNS])
    aligned = np.zeros((len(features), len(CLASS_LABELS)), dtype=float)
    for idx, klass in enumerate(model.classes_):
        aligned[:, CLASS_LABELS.index(klass)] = raw[:, idx]
    return aligned


def _calibration_table(confidence: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    table = pd.DataFrame({"PredictedProbability": confidence, "Correct": correct})
    table["Bucket"] = pd.cut(table["PredictedProbability"], bins=np.linspace(0, 1, 6), include_lowest=True)
    grouped = table.groupby("Bucket", observed=False).agg(
        Matches=("Correct", "size"),
        AveragePredictedProbability=("PredictedProbability", "mean"),
        ActualFrequency=("Correct", "mean"),
    ).reset_index()
    grouped["Bucket"] = grouped["Bucket"].astype(str)
    return grouped


def run_backtest(input_path: Path, reports_dir: Path, train_until: str, test_from: str) -> dict[str, object]:
    data = _load_matches(input_path)
    dataset = _features_with_metadata(data).sort_values("Date").reset_index(drop=True)
    train_cutoff = pd.to_datetime(train_until)
    test_start = pd.to_datetime(test_from)
    train = dataset[dataset["Date"] <= train_cutoff].copy()
    test = dataset[dataset["Date"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    if train["FTR"].nunique() < 2 or train["FTR"].value_counts().min() < 2:
        raise ValueError("Training split needs at least two examples for every observed class.")

    model = _new_calibrated_model(train["FTR"])
    model.fit(train[FEATURE_COLUMNS], train["FTR"])
    proba = _aligned_probabilities(model, test)
    model_pick = np.array(CLASS_LABELS)[np.argmax(proba, axis=1)]
    bookmaker_proba = test[["BookmakerHome", "BookmakerDraw", "BookmakerAway"]].to_numpy(dtype=float)
    bookmaker_pick = np.array(CLASS_LABELS)[np.nanargmax(bookmaker_proba, axis=1)]
    y_true = test["FTR"].to_numpy()
    correct_model = model_pick == y_true
    correct_bookmaker = bookmaker_pick == y_true

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_until": str(train_cutoff.date()),
        "test_from": str(test_start.date()),
        "accuracy": float(accuracy_score(y_true, model_pick)),
        "log_loss": float(log_loss(y_true, proba, labels=CLASS_LABELS)),
        "brier_score": float(np.mean([brier_score_loss((y_true == c).astype(int), proba[:, i]) for i, c in enumerate(CLASS_LABELS)])),
        "bookmaker_favorite_accuracy": float(accuracy_score(y_true, bookmaker_pick)),
        "model_accuracy_edge": float(accuracy_score(y_true, model_pick) - accuracy_score(y_true, bookmaker_pick)),
    }

    predictions = test[["Date", "Competition", "HomeTeam", "AwayTeam", "FTR"]].copy()
    predictions["ModelHome"], predictions["ModelDraw"], predictions["ModelAway"] = proba[:, 0], proba[:, 1], proba[:, 2]
    predictions["BookmakerHome"] = test["BookmakerHome"].to_numpy()
    predictions["BookmakerDraw"] = test["BookmakerDraw"].to_numpy()
    predictions["BookmakerAway"] = test["BookmakerAway"].to_numpy()
    predictions["ModelPick"] = model_pick
    predictions["BookmakerPick"] = bookmaker_pick
    predictions["CorrectModel"] = correct_model
    predictions["CorrectBookmaker"] = correct_bookmaker
    predictions = predictions[PREDICTION_COLUMNS]

    calibration = _calibration_table(proba.max(axis=1), correct_model.astype(int))
    matrix = pd.DataFrame(confusion_matrix(y_true, model_pick, labels=CLASS_LABELS), index=[f"Actual {c}" for c in CLASS_LABELS], columns=[f"Predicted {c}" for c in CLASS_LABELS])
    comparison = pd.DataFrame({"Outcome": ["Home", "Draw", "Away"], "AverageModelProbability": proba.mean(axis=0), "AverageBookmakerProbability": np.nanmean(bookmaker_proba, axis=0)})

    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(reports_dir / "backtest_predictions.csv", index=False)
    calibration.to_csv(reports_dir / "backtest_calibration.csv", index=False)
    matrix.to_csv(reports_dir / "backtest_confusion_matrix.csv")
    comparison.to_csv(reports_dir / "backtest_probability_comparison.csv", index=False)
    (reports_dir / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"metrics": metrics, "calibration": calibration, "confusion_matrix": matrix, "comparison": comparison}


def main() -> None:
    args = parse_args()
    result = run_backtest(args.input, args.reports_dir, args.train_until, args.test_from)
    print(json.dumps(result["metrics"], indent=2))
    print("\nCalibration table")
    print(result["calibration"].to_string(index=False))
    print("\nConfusion matrix")
    print(result["confusion_matrix"].to_string())
    print("\nModel vs bookmaker probability comparison")
    print(result["comparison"].to_string(index=False))
    print(f"\nSaved reports to {args.reports_dir}")


if __name__ == "__main__":
    main()
