"""Synthetic smoke test for football prediction quality helpers."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.elo import current_elo_ratings, dynamic_k_factor
from src.features import build_match_features
from src.odds import add_implied_probabilities
from src.predictor import evaluate_model_by_date, poisson_goal_model, train_baseline_model


def synthetic_matches(n: int = 90) -> pd.DataFrame:
    teams = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    rows = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 2) % len(teams)]
        hg = 2 if i % 4 in {0, 1} else 1
        ag = 0 if i % 5 == 0 else (2 if i % 7 == 0 else 1)
        rows.append(
            {
                "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                "Competition": "Premier League" if i % 9 else "International Friendly",
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": hg,
                "FTAG": ag,
                "FTR": "H" if hg > ag else ("A" if ag > hg else "D"),
                "B365H": 2.1,
                "B365D": 3.3,
                "B365A": 3.4,
            }
        )
    return add_implied_probabilities(pd.DataFrame(rows), "Bet365")


def main() -> None:
    data = synthetic_matches()
    assert dynamic_k_factor(2, 2, "International Friendly") < dynamic_k_factor(2, 2, "FIFA World Cup")
    assert current_elo_ratings(data)
    features = build_match_features(data)
    assert {"HomeWPPG3", "AwayWGF5", "HomeFormConsistency"}.issubset(features.columns)
    model, training = train_baseline_model(data)
    assert model is not None and not training.empty
    xh, xa, scorelines = poisson_goal_model(training.tail(1))
    assert xh > 0 and xa > 0 and len(scorelines) == 5
    metrics, calibration = evaluate_model_by_date(data)
    assert not metrics.empty and not calibration.empty


if __name__ == "__main__":
    main()
