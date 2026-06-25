from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from scripts.data_sources import UPCOMING_COLUMNS, normalize_upcoming_frame
from scripts.predict_next_48h import generate_next_48h_predictions
from scripts.update_international_fixtures import update_international_fixtures
from src.fixture_sources.manual_csv import ManualCsvFixtureSource
from src.fixture_sources.worldcup_static import WorldCupStaticFixtureSource
from src.odds import odds_to_probabilities


def test_worldcup_fixture_source_creates_and_loads_template(tmp_path: Path) -> None:
    path = tmp_path / "worldcup_2026_fixtures.csv"

    result = WorldCupStaticFixtureSource(path=path).load()

    assert path.exists()
    assert list(pd.read_csv(path).columns) == [
        "Date",
        "Time",
        "Competition",
        "HomeTeam",
        "AwayTeam",
        "HomeOdds",
        "DrawOdds",
        "AwayOdds",
        "Over25Odds",
        "Under25Odds",
        "OddsSource",
    ]
    assert not result.fixtures.empty
    assert result.fixtures.iloc[0]["Competition"] == "FIFA World Cup"


def test_manual_csv_source_loads_missing_and_present_odds(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    path.write_text(
        "Date,Time,Competition,HomeTeam,AwayTeam,HomeOdds,DrawOdds,AwayOdds,Over25Odds,Under25Odds,OddsSource\n"
        "2026-07-01,20:00,FIFA World Cup,Spain,Brazil,,,,,,Manual no odds\n"
        "2026-07-02,20:00,FIFA World Cup,France,Germany,2.1,3.2,3.4,1.9,1.8,Manual odds\n"
    )

    fixtures = ManualCsvFixtureSource(path).load().fixtures

    assert len(fixtures) == 2
    assert fixtures.loc[fixtures["HomeTeam"] == "Spain", "HomeOdds"].isna().all()
    assert fixtures.loc[fixtures["HomeTeam"] == "France", "HomeOdds"].iloc[0] == 2.1


def test_update_international_fixtures_writes_normalized_manual_csv(tmp_path: Path) -> None:
    source = tmp_path / "worldcup.csv"
    output = tmp_path / "international_fixtures.csv"
    source.write_text(
        "Date,Time,Competition,HomeTeam,AwayTeam,HomeOdds,DrawOdds,AwayOdds,OddsSource\n"
        "2026-07-03,18:00,FIFA World Cup,Argentina,England,,,,Unavailable\n"
    )

    written = update_international_fixtures(source_csv=str(source), output=output)

    assert output.exists()
    saved = pd.read_csv(output)
    assert len(written) == 1
    assert saved.iloc[0]["HomeTeam"] == "Argentina"
    assert "Over25Odds" in saved.columns


def test_predictions_probability_inputs_work_without_and_with_odds() -> None:
    assert all(pd.isna(value) for value in odds_to_probabilities(float("nan"), float("nan"), float("nan")))

    home, draw, away = odds_to_probabilities(2.0, 3.5, 4.0)

    assert home + draw + away == pytest.approx(1.0, abs=0.001)
    assert home > draw > away


def test_normalize_upcoming_frame_creates_missing_total_odds_columns() -> None:
    fixtures = normalize_upcoming_frame(
        pd.DataFrame(
            [
                {
                    "Date": "2026-07-01",
                    "Time": "20:00",
                    "Competition": "FIFA World Cup",
                    "HomeTeam": "Spain",
                    "AwayTeam": "Brazil",
                }
            ]
        )
    )

    assert list(fixtures.columns) == UPCOMING_COLUMNS
    assert pd.isna(fixtures.loc[0, "Over25Odds"])
    assert pd.isna(fixtures.loc[0, "Under25Odds"])


def test_normalize_upcoming_frame_accepts_total_odds_typos() -> None:
    fixtures = normalize_upcoming_frame(
        pd.DataFrame(
            [
                {
                    "Date": "2026-07-01",
                    "Time": "20:00",
                    "Competition": "FIFA World Cup",
                    "HomeTeam": "Spain",
                    "AwayTeam": "Brazil",
                    "Over250dds": 1.91,
                    "Under250dds": 1.83,
                }
            ]
        )
    )

    assert fixtures.loc[0, "Over25Odds"] == pytest.approx(1.91)
    assert fixtures.loc[0, "Under25Odds"] == pytest.approx(1.83)


def _write_minimal_international_history(path: Path) -> None:
    teams = ["Mexico", "Canada", "United States", "Japan", "Spain", "Brazil", "France", "Germany"]
    rows = []
    for i in range(48):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        home_goals = i % 4
        away_goals = (i + 2) % 3
        rows.append(
            {
                "Date": (pd.Timestamp("2022-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "Competition": "FIFA World Cup" if i % 3 == 0 else "International Friendly",
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_worldcup_update_creates_international_fixtures_from_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    written = update_international_fixtures(output=Path("data/upcoming/international_fixtures.csv"))

    assert Path("data/raw/worldcup_2026_fixtures.csv").exists()
    assert Path("data/upcoming/international_fixtures.csv").exists()
    assert not written.empty
    assert set(UPCOMING_COLUMNS).issubset(pd.read_csv("data/upcoming/international_fixtures.csv").columns)


def test_generate_next_48h_predictions_without_odds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_minimal_international_history(Path("data/processed/international_matches.csv"))
    kickoff = (pd.Timestamp.now(tz=None) + pd.Timedelta(hours=24)).strftime("%Y-%m-%d")
    source = Path("data/raw/worldcup_2026_fixtures.csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "Date,Time,Competition,HomeTeam,AwayTeam,OddsSource\n"
        f"{kickoff},19:00,FIFA World Cup,Mexico,Canada,Manual no odds\n"
    )
    update_international_fixtures(source_csv=str(source), output=Path("data/upcoming/international_fixtures.csv"))

    predictions = generate_next_48h_predictions(competition="FIFA World Cup")

    assert not predictions.empty
    assert predictions.iloc[0]["ValueSignal"] == "No odds available"


def test_zero_fixture_prediction_message_does_not_claim_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    _write_minimal_international_history(Path("data/processed/international_matches.csv"))
    Path("data/upcoming").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=UPCOMING_COLUMNS).to_csv("data/upcoming/international_fixtures.csv", index=False)

    predictions = generate_next_48h_predictions(competition="FIFA World Cup")
    output = capsys.readouterr().out

    assert predictions.empty
    assert "No international fixtures available" in output
    assert "wrote 0 predictions" not in output
