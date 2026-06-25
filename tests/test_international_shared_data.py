"""Shared international historical and fixtures handling tests."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import UPCOMING_COLUMNS, normalize_upcoming_frame
from scripts.update_international_data import update_international_data
from src.fixtures import _filter_competition
from src.predictor import train_baseline_model
from src.preprocessing import clean_international_match_data


def test_fifa_world_cup_filters_international_historical_rows() -> None:
    raw = pd.DataFrame(
        [
            {"date": "2022-11-20", "home_team": "Qatar", "away_team": "Ecuador", "home_score": 0, "away_score": 2, "tournament": "FIFA World Cup"},
            {"date": "2022-09-22", "home_team": "France", "away_team": "Austria", "home_score": 2, "away_score": 0, "tournament": "UEFA Nations League"},
        ]
    )
    cleaned = clean_international_match_data(raw)
    filtered = _filter_competition(cleaned, "FIFA World Cup")
    assert len(filtered) == 1
    assert filtered.iloc[0]["Competition"] == "FIFA World Cup"


def test_fifa_world_cup_filters_international_upcoming_fixtures() -> None:
    fixtures = normalize_upcoming_frame(
        pd.DataFrame(
            [
                {"Date": "2030-06-13", "Time": "20:00", "Competition": "FIFA World Cup", "HomeTeam": "Spain", "AwayTeam": "Brazil"},
                {"Date": "2030-06-14", "Time": "18:00", "Competition": "International Friendly", "HomeTeam": "Germany", "AwayTeam": "Japan"},
            ]
        )
    )
    filtered = _filter_competition(fixtures, "FIFA World Cup")
    assert len(filtered) == 1
    assert filtered.iloc[0]["HomeTeam"] == "Spain"


def test_training_does_not_require_odds() -> None:
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(40):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        hg = i % 3
        ag = (i + 1) % 3
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "home_team": home,
                "away_team": away,
                "home_score": hg,
                "away_score": ag,
                "tournament": "FIFA World Cup" if i % 5 == 0 else "International Friendly",
            }
        )
    cleaned = clean_international_match_data(pd.DataFrame(rows))
    model, training = train_baseline_model(cleaned)
    assert not training.empty
    assert {"ImpHome", "ImpDraw", "ImpAway"}.issubset(training.columns)


def test_manual_raw_international_csv_is_processed_into_non_empty_file(tmp_path: Path) -> None:
    raw = tmp_path / "international_matches.csv"
    output = tmp_path / "processed_international_matches.csv"
    raw.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,neutral,country\n"
        "2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,True,Qatar\n"
        "2022-09-22,France,Austria,2,0,UEFA Nations League,False,France\n"
    )

    processed = update_international_data(raw, output)

    assert output.exists()
    assert len(processed) == 2
    assert {
        "Date",
        "Competition",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "Neutral",
        "Country",
        "SourceFile",
    }.issubset(processed.columns)
    assert set(processed["FTR"]) == {"A", "H"}


def test_empty_input_does_not_overwrite_valid_processed_file(tmp_path: Path) -> None:
    raw = tmp_path / "international_matches.csv"
    output = tmp_path / "processed_international_matches.csv"
    valid_contents = (
        "Date,Competition,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Neutral,Country,SourceFile\n"
        "2022-11-20,FIFA World Cup,Qatar,Ecuador,0,2,A,True,Qatar,seed.csv\n"
    )
    output.write_text(valid_contents)
    raw.write_text("date,home_team,away_team,home_score,away_score,tournament,neutral\n")

    try:
        update_international_data(raw, output)
    except SystemExit as exc:  # pragma: no cover - defensive for CLI-like failures
        assert "validation" in str(exc).lower() or "invalid" in str(exc).lower()
    except ValueError as exc:
        assert "rows > 0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty input should fail validation")

    assert output.read_text() == valid_contents


def test_automatic_downloader_normalizes_public_results_csv(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "international_matches.csv"
    output = tmp_path / "processed_international_matches.csv"
    csv_bytes = (
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2022-11-20,Qatar,Ecuador,0,2,FIFA World Cup,Al Khor,Qatar,TRUE\n"
        "2022-09-22,France,Austria,2,0,UEFA Nations League,Paris,France,FALSE\n"
        "2022-09-23,England,Italy,0,0,UEFA Nations League,London,England,FALSE\n"
    ).encode()

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return csv_bytes

    def fake_urlopen(url: str, timeout: int = 30) -> MockResponse:
        assert url == "https://example.test/results.csv"
        assert timeout == 30
        return MockResponse()

    monkeypatch.setattr("scripts.update_international_data.urlopen", fake_urlopen)

    processed = update_international_data(raw, output, "https://example.test/results.csv")

    assert raw.exists()
    assert output.exists()
    assert list(processed.columns) == [
        "Date",
        "Competition",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "Neutral",
        "Country",
        "SourceFile",
    ]
    assert len(processed) == 3
    assert set(processed["FTR"]) == {"A", "H", "D"}
    assert processed.loc[processed["HomeTeam"] == "Qatar", "Competition"].iloc[0] == "FIFA World Cup"


def test_fifa_world_cup_filter_returns_only_world_cup_rows() -> None:
    rows = pd.DataFrame(
        [
            {"Date": "2022-11-20", "Competition": "FIFA World Cup", "HomeTeam": "Qatar", "AwayTeam": "Ecuador", "FTHG": 0, "FTAG": 2, "FTR": "A"},
            {"Date": "2022-12-18", "Competition": " FIFA World Cup ", "HomeTeam": "Argentina", "AwayTeam": "France", "FTHG": 3, "FTAG": 3, "FTR": "D"},
            {"Date": "2022-09-22", "Competition": "UEFA Nations League", "HomeTeam": "France", "AwayTeam": "Austria", "FTHG": 2, "FTAG": 0, "FTR": "H"},
        ]
    )

    filtered = _filter_competition(rows, "FIFA World Cup")

    assert len(filtered) == 2
    assert set(filtered["Competition"].str.strip()) == {"FIFA World Cup"}


def test_invalid_download_does_not_overwrite_valid_processed_file(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "international_matches.csv"
    output = tmp_path / "processed_international_matches.csv"
    valid_contents = (
        "Date,Competition,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Neutral,Country,SourceFile\n"
        "2022-11-20,FIFA World Cup,Qatar,Ecuador,0,2,A,True,Qatar,seed.csv\n"
    )
    output.write_text(valid_contents)

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return b"date,home_team,away_team,home_score,away_score,tournament,neutral,country\n"

    monkeypatch.setattr("scripts.update_international_data.urlopen", lambda *args, **kwargs: MockResponse())

    try:
        update_international_data(raw, output, "https://example.test/empty.csv")
    except ValueError as exc:
        assert "rows > 0" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid download should fail validation")

    assert output.read_text() == valid_contents


def _mixed_scope_rows() -> pd.DataFrame:
    rows = []
    intl_teams = ["France", "Norway", "Brazil", "Japan"]
    club_teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    for i in range(70):
        home = intl_teams[i % 4]
        away = intl_teams[(i + 1) % 4]
        rows.append({"Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "Competition": "FIFA World Cup" if i % 4 == 0 else "International Friendly", "HomeTeam": home, "AwayTeam": away, "FTHG": i % 4, "FTAG": (i + 1) % 3, "FTR": "H" if i % 3 == 0 else "A" if i % 3 == 1 else "D", "CompetitionType": "international", "MatchType": "international"})
        rows.append({"Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "Competition": "Premier League", "HomeTeam": club_teams[i % 4], "AwayTeam": club_teams[(i + 1) % 4], "FTHG": (i + 2) % 4, "FTAG": i % 3, "FTR": "H" if i % 2 == 0 else "A", "CompetitionType": "club", "MatchType": "club"})
    return pd.DataFrame(rows)


def test_world_cup_predictions_never_use_club_rows(tmp_path: Path, monkeypatch) -> None:
    import src.fixtures as fixtures
    path = tmp_path / "international.csv"
    clean_international_match_data(_mixed_scope_rows()[lambda df: df["CompetitionType"] == "international"]).to_csv(path, index=False)
    monkeypatch.setattr(fixtures, "INTERNATIONAL_HISTORICAL", path)

    loaded, _, _ = fixtures.load_historical_matches_for_competition("FIFA World Cup")

    assert not loaded.empty
    assert set(loaded["CompetitionType"].astype(str)) == {"international"}
    assert set(loaded["Competition"].str.strip()) == {"FIFA World Cup", "International Friendly"}
    assert not loaded["HomeTeam"].isin(["Arsenal", "Chelsea", "Liverpool", "Everton"]).any()


def test_club_predictions_never_use_international_rows(tmp_path: Path, monkeypatch) -> None:
    import src.fixtures as fixtures
    path = tmp_path / "historical.csv"
    _mixed_scope_rows().to_csv(path, index=False)
    monkeypatch.setattr(fixtures, "CLUB_HISTORICAL", path)

    loaded, _, _ = fixtures.load_historical_matches_for_competition("Premier League")

    assert not loaded.empty
    assert set(loaded["CompetitionType"].astype(str)) == {"club"}
    assert set(loaded["Competition"].str.strip()) == {"Premier League"}
    assert not loaded["HomeTeam"].isin(["France", "Norway", "Brazil", "Japan"]).any()


def test_similar_world_cup_matches_are_international_only() -> None:
    from src.odds import add_implied_probabilities, odds_to_probabilities
    from src.predictor import predict_match, similar_historical_matches, train_baseline_model

    intl = clean_international_match_data(_mixed_scope_rows()[lambda df: df["CompetitionType"] == "international"])
    intl = add_implied_probabilities(intl)
    model, training = train_baseline_model(intl, "FIFA World Cup")
    assert model is not None
    _, feature_row, _ = predict_match(model, intl, "France", "Norway", odds_to_probabilities(float("nan"), float("nan"), float("nan")), competition="FIFA World Cup", neutral=True)

    similar = similar_historical_matches(training, feature_row)

    assert not similar.empty
    assert set(similar["CompetitionType"].astype(str)) == {"international"}
    assert not similar["Competition"].str.contains("Premier League|La Liga|Bundesliga", case=False, na=False).any()


def test_missing_odds_still_allow_world_cup_prediction() -> None:
    from src.predictor import predict_match, train_baseline_model

    intl = clean_international_match_data(_mixed_scope_rows()[lambda df: df["CompetitionType"] == "international"])
    model, _ = train_baseline_model(intl, "FIFA World Cup")
    assert model is not None

    prediction, feature_row, notes = predict_match(model, intl, "France", "Norway", (float("nan"), float("nan"), float("nan")), competition="FIFA World Cup", neutral=True)

    assert not prediction.empty
    assert feature_row[["ImpHome", "ImpDraw", "ImpAway"]].notna().all(axis=None)
    assert any("Tournament category influence" in note for note in notes)


def test_international_fixture_downloader_keeps_existing_file_on_invalid_download(tmp_path: Path, monkeypatch) -> None:
    from scripts.update_international_fixtures import update_international_fixtures
    import scripts.update_international_fixtures as updater

    output = tmp_path / "international_fixtures.csv"
    valid = "Date,Time,Competition,HomeTeam,AwayTeam,HomeOdds,DrawOdds,AwayOdds,OddsSource\n2030-06-13,20:00,FIFA World Cup,Spain,Brazil,,,,Unavailable\n"
    output.write_text(valid)
    monkeypatch.setattr(updater, "_source_urls", lambda: ["https://example.test/empty.csv", "https://example.test/fail.csv"])
    monkeypatch.setattr(updater, "_download", lambda url, timeout=30: b"Date,Competition,HomeTeam,AwayTeam\n")

    result = update_international_fixtures(output=output)

    assert len(result) == 1
    assert output.read_text() == valid
