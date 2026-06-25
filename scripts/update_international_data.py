"""Build the shared international historical matches CSV.

The updater prefers a manual local CSV at ``data/raw/international_matches.csv``.
When that override is absent, it downloads the public martj42 international
results dataset and stores the raw copy at the same path before processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_international_match_data, parse_dates

RAW_INTERNATIONAL = Path("data/raw/international_matches.csv")
PROCESSED_INTERNATIONAL = Path("data/processed/international_matches.csv")
INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
INTERNATIONAL_COLUMNS = [
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
REQUIRED_SOURCE_COLUMNS = {
    "date",
    "tournament",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}
DOWNLOAD_SOURCE_NAME = "martj42/international_results results.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding_errors="ignore")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read international source CSV {path}: {exc}") from exc


def download_source(source_csv: Path = RAW_INTERNATIONAL, source_url: str = INTERNATIONAL_RESULTS_URL) -> pd.DataFrame:
    """Download the public international results CSV and save a raw local copy."""
    try:
        with urlopen(source_url, timeout=30) as response:  # noqa: S310 - fixed public CSV URL
            contents = response.read()
    except (OSError, URLError) as exc:
        raise ValueError(f"Could not download international results from {source_url}: {exc}") from exc
    if not contents:
        raise ValueError(f"Downloaded international results from {source_url} were empty")
    source_csv.parent.mkdir(parents=True, exist_ok=True)
    source_csv.write_bytes(contents)
    frame = _read_csv(source_csv)
    frame["SourceFile"] = DOWNLOAD_SOURCE_NAME
    return frame


def load_source(source_csv: Path = RAW_INTERNATIONAL, source_url: str = INTERNATIONAL_RESULTS_URL) -> pd.DataFrame:
    """Load national-team match rows from manual override or the public dataset."""
    if source_csv.exists():
        frame = _read_csv(source_csv)
        frame["SourceFile"] = source_csv.name
        return frame
    return download_source(source_csv, source_url)


def normalize_international_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize common national-team CSV schemas to the processed contract."""
    if raw.empty:
        raise ValueError("Invalid international historical data: rows > 0 validation failed")
    lowered_columns = {str(column).strip().lower() for column in raw.columns}
    source_missing = sorted(REQUIRED_SOURCE_COLUMNS - lowered_columns)
    if source_missing:
        raise ValueError("Invalid international historical data: missing required columns: " + ", ".join(source_missing))
    cleaned = clean_international_match_data(raw)
    normalized = cleaned.copy()
    if "Competition" not in normalized.columns:
        normalized["Competition"] = pd.NA
    if "Neutral" not in normalized.columns:
        normalized["Neutral"] = pd.NA
    if "Country" not in normalized.columns:
        normalized["Country"] = pd.NA
    if "SourceFile" not in normalized.columns:
        normalized["SourceFile"] = pd.NA
    normalized = normalized[INTERNATIONAL_COLUMNS].copy()
    normalized["Date"] = parse_dates(normalized["Date"])
    normalized = normalized.drop_duplicates(
        ["Date", "Competition", "HomeTeam", "AwayTeam", "FTHG", "FTAG"], keep="last"
    )
    return normalized.sort_values("Date").reset_index(drop=True)


def validate_international_matches(data: pd.DataFrame) -> None:
    """Raise a clear error if processed international data is unusable."""
    missing = [column for column in INTERNATIONAL_COLUMNS if column not in data.columns]
    errors: list[str] = []
    if data.empty:
        errors.append("rows > 0 validation failed")
    if missing:
        errors.append(f"missing required columns: {', '.join(missing)}")
    if "Date" in data.columns and parse_dates(data["Date"]).isna().any():
        errors.append("Date parses correctly validation failed")
    if "FTR" in data.columns:
        invalid = sorted(set(data["FTR"].dropna().astype(str).str.upper()) - {"H", "D", "A"})
        if invalid or data["FTR"].isna().any():
            errors.append("FTR contains only H/D/A validation failed")
    if errors:
        raise ValueError("Invalid international historical data: " + "; ".join(errors))


def update_international_data(
    source_csv: Path = RAW_INTERNATIONAL,
    output: Path = PROCESSED_INTERNATIONAL,
    source_url: str = INTERNATIONAL_RESULTS_URL,
) -> pd.DataFrame:
    """Process international data and write it only after validation succeeds."""
    raw = load_source(source_csv, source_url)
    normalized = normalize_international_matches(raw)
    validate_international_matches(normalized)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    print(f"wrote {len(normalized):,} international rows to {output}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, default=RAW_INTERNATIONAL)
    parser.add_argument("--output", type=Path, default=PROCESSED_INTERNATIONAL)
    parser.add_argument("--source-url", default=INTERNATIONAL_RESULTS_URL)
    args = parser.parse_args()
    try:
        update_international_data(args.source_csv, args.output, args.source_url)
    except Exception as exc:  # noqa: BLE001 - CLI should show concise failure
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
