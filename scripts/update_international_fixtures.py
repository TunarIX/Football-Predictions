"""Manual/API-ready adapter for shared international upcoming fixtures.

For now this adapter reads a supplied CSV and writes the canonical shared file.
A live API adapter can later replace ``read_source`` while keeping the same
``write_international_fixtures`` output contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_sources import UPCOMING_COLUMNS
from src.data_loader import safe_read_csv
from src.fixtures import INTERNATIONAL_UPCOMING, write_international_fixtures


def read_source(source_csv: str | None = None) -> pd.DataFrame:
    if not source_csv:
        return pd.DataFrame(columns=UPCOMING_COLUMNS)
    return safe_read_csv(source_csv, UPCOMING_COLUMNS)


def update_international_fixtures(source_csv: str | None = None, output: Path = INTERNATIONAL_UPCOMING) -> pd.DataFrame:
    fixtures = write_international_fixtures(read_source(source_csv), output)
    if fixtures.empty:
        print(
            f"No international fixtures found. Wrote headers only to {output}. "
            "Add a CSV with Date, Time, Competition, HomeTeam, AwayTeam, HomeOdds, DrawOdds, AwayOdds, OddsSource."
        )
    else:
        print(f"wrote {len(fixtures):,} international fixtures to {output}")
    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", help="Manual international fixtures CSV")
    parser.add_argument("--output", type=Path, default=INTERNATIONAL_UPCOMING)
    args = parser.parse_args()
    update_international_fixtures(args.source_csv, args.output)


if __name__ == "__main__":
    main()
