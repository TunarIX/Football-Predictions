"""Local maintained FIFA World Cup 2026 fixture source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.data_sources import UPCOMING_COLUMNS
from .base import FixtureSourceResult, ensure_parent, normalize_fixture_source_frame
from .manual_csv import ManualCsvFixtureSource

WORLD_CUP_2026_FIXTURES = Path("data/raw/worldcup_2026_fixtures.csv")
WORLD_CUP_TEMPLATE_COLUMNS = UPCOMING_COLUMNS
def _template_rows() -> list[dict[str, str]]:
    """Return editable sample fixtures dated soon enough for the next-48h workflow."""

    kickoff = (pd.Timestamp.now(tz=None).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return [
        {
            "Date": kickoff,
            "Time": "19:00",
            "Competition": "FIFA World Cup",
            "HomeTeam": "Mexico",
            "AwayTeam": "Canada",
            "HomeOdds": "",
            "DrawOdds": "",
            "AwayOdds": "",
            "Over25Odds": "",
            "Under25Odds": "",
            "OddsSource": "Manual template",
        },
        {
            "Date": kickoff,
            "Time": "22:00",
            "Competition": "FIFA World Cup",
            "HomeTeam": "United States",
            "AwayTeam": "Japan",
            "HomeOdds": "",
            "DrawOdds": "",
            "AwayOdds": "",
            "Over25Odds": "",
            "Under25Odds": "",
            "OddsSource": "Manual template",
        },
    ]


def ensure_worldcup_template(path: Path = WORLD_CUP_2026_FIXTURES) -> Path:
    """Create an editable World Cup fixture/odds CSV template when missing."""

    if not path.exists():
        ensure_parent(path)
        pd.DataFrame(_template_rows(), columns=WORLD_CUP_TEMPLATE_COLUMNS).to_csv(path, index=False)
    return path


@dataclass(frozen=True)
class WorldCupStaticFixtureSource:
    """Load FIFA World Cup fixtures from the local maintained schedule file."""

    path: Path = WORLD_CUP_2026_FIXTURES
    name: str = "worldcup_2026_static"

    def load(self) -> FixtureSourceResult:
        existed = self.path.exists()
        ensure_worldcup_template(self.path)
        result = ManualCsvFixtureSource(self.path, competition="FIFA World Cup", name=self.name).load()
        fixtures = normalize_fixture_source_frame(result.fixtures, competition="FIFA World Cup")
        messages = list(result.messages)
        if not existed:
            messages.insert(0, f"created editable World Cup fixture template at {self.path}")
        return FixtureSourceResult(fixtures, messages)
