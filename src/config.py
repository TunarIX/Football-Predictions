"""Competition configuration loading for the dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "competitions.yml"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~", ""}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    return value.strip('"\'')


def load_competitions(path: str | Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    """Load simple competition definitions from config/competitions.yml.

    The project uses a deliberately small YAML shape so the app can load the
    competition registry without requiring an additional parser package.
    """
    competitions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "competitions:":
            continue
        if line.startswith("- "):
            if current:
                competitions.append(current)
            current = {}
            line = line[2:]
            if line:
                key, value = line.split(":", 1)
                current[key.strip()] = _parse_scalar(value)
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = _parse_scalar(value)
    if current:
        competitions.append(current)
    return [competition for competition in competitions if competition.get("name")]


def competition_names(path: str | Path = DEFAULT_CONFIG_PATH) -> list[str]:
    return [competition["name"] for competition in load_competitions(path)]
