"""Competition configuration loading."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

DEFAULT_CONFIG = Path("config/competitions.yml")


def _simple_competition_yaml(text: str) -> list[dict]:
    """Parse the small competitions.yml shape when PyYAML is unavailable."""
    competitions: list[dict] = []
    current: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "competitions:":
            continue
        if line.startswith("- "):
            if current:
                competitions.append(current)
            current = {}
            line = line[2:]
        if ":" in line and current is not None:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip() or None
    if current:
        competitions.append(current)
    return competitions


def load_competitions(path: str | Path = DEFAULT_CONFIG) -> list[dict]:
    """Load modular competition metadata from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        return []
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text) or {}
        return payload.get("competitions", [])
    except ImportError:
        return _simple_competition_yaml(text)


def competitions_table(path: str | Path = DEFAULT_CONFIG) -> pd.DataFrame:
    """Return configured competitions as a display-friendly table."""
    return pd.DataFrame(load_competitions(path))
