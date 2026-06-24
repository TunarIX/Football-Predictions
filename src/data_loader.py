"""CSV loading helpers for football-data and international match files."""
from __future__ import annotations

from collections.abc import Iterable
import pandas as pd

from .preprocessing import clean_match_data


def _read_uploaded_csv(file) -> pd.DataFrame:
    raw = pd.read_csv(file, encoding_errors="ignore")
    raw["SourceFile"] = getattr(file, "name", "uploaded_csv")
    return raw


def load_uploaded_files(files: Iterable, match_type: str = "club") -> pd.DataFrame:
    """Read and combine one or more uploaded match CSV files."""
    frames = [_read_uploaded_csv(file) for file in files]
    if not frames:
        return pd.DataFrame()
    return clean_match_data(pd.concat(frames, ignore_index=True), match_type=match_type)


def load_international_files(files: Iterable) -> pd.DataFrame:
    """Read international CSVs with teams, scores, tournaments, dates, and optional odds."""
    return load_uploaded_files(files, match_type="international")
