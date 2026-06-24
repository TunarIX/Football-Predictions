"""CSV loading helpers for the Streamlit app."""
from __future__ import annotations

from collections.abc import Iterable
import pandas as pd

from .preprocessing import clean_match_data


def load_uploaded_files(files: Iterable) -> pd.DataFrame:
    """Read and combine one or more uploaded football-data CSV files."""
    frames: list[pd.DataFrame] = []
    for file in files:
        raw = pd.read_csv(file, encoding_errors="ignore")
        raw["SourceFile"] = getattr(file, "name", "uploaded_csv")
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    return clean_match_data(pd.concat(frames, ignore_index=True))
