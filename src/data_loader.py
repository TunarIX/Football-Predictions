"""CSV loading helpers for the Streamlit app."""

from __future__ import annotations

from collections.abc import Iterable
import pandas as pd

from .preprocessing import clean_international_match_data, clean_match_data


def load_uploaded_files(
    files: Iterable, data_source: str = "football-data.co.uk"
) -> pd.DataFrame:
    """Read and combine uploaded CSV files for the selected data source."""
    frames: list[pd.DataFrame] = []
    for file in files:
        raw = pd.read_csv(file, encoding_errors="ignore")
        raw["SourceFile"] = getattr(file, "name", "uploaded_csv")
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if data_source == "international_csv":
        return clean_international_match_data(combined)
    return clean_match_data(combined)
