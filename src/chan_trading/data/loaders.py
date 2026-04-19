from __future__ import annotations

from pathlib import Path

import pandas as pd


_PRICE_LIKE_COLUMNS = {"open", "high", "low", "close", "adj_close"}


def load_prices_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    """Load a price table from CSV and enforce basic time-series hygiene."""
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    if df.empty:
        raise ValueError("Input CSV is empty")
    if df.index.has_duplicates:
        raise ValueError("Input CSV contains duplicate timestamps")
    if df.isna().all(axis=None):
        raise ValueError("Input CSV contains only NaN values")

    numeric = df.astype(float)
    if not numeric.index.is_monotonic_increasing:
        raise ValueError("Input CSV index must be monotonic increasing")

    for column in numeric.columns:
        lower = column.lower()
        if lower in _PRICE_LIKE_COLUMNS and (numeric[column] <= 0).any():
            raise ValueError(f"Column {column!r} contains non-positive price values")
    return numeric
