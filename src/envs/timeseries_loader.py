"""Load V3 synthetic time-series CSVs used by SmartGridEnvV3."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "step", "hour", "weather_level",
    "demand_level", "renewable_level", "price_level",
)

RANGED_COLUMNS = {
    "hour": (0, 23),
    "weather_level": (0, 2),
    "demand_level": (0, 3),
    "renewable_level": (0, 3),
    "price_level": (0, 2),
}


def _validate_numeric_column(df: pd.DataFrame, column: str, csv_path: Path) -> None:
    """Validate and coerce a required CSV column to discrete integers.

    Args:
        df: Time-series DataFrame being validated in place.
        column: Column name to validate.
        csv_path: Source CSV path used in validation error messages.

    Raises:
        ValueError: If the column contains missing, non-numeric or
            non-integer values.
    """
    numeric_values = pd.to_numeric(df[column], errors="coerce")
    if numeric_values.isna().any():
        raise ValueError(f"Column '{column}' must be numeric in {csv_path.name}.")
    if (numeric_values % 1 != 0).any():
        raise ValueError(f"Column '{column}' must contain discrete integer values in {csv_path.name}.")
    df[column] = numeric_values.astype(int)


def _validate_range(df: pd.DataFrame, column: str, lower: int, upper: int, csv_path: Path) -> None:
    """Validate that a discrete CSV column stays within an expected range.

    Args:
        df: Time-series DataFrame to validate.
        column: Column name to check.
        lower: Inclusive lower bound.
        upper: Inclusive upper bound.
        csv_path: Source CSV path used in validation error messages.

    Raises:
        ValueError: If any row falls outside the expected range.
    """
    invalid_rows = df[(df[column] < lower) | (df[column] > upper)]
    if not invalid_rows.empty:
        raise ValueError(
            f"Column '{column}' must be between {lower} and {upper} in {csv_path.name}."
        )


def load_timeseries(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a synthetic V3 time-series CSV.

    Args:
        csv_path: CSV path containing step, hour, weather, demand, renewable
            and price columns.

    Returns:
        A validated DataFrame with discrete integer columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Timeseries CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")
    if df.empty:
        raise ValueError(f"Timeseries CSV is empty: {csv_path}")
    for column in REQUIRED_COLUMNS:
        _validate_numeric_column(df, column, csv_path)
    if (df["step"] < 0).any():
        raise ValueError(f"Column 'step' must be non-negative in {csv_path.name}.")
    for column, (lower, upper) in RANGED_COLUMNS.items():
        _validate_range(df, column, lower, upper, csv_path)
    return df


def hour_to_period(hour: int) -> int:
    """Map an hour 0..23 to a discrete day period 0..3."""
    return int(hour) // 6
