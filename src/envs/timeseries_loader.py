"""Load V3 synthetic time-series CSVs used by SmartGridEnvV3."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "step", "hour", "weather_level",
    "demand_level", "renewable_level", "price_level",
)


def load_timeseries(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Timeseries CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")
    if df.empty:
        raise ValueError(f"Timeseries CSV is empty: {csv_path}")
    return df


def hour_to_period(hour: int) -> int:
    """Map an hour 0..23 to a discrete day period 0..3."""
    return int(hour) // 6
