"""Generate synthetic time-series CSVs for V3 scenarios.

The CSV format is: step,hour,weather_level,demand_level,renewable_level,price_level
All values are discrete (matches V3 state design from the design doc).

Usage:
    python -m scripts.generate_synthetic_timeseries
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _baseline_demand(hour: int) -> int:
    if 0 <= hour < 6:
        return 1
    if 6 <= hour < 9:
        return 2
    if 9 <= hour < 16:
        return 2
    if 16 <= hour < 21:
        return 3
    return 1


def _baseline_renewable(hour: int) -> int:
    if 6 <= hour < 9:
        return 1
    if 9 <= hour < 16:
        return 3
    if 16 <= hour < 19:
        return 1
    return 0


def _baseline_price(hour: int) -> int:
    if 16 <= hour < 21:
        return 2
    if 9 <= hour < 16:
        return 0
    return 1


def _baseline_weather(rng: np.random.Generator) -> int:
    return int(rng.choice([1, 1, 1, 0, 2], p=[0.4, 0.2, 0.2, 0.1, 0.1]))


def generate(out_path: Path, days: int = 7, seed: int = 0,
             demand_jitter: float = 0.10, renewable_jitter: float = 0.15) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(days * 24):
        hour = step % 24
        demand = _baseline_demand(hour)
        renewable = _baseline_renewable(hour)
        price = _baseline_price(hour)
        weather = _baseline_weather(rng)
        if rng.random() < demand_jitter:
            demand = int(np.clip(demand + rng.choice([-1, 1]), 0, 3))
        if rng.random() < renewable_jitter:
            renewable = int(np.clip(renewable + rng.choice([-1, 1]), 0, 3))
        rows.append([step, hour, weather, demand, renewable, price])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "hour", "weather_level",
                         "demand_level", "renewable_level", "price_level"])
        writer.writerows(rows)


def main() -> None:
    base = Path("data/timeseries")
    generate(base / "baseline_train.csv", days=7, seed=0)
    generate(base / "baseline_eval.csv", days=7, seed=1)
    print(f"Generated baseline_train.csv and baseline_eval.csv in {base}")


if __name__ == "__main__":
    main()
