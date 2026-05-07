"""Generate synthetic time-series CSVs for V3 scenarios."""
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


def _apply_scenario_profile(
    scenario: str,
    hour: int,
    demand: int,
    renewable: int,
    price: int,
    weather: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    if scenario == "winter_peak":
        if hour < 8 or hour >= 17:
            demand += 1
            price = max(price, 2)
        renewable -= 1
        weather = int(rng.choice([0, 0, 1, 2], p=[0.45, 0.25, 0.20, 0.10]))
    elif scenario == "summer_surplus":
        if 10 <= hour < 18:
            renewable += 1
            price = min(price, 1)
        if 13 <= hour < 20:
            demand += 1
        weather = int(rng.choice([2, 2, 1, 0], p=[0.45, 0.25, 0.20, 0.10]))
    elif scenario == "grid_stress":
        if 7 <= hour < 10 or 17 <= hour < 22:
            demand += 1
            price = 2
        if rng.random() < 0.20:
            renewable -= 1
    elif scenario == "renewable_volatility":
        if rng.random() < 0.35:
            renewable += int(rng.choice([-2, -1, 1, 2]))
        if rng.random() < 0.15:
            price += 1
    return (
        int(np.clip(demand, 0, 3)),
        int(np.clip(renewable, 0, 3)),
        int(np.clip(price, 0, 2)),
        int(np.clip(weather, 0, 2)),
    )


def generate(
    out_path: Path,
    days: int = 7,
    seed: int = 0,
    scenario: str = "baseline",
    demand_jitter: float = 0.10,
    renewable_jitter: float = 0.15,
) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(days * 24):
        hour = step % 24
        demand = _baseline_demand(hour)
        renewable = _baseline_renewable(hour)
        price = _baseline_price(hour)
        weather = _baseline_weather(rng)
        demand, renewable, price, weather = _apply_scenario_profile(
            scenario, hour, demand, renewable, price, weather, rng
        )
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
    scenario_seeds: dict[str, tuple[int, int]] = {
        "baseline": (0, 1),
        "winter_peak": (10, 11),
        "summer_surplus": (20, 21),
        "grid_stress": (30, 31),
        "renewable_volatility": (40, 41),
    }
    for scenario, (train_seed, eval_seed) in scenario_seeds.items():
        generate(base / f"{scenario}_train.csv", days=7, seed=train_seed, scenario=scenario)
        generate(base / f"{scenario}_eval.csv", days=7, seed=eval_seed, scenario=scenario)
    print(f"Generated V3 synthetic time-series CSV files in {base}")


if __name__ == "__main__":
    main()
