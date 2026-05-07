"""SmartGridEnvV3 - Gymnasium environment for the V3 milestone."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.scenario_loader import load_scenario
from src.envs.timeseries_loader import hour_to_period, load_timeseries


try:
    from src.envs.fuzzy_risk import compute_risk as _external_compute_risk
except ImportError:
    _external_compute_risk = None


def _fallback_compute_risk(battery: int, demand: int, renewable: int, price: int) -> float:
    risk = 0.0
    if battery <= 1 and demand >= 2:
        risk += 0.5
    if renewable <= 1 and price >= 2:
        risk += 0.3
    if demand >= 3 and price >= 2:
        risk += 0.2
    if battery >= 3 and renewable >= 2:
        risk -= 0.2
    return float(max(0.0, min(1.0, risk)))


def _compute_risk(battery: int, demand: int, renewable: int, price: int) -> float:
    if _external_compute_risk is not None:
        return float(_external_compute_risk(battery, demand, renewable, price))
    return _fallback_compute_risk(battery, demand, renewable, price)


class SmartGridEnvV3(gym.Env):
    """Discrete Gymnasium environment for V3, fed by a synthetic CSV."""

    metadata = {"render_modes": []}

    def __init__(self, scenario_path: str | Path, mode: str = "train",
                 seed: int | None = None) -> None:
        super().__init__()
        self.config = load_scenario(scenario_path)
        if mode == "train":
            csv_path = self.config["train_csv_path"]
        elif mode == "eval":
            csv_path = self.config["eval_csv_path"]
        else:
            raise ValueError(f"Unsupported env mode: {mode}")
        self.timeseries = load_timeseries(csv_path)
        self.battery_capacity = int(self.config["battery_capacity"])
        self.initial_battery = int(self.config["initial_battery"])
        self.max_steps = int(self.config["max_steps"])
        self.reward_weights = dict(self.config["reward_weights"])
        self.observation_space = spaces.MultiDiscrete([
            self.battery_capacity + 1, 4, 4, 3, 4, 3,
        ])
        self.action_space = spaces.Discrete(5)
        self._rng = np.random.default_rng(seed)
        self.battery = self.initial_battery
        self.step_idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        else:
            super().reset()
        self.battery = self.initial_battery
        self.step_idx = 0
        obs = self._build_observation()
        row = self._current_row()
        info: dict[str, Any] = {
            "demand_covered": 0, "unmet_demand": 0, "grid_bought": 0,
            "sold": 0, "wasted_renewable": 0, "invalid_action": 0,
            "risk_score": _compute_risk(
                self.battery,
                int(row["demand_level"]),
                int(row["renewable_level"]),
                int(row["price_level"]),
            ),
            "battery": self.battery,
        }
        return obs, info

    def step(self, action: int):
        if self.step_idx >= len(self.timeseries) or self.step_idx >= self.max_steps:
            obs = self._build_observation()
            return obs, 0.0, False, True, {"battery": self.battery, "risk_score": 0.0}
        row = self.timeseries.iloc[self.step_idx]
        demand = int(row["demand_level"])
        renewable = int(row["renewable_level"])
        price = int(row["price_level"])
        outcomes = self._apply_action(int(action), demand, renewable)
        risk_score = _compute_risk(self.battery, demand, renewable, price)
        reward = self._compute_reward(outcomes, price, risk_score)
        self.step_idx += 1
        truncated = (self.step_idx >= len(self.timeseries)
                     or self.step_idx >= self.max_steps)
        terminated = False
        obs = self._build_observation()
        info: dict[str, Any] = {**outcomes, "risk_score": risk_score, "battery": self.battery}
        return obs, float(reward), terminated, truncated, info

    def _current_row(self):
        idx = min(self.step_idx, len(self.timeseries) - 1)
        return self.timeseries.iloc[idx]

    def _build_observation(self) -> np.ndarray:
        row = self._current_row()
        demand = min(int(row["demand_level"]), 3)
        renewable = min(int(row["renewable_level"]), 3)
        price = min(int(row["price_level"]), 2)
        weather = min(int(row["weather_level"]), 2)
        period = min(hour_to_period(int(row["hour"])), 3)
        battery = max(0, min(int(self.battery), self.battery_capacity))
        return np.array([battery, demand, renewable, price, period, weather],
                        dtype=np.int64)

    def _apply_action(self, action: int, demand: int, renewable: int) -> dict[str, int]:
        demand_covered = min(renewable, demand)
        unmet_demand = max(0, demand - renewable)
        grid_bought = 0
        energy_sold = 0
        wasted_renewable = max(0, renewable - demand)
        invalid_action = 0
        if action == 0:
            if unmet_demand > 0 and self.battery > 0:
                use = min(self.battery, unmet_demand)
                self.battery -= use
                demand_covered += use
                unmet_demand -= use
            else:
                invalid_action = 1
        elif action == 1:
            if unmet_demand > 0:
                grid_bought = unmet_demand
                demand_covered += unmet_demand
                unmet_demand = 0
            else:
                invalid_action = 1
        elif action == 2:
            surplus = wasted_renewable
            free_capacity = self.battery_capacity - self.battery
            if surplus > 0 and free_capacity > 0:
                stored = min(surplus, free_capacity)
                self.battery += stored
                wasted_renewable = surplus - stored
            else:
                invalid_action = 1
        elif action == 3:
            surplus = wasted_renewable
            if surplus > 0:
                energy_sold = surplus
                wasted_renewable = 0
            else:
                invalid_action = 1
        elif action == 4:
            pass
        else:
            invalid_action = 1
        return {
            "demand_covered": int(demand_covered),
            "unmet_demand": int(unmet_demand),
            "grid_bought": int(grid_bought),
            "sold": int(energy_sold),
            "wasted_renewable": int(wasted_renewable),
            "invalid_action": int(invalid_action),
        }

    def _compute_reward(self, outcomes: dict[str, int], price: int,
                        risk_score: float) -> float:
        w = self.reward_weights
        reward = (
            w["demand_covered"] * outcomes["demand_covered"]
            + w["unmet_demand"] * outcomes["unmet_demand"]
            + w["grid_bought"] * outcomes["grid_bought"] * (1 + price)
            + w["sold"] * outcomes["sold"]
            + w["invalid_action"] * outcomes["invalid_action"]
            + w["wasted_renewable"] * outcomes["wasted_renewable"]
            + w["risk"] * risk_score
        )
        return float(reward)
