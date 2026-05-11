"""SmartGridEnvV3 - Gymnasium environment for the V3 milestone."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.fuzzy_risk import compute_energy_risk
from src.envs.reward_v3 import compute_reward_v3
from src.envs.scenario_loader import load_scenario
from src.envs.timeseries_loader import hour_to_period, load_timeseries


class SmartGridEnvV3(gym.Env):
    """Discrete Gymnasium environment for the final V3 pipeline.

    The environment reads JSON scenario configuration and synthetic CSV time
    series. Its observable state is discrete and does not include fuzzy risk,
    although risk is exposed through ``info`` and used by the V3 reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_path: str | Path,
        mode: str = "train",
        seed: int | None = None,
        reward_weights_override: dict | None = None,
    ) -> None:
        """Create a V3 environment from a JSON scenario file.

        Args:
            scenario_path: Path to the V3 scenario JSON file.
            mode: ``"train"`` to use the training CSV or ``"eval"`` for the
                evaluation CSV.
            seed: Optional seed for reproducible environment state.
            reward_weights_override: Optional reward weights used by V3.1 PSO
                experiments. When omitted, the scenario JSON weights are used.
        """
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
        self.reward_weights = dict(
            reward_weights_override
            if reward_weights_override is not None
            else self.config["reward_weights"]
        )
        self.observation_space = spaces.MultiDiscrete([
            self.battery_capacity + 1, 4, 4, 3, 4, 3,
        ])
        self.action_space = spaces.Discrete(5)
        self._rng = np.random.default_rng(seed)
        self.battery = self.initial_battery
        self.step_idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the episode and return the first V3 observation and risk info."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        else:
            super().reset()
        self.battery = self.initial_battery
        self.step_idx = 0
        obs = self._build_observation()
        row = self._current_row()
        demand = int(row["demand_level"])
        renewable = int(row["renewable_level"])
        price = int(row["price_level"])
        risk = compute_energy_risk(self.battery, demand, renewable, price)
        info: dict[str, Any] = {
            "demand_covered": 0, "unmet_demand": 0, "grid_bought": 0,
            "sold": 0, "wasted_renewable": 0, "invalid_action": 0,
            "demand": demand, "renewable": renewable, "price_level": price,
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "risk_level_name": risk["risk_level_name"],
            "battery": self.battery,
        }
        return obs, info

    def step(self, action: int):
        """Apply one V3 action and return the transition tuple."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action outside the action space: {action}")

        if self.step_idx >= len(self.timeseries) or self.step_idx >= self.max_steps:
            obs = self._build_observation()
            row = self._current_row()
            demand = int(row["demand_level"])
            renewable = int(row["renewable_level"])
            price = int(row["price_level"])
            risk = compute_energy_risk(self.battery, demand, renewable, price)
            info = {
                "battery": self.battery,
                "demand": demand,
                "renewable": renewable,
                "price_level": price,
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "risk_level_name": risk["risk_level_name"],
            }
            return obs, 0.0, False, True, info
        row = self.timeseries.iloc[self.step_idx]
        demand = int(row["demand_level"])
        renewable = int(row["renewable_level"])
        price = int(row["price_level"])
        outcomes = self._apply_action(int(action), demand, renewable)
        risk = compute_energy_risk(self.battery, demand, renewable, price)
        info: dict[str, Any] = {
            **outcomes,
            "demand": demand,
            "renewable": renewable,
            "price_level": price,
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "risk_level_name": risk["risk_level_name"],
            "battery": self.battery,
        }
        reward = compute_reward_v3(info, self.reward_weights)
        self.step_idx += 1
        truncated = (self.step_idx >= len(self.timeseries)
                     or self.step_idx >= self.max_steps)
        terminated = False
        obs = self._build_observation()
        return obs, float(reward), terminated, truncated, info

    def _current_row(self):
        idx = min(self.step_idx, len(self.timeseries) - 1)
        return self.timeseries.iloc[idx]

    def _build_observation(self) -> np.ndarray:
        """Build the discrete V3 observation without adding fuzzy risk."""
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
        """Apply the selected action to battery, demand and renewable balance."""
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

