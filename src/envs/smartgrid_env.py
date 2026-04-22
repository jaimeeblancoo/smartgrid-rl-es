from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SmartGridEnv(gym.Env):
    """Small discrete environment for a simplified smart grid.

    State layout: ``(battery, demand, renewable, period)``

    - battery: `0..2`
    - demand: `0..2`
    - renewable: `0..2`
    - period: `0..3`

    Action mapping:

    - `0`: use battery
    - `1`: buy from grid
    - `2`: store surplus energy
    - `3`: sell surplus energy
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, max_steps: int = 24, seed: int | None = None):
        super().__init__()
        self.max_battery = 2
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 4])
        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.int64)
        self.current_step = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        period = int(self._rng.integers(0, 4))
        battery = int(self._rng.integers(0, 3))
        demand = self._sample_demand(period)
        renewable = self._sample_renewable(period)

        self.state = np.array([battery, demand, renewable, period], dtype=np.int64)
        info = self._build_info(
            battery_used=0,
            grid_bought=0,
            stored=0,
            sold=0,
            unmet_demand=max(0, demand - renewable),
            invalid_action=0,
        )
        return self.state.copy(), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Action outside the action space: {action}")

        battery, demand, renewable, period = map(int, self.state)
        surplus = max(0, renewable - demand)
        deficit = max(0, demand - renewable)

        battery_used = 0
        grid_bought = 0
        stored = 0
        sold = 0
        invalid_action = 0

        if action == 0:
            if deficit == 0 or battery == 0:
                invalid_action = 1
            else:
                battery_used = min(battery, deficit)
                battery -= battery_used
        elif action == 1:
            if deficit == 0:
                invalid_action = 1
            else:
                grid_bought = deficit
        elif action == 2:
            free_capacity = self.max_battery - battery
            if surplus == 0 or free_capacity == 0:
                invalid_action = 1
            else:
                stored = min(surplus, free_capacity)
                battery += stored
        elif action == 3:
            if surplus == 0:
                invalid_action = 1
            else:
                sold = surplus

        demand_covered = min(demand, renewable + battery_used + grid_bought)
        unmet_demand = max(0, demand - demand_covered)

        reward = 0.0
        reward += 2.0 * demand_covered
        reward -= 3.0 * unmet_demand
        reward -= 0.6 * grid_bought
        reward += 0.4 * sold
        reward -= 1.0 * invalid_action

        self.current_step += 1
        next_period = self.current_step % 4
        next_demand = self._sample_demand(next_period)
        next_renewable = self._sample_renewable(next_period)
        self.state = np.array([battery, next_demand, next_renewable, next_period], dtype=np.int64)

        terminated = False
        truncated = self.current_step >= self.max_steps
        info = self._build_info(
            battery_used=battery_used,
            grid_bought=grid_bought,
            stored=stored,
            sold=sold,
            unmet_demand=unmet_demand,
            invalid_action=invalid_action,
            demand_covered=demand_covered,
        )
        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self):
        battery, demand, renewable, period = map(int, self.state)
        print(
            f"step={self.current_step} period={period} battery={battery} demand={demand} renewable={renewable}"
        )

    def _sample_demand(self, period: int) -> int:
        distributions = {
            0: [0.20, 0.55, 0.25],
            1: [0.15, 0.45, 0.40],
            2: [0.10, 0.40, 0.50],
            3: [0.25, 0.55, 0.20],
        }
        return int(self._rng.choice([0, 1, 2], p=distributions[period]))

    def _sample_renewable(self, period: int) -> int:
        distributions = {
            0: [0.35, 0.45, 0.20],
            1: [0.15, 0.45, 0.40],
            2: [0.10, 0.35, 0.55],
            3: [0.30, 0.50, 0.20],
        }
        return int(self._rng.choice([0, 1, 2], p=distributions[period]))

    def _build_info(
        self,
        *,
        battery_used: int,
        grid_bought: int,
        stored: int,
        sold: int,
        unmet_demand: int,
        invalid_action: int,
        demand_covered: int | None = None,
    ) -> dict[str, Any]:
        battery, demand, renewable, period = map(int, self.state)
        if demand_covered is None:
            demand_covered = max(0, demand - unmet_demand)
        return {
            "battery_level": battery,
            "demand": demand,
            "renewable": renewable,
            "period": period,
            "battery_used": int(battery_used),
            "grid_bought": int(grid_bought),
            "stored": int(stored),
            "sold": int(sold),
            "demand_covered": int(demand_covered),
            "unmet_demand": int(unmet_demand),
            "invalid_action": int(invalid_action),
        }
