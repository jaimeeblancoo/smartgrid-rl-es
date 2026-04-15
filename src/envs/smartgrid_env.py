from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SmartGridEnv(gym.Env):
    """Small discrete environment for SmartGrid-ES V1.

    State = (battery, demand, renewable, period)
    - battery: 0 low, 1 medium, 2 high
    - demand: 0 low, 1 medium, 2 high
    - renewable: 0 low, 1 medium, 2 high
    - period: 0 morning, 1 afternoon, 2 night

    Actions:
    - 0: use battery to cover deficit
    - 1: buy energy from the grid
    - 2: store surplus if available
    - 3: sell surplus
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, max_steps: int = 8, seed: Optional[int] = None) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.max_battery = 2

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])

        self._rng = np.random.default_rng(seed)
        self.current_step = 0
        self.state = np.array([1, 1, 1, 0], dtype=np.int64)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = 0
        battery = int(self._rng.integers(0, 3))
        period = 0
        demand = self._sample_demand(period)
        renewable = self._sample_renewable(period)
        self.state = np.array([battery, demand, renewable, period], dtype=np.int64)
        return self.state.copy(), {}

    def step(self, action: int):
        battery, demand_level, renewable_level, period = map(int, self.state)

        demand = demand_level + 1
        renewable = renewable_level + 1

        demand_covered = min(demand, renewable)
        deficit = max(demand - renewable, 0)
        surplus = max(renewable - demand, 0)

        battery_used = 0
        grid_bought = 0
        stored = 0
        sold = 0
        invalid_action = 0

        if action == 0:
            battery_used = min(battery, deficit)
            battery -= battery_used
            demand_covered += battery_used
            if deficit == 0:
                invalid_action = 1

        elif action == 1:
            grid_bought = deficit
            demand_covered += grid_bought
            if deficit == 0:
                invalid_action = 1

        elif action == 2:
            available_capacity = self.max_battery - battery
            stored = min(surplus, available_capacity)
            battery += stored
            if surplus == 0:
                invalid_action = 1

        elif action == 3:
            sold = surplus
            if surplus == 0:
                invalid_action = 1

        else:
            raise ValueError(f"Invalid action: {action}")

        unmet_demand = max(demand - demand_covered, 0)

        reward = 0.0
        reward += 2.0 * demand_covered
        reward -= 4.0 * unmet_demand
        reward -= 2.0 * grid_bought
        reward += 1.0 * stored
        reward += 1.5 * sold
        reward += 0.5 * battery_used
        reward -= 1.0 * invalid_action

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        next_period = self.current_step % 3
        next_demand = self._sample_demand(next_period)
        next_renewable = self._sample_renewable(next_period)

        self.state = np.array([battery, next_demand, next_renewable, next_period], dtype=np.int64)

        info = {
            "demand": demand,
            "renewable": renewable,
            "demand_covered": demand_covered,
            "unmet_demand": unmet_demand,
            "grid_bought": grid_bought,
            "battery_used": battery_used,
            "stored": stored,
            "sold": sold,
            "battery_level": battery,
        }

        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        battery, demand, renewable, period = map(int, self.state)
        period_names = {0: "morning", 1: "afternoon", 2: "night"}
        print(
            f"Step={self.current_step} | "
            f"Battery={battery} | Demand={demand} | Renewable={renewable} | "
            f"Period={period_names[period]}"
        )

    def _sample_demand(self, period: int) -> int:
        distributions = {
            0: [0.5, 0.4, 0.1],
            1: [0.2, 0.5, 0.3],
            2: [0.1, 0.4, 0.5],
        }
        return int(self._rng.choice([0, 1, 2], p=distributions[period]))

    def _sample_renewable(self, period: int) -> int:
        distributions = {
            0: [0.2, 0.5, 0.3],
            1: [0.1, 0.3, 0.6],
            2: [0.6, 0.3, 0.1],
        }
        return int(self._rng.choice([0, 1, 2], p=distributions[period]))
