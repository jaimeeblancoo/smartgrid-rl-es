"""Reward-shaping wrapper for V2 smart-grid experiments.

The wrapper adjusts the scalar reward returned by the base environment without
changing its transition dynamics or observation space.
"""
from __future__ import annotations

import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    """
    Adjusts the reward signal without touching the base environment logic.
    """

    def __init__(
        self,
        env: gym.Env,
        demand_covered_bonus: float = 0.50,
        battery_bonus: float = 0.25,
        unmet_demand_penalty: float = 1.00,
        invalid_action_penalty: float = 0.25,
    ) -> None:
        """Create a reward-shaping wrapper.

        Args:
            env: Base Gymnasium environment to wrap.
            demand_covered_bonus: Bonus when all demand is covered.
            battery_bonus: Bonus for covering demand without grid purchases.
            unmet_demand_penalty: Penalty multiplier for unmet demand.
            invalid_action_penalty: Penalty applied to invalid actions.
        """
        super().__init__(env)
        self.demand_covered_bonus = demand_covered_bonus
        self.battery_bonus = battery_bonus
        self.unmet_demand_penalty = unmet_demand_penalty
        self.invalid_action_penalty = invalid_action_penalty

    def reset(self, **kwargs):
        """Reset the wrapped environment without changing its observation."""
        return self.env.reset(**kwargs)

    def step(self, action: int):
        """Run one step and add shaping terms to the base reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)
        shaped_reward = float(reward)

        grid_bought = info.get("grid_bought", 0) > 0
        battery_level = int(obs[0]) if len(obs) > 0 else 0
        unmet_demand = float(info.get("unmet_demand", 0))
        invalid_action = int(info.get("invalid_action", 0))

        if unmet_demand == 0:
            shaped_reward += self.demand_covered_bonus

        if battery_level > 0 and unmet_demand == 0 and not grid_bought:
            shaped_reward += self.battery_bonus

        if unmet_demand > 0:
            shaped_reward -= self.unmet_demand_penalty * unmet_demand

        if invalid_action:
            shaped_reward -= self.invalid_action_penalty

        info["base_reward"] = float(reward)
        info["shaped_reward"] = float(shaped_reward)
        info["grid_bought_flag"] = bool(grid_bought)

        return obs, shaped_reward, terminated, truncated, info
