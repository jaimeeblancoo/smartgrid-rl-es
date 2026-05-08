from __future__ import annotations

import gymnasium as gym
import numpy as np


class BatteryLossWrapper(gym.Wrapper):
    """
    Simulates simple battery inefficiency while keeping the state discrete.

    At most one battery level can be lost per step.
    """

    def __init__(
        self,
        env: gym.Env,
        charge_loss_prob: float = 0.30,
        discharge_loss_prob: float = 0.20,
        leakage_prob: float = 0.12,
    ) -> None:
        super().__init__(env)
        self.charge_loss_prob = charge_loss_prob
        self.discharge_loss_prob = discharge_loss_prob
        self.leakage_prob = leakage_prob

    def reset(self, **kwargs):
        """Reset the environment and initialize battery-loss metadata."""
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["battery_losses"] = 0
        info["battery_level"] = int(obs[0])
        info["loss_reason"] = "none"
        return obs, info

    def step(self, action: int):
        """Run one step and optionally apply a one-level battery loss."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = np.array(obs, dtype=np.int64)
        info = dict(info)

        battery, demand, renewable, period = map(int, obs)
        total_loss = 0
        loss_reason = "none"

        if battery > 0:
            if info.get("stored", 0) > 0 and self.np_random.random() < self.charge_loss_prob:
                battery -= 1
                total_loss = 1
                loss_reason = "charge_loss"
            elif info.get("battery_used", 0) > 0 and self.np_random.random() < self.discharge_loss_prob:
                battery -= 1
                total_loss = 1
                loss_reason = "discharge_loss"
            elif self.np_random.random() < self.leakage_prob:
                battery -= 1
                total_loss = 1
                loss_reason = "leakage"

        updated_obs = np.array([battery, demand, renewable, period], dtype=np.int64)
        self.unwrapped.state = updated_obs.copy()

        info["battery_losses"] = int(total_loss)
        info["battery_level"] = int(battery)
        info["loss_reason"] = loss_reason

        return updated_obs, reward, terminated, truncated, info
