from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np


class BatteryLossWrapper(gym.Wrapper):
    """Applies simple battery inefficiencies while keeping the state discrete."""

    def __init__(
        self,
        env: gym.Env,
        charge_loss_prob: float = 0.30,
        discharge_loss_prob: float = 0.20,
        leakage_prob: float = 0.12,
        loss_penalty: float = 0.75,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.charge_loss_prob = charge_loss_prob
        self.discharge_loss_prob = discharge_loss_prob
        self.leakage_prob = leakage_prob
        self.loss_penalty = loss_penalty
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = np.array(obs, dtype=np.int64)
        battery, demand, renewable, period = map(int, obs)
        total_loss = 0

        if info.get("stored", 0) > 0 and self._rng.random() < self.charge_loss_prob and battery > 0:
            battery -= 1
            total_loss += 1

        if info.get("battery_used", 0) > 0 and self._rng.random() < self.discharge_loss_prob and battery > 0:
            battery -= 1
            total_loss += 1

        if battery > 0 and self._rng.random() < self.leakage_prob:
            battery -= 1
            total_loss += 1

        obs = np.array([battery, demand, renewable, period], dtype=np.int64)
        self.unwrapped.state = obs.copy()

        reward = float(reward) - self.loss_penalty * total_loss

        info = dict(info)
        info["battery_losses"] = total_loss
        info["battery_level"] = int(battery)

        return obs, reward, terminated, truncated, info