from __future__ import annotations

import gymnasium as gym
import numpy as np


class SeasonWrapper(gym.Wrapper):
    """Apply a simple seasonal profile to demand and renewable generation."""

    def __init__(self, env: gym.Env, season: str, seed: int | None = None) -> None:
        """Create a seasonal wrapper for a V2 environment.

        Args:
            env: Base Gymnasium environment to wrap.
            season: Either ``"winter"`` or ``"summer"``.
            seed: Optional seed for reproducible seasonal adjustments.
        """
        super().__init__(env)
        if season not in {"winter", "summer"}:
            raise ValueError(f"Unsupported season: {season}")
        self.season = season
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        """Reset the wrapped environment and apply the seasonal profile."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._apply_season(obs)
        self.unwrapped.state = obs.copy()

        info = dict(info)
        info["season"] = self.season
        info["demand"] = int(obs[1])
        info["renewable"] = int(obs[2])
        return obs, info

    def step(self, action):
        """Run one step and adjust the resulting observation by season."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._apply_season(obs)
        self.unwrapped.state = obs.copy()

        info = dict(info)
        info["season"] = self.season
        info["demand"] = int(obs[1])
        info["renewable"] = int(obs[2])
        return obs, reward, terminated, truncated, info

    def _apply_season(self, obs):
        """Return a season-adjusted V2 observation."""
        battery, demand, renewable, period = map(int, obs)

        if self.season == "winter":
            if self._rng.random() < 0.35:
                demand = min(2, demand + 1)
            if self._rng.random() < 0.25:
                renewable = max(0, renewable - 1)
        else:
            if self._rng.random() < 0.30:
                renewable = min(2, renewable + 1)
            if self._rng.random() < 0.20:
                demand = max(0, demand - 1)

        return np.array([battery, demand, renewable, period], dtype=np.int64)
