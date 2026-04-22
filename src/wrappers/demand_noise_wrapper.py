from __future__ import annotations

import gymnasium as gym
import numpy as np


class DemandNoiseWrapper(gym.Wrapper):
    """
    Adds occasional demand spikes during the episode.

    This wrapper does not modify the initial observation returned by reset().
    Noise is applied only after calling step().
    """

    def __init__(
        self,
        env: gym.Env,
        spike_probability: float = 0.15,
        spike_size: int = 1,
    ) -> None:
        super().__init__(env)
        self.spike_probability = spike_probability
        self.spike_size = spike_size

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["noise_event"] = "none"
        info["demand_noise"] = 0
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = np.array(obs, dtype=np.int64)
        info = dict(info)

        battery, demand, renewable, period = map(int, obs)
        noise = 0
        noise_event = "none"

        if self.np_random.random() < self.spike_probability:
            demand_max = int(self.observation_space.nvec[1] - 1)
            new_demand = min(demand + self.spike_size, demand_max)

            if new_demand != demand:
                noise = new_demand - demand
                demand = new_demand
                noise_event = "demand_spike"

        updated_obs = np.array([battery, demand, renewable, period], dtype=np.int64)
        self.unwrapped.state = updated_obs.copy()

        info["noise_event"] = noise_event
        info["demand_noise"] = int(noise)
        info["post_noise_demand"] = int(demand)

        return updated_obs, reward, terminated, truncated, info
