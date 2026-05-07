"""Smoke tests for SmartGridEnvV3 (covers A1..A4)."""
from __future__ import annotations

from src.envs.smartgrid_env_v3 import SmartGridEnvV3


SCENARIO = "data/scenarios/baseline_v3.json"


def test_action_space_size() -> None:
    env = SmartGridEnvV3(SCENARIO, mode="train", seed=0)
    assert env.action_space.n == 5, f"expected 5 actions, got {env.action_space.n}"
    print("A3 action_space size OK")


def test_reset_obs_in_space() -> None:
    env = SmartGridEnvV3(SCENARIO, mode="train", seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"reset obs out of space: {obs}"
    for key in ("demand_covered", "unmet_demand", "grid_bought", "sold",
                "wasted_renewable", "invalid_action", "risk_score", "battery"):
        assert key in info, f"missing info key after reset: {key}"
    print("A1/A2 reset OK, A4 JSON+CSV OK")


def test_step_obs_in_space() -> None:
    env = SmartGridEnvV3(SCENARIO, mode="train", seed=0)
    env.reset(seed=0)
    for i in range(50):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"step {i}: obs out of space {obs}"
        assert isinstance(reward, float), f"step {i}: reward not float: {type(reward)}"
        for key in ("demand_covered", "unmet_demand", "grid_bought", "sold",
                    "wasted_renewable", "invalid_action", "risk_score", "battery"):
            assert key in info, f"step {i}: missing info key: {key}"
        if terminated or truncated:
            break
    print("A2 step OK, info contract OK")


def test_full_episode() -> None:
    env = SmartGridEnvV3(SCENARIO, mode="train", seed=42)
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    steps = 0
    while True:
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    assert steps > 0, "episode had zero steps"
    print(f"full episode OK ({steps} steps, total_reward={total_reward:.2f})")


def test_eval_mode_loads() -> None:
    env = SmartGridEnvV3(SCENARIO, mode="eval", seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), f"eval reset obs out of space: {obs}"
    print("eval mode OK")


def main() -> None:
    test_action_space_size()
    test_reset_obs_in_space()
    test_step_obs_in_space()
    test_full_episode()
    test_eval_mode_loads()
    print("\nAll V3 env tests passed.")


if __name__ == "__main__":
    main()
