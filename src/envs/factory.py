from __future__ import annotations

from src.envs.smartgrid_env import SmartGridEnv
from src.wrappers.battery_loss_wrapper import BatteryLossWrapper
from src.wrappers.demand_noise_wrapper import DemandNoiseWrapper
from src.wrappers.reward_shaping_wrapper import RewardShapingWrapper
from src.wrappers.season_wrapper import SeasonWrapper


SCENARIO_CONFIGS = {
    "baseline": {},
    "winter": {"season": "winter"},
    "summer": {"season": "summer"},
    "demand_noise": {"spike_prob": 0.20, "drop_prob": 0.05},
    "battery_loss": {
        "charge_loss_prob": 0.30,
        "discharge_loss_prob": 0.20,
        "leakage_prob": 0.12,
    },
    "combined_v2": {
        "season": "winter",
        "spike_prob": 0.20,
        "drop_prob": 0.05,
        "charge_loss_prob": 0.30,
        "discharge_loss_prob": 0.20,
        "leakage_prob": 0.12,
    },
}


def build_env_from_scenario(scenario_name: str, max_steps: int, seed: int | None = None):
    if scenario_name not in SCENARIO_CONFIGS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    config = SCENARIO_CONFIGS[scenario_name]
    env = SmartGridEnv(max_steps=max_steps, seed=seed)

    if scenario_name == "baseline":
        return env

    if scenario_name in {"winter", "summer"}:
        return SeasonWrapper(env, season=config["season"], seed=seed)

    if scenario_name == "demand_noise":
        return DemandNoiseWrapper(
            env,
            spike_prob=config["spike_prob"],
            drop_prob=config["drop_prob"],
            seed=seed,
        )

    if scenario_name == "battery_loss":
        return BatteryLossWrapper(
            env,
            charge_loss_prob=config["charge_loss_prob"],
            discharge_loss_prob=config["discharge_loss_prob"],
            leakage_prob=config["leakage_prob"],
            seed=seed,
        )

    if scenario_name == "combined_v2":
        env = SeasonWrapper(env, season=config["season"], seed=seed)
        env = DemandNoiseWrapper(
            env,
            spike_prob=config["spike_prob"],
            drop_prob=config["drop_prob"],
            seed=seed,
        )
        env = BatteryLossWrapper(
            env,
            charge_loss_prob=config["charge_loss_prob"],
            discharge_loss_prob=config["discharge_loss_prob"],
            leakage_prob=config["leakage_prob"],
            seed=seed,
        )
        env = RewardShapingWrapper(env)
        return env

    raise ValueError(f"Scenario is configured but not implemented in factory: {scenario_name}")