from __future__ import annotations

TRAINING_CONFIG = {
    "seed": 42,
    "episodes": 2000,
    "max_steps_per_episode": 24,
    "alpha": 0.15,
    "gamma": 0.95,
    "epsilon": 1.00,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "log_every": 100,
}

SCENARIO_TRAINING_OVERRIDES = {
    "combined_v2": {
        "episodes": 4000,
        "epsilon_decay": 0.998,
    },
    "battery_loss": {
        "episodes": 3000,
        "epsilon_decay": 0.997,
    },
    "demand_noise": {
        "episodes": 2500,
        "epsilon_decay": 0.997,
    },
}

EVALUATION_CONFIG = {
    "eval_episodes": 100,
    "demo_seed": 123,
    "eval_seed_offset": 10000,
}


def get_training_config_for_scenario(scenario_name: str) -> dict:
    cfg = dict(TRAINING_CONFIG)
    cfg.update(SCENARIO_TRAINING_OVERRIDES.get(scenario_name, {}))
    return cfg