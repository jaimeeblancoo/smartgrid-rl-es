from __future__ import annotations

from pathlib import Path

# ─── V2 configuration (unchanged) ─────────────────────────────────────────────

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


# ─── V3 configuration ──────────────────────────────────────────────────────────

V3_SCENARIOS = [
    "baseline_v3",
    "winter_peak",
    "summer_surplus",
    "grid_stress",
    "renewable_volatility",
]

V3_SCENARIO_PATHS = {
    "baseline_v3": Path("data/scenarios/baseline_v3.json"),
    "winter_peak": Path("data/scenarios/winter_peak.json"),
    "summer_surplus": Path("data/scenarios/summer_surplus.json"),
    "grid_stress": Path("data/scenarios/grid_stress.json"),
    "renewable_volatility": Path("data/scenarios/renewable_volatility.json"),
}

# Base V3 training config. max_steps_per_episode is intentionally absent:
# V3 uses max_steps from the scenario JSON instead.
V3_TRAINING_CONFIG = {
    "seed": 42,
    "episodes": 3000,
    "alpha": 0.15,
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.997,
    "log_every": 100,
}

# Per-scenario overrides for harder V3 scenarios.
V3_SCENARIO_TRAINING_OVERRIDES = {
    "winter_peak": {
        "episodes": 4000,
        "epsilon_decay": 0.998,
    },
    "grid_stress": {
        "episodes": 4000,
        "epsilon_decay": 0.998,
    },
}


def get_v3_scenario_path(scenario_name: str) -> Path:
    if scenario_name not in V3_SCENARIO_PATHS:
        raise ValueError(
            f"Unknown V3 scenario: '{scenario_name}'. "
            f"Available: {list(V3_SCENARIO_PATHS)}"
        )
    return V3_SCENARIO_PATHS[scenario_name]


def get_v3_training_config_for_scenario(scenario_name: str) -> dict:
    cfg = dict(V3_TRAINING_CONFIG)
    cfg.update(V3_SCENARIO_TRAINING_OVERRIDES.get(scenario_name, {}))
    return cfg
