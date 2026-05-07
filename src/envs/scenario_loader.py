"""Load V3 scenario definitions from JSON files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "name", "mode", "train_csv_path", "eval_csv_path",
    "battery_capacity", "initial_battery", "max_steps", "reward_weights",
)

REQUIRED_REWARD_KEYS = (
    "demand_covered", "unmet_demand", "grid_bought", "sold",
    "invalid_action", "wasted_renewable", "risk",
)


def load_scenario(json_path: str | Path) -> dict[str, Any]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    missing = [k for k in REQUIRED_FIELDS if k not in config]
    if missing:
        raise ValueError(f"Missing required fields in scenario {json_path.name}: {missing}")
    if config["mode"] not in {"csv", "markov", "csv_markov_noise"}:
        raise ValueError(f"Unsupported scenario mode: {config['mode']}")
    weights = config["reward_weights"]
    missing_w = [k for k in REQUIRED_REWARD_KEYS if k not in weights]
    if missing_w:
        raise ValueError(f"Missing required reward weights in {json_path.name}: {missing_w}")
    return config
