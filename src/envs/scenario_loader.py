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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_from_repo_root(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def load_scenario(json_path: str | Path) -> dict[str, Any]:
    """Load and validate a V3 JSON scenario configuration.

    Args:
        json_path: Path to the scenario JSON file.

    Returns:
        Scenario dictionary with CSV paths resolved from the repository root.

    Raises:
        FileNotFoundError: If the JSON or referenced CSV files do not exist.
        ValueError: If required fields or reward weights are missing.
        NotImplementedError: If a declared scenario mode is not implemented.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    missing = [k for k in REQUIRED_FIELDS if k not in config]
    if missing:
        raise ValueError(f"Missing required fields in scenario {json_path.name}: {missing}")
    mode = config["mode"]
    if mode in {"markov", "csv_markov_noise"}:
        raise NotImplementedError(
            f"Scenario mode '{mode}' is declared but not implemented in SmartGridEnvV3."
        )
    if mode != "csv":
        raise ValueError(f"Unsupported scenario mode: {mode}")
    weights = config["reward_weights"]
    missing_w = [k for k in REQUIRED_REWARD_KEYS if k not in weights]
    if missing_w:
        raise ValueError(f"Missing required reward weights in {json_path.name}: {missing_w}")
    for path_key in ("train_csv_path", "eval_csv_path"):
        csv_path = _resolve_from_repo_root(config[path_key])
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Scenario {json_path.name} points to a missing CSV file: {config[path_key]}"
            )
        config[path_key] = str(csv_path)
    return config
