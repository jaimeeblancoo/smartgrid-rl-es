"""Reward calculation for SmartGridEnvV3."""
from __future__ import annotations


def compute_reward_v3(info: dict, reward_weights: dict) -> float:
    """Compute the weighted V3 reward from environment metrics.

    Args:
        info: Metrics produced by ``SmartGridEnvV3.step``.
        reward_weights: Scenario or PSO-proposed reward weights.

    Returns:
        Scalar reward used by the Q-learning update.
    """
    demand_covered = float(info.get("demand_covered", 0))
    unmet_demand = float(info.get("unmet_demand", 0))
    grid_bought = float(info.get("grid_bought", 0))
    sold = float(info.get("sold", 0))
    invalid_action = float(info.get("invalid_action", 0))
    wasted_renewable = float(info.get("wasted_renewable", 0))
    risk_score = float(info.get("risk_score", 0))
    price_level = float(info.get("price_level", 0))

    grid_cost = grid_bought * (1 + price_level)

    return float(
        float(reward_weights.get("demand_covered", 0)) * demand_covered
        + float(reward_weights.get("unmet_demand", 0)) * unmet_demand
        + float(reward_weights.get("grid_bought", 0)) * grid_cost
        + float(reward_weights.get("sold", 0)) * sold
        + float(reward_weights.get("invalid_action", 0)) * invalid_action
        + float(reward_weights.get("wasted_renewable", 0)) * wasted_renewable
        + float(reward_weights.get("risk", 0)) * risk_score
    )
