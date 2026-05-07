"""Simple fuzzy energy-risk scoring for SmartGridEnvV3."""
from __future__ import annotations


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _rising(value: float, start: float, end: float) -> float:
    if value <= start:
        return 0.0
    if value >= end:
        return 1.0
    return _clamp((value - start) / (end - start))


def _falling(value: float, start: float, end: float) -> float:
    if value <= start:
        return 1.0
    if value >= end:
        return 0.0
    return _clamp((end - value) / (end - start))


def _triangle(value: float, left: float, center: float, right: float) -> float:
    if value <= left or value >= right:
        return 0.0
    if value == center:
        return 1.0
    if value < center:
        return _clamp((value - left) / (center - left))
    return _clamp((right - value) / (right - center))


def discretize_risk(score: float) -> int:
    """Return 0 for low risk, 1 for medium risk, and 2 for high risk."""
    score = max(0.0, min(100.0, float(score)))
    if score < 34.0:
        return 0
    if score < 67.0:
        return 1
    return 2


def compute_energy_risk(
    battery: int,
    demand: int,
    renewable: int,
    price: int,
) -> dict[str, float | int | str]:
    """Compute a fuzzy energy-risk score from discrete V3 state values."""
    battery_value = max(0.0, min(4.0, float(battery)))
    demand_value = max(0.0, min(3.0, float(demand)))
    renewable_value = max(0.0, min(3.0, float(renewable)))
    price_value = max(0.0, min(2.0, float(price)))

    battery_low = _falling(battery_value, 0.0, 2.0)
    battery_medium = _triangle(battery_value, 1.0, 2.0, 3.0)
    battery_high = _rising(battery_value, 2.0, 4.0)

    demand_medium = _triangle(demand_value, 0.5, 1.5, 2.5)
    demand_high = _rising(demand_value, 1.0, 3.0)
    demand_peak = _rising(demand_value, 2.0, 3.0)

    renewable_low = _falling(renewable_value, 0.0, 2.0)
    renewable_high = _rising(renewable_value, 1.0, 3.0)

    price_high = _rising(price_value, 1.0, 2.0)

    high_risk = max(
        min(battery_low, demand_high),
        min(renewable_low, price_high),
        min(demand_peak, price_high),
    )
    medium_risk = min(battery_medium, demand_medium)
    low_risk = min(battery_high, renewable_high)

    activations = [
        (low_risk, 15.0),
        (medium_risk, 50.0),
        (high_risk, 90.0),
    ]
    total_activation = sum(weight for weight, _ in activations)

    if total_activation:
        risk_score = sum(weight * score for weight, score in activations) / total_activation
    else:
        demand_pressure = demand_value / 3.0
        renewable_shortage = 1.0 - (renewable_value / 3.0)
        battery_shortage = 1.0 - (battery_value / 4.0)
        price_pressure = price_value / 2.0
        risk_score = 100.0 * (
            0.35 * demand_pressure
            + 0.25 * renewable_shortage
            + 0.25 * battery_shortage
            + 0.15 * price_pressure
        )

    risk_score = max(0.0, min(100.0, float(risk_score)))
    risk_level = discretize_risk(risk_score)
    risk_names = {0: "low", 1: "medium", 2: "high"}

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_level_name": risk_names[risk_level],
    }


def compute_risk(battery: int, demand: int, renewable: int, price: int) -> float:
    """Backward-compatible numeric risk score for SmartGridEnvV3."""
    return float(compute_energy_risk(battery, demand, renewable, price)["risk_score"])
