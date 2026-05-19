"""Scikit-fuzzy energy-risk scoring for SmartGridEnvV3."""
from __future__ import annotations

from functools import lru_cache
from math import isfinite

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


RISK_NAMES = {0: "low", 1: "medium", 2: "high"}


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to a closed interval."""
    return max(lower, min(upper, value))


def _normalized_inputs(
    battery: int,
    demand: int,
    renewable: int,
    price: int,
) -> tuple[float, float, float, float]:
    """Convert V3 discrete state values to the fuzzy universes."""
    return (
        _clamp(float(battery), 0.0, 4.0),
        _clamp(float(demand), 0.0, 3.0),
        _clamp(float(renewable), 0.0, 3.0),
        _clamp(float(price), 0.0, 2.0),
    )


@lru_cache(maxsize=1)
def _build_control_system() -> ctrl.ControlSystem:
    """Create the reusable scikit-fuzzy control system."""
    battery = ctrl.Antecedent(np.arange(0.0, 4.01, 0.1), "battery")
    demand = ctrl.Antecedent(np.arange(0.0, 3.01, 0.1), "demand")
    renewable = ctrl.Antecedent(np.arange(0.0, 3.01, 0.1), "renewable")
    price = ctrl.Antecedent(np.arange(0.0, 2.01, 0.1), "price")
    risk = ctrl.Consequent(np.arange(0.0, 100.1, 1.0), "risk")

    battery["low"] = fuzz.trapmf(battery.universe, [0.0, 0.0, 1.0, 2.0])
    battery["medium"] = fuzz.trimf(battery.universe, [1.0, 2.0, 3.0])
    battery["high"] = fuzz.trapmf(battery.universe, [2.0, 3.0, 4.0, 4.0])

    demand["low"] = fuzz.trapmf(demand.universe, [0.0, 0.0, 0.5, 1.5])
    demand["medium"] = fuzz.trimf(demand.universe, [0.5, 1.5, 2.5])
    demand["high"] = fuzz.trapmf(demand.universe, [1.5, 2.0, 3.0, 3.0])
    demand["peak"] = fuzz.trimf(demand.universe, [2.0, 3.0, 3.0])

    renewable["low"] = fuzz.trapmf(renewable.universe, [0.0, 0.0, 1.0, 2.0])
    renewable["medium"] = fuzz.trimf(renewable.universe, [1.0, 2.0, 3.0])
    renewable["high"] = fuzz.trapmf(renewable.universe, [2.0, 2.5, 3.0, 3.0])

    price["low"] = fuzz.trapmf(price.universe, [0.0, 0.0, 0.5, 1.0])
    price["medium"] = fuzz.trimf(price.universe, [0.0, 1.0, 2.0])
    price["high"] = fuzz.trapmf(price.universe, [1.0, 1.5, 2.0, 2.0])

    risk["low"] = fuzz.trapmf(risk.universe, [0.0, 0.0, 20.0, 40.0])
    risk["medium"] = fuzz.trimf(risk.universe, [30.0, 50.0, 70.0])
    risk["high"] = fuzz.trapmf(risk.universe, [60.0, 80.0, 100.0, 100.0])

    high_demand = demand["high"] | demand["peak"]
    rules = [
        ctrl.Rule(battery["low"] & high_demand, risk["high"]),
        ctrl.Rule(renewable["low"] & price["high"], risk["high"]),
        ctrl.Rule(high_demand & price["high"], risk["high"]),
        ctrl.Rule(battery["medium"] & demand["medium"], risk["medium"]),
        ctrl.Rule(battery["high"] & renewable["high"], risk["low"]),
        ctrl.Rule(demand["low"] & renewable["high"], risk["low"]),
        ctrl.Rule(battery["high"] & demand["low"], risk["low"]),
        ctrl.Rule(renewable["high"] & price["low"], risk["low"]),
        ctrl.Rule(demand["low"] & price["low"], risk["low"]),
        ctrl.Rule(battery["low"], risk["medium"]),
        ctrl.Rule(battery["medium"], risk["medium"]),
        ctrl.Rule(battery["high"], risk["low"]),
        ctrl.Rule(renewable["low"] & demand["medium"], risk["medium"]),
        ctrl.Rule(renewable["low"] & price["medium"], risk["medium"]),
        ctrl.Rule(price["high"], risk["medium"]),
        ctrl.Rule(demand["peak"] & battery["low"], risk["high"]),
        ctrl.Rule(demand["peak"] & renewable["low"], risk["high"]),
    ]
    return ctrl.ControlSystem(rules)


def _fallback_risk_score(
    battery_value: float,
    demand_value: float,
    renewable_value: float,
    price_value: float,
) -> float:
    """Compute a deterministic fallback score if fuzzy inference has no output."""
    demand_pressure = demand_value / 3.0
    renewable_shortage = 1.0 - (renewable_value / 3.0)
    battery_shortage = 1.0 - (battery_value / 4.0)
    price_pressure = price_value / 2.0

    return 100.0 * (
        0.35 * demand_pressure
        + 0.25 * renewable_shortage
        + 0.25 * battery_shortage
        + 0.15 * price_pressure
    )


@lru_cache(maxsize=240)
def _compute_risk_score_cached(
    battery_value: float,
    demand_value: float,
    renewable_value: float,
    price_value: float,
) -> float:
    """Compute a risk score with a clean simulation for each cached state."""
    simulation = ctrl.ControlSystemSimulation(_build_control_system())
    simulation.input["battery"] = battery_value
    simulation.input["demand"] = demand_value
    simulation.input["renewable"] = renewable_value
    simulation.input["price"] = price_value
    simulation.compute()

    risk_score = simulation.output.get("risk")
    if risk_score is None or not isfinite(float(risk_score)):
        risk_score = _fallback_risk_score(
            battery_value,
            demand_value,
            renewable_value,
            price_value,
        )

    return _clamp(float(risk_score), 0.0, 100.0)


def discretize_risk(score: float) -> int:
    """Return 0 for low risk, 1 for medium risk, and 2 for high risk."""
    score = _clamp(float(score), 0.0, 100.0)
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
    """Compute a fuzzy energy-risk score from discrete V3 state values.

    Args:
        battery: Current battery level.
        demand: Current demand level.
        renewable: Current renewable generation level.
        price: Current price level.

    Returns:
        Dictionary with ``risk_score``, ``risk_level`` and ``risk_level_name``.
    """
    values = _normalized_inputs(battery, demand, renewable, price)
    risk_score = _compute_risk_score_cached(*values)
    risk_level = discretize_risk(risk_score)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_level_name": RISK_NAMES[risk_level],
    }


def compute_risk(battery: int, demand: int, renewable: int, price: int) -> float:
    """Backward-compatible numeric risk score for SmartGridEnvV3."""
    return float(compute_energy_risk(battery, demand, renewable, price)["risk_score"])
