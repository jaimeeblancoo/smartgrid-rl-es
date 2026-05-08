from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.agents.qlearning_agent import QLearningAgent
from src.envs.factory import build_env_from_scenario
from src.envs.smartgrid_env_v3 import SmartGridEnvV3
from src.training.config import (
    EVALUATION_CONFIG,
    V3_SCENARIOS,
    get_training_config_for_scenario,
    get_v3_scenario_path,
    get_v3_training_config_for_scenario,
)
from src.utils.io_helpers import save_dataframe
from src.utils.plotting import plot_comparison


SCENARIOS = ["baseline", "winter", "summer", "demand_noise", "battery_loss", "combined_v2"]


def current_run_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ─── V2 evaluation (unchanged) ────────────────────────────────────────────────

def evaluate_agent(scenario_name: str) -> dict:
    train_cfg = get_training_config_for_scenario(scenario_name)
    env = build_env_from_scenario(
        scenario_name=scenario_name,
        max_steps=train_cfg["max_steps_per_episode"],
        seed=train_cfg["seed"],
    )
    agent = QLearningAgent(
        state_shape=env.observation_space.nvec,
        action_size=env.action_space.n,
        alpha=train_cfg["alpha"],
        gamma=train_cfg["gamma"],
        epsilon=0.0,
        epsilon_min=train_cfg["epsilon_min"],
        epsilon_decay=train_cfg["epsilon_decay"],
        seed=train_cfg["seed"],
    )
    model_path = Path("results/models") / f"q_table_{scenario_name}.npy"
    agent.load(model_path)

    rewards = []
    coverages = []
    grid_buys = []
    battery_levels = []
    sold_energy = []

    for episode_idx in range(EVALUATION_CONFIG["eval_episodes"]):
        state, _ = env.reset(seed=EVALUATION_CONFIG["eval_seed_offset"] + episode_idx)
        done = False
        reward_total = 0.0
        demand_total = 0.0
        covered_total = 0.0
        grid_total = 0.0
        sold_total = 0.0
        last_battery = 0

        while not done:
            action = agent.greedy_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward_total += reward
            demand_total += info["demand"]
            covered_total += info["demand_covered"]
            grid_total += info["grid_bought"]
            sold_total += info["sold"]
            last_battery = info["battery_level"]

        rewards.append(reward_total)
        coverages.append(covered_total / demand_total if demand_total else 0.0)
        grid_buys.append(grid_total)
        sold_energy.append(sold_total)
        battery_levels.append(last_battery)

    return {
        "scenario": scenario_name,
        "avg_reward": float(sum(rewards) / len(rewards)),
        "avg_coverage": float(sum(coverages) / len(coverages)),
        "avg_grid_bought": float(sum(grid_buys) / len(grid_buys)),
        "avg_battery_end": float(sum(battery_levels) / len(battery_levels)),
        "avg_sold": float(sum(sold_energy) / len(sold_energy)),
    }


def main():
    rows = []
    missing_models = []

    for scenario_name in SCENARIOS:
        model_path = Path("results/models") / f"q_table_{scenario_name}.npy"
        if not model_path.exists():
            missing_models.append(scenario_name)
            continue
        rows.append(evaluate_agent(scenario_name))

    if not rows:
        print("No trained models were found in results/models. Train at least one scenario before evaluation.")
        return

    summary = pd.DataFrame(rows)
    date_tag = current_run_date()
    summary_path = Path("results/summaries") / f"smartgrid_evaluation_summary_{date_tag}.csv"
    plot_path = Path("results/plots") / f"evaluation_comparison_{date_tag}.png"
    save_dataframe(summary, summary_path)
    plot_comparison(summary, str(plot_path))
    print(summary.to_string(index=False))
    print()
    print(f"Evaluation summary saved to: {summary_path}")
    print(f"Comparison plot saved to: {plot_path}")

    if missing_models:
        print()
        print("Skipped scenarios without trained models:")
        for scenario_name in missing_models:
            print(f"- {scenario_name}")


# ─── V3 evaluation ────────────────────────────────────────────────────────────

def evaluate_agent_v3(scenario_name: str) -> dict:
    """Evaluate a trained V3 agent using the eval CSV timeseries."""
    cfg = get_v3_training_config_for_scenario(scenario_name)
    scenario_path = get_v3_scenario_path(scenario_name)
    env = SmartGridEnvV3(scenario_path=scenario_path, mode="eval", seed=cfg["seed"])
    agent = QLearningAgent(
        state_shape=env.observation_space.nvec,
        action_size=env.action_space.n,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon=0.0,
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        seed=cfg["seed"],
    )
    model_path = Path("results/v3/models") / f"q_table_{scenario_name}.npy"
    agent.load(model_path)

    rewards = []
    coverages = []
    grid_buys = []
    battery_levels = []
    sold_energy = []
    unmet_demands = []
    wasted_renewables = []
    risk_scores = []
    invalid_rates = []

    for episode_idx in range(EVALUATION_CONFIG["eval_episodes"]):
        state, _ = env.reset(seed=EVALUATION_CONFIG["eval_seed_offset"] + episode_idx)
        done = False
        reward_total = 0.0
        demand_total = 0.0
        covered_total = 0.0
        grid_total = 0.0
        sold_total = 0.0
        unmet_total = 0.0
        wasted_total = 0.0
        risk_total = 0.0
        invalid_total = 0
        steps = 0
        last_battery = 0

        while not done:
            action = agent.greedy_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward_total += reward
            demand_total += info["demand"]
            covered_total += info["demand_covered"]
            grid_total += info["grid_bought"]
            sold_total += info["sold"]
            unmet_total += info["unmet_demand"]
            wasted_total += info["wasted_renewable"]
            risk_total += info["risk_score"]
            invalid_total += info["invalid_action"]
            last_battery = info["battery"]  # V3 uses "battery", not "battery_level"
            steps += 1

        rewards.append(reward_total)
        coverages.append(covered_total / demand_total if demand_total else 0.0)
        grid_buys.append(grid_total)
        sold_energy.append(sold_total)
        battery_levels.append(last_battery)
        unmet_demands.append(unmet_total)
        wasted_renewables.append(wasted_total)
        risk_scores.append(risk_total / steps if steps else 0.0)
        invalid_rates.append(invalid_total / steps if steps else 0.0)

    n = len(rewards)
    return {
        "scenario": scenario_name,
        "avg_reward": float(sum(rewards) / n),
        "avg_coverage": float(sum(coverages) / n),
        "avg_unmet_demand": float(sum(unmet_demands) / n),
        "avg_grid_bought": float(sum(grid_buys) / n),
        "avg_sold": float(sum(sold_energy) / n),
        "avg_wasted_renewable": float(sum(wasted_renewables) / n),
        "avg_battery_end": float(sum(battery_levels) / n),
        "avg_risk_score": float(sum(risk_scores) / n),
        "invalid_action_rate": float(sum(invalid_rates) / n),
    }


def main_v3(scenario_names: list) -> None:
    """Evaluate all given V3 scenarios and save a combined summary."""
    rows = []
    missing_models = []

    for scenario_name in scenario_names:
        model_path = Path("results/v3/models") / f"q_table_{scenario_name}.npy"
        if not model_path.exists():
            missing_models.append(scenario_name)
            continue
        print(f"Evaluating V3 scenario: {scenario_name}")
        rows.append(evaluate_agent_v3(scenario_name))

    if not rows:
        print(
            "No trained V3 models found in results/v3/models. "
            "Run training for at least one V3 scenario first."
        )
        return

    summary = pd.DataFrame(rows)
    date_tag = current_run_date()
    summary_path = (
        Path("results/v3/summaries")
        / f"smartgrid_v3_evaluation_summary_{date_tag}.csv"
    )
    plot_path = (
        Path("results/v3/plots")
        / f"v3_evaluation_comparison_{date_tag}.png"
    )
    save_dataframe(summary, summary_path)
    plot_comparison(summary, str(plot_path))
    print(summary.to_string(index=False))
    print()
    print(f"V3 evaluation summary saved to: {summary_path}")
    print(f"V3 comparison plot saved to:    {plot_path}")

    if missing_models:
        print()
        print("Skipped V3 scenarios without trained models:")
        for scenario_name in missing_models:
            print(f"- {scenario_name}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartGrid-ES evaluation script.")
    parser.add_argument(
        "--version",
        choices=["v2", "v3"],
        default="v3",
        help="Environment version to evaluate (default: v3).",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Single scenario to evaluate (V3 only). Defaults to baseline_v3.",
    )
    parser.add_argument(
        "--all-v3",
        action="store_true",
        help="Evaluate all V3 scenarios.",
    )
    args = parser.parse_args()

    if args.version == "v2":
        main()
    else:
        if args.all_v3:
            main_v3(V3_SCENARIOS)
        else:
            scenario = args.scenario or "baseline_v3"
            main_v3([scenario])
