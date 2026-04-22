from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.agents.qlearning_agent import QLearningAgent
from src.envs.factory import build_env_from_scenario
from src.training.config import EVALUATION_CONFIG, get_training_config_for_scenario
from src.utils.io_helpers import save_dataframe
from src.utils.plotting import plot_comparison


SCENARIOS = ["baseline", "winter", "summer", "demand_noise", "battery_loss", "combined_v2"]


def current_run_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


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
        coverages.append(min(covered_total / demand_total, 1.0) if demand_total else 0.0)
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


if __name__ == "__main__":
    main()
