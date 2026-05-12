"""Training entry points for SmartGrid-ES agents.

The module supports the legacy V2 comparison scenarios and the final V3
pipeline based on ``SmartGridEnvV3``, synthetic JSON/CSV scenarios and tabular
Q-learning. It writes trained Q-tables, reward plots and demo episode CSVs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.agents.qlearning_agent import QLearningAgent
from src.envs.factory import build_env_from_scenario
from src.envs.smartgrid_env_v3 import SmartGridEnvV3
from src.training.config import (
    V3_SCENARIOS,
    get_training_config_for_scenario,
    get_v3_scenario_path,
    get_v3_training_config_for_scenario,
)
from src.utils.io_helpers import save_dataframe
from src.utils.logger import log_episode
from src.utils.plotting import plot_rewards


def current_run_date() -> str:
    """Return the date tag used in generated result filenames."""
    return datetime.now().strftime("%Y-%m-%d")


# V2 training.

def train_agent(scenario_name: str):
    """Train a tabular Q-learning agent on one V2 scenario.

    Args:
        scenario_name: Name of the configured V2 scenario.

    Returns:
        Tuple containing the trained agent, reward history, environment and
        training configuration.
    """
    cfg = get_training_config_for_scenario(scenario_name)
    env = build_env_from_scenario(
        scenario_name=scenario_name,
        max_steps=cfg["max_steps_per_episode"],
        seed=cfg["seed"],
    )
    obs_space = env.observation_space
    agent = QLearningAgent(
        state_shape=obs_space.nvec,
        action_size=env.action_space.n,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        seed=cfg["seed"],
    )

    rewards_history = []
    coverage_history = []
    grid_history = []

    for episode in range(1, cfg["episodes"] + 1):
        state, _ = env.reset(seed=cfg["seed"] + episode)
        done = False
        total_reward = 0.0
        total_demand = 0.0
        total_covered = 0.0
        total_grid = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            total_demand += info["demand"]
            total_covered += info["demand_covered"]
            total_grid += info["grid_bought"]

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        coverage_history.append(total_covered / total_demand if total_demand else 0.0)
        grid_history.append(total_grid)

        if episode % cfg["log_every"] == 0:
            log_episode(
                episode,
                avg_reward=sum(rewards_history[-cfg["log_every"] :]) / cfg["log_every"],
                avg_coverage=sum(coverage_history[-cfg["log_every"] :]) / cfg["log_every"],
                avg_grid=sum(grid_history[-cfg["log_every"] :]) / cfg["log_every"],
                epsilon=agent.epsilon,
            )

    return agent, rewards_history, env, cfg


def save_demo_episode(env, agent: QLearningAgent, scenario_name: str, seed: int) -> Path:
    """Run one greedy V2 demo episode and save it as a CSV file.

    Args:
        env: V2 Gymnasium-compatible environment.
        agent: Trained tabular Q-learning agent.
        scenario_name: Scenario name used in the output filename.
        seed: Reset seed for reproducible demo generation.

    Returns:
        Path to the saved demo CSV.
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rows = []
    state, _ = env.reset(seed=seed)
    done = False
    step_idx = 0

    while not done:
        action = agent.greedy_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append(
            {
                "step": step_idx,
                "battery": int(state[0]),
                "demand": int(info["demand"]),
                "renewable": int(info["renewable"]),
                "period": int(state[3]),
                "action": int(action),
                "reward": float(reward),
                "grid_bought": int(info["grid_bought"]),
                "demand_covered": int(info["demand_covered"]),
                "unmet_demand": int(info["unmet_demand"]),
            }
        )
        state = next_state
        step_idx += 1

    agent.epsilon = original_epsilon
    df = pd.DataFrame(rows)
    output_path = Path("results/demos") / f"smartgrid_demo_episode_{scenario_name}_{current_run_date()}.csv"
    save_dataframe(df, output_path)
    return output_path


def run_training_pipeline(scenario_name: str) -> None:
    """Run the complete V2 training pipeline for one scenario.

    Args:
        scenario_name: Name of the configured V2 scenario.
    """
    agent, rewards_history, env, cfg = train_agent(scenario_name=scenario_name)

    model_path = Path("results/models") / f"q_table_{scenario_name}.npy"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)

    plot_path = Path("results/plots") / f"training_rewards_{scenario_name}.png"
    plot_rewards(rewards_history, str(plot_path))
    demo_path = save_demo_episode(env=env, agent=agent, scenario_name=scenario_name, seed=cfg["seed"] + 777)

    print()
    print(f"Training completed for {scenario_name}")
    print(f"Model saved to: {model_path}")
    print(f"Training plot saved to: {plot_path}")
    print(f"Demo episode CSV saved to: {demo_path}")


# V3 training.

def train_agent_v3(scenario_name: str):
    """Train a tabular Q-learning agent on a V3 scenario loaded from JSON.

    Args:
        scenario_name: Name of a configured V3 scenario.

    Returns:
        The trained agent, reward history, environment and training config.
    """
    cfg = get_v3_training_config_for_scenario(scenario_name)
    scenario_path = get_v3_scenario_path(scenario_name)
    env = SmartGridEnvV3(scenario_path=scenario_path, mode="train", seed=cfg["seed"])
    agent = QLearningAgent(
        state_shape=env.observation_space.nvec,
        action_size=env.action_space.n,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        seed=cfg["seed"],
    )

    rewards_history = []
    coverage_history = []
    grid_history = []

    for episode in range(1, cfg["episodes"] + 1):
        state, _ = env.reset(seed=cfg["seed"] + episode)
        done = False
        total_reward = 0.0
        total_demand = 0.0
        total_covered = 0.0
        total_grid = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            total_demand += info["demand"]
            total_covered += info["demand_covered"]
            total_grid += info["grid_bought"]

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        coverage_history.append(total_covered / total_demand if total_demand else 0.0)
        grid_history.append(total_grid)

        if episode % cfg["log_every"] == 0:
            log_episode(
                episode,
                avg_reward=sum(rewards_history[-cfg["log_every"] :]) / cfg["log_every"],
                avg_coverage=sum(coverage_history[-cfg["log_every"] :]) / cfg["log_every"],
                avg_grid=sum(grid_history[-cfg["log_every"] :]) / cfg["log_every"],
                epsilon=agent.epsilon,
            )

    return agent, rewards_history, env, cfg


def save_demo_episode_v3(
    env: SmartGridEnvV3, agent: QLearningAgent, scenario_name: str, seed: int
) -> Path:
    """Run one greedy episode and save a detailed V3 demo CSV.

    The demo includes the V3 risk fields so the dashboard can visualize the
    same risk signal used by the reward function.

    Args:
        env: V3 Gymnasium environment.
        agent: Trained tabular Q-learning agent.
        scenario_name: Scenario name used in the output filename.
        seed: Reset seed for reproducible demo generation.

    Returns:
        Path to the saved demo CSV.
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    rows = []
    state, _ = env.reset(seed=seed)
    done = False
    step_idx = 0

    while not done:
        action = agent.greedy_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # V3 state layout: [battery, demand, renewable, price, period, weather]
        rows.append(
            {
                "step": step_idx,
                "battery": int(state[0]),
                "demand": int(info["demand"]),
                "renewable": int(info["renewable"]),
                "price_level": int(state[3]),
                "period": int(state[4]),
                "action": int(action),
                "reward": float(reward),
                "grid_bought": int(info["grid_bought"]),
                "sold": int(info["sold"]),
                "demand_covered": int(info["demand_covered"]),
                "unmet_demand": int(info["unmet_demand"]),
                "wasted_renewable": int(info["wasted_renewable"]),
                "invalid_action": int(info["invalid_action"]),
                "risk_score": float(info["risk_score"]),
                "risk_level": int(info["risk_level"]),
                "risk_level_name": str(info["risk_level_name"]),
            }
        )
        state = next_state
        step_idx += 1

    agent.epsilon = original_epsilon
    df = pd.DataFrame(rows)
    date_tag = current_run_date()
    output_path = (
        Path("results/v3/demos")
        / f"smartgrid_v3_demo_episode_{scenario_name}_{date_tag}.csv"
    )
    save_dataframe(df, output_path)
    return output_path


def run_training_pipeline_v3(scenario_name: str) -> None:
    """Run the complete V3 training pipeline for one scenario.

    Args:
        scenario_name: Name of the configured V3 scenario.
    """
    print(f"\nTraining V3 scenario: {scenario_name}")
    agent, rewards_history, env, cfg = train_agent_v3(scenario_name=scenario_name)

    date_tag = current_run_date()

    model_path = Path("results/v3/models") / f"q_table_{scenario_name}.npy"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)

    plot_path = (
        Path("results/v3/plots")
        / f"training_rewards_{scenario_name}_{date_tag}.png"
    )
    plot_rewards(rewards_history, str(plot_path))

    demo_path = save_demo_episode_v3(
        env=env,
        agent=agent,
        scenario_name=scenario_name,
        seed=cfg["seed"] + 777,
    )

    print()
    print(f"V3 training completed for: {scenario_name}")
    print(f"Model saved to:            {model_path}")
    print(f"Training plot saved to:    {plot_path}")
    print(f"Demo episode CSV saved to: {demo_path}")


# Entry point.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartGrid-ES training script.")
    parser.add_argument(
        "--version",
        choices=["v2", "v3"],
        default="v3",
        help="Environment version to train (default: v3).",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario name to train (default: baseline_v3 for V3, baseline for V2).",
    )
    parser.add_argument(
        "--all-v3",
        action="store_true",
        help="Train all V3 scenarios sequentially.",
    )
    args = parser.parse_args()

    if args.version == "v2":
        scenario = args.scenario or "baseline"
        run_training_pipeline(scenario)
    else:
        scenario = args.scenario or "baseline_v3"
        if args.all_v3:
            for sc in V3_SCENARIOS:
                run_training_pipeline_v3(sc)
        else:
            run_training_pipeline_v3(scenario)
