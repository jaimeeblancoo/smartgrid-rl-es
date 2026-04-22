from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.agents.qlearning_agent import QLearningAgent
from src.envs.factory import build_env_from_scenario
from src.training.config import get_training_config_for_scenario
from src.utils.io_helpers import save_dataframe
from src.utils.logger import log_episode
from src.utils.plotting import plot_rewards


def current_run_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def train_agent(scenario_name: str):
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
        coverage_history.append(min(total_covered / total_demand, 1.0) if total_demand else 0.0)
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


def run_training_pipeline(scenario_name: str):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="baseline")
    args = parser.parse_args()
    run_training_pipeline(args.scenario)
