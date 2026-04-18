from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.agents.qlearning_agent import QLearningAgent
from src.envs.smartgrid_env import SmartGridEnv
from src.training.config import TRAINING_CONFIG


def _state_to_columns(prefix: str, state) -> dict[str, int]:
    battery, demand_level, renewable_level, period = map(int, state)
    return {
        f"{prefix}_battery": battery,
        f"{prefix}_demand_level": demand_level,
        f"{prefix}_renewable_level": renewable_level,
        f"{prefix}_period": period,
    }


def save_demo_episode(env: SmartGridEnv, agent: QLearningAgent, output_path: str) -> None:
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state, _ = env.reset(seed=TRAINING_CONFIG["seed"])
    done = False
    step_num = 0
    rows = []

    action_names = {
        0: "use battery",
        1: "buy from grid",
        2: "store surplus",
        3: "sell surplus",
    }

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rows.append(
            {
                "step": step_num,
                **_state_to_columns("state", state),
                "action": action,
                "action_name": action_names[action],
                "reward": reward,
                "demand": info["demand"],
                "renewable": info["renewable"],
                "covered_demand": info["demand_covered"],
                "unmet_demand": info["unmet_demand"],
                "grid_purchase": info["grid_bought"],
                "battery_used": info["battery_used"],
                "stored_energy": info["stored"],
                "sold_energy": info["sold"],
                "battery_level": info["battery_level"],
                **_state_to_columns("next_state", next_state),
            }
        )

        state = next_state
        step_num += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")

    agent.epsilon = original_epsilon


def train():
    env = SmartGridEnv(
        max_steps=TRAINING_CONFIG["max_steps_per_episode"],
        seed=TRAINING_CONFIG["seed"],
    )

    agent = QLearningAgent(
        state_shape=env.observation_space.nvec,
        n_actions=env.action_space.n,
        alpha=TRAINING_CONFIG["alpha"],
        gamma=TRAINING_CONFIG["gamma"],
        epsilon=TRAINING_CONFIG["epsilon"],
        epsilon_min=TRAINING_CONFIG["epsilon_min"],
        epsilon_decay=TRAINING_CONFIG["epsilon_decay"],
        seed=TRAINING_CONFIG["seed"],
    )

    rewards_history = []

    for episode in range(TRAINING_CONFIG["episodes"]):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Episode {episode + 1} | "
                f"Average reward (last 100): {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("\nTraining finished.")
    print(f"Overall average reward: {np.mean(rewards_history):.2f}")

    plot_path = os.path.join(ROOT_DIR, "results", "plots", "training_rewards.png")
    csv_path = os.path.join(ROOT_DIR, "results", "logs", "demo_episode.csv")

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title("Training rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    save_demo_episode(env, agent, csv_path)

    print(f"Plot saved to: {plot_path}")
    print(f"Demo episode CSV saved to: {csv_path}")

    return agent, rewards_history


if __name__ == "__main__":
    train()