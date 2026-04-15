from __future__ import annotations

import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.agents.qlearning_agent import QLearningAgent
from src.envs.smartgrid_env import SmartGridEnv
from src.training.config import TRAINING_CONFIG


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
    )

    rewards_history = []

    for episode in range(TRAINING_CONFIG["episodes"]):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Episodio {episode + 1} | "
                f"Recompensa media (últimos 100): {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("\nEntrenamiento terminado.")
    print(f"Recompensa media total: {np.mean(rewards_history):.2f}")

    return agent, rewards_history


if __name__ == "__main__":
    train()