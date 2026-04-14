from __future__ import annotations

import random

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        state_shape,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.state_shape = tuple(state_shape)
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros(self.state_shape + (n_actions,), dtype=np.float32)

    def choose_action(self, state) -> int:
        state = tuple(state)

        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        return int(np.argmax(self.q_table[state]))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> None:
        state = tuple(state)
        next_state = tuple(next_state)

        current_q = self.q_table[state + (action,)]
        max_next_q = np.max(self.q_table[next_state])

        target = reward
        if not done:
            target += self.gamma * max_next_q

        self.q_table[state + (action,)] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.q_table, axis=-1)