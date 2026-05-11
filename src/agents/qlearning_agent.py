from __future__ import annotations

from pathlib import Path

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent used by the SmartGrid environments.

    The agent stores one Q-value per discrete state-action pair. It supports
    epsilon-greedy exploration during training and greedy action selection
    during evaluation.
    """

    def __init__(
        self,
        state_shape,
        action_size: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        seed: int | None = None,
    ):
        """Initialize a Q-table and the learning hyperparameters.

        Args:
            state_shape: Shape of the discrete observation space.
            action_size: Number of available discrete actions.
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Initial exploration probability.
            epsilon_min: Minimum exploration probability.
            epsilon_decay: Multiplicative decay applied after each episode.
            seed: Optional random seed for reproducible exploration.
        """
        self.state_shape = tuple(int(x) for x in state_shape)
        self.n_actions = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros(self.state_shape + (action_size,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def choose_action(self, state) -> int:
        """Return an epsilon-greedy action for the current state."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self.greedy_action(state)

    def greedy_action(self, state) -> int:
        """Return the action with the largest learned Q-value."""
        state_t = tuple(int(x) for x in state)
        return int(np.argmax(self.q_table[state_t]))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> None:
        """Apply the standard tabular Q-learning update rule."""
        s = tuple(int(x) for x in state)
        ns = tuple(int(x) for x in next_state)
        current_q = self.q_table[s + (action,)]
        next_q = 0.0 if done else float(np.max(self.q_table[ns]))
        target = reward + self.gamma * next_q
        self.q_table[s + (action,)] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        """Decay exploration while respecting the configured minimum value."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """Return the greedy action index for every discrete state."""
        return np.argmax(self.q_table, axis=-1)

    def save(self, path: str | Path) -> None:
        """Save the Q-table as a NumPy binary file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: str | Path) -> None:
        """Load a previously saved Q-table from disk."""
        q_table = np.load(Path(path))
        expected_shape = self.state_shape + (self.n_actions,)
        if q_table.shape != expected_shape:
            raise ValueError(
                f"Loaded Q-table shape {q_table.shape} does not match "
                f"expected shape {expected_shape}."
            )
        self.q_table = q_table
