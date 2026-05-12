"""Console logging helpers for training progress."""
from __future__ import annotations


def log_episode(episode: int, avg_reward: float, avg_coverage: float, avg_grid: float, epsilon: float) -> None:
    """Print a compact training-progress line.

    Args:
        episode: Current episode number.
        avg_reward: Average reward over the recent logging window.
        avg_coverage: Average demand coverage over the recent logging window.
        avg_grid: Average grid energy purchased over the recent logging window.
        epsilon: Current exploration probability.
    """
    print(
        f"Episode {episode:4d} | avg_reward={avg_reward:7.2f} | "
        f"coverage={avg_coverage:.2%} | grid={avg_grid:.2f} | epsilon={epsilon:.3f}"
    )
