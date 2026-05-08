from __future__ import annotations


def log_episode(episode: int, avg_reward: float, avg_coverage: float, avg_grid: float, epsilon: float) -> None:
    """Print a compact training-progress line."""
    print(
        f"Episode {episode:4d} | avg_reward={avg_reward:7.2f} | "
        f"coverage={avg_coverage:.2%} | grid={avg_grid:.2f} | epsilon={epsilon:.3f}"
    )
