"""Plotting helpers for SmartGrid-ES result artifacts.

The functions in this module save lightweight Matplotlib figures used by the
training, evaluation and comparison scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(values: Sequence[float], window: int = 50) -> list[float]:
    """Compute a simple trailing moving average.

    Args:
        values: Numeric sequence to smooth.
        window: Maximum number of recent values included in each average.

    Returns:
        List of averaged values with the same length as ``values``.
    """
    if not values:
        return []
    output = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output.append(sum(values[start : index + 1]) / (index - start + 1))
    return output


def plot_rewards(rewards: Sequence[float], output_path: str) -> None:
    """Save a training reward curve with a moving-average line.

    Args:
        rewards: Per-episode reward history.
        output_path: Destination image path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="reward")
    plt.plot(moving_average(rewards, window=50), label="moving average (50)")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_comparison(summary_df: pd.DataFrame, output_path: str) -> None:
    """Save a bar plot comparing scenario average rewards.

    Args:
        summary_df: Evaluation summary containing ``scenario`` and
            ``avg_reward`` columns.
        output_path: Destination image path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["scenario"], summary_df["avg_reward"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("avg_reward")
    plt.title("Scenario comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
