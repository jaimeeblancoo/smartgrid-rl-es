"""Optional V3.1 PSO reward-weight tuning experiment.

This module treats Particle Swarm Optimization as a reward-weight search
procedure around the main V3 tabular Q-learning pipeline. PSO is not the main
learning algorithm; each candidate vector trains and evaluates a temporary
Q-learning agent before the best reward weights are written as experiment
artifacts.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyswarm import pso

from src.agents.qlearning_agent import QLearningAgent
from src.envs.smartgrid_env_v3 import SmartGridEnvV3
from src.training.config import get_v3_scenario_path, get_v3_training_config_for_scenario


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

VECTOR_ORDER = [
    "demand_covered",
    "unmet_demand",
    "grid_bought",
    "sold",
    "invalid_action",
    "wasted_renewable",
    "risk",
]

LOWER_BOUND = np.array([1.0, -8.0, -2.0, 0.1, -4.0, -2.0, -0.08], dtype=float)
UPPER_BOUND = np.array([4.0, -2.0, -0.2, 2.0, -0.5, -0.1, -0.005], dtype=float)


def current_run_date() -> str:
    """Return the date tag used in V3.1 output filenames."""
    return datetime.now().strftime("%Y-%m-%d")


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated CLI argument into integers.

    Args:
        value: Comma-separated string from the command line.

    Returns:
        Parsed integer list.
    """
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated CLI argument into floats.

    Args:
        value: Comma-separated string from the command line.

    Returns:
        Parsed float list.
    """
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def vector_to_reward_weights(vector: np.ndarray) -> dict[str, float]:
    """Convert a PSO particle vector into V3 reward weights.

    Args:
        vector: Particle vector ordered according to ``VECTOR_ORDER``.

    Returns:
        Reward-weight dictionary accepted by ``SmartGridEnvV3``.
    """
    return {name: float(value) for name, value in zip(VECTOR_ORDER, vector)}


def build_agent(env: SmartGridEnvV3, cfg: dict[str, Any], seed: int) -> QLearningAgent:
    """Create a Q-learning agent compatible with a V3 environment.

    Args:
        env: V3 environment used to derive observation and action dimensions.
        cfg: Training hyperparameter configuration.
        seed: Random seed for the agent.

    Returns:
        Configured tabular Q-learning agent.
    """
    return QLearningAgent(
        state_shape=env.observation_space.nvec,
        action_size=env.action_space.n,
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        seed=seed,
    )


def train_temporary_agent(
    scenario: str,
    reward_weights: dict[str, float],
    episodes: int,
    seed: int,
) -> QLearningAgent:
    """Train a temporary Q-learning agent for one PSO candidate.

    The agent is not saved to disk because PSO evaluates many candidate reward
    vectors and only the optimization results are persisted.
    """
    cfg = get_v3_training_config_for_scenario(scenario)
    cfg["episodes"] = episodes
    scenario_path = get_v3_scenario_path(scenario)
    env = SmartGridEnvV3(
        scenario_path=scenario_path,
        mode="train",
        seed=seed,
        reward_weights_override=reward_weights,
    )
    agent = build_agent(env=env, cfg=cfg, seed=seed)

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()

    return agent


def evaluate_agent(
    scenario: str,
    agent: QLearningAgent,
    reward_weights: dict[str, float],
    seed: int,
) -> dict[str, float]:
    """Evaluate a temporary PSO-trained agent greedily on the eval CSV.

    Args:
        scenario: V3 scenario name.
        agent: Temporary tabular Q-learning agent.
        reward_weights: Candidate reward weights used by the environment.
        seed: Reset seed for evaluation.

    Returns:
        Dictionary of aggregate evaluation metrics for one deterministic V3
        evaluation episode.
    """
    scenario_path = get_v3_scenario_path(scenario)
    env = SmartGridEnvV3(
        scenario_path=scenario_path,
        mode="eval",
        seed=seed,
        reward_weights_override=reward_weights,
    )
    state, _ = env.reset(seed=seed)
    done = False
    reward_total = 0.0
    demand_total = 0.0
    covered_total = 0.0
    grid_total = 0.0
    sold_total = 0.0
    unmet_total = 0.0
    wasted_total = 0.0
    risk_total = 0.0
    invalid_total = 0
    steps = 0
    last_battery = 0

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    while not done:
        action = agent.greedy_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward_total += reward
        demand_total += info["demand"]
        covered_total += info["demand_covered"]
        grid_total += info["grid_bought"]
        sold_total += info["sold"]
        unmet_total += info["unmet_demand"]
        wasted_total += info["wasted_renewable"]
        risk_total += info["risk_score"]
        invalid_total += info["invalid_action"]
        last_battery = info["battery"]
        steps += 1
    agent.epsilon = original_epsilon

    return {
        "avg_reward": float(reward_total),
        "avg_coverage": float(covered_total / demand_total if demand_total else 0.0),
        "avg_unmet_demand": float(unmet_total),
        "avg_grid_bought": float(grid_total),
        "avg_sold": float(sold_total),
        "avg_wasted_renewable": float(wasted_total),
        "avg_battery_end": float(last_battery),
        "avg_risk_score": float(risk_total / steps if steps else 0.0),
        "invalid_action_rate": float(invalid_total / steps if steps else 0.0),
    }


def compute_fitness(metrics: dict[str, float]) -> float:
    """Compute the PSO fitness score from V3 evaluation metrics.

    The score rewards high episode reward and coverage while penalizing unmet
    demand, energy risk and invalid actions.
    """
    return (
        metrics["avg_reward"]
        + 50 * metrics["avg_coverage"]
        - 2 * metrics["avg_unmet_demand"]
        - 0.5 * metrics["avg_risk_score"]
        - 10 * metrics["invalid_action_rate"]
    )


def evaluate_candidate(
    vector: np.ndarray,
    scenario: str,
    episodes: int,
    seed: int,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Train and evaluate one PSO candidate reward-weight vector.

    Args:
        vector: Candidate particle vector.
        scenario: V3 scenario name.
        episodes: Number of temporary Q-learning episodes to run.
        seed: Seed used for temporary training and evaluation.

    Returns:
        Tuple with fitness, reward-weight dictionary and evaluation metrics.
    """
    reward_weights = vector_to_reward_weights(vector)
    agent = train_temporary_agent(
        scenario=scenario,
        reward_weights=reward_weights,
        episodes=episodes,
        seed=seed,
    )
    metrics = evaluate_agent(
        scenario=scenario,
        agent=agent,
        reward_weights=reward_weights,
        seed=seed + 10000,
    )
    fitness = compute_fitness(metrics)
    return fitness, reward_weights, metrics


def make_objective(scenario: str, episodes: int, seed: int):
    """Create the minimization objective expected by ``pyswarm.pso``.

    Args:
        scenario: V3 scenario name.
        episodes: Number of temporary Q-learning episodes per candidate.
        seed: Base seed used for reproducibility.

    Returns:
        Objective function that maps a PSO vector to negative fitness.
    """
    def objective(vector: np.ndarray) -> float:
        """Return negative fitness because ``pyswarm`` minimizes."""
        fitness, _, _ = evaluate_candidate(
            vector=np.asarray(vector, dtype=float),
            scenario=scenario,
            episodes=episodes,
            seed=seed,
        )
        return -fitness

    return objective


def create_result_record(
    iteration: int,
    scenario: str,
    swarmsize: int,
    omega: float,
    phip: float,
    phig: float,
    maxiter: int,
    episodes: int,
    x_optimum: np.ndarray,
    objective_minimum: float,
    fitness_best: float,
    reward_weights: dict[str, float],
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Create one tabular result row for a PSO hyperparameter combination.

    Args:
        iteration: Grid-search iteration number.
        scenario: V3 scenario name.
        swarmsize: Number of particles used by PSO.
        omega: PSO inertia coefficient.
        phip: Particle best-position coefficient.
        phig: Global best-position coefficient.
        maxiter: Number of PSO iterations.
        episodes: Number of temporary Q-learning episodes per candidate.
        x_optimum: Best particle vector found by PSO.
        objective_minimum: Minimum objective value returned by ``pyswarm``.
        fitness_best: Fitness value corresponding to ``x_optimum``.
        reward_weights: Reward-weight dictionary for ``x_optimum``.
        metrics: Evaluation metrics for the best candidate.

    Returns:
        Flat dictionary suitable for conversion to a results DataFrame.
    """
    record: dict[str, Any] = {
        "iteration": iteration,
        "scenario": scenario,
        "swarmsize": swarmsize,
        "omega": omega,
        "phip": phip,
        "phig": phig,
        "maxiter": maxiter,
        "episodes": episodes,
        "objective_minimum": float(objective_minimum),
        "fitness_best": float(fitness_best),
        "x_optimum": json.dumps([float(value) for value in x_optimum]),
    }
    record.update(reward_weights)
    record.update(metrics)
    return record


def save_outputs(
    results_df: pd.DataFrame,
    scenario: str,
    output_prefix: str | None,
) -> tuple[Path, Path, Path]:
    """Save the V3.1 PSO CSV, best-config JSON and fitness plot.

    Args:
        results_df: DataFrame with one row per PSO hyperparameter combination.
        scenario: V3 scenario name used in output filenames.
        output_prefix: Optional filename prefix from the CLI.

    Returns:
        Paths to the summary CSV, best-config JSON and fitness plot.
    """
    date_tag = current_run_date()
    prefix = f"{output_prefix}_" if output_prefix else ""
    summary_path = (
        Path("results/v3_1/summaries")
        / f"{prefix}pso_reward_tuning_{scenario}_{date_tag}.csv"
    )
    best_config_path = (
        Path("results/v3_1/best_configs")
        / f"{prefix}best_reward_weights_{scenario}_{date_tag}.json"
    )
    plot_path = (
        Path("results/v3_1/plots")
        / f"{prefix}pso_reward_tuning_{scenario}_{date_tag}.png"
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_df = results_df.sort_values("fitness_best", ascending=False).reset_index(drop=True)
    sorted_df.to_csv(summary_path, index=False, encoding="utf-8")

    best_row = sorted_df.iloc[0].to_dict()
    best_config = {
        "scenario": scenario,
        "best_hyperparameters": {
            "swarmsize": int(best_row["swarmsize"]),
            "omega": float(best_row["omega"]),
            "phip": float(best_row["phip"]),
            "phig": float(best_row["phig"]),
            "maxiter": int(best_row["maxiter"]),
            "episodes": int(best_row["episodes"]),
        },
        "best_fitness": float(best_row["fitness_best"]),
        "objective_minimum": float(best_row["objective_minimum"]),
        "reward_weights": {
            name: float(best_row[name])
            for name in VECTOR_ORDER
        },
        "metrics": {
            "avg_reward": float(best_row["avg_reward"]),
            "avg_coverage": float(best_row["avg_coverage"]),
            "avg_unmet_demand": float(best_row["avg_unmet_demand"]),
            "avg_grid_bought": float(best_row["avg_grid_bought"]),
            "avg_sold": float(best_row["avg_sold"]),
            "avg_wasted_renewable": float(best_row["avg_wasted_renewable"]),
            "avg_battery_end": float(best_row["avg_battery_end"]),
            "avg_risk_score": float(best_row["avg_risk_score"]),
            "invalid_action_rate": float(best_row["invalid_action_rate"]),
        },
        "vector_order": VECTOR_ORDER,
        "note": "pyswarm minimizes the objective, so the script minimizes negative fitness.",
    }
    best_config_path.write_text(
        json.dumps(best_config, indent=2),
        encoding="utf-8",
    )

    plot_df = results_df.sort_values("iteration").reset_index(drop=True)
    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["iteration"], plot_df["fitness_best"], marker="o")
    plt.xlabel("PSO hyperparameter combination")
    plt.ylabel("best fitness")
    plt.title("V3.1 PSO reward tuning")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return summary_path, best_config_path, plot_path


def run_pso_grid_search(args: argparse.Namespace) -> tuple[pd.DataFrame, Path, Path, Path]:
    """Run PSO across the configured hyperparameter grid.

    Args:
        args: Parsed command-line arguments defining the PSO grid.

    Returns:
        Results DataFrame and paths to the saved output artifacts.
    """
    swarmsizes = parse_int_list(args.swarmsizes)
    omegas = parse_float_list(args.omegas)
    phips = parse_float_list(args.phips)
    phigs = parse_float_list(args.phigs)
    combinations = list(itertools.product(swarmsizes, omegas, phips, phigs))
    records: List[Dict[str, Any]] = []

    for iteration, (swarmsize, omega, phip, phig) in enumerate(combinations, start=1):
        LOGGER.info(
            "Running PSO %s/%s | scenario=%s swarmsize=%s omega=%s phip=%s phig=%s",
            iteration,
            len(combinations),
            args.scenario,
            swarmsize,
            omega,
            phip,
            phig,
        )
        np.random.seed(args.seed + iteration)
        objective = make_objective(
            scenario=args.scenario,
            episodes=args.episodes,
            seed=args.seed + iteration * 1000,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            x_optimum, objective_minimum = pso(
                objective,
                LOWER_BOUND,
                UPPER_BOUND,
                swarmsize=swarmsize,
                omega=omega,
                phip=phip,
                phig=phig,
                maxiter=args.maxiter,
                debug=False,
            )

        fitness_best, reward_weights, metrics = evaluate_candidate(
            vector=np.asarray(x_optimum, dtype=float),
            scenario=args.scenario,
            episodes=args.episodes,
            seed=args.seed + iteration * 1000,
        )
        records.append(
            create_result_record(
                iteration=iteration,
                scenario=args.scenario,
                swarmsize=swarmsize,
                omega=omega,
                phip=phip,
                phig=phig,
                maxiter=args.maxiter,
                episodes=args.episodes,
                x_optimum=np.asarray(x_optimum, dtype=float),
                objective_minimum=float(objective_minimum),
                fitness_best=fitness_best,
                reward_weights=reward_weights,
                metrics=metrics,
            )
        )

    results_df = pd.DataFrame(records)
    summary_path, best_config_path, plot_path = save_outputs(
        results_df=results_df,
        scenario=args.scenario,
        output_prefix=args.output_prefix,
    )
    return results_df, summary_path, best_config_path, plot_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the V3.1 PSO experiment."""
    parser = argparse.ArgumentParser(
        description="Optional V3.1 PSO reward-weight tuning experiment."
    )
    parser.add_argument("--scenario", default="baseline_v3")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--swarmsizes", default="5")
    parser.add_argument("--omegas", default="0.5,0.7")
    parser.add_argument("--phips", default="1.0,1.5")
    parser.add_argument("--phigs", default="1.0,1.5")
    return parser


def main() -> None:
    """Run the V3.1 PSO reward-weight tuning CLI."""
    args = build_parser().parse_args()
    results_df, summary_path, best_config_path, plot_path = run_pso_grid_search(args)
    best = results_df.sort_values("fitness_best", ascending=False).iloc[0]
    print(results_df.sort_values("fitness_best", ascending=False).to_string(index=False))
    print()
    print(f"Best fitness:       {best['fitness_best']:.4f}")
    print(f"PSO results CSV:    {summary_path}")
    print(f"Best config JSON:   {best_config_path}")
    print(f"Fitness plot saved: {plot_path}")


if __name__ == "__main__":
    main()
