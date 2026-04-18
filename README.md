# SmartGrid-ES

Academic reinforcement learning project using Gymnasium to model a simplified electrical microgrid inspired by a smart grid.

## V1 goal

This first version aims to validate the base architecture of the project with a small, functional, and extensible implementation. At this stage, we are not trying to build a fully realistic smart grid yet, but rather a discrete demo that already includes:

- a Gymnasium environment
- a tabular Q-learning agent
- an episode-based training script
- basic saved results
- a clean repository structure

## What this version does

The environment represents a simplified microgrid where, at each step, the agent observes:

- battery level
- demand level
- renewable generation level
- time period of the day

Based on that state, the agent chooses between four actions:

- use battery
- buy energy from the grid
- store surplus
- sell surplus

The reward encourages covering demand and penalizes leaving demand unmet or relying too much on the grid.

## Technologies

- Python
- Gymnasium
- NumPy
- Matplotlib
- Pandas

## Repository structure

```text
smartgrid-rl-es/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── smartgrid_env.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── qlearning_agent.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── train.py
│   └── utils/
├── results/
│   ├── plots/
│   │   └── .gitkeep
│   └── logs/
│       └── .gitkeep
```

## Installation

```bash
git clone -b v1-main https://github.com/jaimeeblancoo/smartgrid-rl-es.git
cd smartgrid-rl-es
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the project

From the project root:

```bash
python -m src.training.train
```

You can also run it with:

```bash
python src/training/train.py
```

## Saved outputs

When training finishes, the program stores the generated results locally on disk.

Saved files:

- training reward plot:
  - `results/plots/training_rewards.png`
- demo episode log with step-by-step agent decisions:
  - `results/logs/demo_episode.csv`

The script creates the output folders automatically if they do not exist.

These generated files are local execution outputs and are not intended to be tracked in Git.

## Console output

During training, the script prints:

- average reward every 100 episodes
- current epsilon value
- final average reward
- saved output paths
