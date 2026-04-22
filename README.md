# SmartGrid-ES

SmartGrid-ES is an academic reinforcement learning project that uses Gymnasium to model a simplified electrical microgrid inspired by a smart grid.

This branch contains the **V1** version of the project.

V1 is intended as a small and functional first step. The objective is not to build a fully realistic smart grid yet, but to validate the basic architecture of the project with a clean and extensible reinforcement learning pipeline.

In this first version, the project already includes:
- a custom Gymnasium environment,
- a tabular Q-learning agent,
- an episode-based training script,
- basic output generation,
- and a simple repository structure.

---

## Project purpose

The project studies how a reinforcement learning agent can learn simple energy-management decisions inside a small discrete microgrid environment.

At each step, the agent must decide how to manage available energy in order to:
- cover demand,
- reduce unmet demand,
- avoid unnecessary dependence on the external grid,
- and make reasonable use of stored or surplus energy.

V1 focuses on the **base environment and training loop**. It is the foundation on top of which later versions can introduce richer scenarios, wrappers, comparative evaluation, and more advanced visualisation.

---

## What V1 does

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

The reward function encourages the agent to cover demand while penalising unmet demand, invalid actions, and excessive dependence on the grid.

---

## State variables

The environment state is represented as:

```text
(battery, demand, renewable, period)
```

### Meaning of each variable

| Variable | Meaning |
|---|---|
| `battery` | current battery energy level |
| `demand` | current electricity demand level |
| `renewable` | current renewable generation level |
| `period` | simplified time period of the day |

These are **discrete variables**, not continuous real-world measurements.  
This makes the environment small enough to be solved with tabular Q-learning.

---

## Action space

The action space contains 4 discrete actions:

| Action ID | Meaning |
|---|---|
| `0` | use battery |
| `1` | buy from the grid |
| `2` | store surplus energy |
| `3` | sell surplus energy |

### Action intuition

- **use battery**: use stored energy to help cover a demand deficit
- **buy from the grid**: cover demand by importing energy externally
- **store surplus**: save extra renewable energy in the battery
- **sell surplus**: export renewable surplus instead of storing it

---

## Technologies

The project currently uses:
- Python
- Gymnasium
- NumPy
- Matplotlib
- Pandas

---

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

---

## Installation

Clone the repository and install the required packages:

```bash
git clone -b v1-main https://github.com/jaimeeblancoo/smartgrid-rl-es.git
cd smartgrid-rl-es
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### On Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Running the project

From the project root:

```bash
python -m src.training.train
```

You can also run it with:

```bash
python src/training/train.py
```

### Suggested workflow

A simple execution order for V1 is:

1. install dependencies
2. run the training script
3. inspect the generated plot and CSV outputs

### On Ubuntu / Linux

```bash
source .venv/bin/activate
python -m src.training.train
```

### On Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.training.train
```

---

## Saved outputs

When training finishes, the program stores the generated results locally on disk.

### Training reward plot
Stored in:

```text
results/plots/training_rewards.png
```

### Demo episode log
Stored in:

```text
results/logs/demo_episode.csv
```

This CSV stores step-by-step agent decisions during a demo episode.

The script creates the output folders automatically if they do not exist.

These generated files are local execution outputs and are not intended to be tracked in Git.

---

## Console output

During training, the script prints:
- average reward every 100 episodes
- current epsilon value
- final average reward
- saved output paths

---

## Why V1 matters

V1 is important because it validates the base logic of the project:
- environment definition,
- state and action design,
- reward structure,
- Q-learning training loop,
- and local result generation.

Even though this version is intentionally simple, it provides a clean base for later versions with more complex behaviour.
