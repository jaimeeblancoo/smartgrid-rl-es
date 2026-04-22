# SmartGrid-ES

SmartGrid-ES is an academic reinforcement learning project that models a simplified electrical microgrid inspired by a smart grid.

This repository contains the **V2** version of the project.  
V2 keeps the small and discrete base from V1, but extends it with:
- scenario-based environment construction,
- wrapper-based behavior changes,
- comparative evaluation across scenarios,
- generated CSV outputs,
- and a lightweight dashboard to inspect results.

The goal is not to simulate a fully realistic power grid, but to build a **small, functional, and extensible RL system** that is easy to understand, test, and extend in future versions.

---

## Project purpose

The project studies how a tabular Q-learning agent can learn decision rules inside a simplified microgrid.

At each step, the agent must decide how to manage available energy:
- use stored battery energy,
- buy electricity from the external grid,
- store renewable surplus,
- or sell renewable surplus.

The agent is trained to:
- cover demand,
- reduce unmet demand,
- avoid excessive grid dependence,
- and behave reasonably under different operating scenarios.

---

## V2 objective

V1 established the basic architecture:
- a discrete Gymnasium environment,
- a tabular Q-learning agent,
- and a simple training flow.

V2 builds on top of that foundation by adding:
- **scenarios** through an environment factory,
- **wrappers** that modify behavior without rewriting the base environment,
- **evaluation across multiple scenarios**,
- **CSV and plot outputs** for analysis,
- and a **Streamlit dashboard** to inspect generated results.

This makes the project more structured and closer to a real experimentation workflow, while still staying simple and appropriate for an academic team project.

---

## Environment design

The environment represents a simplified microgrid with:
- a battery,
- a current demand level,
- a renewable generation level,
- and a time period of the day.

The environment is discrete, which makes it suitable for tabular Q-learning.

### State variables

The state is:

```text
(battery, demand, renewable, period)
```

Each component is discrete:

| Variable | Meaning | Range |
|---|---|---|
| `battery` | energy currently available in storage | `0..2` |
| `demand` | current electricity demand level | `0..2` |
| `renewable` | current renewable generation level | `0..2` |
| `period` | time period of the day | `0..3` |

### Meaning of each variable

- **battery**: how much stored energy is available in the battery.
- **demand**: how much energy is currently needed.
- **renewable**: how much renewable energy is currently being generated.
- **period**: a simplified time-of-day indicator used to vary demand and renewable patterns.

These are **discrete levels**, not continuous real-world measurements.

---

## Action space

The action space has 4 discrete actions:

| Action ID | Meaning |
|---|---|
| `0` | use battery |
| `1` | buy from grid |
| `2` | store surplus energy |
| `3` | sell surplus energy |

### Action intuition

- **use battery**: use stored energy to help cover a demand deficit.
- **buy from grid**: buy energy externally when renewable generation is not enough.
- **store surplus energy**: save extra renewable energy in the battery when possible.
- **sell surplus energy**: sell renewable surplus instead of storing it.

---

## Reward intuition

The reward is designed to guide the agent toward reasonable energy-management behavior.

In general, the reward:
- **encourages covering demand**,
- **strongly penalizes unmet demand**,
- **penalizes unnecessary dependence on the external grid**,
- **can reward selling surplus energy**,
- and **penalizes invalid actions**.

Some V2 scenarios also apply additional reward shaping through wrappers.

---

## Available scenarios in V2

V2 supports the following scenarios through `src/envs/factory.py`:

- `baseline`
- `winter`
- `summer`
- `demand_noise`
- `battery_loss`
- `combined_v2`

### Scenario meaning

- **baseline**: the base environment with no extra wrapper modifications.
- **winter**: seasonal conditions that reflect a winter profile.
- **summer**: seasonal conditions that reflect a summer profile.
- **demand_noise**: introduces occasional stochastic demand spikes.
- **battery_loss**: simulates simple battery inefficiency and losses.
- **combined_v2**: combines seasonal effects, demand noise, battery loss, and reward shaping.

---

## Wrappers used in V2

V2 extends the environment through wrappers instead of rewriting the base environment.

### `SeasonWrapper`
Applies a seasonal profile to demand and renewable generation.

### `DemandNoiseWrapper`
Introduces occasional demand spikes during the episode.

### `BatteryLossWrapper`
Simulates battery inefficiency while keeping the state discrete.

### `RewardShapingWrapper`
Adjusts the reward signal without modifying the base environment logic.

This wrapper-based design is one of the main differences between V1 and V2.

---

## Training and evaluation

### Training

Training is handled through:

```bash
python -m src.training.train --scenario baseline
```

You can replace `baseline` with any of the supported scenarios.

Training:
- builds the environment through the scenario factory,
- trains a tabular Q-learning agent,
- saves a model,
- saves a training plot,
- and exports a demo episode as CSV.

### Evaluation

Evaluation is handled through:

```bash
python -m src.training.evaluate
```

Evaluation:
- loads trained models from the different scenarios,
- computes summary metrics,
- exports a summary CSV,
- and generates a comparison plot.

---

## Dashboard

The project includes a lightweight Streamlit dashboard:

```bash
streamlit run src/dashboard/app.py
```

The dashboard is designed to:
- load the latest evaluation summary,
- inspect generated demo episode CSV files,
- show tables and simple charts,
- and provide a readable visual summary of the V2 outputs.

The dashboard reads data from:
- `results/summaries/`
- `results/demos/`

---

## Repository structure

```text
smartgrid-rl-es/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── README_data.md
├── src/
│   ├── agents/
│   │   └── qlearning_agent.py
│   ├── dashboard/
│   │   └── app.py
│   ├── envs/
│   │   ├── factory.py
│   │   └── smartgrid_env.py
│   ├── training/
│   │   ├── config.py
│   │   ├── evaluate.py
│   │   └── train.py
│   ├── utils/
│   │   ├── io_helpers.py
│   │   ├── logger.py
│   │   └── plotting.py
│   └── wrappers/
│       ├── season_wrapper.py
│       ├── demand_noise_wrapper.py
│       ├── battery_loss_wrapper.py
│       └── reward_shaping_wrapper.py
└── results/
    ├── demos/
    ├── logs/
    ├── models/
    ├── plots/
    └── summaries/
```

---

## Installation

Clone the repository and install the required packages:

```bash
git clone -b v2-dev https://github.com/jaimeeblancoo/smartgrid-rl-es.git
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

## Dependencies

The project currently uses:
- `gymnasium`
- `numpy`
- `matplotlib`
- `pandas`
- `streamlit`

---

## Outputs generated by V2

After running training and evaluation, the repository can generate the following types of outputs:

### Models
Stored in:

```text
results/models/
```

Example:
- `q_table_baseline.npy`

### Training plots
Stored in:

```text
results/plots/
```

Examples:
- `training_rewards_baseline.png`
- `evaluation_comparison_YYYY-MM-DD.png`

### Demo episode CSV files
Stored in:

```text
results/demos/
```

Examples:
- `smartgrid_demo_episode_baseline_YYYY-MM-DD.csv`
- `smartgrid_demo_episode_combined_v2_YYYY-MM-DD.csv`

These files store step-by-step episode data such as:
- battery level,
- demand,
- renewable generation,
- selected action,
- reward,
- grid usage,
- covered demand,
- and unmet demand.

### Evaluation summary CSV files
Stored in:

```text
results/summaries/
```

Example:
- `smartgrid_evaluation_summary_YYYY-MM-DD.csv`

These files summarize metrics by scenario, such as:
- average reward,
- average coverage,
- average grid energy bought,
- average final battery level,
- and average sold energy.

---

## Notes about data

V2 does **not** use:
- JSON scenario files,
- real external data inputs,
- or external live datasets.

All behavior in V2 is defined directly in Python code through:
- the base environment,
- scenario construction,
- and wrappers.

This keeps the project small, controlled, and appropriate for a second academic iteration.

---

## Future direction

The current V2 is meant to remain:
- simple,
- functional,
- readable,
- and easy to extend.

A future V3 could move toward:
- richer custom environments,
- more realistic system dynamics,
- external data integration,
- or more advanced agent designs.

---

## Summary

SmartGrid-ES V2 is a small but structured reinforcement learning project that extends the original V1 base with:
- scenario-based environment construction,
- wrapper-based modifications,
- comparative evaluation,
- CSV and plot outputs,
- and a lightweight dashboard.

Its main purpose is to provide a clean academic RL pipeline that is simple enough to understand and defend, but rich enough to show meaningful progression beyond the first version.
