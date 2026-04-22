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

### Interpreting discrete state values

The values `0`, `1`, and `2` should be interpreted as **ordered discrete categories**, not as exact physical units.

A simple interpretation is:

- `0` = low or none
- `1` = medium
- `2` = high

So, for example:
- `battery = 0` means the battery is empty or almost empty,
- `battery = 2` means the battery is at a high stored-energy level,
- `demand = 2` means the environment is currently in a high-demand situation,
- `renewable = 2` means renewable generation is currently high.

This representation keeps the environment small enough for tabular Q-learning while still allowing meaningful differences between operating conditions.

### Period meaning

The variable `period` also uses a discrete representation:

- `0` = first daily segment
- `1` = second daily segment
- `2` = third daily segment
- `3` = fourth daily segment

These values do **not** correspond to exact real-world hours.  
Instead, they represent four simplified phases of the day used to change the probability of different demand and renewable generation levels.

This means the agent does not just react to the current battery, demand, and renewable values. It can also learn that different periods tend to have different operating patterns.

### Demand and renewable sampling

The values of `demand` and `renewable` are sampled stochastically, and their distributions depend on the current `period`.

This is important for interpreting results:
- some periods are more likely to produce higher demand,
- some periods are more likely to produce higher renewable generation,
- and the agent must adapt to those changing conditions.

Because of this, the same action can be more or less useful depending on the current period.

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

### When actions make sense

The interpretation of the actions is easier if they are read together with the current energy balance:

- if `demand > renewable`, the system is in a **deficit** situation,
- if `renewable > demand`, the system is in a **surplus** situation,
- if `demand == renewable`, the current renewable generation already matches demand.

So:
- **use battery** and **buy from grid** are mainly relevant in deficit situations,
- **store surplus energy** and **sell surplus energy** are mainly relevant in surplus situations.

If the agent tries to apply an action that does not fit the current situation, the environment can mark it as an invalid action and penalise it.

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

### How to interpret reward values

The reward should not be read as a monetary value.  
Instead, it is a compact signal that combines several objectives:

- serving demand is good,
- leaving demand uncovered is bad,
- using the external grid too much is costly,
- selling surplus can be positive,
- and invalid actions reduce performance.

This means that a higher reward generally indicates a better policy, but reward should still be interpreted together with the evaluation metrics and scenario context.

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

### How to compare scenarios

A scenario should not be judged only by reward in isolation.

A stronger result usually combines:
- **higher `avg_reward`**,
- **higher `avg_coverage`**,
- **lower `avg_grid_bought`**,
- and a reasonable battery behavior depending on the scenario.

More difficult scenarios are expected to produce weaker raw numbers than the baseline. That does not necessarily mean the agent is worse trained; it may simply mean the environment is more demanding.

For example:
- `baseline` is the clean reference case,
- `winter` and `summer` change the operating profile,
- `demand_noise` makes demand less predictable,
- `battery_loss` reduces battery reliability,
- `combined_v2` is usually the hardest scenario because several sources of difficulty are active at the same time.

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

### How wrappers affect interpretation

The wrappers do not all change the environment in the same way:

- **SeasonWrapper** changes the underlying demand and renewable profile.
- **DemandNoiseWrapper** makes demand less predictable during the episode.
- **BatteryLossWrapper** makes stored energy less reliable.
- **RewardShapingWrapper** changes the incentive signal, not the base physical transition itself.

This matters when interpreting results:
- weaker performance in a wrapped scenario may reflect a genuinely harder environment,
- and differences between scenarios should be read as differences in operating conditions, not just as changes in agent quality.

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/jaimeeblancoo/smartgrid-rl-es.git
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

Evaluation expects trained models to exist in:

```text
results/models/
```

So, if no models are available yet, run one or more training commands first.

### Suggested workflow

A simple and recommended execution order for V2 is:

1. Train one or more scenarios:
   - `python -m src.training.train --scenario baseline`
   - `python -m src.training.train --scenario combined_v2`

2. Run the evaluation script:
   - `python -m src.training.evaluate`

3. Open the dashboard:
   - `streamlit run src/dashboard/app.py`

This makes it easier to generate the required files in the correct order before inspecting them in the dashboard.

### On Ubuntu / Linux

```bash
source .venv/bin/activate

python -m src.training.train --scenario baseline
python -m src.training.train --scenario combined_v2
python -m src.training.evaluate
streamlit run src/dashboard/app.py
```

### On Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1

python -m src.training.train --scenario baseline
python -m src.training.train --scenario combined_v2
python -m src.training.evaluate
streamlit run src/dashboard/app.py
```

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

### How to interpret the dashboard

The dashboard is useful because it combines:
- aggregated scenario-level results from evaluation,
- and step-by-step episode data from the demo CSV files.

This allows two complementary views:
- a **summary view**, to compare scenarios globally,
- and a **trajectory view**, to inspect how the agent behaves step by step inside one episode.

If the evaluation summary shows that a scenario is weaker than another, the demo CSV can help understand why:
- more grid use,
- less battery availability,
- more unmet demand,
- or less opportunity to exploit surplus energy.

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

### About `results/logs/`
The `results/logs/` folder is kept in the structure for optional local logs or auxiliary outputs, but the main V2 workflow is centered on:
- `results/models/`
- `results/plots/`
- `results/demos/`
- `results/summaries/`

### How to read evaluation metrics

The most important evaluation metrics are:

- **avg_reward**: average total reward per episode.
- **avg_coverage**: average proportion of demand covered during evaluation episodes.
- **avg_grid_bought**: average amount of energy bought from the grid per episode.
- **avg_battery_end**: average battery level at the end of evaluation episodes.
- **avg_sold**: average amount of surplus energy sold per episode.

A useful interpretation is:
- higher `avg_reward` is usually better,
- higher `avg_coverage` is better,
- lower `avg_grid_bought` is usually better,
- higher `avg_sold` can be positive if surplus is being exploited efficiently,
- and `avg_battery_end` should be read carefully depending on the scenario, because ending with a full battery is not always the main objective if demand was not well covered.

### How to interpret demo episode files

The demo episode CSV files are useful for understanding **how** the agent behaves, not just how well it scores.

They can help answer questions such as:
- when does the agent use the battery?
- when does it rely on the external grid?
- does it exploit renewable surplus well?
- does it keep making invalid actions?
- does it behave differently in more difficult scenarios?

This makes the demo CSVs especially helpful when the summary metrics alone do not explain the behaviour clearly.

---

## Main results and conclusions

The V2 version of SmartGrid-ES provides a more complete experimentation workflow than V1.

By introducing scenarios, wrappers, comparative evaluation, and a dashboard, the project can now be used not only to train an agent, but also to compare behavior across different operating conditions.

The generated outputs make it possible to inspect:
- how reward changes across scenarios,
- how much demand is covered,
- how often the grid is used,
- and how battery-related effects influence performance.

In particular, the `combined_v2` scenario is useful because it brings together several sources of difficulty in a single setup, making it a better approximation of a more realistic decision environment than the baseline case.

Even though the environment is still intentionally simple and discrete, the project now has a solid V2 structure that is easy to explain, test, and extend. This makes it a good foundation for a future V3 with richer dynamics and possibly more realistic data or environment design.

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
