# SmartGrid-ES

## Overview

SmartGrid-ES is an academic reinforcement learning project that models a simplified smart-grid-inspired microgrid using Gymnasium.

The final delivery branch, `v3-main`, contains the third version of the project: a custom Gymnasium environment called `SmartGridEnvV3`, where a tabular Q-learning agent learns energy-management decisions under synthetic demand, renewable-generation, electricity-price and weather scenarios.

V3 is the final main version of the project. V1 and V2 are preserved as historical milestones because they document the evolution from a predesigned environment to wrappers and finally to a dedicated custom environment. The project is designed for interpretability, reproducibility and alignment with the Artificial Intelligence course topics.

## Academic report

The final academic report for the V3 delivery is included as a PDF artifact:

```text
results/smartgrid_es_v3_report.pdf
```

The report provides the full academic explanation of the V3 design, including the custom Gymnasium environment, observation and action variables, tabular Q-learning setup, reward function, fuzzy risk module, scenario evaluation, final results, dashboard and optional V3.1 PSO reward-weight tuning experiment.

## Version history

| Branch | Purpose |
|---|---|
| `v1-main` | Initial version using a predesigned Gymnasium environment |
| `v2-main` | Wrapper-based version with scenario modifications |
| `v3-main` | Final version with a custom Gymnasium environment |

V1 demonstrates the basic agent-environment cycle, V2 introduces wrappers, and V3 introduces a custom environment, external scenarios, extended evaluation, fuzzy risk and optional PSO reward-weight tuning.

## V3 contribution

V3 replaces the V2 wrapper-based approach with a dedicated custom Gymnasium environment: `SmartGridEnvV3`. This makes the environment dynamics, scenario inputs, reward calculation and evaluation outputs explicit within the V3 pipeline.

| Area | V2 | V3 |
|---|---|---|
| Environment design | Base environment modified through wrappers | Custom Gymnasium environment |
| Scenario definition | Python-side configuration | External JSON scenario files |
| Input data | Internal/procedural dynamics | Synthetic CSV time series |
| Evaluation | Basic scenario comparison | Extended metrics, demo episodes, dashboard and risk analysis |

## Course alignment

| Course topic | Project implementation |
|---|---|
| Reinforcement Learning | Tabular Q-learning agent trained through agent-environment interaction |
| Gymnasium | Custom `SmartGridEnvV3` environment following the Gymnasium API |
| Markov Decision Processes | Discrete state representation, action space, transition dynamics and reward function |
| Fuzzy Logic | Energy-risk score based on fuzzy rules and membership functions |
| Evolutionary Algorithms | Optional PSO-based reward-weight tuning experiment |
| Experimental Evaluation | Scenario comparison through CSV summaries, plots and dashboard visualizations |

## V3 environment design

`SmartGridEnvV3` is a custom Gymnasium environment for a simplified microgrid with battery storage, electricity demand, renewable generation, electricity price, time period and weather conditions. Its observable state is:

```text
[battery, demand, renewable, price, period, weather]
```

The state is discrete to keep tabular Q-learning interpretable and to make the learned Q-table directly inspectable.

| State variable | Meaning | Rationale |
|---|---|---|
| `battery` | Stored energy level | Represents short-term storage capacity |
| `demand` | Electricity demand level | Captures consumption pressure |
| `renewable` | Renewable generation level | Represents available low-cost clean energy |
| `price` | Electricity price level | Penalizes expensive grid purchases |
| `period` | Time-of-day period | Captures daily demand/generation patterns |
| `weather` | Simplified weather condition | Represents contextual conditions in the scenario data |

The V3 action space contains five discrete actions:

| Action ID | Action | Interpretation |
|---|---|---|
| `0` | Use battery | Cover unmet demand with stored energy |
| `1` | Buy from grid | Purchase external energy when local supply is insufficient |
| `2` | Store surplus | Save renewable surplus in the battery |
| `3` | Sell surplus | Export unused renewable energy |
| `4` | Hold | Take no active management action |

## Reward design

The V3 reward is interpretable and designed to guide reasonable energy-management behaviour in a simplified academic setting. It is not intended to simulate a real electricity market or a real market-settlement mechanism.

| Component | Effect on reward | Reason |
|---|---|---|
| Covered demand | Positive | The grid should satisfy electricity demand |
| Unmet demand | Strong negative | Failing to serve demand is the main operational failure |
| Grid energy bought | Negative | Buying from the grid has a cost, especially when prices are high |
| Sold surplus | Positive | Selling unused renewable surplus is beneficial |
| Invalid action | Negative | The agent should avoid actions that are not useful in the current state |
| Wasted renewable energy | Negative | Renewable surplus should preferably be stored or sold |
| Fuzzy risk | Negative | High-risk grid states should be discouraged |

## Fuzzy energy-risk score

The fuzzy risk module estimates operational risk from four V3 variables:

- battery level
- demand level
- renewable generation level
- price level

It applies membership functions and fuzzy rules to produce an interpretable risk signal.

| Output | Meaning |
|---|---|
| `risk_score` | Continuous risk value from 0 to 100 |
| `risk_level` | Discrete risk category: 0, 1 or 2 |
| `risk_level_name` | Human-readable label: low, medium or high |

The risk signal is used as:

1. an evaluation metric,
2. a reward penalty,
3. a dashboard visualization signal.

## V3 technical reference

### Observation variables and ranges

| Variable | Range | Meaning |
|---|---:|---|
| `battery` | `0..4` | Stored energy level |
| `demand` | `0..3` | Electricity demand level |
| `renewable` | `0..3` | Renewable generation level |
| `price` | `0..2` | Electricity price level |
| `period` | `0..3` | Time-of-day period |
| `weather` | `0..2` | Simplified weather condition |

- Observation space: `MultiDiscrete([5, 4, 4, 3, 4, 3])`
- Action space: `Discrete(5)`
- Q-table shape: `(5, 4, 4, 3, 4, 3, 5)`

The first six Q-table dimensions correspond to the discrete observation variables, and the final dimension corresponds to the five discrete actions available in V3.

### V3 CSV columns

| Column | Range | Meaning |
|---|---:|---|
| `step` | `>= 0` | Hourly index in the synthetic weekly sequence |
| `hour` | `0..23` | Hour of day |
| `weather_level` | `0..2` | Simplified weather condition |
| `demand_level` | `0..3` | Discrete electricity demand level |
| `renewable_level` | `0..3` | Discrete renewable generation level |
| `price_level` | `0..2` | Discrete electricity price level |

Each V3 episode contains 168 hourly steps, corresponding to one synthetic week.

### Reward weights

| Reward key | Weight | Purpose |
|---|---:|---|
| `demand_covered` | `2.0` | Reward covered demand |
| `unmet_demand` | `-5.0` | Penalize uncovered demand |
| `grid_bought` | `-0.8` | Penalize external grid purchases |
| `sold` | `0.8` | Reward selling surplus energy |
| `invalid_action` | `-1.5` | Penalize useless or impossible actions |
| `wasted_renewable` | `-0.5` | Penalize unused renewable surplus |
| `risk` | `-0.03` | Penalize high fuzzy-risk states |

All final V3 scenarios use the same reward-weight structure, so scenario comparisons are made under a common objective.

### Main evaluation metrics

| Metric | Meaning |
|---|---|
| `avg_reward` | Average reward obtained during greedy evaluation |
| `avg_coverage` | Share of demand covered during evaluation |
| `avg_unmet_demand` | Average unmet demand |
| `avg_grid_bought` | Average energy bought from the grid |
| `avg_sold` | Average surplus energy sold |
| `avg_wasted_renewable` | Average renewable surplus wasted |
| `avg_battery_end` | Battery level at the end of the evaluation episode |
| `avg_risk_score` | Average fuzzy risk score |
| `invalid_action_rate` | Share of invalid or useless actions |

## V3 scenarios

| Scenario | Purpose |
|---|---|
| `baseline_v3` | Reference scenario with balanced demand, renewable generation and prices |
| `winter_peak` | High-demand scenario with lower renewable availability and stronger price pressure |
| `summer_surplus` | Scenario with higher renewable generation and surplus-management opportunities |
| `grid_stress` | Stress scenario with demand peaks and limited renewable support |
| `renewable_volatility` | Scenario with unstable renewable generation and price variation |

Each scenario uses:

- a JSON configuration file in `data/scenarios/`,
- a training CSV time series,
- an evaluation CSV time series.

The scenario and time-series loaders validate the JSON configuration and the referenced CSV files before constructing the V3 environment.

## Training and evaluation

Train one V3 scenario:

```bash
python -m src.training.train --version v3 --scenario baseline_v3
```

Train all V3 scenarios:

```bash
python -m src.training.train --version v3 --all-v3
```

Evaluate one V3 scenario:

```bash
python -m src.training.evaluate --version v3 --scenario baseline_v3
```

Evaluate all V3 scenarios:

```bash
python -m src.training.evaluate --version v3 --all-v3
```

## Dashboard

The Streamlit dashboard reads V3 evaluation summaries and demo episodes from:

- `results/v3/summaries/`
- `results/v3/demos/`

Run the dashboard with:

```bash
streamlit run src/dashboard/app.py
```

## V3.1 PSO reward-weight tuning

V3.1 adds an optional optimization layer based on Particle Swarm Optimization. Q-learning remains the main learning algorithm: PSO does not control the environment directly and does not replace the agent. Instead, PSO searches for reward-weight configurations used by `SmartGridEnvV3`.

Because PSO is an optional optimization experiment rather than the main learning pipeline, its outputs are stored separately under `results/v3_1/`. The main V3 training and evaluation artifacts remain under `results/v3/`.

Each PSO particle represents the following reward-weight vector:

```text
[demand_covered, unmet_demand, grid_bought, sold, invalid_action, wasted_renewable, risk]
```

For each candidate vector, the PSO experiment:

1. creates a temporary V3 environment,
2. trains a temporary Q-learning agent,
3. evaluates the greedy policy,
4. computes a fitness score using reward, coverage, unmet demand, risk and invalid actions.

The `pyswarm` implementation minimizes the objective function, so this project minimizes negative fitness.

Smoke command:

```bash
python -m src.optimization.pso_reward_tuning --scenario baseline_v3 --episodes 50 --maxiter 2 --swarmsizes 3 --omegas 0.5 --phips 1.0 --phigs 1.0
```

Standard command:

```bash
python -m src.optimization.pso_reward_tuning --scenario baseline_v3
```

## Results and outputs

Representative artifacts are included so the reviewer can inspect results without retraining every experiment.

Results under `results/v3/` correspond to the main final pipeline: `SmartGridEnvV3`, tabular Q-learning, scenario evaluation and dashboard outputs.

Results under `results/v3_1/` correspond only to the optional PSO reward-weight tuning experiment. They are separated from the main V3 outputs because PSO is an optimization layer for reward weights, not the main learning algorithm.

| Output type | Folder |
|---|---|
| Final academic report | `results/smartgrid_es_v3_report.pdf` |
| Trained Q-table models | `results/v3/models/` |
| Training and comparison plots | `results/v3/plots/` |
| Greedy demo episodes | `results/v3/demos/` |
| Evaluation summaries | `results/v3/summaries/` |
| PSO summaries | `results/v3_1/summaries/` |
| PSO best configurations | `results/v3_1/best_configs/` |
| PSO plots | `results/v3_1/plots/` |

The repository includes a V3 evaluation summary CSV in `results/v3/summaries/`. It compares the five V3 scenarios using metrics such as:

- average reward
- coverage
- unmet demand
- grid energy bought
- sold energy
- wasted renewable energy
- risk score
- invalid action rate

## Installation

```bash
git clone -b v3-main https://github.com/jaimeeblancoo/smartgrid-rl-es.git
cd smartgrid-rl-es
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

During development, use `v3-dev`; after final promotion, use `v3-main`.

## Reproducing the V3 results

```bash
python -m scripts.generate_synthetic_timeseries
python -m src.training.train --version v3 --all-v3
python -m src.training.evaluate --version v3 --all-v3
streamlit run src/dashboard/app.py
```

## Repository structure

```text
smartgrid-rl-es/
├── README.md
├── requirements.txt
├── data/
│   ├── scenarios/
│   └── timeseries/
├── results/
│   ├── v3/
│   └── v3_1/
├── scripts/
└── src/
    ├── agents/
    ├── dashboard/
    ├── envs/
    ├── optimization/
    ├── training/
    ├── utils/
    └── wrappers/
```

## Scope and limitations

- The environment is discrete and simplified.
- CSV data are synthetic and controlled.
- The Q-learning agent is tabular, not deep reinforcement learning.
- V3 does not implement real multi-agent reinforcement learning.
- V3 does not use live real-world data, external APIs or LLM agents.
- PSO is an optional reward-weight tuning experiment, not the main control algorithm.
- A Markov-chain scenario generator or mode was considered, but it is not part of the final V3 implementation.
