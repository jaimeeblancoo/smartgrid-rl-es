# SmartGrid-ES

SmartGrid-ES is an academic reinforcement learning project for a simplified smart-grid-inspired microgrid.

The project studies how an interpretable tabular Q-learning agent can learn energy-management decisions in controlled synthetic scenarios. The environment is intentionally discrete and simplified: the goal is not to reproduce a detailed real power system, but to build a clear, reproducible and defensible reinforcement learning pipeline for coursework.

V2 introduced wrappers and comparative evaluation. V3 is the current final version of the project, and adds a custom Gymnasium environment, synthetic CSV time series, JSON scenario configuration, fuzzy energy risk, richer metrics, a V3 dashboard and a V3.1 PSO reward-weight tuning feature.

---

## Final V3 overview

V3 moves beyond the V2 wrapper-based extension into a dedicated custom Gymnasium environment: `SmartGridEnvV3`.

The final V3 pipeline includes:
- synthetic CSV time series for reproducible demand, renewable generation, weather and price profiles,
- external scenario configuration through JSON files,
- five V3 scenarios for controlled comparison,
- tabular Q-learning for interpretability,
- fuzzy energy risk as a metric, reward penalty and dashboard signal,
- V3 training and evaluation commands,
- a V3 Streamlit dashboard,
- representative V3 result artifacts,
- and V3.1 PSO reward-weight tuning with `pyswarm`.

V3.1 is included in the final V3 branch as an optimization layer. It does not replace Q-learning: PSO searches for reward-weight configurations, while Q-learning remains the learning agent.

The project remains local, reproducible and academic. It does not use external APIs, LLM agents, live real-world data, deep learning or real multi-agent reinforcement learning.

---

## Course concepts used

| Course concept | Implementation in SmartGrid-ES |
|---|---|
| Reinforcement Learning | Tabular Q-learning agent |
| Gymnasium environments | Custom `SmartGridEnvV3` |
| MDP formulation | Discrete state, actions, reward and transitions |
| Fuzzy logic | Fuzzy energy risk score |
| Evolutionary optimization / PSO | V3.1 reward-weight tuning with `pyswarm` |
| Experimental evaluation | Scenario metrics, plots, CSV outputs and dashboard |

Markov mode is not implemented in the current V3 delivery. It can be considered only as a possible future extension.

---

## V3 environment design

`SmartGridEnvV3` represents a simplified microgrid with battery storage, electricity demand, renewable generation, market price, time period and weather conditions.

The V3 observable state is:

```text
[battery, demand, renewable, price, period, weather]
```

These are discrete levels, not continuous real-world measurements.

| State variable | Meaning |
|---|---|
| `battery` | current stored energy level |
| `demand` | current electricity demand level |
| `renewable` | current renewable generation level |
| `price` | current electricity price level |
| `period` | time-of-day period |
| `weather` | simplified weather condition level |

The V3 action space has five discrete actions:

| Action ID | Meaning |
|---|---|
| `0` | use battery |
| `1` | buy from grid |
| `2` | store surplus |
| `3` | sell surplus |
| `4` | hold |

Fuzzy risk is calculated separately and is not part of `observation_space`. The risk outputs are:
- `risk_score`
- `risk_level`
- `risk_level_name`

Risk is used as:
- an evaluation metric,
- a V3 reward penalty,
- and a dashboard visualization signal.

---

## V3 reward intuition

The V3 reward function is designed to guide the agent toward reasonable energy-management behavior.

It rewards or penalizes:
- covered demand,
- unmet demand,
- grid energy bought,
- surplus energy sold,
- invalid actions,
- wasted renewable energy,
- and fuzzy energy risk.

The reward is still part of a simplified academic simulation. It is not intended to represent a real electricity-market settlement model.

---

## V3 scenarios

V3 includes five scenarios:
- `baseline_v3`
- `winter_peak`
- `summer_surplus`
- `grid_stress`
- `renewable_volatility`

Each V3 scenario uses:
- a JSON configuration file in `data/scenarios/`,
- a training CSV time series,
- and an evaluation CSV time series.

This design separates scenario data from environment code and makes the experiments easier to reproduce.

---

## V3 training, evaluation and dashboard

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

Run the V3 dashboard:

```bash
streamlit run src/dashboard/app.py
```

The dashboard reads the latest V3 summaries and demo episode files from `results/v3/`.

---

## V3.1 PSO reward tuning feature

V3.1 is a PSO-based reward-weight tuning feature included in the final V3 branch.

It follows the style used in class:
- `from pyswarm import pso`
- `numpy`
- `pandas`
- `itertools.product`
- `logging`
- `contextlib.redirect_stdout`
- CSV export of optimization results

Each PSO particle represents a candidate reward-weight vector for `SmartGridEnvV3`. For each candidate vector, the objective function trains a temporary tabular Q-learning agent and evaluates it greedily. Since `pyswarm` minimizes the objective, the implementation minimizes negative fitness.

V3.1 does not replace the main V3 training or evaluation pipeline. It is an additional optimization layer for controlled reward-weight experiments.

Smoke command:

```bash
python -m src.optimization.pso_reward_tuning --scenario baseline_v3 --episodes 50 --maxiter 2 --swarmsizes 3 --omegas 0.5 --phips 1.0 --phigs 1.0
```

Standard command:

```bash
python -m src.optimization.pso_reward_tuning --scenario baseline_v3
```

---

## Outputs

V3 outputs are saved under:

| Output type | Folder |
|---|---|
| Q-table models | `results/v3/models/` |
| Training and comparison plots | `results/v3/plots/` |
| Demo episode CSV files | `results/v3/demos/` |
| Evaluation summaries | `results/v3/summaries/` |

V3.1 PSO outputs are saved under:

| Output type | Folder |
|---|---|
| PSO result CSV files | `results/v3_1/summaries/` |
| Best reward-weight JSON files | `results/v3_1/best_configs/` |
| PSO fitness plots | `results/v3_1/plots/` |

Representative V3 and V3.1 outputs are included so the final branch can be inspected without rerunning every experiment.

---

## V2 context

V2 is kept as useful historical context and remains part of the repository.

V2 introduced:
- scenario-based environment construction through `src/envs/factory.py`,
- wrapper-based behavior changes,
- comparative evaluation across scenarios,
- CSV and plot outputs,
- and an earlier dashboard workflow.

The V2 scenarios are:
- `baseline`
- `winter`
- `summer`
- `demand_noise`
- `battery_loss`
- `combined_v2`

The V2 wrappers are:
- `SeasonWrapper`
- `DemandNoiseWrapper`
- `BatteryLossWrapper`
- `RewardShapingWrapper`

V2 helped establish the experimentation workflow, but V3 is the current final version used for the main project delivery.

---

## Installation

Clone the final V3 development branch and install the required packages:

```bash
git clone -b v3-dev https://github.com/jaimeeblancoo/smartgrid-rl-es.git
cd smartgrid-rl-es
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dependencies

The project uses:
- `gymnasium`
- `numpy`
- `matplotlib`
- `pandas`
- `streamlit`
- `pyswarm`

No external APIs are required.

---

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

---

## Limitations

- The environment is intentionally discrete and simplified.
- CSV data are synthetic and controlled, not live real-world data.
- V3 does not implement real multi-agent reinforcement learning.
- V3 does not use external APIs or LLM agents.
- Markov mode is not implemented in the current V3 delivery.
- PSO tuning is computationally more expensive than normal training, so it is mainly used for controlled reward-weight experiments.

---

## Summary

SmartGrid-ES V3 is the final main version of the project. It provides a local and reproducible academic RL pipeline built around `SmartGridEnvV3`, synthetic scenario data, fuzzy energy risk, tabular Q-learning, evaluation outputs, a dashboard and an optional V3.1 PSO reward-weight tuning feature.

The project remains deliberately interpretable: the agent is tabular, the state and actions are discrete, the data are synthetic and the outputs are designed for inspection through CSV files, plots and the dashboard.
