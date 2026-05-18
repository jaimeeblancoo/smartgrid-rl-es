"""Streamlit dashboard for V3 SmartGrid-ES result files.

The dashboard reads saved V3 evaluation summaries and demo episode CSV files.
It visualizes tabular Q-learning outputs and the fuzzy risk signal exposed by
``SmartGridEnvV3`` through the ``info`` dictionary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


RESULTS_DIR = Path("results")
SUMMARIES_DIR = RESULTS_DIR / "v3" / "summaries"
DEMOS_DIR = RESULTS_DIR / "v3" / "demos"

ACTION_NAMES = {
    0: "Use battery",
    1: "Buy from grid",
    2: "Store surplus",
    3: "Sell surplus",
    4: "Hold",
}


def get_latest_summary_file() -> Path | None:
    """Return the newest V3 evaluation summary CSV, if one exists."""
    if not SUMMARIES_DIR.exists():
        return None
    files = sorted(SUMMARIES_DIR.glob("*.csv"))
    if not files:
        return None
    return files[-1]


def get_demo_files() -> list[Path]:
    """Return available V3 demo episode CSV files."""
    if not DEMOS_DIR.exists():
        return []
    return sorted(DEMOS_DIR.glob("*.csv"))


def load_csv(path: Path) -> pd.DataFrame:
    """Load a dashboard CSV file into a DataFrame.

    Args:
        path: CSV file path to load.

    Returns:
        DataFrame containing the selected dashboard data.
    """
    return pd.read_csv(path)


def add_action_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable V3 action names when an action column is present.

    Args:
        df: Demo episode DataFrame.

    Returns:
        DataFrame with an ``action_name`` column when possible.
    """
    if "action" not in df.columns:
        return df
    df = df.copy()
    df["action_name"] = df["action"].map(ACTION_NAMES).fillna("Unknown")
    return df


def format_percentage(value: float) -> str:
    """Format decimal or already-percent values for display."""
    if pd.isna(value):
        return "n/a"
    percentage = float(value)
    if abs(percentage) <= 1.5:
        percentage *= 100
    return f"{percentage:.1f}%"


def as_percentage_series(series: pd.Series) -> pd.Series:
    """Return a display copy of a series using percentage units."""
    numeric = pd.to_numeric(series, errors="coerce")
    max_abs = numeric.abs().max(skipna=True)
    if pd.notna(max_abs) and max_abs <= 1.5:
        return numeric * 100
    return numeric


def get_scenario_indexed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Use scenario labels as the chart index when available."""
    if "scenario" in df.columns:
        return df.set_index("scenario")
    return df


def scenario_label(df: pd.DataFrame, index: object) -> str:
    """Return a readable scenario label for a summary row."""
    if "scenario" in df.columns:
        return str(df.loc[index, "scenario"])
    return f"Row {index}"


def safe_metric(label: str, value: str, detail: str | None = None) -> None:
    """Render a compact metric when a value is available."""
    if value:
        st.metric(label, value, detail)


def show_summary_kpis(summary_df: pd.DataFrame) -> None:
    """Render compact KPIs for the evaluation summary when columns exist."""
    kpis: list[tuple[str, str, str]] = []

    if "avg_reward" in summary_df.columns:
        rewards = pd.to_numeric(summary_df["avg_reward"], errors="coerce")
        if rewards.notna().any():
            best_index = rewards.idxmax()
            worst_index = rewards.idxmin()
            kpis.append(
                (
                    "Best reward",
                    scenario_label(summary_df, best_index),
                    f"{rewards.loc[best_index]:.2f}",
                )
            )
            kpis.append(
                (
                    "Worst reward",
                    scenario_label(summary_df, worst_index),
                    f"{rewards.loc[worst_index]:.2f}",
                )
            )

    if "avg_coverage" in summary_df.columns:
        coverage = pd.to_numeric(summary_df["avg_coverage"], errors="coerce")
        if coverage.notna().any():
            coverage_index = coverage.idxmax()
            kpis.append(
                (
                    "Highest coverage",
                    scenario_label(summary_df, coverage_index),
                    format_percentage(coverage.loc[coverage_index]),
                )
            )

    if "avg_risk_score" in summary_df.columns:
        risk = pd.to_numeric(summary_df["avg_risk_score"], errors="coerce")
        if risk.notna().any():
            risk_index = risk.idxmax()
            kpis.append(
                (
                    "Highest fuzzy risk",
                    scenario_label(summary_df, risk_index),
                    f"{risk.loc[risk_index]:.2f}",
                )
            )

    if "invalid_action_rate" in summary_df.columns:
        invalid_rate = pd.to_numeric(summary_df["invalid_action_rate"], errors="coerce")
        if invalid_rate.notna().any():
            invalid_index = invalid_rate.idxmin()
            kpis.append(
                (
                    "Lowest invalid rate",
                    scenario_label(summary_df, invalid_index),
                    format_percentage(invalid_rate.loc[invalid_index]),
                )
            )

    if not kpis:
        return

    st.subheader("Summary KPIs")
    columns = st.columns(len(kpis))
    for column, (label, value, detail) in zip(columns, kpis):
        with column:
            safe_metric(label, value, detail)


def show_demo_kpis(demo_df: pd.DataFrame) -> None:
    """Render compact KPIs for the selected demo episode."""
    kpis: list[tuple[str, str]] = []

    if "reward" in demo_df.columns:
        rewards = pd.to_numeric(demo_df["reward"], errors="coerce")
        if rewards.notna().any():
            kpis.append(("Total reward", f"{rewards.sum():.2f}"))
            kpis.append(("Average reward", f"{rewards.mean():.2f}"))

    if "battery" in demo_df.columns and not demo_df.empty:
        final_battery = pd.to_numeric(demo_df["battery"], errors="coerce").dropna()
        if not final_battery.empty:
            kpis.append(("Final battery", f"{final_battery.iloc[-1]:.0f}"))

    for column_name, label in [
        ("grid_bought", "Total grid bought"),
        ("sold", "Total sold"),
        ("unmet_demand", "Total unmet demand"),
    ]:
        if column_name in demo_df.columns:
            values = pd.to_numeric(demo_df[column_name], errors="coerce")
            if values.notna().any():
                kpis.append((label, f"{values.sum():.0f}"))

    if "risk_score" in demo_df.columns:
        risk = pd.to_numeric(demo_df["risk_score"], errors="coerce")
        if risk.notna().any():
            kpis.append(("Average risk", f"{risk.mean():.2f}"))

    if "invalid_action" in demo_df.columns:
        invalid_actions = pd.to_numeric(demo_df["invalid_action"], errors="coerce")
        if invalid_actions.notna().any():
            invalid_count = invalid_actions.sum()
            invalid_rate = invalid_count / len(demo_df) if len(demo_df) else 0
            kpis.append(
                ("Invalid actions", f"{invalid_count:.0f} ({format_percentage(invalid_rate)})")
            )
    elif "invalid_action_rate" in demo_df.columns:
        invalid_rate = pd.to_numeric(demo_df["invalid_action_rate"], errors="coerce")
        if invalid_rate.notna().any():
            kpis.append(("Invalid action rate", format_percentage(invalid_rate.mean())))

    if not kpis:
        return

    st.subheader("Demo KPIs")
    for row_start in range(0, len(kpis), 4):
        columns = st.columns(min(4, len(kpis) - row_start))
        for column, (label, value) in zip(columns, kpis[row_start : row_start + 4]):
            with column:
                safe_metric(label, value)


def main() -> None:
    """Render the Streamlit dashboard for V3 summaries and demo episodes."""
    st.set_page_config(page_title="SmartGrid-ES V3 Dashboard", layout="wide")

    st.title("SmartGrid-ES V3 Dashboard")
    st.write(
        "This dashboard reads saved V3 evaluation summaries and demo episode artifacts. "
        "It does not retrain the agent; it is intended for inspecting the final V3 results "
        "without opening the CSV files manually."
    )

    st.subheader("Evaluation summary")

    summary_file = get_latest_summary_file()
    if summary_file is None:
        st.warning("No V3 evaluation summary files were found in results/v3/summaries.")
    else:
        summary_df = load_csv(summary_file)
        st.caption(f"Loaded file: {summary_file.name}")
        show_summary_kpis(summary_df)
        indexed_summary = get_scenario_indexed_df(summary_df)

        if "avg_reward" in summary_df.columns:
            st.subheader("Average reward by scenario")
            reward_chart = indexed_summary["avg_reward"]
            st.bar_chart(reward_chart)

        if "avg_coverage" in summary_df.columns:
            st.subheader("Average coverage by scenario")
            coverage_chart = as_percentage_series(indexed_summary["avg_coverage"])
            coverage_chart.name = "avg_coverage_pct"
            st.line_chart(coverage_chart)
            st.caption("Coverage is displayed as a percentage.")

        if "avg_risk_score" in summary_df.columns:
            st.subheader("Average fuzzy risk by scenario")
            risk_chart = pd.to_numeric(indexed_summary["avg_risk_score"], errors="coerce")
            risk_chart.name = "avg_risk_score"
            st.line_chart(risk_chart)
            st.caption("Fuzzy risk is a score on a 0-100 scale.")

        energy_columns = [
            col
            for col in ["avg_grid_bought", "avg_sold", "avg_wasted_renewable"]
            if col in summary_df.columns
        ]
        if energy_columns:
            st.subheader("Energy flows by scenario")
            st.bar_chart(indexed_summary[energy_columns])

        if "invalid_action_rate" in summary_df.columns:
            st.subheader("Invalid action rate by scenario")
            invalid_rate_chart = as_percentage_series(indexed_summary["invalid_action_rate"])
            invalid_rate_chart.name = "invalid_action_rate_pct"
            st.bar_chart(invalid_rate_chart)

        st.subheader("Raw evaluation summary")
        st.dataframe(summary_df, use_container_width=True)

    st.divider()

    st.subheader("Demo episode viewer")

    demo_files = get_demo_files()
    if not demo_files:
        st.warning("No V3 demo episode CSV files were found in results/v3/demos.")
        return

    selected_demo_name = st.selectbox(
        "Select a demo episode file",
        options=[path.name for path in demo_files],
    )

    selected_demo_path = DEMOS_DIR / selected_demo_name
    demo_df = add_action_names(load_csv(selected_demo_path))

    st.caption(f"Loaded file: {selected_demo_path.name}")
    show_demo_kpis(demo_df)
    st.dataframe(demo_df, use_container_width=True)

    state_columns = [
        col for col in ["battery", "demand", "renewable", "price_level"] if col in demo_df.columns
    ]
    if state_columns:
        st.subheader("State levels over time")
        st.line_chart(demo_df[state_columns])

    if "reward" in demo_df.columns:
        st.subheader("Reward over time")
        st.line_chart(demo_df["reward"])

    if "risk_score" in demo_df.columns:
        st.subheader("Fuzzy risk over time")
        st.line_chart(demo_df["risk_score"])

    outcome_columns = [
        col
        for col in ["grid_bought", "sold", "demand_covered", "unmet_demand", "wasted_renewable"]
        if col in demo_df.columns
    ]
    if outcome_columns:
        st.subheader("Energy and action outcomes")
        st.line_chart(demo_df[outcome_columns])

    if "action" in demo_df.columns:
        st.subheader("Actions taken")
        if "action_name" in demo_df.columns:
            st.bar_chart(demo_df["action_name"].value_counts().sort_values(ascending=False))
        else:
            st.bar_chart(demo_df["action"].value_counts().sort_values(ascending=False))


if __name__ == "__main__":
    main()
