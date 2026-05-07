"""Compare the latest V2 and V3 evaluation summaries."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


V2_SUMMARIES_DIR = Path("results/summaries")
V3_SUMMARIES_DIR = Path("results/v3/summaries")
COMPARISON_DIR = Path("results/comparison")


def find_latest_csv(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.csv"))
    if not files:
        return None
    return files[-1]


def load_summary(path: Path, version: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.insert(0, "version", version)
    df.insert(1, "source_file", path.name)
    return df


def save_plot(comparison: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available, so the comparison plot was not generated.")
        return

    metric_columns = [
        col
        for col in ["avg_reward", "avg_coverage", "avg_grid_bought", "avg_sold", "avg_risk_score"]
        if col in comparison.columns
    ]
    if not metric_columns:
        print("No comparable metric columns were found, so the comparison plot was not generated.")
        return

    plot_data = comparison.set_index("version")[metric_columns]
    ax = plot_data.plot(kind="bar", figsize=(10, 5))
    ax.set_title("SmartGrid-ES V2 vs V3 summary comparison")
    ax.set_xlabel("Version")
    ax.set_ylabel("Metric value")
    ax.legend(loc="best")
    plt.tight_layout()

    plot_path = COMPARISON_DIR / "v2_vs_v3_summary.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")


def main() -> None:
    v2_summary = find_latest_csv(V2_SUMMARIES_DIR)
    v3_summary = find_latest_csv(V3_SUMMARIES_DIR)

    if v2_summary is None:
        print("No V2 summary CSV was found in results/summaries.")
    else:
        print(f"Latest V2 summary: {v2_summary}")

    if v3_summary is None:
        print("No V3 summary CSV was found in results/v3/summaries.")
    else:
        print(f"Latest V3 summary: {v3_summary}")

    if v2_summary is None and v3_summary is None:
        print("No comparison file was created because both summary inputs are missing.")
        return

    rows = []
    if v2_summary is not None:
        rows.append(load_summary(v2_summary, "V2"))
    if v3_summary is not None:
        rows.append(load_summary(v3_summary, "V3"))

    comparison = pd.concat(rows, ignore_index=True, sort=False)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    output_path = COMPARISON_DIR / "v2_vs_v3_summary.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Comparison summary saved to: {output_path}")

    save_plot(comparison)


if __name__ == "__main__":
    main()
