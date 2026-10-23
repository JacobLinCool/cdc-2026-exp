"""Standalone figures for measured capacity × test-time interactions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOLDER = Path(__file__).resolve().parent
COLORS = {"139": "#2D7364", "499": "#A77520", "1948": "#6B5AAB"}


def bounds(items, key="score", scale=100):
    if key == "elo":
        items = [item["elo"] for item in items]
        key = "estimate"
    center = np.array([item[key] for item in items]) * scale
    intervals = np.array([item["paired_bootstrap_95_ci"] for item in items]) * scale
    return center, np.vstack([center - intervals[:, 0], intervals[:, 1] - center])


def main():
    result = json.loads((FOLDER / "results.json").read_text())
    times = result["time_budgets_ms"]
    nodes = result["node_budgets"]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), layout="constrained")
    for capacity, data in result["anchor"].items():
        items = [data["budgets"][str(t)] for t in times]
        center, error = bounds(items, key="elo", scale=1)
        axes[0, 0].errorbar(
            times,
            center,
            yerr=error,
            marker="o",
            capsize=4,
            color=COLORS[capacity],
            label=f"{int(capacity):,} parameters",
        )
        gains = [
            data["gains"][f"{a}->{b}"]["elo_gain_per_doubling"]
            for a, b in zip(times, times[1:])
        ]
        center, error = bounds(gains, key="estimate", scale=1)
        offsets = {"139": -0.13, "499": 0, "1948": 0.13}
        axes[1, 0].errorbar(
            np.arange(2) + offsets[capacity],
            center,
            yerr=error,
            marker="o",
            linestyle="none",
            capsize=4,
            color=COLORS[capacity],
        )
    axes[0, 0].set_title("Against the same 40 ms opponent")
    axes[0, 0].set_ylabel("Opponent-relative Elo")
    axes[0, 0].set_xlabel("Model's search budget (ms / move)")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(times, [str(t) for t in times])
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0, 0].axhline(0, color="#888888", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Marginal gain from more search")
    axes[1, 0].set_ylabel("Elo gain per budget doubling")
    axes[1, 0].set_xticks([0, 1], ["10 → 40 ms", "40 → 160 ms"])
    axes[1, 0].set_xlim(-0.4, 1.4)
    axes[1, 0].axhline(0, color="#888888", linestyle="--", linewidth=1)
    for ax, family, budgets, xlabel, title in [
        (
            axes[0, 1],
            "direct",
            times,
            "Search budget for both players (ms / move)",
            "Direct match: 1,948 vs 139 parameters",
        ),
        (
            axes[1, 1],
            "node",
            nodes,
            "Search node cap for both players",
            "Direct match with equal node caps",
        ),
    ]:
        items = [result[family]["budgets"][str(t)] for t in budgets]
        center, error = bounds(items)
        ax.errorbar(
            budgets, center, yerr=error, marker="s", capsize=4, color=COLORS["1948"]
        )
        ax.axhline(50, color="#888888", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [f"{t:,}" for t in budgets])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Larger model score (%)")
        ax.set_ylim(
            min(35, np.min(center - error[0]) - 2),
            max(65, np.max(center + error[1]) + 2),
        )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.16)
    fig.suptitle(
        "Does a larger evaluator gain more from additional search?", fontsize=14
    )
    fig.supxlabel(
        "2,592 games · 3 fixed training seeds · deal-paired 95% intervals · three budgets cannot identify an asymptotic ceiling",
        fontsize=9,
    )
    fig.savefig(FOLDER / "test-time-scaling.png", dpi=180)
    fig.savefig(FOLDER / "test-time-scaling.svg")


if __name__ == "__main__":
    main()
