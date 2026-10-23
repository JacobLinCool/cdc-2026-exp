"""Standalone scientific plot of all declared development conditions."""
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).resolve().parent
summary = json.loads((folder / "summary.json").read_text())
conditions = summary["conditions"]
labels = list(conditions)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
colors = {"time": "#136F63", "nodes": "#C27322"}
y = np.arange(len(labels))
for key, offset, title in [
    ("time", -0.12, "25 ms / move"),
    ("nodes", 0.12, "8,192 nodes / move"),
]:
    mean = np.array([conditions[name][key]["score"] * 100 for name in labels])
    bounds = (
        np.array([conditions[name][key]["paired_bootstrap_95_ci"] for name in labels])
        * 100
    )
    axes[0].errorbar(
        mean,
        y + offset,
        xerr=np.vstack([mean - bounds[:, 0], bounds[:, 1] - mean]),
        fmt="o",
        capsize=3,
        color=colors[key],
        label=title,
        markersize=4,
    )
axes[0].axvline(50, color="#888888", linestyle="--", linewidth=1)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlabel("Score against previous structured evaluator (%)")
axes[0].set_title("Three training seeds per condition")
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
for name, m in conditions.items():
    axes[1].scatter(
        m["median_eval_ns"] / 1000, m["time"]["score"] * 100, s=38, color=colors["time"]
    )
    axes[1].annotate(
        name,
        (m["median_eval_ns"] / 1000, m["time"]["score"] * 100),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )
axes[1].axhline(50, color="#888888", linestyle="--", linewidth=1)
axes[1].set_xlabel("Native evaluation latency (μs; local microbenchmark)")
axes[1].set_ylabel("25 ms development score (%)")
axes[1].set_title("Evaluation cost and playing strength")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
fig.suptitle(
    "Initialized evaluator scaling sweep — exploratory development results", fontsize=13
)
fig.savefig(folder / "scaling-results.png", dpi=180)
fig.savefig(folder / "scaling-results.svg")
