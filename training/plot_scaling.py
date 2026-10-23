"""Publication-ready view of the complete data/capacity grid."""
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).resolve().parent
summary = json.loads((folder / "scaling-overview.json").read_text())
records = summary["conditions"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
parameters = np.array([139, 499, 1948])
for games, names, color, marker in [
    (512, ["small", "capacity500", "capacity2000"], "#136F63", "o"),
    (2048, ["data4x", "data4x-capacity500", "data4x-capacity2000"], "#BA5C24", "s"),
]:
    rows = [records[n] for n in names]
    loss = np.array([m["validation_excess_cross_entropy"] for m in rows]) * 1000
    sd = np.array([m["validation_loss_std"] for m in rows]) * 1000
    axes[0].errorbar(
        parameters,
        loss,
        yerr=sd,
        color=color,
        marker=marker,
        capsize=3,
        label=f"{games:,} training games",
    )
    scores = np.array([m["time"]["score"] for m in rows]) * 100
    bounds = np.array([m["time"]["paired_bootstrap_95_ci"] for m in rows]) * 100
    axes[1].errorbar(
        parameters,
        scores,
        yerr=np.vstack([scores - bounds[:, 0], bounds[:, 1] - scores]),
        color=color,
        marker=marker,
        capsize=4,
        label=f"{games:,} training games",
    )
axes[0].set_ylabel("Validation excess cross-entropy (× 10⁻³)")
axes[0].set_title("Fitting deeper teacher labels\nlower is better", fontsize=11)
axes[1].set_ylabel("25 ms development score (%)")
axes[1].set_title(
    "Games against previous structured evaluator\nhigher is better", fontsize=11
)
axes[1].axhline(50, color="#888888", linestyle="--", linewidth=1)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks(parameters, [f"{p:,}" for p in parameters])
    ax.set_xlabel("Trainable parameters")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle(
    "Data × capacity: fitting improves; playing-strength gains remain unproven",
    fontsize=12,
)
fig.supxlabel(
    "Three training seeds per condition · left: seed SD · right: deal-paired 95% intervals · development results",
    fontsize=8,
)
fig.savefig(folder / "scaling-grid.png", dpi=180)
fig.savefig(folder / "scaling-grid.svg")
