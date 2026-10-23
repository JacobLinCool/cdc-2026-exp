"""Render the completed, frozen experiment as a compact report and figure."""

import json
import sys
import numpy as np
from common import ROOT, FOLDER, digest, save, verify

LABELS = {
    "continue-old": "Continue old targets",
    "deep-value": "Deeper value targets",
    "deep-loose": "Deeper targets, wider updates",
    "deep-close": "Emphasize close positions",
    "deep-rank-half": "Add ranking (0.5)",
    "deep-rank": "Add ranking (1.5)",
}


def interval(report):
    low, high = report["paired_bootstrap_95_ci"]
    return f"{100*low:.2f}–{100*high:.2f}%"


def figure(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.65, 1]}
    )
    names = list(result["development"])
    for y, name in enumerate(names):
        item = result["development"][name]
        center = item["score"] * 100
        low, high = np.array(item["paired_bootstrap_95_ci"]) * 100
        color = "#176b65" if name == result["selected"]["condition"] else "#78889a"
        axes[0].errorbar(
            center,
            y,
            xerr=[[center - low], [high - center]],
            fmt="o",
            capsize=3,
            color=color,
            markersize=7,
        )
        axes[0].text(high + 0.5, y, f"{center:.2f}%", va="center", fontsize=9)
    axes[0].set_yticks(range(len(names)), [LABELS[n] for n in names])
    axes[0].invert_yaxis()
    axes[0].set_title(
        "Development: 192 games per condition", loc="left", fontweight="bold"
    )
    axes[0].set_xlim(34, 68)
    if result["holdout"]:
        for y, (stage, item) in enumerate(result["holdout"].items()):
            center = item["score"] * 100
            low, high = np.array(item["paired_bootstrap_95_ci"]) * 100
            axes[1].errorbar(
                center,
                y,
                xerr=[[center - low], [high - center]],
                fmt="o",
                capsize=4,
                color=["#176b65", "#d58936"][y],
                markersize=8,
            )
            axes[1].text(
                center,
                y - 0.2,
                f"{center:.2f}% [{low:.2f}, {high:.2f}]",
                ha="center",
                fontsize=9,
            )
        axes[1].set_yticks(
            range(len(result["holdout"])),
            [
                f"{r['time_ms']} ms\n{r['games']:,} games"
                for r in result["holdout"].values()
            ],
        )
        axes[1].set_ylim(1.55, -0.65)
    else:
        axes[1].text(
            0.5, 0.5, "No condition advanced", ha="center", transform=axes[1].transAxes
        )
        axes[1].set_yticks([])
    axes[1].set_title("Fresh holdout: one frozen model", loc="left", fontweight="bold")
    axes[1].set_xlim(40, 65)
    for ax in axes:
        ax.axvline(50, color="#b4bac1", linestyle="--", linewidth=1)
        ax.set_xlabel("Score against frozen 499 baseline (%)")
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(
        "Can training improve the same 499-parameter evaluator?",
        x=0.02,
        ha="left",
        fontweight="bold",
        fontsize=15,
    )
    fig.text(
        0.02,
        0.015,
        "Draw = half a point. Bars: 95% whole-deal bootstrap intervals. Development selected the candidate; holdout tests it.",
        fontsize=9,
        color="#4a5562",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92], w_pad=2)
    for suffix in ["png", "svg"]:
        fig.savefig(
            FOLDER / f"results.{suffix}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main():
    verification = json.loads((FOLDER / "verification.json").read_text())
    verify(verification["hashes"])
    result = json.loads((FOLDER / "results.json").read_text())
    selected = result["selected"]
    metadata = json.loads((ROOT / selected["model"]).with_suffix(".json").read_text())
    train = json.loads((FOLDER / "train-labels.json").read_text())
    valid = json.loads((FOLDER / "validation-labels.json").read_text())
    conclusion = (
        "The selected candidate establishes an improvement at the primary 40 ms budget."
        if result["confirmed_improvement"]
        else "This experiment does not establish an improvement over the frozen 499-parameter baseline; retain the baseline."
    )
    rows = [
        "# Improving the 499-parameter evaluator",
        "",
        conclusion,
        "",
        f"The experiment completed **18 fits, {result['development_games']:,} development games and {result['holdout_games']:,} games on fresh holdout deals**. Every fit retains 499 trainable parameters (529 stored coefficients, 4,252 bytes). The baseline is [`data4x-capacity500-89.cxe`](../scaling-joint/data4x-capacity500-89.cxe), chosen by the lowest existing common validation loss among the three 499-parameter fits. All new fits start from its exact weights.",
        "",
        "![Experiment results](results.png)",
        "",
        "## Independent confirmation",
        "",
    ]
    if result["holdout"]:
        rows += [
            f"The frozen candidate is [`{selected['model'].split('/')[-1]}`]({selected['model'].split('/')[-1]}), selected by the [predeclared rule](experiment-plan.json).",
            "",
            "| Budget | Games | Wins / draws / losses | Score | Paired 95% interval | Raw win rate |",
            "| --- | ---: | --- | ---: | --- | ---: |",
        ]
        for report in result["holdout"].values():
            rows.append(
                f"| {report['time_ms']} ms | {report['games']} | {report['wins']} / {report['draws']} / {report['losses']} | {report['score']*100:.2f}% | {interval(report)} | {report['win_rate']*100:.2f}% |"
            )
        rows += [
            "",
            "Score is `(wins + draws / 2) / games`. Each hidden deal is played with both seating orders. The interval resamples complete deals 20,000 times. The primary test uses 600 fresh deals; the secondary uses another 200. Both fixed counts were declared before training and completed without optional stopping. The 160 ms interval is a pointwise secondary result.",
        ]
    else:
        rows += [
            "No condition exceeded 50% aggregate development score, so no candidate advanced under the declared rule."
        ]
    rows += [
        "",
        "## Development sweep",
        "",
        "These comparisons select a candidate and are exploratory. Each condition aggregates three optimization seeds, with 64 games per seed; conditions share the same 96 deals and seating orders. Intervals resample deals within each fixed optimization-seed block.",
        "",
        "| Training variant | Mean common validation loss | Development score | Paired 95% interval |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, item in result["development"].items():
        rows.append(
            f"| {LABELS[name]} | {item['mean_validation_loss']:.8f} | {100*item['score']:.2f}% | {interval(item)} |"
        )
    rows += [
        "",
        "The close-position objective and ranking weight 1.5 selected epoch zero for all three seeds. Their exported weights are byte-identical to the baseline; their development scores reflect match variability, not learned changes.",
    ]
    rows += [
        "",
        f"The selected fit's common validation loss changed from {metadata['baseline_validation_loss']:.8f} to {metadata['model_validation_loss']:.8f}; its validation preference agreement changed from {100*metadata['initial_pair_metrics']['agreement']:.2f}% to {100*metadata['model_pair_metrics']['agreement']:.2f}% (ties count as half). These are teacher-fitting diagnostics, not playing-strength evidence.",
        "",
        "## Training and controls",
        "",
        f"The same 9,269 training roots and 1,104 validation roots used by the baseline were reconstructed from saved public states. The deeper teacher is the frozen baseline itself, searched for up to 2,097,152 nodes per root, with a cold transposition table and reset history. It produced **{train['valid_roots']:,} valid training roots and {valid['valid_roots']:,} valid validation roots**, plus {train['pairs']:,} training preference pairs and {valid['pairs']:,} validation pairs. Invalid labels are retained in the raw records but masked identically in every training condition. No validation pair is used for optimization.",
        "",
        "All six variants run 80 epochs for each of seeds 53, 71 and 89, with Adam at 0.01 and root batches of 256. They compare continued fitting to old targets, deeper scalar targets, fourfold update bounds with weaker regularization, emphasizing close positions, and two ranking-loss weights. Ranking fits sample 128 training pairs per root batch. Checkpoint selection always minimizes the same new-teacher validation BCE at temperature 600; the original weights at epoch zero remain eligible. Integer evaluator operations use an explicit straight-through surrogate during backpropagation; search and discrete rule features are not differentiated.",
        "",
        "Teacher labeling fixes the flip threshold at 5 and disables tablebases. Matches use the unchanged engine's production defaults: both players get the same wall-clock budget and tablebases; the flip threshold is 5 at 40 ms and 6 at 160 ms. Thus the secondary budget tests deployment behavior, rather than isolating a fixed-algorithm test-time scaling slope. Wall-clock matches can vary with scheduling; node-limited label generation is scheduling-independent. This is one training-generation experiment and cannot establish a capacity ceiling.",
        "",
        "Old scalar targets came from the residual-network teacher at 160 ms. New targets use the frozen 499 evaluator under a larger fixed node budget. Teacher identity and compute change together, so this experiment compares training recipes and does not isolate a causal effect of search depth alone.",
        "",
        "## Checks and reproduction",
        "",
        "The labeler reconstructs each public state and asserts exact feature and original-evaluator agreement. Its 18-root smoke test produces identical non-timing labels and preference records with one and two threads. Objective derivatives, child-side preference signs, the combined smooth computation graph, selection rules and seed-block resampling pass six focused checks. Every exported candidate matches native Rust predictions exactly on all 4,845 original validation states. Source, model, dataset and game-result hashes are frozen and verified at stage boundaries.",
        "",
        "Run from the repository root. Existing completed artifacts are verified before reuse; unregistered or partial outputs fail explicitly.",
        "",
        "Training was run with Python 3.14.7 and NumPy 2.4.6, recorded in the training freeze. Plot rendering also requires Matplotlib; this report was rendered with Matplotlib 3.10.7 in the available Python 3.11.1 environment.",
        "",
        "```sh",
        "cargo build --release --offline --manifest-path training/improve499/labeler/Cargo.toml",
        "python3 training/improve499/prepare.py",
        "OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 training/improve499/run_fits.py",
        "python3 training/improve499/diagnose.py",
        "python3 training/improve499/test_evaluation.py",
        "python3 training/improve499/evaluate.py",
        "PYENV_VERSION=3.11.1 MPLCONFIGDIR=/tmp/cdc-improve499-mpl python training/improve499/report.py",
        "```",
        "",
        "The root input format is the existing `CDCSCL01` public-state corpus. New `CDCFT001` records are 32 bytes: row id (u32), root-side score (i32), completed depth (u16), best move (u16), nodes (u64), elapsed milliseconds (f64) and validity flag (u32), all little-endian. Preferences use the existing `CDCRNK01` format. Raw binary corpora and build artifacts are ignored; all model files, manifests, metadata and match CSVs are retained. Regenerating the original wall-clock source corpus is not bit-reproducible; exact reruns require its recorded bytes.",
        "",
        "See [the plan](experiment-plan.json), [labeling freeze](labeling-freeze.json), [training freeze](training-freeze.json), [development freeze](development-freeze.json), [selection](selection.json), [full results](results.json), [native parity](native-parity.json) and [verification](verification.json).",
        "",
    ]
    (FOLDER / "README.md").write_text("\n".join(rows))
    figure(result)
    save(
        FOLDER / "report-artifacts.json",
        dict(
            python=sys.version,
            numpy=np.__version__,
            hashes={
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    FOLDER / "report.py",
                    FOLDER / "README.md",
                    FOLDER / "results.png",
                    FOLDER / "results.svg",
                    FOLDER / "diagnose.py",
                    FOLDER / "fit-diagnostics.json",
                ]
            },
            results_sha256=digest(FOLDER / "results.json"),
        ),
    )


if __name__ == "__main__":
    main()
