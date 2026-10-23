"""Combine the registered sweep and the separately declared interaction supplement."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training/scaling"
JOINT = ROOT / "training/scaling-joint"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def score(report):
    low, high = report["paired_bootstrap_95_ci"]
    return f"{100*report['score']:.2f}% [{100*low:.2f}%, {100*high:.2f}%]"


def main():
    original = read(BASE / "summary.json")
    supplement = read(JOINT / "result.json")
    assert original["final"]["hashes_verified"] and supplement["hashes_verified"]
    conditions = original["conditions"].copy()
    timings = {}
    for line in (JOINT / "throughput.txt").read_text().splitlines():
        match = re.match(r"(.+): median_ns=([\d.]+)", line)
        if match:
            timings[match[1]] = float(match[2])
    # Report annotations name the actual comparator; the shared numeric contrast routine
    # was originally written for the one-axis sweep's small-data control.
    for condition, reports in supplement["results"].items():
        for mode in ["time", "nodes"]:
            reports[mode]["contrast_to_data4x"][
                "interpretation"
            ] = "Exploratory matched-deal score difference versus 2048-game, 139-parameter training."
            reports[mode]["contrast_to_same_capacity_small_data"][
                "interpretation"
            ] = "Exploratory matched-deal score difference versus the same capacity trained on 512 games."
        fits = reports["fits"]
        selected = min(fits, key=lambda m: (m["model_validation_loss"], m["seed"]))
        conditions[condition] = {
            "training_positions": selected["training_positions"],
            "trainable_parameters": selected["trainable_parameters"],
            "observed_nonzero_extra_features": selected[
                "observed_nonzero_extra_features"
            ],
            "training_phase_counts": selected["training_phase_counts"],
            "model_bytes": selected["model_bytes"],
            "selected_seed": selected["seed"],
            "validation_loss_mean": statistics.mean(
                m["model_validation_loss"] for m in fits
            ),
            "validation_loss_std": statistics.pstdev(
                m["model_validation_loss"] for m in fits
            ),
            "baseline_validation_loss": selected["baseline_validation_loss"],
            "median_eval_ns": statistics.median(
                timings[f"training/scaling-joint/{condition}-{m['seed']}.cxe"]
                for m in fits
            ),
            "total_fitting_seconds": sum(m["elapsed_seconds"] for m in fits),
            "time": reports["time"],
            "nodes": reports["nodes"],
        }
    supplement[
        "report_annotation"
    ] = "Contrast descriptions were made comparator-specific after completion; all numerical results are unchanged."
    write(JOINT / "result.json", supplement)
    write(JOINT / "development-results.json", supplement["results"])
    entropy = read(BASE / "loss-diagnostics.json")["validation_target_entropy"]
    initial_loss = conditions["small"]["baseline_validation_loss"]
    for record in conditions.values():
        record["validation_excess_cross_entropy"] = (
            record["validation_loss_mean"] - entropy
        )
        record["reducible_validation_error_reduction"] = (
            initial_loss - record["validation_loss_mean"]
        ) / (initial_loss - entropy)
    overview = {
        "conditions": conditions,
        "condition_count": len(conditions),
        "training_fits": len(conditions) * 3,
        "development_games": sum(
            m["time"]["games"] + m["nodes"]["games"] for m in conditions.values()
        ),
        "final_games": sum(m["games"] for m in original["final"]["holdouts"].values()),
        "main_candidate": original["final"]["candidate"]["model"],
        "main_holdouts": original["final"]["holdouts"],
        "supplement_advanced_to_holdout": supplement["advanced_to_holdout"],
        "initial_validation_excess_cross_entropy": initial_loss - entropy,
        "training_job_seconds_sum": sum(
            m["total_fitting_seconds"] for m in conditions.values()
        ),
        "colab_gpu_allocated": False,
        "interpretation": "Larger data and composition tables improve teacher-label fitting; this sweep establishes no reliable playing-strength gain over the preceding structured evaluator.",
    }
    if supplement["advanced_to_holdout"]:
        overview["supplement_holdouts"] = supplement["holdouts"]
        overview["final_games"] += sum(
            m["games"] for m in supplement["holdouts"].values()
        )
    # Verify retained model bytes and frozen computational sources, independently of report text.
    frozen = original["final"]["candidate"]
    assert (
        hashlib.sha256((ROOT / frozen["model"]).read_bytes()).hexdigest()
        == frozen["model_sha256"]
    )
    for path, digest in frozen["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for path, digest in read(JOINT / "run.json")["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    overview["frozen_sources_reverified"] = True
    write(ROOT / "training/scaling-overview.json", overview)
    rows = [
        "# Data and capacity together",
        "",
        "This supplement completes the 2 × 3 data/capacity grid. Its separate protocol was saved after the original seven-condition development sweep and while that sweep’s frozen holdout was running. The original holdout results do not determine advancement. All six supplemental fits and 720 development games are retained.",
        "",
        "| Training games | 139 parameters | 499 parameters | 1,948 parameters |",
        "| ---: | --- | --- | --- |",
        "| 512 | "
        + " | ".join(
            score(conditions[n]["time"])
            for n in ["small", "capacity500", "capacity2000"]
        )
        + " |",
        "| 2,048 | "
        + " | ".join(
            score(conditions[n]["time"])
            for n in ["data4x", "data4x-capacity500", "data4x-capacity2000"]
        )
        + " |",
        "",
        "Cells are 25 ms development scores and deal-paired 95% intervals against the preceding structured evaluator. Each cell combines three training seeds on the same disjoint deal blocks; it is not one model’s held-out strength. The two new conditions scored 52.50% and 49.58%, below the original control family’s 54.375%. The predeclared advancement rule was not met, so no supplemental final matches were started.",
        "",
        "| Additional condition | 8,192-node score [95% interval] | Mean validation loss | Native eval ns |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in ["data4x-capacity500", "data4x-capacity2000"]:
        m = conditions[name]
        rows.append(
            f"| {name} | {score(m['nodes'])} | {m['validation_loss_mean']:.7f} | {m['median_eval_ns']:.0f} |"
        )
    largest = conditions["data4x-capacity2000"]
    rows += [
        "",
        f"The largest-data/largest-capacity condition reduces excess validation cross entropy by {100*largest['reducible_validation_error_reduction']:.2f}% relative to the shared initial evaluator. Excess cross entropy subtracts the fixed teacher-target entropy; it is not an improvement in win rate. Its matched 25 ms score contrast against the 2,048-game, 139-parameter control still includes zero.",
        "",
        "![Data and capacity scaling](../scaling-grid.png)",
        "",
        "The curves cover only two data sizes and three composition-table capacities. They support improved fitting, not a playing-strength scaling law. Node-budget intervals are also inconclusive. Eighty epochs and the normalized mean-weight prior are held fixed; larger datasets therefore receive more optimizer steps, and capacity changes alter the prior’s per-coefficient normalization. These experiments do not isolate every optimization effect. The expansion adds composition interactions, not learned spatial features.",
        "",
        "All models start from the same previous structured evaluator. Data, labels, game splits, initial coefficients, optimizer settings and seed blocks are shared with the corresponding original conditions. Timing measurements are local single-process microbenchmarks, taken separately after each queue stopped; small timing differences should not be treated as statistically established speedups.",
        "",
        "Across both stages: **9 conditions, 27 fits, 3,240 development games and 2,800 final games**. No Colab runtime was allocated. The sum of the individual fitting-job durations is "
        + f"{overview['training_job_seconds_sum']:.1f} seconds; jobs ran concurrently. Search labeling and games dominate the workload.",
        "",
        "Evidence: [supplement protocol](experiment-plan.json), [sources and data hashes](run.json), [all results](result.json), [cross-language checks](verification.json), [main held-out report](../scaling/README.md), and [combined machine-readable summary](../scaling-overview.json).",
        "",
        "Reproduce the supplement from the repository root, with the retained corpora and completed main sweep:",
        "",
        "```sh",
        "python3 training/scaling-joint/run.py",
        "python3 training/scaling_summary.py",
        "MPLCONFIGDIR=/tmp/cdc-scaling-mpl PYENV_VERSION=3.11.1 python training/plot_scaling.py",
        "```",
        "",
        "The conditional final-test seed range was unused and remains available only under a newly specified protocol. No model has been promoted over `training/ranking/value.cse`; default engine behavior is unchanged.",
        "",
    ]
    (JOINT / "README.md").write_text("\n".join(rows))


if __name__ == "__main__":
    main()
