"""Generate the complete report after all predetermined comparisons finish."""
import json
import re
from pathlib import Path
import numpy as np
from data import read, phases, quiet, deep_valid

FOLDER = Path(__file__).resolve().parent


def percent(value):
    return f"{100*value:.2f}%"


def score(report):
    lo, hi = report["paired_bootstrap_95_ci"]
    return f"{percent(report['score'])} [{percent(lo)}, {percent(hi)}]"


def main():
    plan = json.loads((FOLDER / "experiment-plan.json").read_text())
    result = json.loads((FOLDER / "result.json").read_text())
    development = json.loads((FOLDER / "development-results.json").read_text())
    fits = json.loads((FOLDER / "training-results.json").read_text())["fits"]
    timing = {}
    for line in (FOLDER / "throughput.txt").read_text().splitlines():
        match = re.match(r"(.+): median_ns=([\d.]+) checksum=(-?\d+)", line)
        if match:
            timing[match[1]] = {"median_ns": float(match[2]), "checksum": int(match[3])}
    dataset = {}
    for name in ["train", "validation"]:
        records = read(FOLDER / f"{name}.bin")
        paired = deep_valid(records)
        dataset[name] = {
            "positions": len(records),
            "contributing_games": len(np.unique(records["seed"])),
            "phase_counts": np.bincount(phases(records), minlength=3).tolist(),
            "quiet_positions": int(quiet(records).sum()),
            "paired_teacher_positions": int(paired.sum()),
            "low_mean_depth_paired": float(records["depth"][paired].mean()),
            "high_mean_depth_paired": float(records["deep_depth"][paired].mean()),
            "teacher_score_changed_fraction": float(
                np.mean(records["teacher"][paired] != records["deep_teacher"][paired])
            ),
            "teacher_mean_absolute_change": float(
                np.mean(
                    abs(records["teacher"][paired] - records["deep_teacher"][paired])
                )
            ),
            "low_mean_ms_paired": float(records["time_ms"][paired].mean()),
            "high_mean_ms_paired": float(records["deep_time_ms"][paired].mean()),
        }
    summaries = {}
    for variant in plan["training"]["variants"]:
        name = variant["name"]
        group = [fits[f"{name}-{seed}"] for seed in plan["training"]["seeds"]]
        chosen = min(group, key=lambda m: (m["model_validation_loss"], m["seed"]))
        timings = [
            timing[f"training/scaling/{name}-{seed}.cxe"]["median_ns"]
            for seed in plan["training"]["seeds"]
        ]
        summaries[name] = {
            "training_positions": chosen["training_positions"],
            "trainable_parameters": chosen["trainable_parameters"],
            "observed_nonzero_extra_features": chosen[
                "observed_nonzero_extra_features"
            ],
            "training_phase_counts": chosen["training_phase_counts"],
            "model_bytes": chosen["model_bytes"],
            "selected_seed": chosen["seed"],
            "validation_loss_mean": float(
                np.mean([m["model_validation_loss"] for m in group])
            ),
            "validation_loss_std": float(
                np.std([m["model_validation_loss"] for m in group])
            ),
            "baseline_validation_loss": chosen["baseline_validation_loss"],
            "median_eval_ns": float(np.median(timings)),
            "total_fitting_seconds": sum(m["elapsed_seconds"] for m in group),
            "time": development["time"][name],
            "nodes": development["nodes"][name],
        }
    summary = {
        "datasets": dataset,
        "conditions": summaries,
        "timing": timing,
        "final": result,
        "primary_superiority": result["holdouts"]["current"]["paired_bootstrap_95_ci"][
            0
        ]
        > 0.5,
    }
    (FOLDER / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Scaling the initialized evaluator",
        "",
        f"The predeclared sweep finished: 7 conditions × 3 training seeds, 2,520 development games and 2,800 fresh final games. The frozen candidate is `{result['candidate']['model']}`.",
        "",
        "## Final comparisons",
        "",
        "The primary comparison is against the previous structured evaluator at 25 ms/move. All game counts were fixed in advance; each deal is played from both seats. Intervals bootstrap complete deals 20,000 times. The other comparisons are secondary, with pointwise intervals rather than a familywise superiority claim.",
        "",
        "| Opponent / time | W / D / L | Score [95% interval] | Elo estimate |",
        "| --- | ---: | --- | ---: |",
    ]
    for name, report in result["holdouts"].items():
        lines.append(
            f"| {name} | {report['wins']} / {report['draws']} / {report['losses']} | {score(report)} | {report['elo']:+.1f} |"
        )
    primary = result["holdouts"]["current"]
    lines += [
        "",
        (
            "The primary interval is entirely above 50%, supporting superiority at the tested 25 ms control."
            if summary["primary_superiority"]
            else "The primary interval does not establish superiority over the previous structured evaluator. This is not an equivalence or noninferiority test."
        ),
        "",
        "## Controlled sweep",
        "",
        "Each row aggregates three separately trained models. Their disjoint deal blocks are shared across conditions. The development results select a condition; within it, the final training seed is selected only by common validation loss. Development intervals and matched contrasts are exploratory.",
        "",
        "| Condition | Train positions | Trainable coefficients | Validation loss, mean ± SD | 25 ms score [95% interval] | 8,192-node score [95% interval] | Eval ns |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for name, m in summaries.items():
        lines.append(
            f"| {name} | {m['training_positions']:,} | {m['trainable_parameters']:,} | {m['validation_loss_mean']:.6f} ± {m['validation_loss_std']:.6f} | {score(m['time'])} | {score(m['nodes'])} | {m['median_eval_ns']:.0f} |"
        )
    lines += [
        "",
        "All development opponents use the previous structured model. The node budget replaces wall-clock stopping; the flip-depth threshold is fixed to five. It is a maximum number of counted negamax/quiescence entries, not an assertion that different evaluators visit identical trees. Raw CSVs retain nodes, completed depth, thinking time and outcomes.",
        "",
        "![Scaling results](scaling-results.png)",
        "",
        "## Data and controls",
        "",
        f"The actor played 2,048 training games and 256 separate validation games. Collection retained {dataset['train']['positions']:,} training roots and {dataset['validation']['positions']:,} validation roots. A label-independent reservoir samples at most 24 roots per completed game, every sixth ply. The stored state contains public cells, visible pool counts and public repetition history; no secret layout is exported or used by the teacher.",
        "",
        "- `small`: first 512 training games, quiet roots, 160 ms teacher, 139 trainable coefficients, update bound 1.",
        "- `data4x`: the same rows plus quiet roots from the other 1,536 training games.",
        "- `coverage`: the same number of roots as `small`, sampled without replacement across three hidden-piece phases as evenly as available; tactical roots and roots with a searched flip are eligible.",
        "- `capacity500` and `capacity2000`: identical roots and objective, with 499 or 1,948 trainable coefficients.",
        "- `bound4x`: identical roots and model, with the normalized update bound increased from 1 to 4.",
        "- `teacher640`: exactly the same roots as `small`, labeled at 640 rather than 160 ms. Both budgets start with cold transposition tables and otherwise identical configuration. More time is not itself a measured teacher Elo gain.",
        "",
        "Every fit starts from the previous `ranking/value.cse`. The initial checkpoint remains eligible. All conditions use the same 640 ms quiet validation targets; whole-distribution 160 ms phase metrics are reported separately and do not select checkpoints. Training uses scalar sigmoid cross entropy, output-drift regularization and bounded Adam updates. Integer forward rounding is exact; backward rounding uses an explicit straight-through surrogate. Neither rules nor search are differentiated.",
        "",
        "This round uses root-value labels, whereas the preceding round used quiet successor pairs. Comparisons within this sweep isolate the declared axes; differences between rounds cannot be attributed solely to architecture. The coverage intervention changes both phase balance and eligibility, so it does not isolate the effect of either alone.",
        "",
        "## Capacity and deployment cost",
        "",
        "The medium model adds piece-value corrections conditioned on eight hidden-count phases and the feasible number of opposing predators. The large model additionally conditions revealed/hidden piece counts on opposing piece types and counts across five phases. Impossible predator counts and the fully-revealed/hidden-piece group have no coefficients. These are composition features; the expansion does not learn new board geometry. The report includes how many extra columns were actually nonzero in the training corpus.",
        "",
        "All added coefficients initialize to zero. Native inference uses fixed-point integer table lookups and sums, with exact export parity. Serialized files include 30 preserved, inactive base coefficients, so file parameter counts exceed trainable counts. File size excludes prepared base tables and the rest of the engine. Evaluation timings use the same 1,881 public positions, seven runs of 100 passes, after collection, fitting and matches have stopped; they are not whole-search speedups.",
        "",
        "## Compute, evidence and reproduction",
        "",
        "No Colab GPU was allocated. The original 5,367-pair fitting pilot took 0.62 seconds for two epochs; an expanded-model smoke fit also completed two epochs in under one second. Four local NumPy fits run concurrently, and native labeling/matches use at most twelve CPU workers. No GPU speedup or GPU cost is claimed.",
        "",
        "See [protocol](experiment-plan.json), [collection provenance](run.json), [training manifest](training-manifest.json), [all fits](training-results.json), [development results](development-results.json), [frozen candidate](frozen-candidate.json), [final results](result.json), [native parity checks](verification-full.json), and [machine-readable summary](summary.json). Collection depends on wall-clock scheduling; training from the retained hashed corpus is reproducible.",
        "",
        "From the repository root, after building the Rust binaries and collecting the documented corpora:",
        "",
        "```sh",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 training/scaling/run.py fit",
        "python3 training/scaling/run.py development",
        "python3 training/scaling/run.py holdout",
        "python3 training/scaling/report.py",
        "```",
        "",
        "Completed jobs are reused only when their recorded configurations and hashes match. Existing final deals have been used and must not serve as fresh holdouts for future model selection. The engine default is unchanged. These few scale points do not establish a power law or an unlimited scaling advantage.",
        "",
    ]
    (FOLDER / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
