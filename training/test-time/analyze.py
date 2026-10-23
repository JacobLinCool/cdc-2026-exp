"""Paired, seed-stratified analysis of the frozen test-time experiment."""

import json
from pathlib import Path
import sys

import numpy as np

from run import FOLDER, ROOT, configurations, digest, load_plan, verify_csv

sys.path.insert(0, str(ROOT / "training/scaling"))
from stats import read_games, summarize

REPLICATES = 20_000


def elo(score):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 400 * np.log10(np.asarray(score) / (1 - np.asarray(score)))


def estimate(value, samples):
    return {
        "estimate": float(value),
        "paired_bootstrap_95_ci": list(
            map(float, np.quantile(samples, [0.025, 0.975]))
        ),
    }


def bootstrap_indices(blocks, deals_per_block, seed):
    return np.random.default_rng(seed).integers(
        deals_per_block, size=(blocks, REPLICATES, deals_per_block)
    )


def bootstrap_means(values, indices):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape != (indices.shape[0], indices.shape[2]):
        raise ValueError("bootstrap strata do not match deal blocks")
    return values[np.arange(len(values))[:, None, None], indices].mean(axis=(0, 2))


def cell(jobs, indices):
    paths = [FOLDER / f"{job['name']}.csv" for job in jobs]
    blocks = []
    for job, path in zip(jobs, paths):
        verify_csv(path, job)
        record = json.loads(path.with_name(f"{job['name']}-run.json").read_text())
        assert record["config"] == job and record["csv_sha256"] == digest(path)
        _, deals = read_games([path])
        blocks.append(list(deals.values()))
    values = np.asarray(blocks)
    samples = bootstrap_means(values, indices)
    report = summarize(paths)
    report["paired_bootstrap_95_ci"] = estimate(values.mean(), samples)[
        "paired_bootstrap_95_ci"
    ]
    report["elo"] = estimate(elo(values.mean()), elo(samples))
    report["training_seed_scores"] = {
        str(job["training_seed"]): float(block.mean())
        for job, block in zip(jobs, values)
    }
    report[
        "inference"
    ] = "Deal uncertainty conditional on the three fixed training seeds"
    return report, samples


def contrasts(reports, samples, budgets):
    output = {}
    for low, high in [
        (budgets[0], budgets[1]),
        (budgets[1], budgets[2]),
        (budgets[0], budgets[2]),
    ]:
        a, b = str(low), str(high)
        gain = elo(reports[b]["score"]) - elo(reports[a]["score"])
        boot_gain = elo(samples[b]) - elo(samples[a])
        output[f"{low}->{high}"] = {
            "score_gain": estimate(
                reports[b]["score"] - reports[a]["score"], samples[b] - samples[a]
            ),
            "elo_gain": estimate(gain, boot_gain),
            "elo_gain_per_doubling": estimate(
                gain / np.log2(high / low), boot_gain / np.log2(high / low)
            ),
        }
    return output


def finite_json(value):
    """Explicitly encode unbounded Elo endpoints without nonstandard JSON numbers."""
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            raise ValueError("undefined statistic")
        return "+infinity" if value > 0 else "-infinity"
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def analyze():
    plan = load_plan()
    jobs = configurations(plan)
    frozen = json.loads((FOLDER / "run.json").read_text())["manifest"]
    complete = json.loads((FOLDER / "completion.json").read_text())
    assert complete["hashes_verified"] and complete["games"] == 2592
    assert complete["manifest_sha256"] == digest(FOLDER / "run.json")
    for group in ["sources", "models"]:
        for path, expected in frozen[group].items():
            assert digest(ROOT / path) == expected, f"changed input: {path}"
    for path, expected in frozen["metadata"].items():
        assert digest((ROOT / path).with_suffix(".json")) == expected
    assert digest(ROOT / "engine/target/release/arena") == frozen["arena_sha256"]
    assert frozen["jobs"] == jobs
    result = {
        "games": sum(job["games"] for job in jobs),
        "batches": len(jobs),
        "hashes_verified": True,
        "plan_sha256": digest(FOLDER / "experiment-plan.json"),
        "analysis_sha256": digest(Path(__file__)),
        "bootstrap_replicates": REPLICATES,
        "training_seeds": plan["training_seeds"],
        "anchor_time_ms": plan["anchor"]["time_ms"],
        "time_budgets_ms": plan["time_budgets_ms"],
        "node_budgets": plan["node_budgets"],
        "anchor": {},
        "direct": {},
        "node": {},
    }
    sampled = {family: {} for family in ["anchor", "direct", "node"]}
    for family, bootstrap_seed in [("anchor", 4274), ("direct", 4275), ("node", 4276)]:
        count = plan[f"{family}_games_per_training_seed"] // 2
        indices = bootstrap_indices(len(plan["training_seeds"]), count, bootstrap_seed)
        parameters = (
            [model["parameters"] for model in plan["models"]]
            if family == "anchor"
            else [1948]
        )
        budgets = plan["node_budgets"] if family == "node" else plan["time_budgets_ms"]
        for capacity in parameters:
            reports, samples = {}, {}
            for budget in budgets:
                matching = [
                    job
                    for job in jobs
                    if job["family"] == family
                    and job["parameters"] == capacity
                    and job["budget"] == budget
                ]
                assert [job["training_seed"] for job in matching] == plan[
                    "training_seeds"
                ]
                reports[str(budget)], samples[str(budget)] = cell(matching, indices)
            if family == "anchor":
                result[family][str(capacity)] = {
                    "budgets": reports,
                    "gains": contrasts(reports, samples, budgets),
                }
                sampled[family][str(capacity)] = samples
            else:
                result[family] = {
                    "budgets": reports,
                    "changes": contrasts(reports, samples, budgets),
                }
                sampled[family] = samples
    low, _, high = plan["time_budgets_ms"]
    primary = result["direct"]["changes"][f"{low}->{high}"]["score_gain"]
    result["primary_interaction"] = {
        **primary,
        "definition": f"Large-versus-small direct score at {high} ms minus its score at {low} ms",
        "positive_interaction_supported": primary["paired_bootstrap_95_ci"][0] > 0,
        "negative_interaction_supported": primary["paired_bootstrap_95_ci"][1] < 0,
    }
    result["anchored_capacity_gap"] = {}
    small, large = result["anchor"]["139"], result["anchor"]["1948"]
    small_boot, large_boot = sampled["anchor"]["139"], sampled["anchor"]["1948"]
    for budget in plan["time_budgets_ms"]:
        key = str(budget)
        result["anchored_capacity_gap"][key] = estimate(
            large["budgets"][key]["elo"]["estimate"]
            - small["budgets"][key]["elo"]["estimate"],
            elo(large_boot[key]) - elo(small_boot[key]),
        )
    result["anchored_interaction"] = estimate(
        result["anchored_capacity_gap"][str(high)]["estimate"]
        - result["anchored_capacity_gap"][str(low)]["estimate"],
        elo(large_boot[str(high)])
        - elo(small_boot[str(high)])
        - elo(large_boot[str(low)])
        + elo(small_boot[str(low)]),
    )
    bounds = [
        item["paired_bootstrap_95_ci"] for item in result["direct"]["budgets"].values()
    ]
    result["supported_crossover"] = any(lo > 0.5 for lo, _ in bounds) and any(
        hi < 0.5 for _, hi in bounds
    )
    result["csv_sha256"] = {
        job["name"]: digest(FOLDER / f"{job['name']}.csv") for job in jobs
    }
    result["limitations"] = plan["limitations"]
    (FOLDER / "results.json").write_text(
        json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n"
    )
    write_report(result)
    return result


def number(value):
    return f"{value:+.1f}" if np.isfinite(value) else str(value)


def effect(item, scale=1):
    low, high = item["paired_bootstrap_95_ci"]
    return f"{number(item['estimate'] * scale)} [{number(low * scale)}, {number(high * scale)}]"


def score(item):
    low, high = item["paired_bootstrap_95_ci"]
    return f"{item['score'] * 100:.2f}% [{low * 100:.2f}, {high * 100:.2f}]"


def write_report(result):
    diagnostics = json.loads((FOLDER / "model-diagnostics.json").read_text())
    model_differences = {
        capacity: np.mean(
            [
                item["mean_absolute_score_change_from_same_seed_small"]
                for item in diagnostics["models"].values()
                if item["parameters"] == capacity
            ]
        )
        for capacity in [499, 1948]
    }
    primary = result["primary_interaction"]
    if primary["positive_interaction_supported"]:
        finding = "The larger evaluator's relative advantage grows over the measured 10–160 ms range."
    elif primary["negative_interaction_supported"]:
        finding = "The larger evaluator's relative advantage decreases over the measured 10–160 ms range."
    else:
        finding = "These games do not establish that the larger evaluator benefits more from additional search."
    lines = [
        "# Evaluator capacity × test-time search",
        "",
        finding,
        "",
        f"The preregistered primary contrast, large-versus-small score at 160 ms minus its score at 10 ms, is **{effect(primary, 100)} percentage points** (95% paired interval). This measures a finite-budget interaction, not an asymptotic ceiling.",
        "",
        f"Against the fixed opponent, the 139-parameter model gains {effect(result['anchor']['139']['gains']['10->160']['elo_gain'])} Elo and the 1,948-parameter model gains {effect(result['anchor']['1948']['gains']['10->160']['elo_gain'])}. Their difference is {effect(result['anchored_interaction'])} Elo. The direct and anchored interaction point estimates should be read together with their uncertainty; disagreement between point estimates can reflect sampling variation or opponent-dependent effects.",
        "",
        "All 2,592 games and 45 batches completed with verified input/output hashes. Every capacity uses the same 9,269 training positions and three fixed training seeds (53, 71, 89); each receives equal weight. The nine existing models are frozen. No new model is promoted from this experiment.",
        "",
        "## Capacity and wall-clock curves",
        "",
        "The opponent is always the previous structured evaluator at 40 ms per move. Holding this opponent fixed exposes gains that would disappear if both players received more search time. Each cell contains 192 games: 96 deals with seats reversed, split equally among the three training seeds.",
        "",
        "| Trainable parameters | 10 ms score | 40 ms score | 160 ms score | 10→160 ms Elo gain |",
        "|---:|---:|---:|---:|---:|",
    ]
    for capacity, data in result["anchor"].items():
        lines.append(
            f"| {capacity} | {score(data['budgets']['10'])} | {score(data['budgets']['40'])} | {score(data['budgets']['160'])} | {effect(data['gains']['10->160']['elo_gain'])} |"
        )
    lines.extend(
        [
            "",
            "Elo is a descriptive conversion from score against this particular fixed opponent: `400 log10(score / (1 − score))`. It is not an independently calibrated universal rating. Draws count as half a point.",
            "",
            "| Parameters | Elo gain per doubling, 10→40 ms | Elo gain per doubling, 40→160 ms |",
            "|---:|---:|---:|",
        ]
    )
    for capacity, data in result["anchor"].items():
        lines.append(
            f"| {capacity} | {effect(data['gains']['10->40']['elo_gain_per_doubling'])} | {effect(data['gains']['40->160']['elo_gain_per_doubling'])} |"
        )
    lines.extend(
        [
            "",
            f"The difference in anchored 10→160 ms Elo gain between 1,948 and 139 parameters is {effect(result['anchored_interaction'])}. This secondary contrast preserves the common deals across capacities and budgets.",
            "",
            "## Equal-budget direct matches",
            "",
            "The 1,948-parameter model plays the 139-parameter model with the same training seed. These direct matches use different deals from the fixed-anchor matches. A score above 50% favors the larger model.",
            "",
            "| Budget per player | Games | Large model score, 95% interval | Relative Elo, 95% interval |",
            "|---:|---:|---:|---:|",
        ]
    )
    for budget, item in result["direct"]["budgets"].items():
        lines.append(
            f"| {budget} ms | {item['games']} | {score(item)} | {effect(item['elo'])} |"
        )
    lines.extend(
        [
            "",
            f"A crossover meeting the declared pointwise-interval criterion: **{'yes' if result['supported_crossover'] else 'no'}**. Point estimates crossing 50% alone do not establish a crossover; this secondary comparison has no familywise multiplicity correction.",
            "",
            "| Node cap per player | Games | Large model score, 95% interval | Actual ms/move, large / small |",
            "|---:|---:|---:|---:|",
        ]
    )
    for budget, item in result["node"]["budgets"].items():
        lines.append(
            f"| {int(budget):,} | {item['games']} | {score(item)} | {item['a_ms_per_move']:.2f} / {item['b_ms_per_move']:.2f} |"
        )
    node_change = result["node"]["changes"]["16384->262144"]["score_gain"]
    lines.extend(
        [
            "",
            f"The low-to-high node-budget score change is {effect(node_change, 100)} percentage points. A node is a search visit; equal node caps do not guarantee identical trees, evaluation counts, or FLOPs.",
            "",
            "## Search and timing controls",
            "",
            "The search code and models are unchanged. Both sides explicitly use `fmd=5,tb=0,depth=64,idstop=0.65`, with an 18-bit transposition table. Fixing the flip-search threshold avoids the production default's change at 80 ms. Tablebases are disabled to isolate learned evaluation and search. Six randomized batches run concurrently with two CPU workers each; wall-clock stopping is sensitive to system scheduling. Fixed-node runs override time stopping and have an asserted per-move cap.",
            "",
            "This controlled configuration differs from production defaults. In particular, positions with five or fewer pieces were excluded from the training corpus; disabling tablebases exposes learned evaluation to those endgames. The curves therefore describe this fixed search configuration and should be followed by a separate production-setting confirmation before deployment. Nodes and depth are averages over different played trajectories, not a same-position inference microbenchmark.",
            "",
            "| Parameters | Budget ms | Actual ms/move | Nodes/move | Completed depth |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for capacity, data in result["anchor"].items():
        for budget, item in data["budgets"].items():
            lines.append(
                f"| {capacity} | {budget} | {item['a_ms_per_move']:.2f} | {item['a_nodes_per_move']:.0f} | {item['a_mean_completed_depth']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            "A larger evaluator can reduce leaf-evaluation error and improve move ordering, so deeper search may amplify its advantage. Search can also compensate for a smaller evaluator's weaknesses, or amplify systematic evaluation errors; neither curve ordering nor monotonic gains are guaranteed. The relevant empirical object is strength as a joint function of capacity, training data/compute, search budget, and the search algorithm.",
            "",
            "For the fixed opponent, the capacity–time interaction is `[E_large(160) − E_large(10)] − [E_small(160) − E_small(10)]`. Positive values mean the larger model gains more over this interval; a higher score at every time budget by itself does not establish that interaction. The direct-match primary contrast supplies a separate comparison that avoids relying on transitivity of ratings against the anchor.",
            "",
            "There are also two different optimization questions: measuring capacity effects with a fixed search algorithm, as here, and finding the best achievable strength at each total compute budget after tuning search settings. The latter needs a separate validation procedure for each capacity/budget combination. A plateau under a fixed selective search can be caused by excluded branches or its depth limit, not only model expressivity.",
            "",
            "Training and test-time compute tradeoffs have been measured in AlphaZero/Hex, including architecture-dependent search curves, but that does not establish the same law for this engine ([Jones, 2021, Figures 8–9](https://arxiv.org/abs/2104.03113)). Stockfish also emphasizes that evaluation accuracy must be balanced against search throughput ([NNUE documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)).",
            "",
            result["limitations"],
            "",
            f"The frozen model diagnostics on {diagnostics['positions']:,} existing validation roots show mean absolute score differences of {model_differences[499]:.2f} and {model_differences[1948]:.2f} engine units for the 499- and 1,948-parameter models relative to their same-seed 139-parameter counterparts. The additional tables are active, but this is still a modest expansion around the same learned evaluator. These static-score diagnostics are descriptive and do not select models or supply independent evidence of playing strength; exact values are in [model-diagnostics.json](model-diagnostics.json).",
            "",
            "Intervals use 20,000 bootstrap replicates of entire two-game deals, stratified within each fixed training seed and matched across budgets/capacities. They quantify deal uncertainty conditional on these nine trained models, not variation across all possible training runs. Only the declared direct-match endpoint is primary; all other intervals are exploratory and pointwise. None of the tests reuse training or validation deals. No outcome-dependent stopping or model selection was used.",
            "",
            "The endgame solver limit, if exact exhaustive solving were feasible, would be shared by models using the same rules. The finite-budget plateau of this selective search is a separate empirical question. Three time points cannot identify it. The next expansion should be a fresh, declared wider-budget study with sufficient games, followed by stronger training targets or search-aligned objectives if capacity gains remain absent.",
            "",
            "## Reproduction",
            "",
            "```sh",
            "python3 training/test-time/run.py --verify-only",
            "python3 training/test-time/run.py",
            "python3 training/test-time/diagnostics.py",
            "python3 training/test-time/analyze.py",
            "MPLCONFIGDIR=/tmp/cdc-scaling-mpl PYENV_VERSION=3.11.1 python training/test-time/plot.py",
            "```",
            "",
            "The queue rejects changed frozen inputs and validates completed CSV hashes before resuming. [The protocol](experiment-plan.json), [frozen manifest](run.json), [machine-readable results](results.json), per-batch CSVs, and per-batch run records are included. NumPy is needed for analysis and Matplotlib for the standalone figures. The arena binary is the same verified binary used in the preceding experiment. No GPU was allocated: these matches use the existing native CPU search.",
            "",
            "![Capacity and test-time scaling](test-time-scaling.png)",
        ]
    )
    (FOLDER / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    result = analyze()
    print(
        json.dumps(
            {
                "games": result["games"],
                "primary_interaction": result["primary_interaction"],
                "anchored_interaction": result["anchored_interaction"],
                "supported_crossover": result["supported_crossover"],
            },
            indent=2,
        )
    )
