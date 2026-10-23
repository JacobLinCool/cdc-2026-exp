"""Extend the frozen joint bootstrap to 1,600 ms without changing earlier results."""

from itertools import combinations
import json

import numpy as np

from experiment import (
    BASE,
    FOLDER,
    base_analysis as stats,
    cells,
    digest,
    load_plan,
    read_cell,
    save,
    shards,
    verify_inputs,
)


def summarize_cells(all_cells, samples, ids, budgets):
    league, boot = {}, {}
    for budget in budgets:
        matching = sorted(
            (
                c
                for c in all_cells.values()
                if c["config"]["family"] == "rr" and c["config"]["time_ms"] == budget
            ),
            key=lambda c: c["config"]["block"],
        )
        orientations = [(c["config"]["a_id"], c["config"]["b_id"]) for c in matching]
        point = stats.league_scores(
            [c["a"]["score"] for c in matching], orientations, ids
        )
        draws = stats.league_scores(
            [samples[c["config"]["name"]] for c in matching], orientations, ids
        )
        records = {}
        for i, model in enumerate(ids):
            records[model] = stats.pooled_summary(
                [
                    c["a" if c["config"]["a_id"] == model else "b"]
                    for c in matching
                    if model in (c["config"]["a_id"], c["config"]["b_id"])
                ]
            )
            if records[model]["score"] != point[i]:
                raise ValueError("league aggregation differs")
            records[model]["score_interval"] = stats.estimate(point[i], draws[i])
        league[str(budget)] = {
            "ranking": sorted(ids, key=lambda m: records[m]["score"], reverse=True),
            "models": records,
        }
        boot[str(budget)] = draws
    return league, boot


def gain_difference(a_points, b_points, a_samples, b_samples, low, high):
    """Difference of time gains; preserves within-model matched-deal covariance."""
    lo, hi = str(low), str(high)
    return stats.estimate(
        (a_points[hi] - a_points[lo]) - (b_points[hi] - b_points[lo]),
        (a_samples[hi] - a_samples[lo]) - (b_samples[hi] - b_samples[lo]),
    )


def analyze():
    plan = load_plan()
    jobs = shards(plan)
    verify_inputs(plan, jobs)
    completed = json.loads((FOLDER / "completion.json").read_text())
    if (
        not completed["hashes_verified"]
        or completed["games"] != plan["execution"]["expected_games"]
        or completed["manifest_sha256"] != digest(FOLDER / "run.json")
    ):
        raise ValueError("extension completion record differs")
    for job in jobs:
        if completed["csv_sha256"][job["name"]] != digest(
            FOLDER / f"{job['name']}.csv"
        ):
            raise ValueError(f"completion CSV hash differs: {job['name']}")
    prior = json.loads((BASE / "results.json").read_text())
    all_cells = prior["cells"]
    indices = stats.draw_indices(
        11,
        plan["games_per_cell"] // 2,
        plan["analysis"]["bootstrap_replicates"],
        plan["analysis"]["bootstrap_seed"],
    )
    samples = {
        name: stats.sample_means(c["deal_scores"], indices[c["config"]["block"]])
        for name, c in all_cells.items()
    }
    for cell in cells(plan):
        rows, values, sources = read_cell(cell, jobs)
        name = cell["name"]
        if name in all_cells:
            raise ValueError("extension overlaps an existing cell")
        draws = stats.sample_means(values, indices[cell["block"]])
        all_cells[name] = {
            "config": cell,
            "a": stats.row_summary(rows),
            "b": stats.row_summary(rows, "b"),
            "score": stats.estimate(np.mean(values), draws),
            "deal_scores": values,
            "csv_shards": sources,
        }
        samples[name] = draws
    ids = [m["id"] for m in plan["models"]]
    budgets = plan["analysis_budgets_ms"]
    contrasts = plan["analysis"]["contrasts"]
    league, league_samples = summarize_cells(all_cells, samples, ids, budgets)
    for budget in prior["time_budgets_ms"]:
        if league[str(budget)] != prior["league"][str(budget)]:
            raise ValueError("earlier league results or confidence intervals changed")
    league_changes = {
        model: stats.changes(
            {str(t): league[str(t)]["models"][model]["score"] for t in budgets},
            {str(t): league_samples[str(t)][i] for t in budgets},
            contrasts,
        )
        for i, model in enumerate(ids)
    }
    pairs = sorted(
        (
            c["config"]
            for c in all_cells.values()
            if c["config"]["family"] == "rr" and c["config"]["time_ms"] == budgets[0]
        ),
        key=lambda c: c["block"],
    )
    league_differences, direct_changes = {}, {}
    for pair in pairs:
        a, b = pair["a_id"], pair["b_id"]
        key = f"{a}-vs-{b}"
        ai, bi = ids.index(a), ids.index(b)
        league_differences[key] = {
            str(t): stats.estimate(
                league[str(t)]["models"][a]["score"]
                - league[str(t)]["models"][b]["score"],
                league_samples[str(t)][ai] - league_samples[str(t)][bi],
            )
            for t in budgets
        }
        direct_changes[key] = stats.changes(
            {str(t): all_cells[f"rr-{key}-t{t}"]["a"]["score"] for t in budgets},
            {str(t): samples[f"rr-{key}-t{t}"] for t in budgets},
            contrasts,
        )
    anchor, anchor_samples = {}, {}
    for model in ids:
        reports, draws = {}, {}
        for t in budgets:
            family = (
                "rr"
                if model != "original" and t == plan["anchor"]["opponent_time_ms"]
                else "anchor"
            )
            name = f"{family}-{model}-vs-original-t{t}"
            cell = all_cells[name]
            reports[str(t)] = {
                "source_cell": name,
                **cell["a"],
                "score_interval": cell["score"],
            }
            draws[str(t)] = samples[name]
        anchor[model] = {
            "budgets": reports,
            "gains": stats.changes(
                {t: c["score"] for t, c in reports.items()}, draws, contrasts
            ),
        }
        anchor_samples[model] = draws
        for t in prior["time_budgets_ms"]:
            if reports[str(t)] != prior["anchor"][model]["budgets"][str(t)]:
                raise ValueError("earlier anchor results changed")
    interactions = {}
    for a, b in combinations(ids, 2):
        interactions[f"{a}-minus-{b}"] = {
            f"{lo}->{hi}": gain_difference(
                {t: c["score"] for t, c in anchor[a]["budgets"].items()},
                {t: c["score"] for t, c in anchor[b]["budgets"].items()},
                anchor_samples[a],
                anchor_samples[b],
                lo,
                hi,
            )
            for lo, hi in plan["analysis"]["new_contrasts"]
        }
    result = {
        "games": prior["games"] + completed["games"],
        "new_games": completed["games"],
        "cells": all_cells,
        "new_cells": completed["cells"],
        "new_shards": completed["batches"],
        "models": plan["models"],
        "time_budgets_ms": budgets,
        "numpy_version": np.__version__,
        "bootstrap_replicates": plan["analysis"]["bootstrap_replicates"],
        "plan_sha256": digest(FOLDER / "experiment-plan.json"),
        "completion_sha256": digest(FOLDER / "completion.json"),
        "base_results_sha256": digest(BASE / "results.json"),
        "hashes_verified": True,
        "prior_results_reproduced_exactly": True,
        "league": league,
        "league_changes": league_changes,
        "league_differences": league_differences,
        "direct_changes": direct_changes,
        "anchor": anchor,
        "anchor_gain_differences": interactions,
        "limitations": plan["limitations"],
    }
    save(FOLDER / "results.json", result)
    print(
        json.dumps(
            {
                "games": result["games"],
                "new_games": result["new_games"],
                "rankings": {t: v["ranking"] for t, v in league.items()},
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    analyze()
