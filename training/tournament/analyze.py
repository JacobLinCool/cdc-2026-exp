"""Joint deal bootstrap for the frozen tournament and fixed-opponent series."""

import json

import numpy as np

from run import (
    FOLDER,
    configurations,
    digest,
    load_plan,
    read_complete,
    save,
    verify_inputs,
)


def estimate(point, samples):
    return {
        "estimate": float(point),
        "paired_bootstrap_95_ci": list(
            map(float, np.quantile(samples, [0.025, 0.975]))
        ),
    }


def draw_indices(blocks, deals, replicates, seed):
    return np.random.default_rng(seed).integers(deals, size=(blocks, replicates, deals))


def sample_means(values, indices):
    values = np.asarray(values, dtype=float)
    if values.shape != (indices.shape[1],):
        raise ValueError("deal block dimensions differ")
    return values[indices].mean(axis=1)


def league_scores(pair_scores, orientations, model_ids):
    """Map side-A pair scores (possibly bootstrap arrays) to a zero-sum league."""
    scores = np.asarray(pair_scores, dtype=float)
    if scores.shape[0] != len(orientations):
        raise ValueError("pair dimensions differ")
    result = np.zeros((len(model_ids), *scores.shape[1:]))
    opponents = {model: set() for model in model_ids}
    index = {model: i for i, model in enumerate(model_ids)}
    for values, (a, b) in zip(scores, orientations):
        if a == b or b in opponents[a]:
            raise ValueError("duplicated or invalid league pair")
        opponents[a].add(b)
        opponents[b].add(a)
        result[index[a]] += values
        result[index[b]] += 1 - values
    if any(len(value) != len(model_ids) - 1 for value in opponents.values()):
        raise ValueError("incomplete round robin")
    return result / (len(model_ids) - 1)


def row_summary(rows, side="a"):
    scores = np.array([float(row["score"]) for row in rows])
    if side == "b":
        scores = 1 - scores
    moves = sum(int(row[f"{side}_moves"]) for row in rows)
    milliseconds = sum(float(row[f"{side}_ms"]) for row in rows)
    nodes = sum(int(row[f"{side}_nodes"]) for row in rows)
    return {
        "games": len(rows),
        "wins": int((scores == 1).sum()),
        "draws": int((scores == 0.5).sum()),
        "losses": int((scores == 0).sum()),
        "score": float(scores.mean()),
        "win_rate": float((scores == 1).mean()),
        "moves": moves,
        "milliseconds": milliseconds,
        "nodes": nodes,
        "depth_sum": sum(int(row[f"{side}_depth"]) for row in rows),
        "ms_per_move": milliseconds / moves,
        "nodes_per_move": nodes / moves,
        "nodes_per_second": nodes * 1000 / milliseconds,
        "mean_completed_depth": sum(int(row[f"{side}_depth"]) for row in rows) / moves,
    }


def pooled_summary(items):
    """Pool one model's disjoint pair blocks with either arena orientation."""
    totals = {
        key: sum(item[key] for item in items)
        for key in (
            "games",
            "wins",
            "draws",
            "losses",
            "moves",
            "milliseconds",
            "nodes",
            "depth_sum",
        )
    }
    return {
        **totals,
        "score": (totals["wins"] + 0.5 * totals["draws"]) / totals["games"],
        "win_rate": totals["wins"] / totals["games"],
        "ms_per_move": totals["milliseconds"] / totals["moves"],
        "nodes_per_move": totals["nodes"] / totals["moves"],
        "nodes_per_second": 1000 * totals["nodes"] / totals["milliseconds"],
        "mean_completed_depth": totals["depth_sum"] / totals["moves"],
    }


def changes(points, samples, contrasts):
    return {
        f"{low}->{high}": estimate(
            points[str(high)] - points[str(low)], samples[str(high)] - samples[str(low)]
        )
        for low, high in contrasts
    }


def analyze():
    plan = load_plan()
    jobs = configurations(plan)
    verify_inputs(plan, jobs)
    completion = json.loads((FOLDER / "completion.json").read_text())
    if (
        not completion["hashes_verified"]
        or completion["games"] != plan["execution"]["expected_games"]
        or completion["manifest_sha256"] != digest(FOLDER / "run.json")
    ):
        raise ValueError("missing or inconsistent completion record")
    ids = [m["id"] for m in plan["models"]]
    budgets = plan["time_budgets_ms"]
    pairs = sorted(
        [j for j in jobs if j["family"] == "rr" and j["time_ms"] == budgets[0]],
        key=lambda j: j["block"],
    )
    orientations = [(j["a_id"], j["b_id"]) for j in pairs]
    indices = draw_indices(
        len(pairs) + 1,
        plan["games_per_cell"] // 2,
        plan["analysis"]["bootstrap_replicates"],
        plan["analysis"]["bootstrap_seed"],
    )
    cells, cell_samples = {}, {}
    for job in jobs:
        name = job["name"]
        if completion["csv_sha256"][name] != digest(FOLDER / f"{name}.csv"):
            raise ValueError(f"completion hash differs: {name}")
        rows, values = read_complete(job)
        samples = sample_means(values, indices[job["block"]])
        cells[name] = {
            "config": job,
            "a": row_summary(rows),
            "b": row_summary(rows, "b"),
            "score": estimate(np.mean(values), samples),
            "deal_scores": values,
        }
        cell_samples[name] = samples
    league, league_samples = {}, {}
    for budget in budgets:
        matching = sorted(
            [j for j in jobs if j["family"] == "rr" and j["time_ms"] == budget],
            key=lambda j: j["block"],
        )
        points = league_scores(
            [cells[j["name"]]["a"]["score"] for j in matching], orientations, ids
        )
        samples = league_scores(
            [cell_samples[j["name"]] for j in matching], orientations, ids
        )
        records = {}
        for i, model in enumerate(ids):
            records[model] = pooled_summary(
                [
                    cells[j["name"]]["a" if j["a_id"] == model else "b"]
                    for j in matching
                    if model in (j["a_id"], j["b_id"])
                ]
            )
            if not np.isclose(records[model]["score"], points[i]):
                raise ValueError("pooled and equal-weight league scores differ")
            records[model]["score_interval"] = estimate(points[i], samples[i])
        league[str(budget)] = {
            "ranking": sorted(ids, key=lambda m: records[m]["score"], reverse=True),
            "models": records,
        }
        league_samples[str(budget)] = samples
    contrasts = plan["analysis"]["contrasts"]
    league_changes = {
        model: changes(
            {str(t): league[str(t)]["models"][model]["score"] for t in budgets},
            {str(t): league_samples[str(t)][i] for t in budgets},
            contrasts,
        )
        for i, model in enumerate(ids)
    }
    league_differences = {}
    direct_changes = {}
    for pair in pairs:
        a, b = pair["a_id"], pair["b_id"]
        key = f"{a}-vs-{b}"
        ai, bi = ids.index(a), ids.index(b)
        league_differences[key] = {
            str(t): estimate(
                league[str(t)]["models"][a]["score"]
                - league[str(t)]["models"][b]["score"],
                league_samples[str(t)][ai] - league_samples[str(t)][bi],
            )
            for t in budgets
        }
        direct_changes[key] = changes(
            {str(t): cells[f"rr-{key}-t{t}"]["a"]["score"] for t in budgets},
            {str(t): cell_samples[f"rr-{key}-t{t}"] for t in budgets},
            contrasts,
        )
    anchor = {}
    for model in ids:
        reports, samples = {}, {}
        for t in budgets:
            family = (
                "rr"
                if model != "original" and t == plan["anchor"]["opponent_time_ms"]
                else "anchor"
            )
            name = f"{family}-{model}-vs-original-t{t}"
            reports[str(t)] = {
                "source_cell": name,
                **cells[name]["a"],
                "score_interval": cells[name]["score"],
            }
            samples[str(t)] = cell_samples[name]
        anchor[model] = {
            "budgets": reports,
            "gains": changes(
                {t: item["score"] for t, item in reports.items()}, samples, contrasts
            ),
        }
    result = {
        "games": completion["games"],
        "batches": completion["batches"],
        "models": plan["models"],
        "time_budgets_ms": budgets,
        "bootstrap_replicates": plan["analysis"]["bootstrap_replicates"],
        "numpy_version": np.__version__,
        "plan_sha256": digest(FOLDER / "experiment-plan.json"),
        "completion_sha256": digest(FOLDER / "completion.json"),
        "hashes_verified": True,
        "league": league,
        "league_changes": league_changes,
        "league_differences": league_differences,
        "direct_changes": direct_changes,
        "anchor": anchor,
        "cells": cells,
        "limitations": plan["limitations"],
    }
    save(FOLDER / "results.json", result)
    print(
        json.dumps(
            {
                "games": result["games"],
                "rankings": {t: r["ranking"] for t, r in league.items()},
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    analyze()
