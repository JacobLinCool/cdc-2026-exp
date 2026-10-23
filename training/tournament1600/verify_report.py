"""Independently reconcile published counts and point estimates with raw CSVs."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
BASE = ROOT / "training/tournament"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def compare(actual, expected, description):
    if isinstance(actual, int) and isinstance(expected, int):
        require(actual == expected, description)
    else:
        require(isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), description)


def read_json(path):
    return json.loads(path.read_text())


def raw_summary(rows, side):
    values = [float(row["score"]) for row in rows]
    if side == "b":
        values = [1 - value for value in values]
    totals = {
        "games": len(rows),
        "wins": values.count(1),
        "draws": values.count(0.5),
        "losses": values.count(0),
        "moves": sum(int(r[f"{side}_moves"]) for r in rows),
        "milliseconds": sum(float(r[f"{side}_ms"]) for r in rows),
        "nodes": sum(int(r[f"{side}_nodes"]) for r in rows),
        "depth_sum": sum(int(r[f"{side}_depth"]) for r in rows),
    }
    return {
        **totals,
        "score": sum(values) / len(values),
        "win_rate": values.count(1) / len(values),
        "ms_per_move": totals["milliseconds"] / totals["moves"],
        "nodes_per_move": totals["nodes"] / totals["moves"],
        "nodes_per_second": 1000 * totals["nodes"] / totals["milliseconds"],
        "mean_completed_depth": totals["depth_sum"] / totals["moves"],
    }


def main():
    result = read_json(FOLDER / "results.json")
    prior = read_json(BASE / "results.json")
    plan = read_json(FOLDER / "experiment-plan.json")
    run = read_json(FOLDER / "run.json")
    completion = read_json(FOLDER / "completion.json")
    require(result["games"] == 1792 and result["new_games"] == 480, "game totals")
    require(result["time_budgets_ms"] == [25, 100, 400, 1600], "budget sequence")
    require(
        completion["manifest_sha256"] == digest(FOLDER / "run.json"), "manifest hash"
    )
    require(
        result["completion_sha256"] == digest(FOLDER / "completion.json"),
        "completion hash",
    )
    require(
        result["plan_sha256"] == digest(FOLDER / "experiment-plan.json"), "plan hash"
    )
    require(
        result["base_results_sha256"] == digest(BASE / "results.json"),
        "base results hash",
    )
    for path, expected in run["manifest"]["hashes"].items():
        require(digest(ROOT / path) == expected, f"frozen input changed: {path}")
    require(
        digest(ROOT / "engine/target/release/arena") == run["manifest"]["arena_sha256"],
        "arena hash",
    )
    for path, expected in plan["base_artifacts"].items():
        require(digest(BASE / path) == expected, f"base artifact changed: {path}")
    require(len(result["cells"]) == 56, "expected 56 logical cells")
    raw = {}
    sources = set()
    total_games = 0
    new_games = 0
    deal_blocks = {}
    for name, cell in result["cells"].items():
        config = cell["config"]
        if config["time_ms"] == 1600:
            filenames = [FOLDER / item for item in cell["csv_shards"]]
            require(len(filenames) == 8, f"eight shards per cell: {name}")
        else:
            filenames = [BASE / f"{name}.csv"]
            require(cell == prior["cells"][name], f"earlier cell unchanged: {name}")
        rows = []
        for path in filenames:
            require(path not in sources, f"physical CSV counted twice: {path}")
            sources.add(path)
            record = read_json(path.with_name(f"{path.stem}-run.json"))
            require(record["csv_sha256"] == digest(path), f"raw CSV hash: {path}")
            if path.parent == FOLDER:
                require(
                    completion["csv_sha256"][path.stem] == digest(path),
                    f"completion CSV hash: {path}",
                )
            with path.open() as stream:
                batch = list(csv.DictReader(stream))
            require(len(batch) == record["config"]["games"], f"shard row count: {path}")
            rows.extend(batch)
        require(len(rows) == 32, f"cell row count: {name}")
        total_games += len(rows)
        if config["time_ms"] == 1600:
            new_games += len(rows)
        seeds = sorted(set(int(row["seed"]) for row in rows))
        require(
            seeds == list(range(config["first_seed"], config["first_seed"] + 16)),
            f"seed range: {name}",
        )
        block = config["block"]
        if block in deal_blocks:
            require(deal_blocks[block] == seeds, f"same deals across time: {name}")
        else:
            deal_blocks[block] = seeds
        deal_values = []
        for seed in seeds:
            pair = [row for row in rows if int(row["seed"]) == seed]
            require(
                len(pair) == 2 and {r["a_first"] for r in pair} == {"true", "false"},
                f"paired seats: {name}/{seed}",
            )
            values = [float(r["score"]) for r in pair]
            require(
                all(value in (0, 0.5, 1) for value in values),
                f"legal scores: {name}/{seed}",
            )
            deal_values.append(sum(values) / 2)
        require(deal_values == cell["deal_scores"], f"deal scores: {name}")
        raw[name] = {side: raw_summary(rows, side) for side in ("a", "b")}
        for side in ("a", "b"):
            for key, expected in raw[name][side].items():
                compare(
                    cell[side][key], expected, f"cell statistic: {name}/{side}/{key}"
                )
        compare(
            cell["score"]["estimate"],
            raw[name]["a"]["score"],
            f"cell score estimate: {name}",
        )
    require(total_games == 1792 and new_games == 480, "raw game totals")
    require(len(sources) == 161, "raw CSV count")
    require(
        {p for p in sources if p.parent == FOLDER} == set(FOLDER.glob("*.csv")),
        "all new CSVs included",
    )
    require(
        len(deal_blocks) == 11
        and len({s for block in deal_blocks.values() for s in block}) == 176,
        "independent deal blocks",
    )
    ids = [m["id"] for m in plan["models"]]
    budgets = result["time_budgets_ms"]
    for t in budgets:
        league = result["league"][str(t)]["models"]
        compare(
            sum(v["score"] for v in league.values()),
            len(ids) / 2,
            f"zero-sum league: {t}",
        )
        if t != 1600:
            require(
                result["league"][str(t)] == prior["league"][str(t)],
                f"old league unchanged: {t}",
            )
        for model in ids:
            items = []
            for name, cell in result["cells"].items():
                cfg = cell["config"]
                if (
                    cfg["family"] == "rr"
                    and cfg["time_ms"] == t
                    and model in (cfg["a_id"], cfg["b_id"])
                ):
                    items.append(raw[name]["a" if cfg["a_id"] == model else "b"])
            require(len(items) == 4, f"four opponents: {model}/{t}")
            for key in (
                "games",
                "wins",
                "draws",
                "losses",
                "moves",
                "milliseconds",
                "nodes",
                "depth_sum",
            ):
                compare(
                    league[model][key],
                    sum(item[key] for item in items),
                    f"league totals: {model}/{t}/{key}",
                )
            compare(
                league[model]["score"],
                sum(item["score"] for item in items) / 4,
                f"league score: {model}/{t}",
            )
            anchor = result["anchor"][model]["budgets"][str(t)]
            for key, expected in raw[anchor["source_cell"]]["a"].items():
                compare(anchor[key], expected, f"anchor statistic: {model}/{t}/{key}")
            if t != 1600:
                require(
                    anchor == prior["anchor"][model]["budgets"][str(t)],
                    f"old anchor unchanged: {model}/{t}",
                )
    for model in ids:
        for lo, hi in plan["analysis"]["contrasts"]:
            contrast = f"{lo}->{hi}"
            expected = (
                result["league"][str(hi)]["models"][model]["score"]
                - result["league"][str(lo)]["models"][model]["score"]
            )
            compare(
                result["league_changes"][model][contrast]["estimate"],
                expected,
                f"league delta: {model}/{contrast}",
            )
            anchor = result["anchor"][model]
            expected = (
                anchor["budgets"][str(hi)]["score"]
                - anchor["budgets"][str(lo)]["score"]
            )
            compare(
                anchor["gains"][contrast]["estimate"],
                expected,
                f"anchor delta: {model}/{contrast}",
            )
    for pair, contrasts in result["anchor_gain_differences"].items():
        a, b = pair.split("-minus-")
        for contrast, item in contrasts.items():
            expected = (
                result["anchor"][a]["gains"][contrast]["estimate"]
                - result["anchor"][b]["gains"][contrast]["estimate"]
            )
            compare(item["estimate"], expected, f"gain interaction: {pair}/{contrast}")
    for family in ("league_changes", "direct_changes", "league_differences"):
        for pair, comparisons in prior[family].items():
            for contrast, estimate in comparisons.items():
                require(
                    result[family][pair][contrast] == estimate,
                    f"earlier contrast and interval unchanged: {family}/{pair}/{contrast}",
                )
    for model in ids:
        for contrast, estimate in prior["anchor"][model]["gains"].items():
            require(
                result["anchor"][model]["gains"][contrast] == estimate,
                f"earlier anchor gain interval unchanged: {model}/{contrast}",
            )
    artifacts = read_json(FOLDER / "report-artifacts.json")
    for filename, expected in artifacts.items():
        require(digest(FOLDER / filename) == expected, f"rendered artifact: {filename}")
    verification = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "raw_csv_files": len(sources),
        "total_games": total_games,
        "new_games": new_games,
        "logical_cells": len(raw),
        "independent_deal_blocks": len(deal_blocks),
        "unique_deals": 176,
        "paired_seeds_and_seating_verified": True,
        "all_cell_wdl_and_search_metrics_recomputed": True,
        "league_and_anchor_point_estimates_recomputed": True,
        "time_gain_point_estimates_recomputed": True,
        "gain_interaction_point_estimates_recomputed": True,
        "earlier_cells_and_intervals_unchanged": True,
        "earlier_derived_intervals_unchanged": True,
        "frozen_inputs_verified": len(run["manifest"]["hashes"]),
        "arena_hash_verified": True,
        "rendered_artifact_hashes_verified": len(artifacts),
        "results_sha256": digest(FOLDER / "results.json"),
        "report_artifacts_sha256": digest(FOLDER / "report-artifacts.json"),
        "verification_script_sha256": digest(Path(__file__)),
    }
    (FOLDER / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
