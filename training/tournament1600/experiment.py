"""Inputs and integrity checks for the 1,600 ms tournament extension."""

from copy import deepcopy
import json
from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
BASE = ROOT / "training/tournament"
sys.path.insert(0, str(BASE))
import run as base_run
import analyze as base_analysis

digest = base_run.digest
save = base_run.save
timestamp = base_run.timestamp
ARENA = base_run.ARENA


def load_plan():
    return json.loads((FOLDER / "experiment-plan.json").read_text())


def cells(plan):
    configuration = deepcopy(plan)
    configuration["execution"]["expected_batches"] = plan["execution"]["expected_cells"]
    return base_run.configurations(configuration)


def shards(plan):
    count = plan["execution"]["games_per_shard"]
    if count < 4 or count % 2 or plan["games_per_cell"] % count:
        raise ValueError(
            "shards must contain complete paired deals and at least four games"
        )
    output = []
    for cell in cells(plan):
        for index in range(cell["games"] // count):
            output.append(
                {
                    **cell,
                    "name": f"{cell['name']}-s{index}",
                    "cell_name": cell["name"],
                    "shard": index,
                    "first_seed": cell["first_seed"] + index * count // 2,
                    "games": count,
                }
            )
    if len(output) != plan["execution"]["expected_batches"]:
        raise ValueError("unexpected shard count")
    if sum(j["games"] for j in output) != plan["execution"]["expected_games"]:
        raise ValueError("unexpected game count")
    return output


def verify_base(plan):
    previous = base_run.load_plan()
    base_jobs = base_run.configurations(previous)
    base_run.verify_inputs(previous, base_jobs)
    if plan["models"] != previous["models"]:
        raise ValueError("extension models differ from the original tournament")
    for key in ("games_per_cell", "round_robin", "anchor"):
        if plan[key] != previous[key]:
            raise ValueError(f"paired comparison changed: {key}")
    for key in ("tablebases", "tt_bits", "max_depth", "id_stop", "flip_min_depth"):
        if plan["search"][key] != previous["search"][key]:
            raise ValueError(f"search setting changed: {key}")
    for name, expected in plan["base_artifacts"].items():
        if digest(BASE / name) != expected:
            raise ValueError(f"original tournament artifact changed: {name}")
    completed = json.loads((BASE / "completion.json").read_text())
    if not completed["hashes_verified"] or completed["manifest_sha256"] != digest(
        BASE / "run.json"
    ):
        raise ValueError("original tournament completion is inconsistent")
    for job in base_jobs:
        base_run.read_complete(job)
        if completed["csv_sha256"][job["name"]] != digest(BASE / f"{job['name']}.csv"):
            raise ValueError("original tournament CSV changed")
    return previous, base_jobs


def make_manifest(plan, jobs):
    paths = [
        FOLDER / name
        for name in (
            "experiment-plan.json",
            "experiment.py",
            "run_long.py",
            "analyze_long.py",
            "test_extension.py",
        )
    ]
    paths += [
        p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    ]
    paths += [
        *ROOT.glob("engine/src/**/*.rs"),
        *ROOT.glob("engine/tests/*.rs"),
        ROOT / "engine/Cargo.toml",
        ROOT / "engine/Cargo.lock",
    ]
    for model in plan["models"]:
        if "path" in model:
            path = ROOT / model["path"]
            paths.extend([path, path.with_suffix(".json")])
    return {
        "hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))},
        "arena_sha256": digest(ARENA),
        "jobs": jobs,
    }


def verify_inputs(plan, jobs):
    verify_base(plan)
    base_run.validate_models(plan, jobs)
    manifest = make_manifest(plan, jobs)
    record = FOLDER / "run.json"
    if record.exists() and json.loads(record.read_text())["manifest"] != manifest:
        raise ValueError("frozen extension inputs changed")
    return manifest


def read_shard(job):
    path = FOLDER / f"{job['name']}.csv"
    record = json.loads((FOLDER / f"{job['name']}-run.json").read_text())
    if record["config"] != job or record["csv_sha256"] != digest(path):
        raise ValueError(f"completed shard changed: {job['name']}")
    return base_run.verify_csv(path, job)


def read_cell(cell, jobs):
    matching = sorted(
        (j for j in jobs if j["cell_name"] == cell["name"]),
        key=lambda j: j["first_seed"],
    )
    rows, values, next_seed = [], [], cell["first_seed"]
    for job in matching:
        if job["first_seed"] != next_seed or job["block"] != cell["block"]:
            raise ValueError("shard deal order or block differs")
        batch_rows, batch_values = read_shard(job)
        rows.extend(batch_rows)
        values.extend(batch_values)
        next_seed += len(batch_values)
    if len(rows) != cell["games"] or len(values) != cell["games"] // 2:
        raise ValueError(f"incomplete cell: {cell['name']}")
    return rows, values, [f"{j['name']}.csv" for j in matching]
