"""Frozen, resumable five-version tournament using the existing native arena."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
ARENA = ROOT / "engine/target/release/arena"
CSV_FIELDS = {
    "seed",
    "a_first",
    "score",
    "plies",
    "a_ms",
    "b_ms",
    "a_moves",
    "b_moves",
    "a_nodes",
    "b_nodes",
    "a_depth",
    "b_depth",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_plan():
    return json.loads((FOLDER / "experiment-plan.json").read_text())


def configurations(plan):
    models = plan["models"]
    settings = plan["search"]
    if settings["tt_bits"] != 18:
        raise ValueError("arena fixes the transposition table at 18 bits")
    common = [
        f"fmd={settings['flip_min_depth']}",
        f"tb={int(settings['tablebases'])}",
        f"depth={settings['max_depth']}",
        f"idstop={settings['id_stop']}",
    ]

    def spec(model, budget, arena_budget):
        fields = list(common)
        if "path" in model:
            fields.append(f"{model['spec_key']}={model['path']}")
        if "model_scale" in model:
            fields.append(f"model_scale={model['model_scale']}")
        if budget != arena_budget:
            fields.append(f"tmul={budget / arena_budget}")
        return ",".join(fields)

    jobs = []

    def append(family, a, b, budget, opponent_budget, block, first_seed):
        name = f"{family}-{a['id']}-vs-{b['id']}-t{budget}"
        jobs.append(
            {
                "name": name,
                "family": family,
                "a_id": a["id"],
                "b_id": b["id"],
                "time_ms": budget,
                "b_time_ms": opponent_budget,
                "block": block,
                "first_seed": first_seed,
                "games": plan["games_per_cell"],
                "threads": plan["execution"]["threads_per_batch"],
                "a": spec(a, budget, budget),
                "b": spec(b, opponent_budget, budget),
            }
        )

    deals = plan["games_per_cell"] // 2
    anchor = plan["anchor"]
    for block, (a, b) in enumerate(combinations(models, 2)):
        if a["id"] == anchor["opponent"]:
            a, b = b, a
        first_seed = plan["round_robin"]["first_seed"] + block * deals
        for budget in plan["time_budgets_ms"]:
            append("rr", a, b, budget, budget, block, first_seed)
            if b["id"] == anchor["opponent"] and budget != anchor["opponent_time_ms"]:
                append(
                    "anchor",
                    a,
                    b,
                    budget,
                    anchor["opponent_time_ms"],
                    block,
                    first_seed,
                )
    original = next(m for m in models if m["id"] == anchor["opponent"])
    for budget in plan["time_budgets_ms"]:
        append(
            "anchor",
            original,
            original,
            budget,
            anchor["opponent_time_ms"],
            math.comb(len(models), 2),
            anchor["self_control_first_seed"],
        )
    if len(jobs) != plan["execution"]["expected_batches"]:
        raise ValueError("unexpected batch count")
    if sum(j["games"] for j in jobs) != plan["execution"]["expected_games"]:
        raise ValueError("unexpected game count")
    return jobs


def validate_models(plan, jobs):
    metadata = []
    for model in plan["models"]:
        if "path" not in model:
            continue
        path = ROOT / model["path"]
        item = json.loads(path.with_suffix(".json").read_text())
        if digest(path) != model["sha256"] or item["model_sha256"] != model["sha256"]:
            raise ValueError(f"model hash differs: {path}")
        count = (
            item["parameters"]
            if model["spec_key"] == "model"
            else item["trainable_parameters"]
        )
        if count != model["trainable_parameters"]:
            raise ValueError(f"parameter count differs: {path}")
        metadata.append(item)
    for job in jobs:
        seeds = set(range(job["first_seed"], job["first_seed"] + job["games"] // 2))
        for item in metadata:
            if seeds.intersection(item["training_games"] + item["validation_games"]):
                raise ValueError(f"training/validation deal leakage: {job['name']}")


def make_manifest(plan, jobs):
    paths = [
        FOLDER / "experiment-plan.json",
        *[FOLDER / name for name in ("run.py", "analyze.py", "test_tournament.py")],
        *ROOT.glob("engine/src/**/*.rs"),
        *ROOT.glob("engine/tests/*.rs"),
        ROOT / "engine/Cargo.toml",
        ROOT / "engine/Cargo.lock",
    ]
    for model in plan["models"]:
        if "path" in model:
            paths.extend(
                [ROOT / model["path"], (ROOT / model["path"]).with_suffix(".json")]
            )
    return {
        "hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))},
        "arena_sha256": digest(ARENA),
        "jobs": jobs,
    }


def verify_csv(path, config):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != CSV_FIELDS:
            raise ValueError(f"wrong CSV schema: {path}")
        rows = list(reader)
    if len(rows) != config["games"]:
        raise ValueError(f"wrong game count: {path}")
    pairs = {}
    for row in rows:
        if float(row["score"]) not in (0.0, 0.5, 1.0):
            raise ValueError(f"invalid outcome: {path}")
        pairs.setdefault(int(row["seed"]), []).append(row)
        if int(row["a_moves"]) + int(row["b_moves"]) != int(row["plies"]):
            raise ValueError(f"move count differs: {path}")
        for side in ("a", "b"):
            elapsed = float(row[f"{side}_ms"])
            if (
                int(row[f"{side}_moves"]) <= 0
                or int(row[f"{side}_nodes"]) < 0
                or int(row[f"{side}_depth"]) < 0
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError(f"invalid timing/search record: {path}")
    expected = set(
        range(config["first_seed"], config["first_seed"] + config["games"] // 2)
    )
    if set(pairs) != expected or any(
        len(pair) != 2 or {r["a_first"] for r in pair} != {"false", "true"}
        for pair in pairs.values()
    ):
        raise ValueError(f"incomplete/duplicated deal or seat: {path}")
    return rows, [sum(float(r["score"]) for r in pairs[s]) / 2 for s in sorted(pairs)]


def read_complete(job):
    path = FOLDER / f"{job['name']}.csv"
    record = json.loads((FOLDER / f"{job['name']}-run.json").read_text())
    if record["config"] != job or record["csv_sha256"] != digest(path):
        raise ValueError(f"completed match changed: {job['name']}")
    return verify_csv(path, job)


def match(job):
    name = job["name"]
    output = FOLDER / f"{name}.csv"
    record = FOLDER / f"{name}-run.json"
    previous = None
    if record.exists():
        previous = json.loads(record.read_text())
        if previous["config"] != job:
            raise ValueError(f"configuration changed: {name}")
        if "csv_sha256" in previous:
            read_complete(job)
            return
    elif output.exists():
        raise ValueError(f"unregistered output: {output}")
    if output.exists():
        verify_csv(output, job)
        save(
            record,
            {**previous, "csv_sha256": digest(output), "recovered_utc": timestamp()},
        )
        return
    started = time.monotonic()
    status = {"config": job, "started_utc": timestamp()}
    save(record, status)
    print(f"START {name}", flush=True)
    with (FOLDER / f"{name}.log").open("w") as stream:
        subprocess.run(
            [
                str(ARENA),
                str(job["games"]),
                str(job["time_ms"]),
                job["a"],
                job["b"],
                str(job["threads"]),
            ],
            cwd=ROOT,
            env=dict(os.environ, SEED=str(job["first_seed"]), ARENA_CSV=str(output)),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    verify_csv(output, job)
    save(
        record,
        {
            **status,
            "csv_sha256": digest(output),
            "completed_utc": timestamp(),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def verify_inputs(plan, jobs):
    validate_models(plan, jobs)
    manifest = make_manifest(plan, jobs)
    record = FOLDER / "run.json"
    if record.exists() and json.loads(record.read_text())["manifest"] != manifest:
        raise ValueError("frozen inputs changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    plan = load_plan()
    jobs = configurations(plan)
    manifest = verify_inputs(plan, jobs)
    record = FOLDER / "run.json"
    print(
        f"Verified {len(jobs)} batches / {sum(j['games'] for j in jobs)} games",
        flush=True,
    )
    if args.verify_only:
        return
    order = list(jobs)
    random.Random(plan["execution"]["shuffle_seed"]).shuffle(order)
    if not record.exists():
        save(
            record,
            {
                "started_utc": timestamp(),
                "manifest": manifest,
                "schedule": [j["name"] for j in order],
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "logical_cpus": os.cpu_count(),
                },
            },
        )
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=plan["execution"]["parallel_batches"]) as pool:
        pending = {pool.submit(match, j): j for j in order}
        for index, future in enumerate(as_completed(pending), 1):
            future.result()
            print(f"DONE {index}/{len(jobs)} {pending[future]['name']}", flush=True)
    if verify_inputs(plan, jobs) != manifest:
        raise ValueError("inputs changed during evaluation")
    for job in jobs:
        read_complete(job)
    save(
        FOLDER / "completion.json",
        {
            "completed_utc": timestamp(),
            "wall_seconds_this_invocation": time.monotonic() - started,
            "games": sum(j["games"] for j in jobs),
            "batches": len(jobs),
            "manifest_sha256": digest(record),
            "hashes_verified": True,
            "csv_sha256": {
                j["name"]: digest(FOLDER / f"{j['name']}.csv") for j in jobs
            },
        },
    )


if __name__ == "__main__":
    main()
