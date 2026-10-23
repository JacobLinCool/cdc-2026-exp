"""Frozen, resumable capacity × search-budget experiment queue."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
ARENA = ROOT / "engine/target/release/arena"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_plan():
    return json.loads((FOLDER / "experiment-plan.json").read_text())


def configurations(plan):
    jobs = []
    search = plan["search"]
    assert search["tt_bits"] == 18, "arena uses a fixed 18-bit transposition table"
    common = (
        f",fmd={search['flip_min_depth']},tb={int(search['tablebases'])}"
        f",depth={search['max_depth']},idstop={search['id_stop']}"
    )
    anchor = plan["anchor"]
    for index, seed in enumerate(plan["training_seeds"]):
        models = {
            item["parameters"]: f"{item['prefix']}-{seed}.cxe"
            for item in plan["models"]
        }
        for parameters, model in models.items():
            for budget in plan["time_budgets_ms"]:
                games = plan["anchor_games_per_training_seed"]
                jobs.append(
                    {
                        "name": f"anchor-p{parameters}-t{budget}-s{seed}",
                        "family": "anchor",
                        "parameters": parameters,
                        "training_seed": seed,
                        "budget": budget,
                        "time_ms": budget,
                        "games": games,
                        "first_seed": plan["anchor_first_seed"] + index * (games // 2),
                        "models": [model, anchor["model"]],
                        "a": f"expanded={model}{common}",
                        "b": f"structured={anchor['model']}{common},tmul={anchor['time_ms'] / budget}",
                    }
                )
        for family, budgets in [
            ("direct", plan["time_budgets_ms"]),
            ("node", plan["node_budgets"]),
        ]:
            games = plan[f"{family}_games_per_training_seed"]
            for budget in budgets:
                extra = f",nodes={budget}" if family == "node" else ""
                jobs.append(
                    {
                        "name": f"{family}-b{budget}-s{seed}",
                        "family": family,
                        "parameters": 1948,
                        "training_seed": seed,
                        "budget": budget,
                        "time_ms": anchor["time_ms"] if family == "node" else budget,
                        "games": games,
                        "first_seed": plan[f"{family}_first_seed"]
                        + index * (games // 2),
                        "models": [models[1948], models[139]],
                        "a": f"expanded={models[1948]}{common}{extra}",
                        "b": f"expanded={models[139]}{common}{extra}",
                    }
                )
    for job in jobs:
        job["threads"] = plan["execution"]["threads_per_batch"]
    assert len(jobs) == 45 and sum(job["games"] for job in jobs) == 2592
    return jobs


def validate_models(jobs):
    paths = sorted({path for job in jobs for path in job["models"]})
    metadata = {}
    fingerprints = set()
    for name in paths:
        path = ROOT / name
        item = json.loads(path.with_suffix(".json").read_text())
        if digest(path) != item["model_sha256"]:
            raise ValueError(f"model hash differs: {path}")
        metadata[name] = item
        if path.suffix == ".cxe":
            assert item["training_positions"] == 9269 and item["epochs"] == 80
            fingerprints.add(
                tuple(
                    item[key]
                    for key in [
                        "training_data_sha256",
                        "validation_data_sha256",
                        "training_row_ids_sha256",
                        "initial_sha256",
                    ]
                )
            )
    assert (
        len(fingerprints) == 1
    ), "capacity conditions used different data or initialization"
    for job in jobs:
        deals = set(range(job["first_seed"], job["first_seed"] + job["games"] // 2))
        for name in job["models"]:
            item = metadata[name]
            assert not deals.intersection(
                item["training_games"] + item["validation_games"]
            )
        assert metadata[job["models"][0]]["trainable_parameters"] == job["parameters"]
    return paths


def make_manifest(jobs, paths):
    source_files = [
        FOLDER / "experiment-plan.json",
        Path(__file__),
        *ROOT.glob("engine/src/**/*.rs"),
        *ROOT.glob("engine/tests/*.rs"),
        ROOT / "engine/Cargo.toml",
        ROOT / "engine/Cargo.lock",
    ]
    return {
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sorted(source_files)},
        "models": {p: digest(ROOT / p) for p in paths},
        "metadata": {p: digest((ROOT / p).with_suffix(".json")) for p in paths},
        "arena_sha256": digest(ARENA),
        "jobs": jobs,
    }


def verify_csv(path, config):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == config["games"], f"wrong game count: {path}"
    pairs = {}
    for row in rows:
        score = float(row["score"])
        assert score in [0.0, 0.5, 1.0]
        pairs.setdefault(int(row["seed"]), []).append(row["a_first"])
        for side in ["a", "b"]:
            moves = int(row[f"{side}_moves"])
            nodes = int(row[f"{side}_nodes"])
            assert moves > 0 and nodes >= 0 and float(row[f"{side}_ms"]) >= 0
            if config["family"] == "node":
                assert nodes <= moves * config["budget"]
    assert set(pairs) == set(
        range(config["first_seed"], config["first_seed"] + config["games"] // 2)
    )
    assert all(sorted(pair) == ["false", "true"] for pair in pairs.values())


def match(config):
    name = config["name"]
    output = FOLDER / f"{name}.csv"
    record = FOLDER / f"{name}-run.json"
    if record.exists():
        previous = json.loads(record.read_text())
        assert previous["config"] == config
        if "csv_sha256" in previous:
            assert output.exists() and digest(output) == previous["csv_sha256"]
            verify_csv(output, config)
            return
    elif output.exists():
        raise ValueError(f"unregistered output exists: {output}")
    if output.exists():
        # A completed arena output can survive an interrupted harness; validate it.
        verify_csv(output, config)
        save(record, {"config": config, "csv_sha256": digest(output)})
        return
    save(
        record,
        {"config": config, "started_utc": datetime.now(timezone.utc).isoformat()},
    )
    started = time.monotonic()
    env = dict(os.environ, SEED=str(config["first_seed"]), ARENA_CSV=str(output))
    with (FOLDER / f"{name}.log").open("w") as stream:
        subprocess.run(
            [
                str(ARENA),
                str(config["games"]),
                str(config["time_ms"]),
                config["a"],
                config["b"],
                str(config["threads"]),
            ],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    verify_csv(output, config)
    save(
        record,
        {
            "config": config,
            "csv_sha256": digest(output),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    plan = load_plan()
    jobs = configurations(plan)
    paths = validate_models(jobs)
    manifest = make_manifest(jobs, paths)
    record = FOLDER / "run.json"
    if record.exists():
        assert (
            json.loads(record.read_text())["manifest"] == manifest
        ), "frozen inputs changed"
    elif not args.verify_only:
        save(
            record,
            {
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "manifest": manifest,
            },
        )
    print(
        f"Verified {len(paths)} models, {len(jobs)} batches, {sum(j['games'] for j in jobs)} games",
        flush=True,
    )
    if args.verify_only:
        return
    order = list(jobs)
    random.Random(plan["execution"]["shuffle_seed"]).shuffle(order)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=plan["execution"]["parallel_batches"]) as pool:
        pending = {pool.submit(match, job): job for job in order}
        for index, future in enumerate(as_completed(pending), 1):
            future.result()
            print(
                f"{index}/{len(jobs)} {pending[future]['name']} completed", flush=True
            )
    assert (
        make_manifest(configurations(plan), paths) == manifest
    ), "inputs changed during evaluation"
    save(
        FOLDER / "completion.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds_this_invocation": time.monotonic() - started,
            "games": sum(j["games"] for j in jobs),
            "batches": len(jobs),
            "manifest_sha256": digest(record),
            "hashes_verified": True,
        },
    )


if __name__ == "__main__":
    main()
