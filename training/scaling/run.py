"""Bounded, resumable experiment queue. Never selects candidates on final games."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from data import read, phases, quiet, deep_valid
from stats import summarize, contrast
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "training/scaling"
PLAN = FOLDER / "experiment-plan.json"
ARENA = ROOT / "engine/target/release/arena"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sources():
    files = [
        *ROOT.glob("engine/src/**/*.rs"),
        *ROOT.glob("engine/tests/*.rs"),
        *ROOT.glob("training/structured/*.py"),
        *ROOT.glob("training/scaling/*.py"),
        PLAN,
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def subprocess_job(command, log, environment=None):
    start = time.perf_counter()
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"
    )
    if environment:
        env.update(environment)
    with log.open("w") as stream:
        subprocess.run(
            list(map(str, command)),
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.perf_counter() - start


def parallel(jobs, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for future in as_completed(futures):
            result = future.result()
            print(
                json.dumps({"completed": futures[future], "result": result}), flush=True
            )


def fit(plan):
    for name, games in [("train", 2048), ("validation", 256)]:
        log = (FOLDER / f"collect-{name}.log").read_text()
        if f"completed={games}/{games}" not in log:
            raise ValueError(f"{name} collection is not complete")
    manifest_path = FOLDER / "training-manifest.json"
    current = {
        "sources": sources(),
        "train_sha256": digest(FOLDER / "train.bin"),
        "validation_sha256": digest(FOLDER / "validation.bin"),
        "initial_sha256": digest(ROOT / plan["initial_model"]),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["inputs"] != current:
            raise ValueError("training inputs changed; use a new experiment directory")
    else:
        write(manifest_path, {"started_utc": utc(), "inputs": current})
    jobs = []
    for variant in plan["training"]["variants"]:
        for seed in plan["training"]["seeds"]:
            name = f"{variant['name']}-{seed}"
            model = FOLDER / f"{name}.cxe"

            def job(name=name, variant=variant, seed=seed, model=model):
                metadata = model.with_suffix(".json")
                if model.exists() and metadata.exists():
                    if json.loads(metadata.read_text())["model_sha256"] != digest(
                        model
                    ):
                        raise ValueError(f"model hash mismatch: {model}")
                    return "verified existing fit"
                return subprocess_job(
                    [
                        sys.executable,
                        FOLDER / "train.py",
                        variant["name"],
                        "--seed",
                        seed,
                    ],
                    FOLDER / f"fit-{name}.log",
                )

            jobs.append((name, job))
    parallel(jobs, 4)
    if sources() != current["sources"]:
        raise ValueError("source changed during training")
    metadata = {
        p.stem: json.loads(p.read_text())
        for p in FOLDER.glob("*.json")
        if p.with_suffix(".cxe").exists()
    }
    write(FOLDER / "training-results.json", {"completed_utc": utc(), "fits": metadata})


def match_job(
    name, model, opponent, games, first_seed, time_ms, threads, node_limit=None
):
    csv = FOLDER / f"{name}.csv"
    manifest = FOLDER / f"{name}-run.json"
    model_metadata = json.loads(model.with_suffix(".json").read_text())
    if model_metadata["model_sha256"] != digest(model):
        raise ValueError("candidate metadata and weights disagree")
    expected_seeds = set(range(first_seed, first_seed + games // 2))
    if expected_seeds.intersection(
        model_metadata["training_games"] + model_metadata["validation_games"]
    ):
        raise ValueError("match seeds overlap training or validation")
    opponent_hash = None
    for specification in opponent.split(","):
        key, _, value = specification.partition("=")
        if key in ["model", "structured", "expanded"]:
            opponent_path = ROOT / value
            opponent_metadata = json.loads(
                opponent_path.with_suffix(".json").read_text()
            )
            opponent_hash = digest(opponent_path)
            if opponent_hash != opponent_metadata["model_sha256"]:
                raise ValueError("opponent metadata and weights disagree")
            if expected_seeds.intersection(
                opponent_metadata["training_games"]
                + opponent_metadata["validation_games"]
            ):
                raise ValueError("match seeds overlap opponent training or validation")
    config = {
        "model": str(model.relative_to(ROOT)),
        "model_sha256": digest(model),
        "opponent": opponent,
        "opponent_sha256": opponent_hash,
        "games": games,
        "first_seed": first_seed,
        "time_ms": time_ms,
        "threads": threads,
        "node_limit": node_limit,
        "arena_sha256": digest(ARENA),
    }
    if csv.exists():
        if (
            not manifest.exists()
            or json.loads(manifest.read_text())["config"] != config
        ):
            raise ValueError(f"cannot reuse unmatched result {csv}")
        report = summarize([csv])
        if (
            report["games"] != games
            or report["first_seed"] != first_seed
            or report["last_seed"] != first_seed + games // 2 - 1
        ):
            raise ValueError("incomplete match")
        return report
    spec = f"expanded={model}"
    if node_limit:
        suffix = f",nodes={node_limit},fmd=5"
        spec += suffix
        opponent += suffix
    write(manifest, {"started_utc": utc(), "config": config})
    elapsed = subprocess_job(
        [ARENA, games, time_ms, spec, opponent, threads],
        FOLDER / f"{name}.log",
        {"SEED": str(first_seed), "ARENA_CSV": str(csv)},
    )
    report = summarize([csv])
    if report["games"] != games:
        raise ValueError("wrong game count")
    write(
        manifest,
        {
            "completed_utc": utc(),
            "elapsed_seconds": elapsed,
            "config": config,
            "csv_sha256": digest(csv),
        },
    )
    write(FOLDER / f"{name}.json", report)
    return {"score": report["score"], "seconds": elapsed}


def development(plan):
    before = sources()
    names = [v["name"] for v in plan["training"]["variants"]]
    jobs = []
    for mode, games, key, nodes in [
        ("time", 80, "development_first_seed", None),
        ("nodes", 40, "fixed_nodes_first_seed", 8192),
    ]:
        for condition in names:
            for index, seed in enumerate(plan["training"]["seeds"]):
                name = f"dev-{mode}-{condition}-{seed}"
                model = FOLDER / f"{condition}-{seed}.cxe"
                first = plan["evaluation"][key] + index * (games // 2)

                def job(name=name, model=model, games=games, first=first, nodes=nodes):
                    return match_job(
                        name,
                        model,
                        "structured=training/ranking/value.cse",
                        games,
                        first,
                        25,
                        2,
                        nodes,
                    )

                jobs.append((name, job))
    random.Random(7301).shuffle(jobs)
    parallel(jobs, 6)
    if sources() != before:
        raise ValueError("source changed during development")
    results = {}
    for mode in ["time", "nodes"]:
        control = [
            FOLDER / f"dev-{mode}-small-{seed}.csv"
            for seed in plan["training"]["seeds"]
        ]
        results[mode] = {}
        for condition in names:
            paths = [
                FOLDER / f"dev-{mode}-{condition}-{seed}.csv"
                for seed in plan["training"]["seeds"]
            ]
            report = summarize(paths)
            report["contrast_to_small"] = contrast(paths, control)
            report["training_seeds"] = {
                str(seed): summarize([FOLDER / f"dev-{mode}-{condition}-{seed}.csv"])
                for seed in plan["training"]["seeds"]
            }
            results[mode][condition] = report
    metadata = {
        condition: [
            json.loads((FOLDER / f"{condition}-{seed}.json").read_text())
            for seed in plan["training"]["seeds"]
        ]
        for condition in names
    }
    winner = min(
        names,
        key=lambda name: (
            -results["time"][name]["score"],
            np.mean([m["model_validation_loss"] for m in metadata[name]]),
            name,
        ),
    )
    selected = min(
        metadata[winner], key=lambda m: (m["model_validation_loss"], m["seed"])
    )
    model = FOLDER / f"{winner}-{selected['seed']}.cxe"
    write(FOLDER / "development-results.json", results)
    frozen = {
        "frozen_utc": utc(),
        "condition": winner,
        "seed": selected["seed"],
        "model": str(model.relative_to(ROOT)),
        "model_sha256": digest(model),
        "arena_sha256": digest(ARENA),
        "sources": before,
        "all_model_hashes": {p.name: digest(p) for p in sorted(FOLDER.glob("*.cxe"))},
        "protocol": plan["evaluation"],
        "primary_comparison": "25ms versus training/ranking/value.cse",
        "selection": "Condition by aggregate development score, seed by common validation loss only",
    }
    write(FOLDER / "frozen-candidate.json", frozen)
    print(json.dumps({"selected": frozen["model"]}), flush=True)


def holdout(plan):
    frozen = json.loads((FOLDER / "frozen-candidate.json").read_text())
    if sources() != frozen["sources"] or digest(ARENA) != frozen["arena_sha256"]:
        raise ValueError("frozen code changed")
    model = ROOT / frozen["model"]
    if digest(model) != frozen["model_sha256"]:
        raise ValueError("frozen model changed")
    jobs = []
    for name, opponent in [
        ("original", ""),
        ("current", "structured=training/ranking/value.cse"),
        ("residual", "model=training/strong-teacher.nn,model_scale=0.5"),
    ]:

        def job(name=name, opponent=opponent):
            return match_job(f"holdout-{name}", model, opponent, 800, 18000000, 25, 4)

        jobs.append((name, job))
    parallel(jobs, 3)
    result = match_job(
        "holdout-current-100ms",
        model,
        "structured=training/ranking/value.cse",
        400,
        19000000,
        100,
        12,
    )
    print(
        json.dumps({"completed": "holdout-current-100ms", "result": result}), flush=True
    )
    if (
        sources() != frozen["sources"]
        or digest(model) != frozen["model_sha256"]
        or digest(ARENA) != frozen["arena_sha256"]
    ):
        raise ValueError("frozen artifacts changed during holdout")
    results = {
        name: summarize([FOLDER / f"holdout-{name}.csv"])
        for name in ["original", "current", "residual", "current-100ms"]
    }
    write(
        FOLDER / "result.json",
        {
            "completed_utc": utc(),
            "candidate": frozen,
            "holdouts": results,
            "hashes_verified": True,
            "claim_policy": "Primary endpoint is current structured model at 25ms; other comparisons are secondary and pointwise, not a familywise claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "development", "holdout"])
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = json.loads(PLAN.read_text())
    {"fit": fit, "development": development, "holdout": holdout}[args.stage](plan)


if __name__ == "__main__":
    main()
