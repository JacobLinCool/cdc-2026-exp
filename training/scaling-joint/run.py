"""Bounded supplement completing the 2 × 3 data/capacity grid."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import datetime
import hashlib
import json
import os
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
BASE = ROOT / "training/scaling"
sys.path.insert(0, str(BASE))
from stats import summarize, contrast


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + "\n")


def execute(command, log, extra=None):
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    env.update(extra or {})
    with log.open("w") as stream:
        subprocess.run(
            list(map(str, command)),
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def parallel(jobs, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fn): name for name, fn in jobs}
        for future in as_completed(pending):
            future.result()
            print(pending[future], flush=True)


def match(name, model, opponent, games, seed, nodes=None, threads=2):
    csv = FOLDER / f"{name}.csv"
    config = {
        "model_sha256": digest(model),
        "opponent": opponent,
        "games": games,
        "first_seed": seed,
        "nodes": nodes,
        "time_ms": 25,
        "threads": threads,
        "arena_sha256": digest(ROOT / "engine/target/release/arena"),
    }
    metadata = json.loads(model.with_suffix(".json").read_text())
    assert metadata["model_sha256"] == config["model_sha256"]
    expected = set(range(seed, seed + games // 2))
    assert not expected.intersection(
        metadata["training_games"] + metadata["validation_games"]
    )
    opponent_path = ROOT / opponent.split("=", 1)[1]
    other = json.loads(opponent_path.with_suffix(".json").read_text())
    assert digest(opponent_path) == other["model_sha256"]
    assert not expected.intersection(
        other["training_games"] + other["validation_games"]
    )
    config["opponent_sha256"] = digest(opponent_path)
    record = FOLDER / f"{name}-run.json"
    if csv.exists():
        assert json.loads(record.read_text())["config"] == config
    else:
        a = f"expanded={model}"
        b = opponent
        if nodes:
            a += f",nodes={nodes},fmd=5"
            b += f",nodes={nodes},fmd=5"
        save(record, {"config": config})
        execute(
            [ROOT / "engine/target/release/arena", games, 25, a, b, threads],
            FOLDER / f"{name}.log",
            {"SEED": str(seed), "ARENA_CSV": str(csv)},
        )
    report = summarize([csv])
    assert (
        report["games"] == games
        and report["first_seed"] == seed
        and report["last_seed"] == seed + games // 2 - 1
    )
    save(record, {"config": config, "csv_sha256": digest(csv)})
    save(FOLDER / f"{name}.json", report)


def main():
    os.chdir(ROOT)
    # The preceding bounded queue must finish before this supplement uses the CPU workers.
    assert json.loads((BASE / "result.json").read_text())["hashes_verified"]
    plan = json.loads((FOLDER / "experiment-plan.json").read_text())
    assert (
        digest(BASE / "development-results.json")
        == plan["supplement"]["reference_sha256"]
    )
    source_files = [
        *BASE.glob("*.py"),
        *FOLDER.glob("*.py"),
        *ROOT.glob("training/structured/*.py"),
        *ROOT.glob("engine/src/**/*.rs"),
        FOLDER / "experiment-plan.json",
    ]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(source_files)}
    manifest = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": source_hashes,
        "train_sha256": digest(FOLDER / "train.bin"),
        "validation_sha256": digest(FOLDER / "validation.bin"),
    }
    previous = FOLDER / "run.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        assert {k: v for k, v in old.items() if k != "started_utc"} == {
            k: v for k, v in manifest.items() if k != "started_utc"
        }
    else:
        save(previous, manifest)
    jobs = []
    for variant in plan["training"]["variants"]:
        for seed in plan["training"]["seeds"]:
            name = f"{variant['name']}-{seed}"

            def fit(name=name, variant=variant, seed=seed):
                model = FOLDER / f"{name}.cxe"
                if model.exists():
                    assert json.loads(model.with_suffix(".json").read_text())[
                        "model_sha256"
                    ] == digest(model)
                    return
                execute(
                    [
                        sys.executable,
                        BASE / "train.py",
                        variant["name"],
                        "--seed",
                        seed,
                        "--directory",
                        FOLDER,
                    ],
                    FOLDER / f"fit-{name}.log",
                )

            jobs.append((name, fit))
    parallel(jobs, 4)
    models = sorted(FOLDER.glob("*.cxe"))
    assert len(models) == 6
    command = [sys.executable, BASE / "check.py", BASE / "validation.bin"]
    for model in models:
        command.extend(["--candidate", model])
    execute(command, FOLDER / "verification.json")
    jobs = []
    for variant in plan["training"]["variants"]:
        for i, seed in enumerate(plan["training"]["seeds"]):
            model = FOLDER / f"{variant['name']}-{seed}.cxe"
            for mode, games, first, nodes in [
                ("time", 80, 16000000, None),
                ("nodes", 40, 17000000, 8192),
            ]:
                name = f"dev-{mode}-{variant['name']}-{seed}"

                def job(
                    name=name, model=model, games=games, first=first, i=i, nodes=nodes
                ):
                    match(
                        name,
                        model,
                        "structured=training/ranking/value.cse",
                        games,
                        first + i * (games // 2),
                        nodes,
                    )

                jobs.append((name, job))
    random.Random(7302).shuffle(jobs)
    parallel(jobs, 6)
    results = {}
    for variant in plan["training"]["variants"]:
        name = variant["name"]
        reports = {}
        for mode in ["time", "nodes"]:
            files = [
                FOLDER / f"dev-{mode}-{name}-{seed}.csv"
                for seed in plan["training"]["seeds"]
            ]
            reports[mode] = summarize(files)
            reports[mode]["contrast_to_data4x"] = contrast(
                files,
                [
                    BASE / f"dev-{mode}-data4x-{seed}.csv"
                    for seed in plan["training"]["seeds"]
                ],
            )
            reports[mode]["contrast_to_same_capacity_small_data"] = contrast(
                files,
                [
                    BASE / f"dev-{mode}-{name.removeprefix('data4x-')}-{seed}.csv"
                    for seed in plan["training"]["seeds"]
                ],
            )
        reports["fits"] = [
            json.loads((FOLDER / f"{name}-{seed}.json").read_text())
            for seed in plan["training"]["seeds"]
        ]
        results[name] = reports
    save(FOLDER / "development-results.json", results)
    reference = json.loads((BASE / "development-results.json").read_text())["time"][
        "small"
    ]["score"]
    winner = min(
        results,
        key=lambda name: (
            -results[name]["time"]["score"],
            sum(m["model_validation_loss"] for m in results[name]["fits"]) / 3,
            name,
        ),
    )
    eligible = results[winner]["time"]["score"] > reference
    output = {
        "reference_development_score": reference,
        "best_condition": winner,
        "advanced_to_holdout": eligible,
        "results": results,
    }
    if eligible:
        chosen = min(
            results[winner]["fits"],
            key=lambda m: (m["model_validation_loss"], m["seed"]),
        )
        model = FOLDER / f"{winner}-{chosen['seed']}.cxe"
        frozen = {
            "model": str(model.relative_to(ROOT)),
            "model_sha256": digest(model),
            "sources": source_hashes,
            "arena_sha256": digest(ROOT / "engine/target/release/arena"),
            "protocol": plan["supplement"],
        }
        save(FOLDER / "frozen-candidate.json", frozen)
        jobs = []
        for name, opponent in [
            ("current", "structured=training/ranking/value.cse"),
            ("main-candidate", "expanded=training/scaling/small-53.cxe"),
        ]:

            def job(name=name, opponent=opponent):
                match(f"holdout-{name}", model, opponent, 800, 21000000, threads=6)

            jobs.append((name, job))
        parallel(jobs, 2)
        output["holdouts"] = {
            name: summarize([FOLDER / f"holdout-{name}.csv"])
            for name in ["current", "main-candidate"]
        }
    assert all(digest(ROOT / path) == value for path, value in source_hashes.items())
    output["hashes_verified"] = True
    output["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(FOLDER / "result.json", output)
    print(json.dumps({"best_condition": winner, "advanced": eligible}), flush=True)


if __name__ == "__main__":
    main()
