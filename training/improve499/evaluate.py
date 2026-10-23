"""Frozen development selection and independent paired-game confirmation."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import random
import subprocess
import time
import numpy as np
from common import (
    ROOT,
    FOLDER,
    plan,
    digest,
    relative,
    save,
    freeze,
    fingerprints,
    verify,
    timestamp,
    engine_sources,
)
from stats import summarize, read_games

ARENA = ROOT / "engine/target/release/arena"


def development_jobs(config):
    evaluation = config["evaluation"]
    games = evaluation["development_games_per_fit"]
    return [
        dict(
            name=f"dev-{v['name']}-{seed}",
            condition=v["name"],
            training_seed=seed,
            model=relative(FOLDER / f"{v['name']}-{seed}.cxe"),
            opponent=config["baseline"],
            games=games,
            time_ms=evaluation["development_ms"],
            first_seed=evaluation["development_first_seed"] + i * (games // 2),
            threads=config["compute"]["threads_per_development_batch"],
        )
        for v in config["training"]["variants"]
        for i, seed in enumerate(config["training"]["seeds"])
    ]


def holdout_jobs(config, selected):
    if not selected["advanced"]:
        return []
    return [
        dict(
            name=f"holdout-{stage}",
            condition=selected["condition"],
            training_seed=selected["training_seed"],
            model=selected["model"],
            opponent=config["baseline"],
            games=config["evaluation"][f"{stage}_holdout_games"],
            time_ms=config["evaluation"][f"{stage}_holdout_ms"],
            first_seed=config["evaluation"][f"{stage}_holdout_first_seed"],
            threads=config["compute"]["threads_per_final_batch"],
        )
        for stage in ["primary", "secondary"]
    ]


def verify_csv(path, job):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == job["games"], f"incomplete games: {path}"
    deals = {}
    for row in rows:
        assert float(row["score"]) in [0.0, 0.5, 1.0]
        deals.setdefault(int(row["seed"]), []).append(row["a_first"])
        assert int(row["plies"]) == int(row["a_moves"]) + int(row["b_moves"])
        for side in ["a", "b"]:
            assert int(row[f"{side}_moves"]) > 0
            assert int(row[f"{side}_nodes"]) >= 0
            assert int(row[f"{side}_depth"]) >= 0
            assert (
                np.isfinite(float(row[f"{side}_ms"])) and float(row[f"{side}_ms"]) >= 0
            )
    assert set(deals) == set(
        range(job["first_seed"], job["first_seed"] + job["games"] // 2)
    )
    assert all(sorted(pair) == ["false", "true"] for pair in deals.values())


def registered_output(job):
    path = FOLDER / f"{job['name']}.csv"
    record = json.loads((FOLDER / f"{job['name']}-run.json").read_text())
    assert record["job"] == job and record["csv_sha256"] == digest(path)
    verify_csv(path, job)
    return path


def match(job):
    path = FOLDER / f"{job['name']}.csv"
    record = FOLDER / f"{job['name']}-run.json"
    if record.exists():
        previous = json.loads(record.read_text())
        assert previous["job"] == job
        if "csv_sha256" in previous:
            registered_output(job)
            return
        if path.exists():
            verify_csv(path, job)
            save(
                record,
                dict(
                    **previous, csv_sha256=digest(path), recovered_completed_output=True
                ),
            )
            return
    elif path.exists():
        raise FileExistsError(f"unregistered output: {path}")
    started = time.monotonic()
    before = dict(job=job, started_utc=timestamp())
    save(record, before)
    env = dict(os.environ, SEED=str(job["first_seed"]), ARENA_CSV=str(path))
    with (FOLDER / f"{job['name']}.log").open("w") as stream:
        subprocess.run(
            [
                str(ARENA),
                str(job["games"]),
                str(job["time_ms"]),
                f"expanded={job['model']}",
                f"expanded={job['opponent']}",
                str(job["threads"]),
            ],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    verify_csv(path, job)
    save(
        record,
        dict(
            **before,
            csv_sha256=digest(path),
            elapsed_seconds=time.monotonic() - started,
        ),
    )


def run_jobs(jobs, workers, freeze_path, extra_files=()):
    paths = [
        FOLDER / "experiment-plan.json",
        FOLDER / "evaluate.py",
        FOLDER / "test_evaluation.py",
        FOLDER / "common.py",
        ROOT / "training/scaling/stats.py",
        ARENA,
        FOLDER / "training-completion.json",
        *engine_sources(),
        *extra_files,
    ]
    for job in jobs:
        for name in [job["model"], job["opponent"]]:
            model = ROOT / name
            metadata = json.loads(model.with_suffix(".json").read_text())
            assert digest(model) == metadata["model_sha256"]
            assert metadata["trainable_parameters"] == 499
            deals = set(range(job["first_seed"], job["first_seed"] + job["games"] // 2))
            assert not deals.intersection(
                metadata["training_games"] + metadata["validation_games"]
            )
            paths += [model, model.with_suffix(".json")]
    hashes = fingerprints(paths)
    frozen = freeze(freeze_path, hashes, jobs=jobs)
    assert frozen["jobs"] == jobs
    order = list(jobs)
    random.Random(84531).shuffle(order)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(match, job): job for job in order}
        for count, future in enumerate(as_completed(pending), 1):
            future.result()
            print(f"{count}/{len(jobs)} {pending[future]['name']} complete", flush=True)
    verify(hashes)
    for job in jobs:
        registered_output(job)


def stratified_interval(blocks):
    values = np.asarray(blocks, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("expected equal paired-deal blocks per training seed")
    indices = np.random.default_rng(84271).integers(
        values.shape[1], size=(values.shape[0], 20000, values.shape[1])
    )
    samples = values[np.arange(len(values))[:, None, None], indices].mean(axis=(0, 2))
    return list(map(float, np.quantile(samples, [0.025, 0.975])))


def condition_report(jobs):
    paths = [registered_output(job) for job in jobs]
    result = summarize(paths)
    result["files"] = list(map(relative, paths))
    blocks = [list(read_games([path])[1].values()) for path in paths]
    result["paired_bootstrap_95_ci"] = stratified_interval(blocks)
    result["training_seed_scores"] = {
        str(job["training_seed"]): float(np.mean(block))
        for job, block in zip(jobs, blocks)
    }
    metadata = [
        json.loads((ROOT / job["model"]).with_suffix(".json").read_text())
        for job in jobs
    ]
    result["mean_validation_loss"] = float(
        np.mean([m["model_validation_loss"] for m in metadata])
    )
    result[
        "inference"
    ] = "Exploratory development estimate; whole deals resampled within each fixed training-seed block."
    return result


def choose(reports, metadata):
    condition = min(
        reports,
        key=lambda name: (
            -reports[name]["score"],
            reports[name]["mean_validation_loss"],
            name,
        ),
    )
    chosen = min(
        [m for m in metadata if m["condition"] == condition],
        key=lambda m: (m["model_validation_loss"], m["seed"]),
    )
    return dict(
        advanced=reports[condition]["score"] > 0.5,
        condition=condition,
        training_seed=chosen["seed"],
        model=relative(FOLDER / f"{condition}-{chosen['seed']}.cxe"),
        model_sha256=chosen["model_sha256"],
        development_score=reports[condition]["score"],
        common_validation_loss=chosen["model_validation_loss"],
    )


def selection(config, jobs):
    reports = {
        v["name"]: condition_report(
            [job for job in jobs if job["condition"] == v["name"]]
        )
        for v in config["training"]["variants"]
    }
    metadata = [
        json.loads((ROOT / job["model"]).with_suffix(".json").read_text())
        for job in jobs
    ]
    selected = choose(reports, metadata)
    selected["development_reports"] = reports
    selected["selection_rule"] = config["evaluation"]["selection"]
    path = FOLDER / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == selected, "frozen selection changed"
    else:
        save(path, selected)
    return selected


def summarize_final(config, selected, jobs):
    reports = {}
    for job in jobs:
        report = summarize([registered_output(job)])
        report["files"] = [relative(FOLDER / f"{job['name']}.csv")]
        report["time_ms"] = job["time_ms"]
        report["win_rate"] = report["wins"] / report["games"]
        report["loss_rate"] = report["losses"] / report["games"]
        reports[job["name"].removeprefix("holdout-")] = report
    confirmed = bool(reports and reports["primary"]["paired_bootstrap_95_ci"][0] > 0.5)
    result = dict(
        completed_utc=timestamp(),
        baseline=config["baseline"],
        baseline_sha256=config["baseline_sha256"],
        parameters=499,
        model_bytes=4252,
        selected={k: v for k, v in selected.items() if k != "development_reports"},
        development=selected["development_reports"],
        holdout=reports,
        confirmed_improvement=confirmed,
        recommended_model=selected["model"] if confirmed else config["baseline"],
        development_games=sum(
            r["games"] for r in selected["development_reports"].values()
        ),
        holdout_games=sum(r["games"] for r in reports.values()),
        interpretation="Fixed-count independent primary confirmation; pointwise secondary result. Development was used for selection and is not independent evidence.",
    )
    save(FOLDER / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = plan()
    verify(json.loads((FOLDER / "training-completion.json").read_text())["hashes"])
    verify(json.loads((FOLDER / "training-freeze.json").read_text())["hashes"])
    development = development_jobs(config)
    assert len(development) == 18 and sum(j["games"] for j in development) == 1152
    if not args.verify_only:
        run_jobs(
            development,
            config["compute"]["development_batches"],
            FOLDER / "development-freeze.json",
        )
    selected = selection(config, development)
    print(
        json.dumps({k: v for k, v in selected.items() if k != "development_reports"}),
        flush=True,
    )
    holdout = holdout_jobs(config, selected)
    if holdout and not args.verify_only:
        run_jobs(
            holdout,
            config["compute"]["final_batches"],
            FOLDER / "holdout-freeze.json",
            [FOLDER / "selection.json", FOLDER / "development-freeze.json"],
        )
    for name in ["development-freeze.json"] + (
        ["holdout-freeze.json"] if holdout else []
    ):
        verify(json.loads((FOLDER / name).read_text())["hashes"])
    result = summarize_final(config, selected, holdout)
    save(
        FOLDER / "verification.json",
        dict(
            completed_utc=timestamp(),
            source_and_model_hashes_verified=True,
            paired_deals_and_game_counts_verified=True,
            training_validation_match_seeds_disjoint=True,
            development_games=result["development_games"],
            holdout_games=result["holdout_games"],
            hashes=fingerprints(
                [
                    FOLDER / "results.json",
                    FOLDER / "selection.json",
                    *[FOLDER / f"{job['name']}.csv" for job in development + holdout],
                    *[
                        FOLDER / f"{job['name']}-run.json"
                        for job in development + holdout
                    ],
                ]
            ),
        ),
    )
    print(
        json.dumps(
            dict(
                confirmed_improvement=result["confirmed_improvement"],
                holdout=result["holdout"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
