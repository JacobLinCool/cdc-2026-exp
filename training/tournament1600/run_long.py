"""Run all frozen 1,600 ms cells in paired, resumable four-game shards."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import platform
import random
import subprocess
import threading
import time

from experiment import (
    ARENA,
    FOLDER,
    ROOT,
    base_run,
    digest,
    load_plan,
    read_shard,
    save,
    shards,
    timestamp,
    verify_inputs,
)

PRINT_LOCK = threading.Lock()


def progress(message):
    with PRINT_LOCK:
        print(message, flush=True)


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
            read_shard(job)
            return
    elif output.exists():
        raise ValueError(f"unregistered output: {output}")
    if output.exists():
        base_run.verify_csv(output, job)
        save(
            record,
            {**previous, "csv_sha256": digest(output), "recovered_utc": timestamp()},
        )
        return
    status = {"config": job, "started_utc": timestamp()}
    save(record, status)
    started = time.monotonic()
    progress(f"START {name}")
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
    base_run.verify_csv(output, job)
    save(
        record,
        {
            **status,
            "csv_sha256": digest(output),
            "completed_utc": timestamp(),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    plan = load_plan()
    jobs = shards(plan)
    manifest = verify_inputs(plan, jobs)
    progress(
        f"Verified {len(jobs)} shards / {sum(j['games'] for j in jobs)} games / 15 cells"
    )
    if args.verify_only:
        return
    order = list(jobs)
    random.Random(plan["execution"]["shuffle_seed"]).shuffle(order)
    record = FOLDER / "run.json"
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
        pending = {pool.submit(match, job): job for job in order}
        for index, future in enumerate(as_completed(pending), 1):
            future.result()
            progress(f"DONE {index}/{len(jobs)} {pending[future]['name']}")
    if verify_inputs(plan, jobs) != manifest:
        raise ValueError("extension inputs changed during play")
    for job in jobs:
        read_shard(job)
    save(
        FOLDER / "completion.json",
        {
            "completed_utc": timestamp(),
            "wall_seconds_this_invocation": time.monotonic() - started,
            "games": sum(j["games"] for j in jobs),
            "batches": len(jobs),
            "cells": plan["execution"]["expected_cells"],
            "manifest_sha256": digest(record),
            "hashes_verified": True,
            "csv_sha256": {
                j["name"]: digest(FOLDER / f"{j['name']}.csv") for j in jobs
            },
        },
    )


if __name__ == "__main__":
    main()
