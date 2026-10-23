"""Run all eighteen fixed fits, then verify every export in the native engine."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np
from common import (
    ROOT,
    FOLDER,
    plan,
    digest,
    save,
    freeze,
    fingerprints,
    verify,
    timestamp,
)
from dataset import load_split


def fit(job, frozen_hash):
    name, seed = job
    output = FOLDER / f"{name}-{seed}.cxe"
    metadata = output.with_suffix(".json")
    if output.exists() or metadata.exists():
        if not output.exists() or not metadata.exists():
            raise ValueError(f"incomplete fit requires inspection: {output}")
        record = json.loads(metadata.read_text())
        assert record["model_sha256"] == digest(output)
        assert record["training_freeze_sha256"] == frozen_hash
    else:
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        with output.with_suffix(".log").open("w") as stream:
            subprocess.run(
                [sys.executable, str(FOLDER / "fit.py"), name, "--seed", str(seed)],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        record = json.loads(metadata.read_text())
    return {
        k: record[k]
        for k in [
            "condition",
            "seed",
            "best_epoch",
            "model_validation_loss",
            "elapsed_seconds",
        ]
    }


def main():
    config = plan()
    labeling = json.loads((FOLDER / "labeling-freeze.json").read_text())
    verify(labeling["hashes"])
    for split in ["train", "validation"]:
        verify(json.loads((FOLDER / f"{split}-labels.json").read_text())["hashes"])
        load_split(split, config["data"]["teacher_nodes"])
    subprocess.run(
        [sys.executable, str(FOLDER / "test_objectives.py")], cwd=ROOT, check=True
    )
    sources = [
        FOLDER / name
        for name in [
            "experiment-plan.json",
            "common.py",
            "dataset.py",
            "objectives.py",
            "fit.py",
            "test_objectives.py",
            "run_fits.py",
            "labeling-freeze.json",
            "labeler-checks.json",
        ]
    ]
    sources += [
        FOLDER / f"{split}-{suffix}"
        for split in ["train", "validation"]
        for suffix in ["input.bin", "labels.bin", "pairs.bin", "labels.json"]
    ]
    sources += [
        *ROOT.glob("training/structured/*.py"),
        *ROOT.glob("training/scaling/*.py"),
        ROOT / config["baseline"],
        ROOT / "engine/target/release/scaling_predict",
    ]
    hashes = fingerprints(sources)
    freeze(
        FOLDER / "training-freeze.json",
        hashes,
        python=sys.version,
        numpy=np.__version__,
        platform=platform.platform(),
    )
    started = time.monotonic()
    jobs = [
        (v["name"], seed)
        for v in config["training"]["variants"]
        for seed in config["training"]["seeds"]
    ]
    assert len(jobs) == 18
    with ThreadPoolExecutor(max_workers=config["compute"]["training_workers"]) as pool:
        futures = [
            pool.submit(fit, job, digest(FOLDER / "training-freeze.json"))
            for job in jobs
        ]
        for count, future in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(completed=count, **future.result())), flush=True)
    verify(hashes)
    outputs = [FOLDER / f"{name}-{seed}.cxe" for name, seed in jobs]
    command = [
        sys.executable,
        str(ROOT / "training/scaling/check.py"),
        config["data"]["source_validation"],
    ]
    for output in outputs:
        command += ["--candidate", str(output)]
    checked = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True
    )
    (FOLDER / "native-parity.log").write_text(checked.stderr)
    parity = json.loads(checked.stdout)
    assert len(parity["candidates"]) == 18 and all(
        c["native_exact"] for c in parity["candidates"]
    )
    save(FOLDER / "native-parity.json", parity)
    verify(hashes)
    save(
        FOLDER / "training-completion.json",
        dict(
            completed_utc=timestamp(),
            fits=len(jobs),
            hashes=fingerprints(
                [
                    *outputs,
                    *[p.with_suffix(".json") for p in outputs],
                    FOLDER / "native-parity.json",
                    FOLDER / "training-freeze.json",
                ]
            ),
            wall_seconds_this_invocation=time.monotonic() - started,
            training_checks="3 objective and real-graph gradient checks passed; original exact initialization checks and native parity passed for all models",
        ),
    )


if __name__ == "__main__":
    main()
