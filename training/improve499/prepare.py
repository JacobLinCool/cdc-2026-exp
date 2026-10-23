"""Freeze existing root selection, verify repeatability, then collect deeper labels."""

import hashlib
import json
import subprocess
import tempfile
import time
import numpy as np
from common import (
    ROOT,
    FOLDER,
    digest,
    relative,
    save,
    plan,
    fingerprints,
    verify,
    freeze,
    engine_sources,
)
from data import read, quiet, deep_valid, phases
from dataset import LABEL, PAIR, read_binary, load_split

LABELER = FOLDER / "labeler/target/release/cdc-refine-labeler"


def inputs(config):
    baseline = ROOT / config["baseline"]
    assert digest(baseline) == config["baseline_sha256"]
    metadata = json.loads(baseline.with_suffix(".json").read_text())
    result = {}
    games = []
    for split in ["train", "validation"]:
        source = ROOT / config["data"][f"source_{split}"]
        records = read(source)
        records = records[np.lexsort((records["ply"], records["seed"]))]
        eligible = (
            quiet(records) & (deep_valid(records) | (records["seed"] >= 14000512))
            if split == "train"
            else quiet(records) & deep_valid(records)
        )
        ids = np.flatnonzero(eligible)
        key = "training" if split == "train" else "validation"
        assert len(ids) == config["data"][f"{key}_rows"]
        assert digest(source) == metadata[f"{key}_data_sha256"]
        row_hash = hashlib.sha256(ids.astype("<u8").tobytes()).hexdigest()
        if split == "train":
            assert row_hash == metadata["training_row_ids_sha256"]
        selected = records[ids]
        path = FOLDER / f"{split}-input.bin"
        blob = b"CDCSCL01" + selected.tobytes()
        if path.exists():
            assert path.read_bytes() == blob
        else:
            with path.open("xb") as out:
                out.write(blob)
        games.append(set(map(int, selected["seed"])))
        result[split] = dict(
            rows=len(ids),
            games=len(games[-1]),
            input_sha256=digest(path),
            source=relative(source),
            source_sha256=digest(source),
            source_row_ids_sha256=row_hash,
            phases=np.bincount(phases(selected), minlength=3).tolist(),
        )
    assert not games[0].intersection(games[1])
    return result


def invoke(source, labels, pairs, baseline, nodes, workers, stream):
    subprocess.run(
        [
            str(LABELER),
            str(source),
            str(labels),
            str(pairs),
            str(baseline),
            str(nodes),
            str(workers),
        ],
        cwd=ROOT,
        stdout=stream,
        stderr=subprocess.STDOUT,
        check=True,
    )


def smoke(config):
    records = read(FOLDER / "train-input.bin")
    ids = np.concatenate(
        [np.flatnonzero(phases(records) == phase)[:6] for phase in range(3)]
    )
    with tempfile.TemporaryDirectory(prefix="cdc-improve499-smoke-") as directory:
        directory = __import__("pathlib").Path(directory)
        source = directory / "input.bin"
        source.write_bytes(b"CDCSCL01" + records[ids].tobytes())
        runs = []
        for workers in [1, 2]:
            labels, pairs = [
                directory / f"{kind}-{workers}.bin" for kind in ["labels", "pairs"]
            ]
            invoke(
                source, labels, pairs, ROOT / config["baseline"], 65536, workers, None
            )
            values = read_binary(labels, b"CDCFT001", LABEL).copy()
            assert np.all(values["nodes"] <= 65536)
            values["time_ms"] = 0
            runs.append((values.tobytes(), pairs.read_bytes()))
        assert runs[0] == runs[1], "node-limited teacher depends on scheduling"
    return dict(
        roots=len(ids),
        workers=[1, 2],
        node_limit=65536,
        all_nontiming_labels_and_pairs_identical=True,
        public_state_feature_and_baseline_assertions=True,
    )


def main():
    config = plan()
    description = inputs(config)
    sources = [
        FOLDER / "experiment-plan.json",
        FOLDER / "prepare.py",
        FOLDER / "common.py",
        FOLDER / "dataset.py",
        FOLDER / "labeler/Cargo.toml",
        FOLDER / "labeler/Cargo.lock",
        FOLDER / "labeler/src/main.rs",
        LABELER,
        ROOT / config["baseline"],
        *engine_sources(),
        *[FOLDER / f"{name}-input.bin" for name in ["train", "validation"]],
        *[ROOT / config["data"][f"source_{name}"] for name in ["train", "validation"]],
        ROOT / "training/scaling/data.py",
    ]
    frozen = freeze(
        FOLDER / "labeling-freeze.json",
        fingerprints(sources),
        input_selection=description,
    )
    if not (FOLDER / "labeler-checks.json").exists():
        save(FOLDER / "labeler-checks.json", smoke(config))
    for split in ["train", "validation"]:
        record_path = FOLDER / f"{split}-labels.json"
        labels, pairs = [FOLDER / f"{split}-{kind}.bin" for kind in ["labels", "pairs"]]
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            verify(previous["hashes"])
            load_split(split, config["data"]["teacher_nodes"])
            print(f"Verified existing {split} labels", flush=True)
            continue
        if labels.exists() or pairs.exists():
            raise FileExistsError(
                "unregistered label output; inspect it before resuming"
            )
        started = time.monotonic()
        with (FOLDER / f"{split}-labels.log").open("w") as stream:
            invoke(
                FOLDER / f"{split}-input.bin",
                labels,
                pairs,
                ROOT / config["baseline"],
                config["data"]["teacher_nodes"],
                config["compute"]["label_workers"],
                stream,
            )
        roots, values, preferences, valid = load_split(
            split, config["data"]["teacher_nodes"]
        )
        result = dict(
            hashes=fingerprints([labels, pairs]),
            input_sha256=digest(FOLDER / f"{split}-input.bin"),
            roots=len(roots),
            valid_roots=int(valid.sum()),
            pairs=len(preferences),
            pair_parent_games=len(np.unique(preferences["seed"])),
            invalid_depth=int((values["depth"] < 3).sum()),
            invalid_terminal_score=int((abs(values["score"]) >= 5000).sum()),
            mean_depth=float(values["depth"].mean()),
            mean_nodes=float(values["nodes"].mean()),
            mean_time_ms=float(values["time_ms"].mean()),
            elapsed_seconds=time.monotonic() - started,
        )
        verify(frozen["hashes"])
        save(record_path, result)
        print(json.dumps(dict(split=split, **result)), flush=True)
    print("Both splits complete; all frozen inputs verified", flush=True)


if __name__ == "__main__":
    main()
