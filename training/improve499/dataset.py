"""Validated, split-preserving root labels and quiet successor preferences."""

import numpy as np
from common import FOLDER
from data import FEATURES, read

LABEL = np.dtype(
    [
        ("row", "<u4"),
        ("score", "<i4"),
        ("depth", "<u2"),
        ("move", "<u2"),
        ("nodes", "<u8"),
        ("time_ms", "<f8"),
        ("valid", "<u4"),
    ]
)
PAIR = np.dtype(
    [
        ("seed", "<u8"),
        ("ply", "<u2"),
        ("depth", "<u2"),
        ("moves", "<u2", (2,)),
        ("teacher", "<i4", (2,)),
        ("baseline", "<i4", (2,)),
        ("features", "<i2", (2, FEATURES)),
    ]
)
assert LABEL.itemsize == 32 and PAIR.itemsize == 2180


def read_binary(path, header, dtype):
    with path.open("rb") as stream:
        if stream.read(8) != header:
            raise ValueError(f"invalid header: {path}")
    size = path.stat().st_size - 8
    if size <= 0 or size % dtype.itemsize:
        raise ValueError(f"empty or truncated data: {path}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8)


def load_split(name, node_limit):
    roots = read(FOLDER / f"{name}-input.bin")
    labels = read_binary(FOLDER / f"{name}-labels.bin", b"CDCFT001", LABEL)
    pairs = read_binary(FOLDER / f"{name}-pairs.bin", b"CDCRNK01", PAIR)
    if len(labels) != len(roots) or not np.array_equal(
        labels["row"], np.arange(len(roots))
    ):
        raise ValueError("labels do not align with roots")
    valid = (labels["depth"] >= 3) & (abs(labels["score"]) < 5000)
    if not np.array_equal(labels["valid"], valid.astype(np.uint32)):
        raise ValueError("incorrect teacher validity flags")
    if np.any(labels["nodes"] > node_limit) or np.any(labels["time_ms"] < 0):
        raise ValueError("teacher node cap or timing invalid")
    index = {(int(r["seed"]), int(r["ply"])): i for i, r in enumerate(roots)}
    if len(index) != len(roots):
        raise ValueError("duplicate root state identifier")
    parent = np.array([index[(int(p["seed"]), int(p["ply"]))] for p in pairs])
    gap = pairs["teacher"][:, 0] - pairs["teacher"][:, 1]
    if (
        not np.all(valid[parent])
        or not np.array_equal(pairs["depth"], labels["depth"][parent])
        or np.any((gap < 10) | (gap > 600))
        or np.any(abs(pairs["teacher"]) >= 5000)
        or np.any(pairs["moves"][:, 0] == pairs["moves"][:, 1])
        or np.any(np.bincount(parent, minlength=len(roots)) > 3)
    ):
        raise ValueError("invalid teacher preference or split")
    return roots, labels, pairs, valid
