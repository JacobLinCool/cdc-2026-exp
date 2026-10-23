"""The versioned public-root corpus written by scaling_data (no hidden deal)."""
from pathlib import Path
import numpy as np

FEATURES = 537
DTYPE = np.dtype(
    [
        ("seed", "<u8"),
        ("ply", "<u2"),
        ("flags", "<u2"),
        ("depth", "<u2"),
        ("deep_depth", "<u2"),
        ("teacher", "<i4"),
        ("deep_teacher", "<i4"),
        ("time_ms", "<f4"),
        ("deep_time_ms", "<f4"),
        ("nodes", "<u8"),
        ("deep_nodes", "<u8"),
        ("baseline", "<i4"),
        ("cells", "u1", (32,)),
        ("pool", "u1", (14,)),
        ("alive", "u1", (14,)),
        ("side", "u1"),
        ("hidden", "u1"),
        ("nocap", "<u2"),
        ("history_length", "<u2"),
        ("history", "<u8", (60,)),
        ("features", "<i2", (FEATURES,)),
    ]
)
assert DTYPE.itemsize == 1672


def read(path):
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(8) != b"CDCSCL01":
            raise ValueError("invalid scaling corpus header")
    size = path.stat().st_size - 8
    if size <= 0 or size % DTYPE.itemsize:
        raise ValueError("empty or incomplete scaling corpus")
    records = np.memmap(path, dtype=DTYPE, mode="r", offset=8)
    if (
        np.any(records["depth"] < 3)
        or np.any(abs(records["teacher"]) >= 5000)
        or np.any(records["side"] > 1)
        or np.any(records["history_length"] > 60)
        or np.any(records["hidden"] != records["pool"].sum(axis=1))
        or np.any(records["pool"] > records["alive"])
    ):
        raise ValueError("invalid public state or labels")
    return records


def phases(records):
    return np.where(records["hidden"] == 0, 0, np.where(records["hidden"] < 8, 1, 2))


def quiet(records):
    return records["flags"] & 1 != 0


def deep_valid(records):
    return records["flags"] & 8 != 0
