"""Sparse composition tables, with exact fixed-point forward and explicit STE."""
from pathlib import Path
import struct
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "structured"))
import model as base
from autodiff import Tensor

SMALL = 8 * 45
LARGE = 9 * 7 * 23
PREDATOR_OFFSETS = np.array([0, 6, 8, 12, 18, 26, 36])
COUNTS = np.array([1, 2, 2, 2, 2, 2, 5])
OFFSETS = np.array([0, 2, 5, 8, 11, 14, 17])


def extra_count(tier):
    if tier not in (0, 1, 2):
        raise ValueError("tier must be 0, 1, or 2")
    return [0, SMALL, SMALL + LARGE][tier]


def initialize(path, tier):
    return np.concatenate([base.load(path), np.zeros(extra_count(tier))])


def save(path, parameters, tier):
    p = np.asarray(parameters, dtype="<f8")
    if (
        p.shape != (base.PARAMETERS + extra_count(tier),)
        or not np.isfinite(p).all()
        or np.max(abs(p)) > 4096
    ):
        raise ValueError("invalid expanded coefficients")
    base.validate_weights(p[: base.PARAMETERS])
    Path(path).write_bytes(
        b"CDCEX001" + struct.pack("<I", tier) + b"CDCSE001" + p.tobytes()
    )


def load(path):
    blob = Path(path).read_bytes()
    if len(blob) < 1372 or blob[:8] != b"CDCEX001" or blob[12:20] != b"CDCSE001":
        raise ValueError("invalid expanded model")
    tier = struct.unpack_from("<I", blob, 8)[0]
    if len(blob) != 1372 + extra_count(tier) * 8:
        raise ValueError("invalid expanded model length")
    p = np.frombuffer(blob, dtype="<f8", offset=20).copy()
    if not np.isfinite(p).all() or np.max(abs(p)) > 4096:
        raise ValueError("invalid expanded coefficients")
    base.validate_weights(p[: base.PARAMETERS])
    return p, tier


def design(features, tier):
    x = np.asarray(features, dtype=np.int64)
    z = np.zeros((len(x), extra_count(tier)), dtype=np.float64)
    if tier == 0:
        return z
    alive = x[:, :14].reshape(-1, 2, 7)
    pool = x[:, 14:28].reshape(-1, 2, 7)
    hidden = x[:, base.HIDDEN_COUNT]
    phase = np.searchsorted([0, 2, 5, 8, 12, 18, 25, 32], hidden)
    detailed = np.searchsorted([0, 4, 10, 20, 32], hidden)
    rows = np.arange(len(x))
    predators = (alive[:, ::-1, :] @ base.PREDATORS.T).astype(np.int64)
    for c in range(2):
        sign = 1 if c == 0 else -1
        for t in range(7):
            indices = phase * 45 + PREDATOR_OFFSETS[t] + predators[:, c, t]
            z[rows, indices] += sign * alive[:, c, t] * 7
            if tier == 2:
                for visibility, amount in enumerate(
                    [alive[:, c, t] - pool[:, c, t], pool[:, c, t]]
                ):
                    group = np.where(detailed == 0, 0, detailed * 2 + visibility - 1)
                    start = SMALL + (group * 7 + t) * 23
                    for u in range(7):
                        index = start + OFFSETS[u] + alive[:, 1 - c, u]
                        z[rows, index] += sign * amount
    return z


def linear(parameters, matrix):
    p = Tensor.wrap(parameters)
    return Tensor(matrix @ p.value, ((p, lambda gradient: matrix.T @ gradient),))


def evaluate(parameters, features, tier, matrix=None, quantized=True):
    p = Tensor.wrap(parameters)
    if p.value.shape != (base.PARAMETERS + extra_count(tier),):
        raise ValueError("wrong parameter shape")
    result = base.evaluate(p[: base.PARAMETERS], features, quantized=quantized)
    if tier:
        z = design(features, tier) if matrix is None else matrix
        scaled = p[base.PARAMETERS :] * 256
        if quantized:
            scaled = scaled.trunc()
        correction = linear(scaled, z) / (256 * 7)
        if quantized:
            correction = correction.trunc()
        result = (result + correction).clip(-9000, 9000)
    return result * (features[:, base.ASSIGNED] != 0)
