"""Batched CDCSE001 evaluation; forward rounding mirrors Rust, backwards uses STE."""
from pathlib import Path
import numpy as np
from autodiff import Tensor

PARAMETERS = 169
FEATURES = 537
ALIVE, POOL = 0, 14
CHASE, DANGER = 42, 210
MOBILITY, PROTECT, TRAPPED, HIDDEN_NEAR = 378, 392, 406, 420
CONFINE, PARITY, CANNON_LINE = 434, 504, 518
HIDDEN_COUNT, NOCAP, ASSIGNED, REVEALED = 532, 533, 534, 535
SPLIT, NEAR, CONF, INTERACTIONS = 80, 136, 148, 153
DRAW_PLIES = 60
DTYPE = np.dtype(
    [
        ("seed", "<u8"),
        ("baseline", "<i4"),
        ("teacher", "<i4"),
        ("outcome", "<i2"),
        ("features", "<i2", (FEATURES,)),
    ]
)
ACTIVE = [
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    23,
    24,
    25,
    26,
    27,
    31,
    *range(32, 40),
]
ADJ = np.array(
    [
        [
            t != 6 if a == 0 else t in (0, 6) if a == 6 else False if a == 5 else a <= t
            for t in range(7)
        ]
        for a in range(7)
    ]
)
ANY = ADJ.copy()
ANY[5] = True
PREDATORS = ADJ.T.copy()
PREDATORS[:, 5] = False
np.fill_diagonal(PREDATORS, False)


def load(path):
    blob = Path(path).read_bytes()
    if blob[:8] != b"CDCSE001" or len(blob) != 8 + PARAMETERS * 8:
        raise ValueError("invalid CDCSE001 model")
    weights = np.frombuffer(blob, dtype="<f8", offset=8).copy()
    validate_weights(weights)
    return weights


def save(path, weights):
    validate_weights(weights)
    Path(path).write_bytes(b"CDCSE001" + np.asarray(weights, dtype="<f8").tobytes())


def validate_weights(weights):
    weights = np.asarray(weights)
    if (
        weights.shape != (PARAMETERS,)
        or not np.isfinite(weights).all()
        or np.max(np.abs(weights)) > 4096
    ):
        raise ValueError("invalid model coefficients")
    if np.any(weights[[22, 62]] != 1):
        raise ValueError("only the default exponential piece-value model is supported")
    if np.any((weights[[24, 26, 64, 66]] < 5) | (weights[[24, 26, 64, 66]] > 95)):
        raise ValueError("piece-value ratios must be in [5, 95]")


def read_data(path):
    with Path(path).open("rb") as source:
        if source.read(8) != b"CDCDAT01":
            raise ValueError("invalid dataset header")
    size = Path(path).stat().st_size - 8
    if size <= 0 or size % DTYPE.itemsize:
        raise ValueError("empty or truncated dataset")
    return np.memmap(path, dtype=DTYPE, mode="r", offset=8)


def interactions(x):
    phase = 1 - x[:, HIDDEN_COUNT] / 32
    progress = x[:, NOCAP] / DRAW_PLIES
    revealed = (x[:, ALIVE : ALIVE + 14] - x[:, POOL : POOL + 14]).reshape(-1, 2, 7)
    diff = revealed[:, 0, :] - revealed[:, 1, :]
    z = np.zeros((len(x), 16))
    z[:, :7] = diff * phase[:, None]
    z[:, 7:14] = diff * progress[:, None]
    z[:, 14] = (x[:, 5] * x[:, 13] - x[:, 12] * x[:, 6]) / 5
    z[:, 15] = (x[:, REVEALED] - x[:, REVEALED + 1]) * phase * progress
    return z


def evaluate(parameters, features, quantized=True):
    p = Tensor.wrap(parameters)
    x = np.asarray(features, dtype=np.float64)
    if p.value.shape != (PARAMETERS,) or x.ndim != 2 or x.shape[1] != FEATURES:
        raise ValueError("expected 169 parameters and a batch of 537-feature positions")

    def q(value):
        value = Tensor.wrap(value)
        return value.trunc() if quantized else value

    hidden = x[:, HIDDEN_COUNT]
    gate = ((8 - np.minimum(hidden, 8)) * 8)[:, None]
    w = q((q(p[:40]) * (64 - gate) + q(p[40:80]) * gate) / 64)
    split = q(
        (q(p[SPLIT : SPLIT + 28]) * (64 - gate) + q(p[SPLIT + 28 : NEAR]) * gate) / 64
    ).reshape(-1, 4, 7)
    alive = x[:, ALIVE : ALIVE + 14].reshape(-1, 2, 7)
    pool = x[:, POOL : POOL + 14].reshape(-1, 2, 7)
    opposite = alive[:, ::-1, :]
    counts = opposite @ PREDATORS.T
    exponents = counts.copy()
    exponents[:, :, 0] = opposite[:, :, 6]
    exponents[:, :, 5] = 0
    king = np.zeros((1, 1, 7))
    king[:, :, 0] = 1
    cannon = np.zeros((1, 1, 7))
    cannon[:, :, 5] = 1
    ratio = (
        w[:, 24].reshape(-1, 1, 1) * (1 - king) + w[:, 26].reshape(-1, 1, 1) * king
    ) / 100
    raw = (
        w[:, 23].reshape(-1, 1, 1) * ratio.pow(exponents) * (1 - cannon)
        + w[:, 25].reshape(-1, 1, 1) * cannon
    )
    value = q(raw) + w[:, 32:39].reshape(-1, 1, 7)
    no_prey = (opposite @ ANY.T) == 0
    value = (q(value / 3) * no_prey + value * (~no_prey)).clip(2, np.inf)
    score = (
        (alive - pool) * value + q(pool * value * w[:, 39].reshape(-1, 1, 1) / 100)
    ).sum(axis=2)
    total = alive.sum(axis=2)
    score = score - w[:, 19].reshape(-1, 1) * np.maximum(4 - total, 0) * (total <= 3)
    chase_w = w[:, 10] + q(w[:, 17] * (32 - hidden) / 32)
    # Sum distance histograms before applying the original term's truncation.
    for c in range(2):
        select = np.array([[1.0, 0.0] if c == 0 else [0.0, 1.0]])
        chase = (
            Tensor(x[:, CHASE + c * 84 : CHASE + (c + 1) * 84].reshape(-1, 7, 12))
            * q(p[NEAR:CONF]).reshape(1, 1, 12)
        ).sum(axis=2)
        danger = (
            Tensor(x[:, DANGER + c * 84 : DANGER + (c + 1) * 84].reshape(-1, 7, 12))
            * q(p[NEAR:CONF]).reshape(1, 1, 12)
        ).sum(axis=2)
        confine = (
            Tensor(x[:, CONFINE + c * 35 : CONFINE + (c + 1) * 35].reshape(-1, 7, 5))
            * q(p[CONF:INTERACTIONS]).reshape(1, 1, 5)
        ).sum(axis=2)
        term = q((chase * value[:, 1 - c, :]).sum(axis=1) * chase_w / 4096)
        term = term - q((danger * value[:, c, :]).sum(axis=1) * w[:, 11] / 4096)
        term = term - q(
            (confine * value[:, c, :]).sum(axis=1)
            * w[:, 31]
            * (32 - hidden)
            / (4096 * 32)
        )
        term = term + (
            Tensor(x[:, MOBILITY + c * 7 : MOBILITY + (c + 1) * 7])
            * (w[:, 12].reshape(-1, 1) + split[:, 0, :])
        ).sum(axis=1)
        term = term + (
            Tensor(x[:, PROTECT + c * 7 : PROTECT + (c + 1) * 7])
            * (w[:, 15].reshape(-1, 1) + split[:, 1, :])
        ).sum(axis=1)
        term = term - q(
            (
                Tensor(x[:, TRAPPED + c * 7 : TRAPPED + (c + 1) * 7])
                * value[:, c, :]
                * (w[:, 16].reshape(-1, 1) + split[:, 2, :])
            ).sum(axis=1)
            / 64
        )
        term = term + q(
            (
                Tensor(x[:, HIDDEN_NEAR + c * 7 : HIDDEN_NEAR + (c + 1) * 7])
                * (w[:, 21].reshape(-1, 1) + split[:, 3, :])
            ).sum(axis=1)
            * hidden
            / 32
        )
        term = term + (
            Tensor(x[:, CANNON_LINE + c * 7 : CANNON_LINE + (c + 1) * 7])
            * q(w[:, 20].reshape(-1, 1) * value[:, 1 - c, :] / 64)
        ).sum(axis=1)
        term = term + q(
            (
                Tensor(x[:, PARITY + c * 7 : PARITY + (c + 1) * 7]) * value[:, 1 - c, :]
            ).sum(axis=1)
            * w[:, 27]
            / (64 * (1 + x[:, REVEALED + c]))
        ) * (hidden <= 2)
        score = score + term.reshape(-1, 1) * select
    result = (
        score[:, 0]
        - score[:, 1]
        + w[:, 13]
        + q((Tensor(interactions(x)) * p[INTERACTIONS:]).sum(axis=1))
    )
    fraction = np.minimum(np.trunc((x[:, NOCAP] - 10) * 100 / (DRAW_PLIES - 10)), 100)
    factor = 100 - q((100 - w[:, 18].clip(0, 100)) * fraction / 100)
    result = q(result * factor / 100) * (x[:, NOCAP] > 10) + result * (
        x[:, NOCAP] <= 10
    )
    return result.clip(-9000, 9000) * (x[:, ASSIGNED] != 0)
