#!/usr/bin/env python3
"""Train exact-initialized evaluators with explicit quantization-aware surrogate gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from autodiff import Tensor
from model import ACTIVE, evaluate, load, read_data, save


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("initial", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--capacity", choices=["base", "split", "interactions"], required=True
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--outcome-weight", type=float, default=0)
    a = p.parse_args()
    if a.epochs < 1 or not 0 <= a.outcome_weight <= 1:
        p.error("invalid epochs or outcome weight")
    data = read_data(a.data)
    initial = load(a.initial)
    rng = np.random.default_rng(a.seed)
    games = np.unique(data["seed"])
    rng.shuffle(games)
    if len(games) < 10:
        raise ValueError("at least ten games are required")
    validation_games = games[: len(games) // 5]
    valid = np.isin(data["seed"], validation_games)
    train_ids = np.flatnonzero(~valid)
    valid_ids = np.flatnonzero(valid)
    mask = np.zeros(169)
    mask[ACTIVE] = 1
    mask[[40 + i for i in ACTIVE]] = 1
    if a.capacity != "base":
        mask[80:153] = 1
    if a.capacity == "interactions":
        mask[153:] = 1
    scale = np.maximum(np.minimum(abs(initial) * 0.1, 20), 1)
    scale[80:136] = 4
    scale[136:153] = 4
    scale[153:] = 5
    scale[[24, 26, 64, 66]] = 1
    target = (1 - a.outcome_weight) * sigmoid(
        data["teacher"].astype(float) / 600
    ) + a.outcome_weight * data["outcome"] / 2
    delta = np.zeros(169)
    m = np.zeros(169)
    v = np.zeros(169)
    step = 0

    def weights(d):
        w = initial + d * scale * mask
        w[[24, 26, 64, 66]] = np.clip(w[[24, 26, 64, 66]], 5, 95)
        w[136:148] = np.clip(w[136:148], 0, 128)
        w[148:153] = np.clip(w[148:153], 0, 32)
        return w

    def loss(w, ids):
        total = 0
        for start in range(0, len(ids), 1024):
            batch = ids[start : start + 1024]
            z = evaluate(w, data["features"][batch]).value / 600
            total += np.sum(np.logaddexp(0, z) - target[batch] * z)
        return float(total / len(ids))

    baseline = loss(initial, valid_ids)
    best = baseline
    best_epoch = 0
    best_weights = initial.copy()
    history = []
    for epoch in range(a.epochs):
        rng.shuffle(train_ids)
        for start in range(0, len(train_ids), 512):
            ids = train_ids[start : start + 512]
            parameter = Tensor(weights(delta))
            result = evaluate(parameter, data["features"][ids])
            z = result.value / 600
            result.backward((sigmoid(z) - target[ids]) / (600 * len(ids)))
            grad = (parameter.grad * scale + 0.002 * delta / mask.sum()) * mask
            if not np.isfinite(grad).all():
                raise ValueError("nonfinite gradient")
            step += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            delta -= (
                0.015
                * (m / (1 - 0.9**step))
                / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
            )
            # Bounded changes retain the initial expert's useful scale and keep export safe.
            delta = np.clip(delta, -2, 2) * mask
            delta = (weights(delta) - initial) / scale
        validation = loss(weights(delta), valid_ids)
        history.append(validation)
        if validation < best:
            best = validation
            best_epoch = epoch + 1
            best_weights = weights(delta).copy()
        if (epoch + 1) % 5 == 0:
            print(
                f"epoch={epoch+1} loss={validation:.7f} best={best:.7f} baseline={baseline:.7f}",
                flush=True,
            )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save(a.output, best_weights)
    report = dict(
        capacity=a.capacity,
        parameters=169,
        trainable_parameters=int(mask.sum()),
        model_bytes=a.output.stat().st_size,
        model_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
        initial_sha256=hashlib.sha256(a.initial.read_bytes()).hexdigest(),
        seed=a.seed,
        training_games=sorted(map(int, games[len(games) // 5 :])),
        validation_games=sorted(map(int, validation_games)),
        rows=len(data),
        train_rows=len(train_ids),
        validation_rows=len(valid_ids),
        best_epoch=best_epoch,
        baseline_validation_loss=baseline,
        model_validation_loss=best,
        history=history,
        outcome_weight=a.outcome_weight,
        command=sys.argv,
        numpy_version=np.__version__,
        gradient="exact forward truncation; straight-through derivative; no gradient through board rules or search",
    )
    a.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "capacity",
                    "best_epoch",
                    "baseline_validation_loss",
                    "model_validation_loss",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
