#!/usr/bin/env python3
"""Fit scalar values or teacher preferences on the same quiet successor pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structured"))
from autodiff import Tensor
from model import ACTIVE, FEATURES, PARAMETERS, evaluate, load, save

DTYPE = np.dtype(
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


def read_pairs(path):
    with path.open("rb") as source:
        if source.read(8) != b"CDCRNK01":
            raise ValueError("invalid preference dataset")
    size = path.stat().st_size - 8
    if size <= 0 or size % DTYPE.itemsize:
        raise ValueError("empty or truncated preference dataset")
    data = np.memmap(path, dtype=DTYPE, offset=8, mode="r")
    gaps = data["teacher"][:, 0] - data["teacher"][:, 1]
    if np.any(data["depth"] < 3) or np.any((gaps < 10) | (gaps > 600)):
        raise ValueError("invalid teacher preferences")
    return data


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def objective(predicted, teacher, baseline, kind):
    """Return mean loss and its derivative with respect to child-side scores."""
    count = len(predicted)
    z = predicted / 600
    target = sigmoid(-teacher.astype(float) / 600)
    value_loss = np.mean(np.logaddexp(0, z) - target * z)
    value_gradient = (sigmoid(z) - target) / (600 * 2 * count)
    drift = (predicted - baseline) / 600
    loss = value_loss + 0.02 * np.mean(drift * drift)
    gradient = value_gradient + 0.04 * drift / (600 * 2 * count)
    if kind == "rank":
        # A better move gives the opponent a worse child-side evaluation.
        ranking_logit = (predicted[:, 1] - predicted[:, 0]) / 200
        target_preference = sigmoid((teacher[:, 0] - teacher[:, 1]).astype(float) / 200)
        loss += (
            np.mean(np.logaddexp(0, ranking_logit) - target_preference * ranking_logit)
            - 0.9 * value_loss
        )
        rank_gradient = (sigmoid(ranking_logit) - target_preference) / (200 * count)
        gradient -= 0.9 * value_gradient
        gradient[:, 0] -= rank_gradient
        gradient[:, 1] += rank_gradient
    return float(loss), gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("initial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--objective", choices=["value", "rank"], required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    data = read_pairs(args.data)
    initial = load(args.initial)
    rng = np.random.default_rng(args.seed)
    games = np.unique(data["seed"])
    if len(games) < 10:
        raise ValueError("at least ten contributing games required")
    rng.shuffle(games)
    validation_games = games[: len(games) // 5]
    valid = np.isin(data["seed"], validation_games)
    train_ids, valid_ids = np.flatnonzero(~valid), np.flatnonzero(valid)
    mask = np.zeros(PARAMETERS)
    mask[[*ACTIVE, *[40 + i for i in ACTIVE], *range(80, PARAMETERS)]] = 1
    scale = np.clip(abs(initial) * 0.1, 1, 20)
    scale[80:153] = 4
    scale[153:] = 5
    scale[[24, 26, 64, 66]] = 1
    delta, first, second = (np.zeros(PARAMETERS) for _ in range(3))
    step = 0

    def weights(update):
        result = initial + update * scale * mask
        result[[24, 26, 64, 66]] = np.clip(result[[24, 26, 64, 66]], 5, 95)
        result[136:148] = np.clip(result[136:148], 0, 128)
        result[148:153] = np.clip(result[148:153], 0, 32)
        return result

    def validation(parameters):
        losses, correct, total = 0.0, 0, 0
        for start in range(0, len(valid_ids), 512):
            ids = valid_ids[start : start + 512]
            predicted = evaluate(
                parameters, data["features"][ids].reshape(-1, FEATURES)
            ).value.reshape(-1, 2)
            loss, _ = objective(
                predicted, data["teacher"][ids], data["baseline"][ids], args.objective
            )
            losses += loss * len(ids)
            correct += np.count_nonzero(predicted[:, 1] > predicted[:, 0])
            total += len(ids)
        return losses / total, correct / total

    baseline_loss, baseline_accuracy = validation(initial)
    best_loss, best_accuracy = baseline_loss, baseline_accuracy
    best_weights, best_epoch = initial.copy(), 0
    history = []
    for epoch in range(args.epochs):
        rng.shuffle(train_ids)
        for start in range(0, len(train_ids), 256):
            ids = train_ids[start : start + 256]
            parameter = Tensor(weights(delta))
            output = evaluate(parameter, data["features"][ids].reshape(-1, FEATURES))
            _, seed = objective(
                output.value.reshape(-1, 2),
                data["teacher"][ids],
                data["baseline"][ids],
                args.objective,
            )
            output.backward(seed.reshape(-1))
            gradient = (parameter.grad * scale + 0.002 * delta / mask.sum()) * mask
            if not np.isfinite(gradient).all():
                raise ValueError("nonfinite training gradient")
            step += 1
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            delta -= (
                0.01
                * (first / (1 - 0.9**step))
                / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            )
            delta = np.clip(delta, -1, 1) * mask
            delta = (weights(delta) - initial) / scale
        loss, accuracy = validation(weights(delta))
        history.append(dict(loss=loss, pair_accuracy=accuracy))
        if loss < best_loss:
            best_loss, best_accuracy, best_epoch = loss, accuracy, epoch + 1
            best_weights = weights(delta).copy()
        if (epoch + 1) % 10 == 0:
            print(
                f"epoch={epoch+1} loss={loss:.7f} best={best_loss:.7f} pair_accuracy={accuracy:.4f}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, best_weights)
    metadata = dict(
        objective=args.objective,
        model_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        initial_sha256=hashlib.sha256(args.initial.read_bytes()).hexdigest(),
        model_bytes=args.output.stat().st_size,
        parameters=PARAMETERS,
        trainable_parameters=int(mask.sum()),
        training_games=sorted(map(int, games[len(games) // 5 :])),
        validation_games=sorted(map(int, validation_games)),
        pairs=len(data),
        train_pairs=len(train_ids),
        validation_pairs=len(valid_ids),
        best_epoch=best_epoch,
        baseline_validation_loss=baseline_loss,
        model_validation_loss=best_loss,
        baseline_pair_accuracy=baseline_accuracy,
        model_pair_accuracy=best_accuracy,
        seed=args.seed,
        command=sys.argv,
        numpy_version=np.__version__,
        history=history,
        gradient="exact quantized forward, STE for truncation; no differentiation through search or rules",
        objective_settings=dict(
            value_temperature=600,
            rank_temperature=200,
            rank_value_mix=0.1,
            drift_penalty=0.02,
            normalized_weight_bound=1,
        ),
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "objective",
                    "pairs",
                    "best_epoch",
                    "baseline_pair_accuracy",
                    "model_pair_accuracy",
                    "model_validation_loss",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
