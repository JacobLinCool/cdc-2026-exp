"""Continue a frozen 499-parameter evaluator with one preregistered objective."""

import argparse
import json
import time
import numpy as np
from common import ROOT, FOLDER, digest, plan, save, verify
from dataset import load_split
from expanded import load, save as save_model, design, evaluate, base
from autodiff import Tensor
from objectives import root_objective, pair_objective, value_loss


def predict(weights, features, matrix):
    return np.concatenate(
        [
            evaluate(weights, features[i : i + 512], 1, matrix[i : i + 512]).value
            for i in range(0, len(features), 512)
        ]
    )


def parameterization(initial):
    mask = np.zeros_like(initial)
    mask[[*base.ACTIVE, *[40 + i for i in base.ACTIVE], *range(80, len(initial))]] = 1
    assert int(mask.sum()) == 499
    scale = np.clip(abs(initial) * 0.1, 1, 20)
    scale[80:153] = 4
    scale[153:169] = 5
    scale[169:] = 8
    scale[[24, 26, 64, 66]] = 1

    def weights(delta):
        result = initial + delta * scale * mask
        result[[24, 26, 64, 66]] = np.clip(result[[24, 26, 64, 66]], 5, 95)
        result[136:148] = np.clip(result[136:148], 0, 128)
        result[148:153] = np.clip(result[148:153], 0, 32)
        return result

    return mask, scale, weights


def pair_metrics(weights, records):
    features = records["features"].reshape(-1, base.FEATURES).astype(float)
    scores = predict(weights, features, design(features, 1)).reshape(-1, 2)
    gap = scores[:, 1] - scores[:, 0]
    loss, _ = pair_objective(scores, records["teacher"], scores, 0)
    return dict(
        loss=loss,
        agreement=float(np.mean((gap > 0) + 0.5 * (gap == 0))),
        strict_agreement=float(np.mean(gap > 0)),
        ties=float(np.mean(gap == 0)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = plan()
    frozen = json.loads((FOLDER / "training-freeze.json").read_text())
    verify(frozen["hashes"])
    variant = {v["name"]: v for v in config["training"]["variants"]}[args.condition]
    assert args.seed in config["training"]["seeds"]
    output = FOLDER / f"{args.condition}-{args.seed}.cxe"
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    started = time.monotonic()
    initial, tier = load(ROOT / config["baseline"])
    assert tier == 1 and len(initial) == 529
    train, labels, pairs, eligible = load_split(
        "train", config["data"]["teacher_nodes"]
    )
    valid, valid_labels, valid_pairs, valid_eligible = load_split(
        "validation", config["data"]["teacher_nodes"]
    )
    assert not set(train["seed"]).intersection(valid["seed"])
    x = train["features"][eligible].astype(float)
    vx = valid["features"][valid_eligible].astype(float)
    target = (train["teacher"] if variant["teacher"] == "old" else labels["score"])[
        eligible
    ].astype(float)
    valid_target = valid_labels["score"][valid_eligible].astype(float)
    assert min(len(x), len(vx)) > 100
    z, vz = design(x, 1), design(vx, 1)
    init_scores, init_valid = predict(initial, x, z), predict(initial, vx, vz)
    if variant["rank_weight"]:
        pair_x = pairs["features"].reshape(-1, base.FEATURES).astype(float)
        pair_z = design(pair_x, 1)
        init_pairs = predict(initial, pair_x, pair_z).reshape(-1, 2)
    mask, scale, weights = parameterization(initial)
    delta, first, second = [np.zeros_like(initial) for _ in range(3)]
    baseline_loss = value_loss(init_valid, valid_target)
    best_loss, best_epoch, best_weights = baseline_loss, 0, initial.copy()
    rng = np.random.default_rng(args.seed)
    order, history, steps = np.arange(len(x)), [], 0
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            features, matrix = x[batch], z[batch]
            if variant["rank_weight"]:
                pair_ids = rng.integers(0, len(pairs), size=128)
                child_ids = (pair_ids[:, None] * 2 + np.arange(2)).reshape(-1)
                features = np.concatenate([features, pair_x[child_ids]])
                matrix = np.concatenate([matrix, pair_z[child_ids]])
            parameter = Tensor(weights(delta))
            prediction = evaluate(parameter, features, 1, matrix)
            _, derivative = root_objective(
                prediction.value[: len(batch)],
                target[batch],
                init_scores[batch],
                variant,
            )
            if variant["rank_weight"]:
                _, pair_derivative = pair_objective(
                    prediction.value[len(batch) :].reshape(-1, 2),
                    pairs["teacher"][pair_ids],
                    init_pairs[pair_ids],
                    variant["drift"],
                )
                derivative = np.concatenate(
                    [derivative, variant["rank_weight"] * pair_derivative.reshape(-1)]
                )
            prediction.backward(derivative)
            gradient = (
                parameter.grad * scale + variant["weight_penalty"] * delta / mask.sum()
            ) * mask
            if not np.isfinite(gradient).all():
                raise ValueError("nonfinite gradient")
            steps += 1
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            delta -= (
                config["training"]["learning_rate"]
                * (first / (1 - 0.9**steps))
                / (np.sqrt(second / (1 - 0.999**steps)) + 1e-8)
            )
            delta = np.clip(delta, -variant["bound"], variant["bound"]) * mask
            delta = (weights(delta) - initial) / scale
        current = weights(delta)
        loss = value_loss(predict(current, vx, vz), valid_target)
        history.append(dict(epoch=epoch, validation_value_loss=loss))
        if loss < best_loss:
            best_loss, best_epoch, best_weights = loss, epoch, current.copy()
        if epoch % 10 == 0:
            print(
                f"{args.condition}-{args.seed} epoch={epoch} validation={loss:.8f} best={best_loss:.8f}",
                flush=True,
            )
    verify(frozen["hashes"])
    assert np.array_equal(best_weights[mask == 0], initial[mask == 0])
    save_model(output, best_weights, 1)
    metadata = dict(
        condition=args.condition,
        variant=variant,
        seed=args.seed,
        tier=1,
        parameters=529,
        trainable_parameters=499,
        epochs=epochs,
        training_positions=len(x),
        validation_positions=len(vx),
        training_pairs=len(pairs),
        validation_pairs=len(valid_pairs),
        training_games=sorted(map(int, np.unique(train["seed"]))),
        validation_games=sorted(map(int, np.unique(valid["seed"]))),
        initial_sha256=digest(ROOT / config["baseline"]),
        model_sha256=digest(output),
        training_freeze_sha256=digest(FOLDER / "training-freeze.json"),
        model_bytes=output.stat().st_size,
        best_epoch=best_epoch,
        baseline_validation_loss=baseline_loss,
        model_validation_loss=best_loss,
        initial_pair_metrics=pair_metrics(initial, valid_pairs),
        model_pair_metrics=pair_metrics(best_weights, valid_pairs),
        common_training_value_loss=value_loss(
            predict(best_weights, x, z), labels["score"][eligible]
        ),
        at_bound=int(
            np.count_nonzero(
                abs((best_weights - initial) / scale) >= variant["bound"] - 0.001
            )
        ),
        changed_parameters=int(np.count_nonzero(best_weights != initial)),
        max_normalized_change=float(np.max(abs((best_weights - initial) / scale))),
        gradient="Exact integer forward; explicit STE through truncation. Discrete features, search, and teacher targets are not differentiated.",
        history=history,
        elapsed_seconds=time.monotonic() - started,
        numpy_version=np.__version__,
    )
    save(output.with_suffix(".json"), metadata)
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in [
                    "condition",
                    "seed",
                    "best_epoch",
                    "model_validation_loss",
                    "elapsed_seconds",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
