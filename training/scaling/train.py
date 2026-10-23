"""One predeclared scaling condition and training seed; frozen game validation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from data import read, phases, quiet, deep_valid
from expanded import initialize, save, design, evaluate, base
from autodiff import Tensor


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def objective(predicted, teacher, initial):
    target = sigmoid(teacher / 600)
    logits = predicted / 600
    value = np.mean(np.logaddexp(0, logits) - target * logits)
    drift = (predicted - initial) / 600
    loss = value + 0.02 * np.mean(drift * drift)
    gradient = (sigmoid(logits) - target + 0.04 * drift) / (600 * len(predicted))
    return float(loss), gradient


def balanced_ids(records, eligible, count):
    rng = np.random.default_rng(2431)
    groups = [
        np.flatnonzero(eligible & (phases(records) == phase)) for phase in range(3)
    ]
    for group in groups:
        rng.shuffle(group)
    sizes = [0, 0, 0]
    while sum(sizes) < count:
        available = [i for i in range(3) if sizes[i] < len(groups[i])]
        if not available:
            raise ValueError("insufficient coverage positions")
        chosen = min(available, key=lambda i: (sizes[i], i))
        sizes[chosen] += 1
    return np.concatenate([group[:n] for group, n in zip(groups, sizes)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    folder = args.directory
    plan = json.loads((folder / "experiment-plan.json").read_text())
    variants = {v["name"]: v for v in plan["training"]["variants"]}
    variant = variants[args.condition]
    if args.seed not in plan["training"]["seeds"]:
        raise ValueError("training seed outside plan")
    output = args.output or folder / f"{args.condition}-{args.seed}.cxe"
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    train = read(folder / "train.bin")
    valid = read(folder / "validation.bin")
    # Stable row ordering makes fitting reproducible despite concurrent collection.
    train = train[np.lexsort((train["ply"], train["seed"]))]
    valid = valid[np.lexsort((valid["ply"], valid["seed"]))]
    if set(train["seed"]).intersection(valid["seed"]):
        raise ValueError("training and validation games overlap")
    first = plan["data"]["training_first_seed"]
    small = train["seed"] < first + plan["data"]["small_training_games"]
    shared_quiet = quiet(train) & (~small | deep_valid(train))
    reference = np.flatnonzero(small & shared_quiet)
    eligible = (train["seed"] >= first) & (train["seed"] < first + variant["games"])
    if variant["coverage"] == "balanced":
        ids = balanced_ids(train, eligible, len(reference))
    else:
        ids = np.flatnonzero(eligible & shared_quiet)
    valid_ids = np.flatnonzero(quiet(valid) & deep_valid(valid))
    if len(ids) < 100 or len(valid_ids) < 100:
        raise ValueError("insufficient fitting or validation positions")
    x = train["features"][ids].astype(float)
    vx = valid["features"][valid_ids].astype(float)
    teacher_key = "deep_teacher" if variant["teacher_ms"] == 640 else "teacher"
    targets = train[teacher_key][ids].astype(float)
    # A single, deeper validation target serves every condition and seed.
    vtargets = valid["deep_teacher"][valid_ids].astype(float)
    initial = initialize(plan["initial_model"], variant["tier"])
    tier = variant["tier"]
    z, vz = design(x, tier), design(vx, tier)

    def predict(parameters, features, matrix):
        return np.concatenate(
            [
                evaluate(
                    parameters, features[i : i + 512], tier, matrix[i : i + 512]
                ).value
                for i in range(0, len(features), 512)
            ]
        )

    initial_scores = predict(initial, x, z)
    initial_valid = predict(initial, vx, vz)
    mask = np.zeros_like(initial)
    mask[[*base.ACTIVE, *[40 + i for i in base.ACTIVE], *range(80, len(initial))]] = 1
    scale = np.clip(abs(initial) * 0.1, 1, 20)
    scale[80:153] = 4
    scale[153:169] = 5
    scale[169:] = 8
    scale[[24, 26, 64, 66]] = 1
    delta, first_moment, second_moment = [np.zeros_like(initial) for _ in range(3)]

    def parameters(update):
        p = initial + update * scale * mask
        p[[24, 26, 64, 66]] = np.clip(p[[24, 26, 64, 66]], 5, 95)
        p[136:148] = np.clip(p[136:148], 0, 128)
        p[148:153] = np.clip(p[148:153], 0, 32)
        return p

    def value_loss(predicted, scores):
        q = predicted / 600
        return float(np.mean(np.logaddexp(0, q) - sigmoid(scores / 600) * q))

    baseline_loss = value_loss(initial_valid, vtargets)
    best_loss, best_weights, best_epoch = baseline_loss, initial.copy(), 0
    history, steps = [], 0
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(x))
    epochs = args.epochs or plan["training"]["epochs"]
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), 256):
            batch = order[start : start + 256]
            parameter = Tensor(parameters(delta))
            prediction = evaluate(parameter, x[batch], tier, z[batch])
            _, derivative = objective(
                prediction.value, targets[batch], initial_scores[batch]
            )
            prediction.backward(derivative)
            gradient = (parameter.grad * scale + 0.002 * delta / mask.sum()) * mask
            if not np.isfinite(gradient).all():
                raise ValueError("nonfinite gradient")
            steps += 1
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
            delta -= (
                0.01
                * (first_moment / (1 - 0.9**steps))
                / (np.sqrt(second_moment / (1 - 0.999**steps)) + 1e-8)
            )
            delta = np.clip(delta, -variant["bound"], variant["bound"]) * mask
            delta = (parameters(delta) - initial) / scale
        current = parameters(delta)
        vpred = predict(current, vx, vz)
        loss = value_loss(vpred, vtargets)
        train_loss = value_loss(predict(current, x, z), targets)
        history.append(
            {
                "epoch": epoch + 1,
                "validation_value_loss": loss,
                "training_value_loss": train_loss,
            }
        )
        if loss < best_loss:
            best_loss, best_weights, best_epoch = loss, current.copy(), epoch + 1
        if (epoch + 1) % 10 == 0:
            print(
                f"epoch={epoch+1} validation={loss:.8f} best={best_loss:.8f}",
                flush=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, best_weights, tier)
    # Full-distribution, 160ms phase diagnostics are independent of checkpoint selection.
    all_x = valid["features"].astype(float)
    all_z = design(all_x, tier)
    all_pred = predict(best_weights, all_x, all_z)
    all_initial = predict(initial, all_x, all_z)
    phase_metrics = []
    for phase in range(3):
        use = phases(valid) == phase
        phase_metrics.append(
            {
                "phase": phase,
                "positions": int(use.sum()),
                "initial_value_loss": value_loss(
                    all_initial[use], valid["teacher"][use]
                ),
                "model_value_loss": value_loss(all_pred[use], valid["teacher"][use]),
            }
        )
    metadata = {
        "condition": args.condition,
        "variant": variant,
        "seed": args.seed,
        "tier": tier,
        "parameters": len(initial),
        "trainable_parameters": int(mask.sum()),
        "observed_nonzero_extra_features": int(np.any(z != 0, axis=0).sum()),
        "training_games": sorted(map(int, np.unique(train["seed"][ids]))),
        "validation_games": sorted(map(int, np.unique(valid["seed"]))),
        "training_positions": len(ids),
        "validation_positions": len(valid_ids),
        "training_phase_counts": np.bincount(phases(train)[ids], minlength=3).tolist(),
        "training_row_ids_sha256": hashlib.sha256(
            ids.astype("<u8").tobytes()
        ).hexdigest(),
        "training_data_sha256": hashlib.sha256(
            (folder / "train.bin").read_bytes()
        ).hexdigest(),
        "validation_data_sha256": hashlib.sha256(
            (folder / "validation.bin").read_bytes()
        ).hexdigest(),
        "initial_sha256": hashlib.sha256(
            Path(plan["initial_model"]).read_bytes()
        ).hexdigest(),
        "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "model_bytes": output.stat().st_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "baseline_validation_loss": baseline_loss,
        "model_validation_loss": best_loss,
        "phase_metrics": phase_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "numpy_version": np.__version__,
        "gradient": "Exact integer forward; explicit STE through truncation; discrete features and search are not differentiated",
        "validation_target": "640ms teacher, same quiet positions for every condition",
        "at_bound": int(
            np.count_nonzero(
                abs((best_weights - initial) / scale) >= variant["bound"] - 0.001
            )
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
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
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
