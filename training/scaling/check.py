"""Initialization, analytic gradients, and independent native/export parity."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from data import read
from expanded import base, initialize, design, evaluate, load, save
from autodiff import Tensor
from train import objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument(
        "--initial", type=Path, default=Path("training/ranking/value.cse")
    )
    parser.add_argument(
        "--predictor", type=Path, default=Path("engine/target/release/scaling_predict")
    )
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    args = parser.parse_args()
    records = read(args.data)
    x = records["features"].astype(float)
    initial = base.load(args.initial)
    reference = np.concatenate(
        [base.evaluate(initial, x[i : i + 256]).value for i in range(0, len(x), 256)]
    )
    checks = []
    max_error = 0.0
    with tempfile.TemporaryDirectory(prefix="cdc-scaling-check-") as directory:
        directory = Path(directory)
        for tier in range(3):
            weights = initialize(args.initial, tier)
            for start in range(0, len(x), 256):
                np.testing.assert_array_equal(
                    evaluate(weights, x[start : start + 256], tier).value,
                    reference[start : start + 256],
                )
            # Differentiate the real-arithmetic graph. STE is tested as an explicit surrogate, not finite-differenced.
            small = x[:16]
            rng = np.random.default_rng(891 + tier)
            weights[169:] = rng.normal(0, 0.5, len(weights) - 169)
            parameter = Tensor(weights)
            output = evaluate(parameter, small, tier, quantized=False)
            _, derivative = objective(
                output.value, records["teacher"][:16], reference[:16]
            )
            output.backward(derivative)
            active = np.flatnonzero(abs(parameter.grad) > 1e-12)
            for index in active[:: max(1, len(active) // 24)]:
                losses = []
                for direction in [1, -1]:
                    perturbed = weights.copy()
                    perturbed[index] += direction * 1e-4
                    value = evaluate(perturbed, small, tier, quantized=False).value
                    losses.append(
                        objective(value, records["teacher"][:16], reference[:16])[0]
                    )
                error = abs((losses[0] - losses[1]) / 2e-4 - parameter.grad[index])
                max_error = max(max_error, float(error))
                assert error < 1e-7, (tier, index, error)
            path = directory / f"perturbed-{tier}.cxe"
            save(path, weights, tier)
            native = directory / f"native-{tier}.i32"
            subprocess.run(
                [str(args.predictor), str(path), str(args.data), str(native)],
                check=True,
            )
            prediction = np.concatenate(
                [
                    evaluate(weights, x[i : i + 256], tier).value
                    for i in range(0, len(x), 256)
                ]
            )
            np.testing.assert_array_equal(prediction, np.fromfile(native, dtype="<i4"))
            checks.append(
                {
                    "tier": tier,
                    "positions": len(x),
                    "initial_exact": True,
                    "perturbed_native_exact": True,
                }
            )
        candidates = []
        for index, path in enumerate(args.candidate):
            weights, tier = load(path)
            native = directory / f"candidate-{index}.i32"
            subprocess.run(
                [str(args.predictor), str(path), str(args.data), str(native)],
                check=True,
            )
            prediction = np.concatenate(
                [
                    evaluate(weights, x[i : i + 256], tier).value
                    for i in range(0, len(x), 256)
                ]
            )
            np.testing.assert_array_equal(prediction, np.fromfile(native, dtype="<i4"))
            candidates.append(
                {"path": str(path), "positions": len(x), "native_exact": True}
            )
    # These are root-side labels: a higher teacher value must push prediction upwards.
    _, sign = objective(np.array([0.0]), np.array([300.0]), np.array([0.0]))
    assert sign[0] < 0
    print(
        json.dumps(
            {
                "tiers": checks,
                "candidates": candidates,
                "max_gradient_error": max_error,
                "root_label_sign": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
