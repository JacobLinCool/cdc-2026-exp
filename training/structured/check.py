#!/usr/bin/env python3
"""Verify initialization, exported inference, and the smooth surrogate's gradient."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from autodiff import Tensor
from model import ACTIVE, evaluate, load, read_data


def predict(weights, data):
    return np.concatenate(
        [
            evaluate(weights, data["features"][i : i + 512]).value
            for i in range(0, len(data), 512)
        ]
    )


def check_gradient(weights, features):
    # Finite differences verify the smooth graph, NOT integer quantization's STE.
    rng = np.random.default_rng(441)
    probe = weights + rng.normal(0, 0.03, weights.shape)
    parameter = Tensor(probe)
    output = evaluate(parameter, features, quantized=False)
    seed = rng.normal(size=len(features))
    output.backward(seed)
    indices = [*ACTIVE, *[40 + i for i in ACTIVE], *range(80, 169)]
    errors = []
    for i in indices:
        eps = 1e-5
        plus, minus = probe.copy(), probe.copy()
        plus[i] += eps
        minus[i] -= eps
        numerical = (
            (
                evaluate(plus, features, False).value
                - evaluate(minus, features, False).value
            )
            * seed
        ).sum() / (2 * eps)
        actual = parameter.grad[i]
        error = abs(numerical - actual) / (1 + abs(numerical))
        if error > 1e-5:
            raise AssertionError((i, numerical, actual, error))
        errors.append(error)

    # A separate check specifies the intentionally nonmathematical STE rule.
    parameter = Tensor(np.array([-1.7, 2.9]))
    result = (parameter.trunc() * parameter).sum()
    result.backward()
    np.testing.assert_allclose(result.value, 7.5)
    np.testing.assert_allclose(parameter.grad, [-2.7, 4.9])
    return dict(
        gradient_parameters_checked=len(indices),
        max_relative_gradient_error=max(errors),
        gradient_check="finite difference on unquantized graph; separate STE rule check",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("initial", type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--predictor", type=Path)
    args = parser.parse_args()
    if args.candidate and not args.predictor:
        parser.error("--candidate requires the native --predictor executable")
    data = read_data(args.data)
    weights = load(args.initial)
    predictions = predict(weights, data)
    np.testing.assert_array_equal(predictions, data["baseline"])
    report = dict(
        initial_exact_positions=len(data),
        **check_gradient(weights, data["features"][:16])
    )
    if args.predictor:
        report["native_export_parity"] = {}
        with TemporaryDirectory(prefix="cdc-parity-") as directory:
            output = Path(directory) / "predictions.bin"
            for model in [args.initial, *args.candidate]:
                subprocess.run(
                    [str(args.predictor), str(model), str(args.data), str(output)],
                    check=True,
                )
                native = np.fromfile(output, dtype="<i4")
                np.testing.assert_array_equal(predict(load(model), data), native)
                report["native_export_parity"][model.name] = dict(
                    positions=len(data), mismatches=0
                )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
