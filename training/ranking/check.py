#!/usr/bin/env python3
"""Validate preference sign, loss gradients and public-feature initial scores."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from train import DTYPE, objective, read_pairs
from model import DTYPE as VALUE_DTYPE, evaluate, load


def check_objective():
    rng = np.random.default_rng(18)
    predicted = rng.normal(0, 300, (9, 2))
    teacher = np.sort(rng.normal(0, 400, (9, 2)), axis=1)[:, ::-1]
    baseline = rng.normal(0, 300, (9, 2))
    errors = []
    for kind in ["value", "rank"]:
        _, analytic = objective(predicted, teacher, baseline, kind)
        for index in np.ndindex(predicted.shape):
            plus, minus = predicted.copy(), predicted.copy()
            plus[index] += 1e-3
            minus[index] -= 1e-3
            numerical = (
                objective(plus, teacher, baseline, kind)[0]
                - objective(minus, teacher, baseline, kind)[0]
            ) / 2e-3
            errors.append(abs(numerical - analytic[index]))
    assert max(errors) < 1e-9, max(errors)
    _, gradient = objective(
        np.zeros((1, 2)), np.array([[200, -200]]), np.zeros((1, 2)), "rank"
    )
    assert gradient[0, 0] > 0 and gradient[0, 1] < 0
    return max(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("initial", type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--predictor", type=Path)
    args = parser.parse_args()
    if args.candidate and not args.predictor:
        parser.error("--candidate requires --predictor")
    data = read_pairs(args.data)
    initial = load(args.initial)
    for start in range(0, len(data), 512):
        rows = data[start : start + 512]
        predicted = evaluate(initial, rows["features"].reshape(-1, 537)).value.reshape(
            -1, 2
        )
        np.testing.assert_array_equal(predicted, rows["baseline"])
        # Every candidate is a move, not a flip, and stays in the same hidden phase.
        assert np.all(rows["features"][:, 0, 532] == rows["features"][:, 1, 532])
    depths, counts = np.unique(data["depth"], return_counts=True)
    export_checks = {}
    if args.candidate:
        with TemporaryDirectory(prefix="cdc-ranking-parity-") as directory:
            records = np.empty(2 * len(data), dtype=VALUE_DTYPE)
            records["seed"] = np.repeat(data["seed"], 2)
            records["baseline"] = data["baseline"].reshape(-1)
            records["teacher"] = -data["teacher"].reshape(-1)
            records["outcome"] = 1  # Unused in numerical parity.
            records["features"] = data["features"].reshape(-1, 537)
            source = Path(directory) / "positions.bin"
            source.write_bytes(b"CDCDAT01" + records.tobytes())
            output = Path(directory) / "predictions.bin"
            for candidate in args.candidate:
                subprocess.run(
                    [str(args.predictor), str(candidate), str(source), str(output)],
                    check=True,
                )
                weights = load(candidate)
                expected = np.concatenate(
                    [
                        evaluate(
                            weights, records["features"][start : start + 512]
                        ).value
                        for start in range(0, len(records), 512)
                    ]
                )
                np.testing.assert_array_equal(
                    expected, np.fromfile(output, dtype="<i4")
                )
                export_checks[candidate.name] = dict(
                    positions=len(records), mismatches=0
                )
    print(
        json.dumps(
            dict(
                pairs=len(data),
                positions=2 * len(data),
                contributing_games=len(np.unique(data["seed"])),
                record_bytes=DTYPE.itemsize,
                first_seed=int(data["seed"].min()),
                last_seed=int(data["seed"].max()),
                teacher_depths={str(int(d)): int(c) for d, c in zip(depths, counts)},
                max_objective_gradient_error=check_objective(),
                initial_score_mismatches=0,
                root_to_child_preference_sign_verified=True,
                native_export_parity=export_checks,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
