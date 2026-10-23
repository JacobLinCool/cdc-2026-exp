#!/usr/bin/env python3
"""Train the CDCNN001 residual evaluator using NumPy and game-disjoint validation."""
import argparse
import json
import pathlib
import hashlib
import sys
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--outcome-weight', type=float, default=0.25)
    parser.add_argument(
        '--quiet-only', action='store_true',
        help='fit only positions where neither side has an immediate capture',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.outcome_weight <= 1:
        parser.error('epochs must be positive and outcome-weight must be in [0, 1]')
    data = np.loadtxt(args.data, delimiter=',', dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 68 or not np.isfinite(data).all():
        raise ValueError('expected finite game, baseline, teacher, outcome, 64-feature rows')
    if not np.isin(data[:, 3], [0.0, 0.5, 1.0]).all() or not (data[:, 0] == np.floor(data[:, 0])).all():
        raise ValueError("invalid outcome or nonintegral game seed")
    source_rows = len(data)
    if args.quiet_only:
        data = data[np.max(data[:, 32:46], axis=1) == 0]
    rng = np.random.default_rng(args.seed)
    games = np.unique(data[:, 0])
    if len(games) < 10:
        raise ValueError('at least ten complete games required')
    rng.shuffle(games)
    valid = np.isin(data[:, 0], games[:max(1, len(games) // 5)])
    x = data[:, 4:].astype(np.float32)
    base = data[:, 1].astype(np.float32) / 600
    target = ((1 - args.outcome_weight) * sigmoid(data[:, 2] / 600) + args.outcome_weight * data[:, 3]).astype(np.float32)
    train_indices = np.flatnonzero(~valid)
    params = [
        rng.normal(0, 0.15, (16, 64)).astype(np.float32),
        np.zeros(16, np.float32),
        np.zeros(16, np.float32),
        np.zeros(1, np.float32),
    ]
    if args.quiet_only:
        # Unobserved threat channels must not retain random, untrained weights.
        params[0][:, 28:42] = 0
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step = 0
    best = float('inf')
    history = []

    def loss(mask):
        w, b, out, bias = params
        z = base[mask] + np.maximum(x[mask] @ w.T + b, 0) @ out + bias[0]
        return float(np.mean(np.logaddexp(0, z) - target[mask] * z))

    baseline_loss = float(np.mean(np.logaddexp(0, base[valid]) - target[valid] * base[valid]))
    for epoch in range(args.epochs):
        rng.shuffle(train_indices)
        for start in range(0, len(train_indices), 512):
            ids = train_indices[start:start + 512]
            xx = x[ids]
            w, b, out, bias = params
            pre = xx @ w.T + b
            hidden = np.maximum(pre, 0)
            residual = hidden @ out + bias[0]
            z = base[ids] + residual
            dz = (sigmoid(z) - target[ids]) / len(ids)
            # Residual regularization keeps unsupported corrections close to zero.
            dz += 0.002 * residual / len(ids)
            dh = dz[:, None] * out[None, :] * (pre > 0)
            gradients = [
                dh.T @ xx + 1e-5 * w,
                dh.sum(axis=0),
                hidden.T @ dz + 1e-5 * out,
                np.array([dz.sum()], np.float32),
            ]
            step += 1
            for p, grad, mm, vv in zip(params, gradients, m, v):
                mm *= 0.9
                mm += 0.1 * grad
                vv *= 0.999
                vv += 0.001 * grad * grad
                p -= 0.002 * (mm / (1 - 0.9 ** step)) / (np.sqrt(vv / (1 - 0.999 ** step)) + 1e-8)
        val = loss(valid)
        history.append(val)
        if val < best:
            best = val
            best_epoch = epoch + 1
            saved = [p.copy() for p in params]
        if (epoch + 1) % 10 == 0:
            print(f'epoch={epoch+1} validation_loss={val:.6f} baseline={baseline_loss:.6f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights = np.concatenate([p.ravel() for p in saved])
    assert weights.size == 1057 and np.isfinite(weights).all() and np.max(np.abs(weights)) <= 100
    args.output.write_bytes(b'CDCNN001' + weights.astype('<f4').tobytes())
    report = dict(
        quiet_only=args.quiet_only,
        source_rows=source_rows,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        numpy_version=np.__version__,
        command=sys.argv,
        seed=args.seed,
        training_games=sorted(map(int, games[len(games)//5:])),
        validation_games=sorted(map(int, games[:len(games)//5])),
        rows=len(data),
        train_rows=int((~valid).sum()),
        validation_rows=int(valid.sum()),
        outcome_weight=args.outcome_weight,
        best_epoch=best_epoch,
        baseline_validation_loss=baseline_loss,
        model_validation_loss=best,
        history=history,
        parameters=1057,
        model_bytes=args.output.stat().st_size,
    )
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    # Cross-language fixture for testing exported weights and feature order.
    w, b, out, bias = saved
    prediction = base[:16] * 600 + 600 * (np.maximum(x[:16] @ w.T + b, 0) @ out + bias[0])
    np.savetxt(args.output.with_suffix('.fixture.txt'), np.column_stack((data[:16, 1], x[:16], prediction)), delimiter=',', fmt='%.9g')
    print(json.dumps({k: report[k] for k in ('rows', 'best_epoch', 'baseline_validation_loss', 'model_validation_loss', 'model_bytes')}, indent=2))


if __name__ == '__main__':
    main()
