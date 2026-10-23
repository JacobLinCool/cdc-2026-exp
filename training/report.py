#!/usr/bin/env python3
"""Summarize paired arena games; bootstrap whole deals, never individual games."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--purpose", choices=["development", "holdout"], required=True)
    parser.add_argument("--time-ms", type=float, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--opponent-model", type=Path)
    parser.add_argument("--opponent-scale", type=float, default=1.0)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    pairs = defaultdict(list)
    for row in rows:
        pairs[int(row["seed"])].append(row)
    if len(pairs) < 2:
        raise ValueError("at least two complete deals required")
    for pair in pairs.values():
        if len(pair) != 2 or {r["a_first"] for r in pair} != {"true", "false"}:
            raise ValueError("every seed must have exactly one game in each seat")
    metadata = json.loads(args.model.with_suffix(".json").read_text())
    training_seeds = set(metadata["training_games"]) | set(metadata["validation_games"])
    if training_seeds.intersection(pairs):
        raise ValueError("arena deals overlap training or validation deals")
    if metadata["model_sha256"] != hashlib.sha256(args.model.read_bytes()).hexdigest():
        raise ValueError("model does not match its training metadata")
    opponent = {"kind": "baseline"}
    if args.opponent_model:
        opponent_metadata = json.loads(
            args.opponent_model.with_suffix(".json").read_text()
        )
        opponent_hash = hashlib.sha256(args.opponent_model.read_bytes()).hexdigest()
        if opponent_metadata["model_sha256"] != opponent_hash:
            raise ValueError("opponent model does not match its training metadata")
        opponent_seeds = set(opponent_metadata["training_games"]) | set(
            opponent_metadata["validation_games"]
        )
        if opponent_seeds.intersection(pairs):
            raise ValueError(
                "arena deals overlap opponent training or validation deals"
            )
        opponent = dict(
            kind="learned",
            model=str(args.opponent_model),
            model_sha256=opponent_hash,
            model_scale=args.opponent_scale,
        )
    if args.purpose == "holdout":
        frozen = json.loads((args.csv.parent / "frozen-candidate.json").read_text())
        protocol = frozen["protocol"]
        expected_seeds = set(
            range(
                protocol["holdout_first_seed"],
                protocol["holdout_first_seed"] + protocol["holdout_games"] // 2,
            )
        )
        if set(pairs) != expected_seeds or len(rows) != protocol["holdout_games"]:
            raise ValueError("incomplete or wrong held-out deal set")
        if (
            args.scale != frozen["model_scale"]
            or args.time_ms != protocol["holdout_time_ms"]
        ):
            raise ValueError("evaluation settings differ from the frozen candidate")
        if metadata["model_sha256"] != frozen["model_sha256"]:
            raise ValueError("model differs from the frozen candidate")
        if args.opponent_model and opponent not in protocol["opponents"]:
            raise ValueError("opponent differs from the frozen protocol")
    scores = np.array([float(row["score"]) for row in rows])
    if not np.isin(scores, [0, 0.5, 1]).all():
        raise ValueError("invalid score")
    means = np.array(
        [sum(float(r["score"]) for r in pair) / 2 for pair in pairs.values()]
    )
    rng = np.random.default_rng(4271)
    boot = np.concatenate(
        [
            rng.choice(means, (1000, len(means)), replace=True).mean(axis=1)
            for _ in range(20)
        ]
    )
    low, high = np.quantile(boot, [0.025, 0.975])
    score = float(scores.mean())
    report = dict(
        purpose=args.purpose,
        model=str(args.model),
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        model_scale=args.scale,
        opponent=opponent,
        time_ms_per_move=args.time_ms,
        csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        first_seed=min(pairs),
        last_seed=max(pairs),
        games=len(rows),
        independent_deals=len(pairs),
        wins=int((scores == 1).sum()),
        draws=int((scores == 0.5).sum()),
        losses=int((scores == 0).sum()),
        score=score,
        paired_bootstrap_95_ci=[float(low), float(high)],
        elo=float(400 * np.log10(score / (1 - score))) if 0 < score < 1 else None,
        exceeds_50_percent_at_95_confidence=bool(low > 0.5),
        model_ms_per_move=sum(float(r["a_ms"]) for r in rows)
        / sum(int(r["a_moves"]) for r in rows),
        opponent_ms_per_move=sum(float(r["b_ms"]) for r in rows)
        / sum(int(r["b_moves"]) for r in rows),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
