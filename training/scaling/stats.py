"""Deal-paired descriptive intervals and matched scaling contrasts."""
import csv
from pathlib import Path
import numpy as np


def read_games(paths):
    rows = []
    for path in paths:
        with Path(path).open() as stream:
            rows.extend(csv.DictReader(stream))
    pairs = {}
    for row in rows:
        pairs.setdefault(int(row["seed"]), []).append(row)
    for seed, pair in pairs.items():
        if len(pair) != 2 or {r["a_first"] for r in pair} != {"true", "false"}:
            raise ValueError(f"incomplete or duplicated deal {seed}")
    return rows, {
        seed: sum(float(r["score"]) for r in pair) / 2
        for seed, pair in sorted(pairs.items())
    }


def interval(values):
    values = np.asarray(values)
    if len(values) < 2:
        raise ValueError("need independent paired deals")
    rng = np.random.default_rng(4271)
    boot = np.concatenate(
        [
            rng.choice(values, (1000, len(values)), replace=True).mean(axis=1)
            for _ in range(20)
        ]
    )
    return list(map(float, np.quantile(boot, [0.025, 0.975])))


def summarize(paths):
    rows, pairs = read_games(paths)
    scores = np.array([float(row["score"]) for row in rows])
    if not np.isin(scores, [0, 0.5, 1]).all():
        raise ValueError("invalid outcome")
    score = float(scores.mean())
    report = {
        "files": list(map(str, paths)),
        "games": len(rows),
        "independent_deals": len(pairs),
        "first_seed": min(pairs),
        "last_seed": max(pairs),
        "wins": int((scores == 1).sum()),
        "draws": int((scores == 0.5).sum()),
        "losses": int((scores == 0).sum()),
        "score": score,
        "paired_bootstrap_95_ci": interval(list(pairs.values())),
        "elo": float(400 * np.log10(score / (1 - score))) if 0 < score < 1 else None,
    }
    for side in ["a", "b"]:
        moves = sum(int(r[f"{side}_moves"]) for r in rows)
        elapsed = sum(float(r[f"{side}_ms"]) for r in rows)
        nodes = sum(int(r[f"{side}_nodes"]) for r in rows)
        report[f"{side}_ms_per_move"] = elapsed / moves
        report[f"{side}_nodes_per_move"] = nodes / moves
        report[f"{side}_nodes_per_second"] = nodes / elapsed * 1000
        report[f"{side}_mean_completed_depth"] = (
            sum(int(r[f"{side}_depth"]) for r in rows) / moves
        )
    return report


def contrast(paths, control_paths):
    _, candidate = read_games(paths)
    _, control = read_games(control_paths)
    if candidate.keys() != control.keys():
        raise ValueError("contrast deal sets differ")
    differences = np.array([candidate[seed] - control[seed] for seed in candidate])
    return {
        "score_difference": float(differences.mean()),
        "paired_bootstrap_95_ci": interval(differences),
        "interpretation": "Exploratory difference against the small-data condition on matched deals, with the same training-seed blocks",
    }
