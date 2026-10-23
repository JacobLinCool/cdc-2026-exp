"""Describe frozen model differences on the existing validation positions."""

import json
from pathlib import Path
import sys

import numpy as np

from run import FOLDER, ROOT, digest, load_plan

sys.path.insert(0, str(ROOT / "training/scaling"))
from data import read, phases
from expanded import base, evaluate, load


def main():
    plan = load_plan()
    path = ROOT / "training/scaling/validation.bin"
    records = read(path)
    features = records["features"].astype(float)
    anchor = base.load(ROOT / plan["anchor"]["model"])

    def predict(weights, tier):
        return np.concatenate(
            [
                evaluate(weights, features[start : start + 256], tier).value
                for start in range(0, len(features), 256)
            ]
        )

    reference = predict(anchor, 0)
    result = {
        "interpretation": "Descriptive static-score differences on the existing validation distribution, not fresh evidence of playing strength or model selection.",
        "positions": len(records),
        "validation_sha256": digest(path),
        "models": {},
    }
    for seed in plan["training_seeds"]:
        small = None
        for item in plan["models"]:
            model = ROOT / f"{item['prefix']}-{seed}.cxe"
            weights, tier = load(model)
            metadata = json.loads(model.with_suffix(".json").read_text())
            assert digest(model) == metadata["model_sha256"]
            assert digest(path) == metadata["validation_data_sha256"]
            scores = predict(weights, tier)
            if tier == 0:
                small = scores
            base_only = predict(weights[:169], 0)
            extra = np.trunc(weights[169:] * 256)
            report = {
                "parameters": item["parameters"],
                "training_seed": seed,
                "model_sha256": digest(model),
                "extra_coefficients": len(extra),
                "nonzero_quantized_extra_coefficients": int(np.count_nonzero(extra)),
                "mean_absolute_score_change_from_anchor": float(
                    abs(scores - reference).mean()
                ),
                "mean_absolute_score_change_from_same_seed_small": float(
                    abs(scores - small).mean()
                ),
                "same_score_fraction_as_same_seed_small": float(
                    (scores == small).mean()
                ),
                "mean_absolute_extra_component": float(abs(scores - base_only).mean()),
                "max_absolute_extra_component": float(abs(scores - base_only).max()),
                "phase_mean_absolute_change_from_small": {
                    name: float(abs(scores - small)[phases(records) == phase].mean())
                    for phase, name in enumerate(
                        ["fully_revealed", "1_to_7_hidden", "8_plus_hidden"]
                    )
                },
            }
            result["models"][str(model.relative_to(ROOT))] = report
    (FOLDER / "model-diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    for capacity in [139, 499, 1948]:
        rows = [r for r in result["models"].values() if r["parameters"] == capacity]
        print(
            capacity,
            "parameters:",
            "mean score difference from small",
            round(
                np.mean(
                    [r["mean_absolute_score_change_from_same_seed_small"] for r in rows]
                ),
                4,
            ),
            "extra component",
            round(np.mean([r["mean_absolute_extra_component"] for r in rows]), 4),
        )


if __name__ == "__main__":
    main()
