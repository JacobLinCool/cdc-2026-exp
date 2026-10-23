"""Descriptive fitting diagnostics, independent of playing-strength selection."""

import json
import numpy as np
from common import ROOT, FOLDER, plan, digest, save, verify
from dataset import load_split
from expanded import load, design
from fit import predict
from objectives import value_loss, sigmoid


def main():
    verify(json.loads((FOLDER / "training-completion.json").read_text())["hashes"])
    config = plan()
    roots, labels, _, use = load_split("validation", config["data"]["teacher_nodes"])
    features = roots["features"][use].astype(float)
    matrix = design(features, 1)
    target = labels["score"][use].astype(float)
    probability = sigmoid(target / 600)
    entropy = float(
        np.mean(
            -probability * np.log(probability)
            - (1 - probability) * np.log1p(-probability)
        )
    )
    baseline, _ = load(ROOT / config["baseline"])
    initial = predict(baseline, features, matrix)
    initial_loss = value_loss(initial, target)
    models = []
    initial_agreements = set()
    for variant in config["training"]["variants"]:
        for seed in config["training"]["seeds"]:
            path = FOLDER / f"{variant['name']}-{seed}.cxe"
            parameters, _ = load(path)
            prediction = predict(parameters, features, matrix)
            metadata = json.loads(path.with_suffix(".json").read_text())
            initial_agreements.add(metadata["initial_pair_metrics"]["agreement"])
            models.append(
                dict(
                    condition=variant["name"],
                    seed=seed,
                    model_sha256=digest(path),
                    best_epoch=metadata["best_epoch"],
                    changed_parameters=metadata["changed_parameters"],
                    at_bound=metadata["at_bound"],
                    common_validation_loss=metadata["model_validation_loss"],
                    teacher_excess_loss=value_loss(prediction, target) - entropy,
                    excess_loss_reduction=(
                        initial_loss - value_loss(prediction, target)
                    )
                    / (initial_loss - entropy),
                    mean_absolute_score_change=float(
                        np.mean(abs(prediction - initial))
                    ),
                    p95_absolute_score_change=float(
                        np.quantile(abs(prediction - initial), 0.95)
                    ),
                    pair_agreement=metadata["model_pair_metrics"]["agreement"],
                )
            )
    assert len(initial_agreements) == 1
    save(
        FOLDER / "fit-diagnostics.json",
        dict(
            teacher_entropy=entropy,
            initial_loss=initial_loss,
            initial_excess_loss=initial_loss - entropy,
            initial_pair_agreement=initial_agreements.pop(),
            models=models,
            interpretation="Descriptive teacher-fitting diagnostics on the common validation roots; these do not measure playing strength. Excess loss subtracts the fixed teacher target entropy.",
        ),
    )


if __name__ == "__main__":
    main()
