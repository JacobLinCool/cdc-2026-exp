"""Differentiable losses on exact evaluator scores; search labels stay constant."""

import numpy as np


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def value_loss(predicted, teacher, temperature=600):
    logits = predicted / temperature
    return float(
        np.mean(np.logaddexp(0, logits) - sigmoid(teacher / temperature) * logits)
    )


def root_objective(predicted, teacher, initial, variant):
    temperature = variant["temperature"]
    logits = predicted / temperature
    target = sigmoid(teacher / temperature)
    importance = (
        1 / (1 + (teacher / 600) ** 2) if variant["focus"] else np.ones_like(predicted)
    )
    importance /= importance.sum()
    drift = (predicted - initial) / 600
    loss = np.sum(importance * (np.logaddexp(0, logits) - target * logits)) + variant[
        "drift"
    ] * np.mean(drift * drift)
    derivative = importance * (sigmoid(logits) - target) / temperature + 2 * variant[
        "drift"
    ] * drift / (600 * len(predicted))
    return float(loss), derivative


def pair_objective(predicted, root_teacher, initial, drift_penalty):
    """The better root move has the *lower* opponent-side child evaluation."""
    count = len(predicted)
    child_target = sigmoid(-root_teacher.astype(float) / 600)
    logits = predicted / 600
    value = np.mean(np.logaddexp(0, logits) - child_target * logits)
    derivative = 0.1 * (sigmoid(logits) - child_target) / (600 * 2 * count)
    preference_logit = (predicted[:, 1] - predicted[:, 0]) / 200
    preference_target = sigmoid(
        (root_teacher[:, 0] - root_teacher[:, 1]).astype(float) / 200
    )
    ranking = np.mean(
        np.logaddexp(0, preference_logit) - preference_target * preference_logit
    )
    rank_derivative = (sigmoid(preference_logit) - preference_target) / (200 * count)
    derivative[:, 0] -= rank_derivative
    derivative[:, 1] += rank_derivative
    drift = (predicted - initial) / 600
    derivative += 2 * drift_penalty * drift / (600 * 2 * count)
    return (
        float(ranking + 0.1 * value + drift_penalty * np.mean(drift * drift)),
        derivative,
    )
