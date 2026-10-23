"""Loss derivatives, preference orientation, and end-to-end real-graph gradients."""

import unittest
import numpy as np
from common import ROOT, FOLDER, plan
from data import read
from expanded import load, evaluate
from autodiff import Tensor
from objectives import root_objective, pair_objective


class ObjectiveTests(unittest.TestCase):
    def check_derivative(self, function, predicted):
        _, derivative = function(predicted)
        for index in np.ndindex(predicted.shape):
            plus, minus = predicted.copy(), predicted.copy()
            plus[index] += 0.001
            minus[index] -= 0.001
            finite = (function(plus)[0] - function(minus)[0]) / 0.002
            self.assertAlmostEqual(finite, derivative[index], delta=1e-9)

    def test_root_losses(self):
        predicted = np.array([-900.0, -100.0, 0.0, 180.0, 900.0])
        teacher = np.array([-2100.0, 40.0, 150.0, -300.0, 1600.0])
        initial = predicted + 80
        for variant in plan()["training"]["variants"]:
            self.check_derivative(
                lambda p: root_objective(p, teacher, initial, variant), predicted
            )

    def test_pair_loss_and_orientation(self):
        predicted = np.array([[-200.0, 300.0], [100.0, -50.0]])
        teacher = np.array([[500.0, 200.0], [-100.0, -110.0]])
        initial = predicted + 50
        self.check_derivative(
            lambda p: pair_objective(p, teacher, initial, 0.02), predicted
        )
        _, gradient = pair_objective(
            np.zeros((1, 2)), np.array([[200.0, -200.0]]), np.zeros((1, 2)), 0
        )
        self.assertGreater(
            gradient[0, 0], 0
        )  # Gradient descent lowers the better move's child score.
        self.assertLess(gradient[0, 1], 0)

    def test_combined_real_graph_gradient(self):
        weights, tier = load(ROOT / plan()["baseline"])
        records = read(FOLDER / "train-input.bin")
        features = records["features"][:24].astype(float)
        teacher = np.linspace(-800, 800, 8)
        pair_teacher = np.column_stack([teacher, teacher - 160])
        initial = evaluate(weights, features, tier, quantized=False).value
        variant = plan()["training"]["variants"][-1]

        def objective(scores):
            loss, grad = root_objective(scores[:8], teacher, initial[:8], variant)
            pair_loss, pair_grad = pair_objective(
                scores[8:].reshape(-1, 2),
                pair_teacher,
                initial[8:].reshape(-1, 2),
                variant["drift"],
            )
            return loss + variant["rank_weight"] * pair_loss, np.concatenate(
                [grad, variant["rank_weight"] * pair_grad.reshape(-1)]
            )

        parameter = Tensor(weights)
        predicted = evaluate(parameter, features, tier, quantized=False)
        predicted.backward(objective(predicted.value)[1])
        active = np.flatnonzero(abs(parameter.grad) > 1e-10)
        self.assertGreater(len(active), 10)
        for index in active[:: max(1, len(active) // 24)]:
            plus, minus = weights.copy(), weights.copy()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            finite = (
                objective(evaluate(plus, features, tier, quantized=False).value)[0]
                - objective(evaluate(minus, features, tier, quantized=False).value)[0]
            ) / 2e-4
            self.assertAlmostEqual(finite, parameter.grad[index], delta=1e-7)


if __name__ == "__main__":
    unittest.main()
