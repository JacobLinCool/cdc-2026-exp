"""Protect independent deal resampling and the declared selection rule."""

import unittest
import numpy as np
from evaluate import stratified_interval, choose, development_jobs, holdout_jobs
from common import plan


class EvaluationTests(unittest.TestCase):
    def test_seed_strata_stay_fixed(self):
        blocks = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        self.assertEqual(stratified_interval(blocks), [0.5, 0.5])

    def test_selection_uses_family_score_then_validation(self):
        reports = {
            "a": dict(score=0.51, mean_validation_loss=0.48),
            "b": dict(score=0.52, mean_validation_loss=0.49),
        }
        metadata = [
            dict(condition=c, seed=s, model_validation_loss=loss, model_sha256="test")
            for c, s, loss in [("a", 53, 0.48), ("b", 53, 0.49), ("b", 71, 0.48)]
        ]
        selected = choose(reports, metadata)
        self.assertTrue(selected["advanced"])
        self.assertEqual((selected["condition"], selected["training_seed"]), ("b", 71))
        reports["a"]["score"] = reports["b"]["score"] = 0.5
        self.assertFalse(choose(reports, metadata)["advanced"])
        self.assertEqual(choose(reports, metadata)["condition"], "a")

    def test_game_budgets_and_seed_separation(self):
        config = plan()
        development = development_jobs(config)
        holdout = holdout_jobs(
            config,
            dict(advanced=True, condition="deep-value", training_seed=53, model="test"),
        )
        self.assertEqual(sum(j["games"] for j in development), 1152)
        self.assertEqual(sum(j["games"] for j in holdout), 1600)
        development_deals = {
            s
            for j in development
            for s in range(j["first_seed"], j["first_seed"] + j["games"] // 2)
        }
        for j in holdout:
            self.assertFalse(
                development_deals.intersection(
                    range(j["first_seed"], j["first_seed"] + j["games"] // 2)
                )
            )


if __name__ == "__main__":
    unittest.main()
