"""Check paired sharding and time interactions before launching the extension."""

import unittest
from unittest.mock import patch

import numpy as np

from experiment import base_run, cells, load_plan, read_cell, shards
from analyze_long import gain_difference


class ExtensionTests(unittest.TestCase):
    def test_same_deals_models_and_seating_at_1600(self):
        plan = load_plan()
        prior = base_run.load_plan()
        self.assertEqual(plan["models"], prior["models"])
        original = {j["name"]: j for j in base_run.configurations(prior)}
        new = cells(plan)
        self.assertEqual(len(new), 15)
        self.assertEqual(sum(j["games"] for j in new), 480)
        for job in new:
            old = original[job["name"].replace("t1600", "t400")]
            for key in ("a_id", "b_id", "block", "first_seed", "games", "threads"):
                self.assertEqual(job[key], old[key])
            self.assertEqual(job["time_ms"], 1600)
            self.assertEqual(
                job["b_time_ms"], 25 if job["family"] == "anchor" else 1600
            )
            if job["family"] == "anchor":
                self.assertEqual(float(job["b"].split("tmul=")[1]) * 1600, 25)

    def test_shards_preserve_complete_disjoint_pairs(self):
        plan = load_plan()
        jobs = shards(plan)
        self.assertEqual(len(jobs), 120)
        self.assertEqual(len({j["name"] for j in jobs}), 120)
        for cell in cells(plan):
            matching = [j for j in jobs if j["cell_name"] == cell["name"]]
            self.assertEqual(len(matching), 8)
            seeds = [
                seed
                for j in matching
                for seed in range(j["first_seed"], j["first_seed"] + j["games"] // 2)
            ]
            self.assertEqual(
                seeds, list(range(cell["first_seed"], cell["first_seed"] + 16))
            )
            self.assertTrue(
                all(
                    j["games"] == 4 and j["a"] == cell["a"] and j["b"] == cell["b"]
                    for j in matching
                )
            )

    def test_reassembly_restores_deal_order_and_rejects_missing_shard(self):
        plan = load_plan()
        jobs = shards(plan)
        cell = cells(plan)[0]

        def reader(job):
            return [None] * job["games"], list(
                range(job["first_seed"], job["first_seed"] + job["games"] // 2)
            )

        with patch("experiment.read_shard", side_effect=reader):
            rows, values, sources = read_cell(cell, list(reversed(jobs)))
            self.assertEqual(len(rows), 32)
            self.assertEqual(len(sources), 8)
            self.assertEqual(
                values, list(range(cell["first_seed"], cell["first_seed"] + 16))
            )
            with self.assertRaises(ValueError):
                read_cell(cell, jobs[1:])

    def test_gain_interaction_uses_within_model_covariance(self):
        x = np.linspace(0.2, 0.6, 100)
        a_samples = {"400": x, "1600": x + 0.2}
        b_samples = {"400": x[::-1], "1600": x[::-1] + 0.05}
        result = gain_difference(
            {"400": 0.4, "1600": 0.6},
            {"400": 0.4, "1600": 0.45},
            a_samples,
            b_samples,
            400,
            1600,
        )
        self.assertAlmostEqual(result["estimate"], 0.15)
        np.testing.assert_allclose(result["paired_bootstrap_95_ci"], [0.15, 0.15])


if __name__ == "__main__":
    unittest.main()
