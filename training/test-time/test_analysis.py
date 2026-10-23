"""Check the uncertainty model used for paired scaling interactions."""

import unittest

import numpy as np

from analyze import bootstrap_indices, bootstrap_means, contrasts, elo


class PairedScalingStatistics(unittest.TestCase):
    def test_training_seeds_remain_fixed_strata(self):
        values = np.repeat([[0.0], [0.5], [1.0]], 32, axis=1)
        samples = bootstrap_means(values, bootstrap_indices(3, 32, 4274))
        np.testing.assert_array_equal(samples, np.full(len(samples), 0.5))

    def test_matching_removes_shared_deal_noise_from_interaction(self):
        base = np.random.default_rng(53).uniform(0.1, 0.7, (3, 32))
        indices = bootstrap_indices(3, 32, 4275)
        small_low = bootstrap_means(base, indices)
        small_high = bootstrap_means(base + 0.1, indices)
        large_low = bootstrap_means(base + 0.02, indices)
        large_high = bootstrap_means(base + 0.16, indices)
        interaction = (large_high - large_low) - (small_high - small_low)
        np.testing.assert_allclose(interaction, 0.04, atol=1e-15)

    def test_time_gains_use_log_budget_and_score_elo(self):
        budgets = [10, 40, 160]
        means = [0.4, 0.5, 0.6]
        reports = {str(t): {"score": s} for t, s in zip(budgets, means)}
        samples = {str(t): np.full(20_000, s) for t, s in zip(budgets, means)}
        report = contrasts(reports, samples, budgets)
        self.assertAlmostEqual(report["10->160"]["score_gain"]["estimate"], 0.2)
        expected = float(elo(0.6) - elo(0.4)) / 4
        self.assertAlmostEqual(
            report["10->160"]["elo_gain_per_doubling"]["estimate"], expected
        )
        self.assertEqual(
            report["10->160"]["elo_gain_per_doubling"]["paired_bootstrap_95_ci"],
            [expected, expected],
        )

    def test_invalid_strata_fail(self):
        with self.assertRaisesRegex(ValueError, "strata"):
            bootstrap_means(np.zeros((2, 32)), bootstrap_indices(3, 32, 4274))


if __name__ == "__main__":
    unittest.main()
