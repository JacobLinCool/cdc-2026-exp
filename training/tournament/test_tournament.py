"""Protocol and statistical-dependence checks, with no engine changes."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze import draw_indices, league_scores, sample_means
from run import CSV_FIELDS, configurations, load_plan, verify_csv


class ProtocolTests(unittest.TestCase):
    def test_complete_schedule_and_anchor_reuse(self):
        plan = load_plan()
        jobs = configurations(plan)
        self.assertEqual(len(jobs), 41)
        self.assertEqual(sum(j["games"] for j in jobs), 1312)
        self.assertEqual(len({j["name"] for j in jobs}), 41)
        rr = [j for j in jobs if j["family"] == "rr"]
        self.assertEqual(len(rr), 30)
        blocks = {}
        for job in jobs:
            blocks.setdefault(job["block"], set()).add(job["first_seed"])
            if job["family"] == "rr":
                self.assertNotEqual(job["a_id"], "original")
        self.assertEqual(len(blocks), 11)
        self.assertTrue(all(len(s) == 1 for s in blocks.values()))
        deals = [set(range(next(iter(s)), next(iter(s)) + 16)) for s in blocks.values()]
        self.assertEqual(sum(map(len, deals)), len(set.union(*deals)))
        for model in plan["models"][1:]:
            shared = next(
                j
                for j in rr
                if j["a_id"] == model["id"]
                and j["b_id"] == "original"
                and j["time_ms"] == 25
            )
            extras = [
                j for j in jobs if j["family"] == "anchor" and j["a_id"] == model["id"]
            ]
            self.assertEqual({j["time_ms"] for j in extras}, {100, 400})
            self.assertTrue(
                all(
                    j["block"] == shared["block"]
                    and j["first_seed"] == shared["first_seed"]
                    for j in extras
                )
            )
            for job in extras:
                ratio = float(job["b"].split("tmul=")[1])
                self.assertEqual(ratio * job["time_ms"], 25)

    def test_duplicate_seat_rejected(self):
        plan = load_plan()
        job = configurations(plan)[0]
        rows = []
        for i in range(32):
            rows.append(
                {
                    "seed": job["first_seed"] + i // 2,
                    "a_first": str(i % 2 == 0).lower(),
                    "score": 0.5,
                    "plies": 2,
                    "a_moves": 1,
                    "b_moves": 1,
                    "a_ms": 1,
                    "b_ms": 1,
                    "a_nodes": 2,
                    "b_nodes": 2,
                    "a_depth": 1,
                    "b_depth": 1,
                }
            )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "games.csv"

            def write():
                with path.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted(CSV_FIELDS))
                    writer.writeheader()
                    writer.writerows(rows)

            write()
            self.assertEqual(verify_csv(path, job)[1], [0.5] * 16)
            rows[1]["a_first"] = "true"
            write()
            with self.assertRaises(ValueError):
                verify_csv(path, job)


class StatisticsTests(unittest.TestCase):
    def test_reversed_orientation_and_zero_sum(self):
        samples = np.array([[0.2, 0.3], [0.6, 0.7], [0.8, 0.9]])
        actual = league_scores(
            samples, [("b", "a"), ("a", "c"), ("b", "c")], ["a", "b", "c"]
        )
        np.testing.assert_allclose(actual, [[0.7, 0.7], [0.5, 0.6], [0.3, 0.2]])
        np.testing.assert_allclose(actual.mean(axis=0), [0.5, 0.5])

    def test_incomplete_league_rejected(self):
        with self.assertRaises(ValueError):
            league_scores([0.5], [("a", "b")], ["a", "b", "c"])

    def test_shared_deal_resampling_preserves_matched_change(self):
        indices = draw_indices(2, 4, 1000, 13)
        values = np.array([0, 0.25, 0.5, 0.75])
        low = sample_means(values, indices[0])
        high = sample_means(values + 0.25, indices[0])
        np.testing.assert_allclose(high - low, 0.25)
        self.assertFalse(np.array_equal(indices[0], indices[1]))


if __name__ == "__main__":
    unittest.main()
