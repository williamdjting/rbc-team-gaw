"""Small synthetic checks for diagnostic correctness; no production inputs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import run_experiment as experiment
from analyze_quiet_scores import daily_budgets


OBSERVED = []


class RecordingModel(HistGradientBoostingClassifier):
    def fit(self, X, y, **kwargs):
        OBSERVED.append(np.array(y))
        return super().fit(X, y, **kwargs)


class DiagnosticsTests(unittest.TestCase):
    def test_budget_ties_use_random_expectation_not_label_order(self):
        result = daily_budgets([1, 0, 1, 0], [.9, .5, .5, .1], [1, 2, 10])
        self.assertEqual([r["expected_tp"] for r in result], [1., 1.5, 2.])
        self.assertEqual(result[1]["cutoff_tie_size"], 2)
        self.assertEqual(result[2]["alerts"], 4)
        permuted = daily_budgets([0, 1, 0, 1], [.1, .5, .5, .9], [2])
        self.assertEqual(permuted[0]["expected_tp"], 1.5)

    def test_constant_scores_match_random_baseline(self):
        for result in daily_budgets([1, 0, 0, 0], [.1] * 4, [1, 2, 4]):
            self.assertAlmostEqual(result["expected_tp"], result["random_tp"])
        self.assertEqual(daily_budgets([], [], [10])[0]["alerts"], 0)

    def test_activity_groups_are_disjoint(self):
        frame = pd.DataFrame({"local_count_7d": [1, 0, 0, 1], "neighbour_count_7d": [1, 1, 0, 0]})
        self.assertEqual(experiment.groups(frame).tolist(), ["local_active", "neighbour_only", "quiet", "local_active"])

    def test_join_rejects_duplicate_and_missing_keys(self):
        left = pd.DataFrame({"key": [1, 2]})
        with self.assertRaises(ValueError):
            experiment.checked_merge(left, pd.DataFrame({"key": [1]}), ["key"], "many_to_one")
        with self.assertRaises(pd.errors.MergeError):
            experiment.checked_merge(left, pd.DataFrame({"key": [1, 1, 2]}), ["key"], "many_to_one")

    def test_metrics_and_empty_positive_group(self):
        m = experiment.metrics([1, 0, 1, 0], [.9, .8, .2, .1], .5)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 1))
        self.assertAlmostEqual(m["f1"], .5)
        self.assertTrue(np.isnan(experiment.metrics([0], [.1], .5)["ap"]))

    def test_paired_training_uses_same_labels_and_writes_keys(self):
        cems = experiment.module("cems_test08", "03a_exp_learned_cems_bc")
        def load(region, year):
            n = 300
            frame = pd.DataFrame({c: np.ones(n) for c in experiment.base.CELL + experiment.base.NEIGHBOUR + cems.CEMS})
            frame["target_date"] = pd.Timestamp(year, 6, 1)
            frame["h3_cell"] = [str(i) for i in range(n)]
            frame[experiment.TARGET] = np.arange(n) % 3 == 0
            return frame
        OBSERVED.clear()
        factory = lambda: RecordingModel(max_iter=2, min_samples_leaf=10)
        with tempfile.TemporaryDirectory() as temp, patch.object(experiment.base, "gradient_boosting", factory), patch.object(cems, "gradient_boosting", factory):
            models = experiment.fit_pair(["Kamloops"], 42, Path(temp), cems, load)
            np.testing.assert_array_equal(OBSERVED[0], OBSERVED[1])
            self.assertEqual(len(list(Path(temp).glob("sample_*.parquet"))), 4)
            for name, helper in [("firms", experiment.base), ("cems", cems)]:
                scores = helper.predict(models[name], load("Kamloops", 2023), experiment.FEATURE_SET)
                self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
