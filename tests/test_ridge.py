import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from ridge_baseline import fit_grouped_ridge


class RidgeContracts(unittest.TestCase):
    def test_group_isolation_train_only_scaling_and_prediction(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(6), 20)
        features = rng.normal(size=(len(groups), 4)) + groups[:, None] * 5
        targets = features @ np.array([[1., 2.], [2., -1.], [3., 0.5], [-1., 2.]])
        model, train, test, score = fit_grouped_ridge(features, targets, groups, alpha=0.01)
        self.assertFalse(set(groups[train]) & set(groups[test]))
        np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, features[train].mean(axis=0))
        self.assertGreater(score, 0.99)
        self.assertEqual(model.predict(features[test]).shape, targets[test].shape)

    def test_nonfinite_inputs_and_single_group_are_rejected(self):
        features = np.ones((4, 2))
        with self.assertRaises(ValueError):
            fit_grouped_ridge(features, features, np.zeros(4, dtype=int))
        features[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_grouped_ridge(features, features, np.array([0, 0, 1, 1]))

    def test_cli_accepts_numeric_arrays_and_rejects_pickle_objects(self):
        runner = Path(__file__).resolve().parents[1] / "ridge_baseline.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays, output = root / "inputs.npz", root / "output.json"
            rng = np.random.default_rng(7)
            features = rng.normal(size=(30, 3))
            np.savez(arrays, features=features, targets=features[:, :1], groups=np.repeat(np.arange(3), 10))
            result = subprocess.run([sys.executable, "-B", str(runner), "--arrays", str(arrays), "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(json.loads(output.read_text())), {"train_rows", "test_rows", "r2", "seed"})
            np.savez(arrays, features=np.array([{"bad": "object"}], dtype=object), targets=features[:, :1], groups=np.arange(30))
            result = subprocess.run([sys.executable, "-B", str(runner), "--arrays", str(arrays), "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Object arrays cannot be loaded", result.stderr)


if __name__ == "__main__":
    unittest.main()
