"""Scientific safeguards and reproducibility checks; standard-library unittest."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from telo_tabular import gp_core as g
from telo_tabular.data import ROOT, MeanPreprocessor, MaskGenerator, load_dataset
from telo_tabular.confidence import cross_fit, ReliabilityEstimator, confidence_metrics, shuffle_missing
from telo_tabular.experiment import read_config, prepare, confidence_stage, train_stage, evaluate_stage
from telo_tabular.models import PilotMLP, fixed_tree, predict, tensors, scores, train
from telo_tabular.storage import bind_run
from telo_tabular.search import search


class ScientificChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = read_config(ROOT / "configs" / "smoke.json")
        cls.x, cls.y, cls.meta = load_dataset("synthetic")

    def test_original_core_provenance(self):
        def definitions(path):
            out = {}
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    out[node.name] = ast.dump(node)
                elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    out[node.targets[0].id] = ast.dump(node)
            return out
        original = definitions(ROOT / "reference/gecco2026/pima_original.py")
        extracted = definitions(ROOT / "telo_tabular/gp_core.py")
        self.assertEqual(len(extracted), 36)
        for name, value in extracted.items():
            self.assertEqual(value, original[name], name)

    def test_supplied_datasets(self):
        for name, shape in (("pima", (768, 8)), ("wdbc", (569, 30))):
            x, y, meta = load_dataset(name)
            self.assertEqual(x.shape, shape)
            self.assertEqual(len(y), shape[0])
            self.assertEqual(len(meta["classes"]), 2)

    def test_mean_imputation_and_original_confidence(self):
        x = np.array([[1., 2.], [3., np.nan], [5., 8.]])
        p = MeanPreprocessor.fit(x)
        z, m, c = p.transform(np.array([[np.nan, 100.], [9., np.nan]]))
        np.testing.assert_allclose(p.mean, [3., 5.])
        np.testing.assert_allclose(c, [[1., 1.], [1., 2 / 3]])
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_equal(m, [[1., 0.], [0., 1.]])

    def test_missingness_deterministic_and_frozen(self):
        generator = MaskGenerator.fit(self.x[:100])
        before = copy.deepcopy(generator.__dict__)
        for mechanism in ("mcar", "mar", "mnar", "group"):
            a = generator.mask(self.x[100:], mechanism, .4, 42)
            np.testing.assert_array_equal(a, generator.mask(self.x[100:], mechanism, .4, 42))
            self.assertEqual(a.shape, self.x[100:].shape)
            if mechanism == "mar":
                self.assertFalse(a[:, 0].any())
        for key in before:
            np.testing.assert_array_equal(before[key], generator.__dict__[key])

    def test_channelprop_bounds_and_redundancy(self):
        layer = torch.nn.Linear(5, 3)
        prop = g.ChannelProp(layer)
        x, m = torch.randn(12, 5), torch.randint(0, 2, (12, 5)).float()
        c = 1 - .2 * m
        _, mm, cc = prop(x, m, c)
        torch.testing.assert_close(cc, 1 - .2 * mm)
        self.assertTrue(((cc >= 0) & (cc <= 1)).all())
        cc.square().sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertTrue(torch.isfinite(layer.weight.grad).all())

    def test_full_model_matches_original(self):
        tree = g.create_binary_node("multiply", g.create_input_node("value"),
                                    g.create_input_node("confidence"))
        torch.manual_seed(3)
        actual = PilotMLP(5, 2, [7, 4], tree)
        expected = g.ThreeChannelMLP(5, [7, 4], 2, tree)
        expected.load_state_dict(actual.network.state_dict())
        x, m, c = torch.randn(10, 5), torch.rand(10, 5), torch.rand(10, 5)
        torch.testing.assert_close(actual(x, m, c), expected(x, m, c))

    def test_no_target_value_in_reliability_context(self):
        model = ReliabilityEstimator(self.cfg["confidence"], 12)
        pre = MeanPreprocessor.fit(self.x)
        changed = self.x.copy()
        changed[:, 2] = 1e9
        np.testing.assert_array_equal(model._context(self.x, pre, 2),
                                      model._context(changed, pre, 2))

    def test_cross_fitting_disjoint_and_reproducible(self):
        masked, _ = MaskGenerator.fit(self.x).corrupt(self.x, "mcar", .3, 7)
        c, constant, final, audit, _ = cross_fit(masked, self.cfg["confidence"], 19)
        coverage = []
        for item in audit:
            fit, held = item["fit"], item["held_out"]
            self.assertFalse(set(fit) & set(held))
            coverage.extend(held)
            direct = ReliabilityEstimator(self.cfg["confidence"], 19 + item["fold"] * 101).fit(masked[fit])
            np.testing.assert_allclose(c[held], direct.predict(masked[held]))
            for inner in direct.inner_audit:
                self.assertFalse(set(inner["fit"]) & set(inner["held_out"]))
        self.assertEqual(sorted(coverage), list(range(len(masked))))
        self.assertTrue(np.all(c[np.isfinite(masked)] == 1))
        self.assertTrue(np.all((c >= 0) & (c <= 1)))
        self.assertTrue(np.isfinite(constant).all())
        np.testing.assert_allclose(final.pre.mean, MeanPreprocessor.fit(masked).mean)

    def test_test_rows_do_not_affect_preparation(self):
        cfg = copy.deepcopy(self.cfg)
        fixed = {"train": np.arange(100), "val": np.arange(100, 140), "test": np.arange(140, 180)}
        altered_x, altered_y = self.x.copy(), self.y.copy()
        altered_x[140:] = 1e8
        altered_y[140:] = 1 - altered_y[140:]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("telo_tabular.experiment.split_indices", return_value=fixed), patch(
                "telo_tabular.experiment._plot_reliability"):
                with patch("telo_tabular.experiment.load_dataset", return_value=(self.x, self.y, self.meta)):
                    a = prepare(cfg, "synthetic", 31, Path(tmp) / "a")
                with patch("telo_tabular.experiment.load_dataset", return_value=(altered_x, altered_y, self.meta)):
                    b = prepare(cfg, "synthetic", 31, Path(tmp) / "b")
            for split in ("train", "val"):
                for aa, bb in zip(a[split], b[split]):
                    np.testing.assert_array_equal(aa, bb)
                np.testing.assert_array_equal(a["learned_" + split], b["learned_" + split])
            self.assertEqual(a["validation_confidence"], b["validation_confidence"])

    def test_shuffle_preserves_observed_and_column_distribution(self):
        m = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
        c = np.array([[.1, 1], [.9, .3], [1, .7]])
        shuffled = shuffle_missing(c, m, 99)
        np.testing.assert_array_equal(shuffled[m == 0], c[m == 0])
        for j in range(2):
            np.testing.assert_array_equal(np.sort(shuffled[m[:, j] == 1, j]),
                                         np.sort(c[m[:, j] == 1, j]))

    def test_calibration_metric_known_example(self):
        clean = np.array([[0.], [2.]])
        p = MeanPreprocessor.fit(clean)
        c = np.array([[.25], [.75]])
        metrics = confidence_metrics(clean, np.full_like(clean, np.nan), c, p, .5)
        self.assertAlmostEqual(metrics["brier"], (.25**2 + .75**2) / 2)
        self.assertEqual(sum(b["n"] for b in metrics["bins"]), 2)

    def test_nonfinite_predictions_rejected(self):
        with self.assertRaises(FloatingPointError):
            scores(np.array([0, 1]), np.full((2, 2), np.nan))
        class Invalid(torch.nn.Module):
            def forward(self, x, m, c):
                return torch.full((len(x), 2), float("nan"))
        data = tensors((self.x, np.zeros_like(self.x), np.ones_like(self.x), self.y), "cpu")
        with self.assertRaises(FloatingPointError):
            predict(Invalid(), data)

    def test_changed_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bind_run(tmp, self.cfg)
            changed = copy.deepcopy(self.cfg)
            changed["confidence"]["tolerance_std"] = 1.
            with self.assertRaises(ValueError):
                bind_run(tmp, changed)

    def test_search_resume_matches_uninterrupted(self):
        def fake_train(*args, **kwargs):
            return {"valid": True, "best_val_accuracy": .6, "epochs_run": 1, "runtime_s": 0., "failure": None}
        cfg = copy.deepcopy(self.cfg["search"])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("telo_tabular.search.train", side_effect=fake_train):
                expected = search(lambda tree: None, None, None, cfg, 99, Path(tmp) / "whole", "test")
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return fake_train()
            with patch("telo_tabular.search.train", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    search(lambda tree: None, None, None, cfg, 99, Path(tmp) / "resume", "test")
            with patch("telo_tabular.search.train", side_effect=fake_train):
                actual = search(lambda tree: None, None, None, cfg, 99, Path(tmp) / "resume", "test")
            self.assertEqual([r["infix"] for r in actual["records"]],
                             [r["infix"] for r in expected["records"]])
            self.assertEqual(actual["best"]["tree"], expected["best"]["tree"])

    def test_synthetic_end_to_end_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = confidence_stage(self.cfg, tmp)
            self.assertEqual(len(rows), 4)
            trained = train_stage(self.cfg, tmp)
            self.assertEqual(len(trained), 6)
            self.assertTrue(any(r["status"] == "trained" and r["variant"] == "learned" for r in trained))
            self.assertFalse((Path(tmp) / "test_results.csv").exists())
            evaluated = evaluate_stage(tmp)
            self.assertTrue(evaluated)
            self.assertEqual(evaluated, evaluate_stage(tmp))
            masks = {}
            for r in evaluated:
                key = (r["mechanism"], r["nominal_rate"])
                masks.setdefault(key, set()).add(r["mask_fingerprint"])
            self.assertTrue(all(len(v) == 1 for v in masks.values()))
            self.assertTrue((Path(tmp) / "test_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()


class GeccoDatasetRegistryTests(unittest.TestCase):
    def test_all_registered_datasets_load_with_integer_labels_and_no_label_in_features(self):
        from telo_tabular.datasets_gecco import REGISTRY, load_gecco_dataset
        for name in REGISTRY:
            x, y, meta = load_gecco_dataset(name)
            self.assertEqual(len(x), len(y), name)
            self.assertEqual(x.dtype, np.float64, name)
            self.assertTrue(np.issubdtype(y.dtype, np.integer), name)
            self.assertEqual(set(np.unique(y)), set(range(len(meta["classes"]))), name)
            self.assertEqual(x.shape[1], len(meta["features"]), name)
            self.assertGreaterEqual(len(meta["classes"]), 2, name)
            self.assertFalse(np.isnan(x).all(axis=0).any(), name)   # all-NaN columns dropped

    def test_known_shapes(self):
        from telo_tabular.datasets_gecco import load_gecco_dataset
        x, y, m = load_gecco_dataset("pima"); self.assertEqual(x.shape, (768, 8)); self.assertEqual(m["classes"], ["0", "1"])
        x, y, m = load_gecco_dataset("wdbc"); self.assertEqual(x.shape, (569, 30)); self.assertEqual(m["classes"], ["B", "M"])
        x, y, m = load_gecco_dataset("housevote"); self.assertEqual(x.shape[1], 16); self.assertTrue(np.isnan(x).any())
        x, y, m = load_gecco_dataset("horse"); self.assertEqual(len(m["classes"]), 3)
