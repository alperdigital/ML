"""Regressions for Ames data semantics and split-local experiment caches."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from base_stage import search_file
from design import load_data
from recipe_cache import cached_oof
from run import foundation_availability, mean_fold_rmsle


ROOT = Path(__file__).resolve().parents[1]


class ExperimentInputTests(unittest.TestCase):
    def test_registered_csv_preserves_masonry_absence(self):
        raw = load_data(ROOT / "data/train.csv")
        self.assertEqual(raw.MasVnrType.eq("None").sum(), 864)
        self.assertEqual(raw.MasVnrType.isna().sum(), 8)

    def test_foundation_worker_reads_literal_and_numeric_looking_categories(self):
        # Exercise the worker's actual CSV read and target transform while
        # replacing only the optional heavyweight dependency and its weights.
        modules = {name: MagicMock() for name in ("torch", "tabpfn", "tabpfn.constants")}
        with patch.dict(sys.modules, modules), tempfile.TemporaryDirectory() as td:
            worker = runpy.run_path(str(ROOT / "ames-v3/tabpfn_worker.py"))
            main = worker["main"]
            checkpoint = b"test-checkpoint"
            with patch.dict(main.__globals__, CHECKPOINT_SHA256=hashlib.sha256(checkpoint).hexdigest(),
                            version=lambda _: "10.2.0"):
                folder = Path(td)
                (folder / "tabpfn-v2-regressor.ckpt").write_bytes(checkpoint)
                (folder / "categories.txt").write_text("veneer\ncategory\n")
                csv = "veneer,category\nNone,001\n,02\n"
                for filename in ("training.csv", "evaluation.csv"):
                    (folder / filename).write_text(csv)
                np.save(folder / "target.npy", [10., 20.])
                model = modules["tabpfn"].TabPFNRegressor.create_default_for_version.return_value
                model.predict.return_value = np.log1p([15., 25.])
                main(td, td)
                for frame in (model.fit.call_args.args[0], model.predict.call_args.args[0]):
                    self.assertEqual(frame.veneer.iloc[0], "None")
                    self.assertTrue(pd.isna(frame.veneer.iloc[1]))
                    self.assertEqual(frame.category.tolist(), ["001", "02"])
                np.testing.assert_allclose(model.fit.call_args.args[1], np.log1p([10., 20.]))
                np.testing.assert_allclose(np.load(folder / "output.npy"), [15., 25.])

    def test_fixed_recipe_cache_changes_with_targets_or_fitting_indices(self):
        x = pd.DataFrame({"feature": [2., 4., 8.]})
        y = pd.Series([10., 20., 30.])
        split = [(np.array([0, 1]), np.array([2]))]
        with tempfile.TemporaryDirectory() as td, patch("models_v3.crossfit", return_value=np.arange(3)) as fit:
            spec = {"model": "ridge"}
            for prices, folds in [(y, split), (y.copy(), split), (y + 1, split),
                                  (y, [(np.array([0]), np.array([2]))])]:
                cached_oof(x, prices, folds, "ridge", spec, td)
            self.assertEqual(fit.call_count, 3)

    def test_nested_cache_changes_with_masonry_or_price_even_for_same_ids(self):
        train = pd.DataFrame({"Id": [1, 2, 3], "MasVnrType": ["None"] * 3, "SalePrice": [1., 2., 3.]})
        ref = search_file("/tmp", "test", train)
        self.assertEqual(ref, search_file("/tmp", "test", train.copy()))
        self.assertNotEqual(ref, search_file("/tmp", "test", train.assign(MasVnrType=np.nan)))
        self.assertNotEqual(ref, search_file("/tmp", "test", train.assign(SalePrice=[3., 2., 1.])))

    def test_inner_rmsle_pairs_column_targets_with_each_price(self):
        targets = pd.DataFrame({"SalePrice": [100., 500., 300.]})
        folds = [(np.array([0, 2]), np.array([1])), (np.array([1]), np.array([0, 2]))]
        self.assertEqual(mean_fold_rmsle(targets, [100., 500., 300.], folds), 0.)
        with self.assertRaises(ValueError):
            mean_fold_rmsle(targets, [[100., 500., 300.]], folds)

    def test_optional_worker_is_checked_before_selection(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(foundation_availability()["available"])
        with patch.dict(os.environ, TABPFN_MODEL_CACHE_DIR="/example"):
            with patch("run.runtime_signature", return_value=json.dumps(["3.12", "arm64", {"tabpfn": "1"}])):
                self.assertFalse(foundation_availability()["available"])
            runtime = json.dumps(["3.12", "arm64", {"tabpfn": "10.2.0"}])
            with patch("run.runtime_signature", return_value=runtime), patch("run.subprocess.run") as probe:
                probe.return_value = SimpleNamespace(returncode=1, stderr="ImportError: torch")
                self.assertEqual(foundation_availability(), {"available": False, "reason": "ImportError: torch"})
                probe.return_value = SimpleNamespace(returncode=0)
                self.assertTrue(foundation_availability()["available"])
            with patch("run.runtime_signature", side_effect=ValueError("fingerprint mismatch")):
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    foundation_availability()
