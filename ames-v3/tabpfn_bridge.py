"""Fit/predict bridge for the pinned CPU foundation model, separate from V2 wheels."""
import hashlib
import os
import json
import time
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CHECKPOINT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"


class FoundationRegressor:
    """Context-learning estimator. Store only the training context until prediction."""
    def __init__(self, X, y):
        self.training = X.drop(columns=["Id", "SalePrice"], errors="ignore").copy()
        self.target = np.asarray(y, dtype=float).copy()

    def predict(self, X):
        context = self.training
        evaluation = X.loc[:, context.columns]
        python = os.environ.get("AMES_TABPFN_PYTHON", sys.executable)
        weights = os.environ.get("TABPFN_MODEL_CACHE_DIR")
        if not weights:
            raise ValueError("Set TABPFN_MODEL_CACHE_DIR to the pinned regression checkpoint directory")
        worker = Path(__file__).with_name("tabpfn_worker.py")
        cache = os.environ.get("AMES_TABPFN_PRED_CACHE")
        with tempfile.TemporaryDirectory(prefix="ames-tabpfn-") as tmp:
            path = Path(tmp)
            context.to_csv(path/"training.csv", index=False)
            evaluation.to_csv(path/"evaluation.csv", index=False)
            np.save(path/"target.npy", self.target, allow_pickle=False)
            cats = context.select_dtypes(exclude=np.number).columns
            (path/"categories.txt").write_text("\n".join(cats)+"\n")
            if cache:
                fingerprint = hashlib.sha256(worker.read_bytes()+CHECKPOINT_SHA256.encode())
                for name in ("training.csv", "evaluation.csv", "target.npy", "categories.txt"):
                    fingerprint.update((path/name).read_bytes())
                target = Path(cache) / (fingerprint.hexdigest()+".npy")
                if target.exists():
                    return np.load(target, allow_pickle=False)
            env = dict(os.environ, TABPFN_NO_BROWSER="1", TABPFN_ALLOW_CPU_LARGE_DATASET="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       SKB_DATA_DIRECTORY=str(path/"skrub"))
            # Prevent the V2 Python 3.9 wheel directory from leaking into Python 3.12.
            env.pop("PYTHONPATH", None)
            # Each slot permits one in-flight inference. The normal evaluation uses
            # slot 1; a separate disjoint-fold precompute can opt into slot 2.
            from filelock import FileLock
            slot = os.environ.get("AMES_TABPFN_SLOT", "1")
            if slot not in ("1", "2"):
                raise ValueError("AMES_TABPFN_SLOT must be 1 or 2")
            lock_name = "ames-v3-tabpfn-cpu.lock" if slot == "1" else "ames-v3-tabpfn-cpu-2.lock"
            with FileLock(str(Path(tempfile.gettempdir()) / lock_name)):
                if cache and target.exists():
                    return np.load(target, allow_pickle=False)
                start = time.monotonic()
                subprocess.run([python, str(worker), tmp, weights], check=True, env=env)
                result = np.load(path/"output.npy", allow_pickle=False)
                if cache:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(f".{os.getpid()}.npy")
                    np.save(temp, result, allow_pickle=False)
                    temp.replace(target)
                    profile = {"train_rows": len(context), "eval_rows": len(evaluation),
                               "prediction_seconds": time.monotonic()-start}
                    target.with_suffix(".json").write_text(json.dumps(profile)+"\n")
                return result
