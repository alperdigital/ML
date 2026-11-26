"""Fit/predict bridge for the pinned CPU foundation model, separate from V2 wheels."""
import hashlib
import os
import json
import time
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

CHECKPOINT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"


def cpu_threads():
    threads = int(os.environ.get("AMES_TABPFN_THREADS", "3"))
    if not 1 <= threads <= 4:
        raise ValueError("AMES_TABPFN_THREADS must be between one and four")
    return threads


@lru_cache(maxsize=None)
def foundation_runtime(python, checkpoint, mtime_ns, size):
    """Identify the checkpoint and the worker's actual package versions."""
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("TabPFNv2 regression checkpoint fingerprint mismatch")
    probe = ("import json,platform; from importlib.metadata import version; "
             "pkgs=('tabpfn','torch','numpy','pandas','scipy','scikit-learn','skrub'); "
             "print(json.dumps([platform.python_version(),platform.machine(),"
             "{k:version(k) for k in pkgs}], sort_keys=True))")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.check_output([python, "-c", probe], text=True, env=env).strip()


def runtime_signature():
    python = os.environ.get("AMES_TABPFN_PYTHON", sys.executable)
    checkpoint = Path(os.environ["TABPFN_MODEL_CACHE_DIR"]) / "tabpfn-v2-regressor.ckpt"
    stat = checkpoint.stat()
    return foundation_runtime(python, str(checkpoint.resolve()), stat.st_mtime_ns, stat.st_size)


class FoundationRegressor:
    """Context-learning estimator. Store only the training context until prediction."""
    def __init__(self, X, y):
        self.training = X.drop(columns=["Id", "SalePrice"], errors="ignore").copy()
        self.target = np.asarray(y, dtype=float).copy()

    def _cached(self, path):
        profile = json.loads(path.with_suffix(".json").read_text())
        self.cached_inference_seconds = float(profile["prediction_seconds"])
        return np.load(path, allow_pickle=False)

    def predict(self, X):
        self.cached_inference_seconds = None
        context = self.training
        evaluation = X.loc[:, context.columns]
        python = os.environ.get("AMES_TABPFN_PYTHON", sys.executable)
        weights = os.environ.get("TABPFN_MODEL_CACHE_DIR")
        if not weights:
            raise ValueError("Set TABPFN_MODEL_CACHE_DIR to the pinned regression checkpoint directory")
        worker = Path(__file__).with_name("tabpfn_worker.py")
        cache = os.environ.get("AMES_TABPFN_PRED_CACHE")
        threads = cpu_threads()
        with tempfile.TemporaryDirectory(prefix="ames-tabpfn-") as tmp:
            path = Path(tmp)
            context.to_csv(path/"training.csv", index=False)
            evaluation.to_csv(path/"evaluation.csv", index=False)
            np.save(path/"target.npy", self.target, allow_pickle=False)
            cats = context.select_dtypes(exclude=np.number).columns
            (path/"categories.txt").write_text("\n".join(cats)+"\n")
            if cache:
                fingerprint = hashlib.sha256(worker.read_bytes() + CHECKPOINT_SHA256.encode()
                                             + runtime_signature().encode())
                for name in ("training.csv", "evaluation.csv", "target.npy", "categories.txt"):
                    fingerprint.update((path/name).read_bytes())
                target = Path(cache) / f"threads-{threads}" / (fingerprint.hexdigest()+".npy")
                if target.exists():
                    return self._cached(target)
            env = dict(os.environ, TABPFN_NO_BROWSER="1", TABPFN_ALLOW_CPU_LARGE_DATASET="1",
                       AMES_TABPFN_THREADS=str(threads),
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
                    return self._cached(target)
                start = time.monotonic()
                subprocess.run([python, str(worker), tmp, weights], check=True, env=env)
                result = np.load(path/"output.npy", allow_pickle=False)
                if cache:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(f".{os.getpid()}.npy")
                    np.save(temp, result, allow_pickle=False)
                    profile = {"train_rows": len(context), "eval_rows": len(evaluation), "threads": threads,
                               "prediction_seconds": time.monotonic()-start}
                    profile_path = temp.with_suffix(".json")
                    profile_path.write_text(json.dumps(profile)+"\n")
                    profile_path.replace(target.with_suffix(".json"))
                    temp.replace(target)
                return result
