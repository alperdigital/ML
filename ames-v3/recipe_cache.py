"""Content-addressed caches for the nested search and fixed candidate predictions."""
import hashlib
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]


def fit_fingerprint(X, y, splits, settings, sources, packages):
    """Bind cached prices to ordered inputs, both sides of every fold, and fit implementation."""
    hasher = hashlib.sha256()
    environment = {p: version(p) for p in packages}
    environment.update(python=platform.python_version(), machine=platform.machine())
    hasher.update(json.dumps([settings, environment], sort_keys=True).encode())
    hasher.update(joblib.hash((X, y, list(splits)), hash_name="sha1").encode())
    for source in sources:
        hasher.update(hashlib.sha256((ROOT / source).read_bytes()).digest())
    return hasher.hexdigest()


def cached_oof(X, y, splits, name, spec, work):
    from models_v3 import crossfit
    splits = list(splits)
    if work is None:
        return crossfit(spec, X, y, splits), 0.0
    settings = dict(name=name, spec=spec)
    sources = ["ames-v3/models_v3.py", "src/ames_data.py"]
    if spec.get("model") == "tabpfn":
        from tabpfn_bridge import CHECKPOINT_SHA256, cpu_threads, runtime_signature
        sources += ["ames-v3/tabpfn_bridge.py", "ames-v3/tabpfn_worker.py"]
        settings.update(worker_runtime=runtime_signature(), checkpoint=CHECKPOINT_SHA256, threads=cpu_threads())
    digest = fit_fingerprint(X, y, splits, settings, sources,
                             ["numpy", "pandas", "scikit-learn", "catboost", "scipy", "joblib"])
    path = Path(work)/"recipes"/(digest+".joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return joblib.load(path)
    start = time.monotonic()
    timing = {"cache_adjustment": 0.0}
    predictions = crossfit(spec, X, y, splits, timing=timing)
    result = predictions, time.monotonic() - start + timing["cache_adjustment"]
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    joblib.dump(result, temp)
    temp.replace(path)
    return result
