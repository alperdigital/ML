"""Cache fold-local OOF predictions for fixed recipes across interrupted runs."""
import hashlib
import json
import os
import time
from pathlib import Path

import joblib


def cached_oof(X, y, splits, name, spec, work):
    from models_v3 import crossfit
    if work is None:
        return crossfit(spec, X, y, splits), 0.0
    digest = hashlib.sha256(json.dumps([X.Id.tolist(), name, spec, [val.tolist() for _,val in splits]],
                                       sort_keys=True).encode()).hexdigest()
    path = Path(work)/"recipes"/(digest+".joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return joblib.load(path)
    start = time.monotonic()
    result = crossfit(spec, X, y, splits), time.monotonic()-start
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    joblib.dump(result, temp)
    temp.replace(path)
    return result
