"""Restore completed data-only Ames experiments into the local content-addressed caches."""
import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write_new(path, writer):
    """Install a whole file and leave earlier local progress in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with tmp.open("wb") as output:
        writer(output)
    tmp.replace(path)


def valid_digest(value):
    return bool(re.fullmatch(r"[0-9a-f]{64}", value))


def restore(work):
    manifest = json.loads((HERE / "manifest.json").read_text())
    blob = (HERE / "progress.json.gz").read_bytes()
    for content, expected in ((blob, manifest["payload_sha256"]),
                              ((ROOT / "data/train.csv").read_bytes(), manifest["source_csv_sha256"])):
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Checkpoint or source CSV checksum differs")
    state = json.loads(gzip.decompress(blob))
    for key, result in state["searches"].items():
        split, digest = key.split("-", 1)
        if split not in manifest["search_splits"] or not valid_digest(digest):
            raise ValueError("Unexpected nested search key")
        result["oof"] = {k: np.array(v, dtype=float) for k, v in result["oof"].items()}
        write_new(work / "searches" / (key + ".joblib"), lambda out: joblib.dump(result, out))
    for key, entry in state["recipes"].items():
        if not valid_digest(key):
            raise ValueError("Unexpected candidate key")
        values = (np.asarray(entry["prices"], dtype=float), entry["seconds"])
        write_new(work / "recipes" / (key + ".joblib"), lambda out: joblib.dump(values, out))
    dest = work / "foundation-cache" / f"threads-{manifest['threads_per_tabpfn_job']}"
    for digest, entry in state["foundation"].items():
        if not valid_digest(digest):
            raise ValueError("Unexpected TabPFN key")
        profile = entry["profile"]
        prices = np.asarray(entry["prices"], dtype=entry["dtype"])
        if prices.shape != (profile["eval_rows"],) or not np.isfinite(prices).all() or (prices < 0).any():
            raise ValueError("Incomplete TabPFN predictions")
        write_new(dest / (digest + ".npy"), lambda out: np.save(out, prices, allow_pickle=False))
        write_new(dest / (digest + ".json"), lambda out: out.write((json.dumps(profile) + "\n").encode()))
    print(f"Restored checkpoint in {work.resolve()}: {len(state['searches'])} searches, "
          f"{len(state['recipes'])} candidate entries, {len(state['foundation'])} TabPFN batches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    restore(parser.parse_args().work)
