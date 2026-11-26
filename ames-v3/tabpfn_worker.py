"""Run pinned TabPFNv2 regression in its dedicated Python environment."""
import hashlib
import os
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

CHECKPOINT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"


def main(folder, weight_dir):
    folder = Path(folder)
    weights = Path(weight_dir) / "tabpfn-v2-regressor.ckpt"
    if version("tabpfn") != "10.2.0":
        raise RuntimeError("Use tabpfn==10.2.0 in the foundation-model environment")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("TabPFNv2 regression checkpoint fingerprint mismatch")
    # On this CPU the 1- and 4-thread eight-estimator parity probe produced
    # identical predictions. A finite thread budget keeps memory usage bounded.
    threads = int(os.environ.get("AMES_TABPFN_THREADS", "4"))
    if not 1 <= threads <= 4:
        raise ValueError("AMES_TABPFN_THREADS must be between one and four")
    torch.set_num_threads(threads)
    from tabpfn.constants import ModelVersion
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=weights, device="cpu", random_state=42)
    training = pd.read_csv(folder/"training.csv")
    target = np.load(folder/"target.npy")
    evaluation = pd.read_csv(folder/"evaluation.csv")
    cats = (folder/"categories.txt").read_text().splitlines()
    # Preserve known training dtypes even for a single row with missing categories.
    for col in cats:
        training[col] = training[col].astype("string")
        evaluation[col] = evaluation[col].astype("string")
    model.fit(training, np.log1p(target))
    pred = np.asarray(np.maximum(np.expm1(model.predict(evaluation)), 0.), dtype=float)
    if pred.shape != (len(evaluation),) or not np.isfinite(pred).all():
        raise ValueError("Invalid foundation-model output")
    np.save(folder/"output.npy", pred, allow_pickle=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
