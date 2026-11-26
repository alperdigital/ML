"""Compute the expensive nested V2 search once for each independent training split."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import joblib

from design import load_data, partitions, inner_splits


def base_task(data, name, work):
    from baselines import select_v2
    df = load_data(data)
    train, _, _, _ = partitions(df)[name]
    training = df.iloc[train].copy()
    y = training.pop("SalePrice")
    start = time.monotonic()
    result = select_v2(training, y, inner_splits(training))
    result["elapsed_seconds"] = time.monotonic() - start
    result["train_ids"] = training.Id.tolist()
    destination = Path(work)/f"{name}.joblib"
    temp = destination.with_suffix(f".{os.getpid()}.tmp")
    joblib.dump(result, temp)
    temp.replace(destination)
    print("Nested V2 search complete", name, round(result["elapsed_seconds"],1), flush=True)


def build(data, work, jobs):
    Path(work).mkdir(parents=True, exist_ok=True)
    tasks = partitions(load_data(data))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(base_task, data, k, work) for k in tasks if not (Path(work)/f"{k}.joblib").exists()]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--jobs", type=int, default=3)
    a = p.parse_args()
    build(a.data, a.work, a.jobs)
