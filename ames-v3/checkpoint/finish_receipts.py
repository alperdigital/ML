"""Verify the completed 12-split run and reproduce its timing and single-house receipts."""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
ROOT = V3.parent
sys.path.insert(0, str(V3))
from design import DATA_SHA, inner_splits, load_data, partitions
from models_v3 import blend_predict
from predict import prepare_row, recommended_version
from src.ames_data import INPUT_SCHEMA, read_ames_csv
from tabpfn_bridge import CHECKPOINT_SHA256


def read(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_run(work, out, manifest, data, summary, frozen):
    plan = manifest["tabpfn_plan"]
    tasks = partitions(data)
    audits = json.loads(gzip.decompress((out / "split-audits.json.gz").read_bytes()))
    check(set(plan) == set(tasks) == set(audits) and len(tasks) == 12, "Expected all 12 registered splits")
    check(summary["dataset_sha256"] == manifest["source_csv_sha256"] == DATA_SHA and
          summary["input_schema"] == manifest["input_schema"] == INPUT_SCHEMA, "Incompatible data or schema")
    check(all(audits["final"][k] == v for k, v in frozen.items()), "Frozen recipe differs from final audit")
    outer = pd.read_csv(out / "outer-predictions.csv.gz")
    check(len(outer) == 2336, "Incomplete outer predictions")
    cache = work / "foundation-cache" / f"threads-{manifest['threads_per_tabpfn_job']}"
    seconds = {}
    for name, (train, evaluation, repeat, fold) in tasks.items():
        audit = audits[name]
        trials = audit["v2_search"]["trials"]
        check(audit["input_schema"] == INPUT_SCHEMA and audit["foundation"] == summary["foundation"]
              and audit["foundation"]["available"] and "tabpfn_v2" in audit["specs"],
              f"{name} lacks the corrected foundation-model run")
        check(all([t["trial"] for t in trials[family]] == list(range(20))
                  for family in ("ridge", "xgboost", "lightgbm", "catboost")), f"Incomplete trials for {name}")
        frame = (pd.read_csv(out / f"{name}.csv") if name in ("final", "temporal")
                 else outer[(outer["repeat"] == repeat) & (outer["fold"] == fold)])
        np.testing.assert_array_equal(frame[["Id", "SalePrice"]], data.iloc[evaluation][["Id", "SalePrice"]])
        np.testing.assert_array_equal(frame.v3, frame[audit["recipe"]])
        sizes = [(len(t), len(v)) for t, v in inner_splits(data.iloc[train])] + [(len(train), len(evaluation))]
        check([(t, v) for _, t, v in plan[name]] == sizes, f"Wrong batch plan for {name}")
        seconds[name] = []
        for index, (key, train_rows, eval_rows) in enumerate(plan[name]):
            profile = read(cache / (key + ".json"))
            pred = np.load(cache / (key + ".npy"), allow_pickle=False)
            check((profile["train_rows"], profile["eval_rows"], profile["threads"]) ==
                  (train_rows, eval_rows, manifest["threads_per_tabpfn_job"]), f"Incorrect profile {key}")
            check(pred.shape == (eval_rows,) and np.isfinite(pred).all() and (pred >= 0).all(),
                  f"Invalid prices for {key}")
            duration = profile["prediction_seconds"]
            check(np.isfinite(duration) and duration > 0, f"Invalid duration for {key}")
            seconds[name].append(duration)
            if index == 3:
                np.testing.assert_allclose(frame.cand_tabpfn_v2, pred, atol=1e-7, rtol=0)
    return seconds


def finish(work, out):
    manifest = read(HERE / "manifest.json")
    summary, frozen = read(out / "summary.json"), read(out / "frozen-final.json")
    data = load_data(ROOT / "data/train.csv")
    seconds = verify_run(work, out, manifest, data, summary, frozen)
    fit = work / "development-fit.joblib"
    bundle = joblib.load(fit)  # Trusted local artifact from run.py.
    check(bundle["input_schema"] == INPUT_SCHEMA and all(bundle["audit"][k] == v for k, v in frozen.items()),
          "Development fit differs from the verified frozen recipe")
    threads = manifest["threads_per_tabpfn_job"]
    os.environ.update(AMES_TABPFN_PRED_CACHE=str(work / "foundation-cache"), AMES_TABPFN_THREADS=str(threads))
    python, architecture, versions = summary["foundation"]["runtime"]
    timing = {"model": "TabPFNv2 regression", "checkpoint": "tabpfn-v2-regressor.ckpt",
              "input_schema": INPUT_SCHEMA, "dataset_sha256": DATA_SHA, "checkpoint_sha256": CHECKPOINT_SHA256,
              "worker_sha256": hashlib.sha256((V3 / "tabpfn_worker.py").read_bytes()).hexdigest(),
              "runtime": dict(versions, python=python, architecture=architecture, device="cpu", threads_per_job=threads),
              "measured_benchmark_jobs": {
                  "batch_count": sum(map(len, seconds.values())),
                  "total_inference_job_seconds": round(sum(map(sum, seconds.values())), 2),
                  "by_task": {k: {"inner_oof_seconds": round(sum(s[:3]), 2), "evaluation_seconds": round(s[3], 2)}
                              for k, s in seconds.items()},
                  "definition": "Sum of elapsed inference jobs, including worker startup. Concurrent jobs overlap. "
                                "Excludes lock wait, installation and separate single-house queries."}}

    def cli(*args):
        cmd = [sys.executable, str(V3 / "predict.py"), "--results", str(out), "--bundle", str(fit), *args]
        return json.loads(subprocess.check_output(cmd, text=True))

    held = pd.read_csv(out / "final.csv").set_index("Id")
    batch = float(held.at[899, "v3"])
    checks = [{"id": r["Id"], "partition": "outer repeat 0, fold 0; inner fold 0", "model": "tabpfn_v2",
               "training_houses": r["train_rows"], "cohort_query_houses": r["batch_rows"],
               "cohort_price_usd": r["batch_usd"], "single_price_usd": r["one_row_usd"],
               "single_minus_cohort_usd": round(r["one_row_usd"] - r["batch_usd"], 2),
               "unrounded_single_minus_cohort_usd": r["delta_usd"]} for r in manifest["development_query_checks"]]
    with tempfile.TemporaryDirectory(dir=out) as directory:
        tmp = Path(directory)
        single = tmp / "house-899.csv"
        # Keep the original category spellings in this stand-alone, unlabeled Ames CSV.
        raw = read_ames_csv(ROOT / "data/train.csv", dtype=str)
        raw.loc[raw.Id.eq("899")].drop(columns="SalePrice").to_csv(single, index=False, na_rep="NA")
        historical = cli("--version", "v3", "--holdout-id", "899")
        standalone = cli("--version", "v3", "--input", str(single))
        check(historical["Id"] == standalone["Id"] == 899 and
              historical["recipe"] == standalone["recipe"] == frozen["recipe"] and
              historical["predicted_sale_price_usd"] == round(batch, 2), "House 899 differs from final.csv")
        row = prepare_row(read_ames_csv(single, dtype=str), bundle["columns"], bundle["dtypes"])
        recipe = frozen["recipe"]
        if recipe in ("equal_blend", "weighted_blend"):
            members = frozen["member_order"]
            weights = frozen["weights"] if recipe == "weighted_blend" else np.full(len(members), 1 / len(members))
            raw_price = blend_predict(np.column_stack([bundle["models"][k].predict(row) for k in members]), weights)
        else:
            raw_price = bundle["models"][recipe].predict(row)
        single_usd = standalone["predicted_sale_price_usd"]
        check(round(float(raw_price[0]), 2) == single_usd, "Stand-alone CLI differs from the development fit")
        checks.append({"id": 899, "partition": "historical holdout", "model": recipe, "best_new": frozen["best_new"],
                       "training_houses": len(data) - len(held), "cohort_query_houses": len(held),
                       "cohort_price_usd": round(batch, 2), "holdout_id_cli_price_usd": historical["predicted_sale_price_usd"],
                       "single_price_usd": single_usd, "single_minus_cohort_usd": round(single_usd - round(batch, 2), 2),
                       "unrounded_single_minus_cohort_usd": float(raw_price[0]) - batch})
        query = {"dataset_sha256": DATA_SHA, "input_schema": INPUT_SCHEMA, "checkpoint_sha256": CHECKPOINT_SHA256,
                 "threads_per_inference_job": threads, "single_query_houses": 1,
                 "amounts": "USD. Differences subtract the cohort price from the same house predicted alone.",
                 "checks": checks}
        example = cli("--version", "recommended", "--input", str(ROOT / "data/test.csv"), "--id", "1461")
        check(example["Id"] == 1461 and example["version"] == recommended_version(summary), "Incorrect example")
        files = {"tabpfn-runtime.json": timing, "query-batch-check.json": query, "example.json": example}
        for name, value in files.items():
            (tmp / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        for name in files:
            (tmp / name).replace(out / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    finish(args.work.resolve(), args.out.resolve())
