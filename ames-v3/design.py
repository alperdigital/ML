"""Predeclared train/evaluation indices and data identity."""
import hashlib
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ames_data import read_ames_csv

DATA_SHA = "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41"


def load_data(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("The input bytes differ from the registered V2 reference CSV.")
    df = read_ames_csv(path)
    if not df.Id.is_unique or df.SalePrice.isna().any() or (df.SalePrice < 0).any():
        raise ValueError("Expected unique house IDs and nonnegative, complete targets.")
    return df


def partitions(df):
    dev, historical = train_test_split(np.arange(len(df)), test_size=.2, random_state=42)
    tasks = {}
    for repeat, seed in enumerate([2603, 2604]):
        for fold, (train, val) in enumerate(KFold(5, shuffle=True, random_state=seed).split(dev)):
            tasks[f"r{repeat}f{fold}"] = (dev[train], dev[val], repeat, fold)
    year = df.YrSold.to_numpy()[dev]
    tasks["temporal"] = (dev[year < 2010], dev[year == 2010], 0, 0)
    tasks["final"] = (dev, historical, 0, 0)
    return tasks


def inner_splits(X):
    return list(KFold(3, shuffle=True, random_state=42).split(X))
