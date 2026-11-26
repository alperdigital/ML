"""Shape-safe, finite, paired price metrics for the Ames pipelines."""
import numpy as np


def _price_vector(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("Prices must be a nonempty finite vector or a single column")
    return array


def price_vectors(y, pred):
    """Validate one prediction per house and apply the nonnegative price floor."""
    y, pred = _price_vector(y), _price_vector(pred)
    if y.shape != pred.shape:
        raise ValueError("Targets and predictions must have the same number of rows")
    if (y < 0).any():
        raise ValueError("Negative target")
    return y, np.maximum(pred, 0)


def rmsle(y, pred):
    y, pred = price_vectors(y, pred)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(pred)) ** 2)))
