"""Smoke tests: the pipeline runs and produces sane metrics."""
import numpy as np
from src.data import load_data
from src.evaluate import compute_metrics

CFG = {
    "seed": 0,
    "data": {"source": "synthetic", "n_samples": 400, "n_features": 10, "test_size": 0.25},
}


def test_load_data_shapes():
    Xtr, Xte, ytr, yte = load_data(CFG)
    assert Xtr.shape[1] == Xte.shape[1] == 10
    assert len(ytr) == 300 and len(yte) == 100


def test_metrics_range():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_metrics(y, p)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["accuracy"] == 1.0
