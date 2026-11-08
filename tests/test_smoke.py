"""Tests: width table, folds/leakage, metrics, stub contracts, end-to-end smoke run.

The GIST and attack contract tests skip until those functions are implemented.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src import attack, features
from src.data import (byteplot, duplicate_groups, image_bytes, make_folds, make_synthetic,
                      pixel_sha256, untestable_families, width_for_size)
from src.evaluate import compute_metrics, mcnemar, robustness_auc
from src.train import load_config, main

CFG = load_config("config.ci.yaml")
WT = CFG["data"]["width_table"]


# ---------------------------------------------------------------- width table / byteplot
@pytest.mark.parametrize("kb,width", [(0.5, 32), (9.99, 32), (10, 64), (29.9, 64), (30, 128),
                                      (99, 256), (100, 384), (450, 512), (999, 768), (5000, 1024)])
def test_width_table_lookup(kb, width):
    assert width_for_size(int(kb * 1024), WT) == width


def test_byteplot_round_trip_and_padding():
    raw = bytes(range(256)) * 41 + b"\x07"            # 10,497 bytes -> width 64
    img = byteplot(raw, WT)
    assert img.shape == (-(-len(raw) // 64), 64)
    assert image_bytes(img)[: len(raw)] == raw
    assert not any(image_bytes(img)[len(raw):])     # padding is zeros


# ---------------------------------------------------------------- dedup / folds
@pytest.fixture(scope="module")
def syn():
    return make_synthetic(CFG["data"]["synthetic"], WT, seed=0)


def test_exact_duplicates_found(syn):
    hashes = [pixel_sha256(syn.load(i)) for i in range(len(syn))]
    groups = duplicate_groups(hashes, None, None)
    assert groups.max() + 1 < len(syn)               # the generator injects duplicates


def test_untestable_families():
    """A family whose samples share one duplicate group can never be both trained
    and tested under a grouped split (Malimg's Yuner.A)."""
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    groups = np.array([5, 5, 5, 6, 7, 8, 8, 9])     # family 0: one group; 1 and 2: two each
    assert untestable_families(groups, labels) == [0]
    groups[3:5] = 5                                    # family 1 merges into family 0's group
    assert untestable_families(groups, labels) == [0, 1]


def test_no_duplicate_group_spans_two_folds(syn):
    """Direct leakage test for E2."""
    hashes = [pixel_sha256(syn.load(i)) for i in range(len(syn))]
    feats = np.stack([features.thumbnail(syn.load(i), 16) for i in range(len(syn))])
    groups = duplicate_groups(hashes, feats, threshold=0.5)
    for seed in (0, 1):
        fold = make_folds(syn.labels, 3, seed, groups)
        for g in np.unique(groups):
            assert len(np.unique(fold[groups == g])) == 1


def test_stratified_folds_cover_everything(syn):
    fold = make_folds(syn.labels, 3, 0)
    assert set(np.unique(fold)) == {0, 1, 2}
    assert len(fold) == len(syn)


# ---------------------------------------------------------------- metrics
def test_metrics_multiclass_with_missing_class():
    y = np.array([0, 1, 2, 2])
    prob = np.eye(4, dtype=np.float32)[[0, 1, 2, 2]]  # 4 columns, class 3 never appears
    m = compute_metrics(y, prob)
    assert m["accuracy"] == m["f1"] == m["roc_auc"] == 1.0
    assert len(m["per_family_recall"]) == 4


def test_mcnemar_and_robustness_auc():
    y = np.zeros(10, int)
    assert mcnemar(y, y, y)["p_value"] == 1.0
    assert robustness_auc([0, 0.1], [1.0, 1.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------- stub contracts
def test_gist_contract(syn):
    g = CFG["features"]["gist"]
    try:
        v = features.gist(syn.load(0), g)
    except NotImplementedError:
        pytest.skip("GIST not implemented yet")
    assert v.shape == (features.gist_dim(g),)
    assert np.isfinite(v).all()
    np.testing.assert_array_equal(v, features.gist(syn.load(0), g))


@pytest.mark.parametrize("kind", ["random", "zero", "other_family"])
def test_append_contract(syn, kind):
    img, donor = syn.load(0), syn.load(len(syn) - 1)
    rng = np.random.default_rng(0)
    try:
        out = attack.append_bytes(img, 0.10, kind, rng, WT, donor=donor)
    except NotImplementedError:
        pytest.skip("append_bytes not implemented yet")
    n = img.size
    assert out.dtype == np.uint8 and out.ndim == 2
    assert out.shape[1] == width_for_size(n + int(round(0.10 * n)), WT)  # re-binned width
    assert image_bytes(out)[:n] == image_bytes(img)                       # original bytes untouched
    np.testing.assert_array_equal(attack.append_bytes(img, 0.0, kind, rng, WT, donor=donor), img)


# ---------------------------------------------------------------- end to end
def test_end_to_end_ci_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"inherit: {Path('config.ci.yaml').resolve().as_posix()}\n"
                        f"output: {{dir: {tmp_path.as_posix()}/out}}\n")
    main(str(cfg_path))
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= m[k] <= 1.0
    assert "e1" in m["experiments"] and "e2" in m["experiments"]
    assert (tmp_path / "out" / "splits" / "e2_seed0.csv").exists()

    # Resume: a second full run reuses the per-seed results instead of recomputing
    # them, and still rebuilds metrics.json (this is what `make reproduce` does with
    # the committed results).
    run_file = tmp_path / "out" / "e1" / "runs" / "seed0.json"
    rec = json.loads(run_file.read_text())
    assert rec["config_fingerprint"]
    before = run_file.stat().st_mtime_ns
    (tmp_path / "out" / "metrics.json").unlink()
    main(str(cfg_path))
    assert run_file.stat().st_mtime_ns == before
    m2 = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert m2["accuracy"] == m["accuracy"]

    # Results made with other settings are never reused silently.
    changed = tmp_path / "changed.yaml"
    changed.write_text(cfg_path.read_text() + "models: {gist_knn: {k: 5}}\n")
    with pytest.raises(SystemExit, match="different settings"):
        main(str(changed))
