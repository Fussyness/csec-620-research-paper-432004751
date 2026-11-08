"""Evidence for the near-duplicate threshold (assumption A24; config dedup.near_threshold).

For one dataset (default Malimg) this reports:
  1. the distance from each sample to its nearest other sample (GIST), binned, with
     the fraction of nearest neighbours that belong to the same family;
  2. for each candidate threshold: number of duplicate groups, share of samples in a
     group, the largest group, mixed-family groups, families that collapse into a
     single group, and GIST + k-NN accuracy under a grouped split (all families and
     testable families only, mean over the configured seeds);
  3. a contact sheet of nearest-neighbour pairs at several distances, to check by eye
     what counts as a near-duplicate.

Outputs go to --out (default results/dedup_threshold/): summary.json (read by
src/tables.py for the paper's table_dedup.tex), summary.txt and pairs.png.

Like src.train, it resumes: if summary.json exists for the same settings it does
nothing, and if the dataset is missing but summary.json exists it keeps that.
--fresh recomputes.

    python scripts/dedup_threshold.py [--config config.yaml] [--dataset malimg] [--fresh]
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from src.data import duplicate_groups, load_dataset, make_folds, untestable_families  # noqa: E402
from src.model import build  # noqa: E402
from src.train import compute_representations, config_fingerprint, load_config  # noqa: E402

BINS = [0, 1e-9, 0.02, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20, np.inf]
PAIR_BANDS = [(0.0, 1e-9), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.10, 0.13)]


def grouped_accuracy(G, y, groups, cfg, n_classes):
    """GIST + k-NN accuracy (all, testable) under a grouped split, mean over seeds."""
    bad = untestable_families(groups, y)
    ok_rows = ~np.isin(y, bad)
    acc_all, acc_t = [], []
    for seed in cfg["seeds"]:
        fold = make_folds(y, cfg["splits"]["n_folds"], seed, groups)
        pred = np.zeros_like(y)
        for k in range(cfg["splits"]["n_folds"]):
            tr, te = fold != k, fold == k
            m = build("gist_knn", cfg)
            m.fit(G[tr], y[tr], n_classes, seed)
            pred[te] = m.predict_proba(G[te]).argmax(1)
        acc_all.append((pred == y).mean())
        acc_t.append((pred == y)[ok_rows].mean())
    return float(np.mean(acc_all)), float(np.mean(acc_t))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default="malimg")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.07, 0.09])
    ap.add_argument("--out", default="results/dedup_threshold")
    ap.add_argument("--fresh", action="store_true", help="recompute even if summary.json matches")
    a = ap.parse_args()

    cfg = load_config(a.config)
    os.makedirs(a.out, exist_ok=True)
    fp = config_fingerprint(cfg)
    summary_path = os.path.join(a.out, "summary.json")
    old = None
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            old = json.load(f)
        same = (old.get("config_fingerprint") == fp and old.get("dataset") == a.dataset
                and old.get("thresholds") == a.thresholds)
        if same and not a.fresh:
            print(f"{summary_path} is up to date (settings {fp}); nothing to do. Use --fresh to recompute.")
            return
    try:
        ds = load_dataset(a.dataset, cfg)
    except FileNotFoundError as e:
        if old is not None:
            print(f"Dataset not available ({e}); keeping the existing {summary_path} "
                  f"(settings {old.get('config_fingerprint')}).")
            return
        raise
    y, fams, nc = ds.labels, list(ds.families), len(ds.families)
    g = build("gist_knn", cfg)
    reps, hashes = compute_representations(ds, {"gist_knn": (g.rep_key(), g.represent)}, cfg, [])
    G = reps["gist_knn"]
    chosen = cfg["dedup"]["near_threshold"] or 0.0
    lines = [f"dataset {a.dataset}: {len(y)} samples, {nc} families; configured threshold {chosen}"]

    # 1. nearest-neighbour distances
    d, i = NearestNeighbors(n_neighbors=2).fit(G).kneighbors(G)
    nd, ni = d[:, 1], i[:, 1]
    lines += ["", "nearest-neighbour distance | samples | NN in same family"]
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (nd >= lo) & (nd < hi)
        if m.any():
            lines.append(f"  [{lo:.2f}, {hi:.2f}) | {m.sum():6d} | {(y[ni[m]] == y[m]).mean():.3f}")

    # 2. thresholds
    lines += ["", "threshold | groups | in a group | largest group | mixed groups | "
              "single-group families | acc all | acc testable"]
    rows = []
    for t in a.thresholds:
        grp = duplicate_groups(list(hashes), G, t or None)
        u, cnt = np.unique(grp, return_counts=True)
        big = u[cnt.argmax()]
        big_fams = dict(Counter(fams[c] for c in y[grp == big]))
        mixed = sum(len(set(y[grp == gg])) > 1 for gg in u[cnt > 1])
        single = [fams[c] for c in untestable_families(grp, y)]
        share = float((cnt[np.searchsorted(u, grp)] > 1).mean())
        acc_all, acc_t = grouped_accuracy(G, y, grp, cfg, nc)
        rows.append((t, len(u), share, int(cnt.max()), mixed, single, acc_all, acc_t))
        lines.append(f"  {t:<8} | {len(u):6d} | {share:6.1%} | {cnt.max():5d} {big_fams} | {mixed} | "
                     f"{single} | {acc_all:.3f} | {acc_t:.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # Machine-readable summary; src/tables.py turns it into report/generated/table_dedup.tex
    summary = {
        "config_fingerprint": fp, "dataset": a.dataset, "thresholds": a.thresholds,
        "chosen": chosen, "n_folds": cfg["splits"]["n_folds"], "n_seeds": len(cfg["seeds"]),
        "nn_bins": [{"lo": lo, "hi": None if hi == np.inf else hi, "n": int(((nd >= lo) & (nd < hi)).sum()),
                     "same_family": float((y[ni[(nd >= lo) & (nd < hi)]] == y[(nd >= lo) & (nd < hi)]).mean())
                     if ((nd >= lo) & (nd < hi)).any() else None}
                    for lo, hi in zip(BINS[:-1], BINS[1:])],
        "rows": [{"threshold": t, "groups": ng, "share_in_group": share, "largest_group": big,
                  "mixed_groups": mixed, "single_group_families": single,
                  "acc_all": acc_all, "acc_testable": acc_t}
                 for t, ng, share, big, mixed, single, acc_all, acc_t in rows],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 3. contact sheet of nearest-neighbour pairs
    rng = np.random.default_rng(0)
    per_band = 4
    fig, axes = plt.subplots(len(PAIR_BANDS), 2 * per_band, figsize=(2 * per_band * 1.3, len(PAIR_BANDS) * 1.9))
    for r, (lo, hi) in enumerate(PAIR_BANDS):
        cand = np.where((nd >= lo) & (nd < hi))[0]
        pick = rng.choice(cand, min(per_band, len(cand)), replace=False) if len(cand) else []
        for c in range(per_band):
            for s in range(2):
                ax = axes[r, 2 * c + s]
                ax.set_xticks([]); ax.set_yticks([])
                if c < len(pick):
                    j = int(pick[c]) if s == 0 else int(ni[pick[c]])
                    ax.imshow(ds.load(j), cmap="gray", aspect="auto", vmin=0, vmax=255)
                    ax.set_title(f"{fams[y[j]][:12]}" + (f"\nd={nd[pick[c]]:.3f}" if s == 0 else "\n"), fontsize=6)
        axes[r, 0].set_ylabel(f"[{lo:.2f}, {hi:.2f})", fontsize=7)
    fig.suptitle("Nearest-neighbour pairs by GIST distance (left: sample, right: its neighbour)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "pairs.png"), dpi=130)
    print(f"\nwrote {a.out}/summary.json, summary.txt and pairs.png")


if __name__ == "__main__":
    main()
