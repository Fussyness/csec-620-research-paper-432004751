"""Single entry point: `python -m src.train --config config.yaml`.

Runs every experiment listed in `run.experiments` for every seed and fold and
writes, under `output.dir`:

  splits/<exp>_seed<s>.csv     fold (and duplicate-group) assignment per sample
  <exp>/runs/seed<s>.json      raw per-seed results (one file per job)
  <exp>/summary.json           mean ± std over seeds for every model
  <exp>/cm_<model>.png         confusion matrix (first seed, pooled out-of-fold)
  e2/nn_distance_hist.png      near-duplicate threshold tuning aid
  e4/budget_curves.png         accuracy vs. append budget
  metrics.json                 headline numbers + all summaries + anything skipped

Parallel use (e.g. one GPU per job, see scripts/lovegood/):
  --prepare                    build the descriptor caches once, train nothing
  --experiments e2 --seeds 0   run one slice; writes only its runs/seed0.json
  --merge                      combine all runs/seed*.json into the summaries

Without --experiments/--seeds the run does everything and merges at the end.
Parts that are not implemented yet (GIST, the attacks) are skipped and listed
under "skipped" in metrics.json instead of failing the run.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from itertools import combinations

import numpy as np
import torch
import yaml

from . import attack, features
from .data import (duplicate_groups, duplicate_rate_by_family, load_dataset, make_folds,
                   nearest_neighbor_distances, pixel_sha256, save_folds, untestable_families)
from .evaluate import (compute_metrics, evasion_rate, mcnemar, mean_std, robustness_auc,
                       save_budget_curves, save_confusion_matrix, save_nn_distance_hist)
from .model import build

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    """YAML config; `inherit: other.yaml` deep-merges this file over another."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(os.path.join(os.path.dirname(path), parent)), cfg)
    return cfg


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _dedupe(skipped):
    return [dict(t) for t in dict.fromkeys(tuple(s.items()) for s in skipped)]


# ---------------------------------------------------------------- representations
def _cache_path(cfg, ds, key):
    ident = hashlib.sha1(("|".join(ds.names) + key).encode()).hexdigest()[:12]
    return os.path.join(cfg["data"]["cache_dir"], f"{ds.name}_{ident}.npy")


def _save_atomic(path, arr):
    """Write then rename, so a parallel job never reads a half-written cache file."""
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def compute_representations(ds, reps_needed: dict, cfg, skipped: list):
    """reps_needed: name -> (cache key, fn(img) -> array). Returns (reps, pixel hashes).

    One pass over the images; cached to data.cache_dir for real datasets. A
    representation whose function raises NotImplementedError is dropped and
    recorded in `skipped`.
    """
    use_cache = cfg["data"]["source"] != "synthetic"
    reps, todo = {}, {}
    for name, (key, fn) in reps_needed.items():
        path = _cache_path(cfg, ds, key)
        if use_cache and os.path.exists(path):
            reps[name] = np.load(path)
        else:
            todo[name] = (path, fn, [])
    hash_path = _cache_path(cfg, ds, "sha256")
    hashes = list(np.load(hash_path)) if use_cache and os.path.exists(hash_path) else None
    if todo or hashes is None:
        new_hashes = []
        t0 = time.time()
        for i in range(len(ds)):
            img = ds.load(i)
            new_hashes.append(pixel_sha256(img))
            for name in list(todo):
                try:
                    todo[name][2].append(todo[name][1](img))
                except NotImplementedError as e:
                    skipped.append({"what": name, "reason": str(e)})
                    del todo[name]
            if (i + 1) % 1000 == 0:
                print(f"  [{ds.name}] represented {i + 1}/{len(ds)} ({time.time() - t0:.0f}s)")
        hashes = new_hashes
        if use_cache:
            os.makedirs(cfg["data"]["cache_dir"], exist_ok=True)
            _save_atomic(hash_path, np.array(hashes))
        for name, (path, _, vals) in todo.items():
            reps[name] = np.stack(vals)
            if use_cache:
                _save_atomic(path, reps[name])
    return reps, hashes


# ---------------------------------------------------------------- attacks
def perturb(kind, img, budget, rng, cfg, model, label, donors):
    wt = cfg["data"]["width_table"]
    if kind == "fgsm":
        return attack.fgsm_append(model, img, label, budget, cfg["attack"]["fgsm_eps"], wt)
    donor = donors() if kind == "other_family" else None
    return attack.append_bytes(img, budget, kind, rng, wt, donor=donor)


def attack_fold(model, ds, te, tr, clean_pred, cfg, seed, fold_k):
    """Predictions on the test fold under every (attack, budget). Budget 0 reuses
    the clean predictions. Returns {attack: {budget: preds}}."""
    a = cfg["attack"]
    rng = np.random.default_rng(seed * 7919 + fold_k)
    imgs = [ds.load(i) for i in te]          # load once, reuse for every attack and budget
    out = {}
    for kind in a["types"]:
        if kind == "fgsm" and model.name != "cnn":
            continue                      # white-box gradient attack needs a differentiable model
        out[kind] = {}
        for b in a["budgets"]:
            if b == 0:
                out[kind][b] = clean_pred
                continue
            reps = []
            for i, img in zip(te, imgs):
                label = int(ds.labels[i])

                def donors():             # a training sample from another family
                    pool = tr[ds.labels[tr] != label]
                    return ds.load(int(rng.choice(pool)))

                reps.append(model.represent(perturb(kind, img, b, rng, cfg, model, label, donors)))
            out[kind][b] = model.predict_proba(np.stack(reps)).argmax(1)
    return out


# ---------------------------------------------------------------- one experiment
def prepare_experiment(exp, spec, cfg, out, skipped):
    """Load the dataset, compute (or load cached) representations, and build the
    duplicate groups. Returns (ds, models, reps, groups, info), or None if a
    required descriptor is unavailable."""
    print(f"\n=== {exp}: dataset={spec['dataset']} split={spec['split']}")
    ds = load_dataset(spec["dataset"], cfg)
    y, n_classes = ds.labels, len(ds.families)
    exp_dir = os.path.join(out, exp); os.makedirs(exp_dir, exist_ok=True)

    models = [build(m, cfg) for m in cfg["run"]["models"]]
    needed = {m.name: (m.rep_key(), m.represent) for m in models}
    dd = cfg["dedup"]
    grouped = spec["split"] == "grouped"
    if grouped and dd["feature"] == "thumbnail":
        size = cfg["features"]["thumbnail_size"]
        needed["thumbnail"] = (f"thumb{size}", lambda img: features.thumbnail(img, size))
    elif grouped and "gist_knn" not in needed:
        g = build("gist_knn", cfg)
        needed["gist_knn"] = (g.rep_key(), g.represent)
    reps, hashes = compute_representations(ds, needed, cfg, skipped)
    models = [m for m in models if m.name in reps]

    info = {"dataset": ds.name, "n_samples": len(ds), "n_families": n_classes,
            "families": ds.families, "split": spec["split"],
            "family_counts": np.bincount(y, minlength=n_classes).tolist(),
            "n_exact_duplicate_groups": int(len(set(hashes))),
            "n_folds": cfg["splits"]["n_folds"]}

    groups = None
    if grouped:
        dfeat = reps.get("thumbnail" if dd["feature"] == "thumbnail" else "gist_knn")
        if dfeat is not None:
            dists = nearest_neighbor_distances(dfeat)
            save_nn_distance_hist(dists, dd["near_threshold"],
                                  os.path.join(exp_dir, "nn_distance_hist.png"), dd["feature"])
            info["nn_distance_quantiles"] = {q: float(np.quantile(dists, q))
                                             for q in (0.01, 0.05, 0.1, 0.25, 0.5)}
        elif dd["near_threshold"]:
            skipped.append({"what": exp, "reason": f"near-duplicate grouping needs the "
                            f"'{dd['feature']}' descriptor, which is unavailable"})
            return None
        groups = duplicate_groups(hashes, dfeat, dd["near_threshold"])
        info.update(near_threshold=dd["near_threshold"], dedup_feature=dd["feature"],
                    n_groups=int(groups.max() + 1),
                    duplicate_rate_by_family=duplicate_rate_by_family(groups, y, ds.families),
                    untestable_families=[ds.families[c] for c in untestable_families(groups, y)])
        if info["untestable_families"]:
            print(f"  families in a single duplicate group (excluded from the 'testable' "
                  f"metrics): {info['untestable_families']}")
    return ds, models, reps, groups, info


def run_seed(exp, seed, prepared, cfg, out, skipped, attack_exp):
    """Train and evaluate every model on every fold of one seed; write
    <exp>/runs/seed<seed>.json."""
    ds, models, reps, groups, info = prepared
    y, n_classes = ds.labels, len(ds.families)
    t_seed = time.time()
    set_seed(seed)
    fold = make_folds(y, cfg["splits"]["n_folds"], seed, groups)
    save_folds(os.path.join(out, "splits", f"{exp}_seed{seed}.csv"), ds, fold, groups)
    do_attack = attack_exp is not None
    # Grouped splits: families that form a single duplicate group are never in the
    # training data when tested. "testable" metrics leave them out (all samples of
    # the other families are kept); attack metrics use the same subset.
    untestable = [ds.families.index(f) for f in info.get("untestable_families", [])]
    testable = ~np.isin(y, untestable)
    record = {"exp": exp, "seed": seed, "info": info, "models": {}, "attack": {},
              "mcnemar": [], "skipped": []}
    preds = {}
    for m in models:
        max_folds = cfg["models"].get(m.name, {}).get("max_folds") or cfg["splits"]["n_folds"]
        prob = np.zeros((len(y), n_classes), dtype=np.float32)
        evaluated = np.zeros(len(y), dtype=bool)
        adv_pred = {}
        t0 = time.time()
        for k in range(max_folds):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            m.fit(reps[m.name][tr], y[tr], n_classes, seed * 1000 + k)
            prob[te] = m.predict_proba(reps[m.name][te])
            evaluated[te] = True
            if do_attack:
                try:
                    fold_adv = attack_fold(m, ds, te, tr, prob[te].argmax(1), cfg, seed, k)
                except NotImplementedError as e:
                    skipped.append({"what": attack_exp, "reason": str(e)})
                    do_attack = False
                    continue
                for kind, by_b in fold_adv.items():
                    for b, p in by_b.items():
                        adv_pred.setdefault(kind, {}).setdefault(b, np.full(len(y), -1))[te] = p
        ev = np.where(evaluated)[0]
        met = compute_metrics(y[ev], prob[ev])
        met["fit_predict_seconds"] = time.time() - t0
        met["n_evaluated"] = int(len(ev))
        msg = ""
        if untestable:
            evt = ev[testable[ev]]
            met["testable"] = {k: v for k, v in compute_metrics(y[evt], prob[evt]).items()
                               if k in METRICS}
            met["testable"]["n_evaluated"] = int(len(evt))
            msg = f" | testable families: acc={met['testable']['accuracy']:.4f}"
        record["models"][m.name] = met
        preds[m.name] = (ev, prob.argmax(1))
        print(f"  {exp} seed {seed} {m.name}: acc={met['accuracy']:.4f} "
              f"macroF1={met['f1']:.4f}{msg} ({met['fit_predict_seconds']:.0f}s)")
        if seed == cfg["seeds"][0]:
            save_confusion_matrix(y[ev], prob[ev].argmax(1), ds.families,
                                  os.path.join(out, exp, f"cm_{m.name}.png"), f"{exp} {m.name}")
        if do_attack:
            eva = ev[testable[ev]]
            clean = prob.argmax(1)[eva]
            # per budget, in cfg order: [accuracy, evasion rate], testable families only
            record["attack"][m.name] = {
                kind: [[float((by_b[b][eva] == y[eva]).mean()), evasion_rate(y[eva], clean, by_b[b][eva])]
                       for b in cfg["attack"]["budgets"]]
                for kind, by_b in adv_pred.items()}
    for a, b in combinations(preds, 2):
        common = np.intersect1d(preds[a][0], preds[b][0])
        record["mcnemar"].append({"seed": seed, "a": a, "b": b,
                                  **mcnemar(y[common], preds[a][1][common], preds[b][1][common])})
    record["skipped"] = _dedupe(skipped)
    record["seconds"] = time.time() - t_seed
    _write_json(os.path.join(out, exp, "runs", f"seed{seed}.json"), record)


# ---------------------------------------------------------------- merge
def _load_runs(out, exp, seeds):
    runs, missing = [], []
    for s in seeds:
        p = os.path.join(out, exp, "runs", f"seed{s}.json")
        if os.path.exists(p):
            with open(p) as f:
                runs.append(json.load(f))
        else:
            missing.append(s)
    return runs, missing


def summarize(cfg, out, config_path, extra_skipped=()):
    """Combine <exp>/runs/seed*.json for the configured seeds into
    <exp>/summary.json, the attack summary, figures, and metrics.json."""
    specs, wanted = cfg["experiments"], cfg["run"]["experiments"]
    attack_of = {specs[e]["attack_on"]: e for e in wanted if "attack_on" in specs[e]}
    trained = [e for e in wanted if "attack_on" not in specs[e]]
    trained += [b for b in attack_of if b not in trained]
    skipped, summaries, seconds = list(extra_skipped), {}, 0.0
    budgets = cfg["attack"]["budgets"]

    for exp in trained:
        runs, missing = _load_runs(out, exp, cfg["seeds"])
        if missing:
            print(f"WARNING {exp}: no results for seeds {missing}; summarising {len(runs)} seed(s)")
        if not runs:
            continue
        for r in runs:
            skipped += r["skipped"]
            seconds += r["seconds"]
        summary = {"info": {**runs[0]["info"], "seeds": [r["seed"] for r in runs]},
                   "models": {}, "mcnemar": [t for r in runs for t in r["mcnemar"]]}
        for name in runs[0]["models"]:
            per = [r["models"][name] for r in runs]
            s = {k: mean_std([p[k] for p in per]) for k in METRICS}
            s["per_family_recall"] = np.mean([p["per_family_recall"] for p in per], 0).tolist()
            s["fit_predict_seconds"] = mean_std([p["fit_predict_seconds"] for p in per])
            s["per_seed"] = [{k: p[k] for k in METRICS} for p in per]
            if all("testable" in p for p in per):
                s["testable"] = {k: mean_std([p["testable"][k] for p in per]) for k in METRICS}
                s["testable"]["n_evaluated"] = per[0]["testable"]["n_evaluated"]
            summary["models"][name] = s
        _write_json(os.path.join(out, exp, "summary.json"), summary)
        if exp in wanted:
            summaries[exp] = summary

        attack_exp = attack_of.get(exp)
        adv_runs = [r["attack"] for r in runs if r["attack"]]
        if attack_exp and adv_runs:
            asum = {"budgets": budgets, "seeds": [r["seed"] for r in runs if r["attack"]],
                    "models": {}}
            for name in adv_runs[0]:
                seeds = [a[name] for a in adv_runs if name in a]
                asum["models"][name] = {
                    kind: {"accuracy": [mean_std([s[kind][j][0] for s in seeds]) for j in range(len(budgets))],
                           "evasion_rate": [mean_std([s[kind][j][1] for s in seeds]) for j in range(len(budgets))],
                           "robustness_auc": mean_std([robustness_auc(budgets, [v[0] for v in s[kind]])
                                                       for s in seeds])}
                    for kind in seeds[0]}
            _write_json(os.path.join(out, attack_exp, "summary.json"), asum)
            save_budget_curves(asum["models"], budgets,
                               os.path.join(out, attack_exp, "budget_curves.png"))
            summaries[attack_exp] = asum

    metrics = {"config": config_path, "seeds": cfg["seeds"],
               "compute_seconds": round(seconds, 1)}
    for exp, model in (h.split("/") for h in cfg["run"]["headline"]):
        res = summaries.get(exp, {}).get("models", {}).get(model)
        if res and "accuracy" in res:
            metrics["headline_source"] = f"{exp}/{model} (mean over seeds)"
            metrics.update({k: res[k]["mean"] for k in METRICS})
            break
    metrics["skipped"] = _dedupe(skipped)
    metrics["experiments"] = summaries
    _write_json(os.path.join(out, "metrics.json"), metrics)
    print("\nResults written to", out)
    print(json.dumps({k: metrics.get(k) for k in ("headline_source", *METRICS)}, indent=2))
    for s in metrics["skipped"]:
        print(f"SKIPPED {s['what']}: {s['reason']}")


# ---------------------------------------------------------------- main
def main(config_path: str, experiments=None, seeds=None, prepare=False, merge=False):
    cfg = load_config(config_path)
    out = cfg["output"]["dir"]; os.makedirs(out, exist_ok=True)
    if merge:
        summarize(cfg, out, config_path)
        return
    specs, wanted = cfg["experiments"], cfg["run"]["experiments"]
    attack_of = {specs[e]["attack_on"]: e for e in wanted if "attack_on" in specs[e]}
    to_train = [e for e in wanted if "attack_on" not in specs[e]]
    to_train += [b for b in attack_of if b not in to_train]
    if experiments:
        unknown = set(experiments) - set(to_train)
        if unknown:
            raise SystemExit(f"--experiments {sorted(unknown)} are not trained experiments "
                             f"here (choose from {to_train}; attacks run with their base)")
        to_train = [e for e in to_train if e in experiments]
    run_seeds = seeds if seeds is not None else cfg["seeds"]

    skipped = []
    for exp in to_train:
        prepared = prepare_experiment(exp, specs[exp], cfg, out, skipped)
        if prepared is None or prepare:
            continue
        for seed in run_seeds:
            run_seed(exp, seed, prepared, cfg, out, list(skipped), attack_of.get(exp))
    if prepare:
        print("\nCaches ready.", *(f"SKIPPED {s['what']}: {s['reason']}" for s in _dedupe(skipped)),
              sep="\n")
    elif experiments is None and seeds is None:
        summarize(cfg, out, config_path, skipped)
    else:
        print(f"\nPartial run done ({to_train} x seeds {run_seeds}). "
              "Run with --merge once every job has finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--experiments", nargs="+", help="run only these trained experiments")
    ap.add_argument("--seeds", nargs="+", type=int, help="run only these seeds")
    ap.add_argument("--prepare", action="store_true", help="build descriptor caches only")
    ap.add_argument("--merge", action="store_true", help="summarise existing per-seed runs")
    a = ap.parse_args()
    main(a.config, a.experiments, a.seeds, a.prepare, a.merge)
