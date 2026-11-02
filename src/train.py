"""Single entry point: `python -m src.train --config config.yaml`.

Runs every experiment listed in `run.experiments` for every seed and fold and
writes, under `output.dir`:

  splits/<exp>_seed<s>.csv     fold (and duplicate-group) assignment per sample
  <exp>/summary.json           mean ± std over seeds for every model
  <exp>/cm_<model>.png         confusion matrix (seed 0, pooled out-of-fold)
  e2/nn_distance_hist.png      near-duplicate threshold tuning aid
  e4/budget_curves.png         accuracy vs. append budget
  metrics.json                 headline numbers + all summaries + anything skipped

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
                   nearest_neighbor_distances, pixel_sha256, save_folds)
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


# ---------------------------------------------------------------- representations
def _cache_path(cfg, ds, key):
    ident = hashlib.sha1(("|".join(ds.names) + key).encode()).hexdigest()[:12]
    return os.path.join(cfg["data"]["cache_dir"], f"{ds.name}_{ident}.npy")


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
            np.save(hash_path, np.array(hashes))
        for name, (path, _, vals) in todo.items():
            reps[name] = np.stack(vals)
            if use_cache:
                np.save(path, reps[name])
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
            for i in te:
                img, label = ds.load(i), int(ds.labels[i])

                def donors():             # a training sample from another family
                    pool = tr[ds.labels[tr] != label]
                    return ds.load(int(rng.choice(pool)))

                reps.append(model.represent(perturb(kind, img, b, rng, cfg, model, label, donors)))
            out[kind][b] = model.predict_proba(np.stack(reps)).argmax(1)
    return out


# ---------------------------------------------------------------- one experiment
def run_experiment(exp, spec, cfg, out, skipped, attack_exp=None):
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
            "n_folds": cfg["splits"]["n_folds"], "seeds": cfg["seeds"]}

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
            return None, None
        groups = duplicate_groups(hashes, dfeat, dd["near_threshold"])
        info.update(near_threshold=dd["near_threshold"], dedup_feature=dd["feature"],
                    n_groups=int(groups.max() + 1),
                    duplicate_rate_by_family=duplicate_rate_by_family(groups, y, ds.families))

    do_attack = attack_exp is not None
    per_seed = {m.name: [] for m in models}
    adv = {m.name: [] for m in models}     # per seed: {attack: {budget: (acc, evasion)}}
    tests = []
    for seed in cfg["seeds"]:
        set_seed(seed)
        fold = make_folds(y, cfg["splits"]["n_folds"], seed, groups)
        save_folds(os.path.join(out, "splits", f"{exp}_seed{seed}.csv"), ds, fold, groups)
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
            per_seed[m.name].append(met)
            preds[m.name] = (ev, prob.argmax(1))
            print(f"  seed {seed} {m.name}: acc={met['accuracy']:.4f} "
                  f"macroF1={met['f1']:.4f} ({met['fit_predict_seconds']:.0f}s)")
            if seed == cfg["seeds"][0]:
                save_confusion_matrix(y[ev], prob[ev].argmax(1), ds.families,
                                      os.path.join(exp_dir, f"cm_{m.name}.png"), f"{exp} {m.name}")
            if do_attack:
                clean = prob.argmax(1)[ev]
                adv[m.name].append({kind: {b: (float((p[ev] == y[ev]).mean()),
                                               evasion_rate(y[ev], clean, p[ev]))
                                           for b, p in by_b.items()}
                                    for kind, by_b in adv_pred.items()})
        for a, b in combinations(preds, 2):
            common = np.intersect1d(preds[a][0], preds[b][0])
            tests.append({"seed": seed, "a": a, "b": b,
                          **mcnemar(y[common], preds[a][1][common], preds[b][1][common])})

    summary = {"info": info, "models": {}, "mcnemar": tests}
    for name, runs in per_seed.items():
        s = {k: mean_std([r[k] for r in runs]) for k in METRICS}
        s["per_family_recall"] = np.mean([r["per_family_recall"] for r in runs], 0).tolist()
        s["fit_predict_seconds"] = mean_std([r["fit_predict_seconds"] for r in runs])
        s["per_seed"] = [{k: r[k] for k in METRICS} for r in runs]
        summary["models"][name] = s
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    attack_summary = None
    if do_attack and any(adv.values()):
        budgets = cfg["attack"]["budgets"]
        attack_summary = {"budgets": budgets, "models": {}}
        for name, seeds in adv.items():
            attack_summary["models"][name] = {
                kind: {"accuracy": [mean_std([s[kind][b][0] for s in seeds]) for b in budgets],
                       "evasion_rate": [mean_std([s[kind][b][1] for s in seeds]) for b in budgets],
                       "robustness_auc": mean_std([robustness_auc(budgets, [s[kind][b][0] for b in budgets])
                                                   for s in seeds])}
                for kind in seeds[0]} if seeds else {}
        adir = os.path.join(out, attack_exp); os.makedirs(adir, exist_ok=True)
        with open(os.path.join(adir, "summary.json"), "w") as f:
            json.dump(attack_summary, f, indent=2)
        save_budget_curves(attack_summary["models"], budgets, os.path.join(adir, "budget_curves.png"))
    return summary, attack_summary


# ---------------------------------------------------------------- main
def main(config_path: str):
    cfg = load_config(config_path)
    out = cfg["output"]["dir"]; os.makedirs(out, exist_ok=True)
    specs, wanted = cfg["experiments"], cfg["run"]["experiments"]
    attack_of = {specs[e]["attack_on"]: e for e in wanted if "attack_on" in specs[e]}
    to_train = [e for e in wanted if "attack_on" not in specs[e]]
    to_train += [b for b in attack_of if b not in to_train]

    skipped, summaries = [], {}
    t0 = time.time()
    for exp in to_train:
        s, a = run_experiment(exp, specs[exp], cfg, out, skipped, attack_of.get(exp))
        if s is not None and exp in wanted:
            summaries[exp] = s
        if a is not None:
            summaries[attack_of[exp]] = a

    metrics = {"config": config_path, "seeds": cfg["seeds"],
               "wall_clock_seconds": round(time.time() - t0, 1)}
    for exp, model in (h.split("/") for h in cfg["run"]["headline"]):
        res = summaries.get(exp, {}).get("models", {}).get(model)
        if res and "accuracy" in res:
            metrics["headline_source"] = f"{exp}/{model} (mean over seeds)"
            metrics.update({k: res[k]["mean"] for k in METRICS})
            break
    skipped = [dict(t) for t in dict.fromkeys(tuple(s.items()) for s in skipped)]
    metrics["skipped"] = skipped
    metrics["experiments"] = summaries
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nResults written to", out)
    print(json.dumps({k: metrics.get(k) for k in ("headline_source", *METRICS)}, indent=2))
    for s in skipped:
        print(f"SKIPPED {s['what']}: {s['reason']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
