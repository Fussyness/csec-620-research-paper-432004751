"""Generate the paper's LaTeX tables and figures from results/ so that every
number in the report comes straight from metrics.json.

    python -m src.tables --results results      (also run by `make reproduce`)

Writes report/generated/*.tex and copies figures into report/figures/.
"""
from __future__ import annotations
import argparse, json, os, shutil

MODEL_NAMES = {"gist_knn": "GIST + $k$-NN [1]", "cnn": "ResNet (fine-tuned)",
               "byte_hist_lr": "Byte histogram + LR"}
ATTACK_NAMES = {"random": "Random", "zero": "Zero pad", "other_family": "Other family",
                "fgsm": "FGSM (white-box)"}


def pm(s, scale=100, digits=2):
    return f"{scale * s['mean']:.{digits}f} $\\pm$ {scale * s['std']:.{digits}f}"


def esc(s):
    return s.replace("_", r"\_").replace("&", r"\&")


def _has_testable(exp: dict) -> bool:
    return any("testable" in r for r in exp["models"].values())


def main_results(exps: dict) -> str:
    """E1-E3 accuracy and macro-F1. For grouped experiments with families that form a
    single duplicate group, the main column covers the testable families only (marked
    with a dagger) and an extra "all families" column follows."""
    base = [e for e in ("e1", "e2", "e3") if e in exps]
    cols = []                                   # (exp, use_testable, header)
    for e in base:
        if _has_testable(exps[e]):
            cols += [(e, True, e.upper() + r"$^\dagger$"), (e, False, e.upper() + " (all fam.)")]
        else:
            cols.append((e, False, e.upper()))
    models = [m for m in MODEL_NAMES if any(m in exps[e]["models"] for e in base)]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for _, _, h in cols)
    lines = [r"\begin{tabular}{l" + "cc" * len(cols) + "}", r"\toprule",
             f"Model & {head} \\\\",
             " & " + " & ".join(["Acc. (\\%)", "Macro-F1 (\\%)"] * len(cols)) + r" \\", r"\midrule"]
    for m in models:
        cells = []
        for e, use_t, _ in cols:
            r = exps[e]["models"].get(m)
            if r and use_t:
                r = r.get("testable")
            cells += [pm(r["accuracy"]), pm(r["f1"])] if r else ["--", "--"]
        lines.append(f"{MODEL_NAMES[m]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    notes = sorted({f for e in base for f in exps[e]["info"].get("untestable_families", [])})
    if notes:
        lines.append(f"\\multicolumn{{{1 + 2 * len(cols)}}}{{l}}{{\\footnotesize $^\\dagger$Excluded "
                     f"(single duplicate group, never in training when tested): "
                     + ", ".join(esc(f) for f in notes) + r".} \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def per_family(exps: dict, model: str) -> str:
    e1, e2 = exps["e1"], exps["e2"]
    fams = e1["info"]["families"]
    dup = e2["info"].get("duplicate_rate_by_family", {})
    r1 = e1["models"][model]["per_family_recall"]
    r2 = e2["models"][model]["per_family_recall"]
    untestable = set(e2["info"].get("untestable_families", []))
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Family & $n$ & Dup. (\%) & Recall E1 & Recall E2 \\", r"\midrule"]
    for i, fam in enumerate(fams):
        mark = r"$^\ddagger$" if fam in untestable else ""
        lines.append(f"{esc(fam)}{mark} & {e1['info']['family_counts'][i]} & {100 * dup.get(fam, 0):.1f} & "
                     f"{100 * r1[i]:.1f} & {100 * r2[i]:.1f} \\\\")
    lines.append(r"\bottomrule")
    if untestable:
        lines.append(r"\multicolumn{5}{l}{\footnotesize $^\ddagger$One duplicate group: "
                     r"never in training when tested (E2).} \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def robustness(e4: dict) -> str:
    attacks = [a for a in ATTACK_NAMES if any(a in v for v in e4["models"].values())]
    lines = [r"\begin{tabular}{l" + "c" * len(attacks) + "}", r"\toprule",
             "Model & " + " & ".join(ATTACK_NAMES[a] for a in attacks) + r" \\", r"\midrule"]
    for m, per in e4["models"].items():
        cells = [pm(per[a]["robustness_auc"], digits=1) if a in per else "--" for a in attacks]
        lines.append(f"{MODEL_NAMES.get(m, esc(m))} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def macros(metrics: dict) -> str:
    """\\newcommand per headline number, e.g. \\EOneGistKnnAcc, for use in prose.
    Where testable-family metrics exist (grouped splits), the plain macro is the
    testable value and the ...All macro (e.g. \\ETwoGistKnnAccAll) covers all families."""
    out = []
    for e, s in metrics["experiments"].items():
        for m, r in s.get("models", {}).items():
            if "accuracy" not in r:
                continue
            name = e.upper().replace("1", "One").replace("2", "Two").replace("3", "Three") + \
                m.replace("_", " ").title().replace(" ", "")
            variants = [("", r.get("testable", r))] + ([("All", r)] if "testable" in r else [])
            for suffix, src in variants:
                for k, short in (("accuracy", "Acc"), ("f1", "Fone")):
                    out.append(f"\\newcommand{{\\{name}{short}{suffix}}}{{{100 * src[k]['mean']:.2f}\\%}}")
    return "\n".join(out)


def main(results: str, report: str = "report"):
    with open(os.path.join(results, "metrics.json")) as f:
        metrics = json.load(f)
    exps = metrics["experiments"]
    gen, figs = os.path.join(report, "generated"), os.path.join(report, "figures")
    os.makedirs(gen, exist_ok=True); os.makedirs(figs, exist_ok=True)
    files = {"numbers.tex": macros(metrics)}
    if any(e in exps for e in ("e1", "e2", "e3")):
        files["table_main.tex"] = main_results(exps)
    if "e1" in exps and "e2" in exps:
        model = next((m for m in MODEL_NAMES if m in exps["e1"]["models"]
                      and m in exps["e2"]["models"]), None)
        if model:
            files["table_per_family.tex"] = per_family(exps, model)
    if "e4" in exps:
        files["table_robustness.tex"] = robustness(exps["e4"])
    for name, body in files.items():
        with open(os.path.join(gen, name), "w", encoding="utf-8") as f:
            f.write(f"% Generated by src/tables.py from {results}/metrics.json; do not edit.\n{body}\n")
    for root, _, names in os.walk(results):
        for n in names:
            if n.endswith(".png"):
                rel = os.path.relpath(os.path.join(root, n), results).replace(os.sep, "_")
                shutil.copy(os.path.join(root, n), os.path.join(figs, rel))
    print(f"Wrote {sorted(files)} to {gen}; figures to {figs}")
    if metrics.get("skipped"):
        print("Note: results are incomplete; skipped:", [s["what"] for s in metrics["skipped"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--report", default="report")
    a = ap.parse_args()
    main(a.results, a.report)
