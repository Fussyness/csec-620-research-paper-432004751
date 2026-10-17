#!/usr/bin/env python3
"""Automated grading harness for CSEC 520/620 project repos.

Runs the OBJECTIVE checks a grading agent needs as evidence, then writes
grading/auto_report.json. The agent reads that file, adds qualitative judgment
(report quality, correctness, security relevance), and produces the final grade
per grading/AGENT_GRADING.md.

Stdlib only. Run from the repo root:  python grading/grade.py
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["config.yaml", "src/train.py", "src/data.py", "src/model.py",
            "src/evaluate.py", "README.md", "requirements.txt"]
METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}

# Prefer the project venv if it exists, so the harness works whether or not it's activated.
_venv = ROOT / ".venv" / "bin" / "python"
PYEXE = str(_venv) if _venv.exists() else sys.executable


def run(cmd, timeout=900):
    """Run a command from ROOT; return (returncode, tail_of_output)."""
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
        out = (p.stdout + p.stderr)[-4000:]
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return 127, str(e)


def check_structure():
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    return {"id": "structure", "status": "pass" if not missing else "fail",
            "evidence": "all present" if not missing else f"missing: {missing}"}


def check_tests():
    rc, out = run([PYEXE, "-m", "pytest", "-q"], timeout=300)
    status = "pass" if rc == 0 else ("dep_missing" if "No module named" in out else "fail")
    return {"id": "tests", "status": status, "returncode": rc, "evidence": out[-800:]}


def check_reproduce():
    rc, out = run([PYEXE, "-m", "src.train", "--config", "config.yaml"])
    if rc == 0:
        status = "pass"
    elif "No module named" in out:
        status = "dep_missing"   # e.g., torch not installed in grader env
    else:
        status = "fail"
    return {"id": "reproduce_runs", "status": status, "returncode": rc,
            "evidence": out[-1200:]}


def check_metrics():
    p = ROOT / "results" / "metrics.json"
    if not p.exists():
        return {"id": "metrics_present", "status": "fail",
                "evidence": "results/metrics.json not found"}
    try:
        m = json.loads(p.read_text())
    except Exception as e:
        return {"id": "metrics_present", "status": "fail", "evidence": f"invalid JSON: {e}"}
    missing = METRIC_KEYS - set(m)
    bad = {k: v for k, v in m.items() if isinstance(v, (int, float)) and not (0 <= v <= 1)}
    ok = not missing and not bad
    return {"id": "metrics_present", "status": "pass" if ok else "fail",
            "evidence": {"metrics": m, "missing_keys": sorted(missing), "out_of_range": bad}}


def check_leakage():
    """Heuristic scan for common data-leakage patterns (agent should confirm)."""
    hits = []
    patterns = [r"fit_transform\s*\(\s*X\s*\)", r"\.fit\s*\(\s*X\s*[\),]",
                r"StandardScaler\(\)\.fit\(X\)"]
    for pyf in (ROOT / "src").glob("*.py"):
        text = pyf.read_text(errors="ignore")
        for pat in patterns:
            for m in re.finditer(pat, text):
                hits.append(f"{pyf.name}: {m.group(0)}")
    return {"id": "leakage_scan",
            "status": "review" if hits else "clean",
            "evidence": hits or "no obvious full-data fit before split (heuristic only)"}


def check_report():
    rdir = ROOT / "report"
    files = [p.name for p in rdir.glob("**/*") if p.suffix.lower() in {".tex", ".pdf", ".md"}] if rdir.exists() else []
    has_real = any(f.lower().endswith((".tex", ".pdf")) for f in files)
    return {"id": "report_present",
            "status": "pass" if has_real else "review",
            "evidence": f"report files: {files}" if files else "no report/ .tex or .pdf found"}


def check_git_hygiene():
    gi = (ROOT / ".gitignore")
    txt = gi.read_text() if gi.exists() else ""
    committed_data = [p.name for p in (ROOT / "data").glob("*")
                      if p.name not in {"README.md", ".gitkeep"}] if (ROOT / "data").exists() else []
    ok = "data/" in txt and "results/" in txt
    return {"id": "git_hygiene", "status": "pass" if ok else "review",
            "evidence": {"gitignore_covers_data_results": ok,
                         "possible_committed_data": committed_data}}


def main():
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    checks = [check_structure(), check_tests(), check_reproduce(), check_metrics(),
              check_leakage(), check_report(), check_git_hygiene()]
    report = {
        "generated": started,
        "repo": str(ROOT.name),
        "checks": checks,
        "note_for_agent": (
            "These are OBJECTIVE signals only. status 'dep_missing' means the grader "
            "environment lacked a dependency (e.g., torch) — set up the env and re-run "
            "before penalizing reproducibility. Confirm heuristic flags (leakage_scan) "
            "by reading the code. Then score every rubric criterion per AGENT_GRADING.md."
        ),
    }
    out = ROOT / "grading" / "auto_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
