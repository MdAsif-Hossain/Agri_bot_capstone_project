"""End-to-end evaluation orchestrator.

Runs all five eval scripts in order, captures progress, and writes a single
summary file at evaluation/results/SUMMARY_latest.json.

Steps:
  1. eval_retrieval   — offline, fast (uses indexes only)
  2. eval_grounding   — needs backend; ~30-60 sec/query × ~50 queries
  3. eval_refusal     — needs backend; ~30-60 sec/query × 60 queries
  4. eval_latency     — needs backend; --runs 3 per query × ~30 queries
  5. stress_test      — needs backend; concurrency=5 × total=20

Usage:
    # Terminal 1
    python api.py

    # Terminal 2
    python evaluation/run_all.py
    python evaluation/run_all.py --skip retrieval grounding   # to re-run only some
    python evaluation/run_all.py --backend http://localhost:8000

The orchestrator pings /v1/health before each backend-dependent step and
aborts cleanly if the backend isn't reachable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
PY = sys.executable  # use the same interpreter we're running under


STEPS = [
    # (step_name, script, needs_backend, extra_args)
    ("retrieval", "evaluation/eval_retrieval.py", False, []),
    ("grounding", "evaluation/eval_grounding.py", True, []),
    ("refusal", "evaluation/eval_refusal.py", True, []),
    ("latency", "evaluation/eval_latency.py", True, ["--runs", "3"]),
    ("stress", "evaluation/stress_test.py", True, ["--concurrency", "5", "--total", "20"]),
]


def _backend_alive(backend: str) -> bool:
    try:
        with urllib.request.urlopen(f"{backend}/v1/health", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def _run_step(name: str, script: str, extra_args: list[str]) -> tuple[bool, float]:
    print("\n" + "=" * 70)
    print(f"  STEP: {name}  ({script})")
    print("=" * 70)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PY, "-u", script, *extra_args],
        cwd=str(ROOT),
        text=True,
    )
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    mark = "✓" if ok else "✗"
    print(f"\n  {mark} {name} exit={proc.returncode} ({elapsed / 60:.1f} min)")
    return ok, elapsed


def _load_latest(name: str) -> dict | None:
    p = RESULTS / f"{name}_latest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://localhost:8000")
    ap.add_argument(
        "--skip", nargs="*", default=[], choices=[s[0] for s in STEPS],
        help="Skip these steps (e.g. --skip stress)",
    )
    ap.add_argument(
        "--only", nargs="*", default=[], choices=[s[0] for s in STEPS],
        help="Run only these steps (overrides --skip when set)",
    )
    args = ap.parse_args()

    selected = [s for s in STEPS if (
        (args.only and s[0] in args.only) or
        (not args.only and s[0] not in args.skip)
    )]
    print(f"Running {len(selected)} step(s): {[s[0] for s in selected]}")

    timings: dict[str, float] = {}
    statuses: dict[str, bool] = {}

    grand_start = time.perf_counter()

    for name, script, needs_backend, extra_args in selected:
        if needs_backend and not _backend_alive(args.backend):
            print(f"\n  ✗ Backend not reachable at {args.backend} — aborting {name} (and remaining steps)")
            statuses[name] = False
            break
        ok, elapsed = _run_step(name, script, extra_args)
        timings[name] = round(elapsed, 1)
        statuses[name] = ok
        if not ok:
            print(f"\n  ✗ {name} failed — continuing with remaining steps")

    grand_elapsed = time.perf_counter() - grand_start

    # ---- Consolidated summary ----
    print("\n" + "=" * 70)
    print(f"  ALL DONE in {grand_elapsed / 60:.1f} min")
    print("=" * 70)

    summary = {
        "ran_at_unix": int(time.time()),
        "total_minutes": round(grand_elapsed / 60, 2),
        "step_status": statuses,
        "step_timings_sec": timings,
        "results_snapshot": {},
    }
    for name, _, _, _ in selected:
        snap = _load_latest(name)
        if snap is not None and "summary" in snap:
            summary["results_snapshot"][name] = snap["summary"]
            print(f"\n--- {name} summary ---")
            for k, v in snap["summary"].items():
                print(f"  {k}: {v}")
        else:
            print(f"\n--- {name} summary: NOT AVAILABLE ---")

    out = RESULTS / f"SUMMARY_{int(time.time())}.json"
    latest = RESULTS / "SUMMARY_latest.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    latest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote consolidated summary: {out.relative_to(ROOT)}")
    print(f"Also: {latest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
