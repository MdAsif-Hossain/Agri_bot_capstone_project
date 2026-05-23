"""Latency evaluation: end-to-end per-stage and total wall-clock latency.

For each in-corpus query, sends a /v1/chat request to the live backend and
records the diagnostics.timings_ms block (per-stage breakdown) plus total
wall-clock. Reports p50, p95, and per-stage averages.

Usage:
    python api.py        # in one shell
    python evaluation/eval_latency.py --runs 3
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict

from evaluation.common import (
    filter_queries,
    load_dataset,
    percentile,
    save_results,
)


def _post(backend: str, query: str, timeout: int = 240) -> tuple[dict, float]:
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        f"{backend}/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - t0) * 1000
    return body, wall_ms


def run(backend: str, runs_per_query: int) -> dict:
    dataset = load_dataset()
    queries = filter_queries(
        dataset, categories={"in_corpus_factual", "in_corpus_procedural"}
    )
    print(
        f"Latency over {len(queries)} in-corpus queries × {runs_per_query} runs via {backend}"
    )

    wall_times: list[float] = []
    stage_times: dict[str, list[float]] = defaultdict(list)
    per_query = []

    for i, q in enumerate(queries, start=1):
        q_walls: list[float] = []
        q_stages: dict[str, list[float]] = defaultdict(list)
        for r in range(runs_per_query):
            try:
                resp, wall_ms = _post(backend, q["query"])
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  [{i}/{len(queries)} run {r + 1}] {q['id']}: FAILED ({e})")
                continue
            q_walls.append(wall_ms)
            wall_times.append(wall_ms)
            timings = resp.get("diagnostics", {}).get("timings_ms", {}) or {}
            for stage, val in timings.items():
                try:
                    val_f = float(val)
                except (TypeError, ValueError):
                    continue
                stage_times[stage].append(val_f)
                q_stages[stage].append(val_f)

        if q_walls:
            per_query.append(
                {
                    "id": q["id"],
                    "wall_ms_mean": round(sum(q_walls) / len(q_walls), 1),
                    "stage_means_ms": {
                        s: round(sum(v) / len(v), 1) for s, v in q_stages.items()
                    },
                }
            )
            print(
                f"  [{i}/{len(queries)}] {q['id']}: "
                f"mean wall = {sum(q_walls) / len(q_walls) / 1000:.2f}s"
            )

    summary = {
        "queries_evaluated": len(per_query),
        "runs_per_query": runs_per_query,
        "wall_ms": {
            "p50": percentile(wall_times, 0.5),
            "p95": percentile(wall_times, 0.95),
            "mean": round(sum(wall_times) / max(1, len(wall_times)), 1),
        },
        "stage_means_ms": {
            stage: round(sum(vals) / len(vals), 1)
            for stage, vals in stage_times.items()
        },
    }

    print("\n=== Latency results ===")
    print(f"  Wall p50: {summary['wall_ms']['p50']} ms")
    print(f"  Wall p95: {summary['wall_ms']['p95']} ms")
    print(f"  Wall mean: {summary['wall_ms']['mean']} ms")
    print("  Per stage (mean ms):")
    for stage, mean in sorted(summary["stage_means_ms"].items(), key=lambda kv: -kv[1]):
        print(f"    {stage:20s} {mean}")

    payload = {"summary": summary, "per_query": per_query}
    out = save_results("latency", payload)
    print(f"\nResults written to: {out}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://localhost:8000")
    p.add_argument("--runs", type=int, default=3, help="Runs per query for stable stats")
    args = p.parse_args()
    run(args.backend, args.runs)


if __name__ == "__main__":
    main()
