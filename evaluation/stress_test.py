"""Concurrent-request stress test.

Floods the backend with N concurrent /v1/chat requests sampled from the eval
dataset to verify:
  - the asyncio.Semaphore-based queuing behaves correctly (no crashes, no
    runaway parallelism on the 4 GB-VRAM GPU)
  - the rate-limiter middleware returns clean 429s rather than dropping
    connections
  - no OOM under sustained load

Usage:
    python api.py
    python evaluation/stress_test.py --concurrency 5 --total 20
    python evaluation/stress_test.py --concurrency 10 --total 50 --queries 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import urllib.error
import urllib.request

from evaluation.common import filter_queries, load_dataset, percentile, save_results


def _post_sync(backend: str, query: str, timeout: int = 300) -> tuple[int, float, str]:
    """Returns (status, wall_ms, error_kind)."""
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        f"{backend}/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            wall_ms = (time.perf_counter() - t0) * 1000
            return resp.status, wall_ms, ""
    except urllib.error.HTTPError as e:
        wall_ms = (time.perf_counter() - t0) * 1000
        return e.code, wall_ms, f"http{e.code}"
    except (urllib.error.URLError, TimeoutError) as e:
        wall_ms = (time.perf_counter() - t0) * 1000
        return 0, wall_ms, type(e).__name__


async def _run_one(loop, executor, backend: str, query: str):
    return await loop.run_in_executor(executor, _post_sync, backend, query)


async def main_async(backend: str, concurrency: int, total: int, n_queries: int):
    from concurrent.futures import ThreadPoolExecutor

    dataset = load_dataset()
    queries = filter_queries(
        dataset, categories={"in_corpus_factual", "in_corpus_procedural"}
    )[:n_queries]
    if not queries:
        raise SystemExit("No queries selected for stress test")

    print(
        f"Stress test: total={total}, concurrency={concurrency}, "
        f"distinct queries={len(queries)} via {backend}"
    )

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=concurrency)
    sem = asyncio.Semaphore(concurrency)
    statuses: list[int] = []
    walls: list[float] = []
    error_kinds: dict[str, int] = {}

    async def bound(idx: int):
        async with sem:
            q = queries[idx % len(queries)]
            status, wall_ms, err = await _run_one(loop, executor, backend, q["query"])
            statuses.append(status)
            walls.append(wall_ms)
            if err:
                error_kinds[err] = error_kinds.get(err, 0) + 1
            mark = "OK" if status == 200 else f"ERR({status or err})"
            print(f"  [{idx + 1:>3}/{total}] {q['id']} {mark} {wall_ms / 1000:.2f}s")

    start = time.perf_counter()
    await asyncio.gather(*(bound(i) for i in range(total)))
    elapsed = time.perf_counter() - start

    success = sum(1 for s in statuses if s == 200)
    rate_limited = sum(1 for s in statuses if s == 429)
    other_errors = total - success - rate_limited

    summary = {
        "total_requests": total,
        "concurrency": concurrency,
        "wallclock_sec": round(elapsed, 1),
        "throughput_rps": round(total / elapsed, 2) if elapsed else 0.0,
        "success_count": success,
        "rate_limited_count": rate_limited,
        "other_errors_count": other_errors,
        "error_breakdown": error_kinds,
        "wall_ms": {
            "p50": percentile(walls, 0.5),
            "p95": percentile(walls, 0.95),
            "max": round(max(walls), 1) if walls else 0.0,
        },
    }

    print("\n=== Stress test results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    payload = {"summary": summary}
    out = save_results("stress", payload)
    print(f"\nResults written to: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--total", type=int, default=20)
    p.add_argument(
        "--queries",
        type=int,
        default=5,
        help="Number of distinct queries to draw from (cycles through)",
    )
    args = p.parse_args()
    asyncio.run(main_async(args.backend, args.concurrency, args.total, args.queries))


if __name__ == "__main__":
    main()
