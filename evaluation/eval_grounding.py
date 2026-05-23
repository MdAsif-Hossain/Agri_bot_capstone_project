"""Groundedness evaluation: do answers cite real evidence, and do they
contain the keywords we expect from a correct answer?

This drives the live backend (/v1/chat) so the full agent pipeline runs,
including the grounding-policy enforcement node. Start the backend first:
    python api.py

Then in another shell:
    python evaluation/eval_grounding.py
    python evaluation/eval_grounding.py --backend http://localhost:8000
"""

from __future__ import annotations

import argparse
import time

import urllib.request
import urllib.error
import json

from evaluation.common import filter_queries, load_dataset, save_results


def _post(backend: str, query: str, timeout: int = 240) -> dict:
    body = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        f"{backend}/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    ans_lower = (answer or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in ans_lower)
    return round(hits / len(keywords), 3)


def run(backend: str) -> dict:
    dataset = load_dataset()
    queries = filter_queries(
        dataset,
        categories={"in_corpus_factual", "in_corpus_procedural", "dialect_bn"},
    )
    print(f"Scoring grounding for {len(queries)} in-corpus queries via {backend}")

    per_query = []
    keyword_scores = []
    citation_counts = []
    verified_count = 0
    grounding_actions: dict[str, int] = {}

    for i, q in enumerate(queries, start=1):
        try:
            t0 = time.perf_counter()
            resp = _post(backend, q["query"])
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  [{i}/{len(queries)}] {q['id']}: REQUEST FAILED ({e})")
            per_query.append({"id": q["id"], "error": str(e)})
            continue

        answer = resp.get("answer", "")
        citations = resp.get("citations", [])
        is_verified = resp.get("is_verified", False)
        grounding_action = resp.get("grounding_action", "pass")

        kw_score = _keyword_coverage(answer, q.get("expected_answer_keywords", []))
        keyword_scores.append(kw_score)
        citation_counts.append(len(citations))
        if is_verified:
            verified_count += 1
        grounding_actions[grounding_action] = grounding_actions.get(grounding_action, 0) + 1

        per_query.append(
            {
                "id": q["id"],
                "category": q["category"],
                "keyword_coverage": kw_score,
                "citation_count": len(citations),
                "citations": citations[:3],
                "is_verified": is_verified,
                "grounding_action": grounding_action,
                "latency_ms": round(elapsed_ms, 1),
                "answer_preview": (answer or "")[:200],
            }
        )
        print(
            f"  [{i}/{len(queries)}] {q['id']}: kw={kw_score} cites={len(citations)} "
            f"verified={is_verified} action={grounding_action} ({elapsed_ms / 1000:.1f}s)"
        )

    n = len(keyword_scores) or 1
    summary = {
        "queries_evaluated": len(per_query),
        "avg_keyword_coverage": round(sum(keyword_scores) / n, 3),
        "avg_citation_count": round(sum(citation_counts) / n, 2),
        "verified_rate": round(verified_count / n, 3),
        "grounding_action_breakdown": grounding_actions,
    }

    print("\n=== Grounding results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    payload = {"summary": summary, "per_query": per_query}
    out = save_results("grounding", payload)
    print(f"\nResults written to: {out}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://localhost:8000")
    args = p.parse_args()
    run(args.backend)


if __name__ == "__main__":
    main()
