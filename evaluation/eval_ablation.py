"""Ablation + naive-baseline evaluation (fix-list items 1 & 2).

Runs the SAME 60-query benchmark under four configurations and scores every
configuration with ONE harness so the rows are directly comparable:

  A  Dense only            FAISS dense retrieval, top-3 → direct generation.
                           No BM25/RRF, no rerank, no KG, no agentic loop,
                           NO refusal policy.  (== the naive RAG baseline.)
  B  + BM25 + RRF          Hybrid dense+sparse fusion, top-3 → generation.
                           No rerank, no KG, single-pass.  + refusal policy.
  C  + Rerank + KG (1-pass) Full retrieval (hybrid + cross-encoder rerank +
                           KG query expansion), single-pass generation
                           (no grade→rewrite→verify loop).  + refusal policy.
  D  Full proposed         The real compiled LangGraph agent (grade→rewrite
                           retry, verify, enforce_policy).  This row is the
                           shipped system, invoked exactly as the API does.

Metrics, computed identically for every config:
  Recall@3, MRR   retrieval hit of an expected source (±2-page tolerance),
                  scored on the 46 in-corpus/dialect queries that carry
                  expected_sources.
  Groundedness    mean keyword coverage of the generated answer over the
                  expected_answer_keywords (the only groundedness proxy that
                  is identical across configs — verified_rate needs the verify
                  node that only D runs).
  Accuracy        refusal correctness over all 60 queries: (TP+TN)/N against
                  the must_refuse gold label, using the same refusal detector
                  as eval_refusal.py.

This is SLOW: ~50 s/query on CPU for the generating configs. A+B+C+D over
60 queries is ~180+ generations ≈ a few hours. Run in the background:

    python evaluation/eval_ablation.py            # all configs, all queries
    python evaluation/eval_ablation.py --smoke    # 2 queries/config, wiring test
    python evaluation/eval_ablation.py --configs A B   # subset

Writes evaluation/results/ablation_latest.json (+ a timestamped copy).
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

from evaluation.common import load_dataset, save_results

# Same refusal markers eval_refusal.py uses, kept in sync deliberately.
REFUSE_MARKERS = [
    "i don't know",
    "i dont know",
    "i cannot answer",
    "cannot provide",
    "consult your local",
    "extension officer",
    "i'm unable",
    "no information",
    "জানি না",
    "জানিনা",
    "তথ্য পাইনি",
    "যাচাই",
]

RETRIEVAL_CATEGORIES = {"in_corpus_factual", "in_corpus_procedural", "dialect_bn"}


# ---------------------------------------------------------------------------
# Scoring helpers (mirrors eval_retrieval / eval_grounding / eval_refusal)
# ---------------------------------------------------------------------------
def _chunk_meta(ev) -> tuple[str, int]:
    inner = getattr(ev, "chunk", ev)
    return str(getattr(inner, "source_file", "")), int(getattr(inner, "page_num", -1))


def _hit_rank(retrieved, expected_sources) -> int | None:
    expected_map: dict[str, set[int]] = defaultdict(set)
    for src in expected_sources:
        expected_map[src["source_file"]].update(src.get("expected_pages", []))
    for rank, ch in enumerate(retrieved, start=1):
        src, page = _chunk_meta(ch)
        if src in expected_map:
            return rank  # source match; page tolerance handled below
    return None


def _hit_rank_pagetol(retrieved, expected_sources, tol: int = 2) -> int | None:
    """1-based rank of first chunk whose source matches and page within ±tol."""
    expected_map: dict[str, set[int]] = defaultdict(set)
    for src in expected_sources:
        expected_map[src["source_file"]].update(src.get("expected_pages", []))
    for rank, ch in enumerate(retrieved, start=1):
        src, page = _chunk_meta(ch)
        if src in expected_map and any(abs(page - ep) <= tol for ep in expected_map[src]):
            return rank
    return None


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    ans = (answer or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in ans)
    return hits / len(keywords)


def _looks_like_refusal(answer: str, grounding_action: str) -> bool:
    if grounding_action == "refuse":
        return True
    ans = (answer or "").lower()
    return any(m in ans for m in REFUSE_MARKERS)


# ---------------------------------------------------------------------------
# Service loading (single load, reused across all configs)
# ---------------------------------------------------------------------------
def _load_services():
    from config import settings
    from sentence_transformers import SentenceTransformer
    from agribot.llm.engine import get_llm
    from agribot.ingestion.index_builder import IndexBundle
    from agribot.retrieval.hybrid import HybridRetriever
    from agribot.retrieval.reranker import Reranker
    from agribot.knowledge_graph.schema import KnowledgeGraph
    from agribot.knowledge_graph.entity_linker import EntityLinker
    from agribot.knowledge_graph.seed_data import seed_knowledge_graph
    from agribot.translation.bangla_t5 import get_translator
    from agribot.agent.graph import build_agent_graph

    print("Loading LLM...")
    llm = get_llm(
        model_path=str(settings.MODEL_PATH),
        n_ctx=settings.LLM_N_CTX,
        n_gpu_layers=settings.LLM_N_GPU_LAYERS,
    )
    print("Loading indexes + embedder...")
    bundle = IndexBundle.load(settings.INDEX_DIR)
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    # Hybrid retriever (configs B/C/D). Pull a generous candidate pool.
    hybrid = HybridRetriever(
        index_bundle=bundle,
        embedding_model=embedder,
        dense_top_k=max(15, settings.DENSE_TOP_K),
        sparse_top_k=max(15, settings.SPARSE_TOP_K),
        dense_weight=settings.DENSE_WEIGHT,
        sparse_weight=settings.SPARSE_WEIGHT,
    )
    # Dense-only retriever (config A): sparse weight 0 ⇒ pure FAISS ordering.
    dense_only = HybridRetriever(
        index_bundle=bundle,
        embedding_model=embedder,
        dense_top_k=max(15, settings.DENSE_TOP_K),
        sparse_top_k=max(15, settings.SPARSE_TOP_K),
        dense_weight=1.0,
        sparse_weight=0.0,
    )
    print("Loading reranker...")
    reranker = Reranker(threshold=settings.RERANK_THRESHOLD, top_n=settings.RERANK_TOP_N)

    print("Loading KG + entity linker...")
    kg = KnowledgeGraph(settings.KG_DB_PATH)
    seed_knowledge_graph(kg)
    entity_linker = EntityLinker(kg)

    print("Loading translator...")
    translator = get_translator(device="cpu")

    print("Building full agent graph (config D)...")
    agent = build_agent_graph(
        llm=llm,
        retriever=hybrid,
        reranker=reranker,
        entity_linker=entity_linker,
        translator=translator,
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=settings.MAX_RETRIES,
        grounding_mode=settings.GROUNDING_MODE,
        on_verify_fail=settings.ON_VERIFY_FAIL,
    )

    return {
        "settings": settings,
        "llm": llm,
        "hybrid": hybrid,
        "dense_only": dense_only,
        "reranker": reranker,
        "entity_linker": entity_linker,
        "translator": translator,
        "agent": agent,
    }


# ---------------------------------------------------------------------------
# Per-config single query runners. Each returns (ranked_chunks, answer,
# grounding_action). ranked_chunks is the ordered evidence list used for
# Recall@3/MRR scoring.
# ---------------------------------------------------------------------------
def _run_single_pass(svc, query: str, *, use_hybrid: bool, use_rerank: bool,
                     use_kg: bool, use_policy: bool, top_k: int = 3):
    """Configs A/B/C: retrieve → (rerank) → direct generate → (policy).

    No grade/rewrite/verify loop. Returns (ranked, answer, grounding_action).
    """
    from agribot.llm.engine import generate_answer
    from agribot.agent.grounding_policy import is_risky_query, REFUSE_EN

    retriever = svc["hybrid"] if use_hybrid else svc["dense_only"]

    # KG query expansion (config C only), mirrors make_kg_link_node.
    retrieval_query = query
    if use_kg:
        try:
            retrieval_query = svc["entity_linker"].expand_query(query)
        except Exception:
            retrieval_query = query

    candidates = retriever.retrieve(retrieval_query, top_n=max(15, top_k * 2))

    if use_rerank and candidates:
        ranked = svc["reranker"].rerank(retrieval_query, candidates)
    else:
        ranked = candidates

    ranked = ranked[:top_k] if ranked else []

    # Build evidence text block exactly like make_rerank_node.
    evidence_text = "\n\n".join(f"[{ev.citation}]: {ev.text}" for ev in ranked)

    grounding_action = "pass"
    if not evidence_text.strip():
        answer = "I don't know based on the provided documents."
    else:
        answer = generate_answer(svc["llm"], query, evidence_text,
                                 max_tokens=svc["settings"].LLM_MAX_TOKENS)
        if not (answer or "").strip():
            answer = "I don't know based on the provided documents."

    # Refusal policy (configs B/C). Config A has NO policy (naive baseline).
    if use_policy and is_risky_query(query):
        answer = REFUSE_EN
        grounding_action = "refuse"

    return ranked, answer, grounding_action


def _run_full_agent(svc, query: str):
    """Config D: the real compiled agent, invoked as api.py does."""
    initial = {
        "query_original": query,
        "query_language": "",
        "query_normalized": "",
        "query_expanded": "",
        "kg_entities": [],
        "evidences": [],
        "evidence_texts": "",
        "evidence_grade": "",
        "answer": "",
        "answer_bn": "",
        "citations": [],
        "is_verified": False,
        "verification_reason": "",
        "retry_count": 0,
        "should_refuse": False,
        "input_mode": "text",
        "input_audio_path": "",
        "image_top_label": "",
        "image_top_confidence": 0.0,
        "image_symptom_hints": [],
        "error": "",
        "trace_id": "",
        "timings_ms": {},
        "grounding_action": "pass",
        "follow_up_suggestions": [],
    }
    final = svc["agent"].invoke(initial)
    ranked = final.get("evidences", []) or []
    return (
        ranked,
        final.get("answer", ""),
        final.get("grounding_action", "pass"),
        bool(final.get("is_verified", False)),
        final.get("citations", []) or [],
    )


CONFIG_RUNNERS = {
    "A": dict(label="Dense only", use_hybrid=False, use_rerank=False,
              use_kg=False, use_policy=False),
    "B": dict(label="+ BM25 + RRF", use_hybrid=True, use_rerank=False,
              use_kg=False, use_policy=True),
    "C": dict(label="+ Rerank + KG (1-pass)", use_hybrid=True, use_rerank=True,
              use_kg=True, use_policy=True),
    "D": dict(label="Full proposed (AgriBot)"),  # special: real agent
}


# ---------------------------------------------------------------------------
# Config-level evaluation
# ---------------------------------------------------------------------------
def _eval_config(svc, cfg_id: str, queries: list[dict], limit: int | None) -> dict:
    cfg = CONFIG_RUNNERS[cfg_id]
    print(f"\n{'=' * 60}\nCONFIG {cfg_id}: {cfg['label']}\n{'=' * 60}")

    ranks_tol: list[int | None] = []   # page-tolerant rank (Recall@3, MRR)
    kw_scores: list[float] = []
    tp = fp = tn = fn = 0
    per_query = []

    eval_queries = queries[:limit] if limit else queries
    for i, q in enumerate(eval_queries, start=1):
        t0 = time.perf_counter()
        if cfg_id == "D":
            ranked, answer, action, is_ver, cites = _run_full_agent(svc, q["query"])
        else:
            ranked, answer, action = _run_single_pass(
                svc, q["query"],
                use_hybrid=cfg["use_hybrid"], use_rerank=cfg["use_rerank"],
                use_kg=cfg["use_kg"], use_policy=cfg["use_policy"],
                top_k=svc["settings"].RERANK_TOP_N,
            )
            is_ver, cites = None, []
        elapsed = time.perf_counter() - t0

        # --- retrieval (only on queries that carry expected_sources) ---
        rank = None
        if q.get("expected_sources"):
            rank = _hit_rank_pagetol(ranked, q["expected_sources"])
            ranks_tol.append(rank)

        # --- groundedness: keyword coverage on answerable queries ---
        if not q.get("must_refuse", False):
            kw = _keyword_coverage(answer, q.get("expected_answer_keywords", []))
            kw_scores.append(kw)
        else:
            kw = None

        # --- refusal accuracy over ALL queries ---
        should_refuse = bool(q.get("must_refuse", False))
        did_refuse = _looks_like_refusal(answer, action)
        if should_refuse and did_refuse:
            tp += 1; verdict = "TP"
        elif should_refuse and not did_refuse:
            fn += 1; verdict = "FN"
        elif not should_refuse and did_refuse:
            fp += 1; verdict = "FP"
        else:
            tn += 1; verdict = "TN"

        per_query.append({
            "id": q["id"], "category": q["category"], "rank": rank,
            "keyword_coverage": kw, "verdict": verdict,
            "grounding_action": action, "latency_s": round(elapsed, 1),
            "is_verified": is_ver, "citation_count": len(cites),
            "answer_preview": (answer or "")[:120],
        })
        print(f"  [{i}/{len(eval_queries)}] {q['id']}: rank={rank} kw={kw} "
              f"{verdict} ({elapsed:.0f}s)")

    n_ret = len(ranks_tol) or 1
    recall_at_3 = sum(1 for r in ranks_tol if r is not None and r <= 3) / n_ret
    mrr = sum(1.0 / r for r in ranks_tol if r is not None) / n_ret
    n_total = tp + fp + tn + fn
    summary = {
        "config": cfg_id,
        "label": cfg["label"],
        "recall_at_3": round(recall_at_3, 4),
        "mrr": round(mrr, 4),
        "groundedness_keyword_coverage": round(sum(kw_scores) / (len(kw_scores) or 1), 4),
        "refusal_accuracy": round((tp + tn) / (n_total or 1), 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_retrieval_scored": len(ranks_tol),
        "n_grounding_scored": len(kw_scores),
        "n_refusal_scored": n_total,
    }
    # Config D is the shipped agent: also emit the verify-node groundedness
    # (verified_rate) + full refusal confusion, matching the standalone
    # eval_grounding.py / eval_refusal.py schemas so they can be written as the
    # canonical per-metric result files (in-process, no flaky HTTP backend).
    if cfg_id == "D":
        gq = [pq for pq in per_query if pq["category"] in RETRIEVAL_CATEGORIES]
        ng = len(gq) or 1
        kcov = [pq["keyword_coverage"] for pq in gq if pq["keyword_coverage"] is not None]
        actions: dict[str, int] = {}
        for pq in gq:
            actions[pq["grounding_action"]] = actions.get(pq["grounding_action"], 0) + 1
        verified = sum(1 for pq in gq if pq["is_verified"])
        summary["grounding"] = {
            "queries_evaluated": len(gq),
            "avg_keyword_coverage": round(sum(kcov) / (len(kcov) or 1), 3),
            "avg_citation_count": round(sum(pq["citation_count"] for pq in gq) / ng, 2),
            "verified_rate": round(verified / ng, 3),
            "grounding_action_breakdown": actions,
        }
        refusal_precision = tp / (tp + fp) if (tp + fp) else 0.0
        refusal_recall = tp / (tp + fn) if (tp + fn) else 0.0
        answer_precision = tn / (tn + fn) if (tn + fn) else 0.0
        summary["refusal"] = {
            "queries_evaluated": n_total,
            "true_positive_refusals": tp,
            "false_positive_refusals": fp,
            "true_negative_answers": tn,
            "false_negative_refusals": fn,
            "accuracy": round((tp + tn) / (n_total or 1), 3),
            "refusal_precision": round(refusal_precision, 3),
            "refusal_recall": round(refusal_recall, 3),
            "answer_precision": round(answer_precision, 3),
        }
    print(f"\n  -> R@3={summary['recall_at_3']} MRR={summary['mrr']} "
          f"Gnd={summary['groundedness_keyword_coverage']} "
          f"Acc={summary['refusal_accuracy']}")
    return {"summary": summary, "per_query": per_query}


def run(config_ids: list[str], limit: int | None) -> dict:
    dataset = load_dataset()
    queries = dataset["queries"]
    svc = _load_services()

    results = {}
    for cfg_id in config_ids:
        results[cfg_id] = _eval_config(svc, cfg_id, queries, limit)

    payload = {
        "meta": {
            "configs": config_ids,
            "limit": limit,
            "note": "Groundedness is mean keyword coverage (identical metric "
                    "across configs). Paper's headline 78.3% is verify-node "
                    "verified_rate, only meaningful for config D.",
        },
        "table": {
            cfg_id: results[cfg_id]["summary"] for cfg_id in config_ids
        },
        "detail": results,
    }
    out = save_results("ablation", payload)
    print(f"\n\nResults written to: {out}")

    # If config D ran, emit the canonical per-metric files (grounding/refusal)
    # from the shipped agent's in-process pass, matching the standalone scripts.
    if "D" in results and "grounding" in results["D"]["summary"]:
        d = results["D"]
        gq_ids = {pq["id"] for pq in d["per_query"]
                  if pq["category"] in RETRIEVAL_CATEGORIES}
        g_payload = {
            "summary": d["summary"]["grounding"],
            "per_query": [pq for pq in d["per_query"] if pq["id"] in gq_ids],
            "meta": {"source": "eval_ablation config D (in-process)"},
        }
        save_results("grounding", g_payload)
        r_payload = {
            "summary": d["summary"]["refusal"],
            "per_query": d["per_query"],
            "meta": {"source": "eval_ablation config D (in-process)"},
        }
        save_results("refusal", r_payload)
        print("Also wrote grounding_latest.json + refusal_latest.json from config D")

    # Pretty table to stdout
    print("\n" + "=" * 72)
    print(f"{'Config':<26}{'R@3':>8}{'MRR':>8}{'Gnd':>8}{'Acc':>8}")
    print("-" * 72)
    for cfg_id in config_ids:
        s = results[cfg_id]["summary"]
        print(f"{cfg_id + ': ' + s['label']:<26}"
              f"{s['recall_at_3']:>8.3f}{s['mrr']:>8.3f}"
              f"{s['groundedness_keyword_coverage']:>8.3f}"
              f"{s['refusal_accuracy']:>8.3f}")
    print("=" * 72)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=["A", "B", "C", "D"],
                   choices=["A", "B", "C", "D"])
    p.add_argument("--smoke", action="store_true",
                   help="Only 2 queries per config to test wiring fast.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of queries per config.")
    args = p.parse_args()
    limit = 2 if args.smoke else args.limit
    run(args.configs, limit)


if __name__ == "__main__":
    main()
