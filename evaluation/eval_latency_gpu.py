"""Full-agent end-to-end GPU latency (fix-list items 13/14).

Runs the SAME in-corpus query set and per-stage instrumentation as
eval_latency.py, but in-process with the generative LLM offloaded to the
GPU (n_gpu_layers=-1) via the cu124 llama-cpp-python build in venv-gpu.
Everything else (BanglaT5 translation, MiniLM embeddings, cross-encoder
rerank, KG) stays CPU-pinned, exactly matching the deployed
hardware-isolation design. The output schema matches eval_latency.py so the
GPU p50/p95 are directly comparable to the CPU headline numbers.

Distinct from gpu_latency_probe.py, which isolates only the generation
stage on fixed prompts; this script measures the real end-to-end agent.

Run (from the GPU venv):
    venv-gpu\\Scripts\\python.exe evaluation\\eval_latency_gpu.py --runs 3

Writes evaluation/results/latency_gpu_latest.json (+ timestamped copy).
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from collections import defaultdict
from pathlib import Path

# Ensure project root is importable when run as `python evaluation/eval_latency_gpu.py`
# (the script dir, not the project root, is what Python puts on sys.path[0]).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- CUDA DLL path preamble: MUST run before importing llama_cpp ---
_site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
_llama_lib = Path(sys.prefix) / "Lib" / "site-packages" / "llama_cpp" / "lib"
for _d in list(_site.glob("*/bin")) + [_llama_lib]:
    if _d.is_dir():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")

from evaluation.common import load_dataset, filter_queries, percentile, save_results  # noqa: E402


def _build_agent(n_gpu_layers: int):
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

    print(f"Loading LLM on GPU (n_gpu_layers={n_gpu_layers})...")
    llm = get_llm(model_path=str(settings.MODEL_PATH),
                  n_ctx=settings.LLM_N_CTX, n_gpu_layers=n_gpu_layers)
    print("Loading indexes + embedder (CPU)...")
    bundle = IndexBundle.load(settings.INDEX_DIR)
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    retriever = HybridRetriever(
        index_bundle=bundle, embedding_model=embedder,
        dense_top_k=settings.DENSE_TOP_K, sparse_top_k=settings.SPARSE_TOP_K,
        dense_weight=settings.DENSE_WEIGHT, sparse_weight=settings.SPARSE_WEIGHT)
    reranker = Reranker(threshold=settings.RERANK_THRESHOLD, top_n=settings.RERANK_TOP_N)
    kg = KnowledgeGraph(settings.KG_DB_PATH)
    seed_knowledge_graph(kg)
    entity_linker = EntityLinker(kg)
    translator = get_translator(device="cpu")
    agent = build_agent_graph(
        llm=llm, retriever=retriever, reranker=reranker,
        entity_linker=entity_linker, translator=translator,
        max_tokens=settings.LLM_MAX_TOKENS, max_retries=settings.MAX_RETRIES,
        grounding_mode=settings.GROUNDING_MODE, on_verify_fail=settings.ON_VERIFY_FAIL)
    return agent


def _initial_state(query: str) -> dict:
    return {
        "query_original": query, "query_language": "", "query_normalized": "",
        "query_expanded": "", "kg_entities": [], "evidences": [],
        "evidence_texts": "", "evidence_grade": "", "answer": "", "answer_bn": "",
        "citations": [], "is_verified": False, "verification_reason": "",
        "retry_count": 0, "should_refuse": False, "input_mode": "text",
        "input_audio_path": "", "image_top_label": "", "image_top_confidence": 0.0,
        "image_symptom_hints": [], "error": "", "trace_id": "", "timings_ms": {},
        "grounding_action": "pass", "follow_up_suggestions": [],
    }


def run(runs_per_query: int, n_gpu_layers: int, limit: int | None) -> dict:
    dataset = load_dataset()
    queries = filter_queries(dataset, categories={"in_corpus_factual", "in_corpus_procedural"})
    if limit:
        queries = queries[:limit]
    agent = _build_agent(n_gpu_layers)
    print(f"GPU end-to-end latency over {len(queries)} in-corpus queries x {runs_per_query} runs")

    wall_times: list[float] = []
    stage_times: dict[str, list[float]] = defaultdict(list)
    per_query = []

    for i, q in enumerate(queries, start=1):
        q_walls: list[float] = []
        q_stages: dict[str, list[float]] = defaultdict(list)
        for _ in range(runs_per_query):
            t0 = time.perf_counter()
            final = agent.invoke(_initial_state(q["query"]))
            wall_ms = (time.perf_counter() - t0) * 1000
            q_walls.append(wall_ms)
            wall_times.append(wall_ms)
            for stage, val in (final.get("timings_ms", {}) or {}).items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                stage_times[stage].append(v)
                q_stages[stage].append(v)
        per_query.append({
            "id": q["id"],
            "wall_ms_mean": round(sum(q_walls) / len(q_walls), 1),
            "stage_means_ms": {s: round(sum(v) / len(v), 1) for s, v in q_stages.items()},
        })
        print(f"  [{i}/{len(queries)}] {q['id']}: mean wall = {sum(q_walls)/len(q_walls)/1000:.2f}s")

    summary = {
        "queries_evaluated": len(per_query),
        "runs_per_query": runs_per_query,
        "device": f"GPU (n_gpu_layers={n_gpu_layers}, RTX 3050 4GB)",
        "wall_ms": {
            "p50": percentile(wall_times, 0.5),
            "p95": percentile(wall_times, 0.95),
            "mean": round(sum(wall_times) / max(1, len(wall_times)), 1),
        },
        "stage_means_ms": {s: round(sum(v) / len(v), 1) for s, v in stage_times.items()},
    }
    print("\n=== GPU end-to-end latency results ===")
    print(f"  Wall p50: {summary['wall_ms']['p50']} ms")
    print(f"  Wall p95: {summary['wall_ms']['p95']} ms")
    print(f"  Wall mean: {summary['wall_ms']['mean']} ms")
    for stage, mean in sorted(summary["stage_means_ms"].items(), key=lambda kv: -kv[1]):
        print(f"    {stage:20s} {mean}")

    payload = {"summary": summary, "per_query": per_query}
    out = save_results("latency_gpu", payload)
    print(f"\nResults written to: {out}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    run(args.runs, args.n_gpu_layers, args.limit)


if __name__ == "__main__":
    main()
