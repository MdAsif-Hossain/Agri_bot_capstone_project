# G-D · EdgeRAG: Online-Indexed RAG for Edge Devices (Seemakhupt et al., arXiv 2024)

> **Verified from PDF.** Status: ❌ Original review had **fabricated authors** (Kakolyris is wrong) and **unverified venue** (ICML 2025 claim not supported by PDF). Methodology summary is otherwise solid; numbers updated below.

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | EdgeRAG: Online-Indexed RAG for Edge Devices |
| **Authors** | **Korakit Seemakhupt¹, Sihang Liu², Samira Khan¹,³** |
| **Affiliations** | ¹ University of Virginia, ² University of Waterloo, ³ Google |
| **Venue** | **arXiv:2412.21023v2 [cs.LG]**, 31 Dec 2024 — **preprint only**, no peer-reviewed venue confirmed in PDF |
| **License** | arXiv (preprint) |

### Audit of the original review (`paper_analysis_group_d.md` Paper 1)
- ❌ **Authors**: Original said "A. K. Kakolyris et al." — this is **fabricated**. Actual authors are **Korakit Seemakhupt, Sihang Liu, Samira Khan**.
- ❌ **Venue**: Original said "ICML 2025 (under review, arXiv preprint)" — only the arXiv preprint is confirmed; **no ICML acceptance is visible in the PDF**. Should be cited as arXiv preprint only.
- ✅ **Hardware**: Nvidia Jetson Orin Nano (8 GB shared memory) — confirmed.
- ✅ **Method**: IVF index pruning + on-the-fly embedding regeneration + selective caching — confirmed.
- ⚠️ **Headline numbers**: Original said "1.22× average retrieval latency improvement; 3.69× for datasets exceeding memory." The PDF says **1.8× faster TTFT on average over IVF baseline; 3.82× for larger datasets** (Section 1, conclusion). Numbers updated.
- ⚠️ **Models**: Original claimed "Sheared-LLaMA-2.7B (LLM), GTE-large (embedding model)" — model choices not explicitly confirmed in read portion; verify in §5 (Evaluation) before citing.
- ✅ **Benchmark**: BEIR suite (6 workloads including nq, hotpotqa, fever) — confirmed.

---

## 2. Paper Summary

### 2.1 Problem
Deploying RAG on **edge devices** (Jetson Orin Nano, iPhone 16 Pro, Galaxy S24 — all with 8 GB shared CPU+GPU memory) is bottlenecked by **memory**, not compute. Example: the FEVER dataset has 5.23M records → embedding index is **18.5 GB**, far exceeding device memory. Two failure modes:
1. **Flat index**: sequential search over all embeddings → memory thrashing.
2. **IVF (Inverted File) index**: clusters embeddings into centroids → fewer searches, but holding all embeddings still thrashes.

### 2.2 Method — Online-Indexed RAG
**Key insight**: Most embeddings in IVF clusters are never accessed during retrieval; access is highly skewed. So instead of caching all embeddings:

1. **Prune second-level embeddings** within IVF clusters → freed memory.
2. **Regenerate embeddings on-demand** during retrieval using the embedding model.
3. **Pre-compute + store embeddings for "tail clusters"** (large clusters where regeneration would dominate latency) — avoids long-tail penalties.
4. **Adaptive caching** of recently-generated embeddings to amortize cost across repeated queries.

**Timing breakdown** (Figure 6): TTFT (time-to-first-token) = Query Embed + First-level Lookup + Second-level Lookup + Prefill. EdgeRAG optimizes the second-level lookup phase.

### 2.3 Key Results (on Jetson Orin Nano 8 GB, BEIR benchmark)
| Metric | EdgeRAG vs. IVF baseline |
|---|---|
| **TTFT** (average) | **1.8× faster** |
| **TTFT** (larger datasets) | **3.82× faster** |
| Recall / generation quality | Within **5%** of Flat Index baseline |
| Memory footprint | Fits all evaluated datasets into 8 GB (avoids thrashing) |

LLM-as-judge evaluation via GPT-4o for generation quality.

---

## 3. Relevance to AgriBot

| Dimension | EdgeRAG | AgriBot |
|---|---|---|
| **Edge deployment** | Jetson Orin Nano (8 GB shared CPU+GPU) | Consumer laptop with **RTX 3050 (4 GB VRAM + system RAM)** |
| **Memory strategy** | Online embedding regeneration + selective caching for large indexes | **Small corpus** (5–25 PDFs) + lightweight all-MiniLM-L6-v2 → FAISS index fits in RAM trivially. No regeneration needed. |
| **LLM** | Sheared-LLaMA-2.7B (claimed) | **Qwen2.5-1.5B Q8 GGUF** (smaller, more aggressively quantized) |
| **Retrieval** | Pure dense (FAISS IVF) | **Hybrid FAISS + BM25 + RRF + cross-encoder rerank** |
| **KG** | None | SQLite KG for dialect bridging |
| **Agentic pipeline** | None | LangGraph state machine with grading + retry + verify + enforce |
| **Domain** | General BEIR benchmarks | Agricultural advisory (Bangladesh) |
| **Hardware isolation** | Shares GPU between LLM and embedding model | **Strict CPU/GPU split** — GPU exclusively for LLM; embeddings/ASR/vision/translation/rerank all on CPU |

---

## 4. Gap AgriBot Fills

1. **Pure systems-optimization paper, no RAG quality improvements.** EdgeRAG optimizes memory/latency but doesn't improve retrieval precision, answer quality, or grounding. AgriBot improves the **entire RAG pipeline** quality (hybrid retrieval, rerank, KG expansion, evidence grading, verification, policy enforcement).
2. **No safety enforcement.** EdgeRAG answers whatever the LLM generates. AgriBot **refuses unverified dosage advice**.
3. **No agentic self-correction.** Single-pass retrieve→generate. AgriBot grades evidence before generation and retries with rewritten queries.
4. **No bilingual / dialect / multimodal / voice support.**
5. **No domain application.** General BEIR benchmarks vs. AgriBot's deployable agricultural advisory.
6. **Hardware isolation not addressed.** EdgeRAG shares GPU between LLM and embedder. AgriBot explicitly **isolates GPU for LLM only** — embeddings, vision, ASR, translation, rerank all on CPU to avoid VRAM contention on a 4 GB card.

## 5. Strengths to Acknowledge

- **EdgeRAG's memory technique is genuinely sophisticated.** The pre-compute-tail / regenerate-others / adaptively-cache decomposition is well-motivated by the access skew analysis in §3.
- **Honest negative trade-off**: 5% recall drop in exchange for memory fit is reported transparently.
- **Hardware comparison table** (iPhone 16 Pro, Galaxy S24, Jetson Orin Nano, Nvidia L40) is a useful taxonomy for the "edge AI device landscape" section of AgriBot's paper.
- **TTFT framing**: User-perceived latency = query embed + lookup + prefill. This decomposition is exactly what AgriBot's per-stage diagnostics module captures — citing EdgeRAG legitimizes our timing breakdown approach.
- **Pattern transferability**: If AgriBot's corpus ever grows to where the FAISS index doesn't fit in RAM, EdgeRAG's on-demand regeneration approach is a viable scaling path.

## 6. BibTeX

```bibtex
@misc{Seemakhupt2024EdgeRAG,
  author        = {Seemakhupt, Korakit and Liu, Sihang and Khan, Samira},
  title         = {{EdgeRAG}: Online-Indexed {RAG} for Edge Devices},
  year          = {2024},
  eprint        = {2412.21023},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2412.21023}
}
```

## 7. Final Verdict

> **INCLUDE — Tier 2 (edge-systems complement).**
> arXiv preprint (not yet peer-reviewed), but reputable authors (UVA + Waterloo + Google). Best positioned as a **complementary systems-engineering reference**: EdgeRAG solves the memory problem for edge RAG; AgriBot solves the quality/safety/accessibility problem with a different memory strategy (small corpus, lightweight models, hardware isolation). Don't claim it as a direct competitor — the design objectives are orthogonal.

**Positioning sentence:**
> "Seemakhupt et al. (EdgeRAG, arXiv 2024) optimize RAG memory footprint on edge devices (Jetson Orin Nano, 8 GB) by pruning IVF cluster embeddings and regenerating them on-demand, achieving up to 3.82× TTFT improvement on large datasets within 5% of Flat-Index recall. Our system takes a complementary approach: targeting a smaller corpus and lighter embedding model that fits trivially in memory, then investing the saved memory in **hybrid (dense + sparse) retrieval, cross-encoder reranking, dialect-bridging knowledge graph expansion, and an agentic self-correction loop** — trading EdgeRAG's index-scaling sophistication for higher-quality, safer, bilingual answers on a 4 GB VRAM consumer GPU."
