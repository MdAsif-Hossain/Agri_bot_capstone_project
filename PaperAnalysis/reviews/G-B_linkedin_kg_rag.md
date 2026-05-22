# G-B · LinkedIn KG-RAG for Customer Service QA

> **Verified from PDF.** Status: ✅ Existing review is **accurate**. No corrections needed beyond using the verbatim title.

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering |
| **Authors** | Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Deshpande, Xiaofeng Wang, Zheng Li |
| **Affiliation** | LinkedIn Corporation, Sunnyvale, CA |
| **Venue** | SIGIR '24 — 47th International ACM SIGIR Conference, July 14–18, 2024, Washington DC |
| **DOI** | 10.1145/3626772.3661370 |
| **arXiv** | 2404.17723v2 (May 2024) |
| **Length** | 5 pages (short paper, industry track) |

### Audit of the original review (`paper_analysis_group_b.md` Paper 3)
- ✅ Authors: correct (listed as "Zhentao Xu et al. (LinkedIn)").
- ✅ Venue: correct.
- ✅ Headline numbers: **77.6% MRR improvement** and **28.6% reduction in median per-issue resolution time** — both quoted verbatim from the abstract.
- ⚠️ Title used in the original review is a paraphrase ("RAG with Knowledge Graphs for Customer Service QA") — use the verbatim title above for the bibliography.

---

## 2. Paper Summary

### 2.1 Problem
Conventional RAG over customer-service tickets treats each ticket as plain text and chunks it for embedding. This loses (1) **intra-issue structure** (a ticket is really a record of `Summary / Description / Steps / Priority / Root Cause / Resolution / Comments`) and (2) **inter-issue relations** (Jira "clone/blocks/relates-to" links + implicit semantic similarity). Result: lossy retrieval and incoherent answers.

### 2.2 Method — Dual-level Knowledge Graph
- **Intra-issue Tree** `T(N, E, R)`: each ticket is parsed into a tree of typed sections via a hybrid **rule-based + LLM** parser guided by a YAML template `T_template`. Rule-based handles predictable fields (priority, code blocks); LLM handles free-text sections.
- **Inter-issue Graph** `G(T, E, R)`: trees are connected via **explicit links** (Jira clone/blocks/relates-to) + **implicit links** (cosine similarity over ticket-title embeddings, threshold-gated).
- **Node embedding generation**: text-embedding models (BERT, E5) embed text-rich nodes (`summary`, `description`, `steps_to_reproduce`). Embeddings stored in **QDrant** vector DB; structure in a graph DB.

### 2.3 Query-time pipeline
1. **Entity detection + intent classification** via LLM — parses query into `(field, value)` pairs aligned to `T_template` plus an intent label.
2. **Embedding-based sub-graph retrieval** using detected entities.
3. **Filtering** by detected entities + intent.
4. **LLM answer generation** from retrieved sub-graph.

### 2.4 Headline Results
| Metric | Improvement vs. baseline |
|---|---|
| MRR (retrieval) | **+77.6%** |
| BLEU (generation) | **+0.32** |
| ROUGE / NDCG@K / METEOR | (reported in §4 — all consistently improved) |
| Production: median resolution time | **−28.6%** at LinkedIn Customer Support over ~6 months |

---

## 3. Relevance to AgriBot

| Dimension | LinkedIn KG-RAG | AgriBot |
|---|---|---|
| **KG purpose** | Encode *document/ticket structure* — preserve hierarchy + cross-doc references | Encode *linguistic ontology* — map dialect terms (Sylheti/Chittagong/Barishal) to canonical scientific entities |
| **Entity detection** | LLM parses query → field/value pairs aligned to a YAML template | Token-based n-gram matching against KG aliases (lightweight, no LLM call per query) |
| **Retrieval** | Embedding-based sub-graph extraction + filtering | Hybrid FAISS dense + BM25 sparse + RRF fusion + FlashRank rerank |
| **Generation** | Cloud LLM with retrieved sub-graph | Local Qwen2.5-1.5B GGUF with cited evidence chunks |
| **Deployment** | LinkedIn production (cloud) | Offline, RTX 3050 4GB VRAM |
| **Domain stakes** | Customer support — wrong answer = inconvenience | Agricultural advisory — wrong dosage = physical harm |

---

## 4. Gap AgriBot Fills

1. **No agentic self-correction.** LinkedIn's pipeline is single-pass (parse → retrieve → filter → generate). AgriBot's LangGraph state machine **grades evidence before generation** and retries with **LLM-rewritten queries** (bounded retry, max 2 by default).
2. **No grounding verification or safety enforcement.** No mechanism for refusing or disclaiming low-confidence answers. AgriBot has **bilingual regex risk detection** (dosage/chemicals in EN + BN) feeding a configurable grounding policy (refuse / disclaim / cited-facts-only).
3. **Heavy infrastructure.** Requires graph DB (Neo4j-class) + QDrant vector DB + cloud LLM. AgriBot achieves comparable KG-enhanced retrieval with **SQLite + FAISS + llama-cpp-python**.
4. **KG covers documents, not language.** Their KG bridges *tickets to each other*; AgriBot's KG bridges *farmer speech to scientific terminology* — orthogonal contribution.
5. **No bilingual / dialect / multimodal / voice support.** All single-channel English text.

---

## 5. Strengths to Acknowledge

- **Production-validated.** 6-month live deployment at LinkedIn with measurable business KPI improvement (−28.6% resolution time) — strong external validation of the "KG-enhanced RAG actually works" claim.
- **Practical hybrid parsing.** Rule-based for predictable fields + LLM for free text is a pattern AgriBot's ingestion pipeline could borrow for structured agricultural manuals (e.g., extension officer reports with predictable sections).
- **Short, well-scoped paper.** 5-page SIGIR industry-track format = easy to cite and clearly positioned.

---

## 6. BibTeX

```bibtex
@inproceedings{Xu2024LinkedInKGRAG,
  author    = {Xu, Zhentao and Cruz, Mark Jerome and Guevara, Matthew and
               Wang, Tie and Deshpande, Manasi and Wang, Xiaofeng and Li, Zheng},
  title     = {Retrieval-Augmented Generation with Knowledge Graphs for
               Customer Service Question Answering},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on
               Research and Development in Information Retrieval (SIGIR '24)},
  year      = {2024},
  pages     = {1--5},
  publisher = {ACM},
  doi       = {10.1145/3626772.3661370},
  address   = {Washington, DC, USA}
}
```

---

## 7. Final Verdict

> **INCLUDE — Tier 1 (high priority).** 
> Top-tier IR venue (SIGIR), production-deployed, clean KG-RAG architecture with directly transferable design lessons. Strongest counter-example for AgriBot's "KG-enhanced retrieval is feasible and effective in domain-specific QA" claim.

**Positioning sentence for your paper:**
> "Xu et al. demonstrate that integrating a structured knowledge graph into RAG significantly improves retrieval and generation quality in production customer-service QA at LinkedIn (+77.6% MRR, −28.6% median resolution time). We extend this paradigm to the safety-critical agricultural domain by (i) using the KG to bridge **linguistic** rather than **structural** gaps (farmer dialect → canonical scientific terms), (ii) adding **agentic evidence grading and self-correction** before generation, and (iii) deploying the entire stack **offline on consumer-grade hardware** for rural Bangladesh, where neither cloud LLMs nor managed graph databases are accessible."
