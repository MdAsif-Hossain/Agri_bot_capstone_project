# G-B · Document GraphRAG (Manufacturing Domain)

> **Verified from PDF.** Status: ⚠️ Existing review is **mostly accurate**; sub-component names confirmed (DKG, IKG, ICS, IKS, UKS) but the original review under-specified the third retrieval strategy and conflated the two graph layers. Datasets confirmed: **SQuAD + HotpotQA + a proprietary German automotive dataset**.

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | Document GraphRAG: Knowledge Graph Enhanced Retrieval Augmented Generation for Document Question Answering Within the Manufacturing Domain |
| **Authors** | Simon Knollmeyer, Oğuz Caymazer, Daniel Grossmann |
| **Affiliation** | AImotion Bavaria, Technische Hochschule Ingolstadt (Knollmeyer, Grossmann); Department of Information Systems, University of Münster (Caymazer) |
| **Venue** | *Electronics* (MDPI), Vol. 14, Issue 11, Article 2102 |
| **DOI** | 10.3390/electronics14112102 |
| **Dates** | Received 14 Mar 2025; Revised 13 May 2025; Accepted 15 May 2025; Published 22 May 2025 |
| **Length** | 37 pages (full research article) |

### Audit of the original review (`paper_analysis_group_b.md` Paper 1)
- ✅ **Authors**: S. Knollmeyer, O. Caymazer, D. Grossmann — confirmed.
- ✅ **Venue**: Electronics 2025, 14(11), 2102 — confirmed.
- ✅ **Sub-component names**: "Document Knowledge Graph (DKG)" and "Information Knowledge Graph (IKG)" both appear verbatim (Section 2.2, Figure 4). "Informed Chapter Search (ICS)" appears verbatim as one of three named graph-retrieval strategies (Section 5.2). The paper also defines two further strategies the original review missed: **Informed Keyword Search (IKS)** and **Uninformed Keyword Search (UKS)**.
- ⚠️ "Structure-aware search" is **not** a verbatim term in the paper — the closest concepts are "structural retrieval" / "graph-based retrieval strategies".
- ✅ **Datasets**: SQuAD + HotpotQA + a proprietary internal manufacturing dataset (German-language automotive production planning, 17 documents, ~5,500 pages).
- ⚠️ Methodology framing: paper explicitly follows **Design Science Research (DSR)** — the original review did not mention this, but it is central to how the artifact is positioned.

---

## 2. Paper Summary

### 2.1 Problem
Naive RAG over long technical manufacturing documents loses (1) intra-document structure (chapter/section hierarchy) and (2) inter-document semantic links, hurting multi-hop QA. Constructing full ontology-based Information Knowledge Graphs (IKGs) is labor-intensive and rarely feasible in industry; lightweight **Document Knowledge Graphs (DKGs)** are proposed as the practical alternative.

### 2.2 Method — Document GraphRAG
- **DKG (hierarchical layer)**: tree-structured graph encoding `Document → Chapter → Sub-chapter → Chunk`. Each chunk receives a URI that also links to the vector DB row. Constructed deterministically from extracted document structure — no LLM call needed.
- **IKG (semantic layer)**: keywords extracted per chunk (YAKE algorithm, 5 keywords/chunk default; LLM-based extraction with Claude 3 Haiku tested as alternative) form cross-chunk and cross-document edges.
- **Three graph retrieval strategies** (all defined in §5.2):
  1. **Informed Chapter Search (ICS)** — vector top-k → expand to sibling chunks in the same chapter via DKG.
  2. **Informed Keyword Search (IKS)** — vector top-k → expand via shared IKG keywords (enables cross-document links).
  3. **Uninformed Keyword Search (UKS)** — extract keywords from the *query itself*, fetch chunks linked to them in the IKG, merge with vector results.
- **Post-processing**: URI-based deduplication → LLM reranking (Claude 3 Haiku, 1–10 score + justification) → pass-conditions filter top-k for generation (Claude Sonnet 3.5).
- **Implementation**: AWS Textract (OCR), LangChain Character Text Splitter, AWS Titan embeddings, Chroma vector DB, GraphDB (RDF/SPARQL), Python 3.10.

### 2.3 Headline Results (verbatim from Table 1, standard pipeline: chunk size 1000, 5 keywords, top-k=10)

| Dataset | Metric | Naive RAG | GraphRAG |
|---|---|---|---|
| SQuAD | Recall@10 / MRR@10 / CP / CR | 0.725 / 0.498 / 0.510 / 0.723 | **0.760 / 0.613 / 0.635 / 0.793** |
| HotpotQA | Recall@10 / MRR@10 / CP / CR | 0.500 / 0.432 / 0.703 / 0.736 | **0.650 / 0.537 / 0.770 / 0.759** |
| Internal | Recall@10 / MRR@10 / CP / CR | 0.055 / 0.040 / 0.485 / 0.729 | **0.075 / 0.044 / 0.498 / 0.794** |

- Average Context Relevance improvement: **+15.5% (SQuAD), +18.3% (HotpotQA), +14.5% (internal)**.
- Average generation-metric improvement: +1.8% / +8.4% / +6.4% respectively.
- **Latency (standard config)**: Naive RAG 1.7 s end-to-end (0.2 s retrieval + 1.5 s generation); GraphRAG 9.8 s (7.4 s retrieval + 0.9 s rerank + 1.5 s generation) — a 5.7× slowdown, accepted by the focus group as below the 10 s usability threshold.
- KG construction for the 5,500-page internal corpus took 195 s (one-shot index time).

---

## 3. Relevance to AgriBot

| Dimension | Document GraphRAG | AgriBot |
|---|---|---|
| **KG purpose** | Two-layer KG: DKG (document hierarchy) + IKG (keyword-based cross-doc links) | SQLite linguistic ontology: dialect tokens → canonical scientific terms |
| **KG construction** | Deterministic DKG from document parse; YAKE/LLM keywords for IKG | Hand-curated dialect aliases; no per-chunk keyword extraction |
| **Retrieval** | Vector + 3 graph strategies (ICS/IKS/UKS) → LLM rerank | FAISS dense + BM25 sparse + RRF fusion + FlashRank cross-encoder |
| **Embedding / LLM** | AWS Titan + Claude 3 Haiku (rerank) + Claude Sonnet 3.5 (gen) | Local Qwen2.5-1.5B Q8 GGUF + CPU embeddings; entirely offline |
| **Latency** | ~9.8 s/query (cloud APIs) | ~2–6 s/query (local, 4 GB VRAM) |
| **Domain** | German automotive manufacturing | Bilingual (Bengali–English) agricultural advisory |
| **Stakes** | Engineering doc lookup — wrong answer = rework | Pesticide/dosage — wrong answer = physical harm |

---

## 4. Gap AgriBot Fills

1. **Cloud-only architecture.** GraphRAG depends on AWS Textract + Bedrock (Claude). AgriBot replicates the *intent* of KG-augmented retrieval with **SQLite + FAISS + llama-cpp-python**, all offline on a 4 GB GPU.
2. **No agentic self-correction or evidence grading.** GraphRAG is single-pass (retrieve → rerank → generate). AgriBot's LangGraph state machine grades evidence and triggers **bounded retries with LLM-rewritten queries**.
3. **No safety layer.** GraphRAG cites sources but has no domain-specific harm-prevention. AgriBot adds **bilingual regex risk detection** (dosages, chemicals in EN + BN) feeding a refuse / disclaim / cited-facts-only policy.
4. **KG layers structure, not language.** GraphRAG's DKG/IKG bridge *documents to documents*; AgriBot's KG bridges *farmer dialect to canonical terms* — an orthogonal contribution explicitly flagged as future work by Knollmeyer et al. (§9: "Extending GraphRAG to multilingual settings is a natural next step").
5. **No multimodal / voice / dialect input.** GraphRAG is German/English text only; AgriBot ingests Bengali ASR + image + text.
6. **Latency budget mismatch.** GraphRAG's 9.8 s/query depends on cloud GPUs; AgriBot must hit similar latency on a single consumer card with no network.

---

## 5. Strengths to Acknowledge

- **Rigorous DSR methodology** — problem identification, design requirements (D1–D4), validated solution architecture, demonstration, evaluation. A useful template for AgriBot's own methodology section.
- **Three orthogonal graph-retrieval strategies (ICS / IKS / UKS)** — the cleanest published taxonomy of "what a KG can do for RAG retrieval" and directly informs how AgriBot's KG-aware query expansion can be framed.
- **Honest evaluation including a custom domain dataset** — they openly report that lexical retrieval metrics drop on the noisy real-world internal corpus (Recall@10 = 0.075) while LLM-as-Judge metrics (CR = 0.794) hold up, a transparency lesson worth emulating.
- **Practical pass-conditions + reranking design** — chunk size 1000, 5 keywords, top-k=10 as a defensible default after ablation across {700, 1000, 1300} × {3, 5, 8} × {5, 10, 20}.

---

## 6. BibTeX

```bibtex
@article{Knollmeyer2025DocumentGraphRAG,
  author    = {Knollmeyer, Simon and Caymazer, O\u{g}uz and Grossmann, Daniel},
  title     = {Document GraphRAG: Knowledge Graph Enhanced Retrieval Augmented
               Generation for Document Question Answering Within the Manufacturing
               Domain},
  journal   = {Electronics},
  volume    = {14},
  number    = {11},
  pages     = {2102},
  year      = {2025},
  publisher = {MDPI},
  doi       = {10.3390/electronics14112102}
}
```

---

## 7. Final Verdict

> **INCLUDE — Tier 1 (high priority).**
> Best published exemplar of a *layered* KG-RAG architecture (DKG + IKG) with a clean three-way taxonomy of graph-retrieval strategies and a rigorous DSR evaluation across two public + one industrial dataset. Strongest reference for AgriBot's claim that KG-augmented retrieval improves QA in domain-specific, long-document settings.

**Positioning sentence for your paper:**
> "Knollmeyer et al. show that layering a Document Knowledge Graph (capturing chapter hierarchy) over an Information Knowledge Graph (capturing keyword-based cross-chunk links) yields +15–18% Context Relevance over naive RAG on SQuAD, HotpotQA, and a 5,500-page automotive corpus, at a 9.8 s/query latency reliant on AWS Textract + Anthropic Claude. AgriBot adopts the same intuition that structure should be made first-class in retrieval, but (i) replaces the document-structural KG with a **linguistic** KG that maps Bengali dialect terms to canonical agricultural entities, (ii) embeds the graph in **SQLite** with FAISS + BM25 + RRF + cross-encoder rerank in place of cloud OCR + LLM rerank, and (iii) wraps the whole pipeline in a **LangGraph self-correcting state machine with bilingual safety enforcement** — delivering KG-aware multilingual agricultural QA fully offline on a 4 GB consumer GPU."
