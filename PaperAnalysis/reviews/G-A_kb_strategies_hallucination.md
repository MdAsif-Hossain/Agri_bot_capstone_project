# G-A · KB Representation Strategies for Hallucination Reduction (Cheng-Hsiu Li, 2025)

> **Verified from PDF.** Status: ⚠️ Original review had **two author/methodology errors** plus a counting error. Substantive findings hold up.

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | Knowledge Representation Strategies for Reducing Hallucinations in Retrieval-Augmented Domain-Specific Question Answering |
| **Author** | **Cheng-Hsiu Li** (single author) |
| **Affiliation** | Department of Information Management, **National Taitung Junior College**, Taitung, Taiwan, ROC |
| **Email** | chsiu.lee@gmail.com |
| **Venue** | International Journal of Engineering and Technology Innovation, **vol. 15, no. 4, 2025, pp. 476–487** |
| **DOI** | 10.46604/ijeti.2025.15708 |
| **Dates** | Received 18 Sep 2025, revised 17 Oct 2025, accepted 20 Oct 2025 |
| **Length** | 12 pages |

### Audit of the original review (`paper_analysis_group_a.md` Paper 3)
- ❌ **Authors**: Original said "Taiwan-based team, names in paper" — it's actually a **single-author paper** by **Cheng-Hsiu Li**. Affiliation is **National Taitung Junior College**, not generic "Taiwan."
- ❌ **Counting error**: Original said "5 knowledge base representation formats (Type0–Type5)" — that range is **6 types** (Type0, 1, 2, 3, 4, 5).
- ⚠️ **Methodology framing**: Original said "QA-format knowledge base (Type2) outperforms raw text" and "Enhanced QA+explanations (Type5) is best." Type5 best is correct; verbatim numbers are Qwen2 = **98 points**, LLaMA2 = **73 points** (out of 100 across 10 standardized questions, scored by domain experts).
- ✅ Models, framework (Ollama + FAISS), domain (Taiwan Level B computer hardware repair certification) — all correctly identified.
- ✅ RLHF-like user feedback loop is real, with **60% weight on user feedback** in the ensemble retriever (this specific weighting was not in the original review).

---

## 2. Paper Summary

### 2.1 Problem
Hallucinations remain a critical failure mode for domain-specific LLM tutors. Existing RAG implementations vary widely in how they format the underlying knowledge base (raw text vs. structured vs. Q&A pairs), and no systematic comparison exists for the impact of **knowledge representation format** on response accuracy.

### 2.2 Method — Six KB Format Variants
Develops an AI teaching assistant chatbot for the **Taiwan Level B certification for computer hardware fabrication** (a vocational skill exam with a ~38% pass rate). Six knowledge-base structures are constructed and tested:

| Type | Content structure |
|---|---|
| Type0 | Raw textbook text (baseline) |
| Type1 | Text with images |
| Type2 | Q&A format |
| Type3 | Keyword-enriched |
| Type4 | Q&A + keywords |
| Type5 | **Q&A + explanations (enhanced)** |

**System stack** (all open-source, local):
- **Framework**: Ollama + Python + VS Code
- **LLMs**: **LLaMA2 7B** and **Qwen2 7B** (both locally deployed)
- **PDF processing**: PyPDFLoader + RecursiveCharacterTextSplitter
- **Embeddings**: OllamaEmbeddings
- **Vector DB**: FAISS (with persistence + caching, hash-based deduplication)
- **RAG chain**: ConversationalRetrievalChain + EnsembleRetriever
- **Feedback loop**: JSONL-logged user feedback re-vectorized and weighted at **60% in retrieval** vs. 40% document — an RLHF-style continuous-learning loop.

### 2.3 Key Results
| Configuration | Score (out of 100, expert-evaluated, 10 questions) |
|---|---|
| **Qwen2 + Type5** | **98** (best overall) |
| LLaMA2 + Type5 | 73 (best for LLaMA2) |
| Both models, Type0 (baseline) | Significantly lower |

Statistical tests confirm significant improvements across knowledge types and between models. The author concludes that **KB representation matters as much as model choice**.

---

## 3. Relevance to AgriBot

| Dimension | Li 2025 (this paper) | AgriBot |
|---|---|---|
| **Local LLM + FAISS RAG stack** | Ollama + Qwen2 7B + FAISS + ConversationalRetrievalChain | llama-cpp-python + Qwen2.5-1.5B Q8 + FAISS + custom LangGraph pipeline |
| **Domain-specific QA** | Taiwan technician certification prep | Bangladesh agricultural advisory |
| **Knowledge representation study** | Compares 6 static KB formats post-hoc | Uses **layout-aware Marker PDF extraction** + page classification (content/TOC/reference/low_signal) + noise filtering — single, sophisticated pipeline |
| **Reranking** | Ensemble retriever (60/40 doc+feedback) | **FlashRank cross-encoder** with threshold filtering |
| **Self-correction** | None — single-pass retrieval | **LangGraph evidence grading + bounded retry + LLM query rewriting** |
| **Verification / safety** | None | **LLM answer verification + bilingual grounding policy** |
| **Feedback loop** | RLHF-like JSONL logging + re-vectorization | Not implemented (flagged as future work in TODO.md) |
| **Language** | Traditional Chinese | **Bengali–English bilingual + dialect mapping** |
| **Modality** | Text only | **Text + voice + image** |
| **Domain stakes** | Wrong answer = exam failure (low risk) | Wrong dosage = physical harm (high risk) |

---

## 4. Gap AgriBot Fills

1. **No agentic pipeline.** Li uses a single-pass ConversationalRetrievalChain. AgriBot's LangGraph state machine grades evidence before generation and retries with LLM-rewritten queries (bounded retry).
2. **No cross-encoder reranking.** Li relies on ensemble dense retrieval; AgriBot adds FlashRank ms-marco-MiniLM-L-12-v2 reranking with threshold filtering — empirically better precision in the top-K.
3. **No verification or safety enforcement.** Li's system shows whatever Qwen2 generates from retrieved chunks. AgriBot verifies the answer against evidence and enforces grounding policies (refuse / disclaim / cited-facts-only).
4. **No domain-specific risk detection.** Certification prep has no physical-safety dimension. AgriBot has bilingual regex (EN + BN) for dosage/chemical queries triggering mandatory refusal.
5. **No bilingual / dialect / multimodal / voice support.**
6. **KB study scope is narrow** — Li compares 6 static text formats. AgriBot's contribution is a *dynamic, layout-aware* extraction pipeline that handles multi-column manuals, tables-as-Markdown, scanned pages (OCR fallback), and page-level relevance weighting — orthogonal to the KB-format question.

## 5. Strengths to Acknowledge

- **The KB-format finding is genuinely actionable**: Type5 (Q&A + explanations) substantially outperforming raw text (Type0) is a useful design lesson AgriBot could incorporate. For the most queried agricultural disease cases, **pre-curating Q&A pairs with explanations** alongside the raw PDF chunks may improve recall and answer quality.
- The **60/40 ensemble weighting toward user-feedback** is a practical implementation pattern for RLHF lite — worth borrowing when AgriBot's feedback loop ships.
- **Honest negative result**: LLaMA2 plateaus at 73 even with Type5 — concrete evidence that model choice matters as much as KB format.

## 6. BibTeX

```bibtex
@article{Li2025KBHallucination,
  author  = {Li, Cheng-Hsiu},
  title   = {Knowledge Representation Strategies for Reducing Hallucinations in
             Retrieval-Augmented Domain-Specific Question Answering},
  journal = {International Journal of Engineering and Technology Innovation},
  volume  = {15},
  number  = {4},
  pages   = {476--487},
  year    = {2025},
  doi     = {10.46604/ijeti.2025.15708}
}
```

## 7. Final Verdict

> **INCLUDE — Tier 2 (supporting reference).**
> The journal (IJETI) is solid but not top-tier. Its real value is providing **direct empirical evidence** that KB representation format significantly affects RAG quality in domain-specific QA — exactly the design dimension AgriBot optimizes via Marker layout-aware extraction. Not a methodological competitor; a foundational design-rationale citation.

**Positioning sentence:**
> "Li (2025) demonstrates that knowledge-base representation format — comparing six text structures from raw text to Q&A-with-explanations — significantly affects RAG accuracy in domain-specific QA, with structured Q&A formats outperforming raw text by up to 25 percentage points on a Qwen2-7B local LLM. This motivates our choice of layout-aware document extraction with page-level relevance weighting, treating agricultural manual sections (tables, figures, dosage charts) as first-class structured chunks rather than uniform fixed-length text segments."
