# G-C · Call Center RAG over Structured + Unstructured Data (Manulife/John Hancock, NAACL 2025 Industry)

> **Verified from PDF.** Status: ❌ Original review had **wrong venue** (ACL 2024 → NAACL 2025) and **vague author attribution**. Industry context (insurance) is actually correct — Manulife + John Hancock are insurance/financial services subsidiaries.

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | Implementing Retrieval Augmented Generation Technique on Unstructured and Structured Data Sources in a Call Center of a Large Financial Institution |
| **Authors** | **Syed Shariyar Murtaza, Yifan Nie, Elias Avan, Utkarsh Soni, Wanyu Liao, Adam Carnegie, Cyril John Mathias, Junlin Jiang, Eugene Wen** |
| **Affiliations** | **Manulife** (200 Bloor St E, Toronto, ON, Canada) + **John Hancock** (200 Berkeley St, Boston, MA, USA) |
| **Venue** | **NAACL 2025 Industry Track**, pages **598–606** — Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies |
| **Date** | April 30, 2025 |
| **Length** | 9 pages |

### Audit of the original review (`paper_analysis_group_c.md` Paper 1)
- ❌ **Venue**: Original said "ACL 2024 Workshop (Industry Track), pp. 597–606" — actually **NAACL 2025 Industry Track, pp. 598–606**. Wrong conference and wrong year.
- ❌ **Authors**: Original said "(Industry team — insurance domain)" — full author list above. Manulife is the primary affiliation.
- ✅ **Domain**: Insurance call center — **correct** (Manulife + John Hancock are insurance subsidiaries).
- ✅ **Methodology**: GPT-3.5-turbo generation + GPT-4 validation + text-embedding-ada-002 + Azure AI Search — confirmed.
- ⚠️ **Headline numbers**: Original claimed "BLEU 0.41, ROUGE 0.64, METEOR 0.52; groundedness 4.55/5." The PDF's abstract reports **average response time 7.33s (down from 21.91s)** as the headline. The specific BLEU/ROUGE/METEOR numbers may appear in §4 (not in read portion). Confirm specific values before citing.
- ✅ **4.5M-row policy DB**: Confirmed verbatim — de-normalized from 50M rows to 4.5M, each row encoded as JSON string (max 1300 tokens) and embedded separately.

---

## 2. Paper Summary

### 2.1 Problem
A call center has both **unstructured** sources (policy PDFs, CSR notes on SharePoint) and **structured** sources (relational DB with 50M+ rows of policy data). Conventional RAG handles the first but stumbles on the second — text-to-SQL approaches are brittle, slow (more LLM calls), and require domain training. CSRs need single-query answers grounded in both.

### 2.2 Method — Hybrid Structured+Unstructured RAG
**Indexing:**
- Unstructured: PDFs chunked at 400 tokens with overlap → text-embedding-ada-002 → 1536-dim vectors → Azure AI Search index.
- **Structured (novel)**: De-normalize and aggregate by business concept → each row converted to **JSON string** (`{"Policy Number": "...", "Section Name": "...", ...}`) up to 1300 tokens → embed and store in **separate** Azure AI Search index. Reduces 50M rows → 4.5M chunks.
- Metadata (policy numbers, state, file location) attached for filtering and citation.
- Nightly incremental re-indexing for updated policy values.

**Inference (Promptflow on Azure Kubernetes Service):**
1. `query_rewording` expands acronyms (e.g., "D.C." → "Death Certificate").
2. Retrieve top-K from both indexes (K = 2–4, priority-tuned).
3. GPT-3.5 generates answer with citation to source numbers.
4. **Separate validator LLM (GPT-4 or Llama-3) rates groundedness 1–5** via few-shot prompting; users see confidence rating + rationale.

**Monitoring**: Devo logging, data versioning, prompt versioning, CI/CD via TeamCity on AKS.

### 2.3 Headline Results
- **Average response time**: **7.33s** (optimized from 21.91s initial launch).
- **Production status**: Live since **May 2024** with consistently grounded, hallucination-free answers.
- Comparison study of open-source and closed-source LLMs included (Section 4 — not visible in read portion).

---

## 3. Relevance to AgriBot

| Dimension | Call Center RAG | AgriBot |
|---|---|---|
| **Dual-source RAG** | Structured DB rows as JSON strings + unstructured PDFs in separate Azure AI Search indexes | **PDFs only** + SQLite KG as ontology (not row-vectorized DB) — different design |
| **Groundedness validation** | GPT-4 scores generation 1–5 (passive — answer shown with score) | **LLM verifier + grounding policy enforcement** (active — refuses/disclaims/strips low-confidence answers) |
| **Query preprocessing** | Acronym expansion ("D.C." → "Death Certificate") | **Bengali dialect → canonical scientific term mapping** via KG entity linking |
| **Citation** | Source-number annotations in bullet points | Citation brackets referencing evidence chunks + per-claim provenance |
| **Deployment** | Azure Cloud (AKS + Azure AI Search + Azure OpenAI) | **Fully offline on RTX 3050 4GB VRAM** |
| **Latency** | 7.33s avg (optimized from 21.91s) | Per-stage diagnostics; comparable target |
| **Hybrid retrieval** | Two separate dense indexes (no BM25) | **FAISS + BM25 + RRF + cross-encoder rerank** |
| **Safety stakes** | Insurance policy lookup — inconvenience risk | **Pesticide dosage** — physical-harm risk → mandatory refusal for unverified dosage |

---

## 4. Gap AgriBot Fills

1. **Validation ≠ Enforcement.** Manulife scores groundedness 1–5 and shows the answer regardless. AgriBot **acts** on the verifier output — configurable modes (refuse / disclaim / cited-facts-only).
2. **No domain-specific risk detection.** Insurance lookups don't trigger refusal logic. AgriBot has **bilingual regex** (English + Bengali) for dosage/chemical queries → mandatory refusal when unverified.
3. **No agentic self-correction.** Single-pass retrieve → generate → validate. AgriBot grades evidence **before** generation and retries with rewritten queries (bounded loop).
4. **No hybrid (dense + sparse) retrieval.** Manulife uses two dense Azure AI Search indexes; AgriBot adds **BM25 sparse** + RRF fusion + cross-encoder reranking.
5. **No bilingual / multimodal / voice / offline support.**
6. **No KG for query expansion.** Acronym expansion is regex-only. AgriBot's SQLite KG performs multi-hop dialect→canonical expansion via graph traversal.

## 5. Strengths to Acknowledge

- **JSON-stringified row vectorization is a clever idea.** Treating each de-normalized DB row as a self-contained JSON chunk avoids text-to-SQL complexity entirely. Could inspire AgriBot's treatment of structured agricultural data (e.g., pesticide dosage tables in extension manuals) — embed each row of a dosage table as a JSON-formatted chunk for direct retrieval.
- **GPT-4 grounding validation with 1–5 rubric + few-shot examples** is a clean evaluation prompt design. AgriBot's current LLM verifier uses simpler SUFFICIENT/INSUFFICIENT — could borrow this rubric for finer-grained grounding scores.
- **Production deployment data** (live since May 2024, sub-10s latency) provides real-world validation that LLM-validated RAG is operationally viable.
- **Honest reporting of latency optimization** (21.91s → 7.33s) — useful benchmark for AgriBot's own latency story.

## 6. BibTeX

```bibtex
@inproceedings{Murtaza2025CallCenterRAG,
  author    = {Murtaza, Syed Shariyar and Nie, Yifan and Avan, Elias and
               Soni, Utkarsh and Liao, Wanyu and Carnegie, Adam and
               Mathias, Cyril John and Jiang, Junlin and Wen, Eugene},
  title     = {Implementing Retrieval Augmented Generation Technique on Unstructured
               and Structured Data Sources in a Call Center of a Large Financial Institution},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas
               Chapter of the Association for Computational Linguistics:
               Human Language Technologies (Industry Track)},
  pages     = {598--606},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}
```

## 7. Final Verdict

> **INCLUDE — Tier 2 (industry RAG comparator).**
> NAACL Industry Track is a respectable venue. This is the strongest industry-deployed RAG-with-groundedness-validation reference for our paper. Best positioned to contrast **passive validation** (their approach) with AgriBot's **active enforcement** for safety-critical domains.

**Positioning sentence:**
> "Murtaza et al. (NAACL Industry 2025) demonstrate a production RAG system at Manulife combining structured insurance policy data and unstructured documents, with GPT-4 scoring groundedness on a 1–5 rubric. Their validator informs but does not enforce — answers below threshold are still returned with a confidence label. Our system extends this paradigm with **active grounding enforcement** (refuse / disclaim / cited-facts-only modes) driven by bilingual risk detection — necessary because incorrect agricultural advice on chemical dosage can cause physical harm, while incorrect call-center answers cause only inconvenience."
