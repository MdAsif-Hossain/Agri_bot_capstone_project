# G-B · KG-RAG (Scientific Reports 2025)

> **Verified from PDF.** Status: ⚠️ Original review **paraphrased the title and misattributed authorship**. Real title, authors, affiliation, and headline numbers (NaturalQuestions EM 47.5 / F1 61.0; PubMedQA 68.6%) all confirmed below.

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | Research on the construction and application of retrieval enhanced generation (RAG) model based on knowledge graph |
| **Authors** | Sili Wang, Heng Yang, Wei Liu |
| **Affiliation** | (1) Key Laboratory of Ecological Safety and Sustainable Development in Arid Lands, Northwest Institute of Eco-Environment and Resources, Chinese Academy of Sciences, Lanzhou 730000, China; (2) Key Laboratory of Knowledge Computing and Intelligent Decision, Lanzhou 730000, Gansu, China |
| **Corresponding author** | wangsl@llas.ac.cn |
| **Venue** | *Scientific Reports* (Nature Portfolio), Vol. 15, Article 40425 (2025) |
| **DOI** | 10.1038/s41598-025-21222-z |
| **Length** | ~17 pages (full research article) |

### Audit of the original review (`paper_analysis_group_b.md` Paper 2)
- ❌ **Title**: original review paraphrased as "KG-RAG: Research on the construction and application of RAG model based on knowledge graph" (or similar). The correct verbatim title uses "**retrieval enhanced generation (RAG) model based on knowledge graph**" — no acronym "KG-RAG" in the title itself; that label appears only inside the body.
- ❌ **Authors**: original review said "Chinese research team" or omitted authors. Real authors are **Sili Wang, Heng Yang, Wei Liu** (CAS Lanzhou — Northwest Institute of Eco-Environment and Resources).
- ✅ **Datasets**: NaturalQuestions + PubMedQA (and WebQuestions, OpenDialKG, MultiWOZ+KG for additional tasks).
- ✅ **Headline number — NaturalQuestions**: full KG-RAG model (Variant C: Soft + Multi-input Fusion) achieves **EM 47.5 / F1 61.0** (Table 1). The main comparison table (Table 5) reports the slightly higher **EM 49.8 / F1 64.2** for the full method against baselines — the discrepancy is because Table 1 is the fusion-strategy ablation while Table 5 is the head-to-head comparison.
- ✅ **Headline number — PubMedQA**: **68.6% accuracy** (Table 1, fusion ablation) and EM 68.9 / F1 73.2 in Table 6 (main comparison).
- ⚠️ The abstract additionally claims "PubMedQA … accuracy rate of 81%. 3% [sic], an improvement of 6.8% points over RAG" — this number does not match the experimental tables (which top out at 68.9% EM); appears to be either an editing artifact or refers to a different metric.
- ✅ Backbone models confirmed: **DPR + BERT (retriever)**, **T5-base / BART-base (generator)**, **Neo4j + UMLS** for graph backend, trained on **NVIDIA A100**.

---

## 2. Paper Summary

### 2.1 Problem
Traditional RAG over unstructured text suffers from (1) fact hallucination, (2) weak semantic/relational reasoning across entities, (3) poor multi-hop reasoning, and (4) inability to track knowledge provenance for high-stakes domains (medicine, law).

### 2.2 Method — KG-RAG
A **dual-channel retrieval architecture** that fuses unstructured-text retrieval with knowledge-graph path retrieval:

- **Text channel**: Domain-adapted fine-tuned DPR (BERT-base encoder) over Wikipedia (NQ) or PubMed corpus (PubMedQA). Contrastive learning with 1 positive + 4 negative samples per query. Reported **Precision@5 lift from 68.2% → 79.6%** (+11.4 pp) after fine-tuning.
- **Graph channel**: KG built with BERT-BiLSTM-CRF for NER, BERT + Multi-Head Attention for relation extraction, Wikidata/CN-DBpedia for entity disambiguation. Triples stored in **Neo4j**; entities embedded with TransE, relations with RotatE; paths scored by a **BiGRU-encoded path-attention** mechanism integrating structural distance + semantic similarity.
- **Diversity constraints**: cosine-similarity penalty (0.7 threshold, ×0.3 score multiplier) on redundant DPR paragraphs; ≥2 distinct relation types required among top-3 graph paths.
- **Hierarchical fusion / generation**:
  - Step 1: cluster top-5 DPR paragraphs (k-means, k=3) into topic-tagged "text knowledge blocks"; convert top-3 KG paths into natural-language "relationship-typed blocks" via templates.
  - Step 2: at each generation step, the model picks the 1 text block + 1 KG block with highest similarity to the partial output.
  - Step 3: concatenate as `[Text Block] + [KG Block] + [Generated text so far]` and feed to T5/BART.
- **Path-attention mechanism**: BiGRU encodes each candidate entity-relation path; learnable attention over (BERT-encoded query, path-vector) pairs; soft-attention weights aggregate paths into a single knowledge vector for prompt fusion.

### 2.3 Headline Results (verbatim)

**Fusion-strategy ablation (Table 1):**
| Variant | NQ (EM / F1) | PubMedQA (Acc) |
|---|---|---|
| A — Soft Fusion only | 42.3 / 56.7 | 66.1% |
| B — Multi-input Fusion only | 44.8 / 58.2 | 67.3% |
| **C — Combined (proposed)** | **47.5 / 61.0** | **68.6%** |

**Open-domain QA vs. baselines (Table 5):**
| Model | EM | F1 |
|---|---|---|
| DPR + BART | 42.3 | 57.8 |
| KG-BART | 44.9 | 59.6 |
| K-BERT | 46.1 | 61.0 |
| **KG-RAG (ours)** | **49.8** | **64.2** |

**Medical QA — PubMedQA (Table 6):** KG-RAG EM **68.9** / F1 **73.2** vs. DPR+BART 61.2 / 66.7.

**Knowledge-enhanced dialogue — OpenDialKG (Table 7):** BLEU-4 **15.2**, ROUGE-L **31.6**, Knowledge-F1 **52.8** (+4.8 / +5.5 / +11.5 vs. DPR+BART).

**Component ablation (Table 8):** removing Path Attention → −2.4 EM / −2.4 F1; removing Dual-Channel → −3.2 EM / −3.2 F1; removing Prompt Fusion → −3.7 EM / −4.8 F1 — confirming all three modules contribute.

**Training setup**: max input 512 tokens, output 128 tokens, batch 16, AdamW, lr 3e-5, 5 epochs (early stopping), NVIDIA A100.

---

## 3. Relevance to AgriBot

| Dimension | KG-RAG (Wang et al.) | AgriBot |
|---|---|---|
| **KG content** | Encyclopedic / medical entity–relation graph (Wikidata, UMLS, CN-DBpedia) | Linguistic dialect-to-canonical mapping (Sylheti / Chittagong / Barishal Bengali) |
| **Retrieval** | DPR (BERT-base) dense + GNN/path-attention over Neo4j | FAISS dense + BM25 sparse + RRF fusion + FlashRank cross-encoder |
| **Path reasoning** | BiGRU-encoded multi-hop paths with learned attention | Token-level KG alias expansion; no multi-hop traversal |
| **Generator** | T5-base / BART-base on A100 | Qwen2.5-1.5B-Instruct Q8 GGUF on RTX 3050 4 GB |
| **Domain** | Open-domain + medical + dialogue (en/zh) | Agricultural advisory (Bengali / English) |
| **Hardware** | NVIDIA A100, cloud-scale | 4 GB consumer GPU, fully offline |
| **Safety** | None reported | Bilingual regex risk detection + refuse/disclaim/cited-facts policy |

---

## 4. Gap AgriBot Fills

1. **Infrastructure mismatch.** KG-RAG requires Neo4j + A100 GPU + T5/BART fine-tuning. AgriBot achieves comparable KG-augmented retrieval with **SQLite + a 1.5B GGUF model** — feasible for rural deployment.
2. **No bounded retry / evidence grading.** KG-RAG is single-pass. AgriBot's **LangGraph state machine** grades retrieved evidence and rewrites the query (max 2 retries) before generation.
3. **No safety enforcement.** KG-RAG's medical results never gate dosage/treatment outputs behind safety checks. AgriBot's pipeline mandates **bilingual regex risk detection** for pesticide/dose mentions and routes to refusal or disclaimed answers.
4. **KG semantics are encyclopedic, not linguistic.** KG-RAG's KG encodes "what entities exist and how they relate"; AgriBot's KG encodes "how farmers *speak* about the same entity in different dialects". This addresses a problem KG-RAG explicitly does not — semantic input drift in low-resource languages.
5. **No multilingual / dialect / voice / image input.** KG-RAG is text-only and treats Wikidata/UMLS as the canonical reference; AgriBot must work with Bengali ASR, photographed pest images, and dialect spelling variants.
6. **No fine-tuning required.** KG-RAG fine-tunes DPR and T5/BART; AgriBot uses off-the-shelf embeddings + a pre-quantized generator, making redeployment trivial on new corpora.

---

## 5. Strengths to Acknowledge

- **Cleanest published ablation of KG-RAG components**: path attention, dual-channel retrieval, and prompt fusion are each isolated in Table 8, giving a clear engineering recipe.
- **Hierarchical fusion mechanism** (k-means topic blocks + relationship-typed graph blocks + per-step dynamic block selection) is a transferable design pattern that AgriBot could adapt for cited-facts-only generation.
- **Path-attention scoring with structural + semantic terms** is a more principled alternative to naive multi-hop traversal, and reportedly raises the relevant-path proportion from 72.5% → 89.3%.
- **Domain-adapted DPR fine-tuning gain (+11.4 pp Precision@5)** is a useful empirical anchor for justifying why retrievers should not be deployed off-the-shelf in specialized domains.

---

## 6. BibTeX

```bibtex
@article{Wang2025KGRAG,
  author    = {Wang, Sili and Yang, Heng and Liu, Wei},
  title     = {Research on the construction and application of retrieval enhanced
               generation (RAG) model based on knowledge graph},
  journal   = {Scientific Reports},
  volume    = {15},
  pages     = {40425},
  year      = {2025},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41598-025-21222-z}
}
```

---

## 7. Final Verdict

> **INCLUDE — Tier 2 (medium priority).**
> Solid Nature-portfolio (Scientific Reports) KG-RAG paper with a clean dual-channel + path-attention + prompt-fusion architecture and rigorous component ablation. Useful as a *methodological* reference for fusion design, but its assumption of A100-class infrastructure, fine-tunable T5/BART, and encyclopedic KGs (Wikidata/UMLS) make it complementary rather than competing with AgriBot's offline edge-deployment story.

**Positioning sentence for your paper:**
> "Wang et al. demonstrate that a dual-channel KG-RAG with BiGRU-encoded path attention and hierarchical text-plus-graph block fusion lifts NaturalQuestions EM from 42.3 to 49.8 and PubMedQA accuracy to 68.6% over a DPR + BART baseline, using Neo4j, UMLS, and an NVIDIA A100 training stack. AgriBot adopts the same core insight — that structured knowledge should be retrieved alongside unstructured text — but (i) replaces the encyclopedic KG with a **linguistic dialect ontology** (Sylheti / Chittagong / Barishal Bengali → canonical agricultural terms) stored in SQLite, (ii) substitutes path-attention with a lightweight FAISS + BM25 + RRF + cross-encoder rerank pipeline that runs on a 4 GB consumer GPU, and (iii) wraps generation in a **LangGraph state machine with bilingual safety enforcement** before any pesticide-dosage answer is emitted — extending KG-augmented RAG into the safety-critical, low-resource, fully-offline regime."
