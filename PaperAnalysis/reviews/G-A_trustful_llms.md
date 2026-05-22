# G-A1 · Trustful LLMs (TrustfulLLM + Dual Decoders + Hallucination Correction)

> **Verified from PDF.** Status: ⚠️ Existing review has **two issues** — the title was truncated and the headline groundedness number was reported as **4.97/5** when the actual value in Table 1 is **5.00/5**. Authors and Microsoft affiliations are correct.

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders |
| **Authors** | Xiaofeng Zhu, Jaya Krishna Mandivarapu |
| **Affiliation** | Microsoft Corporation (Zhu: WA, USA; Mandivarapu: GA, USA) |
| **Venue** | arXiv preprint (industry/applied paper from Microsoft Copilot team) |
| **arXiv** | 2411.07870v6 [cs.CL], 20 Dec 2024 |
| **DOI** | (none assigned — arXiv only) |
| **Length** | ~10 pages incl. appendix |

### Audit of the original review (`paper_analysis_group_a.md` Paper 1)
- ✅ Authors: correct — Xiaofeng Zhu and Jaya Krishna Mandivarapu, both at Microsoft.
- ❌ Title: original review used the trimmed phrase "TrustfulLLM…" — the verbatim title is **"Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders"**.
- ❌ Headline groundedness: original claimed **4.97/5**. Table 1 of the PDF reports **5.00/5** for both `TrustfulLLM + HC + Phi-3.5-mini-instruct` and `HC + Phi-3.5-mini-instruct` on the M365 benchmark. (The same 5.00 score is reproduced on the CNN/DailyMail summarization benchmark in Table 2.)
- ✅ Method summary (dual-decoder + post-hoc knowledge-triplet hallucination correction) is faithful.
- ✅ Domain (Microsoft 365 / Copilot product QA, ~10,000 QA pairs) is correct.

---

## 2. Paper Summary

### 2.1 Problem
Production copilots (Microsoft 365) need LLM answers that are **grounded in private product knowledge** that can change frequently (pricing, feature lists, supported registrars, etc.). RAG alone leaks hallucinations — generated sentences contain entities not present in the retrieved context, or substitute incorrect facts (e.g., wrong price, wrong vendor list).

### 2.2 Method
Two complementary mechanisms, applied on top of any decoder-only LLM:

1. **Hallucination Correction (HC) — Algorithm 1.** A post-processing pass that:
   - Extracts knowledge triplets `(subject, relation, object)` from the LLM's generated text `Ŷ` into a graph `g`.
   - Extracts triplets from the RAG context into a reference graph `G`.
   - For each triplet `t_i ∈ g`: if `subject ∉ G`, eliminate `t_i` (and its associated sentence in `Ŷ`); else replace/keep based on `G`.
   - Final answer `Y` is pruned to a minimum spanning tree of the verified subgraph `Ĝ` (cycle-removal step).
   - Production extension: nodes are embedded via a pre-trained transformer and matched to canonical entities via cosine similarity over a **FAISS** index (e.g., "M365 Business Basic" → "Microsoft 365 Business Basic").

2. **TrustfulLLM dual-decoder (§3.2).** Two weight-shared decoders derived from Phi-3 architecture:
   - Decoder A consumes the prompt; Decoder B consumes the guided/RAG context.
   - Cross-attention `CROSSATTN(H_p, H_g)` uses prompt hidden states as `query` and guided-context hidden states as `key`/`value`.
   - Only the prompt side generates token by token; the guided context contributes via cross-attention only.
   - Fine-tuned via LoRA, >400 epochs (notably higher than typical LoRA).

### 2.3 Headline Results (Table 1 — M365 benchmark, verbatim)

| Model | ROUGE-L | METEOR | Groundedness (1–5) | GPT-Similarity (1–5) | BERTScore |
|---|---|---|---|---|---|
| **TrustfulLLM + HC + Phi-3.5-mini-instruct** | **0.55** | **0.51** | **5.00** | **4.68** | **0.93** |
| TrustfulLLM + Phi-3.5-mini-instruct | 0.50 | 0.50 | 3.98 | 4.30 | 0.90 |
| HC + Phi-3.5-mini-instruct | 0.46 | 0.48 | 5.00 | 4.52 | 0.91 |
| RAG + Phi-3.5-mini-instruct | 0.41 | 0.45 | 3.72 | 3.49 | 0.89 |
| RAG + GPT-4o | 0.40 | 0.46 | 3.77 | 3.52 | 0.91 |
| GPT-4o (no RAG) | 0.23 | 0.31 | 3.64 | 3.97 | 0.83 |

- The percentage of **eliminated entities when HC is applied** drops from **18% (HC on Phi-3.5)** to **6.9% (HC on TrustfulLLM + Phi-3.5)**, indicating the dual-decoder fine-tuning already produces fewer ungrounded entities.
- On CNN/DailyMail summarization (Table 2), TrustfulLLM+HC again hits **Groundedness = 5.00**, GPT-Sim = 4.12, BERTScore = 0.89.

### 2.4 Production Notes
- Upstream intent detection filters out-of-scope queries.
- Pre-calibrated offline triplets stored in FAISS; combined with real-time RAG triplets at inference.
- Red-team evaluation across harmful content, IP infringement, jailbreaking, groundedness reported (no numeric breakdown).

---

## 3. Relevance to AgriBot

| Dimension | TrustfulLLM (Microsoft) | AgriBot |
|---|---|---|
| **KG purpose** | Triplets extracted from RAG context act as a **verification scaffold** — any generated entity absent from the context graph is pruned | KG encodes **dialect → canonical scientific term** mappings (Sylheti/Chittagong/Barishal → standard Bangla/Latin agronomy terms); used at *query rewriting* time, not as a post-hoc filter |
| **Hallucination control** | Post-processing entity-graph diff + minimum-spanning-tree pruning over generated text | Pre-generation **evidence grading** (LangGraph node) + post-generation **regex risk detection** + grounding-policy enforcement (refuse / disclaim / cited-facts-only) |
| **Architecture** | Dual-decoder (weight-shared) fine-tuned via LoRA (~400 epochs) on top of Phi-3.5-mini | Single-decoder Qwen2.5-1.5B-Instruct Q8 GGUF, **no fine-tuning** — orchestrated by a stateful agent graph |
| **Vector store** | FAISS for canonical-entity matching | FAISS dense + BM25 sparse + RRF fusion + FlashRank cross-encoder rerank |
| **Domain** | Microsoft 365 product QA (~10K pairs, mostly English) | Bengali–English agricultural advisory, rural Bangladesh |
| **Deployment** | Cloud Copilot service (Azure) | Fully offline, RTX 3050 4 GB VRAM |
| **Stakes of error** | Wrong price/feature → customer confusion, refund risk | Wrong dosage/chemical → physical harm to farmer or crop |

---

## 4. Gap AgriBot Fills

1. **No bilingual / dialect handling.** TrustfulLLM operates on English product text only. AgriBot's KG is *linguistically* grounded — it solves the problem of farmers using non-standard dialect terms that never appear in a textbook RAG corpus, which TrustfulLLM's entity-matching FAISS cannot do.
2. **No safety enforcement layer.** TrustfulLLM verifies factual entity correctness but does not screen for *harmful* content categories (toxic dosage, restricted pesticides). AgriBot adds **bilingual regex risk detection** that triggers mandatory refusal regardless of grounding score.
3. **No agentic self-correction.** TrustfulLLM is a single forward pass + post-process. AgriBot's LangGraph state machine **rewrites and re-retrieves** (max 2 retries) when the grading node deems evidence insufficient — orthogonal to TrustfulLLM's entity-graph pruning, and could be combined.
4. **Requires LoRA fine-tuning + dual-decoder modification.** Not feasible on 4 GB consumer GPU. AgriBot's grounding policy is **training-free** and runs on a stock GGUF model.
5. **No multimodality / voice.** TrustfulLLM is text-only. AgriBot integrates ASR (faster-whisper + BanglaSpeech2Text + Vosk), TTS, MobileNetV3+CBAM ONNX rice-disease classifier, and OCR/color heuristics for general crop images.
6. **No translation pipeline.** TrustfulLLM assumes a single-language deployment. AgriBot routes through BanglaT5 for bidirectional EN↔BN.

---

## 5. Strengths to Acknowledge

- **Strong empirical headline.** Pushing groundedness to a perfect **5.00/5** on a real Microsoft commercial dataset (M365) while improving ROUGE-L, METEOR, GPT-Similarity, and BERTScore over GPT-4o + RAG is a notable result.
- **Production-validated.** The paper explicitly describes the live Copilot pipeline (intent detection → retrieval → HC → response) and Red Team evaluation across responsible-AI metrics.
- **HC algorithm is model-agnostic.** Algorithm 1 (triplet-graph diff + MST pruning) can wrap *any* LLM output. AgriBot's grounding policy module could borrow this as an additional verification layer between `generate` and `verify` nodes in the LangGraph DAG.
- **FAISS canonical-entity matching at inference** is a concrete pattern AgriBot already echoes (FAISS-backed alias lookup in the SQLite KG).

---

## 6. BibTeX

```bibtex
@misc{Zhu2024TrustfulLLMs,
  author        = {Zhu, Xiaofeng and Mandivarapu, Jaya Krishna},
  title         = {Trustful {LLMs}: Customizing and Grounding Text Generation
                   with Knowledge Bases and Dual Decoders},
  year          = {2024},
  eprint        = {2411.07870},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2411.07870},
  note          = {Version v6, 20 Dec 2024. Microsoft Corporation.}
}
```

---

## 7. Final Verdict

> **INCLUDE — Tier 2 (supporting / methodological reference).**
> Strong industry-grade evidence that **KB-driven post-hoc entity verification + a guided-context decoder** can drive groundedness scores to ceiling on a production QA workload. Useful as a methodological precedent for AgriBot's grounding policy, even though the dual-decoder fine-tuning is out of scope for 4 GB consumer hardware.

**Positioning sentence for your paper:**
> "Zhu and Mandivarapu (Microsoft, 2024) achieve perfect groundedness scores (5.00/5 GPT-4-rated) on Microsoft 365 product QA by combining a knowledge-triplet-based hallucination-correction algorithm with a weight-shared dual-decoder LLM fine-tuned on the target domain. AgriBot adopts the same **'verify generated entities against a structured reference'** principle, but (i) replaces fine-tuning with a training-free LangGraph grounding-policy node compatible with stock GGUF models, (ii) extends entity verification with **bilingual regex-based safety screening** for dosage and chemical risks, and (iii) operates entirely offline on a single RTX 3050 4 GB GPU — constraints that preclude the cloud-hosted Copilot stack TrustfulLLM targets."
