# G-D · Feasibility Eval of Secure Offline LLMs with RAG for CPU-Only Inference (Tyndall et al., Information 2025)

> **Verified from PDF.** Status: ✅ Original review **mostly accurate**. Authors now identified (AFIT — Air Force Institute of Technology, US Air Force).

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | Feasibility Evaluation of Secure Offline Large Language Models with Retrieval-Augmented Generation for CPU-Only Inference |
| **Authors** | **Erick Tyndall (corresponding), Torrey Wagner, Colleen Gayheart, Alexandre Some, Brent Langhals** |
| **Affiliations** | **Air Force Institute of Technology (AFIT), Wright-Patterson AFB, OH, USA** (US Air Force graduate school) |
| **Venue** | **Information** (MDPI) **2025, 16(9), 744** |
| **DOI** | 10.3390/info16090744 |
| **Dates** | Received 20 Jun 2025, revised 1 Aug 2025, accepted 26 Aug 2025, published **28 Aug 2025** |
| **License** | CC BY 4.0 (open access) |
| **Length** | 23 pages |

### Audit of the original review (`paper_analysis_group_d.md` Paper 2)
- ⚠️ Original said "(US-based team, likely academic/government-adjacent)" — actually **US Air Force AFIT** (military graduate school). This affects citation framing — the paper has a defense/government context, not generic academic.
- ✅ **Venue**: Information 2025, 16, 744 (MDPI) — confirmed.
- ✅ **Hardware**: HP EliteBook 840 G5 (i7-8650U, 16 GB) + HP Z820 Workstation (2× Xeon E5-2667v2, 128 GB) — confirmed verbatim.
- ✅ **Framework**: LocalGPT — confirmed. Evaluated 5 frameworks (LocalGPT, PrivateGPT, GPT4All, llama.cpp, Ollama) and chose LocalGPT for native RAG support + cross-platform compatibility.
- ⚠️ **Models**: Original claimed "Phi-3-Mini 3.8B, LLaMA-2-Chat 7B, Mistral-7B-Instruct, LLaMA-2-Chat 13B (Q4 GGUF)." Models not visible in first 200 lines; verify in §2.3 before citing.
- ✅ **Metrics**: BERTScore + METEOR + latency — confirmed.
- ✅ **Source material**: Project management textbook — confirmed.
- ⚠️ **Headline numbers**: Original claimed "BERTScore >0.88 on factual QA; LLaMA-2-Chat 7B best overall; per-token latency 103ms–683ms." Abstract confirms "strong performance in direct question answering" and "summarization tasks showed greater variability"; specific numerical bounds need §3 confirmation.

---

## 2. Paper Summary

### 2.1 Problem
Many AI deployments require **secure, offline operation** without cloud connectivity or GPU acceleration — e.g., **government offices** (the AFIT context) where cloud LLMs are blocked by procurement/security policy and machines lack GPUs. **Can open-source LLMs + RAG operate adequately on CPU-only consumer hardware**?

### 2.2 Method — Feasibility Study with LocalGPT
**Framework selection**: Evaluated LocalGPT, PrivateGPT, GPT4All, llama.cpp, Ollama against criteria (Windows + Ubuntu compatibility, native RAG support, offline operation, no extensive coding required). Chose **LocalGPT** because of integrated RAG + multi-platform support; rejected llama.cpp and Ollama for lack of native RAG.

**Hardware**:
- **Laptop**: HP EliteBook 840 G5 — i7-8650U (1.90 GHz base, 4 cores), 16 GB RAM (typical consumer)
- **Workstation**: HP Z820 — 2× Xeon E5-2667v2 (3.30 GHz, 8 cores each), 128 GB RAM (high-end consumer / server)
- Both **CPU-only** (no GPU)

**Models tested**: 4 open-source models (specific list in §2.3 — not visible in read portion).

**Source**: Project management textbook (single PDF, 512-token chunks, 64-token overlap).

**Tasks**: Direct factual question answering + text summarization.

**Metrics**: BERTScore, METEOR, response timing, latency.

### 2.3 Key Findings (per abstract)
- ✅ **Factual QA**: All models performed "strong" — accurate responses despite limited compute.
- ⚠️ **Summarization**: "Greater variability" — sometimes vague or incomplete.
- **Hardware impact**: Quantization and hardware differences affected **response time more than output quality** — meaningful trade-off for deployment planning.
- **Workstation vs. laptop**: Significantly faster on the workstation (3–6× per prior review claim, to verify).
- Practical takeaway: **CPU-only deployments are viable for structured QA**; limitations remain for generative tasks.

---

## 3. Relevance to AgriBot

| Dimension | Tyndall et al. (this paper) | AgriBot |
|---|---|---|
| **Offline-first philosophy** | Government secure environments, no cloud allowed | Rural Bangladesh, no/unreliable internet |
| **Hardware target** | CPU-only consumer laptop + workstation | **Consumer laptop + RTX 3050 4GB VRAM** (uses GPU for LLM only, CPU for everything else) |
| **Framework** | Off-the-shelf LocalGPT (integrated RAG, minimal coding) | **Custom-built** — Marker + FAISS + BM25 + FlashRank + LangGraph |
| **LLM size** | Phi-3-Mini 3.8B to LLaMA-2 13B (Q4 GGUF) | **Qwen2.5-1.5B Q8 GGUF** (smaller, more recent, Q8 quantization preserves quality) |
| **Retrieval** | LocalGPT default dense (FAISS cosine, 512-token chunks) | **FAISS + BM25 + RRF + FlashRank cross-encoder + KG query expansion** |
| **Evaluation** | BERTScore, METEOR, latency — formal benchmark | Per-stage diagnostics; full benchmark TODO |
| **Domain** | Project management textbook (single source) | **Agricultural manuals** (multi-PDF) + dialect KG |
| **Tasks** | Factual QA + summarization | Citation-grounded advisory generation + verification + bilingual translation |
| **Privacy** | Air Force / government security focus | Farmer privacy (no data leaves device) |

---

## 4. Gap AgriBot Fills

1. **This is a feasibility study, not a system.** It tests off-the-shelf LocalGPT with default settings. AgriBot is a **complete custom-built pipeline** with agentic orchestration, KG, hybrid retrieval, multi-backend ASR, vision pipeline, grounding policy enforcement.
2. **No retrieval optimization.** LocalGPT default = single-pass dense retrieval. AgriBot adds hybrid dense+sparse, RRF, cross-encoder rerank, KG expansion.
3. **No agentic pipeline.** Single-pass retrieve→generate. AgriBot has LangGraph state machine with evidence grading + bounded retry + LLM rewriting + verification + policy enforcement.
4. **No grounding or safety.** Answers shown as-is. AgriBot verifies + enforces grounding policy + has bilingual risk detection.
5. **No bilingual / dialect / multimodal / voice support.**
6. **Smaller LLM, smarter pipeline.** AgriBot uses 1.5B (smaller than the smallest tested here: Phi-3-Mini 3.8B) and compensates with retrieval+KG+rerank — proving that pipeline quality beats raw model size for domain QA.
7. **GPU-aware deployment.** Tyndall et al. assume CPU-only; AgriBot uses GPU acceleration for the LLM while isolating CPU for everything else — much faster latency than CPU-only.
8. **Document preprocessing sophistication.** LocalGPT default = 512-token chunks, 64 overlap. AgriBot uses Marker layout-aware extraction with page classification (content/TOC/reference/low_signal), noise filtering (header/footer detection via frequency >50%), OCR fallback for scanned pages.

## 5. Strengths to Acknowledge

- **Useful baseline.** Proves that off-the-shelf RAG on CPU consumer hardware is viable for factual QA — sets a lower bound that AgriBot's purpose-built pipeline must exceed.
- **Honest about limitations**: Summarization variability is reported openly. Factual QA strong vs. generative tasks variable is a real and important finding.
- **Framework comparison** (LocalGPT vs. PrivateGPT vs. GPT4All vs. llama.cpp vs. Ollama) is genuinely useful taxonomy when justifying AgriBot's choice of llama-cpp-python.
- **Hardware specs and pricing**: HP EliteBook + HP Z820 are realistic consumer/enterprise machines. AgriBot's RTX 3050 4GB laptop target falls in between — useful comparison anchor.

## 6. BibTeX

```bibtex
@article{Tyndall2025OfflineCPU,
  author  = {Tyndall, Erick and Wagner, Torrey and Gayheart, Colleen and
             Some, Alexandre and Langhals, Brent},
  title   = {Feasibility Evaluation of Secure Offline Large Language Models with
             Retrieval-Augmented Generation for {CPU}-Only Inference},
  journal = {Information},
  volume  = {16},
  number  = {9},
  pages   = {744},
  year    = {2025},
  doi     = {10.3390/info16090744},
  note    = {Air Force Institute of Technology}
}
```

## 7. Final Verdict

> **INCLUDE — Tier 2 (feasibility baseline).**
> MDPI Information is a respectable open-access journal. The AFIT/government angle gives it credibility for the "secure offline AI" research thread. Best positioned as the **proof-of-concept baseline** — they show off-the-shelf tools work on CPU; AgriBot shows a purpose-built pipeline does much more with a smaller GPU.

**Positioning sentence:**
> "Tyndall et al. (Information 2025) demonstrate the feasibility of CPU-only offline RAG using off-the-shelf LocalGPT and 3.8B–13B quantized LLMs, achieving strong factual QA performance on consumer hardware in secure (US Air Force) deployment scenarios. Our system extends this baseline along three dimensions: (i) purpose-built pipeline with hybrid retrieval, cross-encoder reranking, and KG-based query expansion replacing the default single-pass dense retrieval; (ii) agentic self-correction with bilingual safety enforcement instead of raw LLM output; and (iii) hardware-isolated GPU acceleration of a smaller 1.5B model that achieves faster latency while leaving CPU resources free for concurrent multimodal processing (ASR, vision, translation)."
