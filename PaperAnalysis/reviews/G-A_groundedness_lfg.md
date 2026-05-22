# G-A2 · Groundedness in Retrieval-Augmented Long-Form Generation

> **Verified from PDF.** Status: ❌ Existing review has **major attribution and venue errors**. The paper is **single-authored by Alessandro Stolfo (ETH Zürich)**, not a four-author team, and was published at **Findings of NAACL 2024**, not "NAACL 2025". The technical findings are summarised correctly.

---

## 1. Verified Citation

| Field | Value |
|---|---|
| **Title (exact)** | Groundedness in Retrieval-augmented Long-form Generation: An Empirical Study |
| **Author** | Alessandro Stolfo (single author) |
| **Affiliation** | ETH Zürich (work partially carried out while at Oracle Labs) |
| **Venue** | Findings of the Association for Computational Linguistics: NAACL 2024, pages 1537–1552 |
| **Date** | June 16–21, 2024 |
| **arXiv** | 2404.07060 |
| **Length** | 16 pages (incl. appendix) |

### Audit of the original review (`paper_analysis_group_a.md` Paper 2)
- ❌ Authors: original review listed **four authors — "Stolfo, Skean, Jin, Sachan"**. The PDF header and footer clearly show **a single author: Alessandro Stolfo**. The other three names appear to be fabricated.
- ❌ Affiliation: original implied a multi-institution team. The PDF lists only **ETH Zürich** (`stolfoa@ethz.ch`), with an acknowledgement that part of the work was done at **Oracle Labs** during an internship.
- ❌ Venue: original review claimed "NAACL 2025". The PDF footer is unambiguous: **"Findings of the Association for Computational Linguistics: NAACL 2024, pages 1537–1552, June 16–21, 2024"**.
- ✅ Headline finding (~25% of EM+ outputs from Falcon 180B remain ungrounded) is correctly summarized.
- ✅ Datasets (ASQA, HotpotQA, StrategyQA) and model families (Pythia, Falcon, MPT, Silo) are correct.
- ✅ Finding that **instruction tuning + beam search** reduce ungrounded generations is faithfully reported.

---

## 2. Paper Summary

### 2.1 Problem
In **long-form QA (LFQA)**, an LLM may produce sentences that *contain* a ground-truth answer string yet are not actually supported by either (a) the **retrieved documents** or (b) any document in the model's **pre-training corpus**. Prior groundedness evaluations focus on short outputs or rely on web-search heuristics. This paper asks: across model families and sizes, how frequently does this happen, and what factors (size, decoding, instruction tuning) move the needle?

### 2.2 Method — dual-source grounding verification

For each generated sentence `s` from `M(p)` the paper computes two grounding flags:

- `g_R(q)(s) = 1` if any document in the retrieved set `R(q)` entails `s` (per the TRUE NLI model — a T5-11B trained on NLI).
- `g_RC(s)(s) = 1` if any of the top-5 documents retrieved from the **model's pre-training corpus `C`** (via MiniLM-v2 sentence embeddings + exact FAISS search) entails `s`.

Sentences are partitioned across the 2×4 grid of (correctness × groundedness): `{EM0, EM+}` × `{None, Retr-only, PT-only, Both}`.

**Manual validation of the TRUE grounding model**: annotators agree with the model on **82% of cases overall** and **98% of the correct-but-ungrounded cases** (the analytically interesting category) — strong support for the methodology.

### 2.3 Experimental Setup
- **Datasets**: ASQA (ambiguous factual questions), HotpotQA (multi-hop Wikipedia), StrategyQA (multi-hop True/False). For HotpotQA and StrategyQA, retrieval is **oracle** (gold passages supplied) to isolate generation from retrieval failures.
- **Models**: Pythia (70M → 12B), Falcon (7B/40B/180B), MPT (7B/30B base + instruct), Silo (1.3B).
- **Pre-training corpora used for post-hoc retrieval**: The Pile (Pythia), Open-license Corpus (Silo), C4 (MPT and Falcon — ~60% of MPT's training).

### 2.4 Headline Findings (verbatim numbers)

- **Pythia 12B on ASQA (Figure 2):**
  - 44% of generated sentences are grounded in at least one source.
  - 48% belong to `EM+` (output contained ≥1 ground-truth answer).
  - **Nearly half** of `EM+` sentences cannot be grounded in either retrieved docs or pre-training corpus — i.e., correct answers are wrapped in plausible-but-fabricated prose.
  - Specific sector areas reported: `None,EM+` = 22.7%, `None,EM0` = 33.1%, `Both,EM+` = 11.6%, `Both,EM0` = 15.9%, `Ret.Docs,EM+` = 6.11%, `Ret.Docs,EM0` = 4.5%, `PT Corpus,EM+` = 2.8%, `PT Corpus,EM0` = 3.2%.

- **Scaling (Figure 4):** Falcon 40B and 180B show clear increases in grounded content vs. Pythia 70M–12B; however **even at Falcon 180B, roughly 25% of EM+ outputs remain ungrounded**.

- **Decoding (Figure 5):** As the groundedness threshold tightens, EM scores for greedy / nucleus sampling fall sharply, but **beam search (b=2, 5, 10) remains nearly flat** — beam search produces more grounded (more extractive) text. Trade-off: beam search outputs are shorter and copy longer LCS spans from retrieved docs (Figure 6).

- **Instruction tuning (Figure 7):** Across MPT-7B / 30B and Falcon 7B / 40B / 180B, the instruction-tuned variant produces both more `EM+` and a higher grounded fraction than its base counterpart.

### 2.5 Failure-mode examples
Table 1 shows Pythia 12B confidently producing the *correct* fact "Brisbane Broncos last won a premiership in 2006" alongside the *fabricated* claim "The Pacific Ocean is connected to the Atlantic Ocean via the Gulf of California." Table 2 shows Falcon 180B saying "Brazil has won the World Cup 5 times" (correct) while attributing 5 titles to Germany (wrong — 4) and inventing Pink Floyd's "hammer in the foreground" album-cover detail. The pattern: **correct entities embedded in subtly fabricated relationships**, exactly the hallucination class that simple retrieval cannot fix.

---

## 3. Relevance to AgriBot

| Dimension | Stolfo 2024 | AgriBot |
|---|---|---|
| **Contribution type** | Diagnostic / measurement study — quantifies *how often* RAG fails to ground long-form outputs | Engineering / system contribution — *enforces* grounding via policy, not just measures it |
| **Grounding signal** | TRUE NLI model (T5-11B) + dual-source check (R(q) and pre-training corpus) | Pre-generation evidence grading (LangGraph `grade` node) + post-generation verification + bilingual regex risk filter |
| **Domain** | Open-domain Wikipedia QA | Closed-domain Bengali–English agricultural advisory |
| **Output length** | Long-form, multi-sentence answers | Short to medium answers; safety-critical recommendations |
| **Compute footprint** | T5-11B grounder + Falcon 180B generator — research-cluster scale | Single RTX 3050 4 GB; T5-11B-class verification is infeasible |
| **Mitigation** | Identifies that **instruction tuning + beam search** help | Operationalizes mitigation as agentic retries + grounding-policy refusal |

---

## 4. Gap AgriBot Fills

1. **Stolfo measures groundedness; AgriBot enforces it.** The paper's contribution is empirical — there is no proposed pipeline, no refusal mechanism, no UX-level mitigation. AgriBot's LangGraph state machine turns the diagnostic insight into a runtime guarantee: low-grounded outputs trigger rewrite + retry (max 2), and if still ungrounded, the grounding policy escalates to disclaim or refuse.
2. **No bilingual or dialect evaluation.** All three benchmarks (ASQA, HotpotQA, StrategyQA) are English. The "correct-entity-wrapped-in-fabricated-relationship" failure mode is *even harder* to detect when the entity is a transliterated Bangla pesticide name. AgriBot's KG normalizes dialect → canonical entity *before* retrieval, eliminating an entire class of false-grounding signals.
3. **No safety taxonomy.** Stolfo treats all ungrounded sentences equally. AgriBot stratifies risk: dosage / chemical / livestock medication mentions trigger mandatory refusal regardless of grounding score.
4. **No edge-device path.** T5-11B grounding model + Falcon 180B generator cannot run on consumer GPUs. AgriBot relies on a much cheaper signal stack: BM25/FAISS retrieval scores + cross-encoder rerank + LLM-self-grade, all CPU/GPU-feasible on 4 GB VRAM.
5. **No multimodal grounding.** Stolfo is text-only. AgriBot grounds visual outputs (rice-disease classification) against MobileNetV3+CBAM ONNX predictions with confidence thresholds.

---

## 5. Strengths to Acknowledge

- **Methodological rigour.** Defining grounding against **both** retrieved docs *and* the model's pre-training corpus is the right framing — most prior work conflates the two or ignores the latter. The manual-validation step (98% agreement on the critical *correct-but-ungrounded* slice) gives strong confidence in the headline numbers.
- **Actionable findings.** Two recommendations transfer directly: (i) prefer **beam search** when grounding matters more than diversity, (ii) prefer **instruction-tuned** variants of any base model. AgriBot already aligns with both (instruction-tuned Qwen2.5-1.5B-Instruct; deterministic decoding for safety-critical paths).
- **Clear scaling story.** Demonstrates that scaling alone does *not* solve hallucination — 180B-parameter Falcon still leaves ~25% of correct outputs ungrounded. Strong justification for AgriBot's "grounding must be architectural, not just bigger model" stance.
- **Generous failure-mode catalogue.** Tables 1 and 2 are concrete enough to cite as illustrative cases of the hallucination class AgriBot's policy must block.

---

## 6. BibTeX

```bibtex
@inproceedings{Stolfo2024Groundedness,
  author    = {Stolfo, Alessandro},
  title     = {Groundedness in Retrieval-augmented Long-form Generation:
               An Empirical Study},
  booktitle = {Findings of the Association for Computational Linguistics:
               NAACL 2024},
  year      = {2024},
  pages     = {1537--1552},
  month     = jun,
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.findings-naacl.100/},
  note      = {arXiv:2404.07060. ETH Zürich; work partially carried out at Oracle Labs.}
}
```

---

## 7. Final Verdict

> **INCLUDE — Tier 1 (high priority for motivation / Related Work).**
> The strongest, cleanest empirical evidence available that **RAG alone does not guarantee groundedness, and scaling does not fix it**. Directly motivates AgriBot's architectural choice to enforce grounding via an agent graph rather than rely on the LLM's intrinsic faithfulness. Single-author, peer-reviewed at a top NLP venue — high citation value.

**Positioning sentence for your paper:**
> "Stolfo (Findings of NAACL 2024) shows empirically that across four model families on three QA benchmarks, RAG-augmented LLMs blend correct answers with ungrounded prose, with **even the 180B-parameter Falcon leaving roughly 25% of correct-answer outputs unsupported by either retrieved documents or pre-training data**. AgriBot operationalises this diagnostic as a runtime guarantee: a LangGraph **evidence-grading node** rejects under-supported drafts before generation, triggers bounded LLM-rewritten retrieval retries, and — for the safety-critical agricultural domain where ungrounded dosage recommendations can harm farmers — enforces a configurable **grounding policy (refuse / disclaim / cited-facts-only)** backed by bilingual regex risk detection, none of which the measurement-only literature provides."
