# AgriBot Paper — Everything That Needs Fixing

**Target:** IEEE ICTAI 2026 | Deadline: June 30, 2026 | Max 8 pages
**Submit at:** https://easychair.org/conferences/?conf=ictai2026

---

## 1. Run a Naive RAG Baseline on Your 60 Queries

**Why:** Without this, reviewers have nothing to compare your numbers against. Your Recall@3 = 0.804 means nothing if a simpler system gets 0.78.

**How:**
Create a stripped-down pipeline that does only this:
- normalize the query (keep language detection)
- translate Bengali → English if needed (keep this, it's preprocessing not a contribution)
- FAISS dense retrieval only — disable BM25 and RRF in your hybrid retriever
- no KG expansion (skip kg_link node)
- no cross-encoder reranking (skip rerank node)
- direct Qwen2.5 generation from top-3 retrieved chunks
- no grade → rewrite retry loop
- no verify step
- no enforce_policy (no refusal)

Run all 60 benchmark queries through this. Record Recall@3, MRR, groundedness (keyword coverage), and accuracy (refusal correctness — the baseline will score 0% refusal recall since it has no refusal policy, meaning all 14 dosage queries get answered unsafely).

Put results in a table:

```
Baseline Results:
  Recall@3:      ___
  MRR:           ___
  Groundedness:  ___
  Accuracy:      ___
```

---

## 2. Run a 4-Row Ablation Study

**Why:** You have 4 novel components (KG, BM25+RRF, cross-encoder rerank, agentic retry loop). Reviewers need to know which ones matter. Right now you cannot prove any of them help.

**How:**
Run your 60 queries under 4 configurations:

**Config A — Dense only (same as baseline above)**

**Config B — Dense + BM25 + RRF**
- Enable BM25 sparse retrieval alongside FAISS
- Enable RRF fusion (your default k_rrf=60, 0.6/0.4 weights)
- No cross-encoder reranking
- No KG expansion
- Single-pass generation (no grade/rewrite loop)
- Include refusal policy (co-occurrence regex)

**Config C — Full retrieval, single-pass generation**
- Everything in retrieval enabled: FAISS + BM25 + RRF + cross-encoder rerank + KG expansion
- Single-pass generation: skip grade → rewrite retry loop
- Skip verify step
- Include refusal policy

**Config D — Full proposed system (already measured)**
- Your current numbers: Recall@3 = 0.804, MRR = 0.793, etc.

For each config record: Recall@3, MRR, Groundedness, Accuracy.

The final table:

```
| Configuration                    | Recall@3 | MRR  | Ground. | Accuracy |
|----------------------------------|----------|------|---------|----------|
| A: Dense only                    |          |      |         |          |
| B: Dense + BM25 + RRF           |          |      |         |          |
| C: Full retrieval, single-pass   |          |      |         |          |
| D: Full proposed (AgriBot)       | 0.804    | 0.793| 78.3%   | 83.3%    |
```

---

## 3. Measure GPU Latency

**Why:** Your title says "edge hardware" and your paper claims 4GB VRAM deployment. But all your latency numbers are CPU-only (p50 = 53 seconds). A 53-second response is not conversational. Reviewers will say your core claim is unverified.

**How:**

```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Recompile llama-cpp-python with GPU support
pip uninstall llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Verify GPU is being used
python -c "from llama_cpp import Llama; m = Llama('path/to/qwen2.5-1.5b-instruct-q8_0.gguf', n_gpu_layers=-1); print('GPU loaded')"

# Re-run your latency benchmark (same 37 in-corpus queries × 3 runs)
python eval/run_latency_eval.py
```

Record GPU p50 and p95 latency. Expected: p50 drops from 53s to roughly 10–15s.

If CUDA installation fails (driver issues, toolkit incompatibility), document the exact error. In the paper, replace "immediate future work" with the specific technical reason.

---

## 4. Get Inter-Annotator Agreement (Cohen's κ)

**Why:** Your benchmark was authored by one team member after inspecting the corpus. This is circular. A κ score from independent annotators proves your labels are not arbitrary.

**How:**
1. Pick 20 queries from your 60 (mix all categories: factual, procedural, out-of-corpus, risk, dialect-bn)
2. Give them to 2 people who are NOT co-authors — classmates, friends, anyone literate
3. For each query, they independently decide:
   - Should the system refuse this? (yes / no)
   - What keywords should appear in a correct answer? (3–5 keywords)
4. Compare their labels to your gold labels
5. Compute Cohen's κ:

```python
from sklearn.metrics import cohen_kappa_score

# For the must_refuse label (binary)
your_labels   = [0, 0, 1, 0, 1, ...]  # your gold labels for 20 queries
their_labels  = [0, 0, 1, 1, 1, ...]  # annotator's labels

kappa = cohen_kappa_score(your_labels, their_labels)
print(f"Cohen's κ: {kappa:.2f}")
# κ > 0.8 = excellent, 0.6–0.8 = substantial, 0.4–0.6 = moderate
```

---

## 5. Fix Reference [27] Author Name

**Why:** Prints as "S. I. Saiful" in the PDF. The author is Saiful Islam.

**How:** In bibliography.bib, find the BanglaSpeech2Text entry and change:

```bibtex
% WRONG:
author = {Saiful, Saiful Islam},

% CORRECT:
author = {{Saiful Islam}},
```

Double braces tell BibTeX to treat the whole string as one name unit.

---

## 6. Fix Reference [28] Author Name

**Why:** Prints as "A. Cephei". Alpha Cephei is the organization, not a person.

**How:** In bibliography.bib, find the Vosk2020 entry and change:

```bibtex
% WRONG:
author = {Alpha Cephei},

% CORRECT:
author = {{Alpha Cephei}},
```

---

## 7. Fix Reference [36] — Add arXiv Number

**Why:** The Nogueira reranking reference prints with nothing after "2019". Looks incomplete.

**How:** In bibliography.bib, find Nogueira2019Reranking and add:

```bibtex
note = {arXiv:1901.04085}
```

---

## 8. Remove "~5× faster projected on GPU" from Figure 3

**Why:** The bar chart image itself contains "~5× faster projected on GPU". You never measured GPU latency, so this is unverified. If you completed item 3, replace with actual numbers. If not, remove entirely.

**How:**
- Open the Python/matplotlib script that generates Figure 3
- Remove or update the "~5× faster projected on GPU" text annotation
- If GPU was measured: replace with actual speedup
- If GPU was not measured: remove the text, change x-axis label to just show CPU
- Regenerate the figure PNG/PDF and replace in LaTeX
- Update figure caption accordingly

---

## 9. Remove \fixme Command from Preamble

**Why:** Development tool. If any `\fixme{text}` calls remain, they render as red bold text. Looks unfinished.

**How:** Delete this line from the preamble:

```latex
\newcommand{\fixme}[1]{\textcolor{red}{\textbf{[FIXME: #1]}}}
```

Search entire .tex file for `\fixme{` and remove any instances.

---

## 10. Move \cmark / \xmark / \pmark to Preamble

**Why:** Currently defined inside the Table I environment. Will cause "already defined" errors if reused.

**How:** Cut these from inside the table:

```latex
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
\newcommand{\pmark}{$\sim$}
```

Paste into preamble (before `\begin{document}`).

---

## 11. Add Ablation Table (Table IV) to Section VI

**Why:** This is where your ablation results from item 2 go into the paper.

**How:** After the refusal correctness analysis in Section VI, insert:

```latex
\begin{table}[t]
\caption{Ablation study on retrieval and agentic components.
All configurations evaluated on the same 60-query benchmark.}
\label{tab:ablation}
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Configuration} & \textbf{R@3} & \textbf{MRR} & \textbf{Gnd.} & \textbf{Acc.} \\
\hline
Dense only              & X.XX & X.XX & XX.X\% & XX.X\% \\
+ BM25 + RRF            & X.XX & X.XX & XX.X\% & XX.X\% \\
+ Rerank + KG (1-pass)  & X.XX & X.XX & XX.X\% & XX.X\% \\
Full proposed (AgriBot)  & 0.804 & 0.793 & 78.3\% & 83.3\% \\
\hline
\end{tabular}
\end{table}
```

Replace X.XX with actual numbers.

---

## 12. Add Baseline Comparison Paragraph After Table IV

**Why:** Explains the ablation results in prose.

**How:** Write 4–6 sentences after Table IV:

"To isolate the contribution of the proposed pipeline, we compare against a naive RAG baseline consisting of dense FAISS retrieval, direct generation, and no refusal policy (first row of Table IV). The full system improves Recall@3 by +[X.X] pp and accuracy by +[X.X] pp over the baseline. The largest single-component gain comes from [name the component with biggest jump between rows]. Critically, the baseline answers all 14 dosage queries without refusal, demonstrating that the safety policy is functionally necessary, not merely conservative."

---

## 13. Update Table III with GPU Latency Row

**Why:** If you completed item 3.

**How:** Add to Table III:

```
Latency (GPU)    p50 / p95 wall    XX.X / XX.X s
```

---

## 14. Update Latency Paragraph in Section VI

**Why:** Current text says CPU-only. Add GPU results if measured.

**How:** After the existing CPU latency numbers, add:

"With GPU offload on the RTX 3050 (4 GB VRAM), the same benchmark yields p50 = [X.X] s and p95 = [X.X] s, a [Y.Y]× speedup over CPU-only inference."

---

## 15. Add Cohen's κ Sentence to Section V

**Why:** If you completed item 4.

**How:** In Section V, after the methodological limits paragraph, add:

"To partially address limit (iii), two independent annotators labelled a 20-query subset; inter-annotator agreement on the must\_refuse label yielded Cohen's $\kappa = [X.XX]$, indicating [excellent/substantial/moderate] agreement."

---

## 16. Update Limitations in Section VII

**Why:** Completed experiments remove some limitations.

**How:**
- If GPU measured: soften limitation (2) to "GPU latency was measured on a single RTX 3050; performance on other edge GPUs remains untested."
- If ablation done: change limitation (3) to "Ablation covers retrieval-side components; per-node agentic loop ablation is left to future work."
- If baseline done: remove limitation (4) entirely.
- Keep limitations (1) self-authored benchmark, (5) narrow corpus, (6) no user study.

---

## 17. Trim Related Work If Paper Exceeds 8 Pages

**Why:** Adding items 11–15 adds ~0.5–0.75 pages. You may need to cut elsewhere.

**What to cut (in order of least loss):**

1. **Remove TimeR4 [12] discussion** — temporal KG QA is tangential. Cut 1–2 sentences. Saves ~3 lines.

2. **Compress grounding subsection** — Keep Stolfo [14] as the key citation. Reduce Zhu [15] and Li [16] to a parenthetical: "Other approaches include dual-decoder architectures [15] and knowledge-representation surveys [16]." Saves ~4 lines.

3. **Shorten stress test paragraph** — Cut to: "A 20-request concurrent stress test at concurrency 5 completed with zero crashes and zero OOM events; the rate-limiter correctly shed excess traffic." Saves ~5 lines.

4. **Shorten safety–coverage trade-off** — Section VII has a detailed two-iteration regex story that largely repeats Section VI. Compress the Section VII version. Saves ~4 lines.

---

## 18. Set hyperref to colorlinks=false

**Why:** Some IEEE submission systems break colored links.

**How:** In preamble:
```latex
\hypersetup{colorlinks=false}
```

---

## 19. Verify Kaggle Dataset Author Name

**Why:** Reference [39] lists "Mishra, Anshul" from the username `anshulm257`. May not be the real name.

**How:** Check https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset for the displayed author name. Update .bib if different.

---

## 20. Run IEEE PDF eXpress

**Why:** IEEE requires format compliance. PDF eXpress catches violations.

**How:**
1. Go to https://ieee-pdf-express.org/
2. Create account with ICTAI 2026 conference ID
3. Upload compiled PDF
4. Fix any violations
5. Download compliant PDF
6. Submit that PDF to EasyChair

---

## PRIORITY ORDER

| Priority | Items | Impact |
|---|---|---|
| 1 | 1–2 (baseline + ablation) | Without these, rejection is almost certain |
| 2 | 3 (GPU latency) | Without this, "edge hardware" is unverified |
| 3 | 5–7 (.bib fixes) | Visible errors that signal carelessness |
| 4 | 8 (Figure 3 GPU projection) | Contradiction between figure and text |
| 5 | 11–14 (write ablation table + baseline + GPU into paper) | Puts results into the paper |
| 6 | 9–10 (\fixme, symbol defs) | Cleanup |
| 7 | 4 (inter-annotator κ) | Strengthens eval, not strictly required |
| 8 | 16–17 (update limitations, trim) | Necessary after adding content |
| 9 | 15, 18–20 (polish) | Final touches |
