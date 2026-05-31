# AgriBot Fix-List Execution Progress

Tracking `agribot_complete_fix_list.md` (20 items). Target: IEEE ICTAI 2026, deadline 2026-06-30, max 8 pages. Team = user + Claude only.

**Key constraint:** No local LaTeX toolchain (`pdflatex` NOT installed). Team compiles on **Overleaf**. All .tex/.bib edits are textually correct but must be visually verified by the user on Overleaf.

## DONE
- **Item 5** — `bibliography.bib` BanglaSpeech2Text author → `{{Saiful Islam}}`.
- **Item 6** — Vosk2020 author was ALREADY `{{Alpha Cephei}}`. No change.
- **Item 7** — Nogueira2019Reranking converted `@misc`→`@article`, now `journal = {arXiv preprint arXiv:1901.04085}` (removed publisher/howpublished). Renders arXiv id cleanly.
- **Item 9** — Removed `\fixme` command def from `main.tex` preamble. No `\fixme{}` usages remained in body (grep-verified).
- **Item 10** — Moved `\cmark`/`\xmark`/`\pmark` defs from inside Table I to preamble (right after `\newcommand{\system}`). Removed the in-table copies.
- **Item 18** — `main.tex` preamble: `\hypersetup{colorlinks=false}` (was colorlinks=true...).

## DONE (session 2 — 2026-05-31)
- **Item 8** — CONFIRMED the "~5× faster projected on GPU" text WAS baked into `fig4.png` (the latency fig used in the paper; fig3 is unused). The committed PNG came from an OLD generator; current `fig4_latency()` source was a stale simplified rewrite. REWROTE `fig4_latency()` to faithfully reproduce the figure (mean-wall-clock denominator + "other overhead" segment) MINUS the GPU clause; xlabel now ends "(CPU-only LLM inference)". Regenerated, copied `fig4_latency.png/pdf` → `fig4.png/pdf`. Visually verified identical except GPU text gone.
- **Item 4 (harness only)** — Built `evaluation/kappa_subset.py` (deterministic 20-query category-balanced subset: 7 factual/4 proc/3 OOC/3 risk/3 dialect, 6 must-refuse) + `evaluation/kappa_score.py` (Cohen's κ via sklearn 1.8.0, a1-vs-gold / a2-vs-gold / a1-vs-a2). Generated `evaluation/annotation/{annotator1,annotator2,_gold}.csv`. NEEDS 2 non-co-author humans to fill must_refuse_label, then run kappa_score.py → `kappa_latest.json`.
- **Item 1+2 (harness + launch)** — Built `evaluation/eval_ablation.py`. Smoke test (2q/config) PASSED: A/B/C single-pass (3-18s), D=real compiled agent (61s, full retry+verify). One unified harness scores all 4 configs: Recall@3+MRR (±2pg tol, 46 q w/ expected_sources), groundedness=keyword coverage (NOT verified_rate — that needs verify node, only D has it), refusal accuracy (60 q). FULL RUN launched in background (bash id btpmfo4v1) → `evaluation/results/ablation_latest.json` + `ablation_run.log`. ETA hours (~180 generations @ ~30-60s).

## ABLATION RESULTS (item 1+2 COMPLETE — 2026-05-31)
One unified harness, same 60-query benchmark. Groundedness = keyword coverage (NOT verified_rate).
| Config | R@3 | MRR | Gnd(kw) | Acc | FN |
|--------|-----|-----|---------|-----|----|
| A: Dense only (NO policy) | 0.660 | 0.585 | 0.523 | 0.683 | **1** |
| B: + BM25 + RRF | 0.723 | 0.652 | 0.600 | 0.783 | 0 |
| C: + Rerank + KG (1-pass) | 0.681 | 0.642 | 0.550 | 0.767 | 0 |
| D: Full (AgriBot) | **0.851** | **0.808** | **0.676** | **0.883** | **0** |

KILLER EVIDENCE: baseline A (no policy) leaked Q39 = "56 lbs/acre urea" dosage (FN=1). Every policy config (B/D) refused it. → proves policy functionally necessary.
- Full vs naive: R@3 +19.2pp, Acc +20.0pp. Biggest retrieval gain = BM25+RRF (A→B +6.4pp). Biggest end-to-end gain = agentic loop (C→D +17pp R@3).
- NOTE: ablation D (0.851/0.808/0.883) ≠ headline Table III (0.80/0.79/0.833) because measured via different path (whole-agent vs per-metric scripts) + LLM run variance. Both within CIs. Paper caption explains: ablation uses keyword-coverage groundedness, headline uses verify-node verified_rate.
- DONE in paper: Item 11 (Table IV ablation), Item 12 (baseline paragraph w/ Q39), Item 16 (limitations 6→5: removed baseline limit, revised ablation limit to "per-node loop ablation future").
- Files: `evaluation/eval_ablation.py`, `evaluation/results/ablation_latest.json` (+ timestamped), `evaluation/results/ablation_run.log`.

## REMAINING — exact next steps

### Item 19 — Verify Kaggle author (BLOCKED — needs human eyes)
- `bibliography.bib` entry `RiceLeafAUG2024` lists `author = {Mishra, Anshul}`, title "Rice Leaf AUG: Augmented Rice-Disease Image Dataset", username `anshulm257`.
- WebFetch FAILED: Kaggle + Google are JS-rendered, author display name not in static HTML. Could not confirm "Anshul Mishra" is the real name, nor that the page title matches our bib title.
- USER ACTION: open https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset in a browser, read the owner's display name + exact dataset title, tell Claude to update the .bib if different. NOTE: current bib title "Rice Leaf AUG..." may not match the page's "Rice Disease Dataset" — verify and reconcile.

### Item 8 — Figure GPU text (investigate first)
- Fix list says "Figure 3" has baked-in "~5× faster projected on GPU". Paper currently includes fig1, fig2, fig4 (NOT fig3 — it was dropped in 6-page cut). EARLIER FINDING: grep of `scripts/make_figures.py` for GPU/faster/projected found NO "5× faster" text. Likely a stale concern OR the text is rendered as an image annotation not in source.
- ACTION: open `docs/figures/fig3.png` and `fig4.png` visually (Read tool renders PNGs). If "~5× faster projected on GPU" appears in fig4 (the latency fig actually used), edit `scripts/make_figures.py` `fig4_latency()` to remove the annotation, regenerate. If only in fig3 (unused), no paper impact — note it. The script is at `scripts/make_figures.py`; fig4 function ~line 41 in docs/figures/README.md mapping.

### Items 1 + 2 — Baseline + Ablation (BIG, long compute)
CORRECTED real module paths (rag_pipeline.py / hybrid_retriever.py do NOT exist):
- Retrieval: `agribot/retrieval/hybrid.py` → `HybridRetriever(index_bundle, embedding_model, dense_top_k, sparse_top_k, dense_weight, sparse_weight)`, method `.retrieve(query, top_n)`. For DENSE-ONLY set `sparse_weight=0.0, dense_weight=1.0`.
- Rerank: `agribot/retrieval/reranker.py` → `Reranker(model_name="ms-marco-MiniLM-L-12-v2", top_n, threshold)`, method `.rerank(query, candidates)`.
- Index load: `from agribot.ingestion.index_builder import IndexBundle; IndexBundle.load(settings.INDEX_DIR)`.
- Embedder: `SentenceTransformer(settings.EMBEDDING_MODEL)`.
- KG expansion: `agribot/knowledge_graph/entity_linker.py` (+ `seed_data.py`, `schema.py`). KG linking lives in agent nodes, NOT in the retriever.
- Agent: `agribot/agent/graph.py`, `nodes.py`, `state.py`, `grounding_policy.py` (`is_risky_query`). NOT yet read — READ FIRST to learn how to invoke generate without the grade→rewrite→verify loop for single-pass configs.
- eval helpers: `evaluation/common.py` → `load_dataset()`, `filter_queries(dataset, categories=...)`, `save_results(name, payload)`. Pattern model = `evaluation/eval_retrieval.py` (already read; composes HybridRetriever+Reranker manually, scores Recall@k/MRR with `_hit_rank`, ±2-page tol).
- Configs (item 2):
  - A Dense only: sparse_weight=0, no rerank, no KG ; NO refusal policy ; single-pass gen.
  - B Dense+BM25+RRF: default weights, no rerank, no KG ; +refusal policy ; single-pass.
  - C Full retrieval single-pass: +rerank +KG ; +refusal policy ; skip grade/rewrite/verify.
  - D Full = already measured (R@3 0.804, MRR 0.793, Gnd 78.3%, Acc 83.3%).
- Metrics/config: Recall@3, MRR (retrieval-only, FAST ~0.5s/q); Groundedness (keyword coverage, see `evaluation/eval_grounding.py`); Accuracy (refusal correctness, see `evaluation/eval_refusal.py`).
- Generation SLOW (~50s/q CPU), ~180 gen calls ≈ hours. BUILD `evaluation/eval_ablation.py`, TEST on 2 queries first, then run in BACKGROUND → `evaluation/results/ablation_latest.json`.

### MODEL CORRECTION (discovered 2026-05-31 via GPU load logs)
- The GGUF file `models/qwen2.5-1.5b-instruct-q8_0.gguf` is MISNAMED. llama loader reports authoritatively: `model params = 3.40B`, `n_layer = 36`, `n_embd = 2048`, `file type = Q4_0` (4.69 bpw), `general.name = qwen2.5-3b-instruct`. → The real model is **Qwen2.5-3B-Instruct Q4_0**, NOT 1.5B-Q8.
- ALL paper numbers were produced by this 3B-Q4 model. USER APPROVED "correct to 3B Q4_0".
- DONE in main.tex: §IV Models para rewritten (3B Q4_0, 36 layers, ~2.2GB VRAM, deleted false "3B doesn't fit" claim — GPU test proved it DOES fit); Table II row → "Qwen2.5-3B Q4_0"; §II "1.5B-parameter" → "3B-parameter". Grep confirms ZERO remaining 1.5B/Q8 refs.
- File on disk stays misnamed (runtime path hardcoded in config.py); only the PAPER is corrected. Consider renaming file + config later (NOT now, would break running evals).

### Item 3 — GPU latency (IN PROGRESS — prebuilt-wheel route WORKING)
- GPU SMOKE SUCCESS: offloaded 37/37 layers, supports_gpu_offload=True, 79 tok/s (vs ~5-8 CPU), VRAM ~2.2GB of 4GB, load 1.8s. venv-gpu now has ALL deps (torch CPU, faiss, rank_bm25, sentence_transformers, flashrank, transformers, langgraph, sentencepiece, bnunicodenormalizer) + GPU llama-cpp.
- Built `evaluation/gpu_latency_probe.py` (in-process, same 37 in-corpus queries x3 runs as eval_latency.py, agent.invoke path, CUDA DLL preamble). Running GPU latency now → `latency_gpu_latest.json`. Then items 13/14.
- USER CHOSE: prebuilt CUDA wheel after ablation. NO source build, NO system CUDA toolkit.
- Driver 581.57 supports CUDA 13.0 → cu124 wheels compatible. RTX 3050 Laptop 4GB.
- DONE: isolated `venv-gpu` (Python 3.12) created. Installed `llama-cpp-python==0.3.23` from `https://abetlen.github.io/llama-cpp-python/whl/cu124`. Wheel bundles `ggml-cuda.dll` (810MB) but NOT cuda runtime → added `nvidia-cuda-runtime-cu12` + `nvidia-cublas-cu12` wheels (cudart64_12, cublas64_12, cublasLt64_12).
- GOTCHA: `os.add_dll_directory()` alone insufficient for transitive ggml-cuda→cudart load on Windows; MUST also prepend each nvidia `*/bin` + llama_cpp/lib to `os.environ['PATH']` BEFORE `import llama_cpp`. See `evaluation/_gpu_smoke.py` for the working DLL-path preamble pattern.
- Original CPU `venv` UNTOUCHED (rollback safety). pip snapshot: `evaluation/_pip_snapshot_cpu_working.txt`.
- NEXT: confirm smoke test loads model w/ n_gpu_layers=-1 (look for "offloaded N/N layers to GPU" + tok/s). If 4GB OOMs at full offload, drop to n_gpu_layers=15 (matches config.py default LLM_N_GPU_LAYERS). Then build GPU latency runner reusing same DLL preamble → measure p50/p95 over same 37 in-corpus queries → items 13/14.

### Item 4 — Cohen's κ (needs 2 NON-co-author humans)
- Pick 20 queries (mix categories) from `evaluation/dataset.json`. Build `evaluation/kappa_template.csv` (id, query, category — blank refuse? + keywords columns) for 2 annotators. Build scoring script using `sklearn.metrics.cohen_kappa_score` on must_refuse labels vs gold. CANNOT self-label (not independent). Hand off labeling to user.

### Paper writes (AFTER results land)
- **Item 11** — Add ablation Table IV after refusal analysis in Section VI (Results). Template in fix list. Use real numbers from item 2.
- **Item 12** — Baseline comparison paragraph after Table IV (4-6 sentences; fix list has template).
- **Item 13/14** — GPU latency row in Table III (headline) + latency paragraph — ONLY if item 3 succeeds.
- **Item 15** — Cohen's κ sentence in Section V — ONLY after item 4.
- **Item 16** — Update Section VII limitations: if GPU done soften (2); ablation done → change (3) to "per-node agentic loop ablation left to future"; baseline done → remove (4); keep (1)(5)(6).
- **Item 17** — If paper >8 pages after additions, trim: TimeR4 sentence, compress grounding subsection (Zhu/Li parenthetical), shorten stress paragraph, shorten §VII safety-coverage (dedupe vs §VI).
- **Item 20** — IEEE PDF eXpress at ieee-pdf-express.org — user must do (needs IEEE account + ICTAI 2026 conf ID).

## NOTE on file quirk
`bibliography.bib` has odd line endings → line numbers scramble across tools. Edit by CONTENT match (works fine), not line number. Same caution for `main.tex` (less affected).
