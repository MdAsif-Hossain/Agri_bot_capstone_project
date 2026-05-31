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

### Item 3 — GPU latency (may be infeasible)
- Needs CUDA toolkit + `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall`. Windows machine, RTX 3050 4GB. Check `nvidia-smi` first. If infeasible/fails, capture EXACT error → feeds Item 16 limitation wording. Do NOT fabricate numbers.

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
