# AgriBot — Full Project Review

> A plain-language audit of what this project is, what's strong about it, what's missing, and concrete prioritized suggestions.

---

## 1. What this project is (in one paragraph)

**AgriBot** is an **offline-first, multimodal RAG assistant for Bangladeshi farmers**. It takes **voice (Bengali), text (BN/EN), or images** as input and returns **bilingual, cited answers** drawn from trusted PDFs (FAO/IRRI/UNICEF manuals). It runs entirely on local hardware (designed around an RTX 3050 4GB), with a deliberate **CPU/GPU split**: the LLM (Qwen 2.5 1.5B GGUF) gets the GPU, while ASR, vision, translation, retrieval, and reranking stay on CPU. A **LangGraph agent** wraps retrieval in a self-correcting loop (retrieve → grade → rewrite → verify → respond), and a **dialect knowledge graph** in SQLite maps colloquial Bengali to canonical scientific terms. Stack: FastAPI backend (~5,000 LOC in [agribot/](agribot/)), React 19 + Vite + Tailwind frontend, ~1,700 LOC of tests.

---

## 2. Potential — what this project has going for it

### 2.1 Strong architectural thinking
- The **strict CPU/GPU isolation** strategy is a genuinely thoughtful answer to a real edge-hardware constraint. Most student/hobby projects ignore VRAM; this one designs around it.
- The **agentic conversational fallback** (asking the user follow-up questions instead of hallucinating with a VLM) is a clever substitute for heavy vision-language models the hardware can't run.
- Hybrid retrieval (FAISS dense + BM25 sparse + ms-marco cross-encoder rerank) is genuinely best-practice RAG.

### 2.2 Real-world relevance
- **Bilingual + dialect-aware + offline-first** is a rare combination. Bangladeshi rural deployment is a well-defined market with weak existing solutions. The differentiation is real.
- Domain choice (agriculture extension) is socially valuable and easy to justify in research contexts.

### 2.3 Solid engineering hygiene
- [config.py](config.py) — centralized pydantic-settings config, env-driven, single source of truth. Clean.
- [api.py:60-62](api.py#L60-L62) — `asyncio.Semaphore` per model lane (LLM=1, STT=2, image=2). Right idea for protecting a single-GPU host.
- Structlog JSON logging, singleton model patterns, trace-friendly design.
- Multi-backend ASR fallback chain (`faster-whisper` → `BanglaSpeech2Text` → `Vosk`) shows pragmatism for low-resource language.

### 2.4 Research-friendly framing
- README already maps out **8 candidate paper titles** across edge-AI, KG-augmented retrieval, agentic RAG, and speech/vision. That's a real publication portfolio if the evaluation gets finished.
- Architecture docs exist: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/MODELS.md](docs/MODELS.md), [docs/PIPELINES_DEEP_DIVE.md](docs/PIPELINES_DEEP_DIVE.md), [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

### 2.5 Decent test coverage
- 11 test files, ~1,700 LOC. Includes agent, retrieval, KG, ingestion, voice, grounding policy, and audio preprocessing tests.

---

## 3. What it's lacking (honest gaps)

### 3.1 The biggest gap: vision pipeline is not wired up
- The ONNX inference path in [agribot/vision/classifier.py:80-177](agribot/vision/classifier.py#L80-L177) is actually fully implemented — but the misleading "Stub implementation" docstring suggests otherwise.
- `IMAGE_CLASSIFIER_ENABLED` defaults to **`False`** in [config.py:81](config.py#L81). The vision feature the README headlines is **off by default**.
- The trained CBAM CNN lives only in a notebook ([cnn/capstone-image-classifier.ipynb](cnn/capstone-image-classifier.ipynb)). The `.onnx` artifact and labels file have never been exported into `models/`.
- [TODO.md](TODO.md) confirms this as Phase 11 outstanding work.
- **Impact:** The README's image-query examples and the "Strict Hardware-Isolated Vision Architecture" pillar are aspirational, not real, in the current repo.

### 3.2 No real evaluation numbers
- README success criteria literally read "**≥X%** responses contain valid citations" — `X` is a placeholder.
- [docs/EVALUATION.md](docs/EVALUATION.md) is noted in the README as "planned, not fully present."
- No ground-truth query set, no Recall@k/MRR/latency numbers, no documented OOM stress-test results.
- **Impact:** Hardest gap to ignore. For a project that wants 8 papers, missing evaluation is the single biggest blocker. Reviewers will reject "we believe it works" without numbers.

### 3.3 No deployment artifacts
- No `Dockerfile`, no `docker-compose.yml`, no deployment runbook (TODO confirms).
- README install path is `python api.py`. For a system that wants to be installed on "fresh offline laptops by field officers," this is not production-grade.

### 3.4 Documentation overpromises vs. reality
- README presents Web + Mobile + Desktop as a "multi-platform ecosystem." Only the **web** app exists. No mobile, no desktop in repo.
- README says vision pipeline is core; in fact it's disabled by default and missing artifacts.
- Reader's first impression won't survive a `git clone`.

### 3.5 Small corpus, small KG
- Only **5 PDFs** in [data/pdfs/](data/pdfs/). Any "retrieval recall" claim is limited by this.
- [agribot/knowledge_graph/seed_data.py](agribot/knowledge_graph/seed_data.py) is 233 hand-written lines. Fine as a seed, but the KG won't scale to "real dialect coverage" by hand.

### 3.6 Security / hardening
- `API_KEY` defaults to empty string — auth disabled by default.
- No rate limiting (TODO flags it).
- No request-size limits beyond per-file MB caps.
- CORS hardcoded to `localhost:5173,localhost:8000` — not deployment-configurable.

### 3.7 Repo hygiene issues
- **`venv/` is committed** — site-packages files are showing up in the repo. This will bloat clones to multi-GB and break on every other developer's machine.
- `BanglaSpeech2Text.log`, `test.wav`, `tts_en_test.wav` are artifacts that don't belong in source control.
- The 1.5GB GGUF model `models/qwen2.5-1.5b-instruct-q8_0.gguf` shouldn't be in git either — use a download script or git-lfs.
- [pyproject.toml](pyproject.toml) is 40 bytes (effectively empty). No build metadata, no dependency declaration in modern form.

### 3.8 Code structure smells
- [agribot/voice/stt.py](agribot/voice/stt.py) is **967 lines** — by far the largest file. Three ASR backends + fallback logic + audio handling all in one module. Hard to test, hard to change one backend without risking the others.
- Voice deps overlap: `faster-whisper`, `vosk`, `BanglaSpeech2Text`, `SpeechRecognition` — four ASR systems. Each pulls its own deps; risk of version conflicts.

### 3.9 No CI/CD visible
- `.github/` directory exists but no mention of automated test runs. For "production-grade" claims, PR-time `pytest` is the minimum bar.

### 3.10 Frontend is thin
- Only `react` and `react-dom` in dependencies. No router, no API client (axios/tanstack-query), no UI library, no state management.
- README claims "Glassmorphism design" — possible, but the dep list suggests a small handcrafted app, not a polished kiosk.

---

## 4. Suggestions (prioritized)

### Top priority — close the credibility gap before showing this to anyone external
1. **Finish the vision pipeline.** Export the CBAM CNN from the notebook to ONNX, drop the `.onnx` + `.txt` into `models/vision/`, flip `IMAGE_CLASSIFIER_ENABLED=True` by default. Without this, the README is misleading.
2. **Build the evaluation harness.** Hand-curate 50–100 colloquial Bengali queries with ground-truth manual+page targets. Run them. Fill in real Recall@k, MRR, citation-grounding, refusal-correctness numbers in [docs/EVALUATION.md](docs/EVALUATION.md). This is the single highest-leverage thing you can do for both research papers and FYP grading.
3. **Sync the README with reality.** Mark each feature ✅ Implemented / 🚧 Partial / 📋 Planned. Move mobile/desktop into a roadmap section. Honest > impressive.
4. **Remove `venv/` from git.** Add it to `.gitignore`, do `git rm -r --cached venv/`. Same for `.log`, `.wav`, and large model files (move models to a download script or git-lfs).
5. **Add a Dockerfile + docker-compose.** A one-command bring-up. This is what "field officer installs on a laptop" actually requires.

### Medium priority — production-readiness polish
6. **Enforce API auth + rate limit.** `API_KEY` middleware is already configured; just require it in non-dev mode. Add `slowapi` for per-IP rate limiting.
7. **Add GitHub Actions CI** — at minimum, `pytest tests/` on every PR. You already have 1,700 lines of tests; let them protect you.
8. **Split [agribot/voice/stt.py](agribot/voice/stt.py)** into `voice/backends/whisper.py`, `voice/backends/banglaspeech2text.py`, `voice/backends/vosk.py`, with a thin coordinator. Easier to test, easier to swap a backend.
9. **Expand the corpus.** 5 PDFs is fine for a demo but won't support real recall claims. Aim for 25–50 documents across crops/diseases/pests.
10. **Document concurrency limits with proof.** Run the stress test that proves the "0 OOM on 4GB VRAM" claim and put the data in EVALUATION.md.

### Lower priority — research and polish
11. **Add an ablation script** (KG-on vs KG-off, rerank-on vs off, agentic-loop vs single-pass). This is paper gold — directly fills Section 3.1 of the README's success criteria.
12. **Add a metrics endpoint** (`/v1/metrics`) exposing latency histograms, semaphore wait times, retrieval cache stats. Cheap and very useful for the "production observability" claim.
13. **Pin dependencies** (`pip freeze > requirements.lock.txt`) so reproducibility is real.
14. **Replace `pyttsx3`** with a proper offline TTS (e.g., `piper-tts`) so Bangla voice quality doesn't depend on the OS having the right voice pack.
15. **Trim overlapping voice deps** — pick one fallback strategy and remove the unused `SpeechRecognition` library if it isn't actually exercised.

---

## 5. Honest summary

**The vision is excellent.** The architecture is the work of someone who has actually thought hard about edge constraints, low-resource languages, and trustworthy RAG. The 8 paper-title portfolio is genuinely fundable.

**The execution is ~70% there.** The retrieval/agent/KG/voice stack is real and works. The vision stack is mostly aspirational. The evaluation story is missing entirely. The deployment story is missing entirely. The README oversells what the repo currently delivers.

**The two unlocks** that turn this from "impressive FYP" into "publishable + deployable" are:
1. Finish the vision pipeline (1–2 weeks of focused work, the notebook already exists).
2. Run a real evaluation and publish the numbers (1 week including dataset curation).

Everything else on the suggestions list is polish. Those two are the gap between the project the README describes and the project the code currently is.

---

## 6. Note on the vision "stub"

On re-reading [agribot/vision/classifier.py:80-177](agribot/vision/classifier.py#L80-L177), the ONNX inference path is **fully implemented** — preprocessing, ImageNet normalization, softmax, top-K filtering, label loading from a companion `.txt`. The "Stub implementation" comment in the `_load_model` docstring is misleading. What's actually missing is just:
- the trained `.onnx` artifact,
- the companion `labels.txt`,
- two config flags flipped (`IMAGE_CLASSIFIER_ENABLED=True`, `IMAGE_CLASSIFIER_MODEL_PATH=...`).

The notebook [cnn/capstone-image-classifier.ipynb](cnn/capstone-image-classifier.ipynb) already defines `ONNX_PATH = "{OUTPUT_DIR}/rice_disease_mobilenetv3_cbam.onnx"` and uses `class_names = full_dataset.classes`. So the export plumbing exists; it just needs to be run end-to-end and the outputs dropped into the project. **This is the next planned step (CNN integration).**
