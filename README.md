# AgriBot: A Production-Grade Offline Multimodal Agentic RAG System with Dialect Knowledge Graph and CPU-Bound Vision for Bilingual Agricultural Decision Support

**📚 Quick Navigation Links:**
- [🖼️ Visual System Workflow Diagram](README_WORKFLOW.md)
- [🏗️ Detailed Architecture](docs/ARCHITECTURE.md)
- [🤖 Local Models & Transformers](docs/MODELS.md)
- [📄 Data Sources & Ingestion](docs/DATA_SOURCES.md)
- [✅ Project Phases Breakdown](PROJECT_PHASES_COMPLETE_BREAKDOWN.txt)

---

## 1. Executive Summary
Farmers and extension officers in rural Bangladesh often lack timely access to reliable, evidence-based agricultural guidance due to unstable internet connectivity, limited expert availability, and a language gap between scientific manuals and colloquial/dialectal Bengali. Meanwhile, crucial knowledge exists in trusted resources (e.g., FAO/IRRI manuals and local extension documents), but these are typically PDF-heavy, difficult to search, and frequently contain complex layouts: tables, figures, scanned pages, and images carrying essential instructions.

This project proposes **AgriBot**, a **production-grade, offline-first** decision-support system that accepts **multimodal inputs**—**voice, text, and images**—and produces **bilingual outputs (Bengali + English)** with **traceable citations** (source + page). AgriBot uses **Retrieval-Augmented Generation (RAG)**, strengthened by three mandatory reliability pillars:

1. **Strict Hardware-Isolated Vision Architecture (CPU-Bound CNN + Heuristics)**
   To adhere to strict edge hardware limits (e.g., RTX 3050 4GB VRAM) and prevent memory-out errors, AgriBot deliberately avoids heavy Vision Language Models (VLMs). Instead, it uses a lightweight, CPU-bound Convolutional Neural Network (MobileNetV3 + CBAM attention module via ONNX) paired with deterministic color-math heuristics to extract disease keywords. The entire GPU is strictly reserved for the LLM inference engine. 

2. **Dialect Knowledge Graph (KG) beyond a Lexicon**  
   AgriBot maintains a versioned, offline **knowledge graph** (stored in **SQLite graph tables** for portability) mapping dialect terms to canonical Bengali/English/scientific entities, and representing relations such as symptom→disease and disease→treatment. The KG is used for entity linking, query expansion, and disambiguation.

3. **Agentic Context-Gathering and Self-Correction (LangGraph)**  
   To compensate for the lack of a generalized VLM, AgriBot relies on conversational intelligence. If the CNN detects a disease but lacks context (e.g., severity or crop stage), the LangGraph agent pauses the RAG loop to proactively ask the farmer follow-up questions. The loop ensures: **Retrieve → Grade → Rewrite/Expand → Verify → Respond (or Ask Follow-up)**.

AgriBot is designed for real deployment: a centralized machine runs the offline models and indexes via FastAPI, while a **multi-platform ecosystem (Web, Mobile App, and Desktop App)** connects over local Wi‑Fi/LAN—delivering high accessibility without forcing heavy on-device inference.

---

## 2. Problem Statement
AgriBot targets the following field-real constraints:

1. **Connectivity constraint:** Cloud AI is unreliable in rural environments; the system must function 100% offline via native Python libraries.
2. **Language + dialect constraint:** Farmers use colloquial/dialect Bengali; manuals use scientific terminology.
3. **Visual-first diagnosis without VLM Hallucination:** General VLMs often hallucinate or crash 4GB GPUs. We require deterministic, low-memory image classification combined with agentic follow-ups.
4. **PDF complexity and noise:** Manuals contain indexes, repeated headers/footers, tables, and scanned content.
5. **Safety and trust:** Agricultural recommendations can be harmful if incorrect; outputs must be evidence-based, cited, and conservative under uncertainty.
6. **Edge hardware constraint:** Strict CPU/GPU memory splitting is required to run LLM, ASR, TTS, and CNN simultaneously on consumer hardware.

---

## 3. Goals and Success Criteria
### 3.1 Primary Goals (Must Deliver)
1. **Offline-first operation** after initial setup using a purely native Python architecture (`llama-cpp-python`, `onnxruntime`).
2. **Multimodal input support**:
   - Voice (Bengali) via offline CPU-bound ASR (`faster-whisper`),
   - Text (BN/EN),
   - Image (farmer crop photo) via CPU-bound CNN and heuristic analysis.
3. **Bilingual outputs (BN + EN)** via `BanglaT5`.
4. **Multimodal PDF ingestion**:
   - Layout-aware text, table, and figure extraction.
   - OCR for scanned content.
5. **Hybrid retrieval** (BM25 + dense vectors) with Cross-Encoder reranking.
6. **Dialect Knowledge Graph** for entity linking and query expansion.
7. **LangGraph agentic self-correction loop** (bounded retries) + conversational follow-ups.
8. **Citations + evidence panel**: every answer includes source/page references.
9. **Production readiness**:
    - Thread-safe singletons and API concurrency control,
    - JSON logging and trace IDs,
    - Reproducible build artifacts.

### 3.2 Success Criteria (Measurable)
- **Groundedness:** ≥X% responses contain valid citations supporting key claims.
- **Refusal correctness:** system safely refuses or asks follow-up questions when evidence or visual context is insufficient.
- **Retrieval quality:** improved Recall@k/MRR with KG expansion vs without.
- **Edge feasibility:** 0 OOM (Out of Memory) crashes during concurrent multimodal use on a 4GB GPU due to strict CPU offloading of auxiliary models.

---

## 4. Target Users and Use Cases
### 4.1 Users
- Extension officers / NGO field staff (primary operators)
- Farmers (via assisted kiosk or mobile client interaction)

### 4.2 Typical Scenarios
1. **Image-Based Query with Agentic Fallback:** A farmer uploads a leaf image. The CPU CNN detects "Early Blight" (92%). Instead of hallucinating severity, the LangGraph Agent asks: *"I detect Early Blight. Are the spots small, or are the leaves wilting?"* The farmer answers, and the agent retrieves the exact dosage for that severity.
2. **Voice-based query:** A farmer speaks Bengali symptoms; the ASR confidence gate triggers. If confidence is high, it processes the query; if low, it asks for user confirmation via the UI.
3. **Dosage query:** User asks for chemical dosage; the system strictly searches the RAG vector space and refuses to guess if the dosage is missing from the manual.

---

## 5. System Overview
AgriBot combines:
- **Offline ASR/TTS** (CPU) for voice interaction,
- **CPU-Bound Vision** (MobileNetV3 + CBAM ONNX classifier + OpenCV color heuristics) generating contextual keywords,
- **Hybrid retrieval** (CPU-based FAISS + BM25 + ms-marco reranker),
- **Dialect Knowledge Graph** (SQLite) for term alignment,
- **LangGraph bounded self-correct loop** with evidence grading and verification,
- **Offline local LLM** (Qwen 2.5 1.5B via `llama-cpp-python`) running strictly on the GPU,
- **Multi-platform ecosystem** (Web Kiosk, Mobile, Desktop) communicating with a unified FastAPI orchestrator.

---

## 6. Architecture (Production View)
### 6.1 Core Modules / Services
1. **Ingestion Pipeline (Offline Build):** Curates PDFs into a chunk store, indexes, and a versioned KG snapshot.
2. **Orchestrator API (`api.py`):** FastAPI exposing concurrent-safe endpoints (`/v1/chat`, `/v1/chat/image`, `/v1/chat/voice`).
3. **Retrieval Service:** Hybrid retrieval + reranking returning structured Markdown evidence.
4. **Knowledge Graph Service:** Entity linking, alias expansion, and dialect-to-science mapping.
5. **ASR/TTS Service:** Whisper ASR with confidence gating + offline Bengali TTS.
6. **Vision Service (`classifier.py`):** ONNX-based PyTorch model for top-K disease classification.
7. **LLM Engine (`engine.py`):** Singleton `llama-cpp-python` instance with thread locks to guarantee GPU stability.

### 6.2 Client Ecosystem (Multi-Platform)
- **Web App:** React/Vite kiosk UI for desktop/laptop (Glassmorphism design).
- **Mobile App:** Native client tailored for farmers in the field (optimized for camera and voice capture).
- **Desktop App:** Standalone application for agricultural researchers requiring heavy offline deep dives.
- **Diagnostic Drawer:** transparently rendering LLM verification reasoning, citations, and node timings across all clients.

---

## 7. Technology-to-Task Mapping (Strict Hardware Allocation)
### 7.1 Execution Modes
- **Offline Runtime Mode (Primary):** All models run locally. Internet is permanently disconnected.

### 7.2 Hardware Splitting Strategy
| Task | Technology | Hardware Target | Output |
|---|---|---|---|
| Text Generation (RAG) | **llama-cpp-python (Qwen 1.5B)** | **GPU (VRAM)** | BN+EN answer + citations |
| Voice → text | **faster-whisper** | **CPU (RAM)** | BN transcript + confidence |
| Image Classification | **MobileNetV3 + CBAM (ONNX Runtime)** | **CPU (RAM)** | Top-K Disease classes / Keywords |
| Symptom Extraction | **PIL/NumPy Color Heuristics** | **CPU (RAM)** | Symptom hints (e.g. "Yellowing") |
| Translation | **BanglaT5** | **CPU (RAM)** | EN/BN translations |
| Dense retrieval | **FAISS + sentence-transformers** | **CPU (RAM)** | Semantic evidence candidates |
| Reranking | **CrossEncoder** | **CPU (RAM)** | Precision-ranked evidence |

---

## 8. Multimodal PDF Filtering and Evidence Quality
AgriBot prevents retrieval pollution by:
- Using layout-aware parsers to extract structured content.
- Removing repeated header/footer noise mathematically.
- Downweighting TOC/index/reference pages during ingestion.
- Treating tables as first-class Markdown evidence chunks to preserve structured relationships.

---

## 9. Dialect Knowledge Graph (Beyond Lexicon)
AgriBot’s KG includes:
- Aliasing and canonical mapping (colloquial dialect → canonical scientific terminology).
- Graph relations connecting symptoms, diseases, pests, and treatments.
The KG materially improves retrieval recall for colloquial queries and provides strong disambiguation during agentic correction.

---

## 10. LangGraph Agentic Self-Correction and Safety
### 10.1 Bounded Self-Correct Loop
**Normalize → KG Link → Vision/Heuristic Fusion → Expand → Retrieve → Grade → (Rewrite/Follow-up) → Generate → Verify → Respond**

### 10.2 Conversational Fallback Rule
Because the system lacks a generalized VLM to perceive arbitrary image contexts, the LangGraph agent is explicitly prompted to **pause and ask the user for visual context** (e.g., "Is the damage on the upper or lower leaves?") if the classification confidence is below threshold or evidence is insufficient.

---

## 11. Production Readiness and Maintainability
AgriBot is engineered for:
- **Strict Concurrency Control:** `asyncio.Semaphore` implementations in `api.py` prevent simultaneous LLM/Vision calls from crashing the system under load.
- Reproducible builds (versioned `.db` and `faiss` artifacts).
- Automated structured JSON logging of trace IDs and node latencies.

---

## 12. Planned Research Publications (Paper Title Portfolio)
These titles maximize competitiveness by emphasizing generalizable CS contributions (agentic fallback, edge hardware optimization, KG-augmented low-resource retrieval, and pure-offline architectures).

**A. Edge AI & Hardware Optimization**
1. “Strict VRAM Isolation for Agentic RAG: Concurrent Multimodal Inference on 4GB Edge Devices”
2. “Agentic Conversational Fallback as a Substitute for VLMs in Low-Resource Edge Deployments”

**B. Low-Resource Language + Knowledge Graph Augmentation**
3. “Dialect-to-Science Alignment via Knowledge Graph-Augmented Retrieval: Robust Evidence Search for Colloquial Bengali Queries”
4. “Graph-Guided Query Expansion for Low-Resource Bilingual RAG: Entity Linking and Provenance-Aware Evidence Selection”

**C. Trustworthy Agentic RAG**
5. “Self-Correcting RAG with Bounded Retrieval Loops: Evidence Grading, Query Rewriting, and Conservative Refusal Offline”
6. “Trustworthy Offline Decision Support with Verified RAG: Transparent Evidence, Safety Policies, and Bounded Self-Correction”

**D. Speech & Vision in Agriculture**
7. “Heuristic-Assisted CNNs for Agricultural Diagnostics: Mitigating Vision Hallucinations on CPU-Bound Edge Systems”
8. “Bilingual Voice Interfaces for Verified RAG: Measuring ASR Error Propagation and Evidence Robustness in Low-Resource Settings”

- [agribot/ingestion](agribot/ingestion)
- [agribot/retrieval](agribot/retrieval)
- [agribot/knowledge_graph](agribot/knowledge_graph)
- [agribot/llm](agribot/llm)
- [agribot/translation](agribot/translation)
- [agribot/voice](agribot/voice)
- [agribot/vision](agribot/vision)
- [frontend](frontend)

## 6. API Endpoints (v1)

Defined in [api.py](api.py):
- GET /v1/health
- POST /v1/chat
- POST /v1/chat/voice
- POST /v1/chat/image
- POST /v1/tts
- GET /v1/kg/stats
- GET /v1/kg/search

## 7. Prerequisites

- Python 3.11+
- Node.js 20+
- ffmpeg available in PATH
- OS Bengali voice pack required for Bangla TTS playback

## 8. Setup and Run

### Environment setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### Ingestion (required before first backend run if indexes are missing)
```bash
python ingest.py
```

### Build frontend
```bash
cd frontend
npm run build
cd ..
```

### Run backend
```bash
python api.py
```

### Makefile commands
Available in [Makefile](Makefile):
- make setup
- make install
- make ingest
- make build-frontend
- make run
- make run-dev
- make test

## 9. Voice Output Notes

- English TTS works with default system voices.
- Bangla TTS requires Bengali voice installation at OS level.
- If Bengali voice is missing, API returns explicit error instead of hanging.

## 10. Testing

Run all tests:
```bash
pytest tests/ -v
```

Focused smoke set:
```bash
pytest tests/test_voice.py tests/test_api.py -q
```

## 11. Recommended Next Milestones

1. Reintroduce evaluation harness with reproducible score reports
2. Reintroduce containerized deployment path
3. Add production observability (metrics, tracing, alerts)
4. Harden security controls and API governance
5. Finalize production vision model path
6. Publish benchmark and ablation results for scholarship/research dossiers

## 12. Related Documents

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/MODELS.md](docs/MODELS.md)
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)
- [PROJECT_PHASES_COMPLETE_BREAKDOWN.txt](PROJECT_PHASES_COMPLETE_BREAKDOWN.txt)

Note: [docs/EVALUATION.md](docs/EVALUATION.md) currently describes planned evaluation capabilities that are not fully present in this repository state.
