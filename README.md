# Project Proposal (Final Year Project)

## Title
**AgriBot: A Production-Grade Offline Multimodal Agentic RAG System with Dialect Knowledge Graph for Bilingual (Bengali–English) Agricultural Decision Support**

---

## 1. Executive Summary
Farmers and extension officers in rural Bangladesh often lack timely access to reliable, evidence-based agricultural guidance due to unstable internet connectivity, limited expert availability, and a language gap between scientific manuals and colloquial/dialectal Bengali. Meanwhile, crucial knowledge exists in trusted resources (e.g., FAO/IRRI manuals and local extension documents), but these are typically PDF-heavy, difficult to search, and frequently contain complex layouts: tables, figures, scanned pages, and images carrying essential instructions.

This project proposes **AgriBot**, a **production-grade, offline-first** decision-support system that accepts **multimodal inputs**—**voice, text, and images**—and produces **bilingual outputs (Bengali + English)** with **traceable citations** (source + page). AgriBot uses **Retrieval-Augmented Generation (RAG)**, strengthened by three mandatory reliability pillars:

1. **Multimodal Document Understanding for PDFs**  
   Instead of brittle hand-written heuristics, AgriBot will use **Marker** or **Surya** (layout-aware document extraction tools) to recover reading order, headings, tables, and figure structure. A lightweight post-processing layer will remove repeated headers/footers, downweight TOC/index/reference sections, and produce high-quality, provenance-rich chunks.

2. **Dialect Knowledge Graph (KG) beyond a Lexicon**  
   AgriBot will maintain a versioned, offline **knowledge graph** (stored in **SQLite graph tables** for portability) mapping dialect terms to canonical Bengali/English/scientific entities, and representing relations such as symptom→disease and disease→treatment. The KG is used for entity linking, query expansion, and disambiguation.

3. **Agentic Self-Correction using LangGraph**  
   AgriBot implements a bounded self-correct loop: **Retrieve → Grade → Rewrite/Expand → Retrieve → Generate → Verify**, ensuring that weak evidence triggers retrieval retry and that unsafe or ungrounded outputs are refused or converted into follow-up questions.

AgriBot is designed for real deployment: a laptop/desktop (consumer GPU e.g., RTX 3050 4GB) runs the offline models and indexes in “kiosk mode,” while a **mobile thin client** connects over local Wi‑Fi/LAN—delivering usability without forcing on-device LLM inference.

---

## 2. Problem Statement
AgriBot targets the following field-real constraints:

1. **Connectivity constraint:** Cloud AI is unreliable in rural environments; the system must function offline.
2. **Language + dialect constraint:** Farmers use colloquial/dialect Bengali; manuals use scientific terminology. Direct translation is insufficient.
3. **Visual-first diagnosis:** Crop issues are often best expressed via images (lesions, pest damage) and voice descriptions.
4. **PDF complexity and noise:** Manuals contain indexes/TOCs/references, repeated headers/footers, tables, diagrams, and scanned content.
5. **Safety and trust:** Agricultural recommendations can be harmful if incorrect; outputs must be evidence-based, cited, and conservative under uncertainty.
6. **Edge hardware constraint:** Deployment must be feasible on consumer hardware used by NGOs/local offices.

---

## 3. Goals and Success Criteria
### 3.1 Primary Goals (Must Deliver)
1. **Offline-first operation** after initial setup (models/indexes local).
2. **Multimodal input support**:
   - Voice (Bengali) via offline ASR,
   - Text (BN/EN),
   - Image (farmer crop photo).
3. **Bilingual outputs (BN + EN)** with aligned terms.
4. **Multimodal PDF ingestion**:
   - robust extraction of text, tables, and figures,
   - OCR for scanned content,
   - filtering/downweighting of irrelevant PDF sections.
5. **Hybrid retrieval** (BM25 + dense vectors) with reranking.
6. **Dialect Knowledge Graph** for entity linking and query expansion.
7. **LangGraph agentic self-correction loop** (bounded retries) + verification.
8. **Citations + evidence panel**: every answer includes source/page references.
9. **Voice output (offline TTS)** in Bengali.
10. **Production readiness**:
    - stable API contracts,
    - logging and metrics,
    - automated tests,
    - reproducible build artifacts.
11. **Deployment**:
    - React kiosk web UI for desktop/laptop,
    - mobile thin client app over LAN/Wi‑Fi,
    - exportable case reports.

### 3.2 Success Criteria (Measurable)
- **Groundedness:** ≥X% responses contain valid citations supporting key claims.
- **Refusal correctness:** system refuses when evidence is insufficient rather than hallucinating.
- **Retrieval quality:** improved Recall@k/MRR with KG expansion vs without.
- **Edge feasibility:** stable operation within consumer hardware limits; bounded latency per query.
- **Usability:** a non-technical user can run it via documented setup and UI.

---

## 4. Target Users and Use Cases
### 4.1 Users
- extension officers / NGO field staff (primary operators)
- farmers (via assisted kiosk or mobile client interaction)

### 4.2 Typical Scenarios
1. **Voice-based query:** farmer speaks Bengali symptoms; system asks follow-up questions; returns cited recommendations and speaks Bengali output.
2. **Image-based query:** farmer uploads a leaf image; system extracts symptoms and retrieves relevant evidence from manuals.
3. **Dosage query:** user asks dosage; system only provides dosage if found and cited; otherwise refuses and references manual sections.

---

## 5. System Overview
AgriBot combines:
- **Offline ASR/TTS** for voice interaction,
- **Multimodal PDF understanding** (Marker/Surya + OCR + post-processing),
- **Hybrid retrieval** (dense + BM25 + reranker),
- **Dialect Knowledge Graph** (SQLite graph) for term alignment,
- **LangGraph bounded self-correct loop** with evidence grading and verification,
- **Offline local LLM** running quantized on consumer GPU/CPU,
- **Kiosk web UI + mobile thin client**.

---

## 6. Architecture (Production View)
### 6.1 Core Modules / Services
1. **Ingestion Pipeline (Offline Build)**
   - Input: curated PDF set
   - Output: chunk store + indexes + KG snapshot (versioned)
2. **Orchestrator API (FastAPI + LangGraph)**
   - endpoint `/chat` executes full agent workflow
3. **Retrieval Service**
   - hybrid retrieval + reranking; returns structured evidence
4. **Knowledge Graph Service**
   - entity linking + alias expansion + neighborhood query expansion
5. **ASR/TTS Service**
   - Whisper ASR + offline Bengali TTS
6. **LLM Inference Service**
   - local quantized LLM (GGUF via llama.cpp)

### 6.2 Client Applications
- **React web kiosk UI** for desktop/laptop
- **mobile thin client** app over LAN/Wi‑Fi

---

## 7. Technology-to-Task Mapping (What is used where, and in which mode)
### 7.1 Execution Modes
- **Offline Runtime Mode (Primary):** all inference, retrieval, KG, and UI operate without internet.
- **Setup/Build Mode (Controlled):** model downloads and index builds may use internet once; runtime remains offline.

### 7.2 Input Processing (User Side)
| Task | Technology | Mode | Output |
|---|---|---|---|
| Voice → text (Bengali ASR) | **Whisper-family ASR** (recommended: **faster-whisper**) | Offline runtime | BN transcript + confidence |
| Image → description / text | **Offline VLM captioning** + **OCR** for text-on-image | Offline runtime | caption/ocr text + symptom hints |
| Text normalization | rule-based cleanup + unit normalization | Offline runtime | normalized BN/EN query |

### 7.3 Multimodal Document Understanding (PDF Ingestion)
| Task | Technology | Mode | Output |
|---|---|---|---|
| Layout-aware extraction (reading order, headings, blocks) | **Marker** or **Surya** | Offline build | structured text/markdown blocks |
| Noise reduction (header/footer, TOC/index/references) | frequency-based removal + page-type downweight | Offline build | cleaned pages + keep_weight |
| Table extraction | extraction to Markdown; OCR fallback | Offline build | table chunks w/ provenance |
| OCR for scanned content | **Tesseract OCR (offline)** | Offline build | OCR chunks w/ page provenance |
| Manual images/figures extraction | PDF image extraction | Offline build | figure artifacts |
| “Read” manual images | **VLM captioning** + OCR | Offline build | image_caption/image_ocr chunks |

### 7.4 Knowledge Graph (Dialect + Agronomy)
| Task | Technology | Mode | Output |
|---|---|---|---|
| KG storage | **SQLite graph tables** (entities, edges, provenance) | Offline runtime | versioned KG snapshot |
| Entity linking | alias match + canonical mapping + optional embedding match | Offline runtime | canonical entities + aliases |
| Graph expansion | 1–2 hop expansion for synonyms and related entities | Offline runtime | expanded bilingual query |

### 7.5 Retrieval and Grounded Generation
| Task | Technology | Mode | Output |
|---|---|---|---|
| Dense retrieval | **FAISS** + sentence-transformers embeddings | Offline runtime | semantic evidence candidates |
| Sparse retrieval | **BM25** (offline) | Offline runtime | keyword evidence candidates |
| Fusion | weighted merge | Offline runtime | combined evidence set |
| Reranking | FlashRank or CrossEncoder | Offline runtime | reranked evidence + scores |
| LLM generation | quantized local LLM via **llama.cpp** | Offline runtime | BN+EN answer + citations |
| Verification | citation coverage + claim support checks | Offline runtime | pass/fail + follow-ups |
| Voice output | offline Bengali **TTS** | Offline runtime | spoken Bengali response |

### 7.6 Product Layer
| Task | Technology | Mode | Output |
|---|---|---|---|
| Desktop UI | React + TypeScript | Offline runtime | kiosk web app |
| Mobile client | thin client over LAN | Offline runtime | voice/photo/text UI |
| API docs | FastAPI OpenAPI/Swagger | Offline runtime | self-documenting API |
| Logging | structured JSON logs | Offline runtime | traceability + debugging |
| Packaging | scripts + artifact versioning | Setup/build | reproducible install |

---

## 8. Multimodal PDF Filtering and Evidence Quality
AgriBot will prevent retrieval pollution by:
- using Marker/Surya to extract structured content,
- removing repeated header/footer noise,
- downweighting TOC/index/reference pages,
- filtering low-signal chunks,
- treating tables and image-derived text as first-class evidence.

All evidence returned is provenance-rich and citation-ready.

---

## 9. Dialect Knowledge Graph (Beyond Lexicon)
AgriBot’s KG will include:
- aliasing and canonical mapping (dialect → canonical)
- relations connecting symptoms, diseases, pests, and treatments
- provenance to support trust and maintainability

The KG will materially improve:
- retrieval recall for colloquial queries,
- disambiguation,
- evidence coverage during agentic correction.

---

## 10. LangGraph Agentic Self-Correction and Safety
### 10.1 Bounded Self-Correct Loop
**Normalize → KG Link → Expand → Retrieve → Rerank → Grade → (Rewrite + Retry) → Generate → Verify → Respond**

### 10.2 Safety Rules (Examples)
- no dosage without evidence chunk explicitly supporting it,
- ask follow-ups if context incomplete (crop stage, location, severity),
- refuse when evidence is missing or contradictory.

---

## 11. Production Readiness and Maintainability
AgriBot will be engineered for:
- reproducible builds (versioned chunk/index/KG artifacts),
- automated testing and regression checks,
- structured logging and measurable performance,
- modular components enabling iterative improvements without rewrites.

---

## 12. Deliverables
1. Production-ready FastAPI backend with LangGraph agent loop
2. Marker/Surya-based multimodal PDF ingestion pipeline with OCR and noise filtering
3. Hybrid retrieval + reranking with provenance-rich evidence
4. Dialect knowledge graph (SQLite) with entity linking and expansion
5. Offline Whisper ASR + offline Bengali TTS
6. React kiosk UI with citations/evidence viewer and report export
7. Mobile thin client app over LAN/Wi‑Fi
8. Evaluation harness and benchmark sets
9. Packaging/runbook + demo kit + documentation

---

# Planned Research Publications (Paper Title Portfolio)
These titles are designed to maximize competitiveness by emphasizing **generalizable CS contributions** (trustworthy offline agentic RAG, multimodal document intelligence, KG-augmented low-resource retrieval, and speech-first interaction) while using agriculture as a high-impact deployment domain.

## A. Flagship (Edge Systems + Trustworthy AI)
1. **“Evidence-Graded Agentic RAG on Consumer GPUs: Offline, Citation-Enforced Bilingual Decision Support for Low-Connectivity Regions”**
2. **“Trustworthy Offline Decision Support with Verified RAG: Transparent Evidence, Safety Policies, and Bounded Self-Correction”**

## B. Multimodal Document Intelligence for RAG
3. **“RAG from Real-World PDFs at Scale: Layout-Aware Extraction, Table/OCR Recovery, and Figure-Grounded Answering”**
4. **“From Figures and Tables to Grounded Answers: Multimodal Manual Ingestion for Evidence-Based Advisory under Offline Constraints”**

## C. Low-Resource Language + Knowledge Graph Augmentation
5. **“Dialect-to-Science Alignment via Knowledge Graph-Augmented Retrieval: Robust Evidence Search for Colloquial Bengali Queries”**
6. **“Graph-Guided Query Expansion for Low-Resource Bilingual RAG: Entity Linking and Provenance-Aware Evidence Selection”**

## D. Agentic Self-Correction + Verification (Method-Focused)
7. **“Self-Correcting RAG with Bounded Retrieval Loops: Evidence Grading, Query Rewriting, and Conservative Refusal Offline”**
8. **“Verification-First RAG for Safety-Critical Advisory: Citation Coverage and Claim Support Checks in Offline Agents”**

## E. Speech-First Grounded Advisory (ASR→RAG→TTS)
9. **“Speech-to-Evidence Advisory Systems: Offline Whisper, Graph Expansion, and Citation-Grounded Responses in Bengali”**
10. **“Bilingual Voice Interfaces for Verified RAG: Measuring ASR Error Propagation and Evidence Robustness in Low-Resource Settings”**
