# Graph Report - .  (2026-07-21)

## Corpus Check
- Large corpus: 124 files · ~581,474 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 1114 nodes · 1852 edges · 79 communities (63 shown, 16 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 130 edges (avg confidence: 0.61)
- Token cost: 0 input · 112,539 output

## Community Hubs (Navigation)
- KG Seeding & Hybrid Retrieval Core
- Audio Preprocessing (ffmpeg)
- React Chat Frontend (App/Composer)
- Grounding Policy & Refusal Enforcement
- Speech-to-Text (faster-whisper cascade)
- Frontend Lint/Build Tooling
- Text-to-Speech (pyttsx3)
- LangGraph Agent Graph Wiring
- Frontend TS App Config
- Chunk Data Model & Provenance
- Paper Figure Generation Scripts
- PDF Loading (Marker/OCR)
- Frontend Node TS Config
- FAISS/BM25 Index Builder
- Singleton Model Getters
- Chat API (image/voice) Models
- Frontend API Client
- PDF Chunking Pipeline
- Ingestion Pipeline Concepts (docs)
- LLM Engine (generate/grade)
- Chat API Request Handling
- BanglaT5 Translator
- Telemetry Logging
- Related-Work Citations (paper)
- Knowledge Graph CRUD
- Entity Linker & Query Expansion
- Config Module Tests
- KG Schema (entities/relations)
- Vision Classifier (CBAM)
- Image Symptom Analysis
- Deployment Runbook Concepts (docs)
- Image Analysis Result Model
- Agent Architecture Concepts (docs)
- KG CRUD Tests
- Grounding Policy Module
- Image Processor & Tests
- Evaluation Methodology Concepts (paper)
- Model Stack Spec (docs/MODELS.md)
- Grade Router & Tests
- Figure/Latency Concepts (paper+docs)
- Reranker & App Config
- Rate Limiter
- KG Search API
- Safety Policy Concepts (paper)
- KG Seed Data Tests
- PDF Page-Type Classification
- KG Entity Alias Lookup
- Reranker Citations (paper)
- Rerank Node
- Structured Logging Config
- Vision Schema Types
- API-Key & Rate-Limit Middleware
- Classifier Predict Impl
- Image Processor Singleton
- TTS API Endpoint
- Eval Runner Script
- Agent State Tests
- Health Check Endpoint
- KG Alias Model
- Frontend ChatInput Component
- Frontend TS Root Config
- Fig 1 Colab Snippet
- Fig 2 Colab Snippet
- Whisper Model Prefetch Script
- Agent Package Init
- Ingestion Package Init
- AgriBot Package Init
- KG Package Init
- LLM Package Init
- Retrieval Package Init
- Translation Package Init
- GPU Launcher Script
- Pytest Fixtures

## God Nodes (most connected - your core abstractions)
1. `KnowledgeGraph` - 40 edges
2. `EntityLinker` - 35 edges
3. `HybridRetriever` - 29 edges
4. `build_agent_graph()` - 26 edges
5. `SpeechToText` - 26 edges
6. `Chunk` - 25 edges
7. `Reranker` - 25 edges
8. `IndexBundle` - 23 edges
9. `TextToSpeech` - 23 edges
10. `AgriBot (paper system)` - 22 edges

## Surprising Connections (you probably didn't know these)
- `AgriBot (paper system)` --semantically_similar_to--> `AgriBot`  [INFERRED] [semantically similar]
  research/paper/main.tex → README.md
- `Domain-Adapted (LoRA) BanglaT5 EN→BN` --semantically_similar_to--> `BanglaT5 EN↔BN Translation`  [INFERRED] [semantically similar]
  research/paper/main.tex → README.md
- `AgriBot Evaluation Report (60-query baseline)` --semantically_similar_to--> `200-Query Bilingual Benchmark`  [INFERRED] [semantically similar]
  docs/EVALUATION.md → research/paper/main.tex
- `Translate-Before-Retrieve Pipeline Ordering` --semantically_similar_to--> `AgriBot Evaluation Report (60-query baseline)`  [INFERRED] [semantically similar]
  research/paper/main.tex → docs/EVALUATION.md
- `Two-Factor Refusal Policy` --semantically_similar_to--> `Two-Factor Refusal Policy`  [INFERRED] [semantically similar]
  research/paper/main.tex → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **LangGraph Bounded Agentic Workflow (normalize → kg_link → retrieve → rerank → grade → generate → translate → verify → enforce_policy)** — readme_agentic_rag_loop, readme_dialect_knowledge_graph, readme_hybrid_retrieval, docs_pipelines_deep_dive_grade_rewrite_failsafe, research_paper_main_two_factor_refusal_policy, research_paper_main_lexical_grounding_backstop, research_paper_main_domain_adapted_banglat5 [EXTRACTED 1.00]
- **Hardware Isolation: CPU-pinned auxiliary models around a GPU-resident LLM** — research_paper_main_hardware_isolation, docs_architecture_cpu_pinned_vision, docs_pipelines_deep_dive_cnn_over_vlm, docs_pipelines_deep_dive_translators_over_native_bn_llm, docs_deployment_single_worker_constraint, readme_qwen25_15b_q8_0 [INFERRED 0.85]
- **Safety Refusal Evidence Chain: policy, its measured trade-off, and the incident that motivated it** — research_paper_main_two_factor_refusal_policy, docs_evaluation_q39_fix, docs_evaluation_refusal_confusion_matrix, docs_evaluation_over_refusal_tradeoff, research_paper_main_four_configuration_ablation [EXTRACTED 1.00]

## Communities (79 total, 16 thin omitted)

### Community 0 - "KG Seeding & Hybrid Retrieval Core"
Cohesion: 0.05
Nodes (69): Seed the KG with initial agricultural terms and relations.      Categories:, seed_knowledge_graph(), get_llm(), Get or create the singleton LLM instance.      Thread-safe lazy initialization, HybridRetriever, Retrieve top-N evidence chunks using hybrid search.          Args:, Combines FAISS dense retrieval with BM25 sparse retrieval     using Reciprocal, Search FAISS index, return (chunk_idx, score) pairs. (+61 more)

### Community 1 - "Audio Preprocessing (ffmpeg)"
Cohesion: 0.06
Nodes (37): check_ffmpeg(), _ffmpeg_available(), preprocess_audio(), _preprocess_ffmpeg(), _preprocess_stdlib(), Path, Audio preprocessing for the AgriBot voice pipeline.  Canonicalizes uploaded audi, Resolve an ffmpeg binary path from env, PATH, or common winget location. (+29 more)

### Community 2 - "React Chat Frontend (App/Composer)"
Cohesion: 0.07
Nodes (34): App(), loadHistory(), uid(), confidenceTone(), ImageFindings(), PIPELINE_LABEL, CAPABILITIES, fade (+26 more)

### Community 3 - "Grounding Policy & Refusal Enforcement"
Cohesion: 0.06
Nodes (28): _extract_cited_facts(), is_risky_query(), make_enforce_policy_node(), Two-factor risk check: a query is risky only if it mentions both a     chemical, Extract only sentences from the answer that have direct support     in the evid, Create the grounding policy enforcement node.      This runs after verify and, GroundingMode, OnVerifyFailAction (+20 more)

### Community 4 - "Speech-to-Text (faster-whisper cascade)"
Cohesion: 0.06
Nodes (29): _build_warnings(), Path, Speech-to-Text using faster-whisper (CTranslate2 Whisper).  Provides offline, GP, Lazy-load the Whisper model on first use., Lazy-load optional BanglaSpeech2Text backend via official package API., Transcribe using optional BanglaSpeech2Text backend., Return rough script counts for Bengali, Devanagari, and Latin letters., Lazy-load Vosk Bengali model for fallback STT. (+21 more)

### Community 5 - "Frontend Lint/Build Tooling"
Cohesion: 0.04
Nodes (47): eslint, @eslint/js, eslint-plugin-react-hooks, eslint-plugin-react-refresh, framer-motion, dependencies, framer-motion, react (+39 more)

### Community 6 - "Text-to-Speech (pyttsx3)"
Cohesion: 0.06
Nodes (27): Path, List all available system TTS voices.          Returns:             List of d, Check if a Bengali voice is available on the system., Speak text aloud using the system TTS engine.          Args:             text, Save spoken text to a WAV audio file.          Args:             text: Text t, Save spoken text to a temporary WAV file.          Useful for Streamlit audio, Offline Text-to-Speech using pyttsx3.      Uses system-installed voices. On Wi, Initialize the TTS engine.          Args:             rate: Speaking rate (wo (+19 more)

### Community 7 - "LangGraph Agent Graph Wiring"
Cohesion: 0.13
Nodes (25): build_agent_graph(), Llama, LangGraph workflow definition.  Wires all agent nodes into a bounded self-corr, Build and compile the LangGraph agent workflow.      Args:         llm: Loade, _clean_answer(), make_generate_node(), make_grade_node(), make_kg_link_node() (+17 more)

### Community 8 - "Frontend TS App Config"
Cohesion: 0.07
Nodes (26): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, jsx, lib, module, moduleDetection, moduleResolution (+18 more)

### Community 9 - "Chunk Data Model & Provenance"
Cohesion: 0.11
Nodes (14): Chunk, A text chunk with full provenance metadata., Serialize for storage alongside FAISS index., Deserialize from stored dict., Human-readable citation string., EvidenceChunk, A retrieved chunk with relevance scoring and provenance., Tests for the retrieval pipeline (hybrid retriever and reranker). These are int (+6 more)

### Community 10 - "Paper Figure Generation Scripts"
Cohesion: 0.12
Nodes (23): _draw_arrow(), _draw_box(), fig1_mpl(), fig2_mpl(), fig3_retrieval(), fig4_latency(), fig5_confusion(), fig6_kg_example() (+15 more)

### Community 11 - "PDF Loading (Marker/OCR)"
Cohesion: 0.13
Nodes (22): _extract_with_fallback(), _extract_with_marker(), _extract_with_ocr(), _get_marker_models(), _get_pdf_page_count(), load_pdfs(), _ocr_empty_pages(), Path (+14 more)

### Community 12 - "Frontend Node TS Config"
Cohesion: 0.09
Nodes (22): compilerOptions, allowImportingTsExtensions, erasableSyntaxOnly, lib, module, moduleDetection, moduleResolution, noEmit (+14 more)

### Community 13 - "FAISS/BM25 Index Builder"
Cohesion: 0.13
Nodes (15): build_indexes(), IndexBundle, Path, Index builder: FAISS (dense) + BM25 (sparse) indexes.  Persists both indexes a, Build FAISS and BM25 indexes from chunks.      Args:         chunks: List of, Container for FAISS index, BM25 index, and chunk metadata., Persist all indexes and metadata to disk., Load all indexes and metadata from disk. (+7 more)

### Community 14 - "Singleton Model Getters"
Cohesion: 0.12
Nodes (20): get_classifier(), Get or create the singleton classifier.      Returns None if model_path is None, get_stt(), Get or create the singleton STT instance.      Thread-safe lazy initialization., get_tts(), Get or create the singleton TTS instance.      Thread-safe lazy initialization, _init_services(), kg_stats() (+12 more)

### Community 15 - "Chat API (image/voice) Models"
Cohesion: 0.13
Nodes (21): chat_image(), chat_voice(), ChatResponseV1, DiagnosticsBlock, ImageBlock, Legacy voice endpoint — use /v1/chat/voice instead., Legacy image endpoint — use /v1/chat/image instead., Per-request diagnostics and traceability. (+13 more)

### Community 16 - "Frontend API Client"
Cohesion: 0.16
Nodes (14): getHealth(), getTTSAudio(), request(), sendChat(), ChatRequest, ChatResponse, EvidenceData, HealthResponse (+6 more)

### Community 17 - "PDF Chunking Pipeline"
Cohesion: 0.19
Nodes (12): chunk_pages(), Provenance-rich text chunker.  Splits extracted pages into chunks while preser, Split pages into provenance-rich chunks.      Args:         pages: List of Pa, _detect_repeated_lines(), PageData, Find lines that appear in more than `freq_threshold` fraction of pages.     The, A single extracted PDF page with metadata., Tests for the ingestion pipeline: PDF loading, chunking, and index building. (+4 more)

### Community 18 - "Ingestion Pipeline Concepts (docs)"
Cohesion: 0.12
Nodes (20): Corpus Ingestion Pipeline, Embedding/Vector Alignment Constraint, ASR Confidence Threshold Fallback Trigger, Offline Ingestion Pipeline (Marker → Chunker → Indexes), Header/Footer Noise Filtering Heuristic, Voice & Audio Pipeline, Frontend HTML App Shell, React + TypeScript + Vite Template (+12 more)

### Community 19 - "LLM Engine (generate/grade)"
Cohesion: 0.18
Nodes (18): _evidence_overlap(), generate(), generate_answer(), grade_evidence(), Llama, _query_coverage(), LLM inference engine: singleton wrapper around llama-cpp-python.  Provides str, Grade whether the retrieved evidence is sufficient to answer the query.      R (+10 more)

### Community 20 - "Chat API Request Handling"
Cohesion: 0.15
Nodes (13): _build_initial_state(), chat(), ChatRequest, HealthResponse, Legacy chat endpoint — use /v1/chat instead., Text-based chat request, also used for voice-confirmed re-submission., System health status., Build the initial agent state for a query. (+5 more)

### Community 21 - "BanglaT5 Translator"
Cohesion: 0.14
Nodes (10): BanglaTranslator, BUET BanglaT5 Translator.  Uses csebuetnlp/banglat5_nmt_en_bn for English→Beng, Apply Bengali unicode normalization if available., Split text into sentences for better translation quality.         T5 models wor, Translate English text to Bengali.          Handles long text by splitting int, Translate one line/block of prose EN->BN (sentence-wise for quality)., Bidirectional English↔Bengali translator using BUET BanglaT5.      Models:, Translate Bengali text to English.          Useful for translating Bengali use (+2 more)

### Community 22 - "Telemetry Logging"
Cohesion: 0.18
Nodes (10): get_telemetry(), _percentile(), Path, Lightweight local telemetry — async fire-and-forget logging of every request int, SQLite percentile via offset trick. Returns 0.0 on empty set., Process-wide singleton accessor., Thread-safe SQLite-backed request log., Insert one request. Swallows exceptions — telemetry must never         break a l (+2 more)

### Community 23 - "Related-Work Citations (paper)"
Cohesion: 0.12
Nodes (17): Abhishek et al. 2025 — AgriTalk, Balpande et al. 2024 — Agriculture RAG Chatbot, Saiful Islam 2023 — BanglaSpeech2Text, BBS 2024 — Labour Force Survey Bangladesh, Singh et al. 2024 — Farmer.Chat, Ramírez 2018 — FastAPI, Wang et al. 2025 — KG-based RAG, Chase 2022 — LangChain / LangGraph (+9 more)

### Community 24 - "Knowledge Graph CRUD"
Cohesion: 0.13
Nodes (8): KnowledgeGraph, Path, Add an entity, return its ID., Add a dialect alias for an entity., Add a directed relation between entities., Close database connection., SQLite-backed dialect knowledge graph for agricultural term alignment., Create tables if they don't exist.

### Community 25 - "Entity Linker & Query Expansion"
Cohesion: 0.20
Nodes (7): EntityLinker, Links query terms to KG entities and expands queries., Split query into candidate tokens and bigrams., Find KG entities that match terms in the query.          Tries exact alias mat, Expand a query with KG-linked terms for better retrieval.          Process:, Tests for entity linking and query expansion., TestEntityLinker

### Community 26 - "Config Module Tests"
Cohesion: 0.14
Nodes (13): Tests for configuration module., Config should load with default values., All paths should be resolved to absolute paths., DATA_DIR should be under BASE_DIR., Dense + sparse weights should sum to ~1.0., Chunk overlap must be less than chunk size., TOC keywords list should have entries., test_config_chunk_overlap_less_than_size() (+5 more)

### Community 27 - "KG Schema (entities/relations)"
Cohesion: 0.18
Nodes (9): Entity linker and query expander using the Knowledge Graph.  Given a user quer, SQLite-based Dialect Knowledge Graph.  Schema: - entities: canonical agricult, Relation, Seed data for the Dialect Knowledge Graph.  Populates the KG with common Benga, kg(), Tests for the Knowledge Graph schema, entity linker, and seed data., Create a fresh KG in a temp directory., KG with seed data loaded. (+1 more)

### Community 28 - "Vision Classifier (CBAM)"
Cohesion: 0.21
Nodes (7): CropClassifier, Check if classifier is loaded and ready., Lightweight crop disease classifier using MobileNetV3 + CBAM.      Loads a model, Load the classification model.          Supported formats:           - .onnx: us, Tests for the image analysis pipeline.  Tests baseline OCR+heuristic path and cl, Test the CropClassifier interface., TestClassifier

### Community 29 - "Image Symptom Analysis"
Cohesion: 0.21
Nodes (7): Path, Generate a caption using an offline Vision Language Model.          Currently, Extract text from image using Tesseract OCR., Fraction of pixels that look like plant foliage, via the Excess         Green i, Analyze image for agricultural symptom indicators using color analysis., Analyze a crop/plant image and return a textual description.          Uses OCR, Analyze a crop/plant image and return structured analysis.          Args:

### Community 30 - "Deployment Runbook Concepts (docs)"
Cohesion: 0.17
Nodes (13): CPU-Pinned Vision Subsystem, AVX2-Only llama-cpp-python Build Requirement, Off-System-Drive Cache Redirection, AgriBot Deployment Runbook, Field-Deployment Security Checklist, Single-Worker Uvicorn Constraint, SQLite Telemetry Store, Lightweight CNN Instead of VLM (+5 more)

### Community 31 - "Image Analysis Result Model"
Cohesion: 0.23
Nodes (5): ImageAnalysisResult, Structured result from image analysis pipeline.      Attributes:         pipelin, Build a combined query string for RAG retrieval.          Retrieval is embedding, Test the ImageAnalysisResult data class., TestImageAnalysisResult

### Community 32 - "Agent Architecture Concepts (docs)"
Cohesion: 0.20
Nodes (12): LangGraph Agent Orchestrator, Hybrid Retriever Layer, Shared Immutable LangGraph State Object, Outlines-Based Structured Decoding, Knowledge Graph Seed Relations, Embedding / Bi-Encoder Specification, Agentic LLM Main Pipeline (LangGraph State Machine), Dialect KG Query Expansion (+4 more)

### Community 34 - "Grounding Policy Module"
Cohesion: 0.22
Nodes (9): Grounding policy enforcement for AgriBot.  Controls how the system handles unv, Build clarifying follow-ups when an image was provided but the     classifier's, _vision_followups(), Wrap a node function with timing instrumentation., _timed(), AgentState, Agent state definition for the LangGraph workflow., Full state for the agentic RAG pipeline. (+1 more)

### Community 35 - "Image Processor & Tests"
Cohesion: 0.27
Nodes (5): ImageProcessor, Offline image analysis for agricultural crop photos.      Capabilities:     1, Initialize the image processor.          Args:             vlm_model_path: Op, Test describe_image_structured (mock OCR + symptoms)., TestImageProcessorStructured

### Community 36 - "Evaluation Methodology Concepts (paper)"
Cohesion: 0.18
Nodes (11): Comparison to Vanilla RAG and Self-RAG, Benchmark Query Categories, Post-Evaluation Improvement Roadmap, Safety-First Over-Refusal Trade-off, Refusal Confusion Matrix, AgriBot Evaluation Report (60-query baseline), Rate-Limit Backpressure Under Load, Lewis et al. 2020 — Retrieval-Augmented Generation (+3 more)

### Community 37 - "Model Stack Spec (docs/MODELS.md)"
Cohesion: 0.18
Nodes (11): Primary LLM Specification (docs/MODELS.md), Bounded LangGraph Agentic RAG Loop, BanglaT5 EN↔BN Translation, Evaluation Harness (200-query benchmark), Qwen2.5-1.5B-Instruct Q8_0 GGUF, AgriBanglaT5 Domain-Adapted LoRA Stack (peft), Bhattacharjee et al. 2023 — BanglaT5, Hasan et al. 2020 — BanglaNMT (+3 more)

### Community 38 - "Grade Router & Tests"
Cohesion: 0.29
Nodes (5): _make_grade_router(), Create a grade router closure with configurable max_retries., Tests for the agent state and graph structure., Tests for the grade routing logic., TestGradeRouter

### Community 39 - "Figure/Latency Concepts (paper+docs)"
Cohesion: 0.20
Nodes (10): Per-Stage Latency Breakdown, Fig 1 — Architecture Diagram, Fig 5 — CBAM CNN Confusion Matrix, Single-Script Figure Generation, IEEE Figure Quality Checklist, Rice Disease Class Labels (6 classes), Howard et al. 2019 — MobileNetV3, Mishra 2024 — Rice Leaf AUG Dataset (+2 more)

### Community 40 - "Reranker & App Config"
Cohesion: 0.22
Nodes (6): Cross-encoder reranker for filtering and reordering retrieved evidence.  Uses, BaseSettings, AgriConfig, AgriBot Central Configuration All tunable parameters in one place., Resolve relative paths after init., Production configuration with sensible defaults for RTX 3050.

### Community 41 - "Rate Limiter"
Cohesion: 0.25
Nodes (5): _Bucket, In-memory token-bucket rate limiter, keyed by client IP.  Single-node design — s, Steady rate of `per_minute` requests per key, with `burst` overflow.      `acqui, Clear bucket state for one key (or all keys when None). Test helper., TokenBucketRateLimiter

### Community 42 - "KG Search API"
Cohesion: 0.22
Nodes (9): kg_search(), KGSearchResponse, Search the knowledge graph for entities., Legacy KG search — use /v1/kg/search instead., ASR diagnostics for voice-input requests., Knowledge graph search results., v1_kg_search(), VoiceBlock (+1 more)

### Community 43 - "Safety Policy Concepts (paper)"
Cohesion: 0.25
Nodes (9): Grounding & Safety Policy Enforcement, AGRIBOT_* Environment Configuration, Q39 Urea-Dosage Safety Fix, Fig 2 — LangGraph Workflow Diagram, Two-Factor Refusal Policy, Bounded Agentic Loop with Verification, GPU Non-Reproducibility Caveat (±3 pp), Lexical Grounding Backstop (+1 more)

### Community 44 - "KG Seed Data Tests"
Cohesion: 0.22
Nodes (4): Tests for KG seed data., Seeding twice should not duplicate data., Colloquial/dialect terms should be present., TestSeedData

### Community 45 - "PDF Page-Type Classification"
Cohesion: 0.36
Nodes (4): _classify_page_type(), Classify a page as content, toc, reference, or low_signal.     Returns (page_ty, Tests for page type classification., TestPageClassification

### Community 46 - "KG Entity Alias Lookup"
Cohesion: 0.29
Nodes (4): Entity, Find entities matching an alias (case-insensitive)., Find entities with aliases containing the text., Get neighboring entities within `hops` distance.          Returns list of (Ent

### Community 47 - "Reranker Citations (paper)"
Cohesion: 0.25
Nodes (8): Cross-Encoder Reranker, Cross-Encoder Reranker Specification, Cormack et al. 2009 — Reciprocal Rank Fusion, Johnson et al. 2021 — FAISS, Nogueira & Cho 2019 — Passage Re-ranking with BERT, Reimers & Gurevych 2019 — Sentence-BERT, Robertson & Zaragoza 2009 — BM25, Two-Stage Hybrid Retrieval with RRF

### Community 48 - "Rerank Node"
Cohesion: 0.33
Nodes (5): make_rerank_node(), Create the reranking node., Reranks evidence chunks using a cross-encoder model., Rerank evidence chunks and filter low-scoring results.          Args:, Reranker

### Community 49 - "Structured Logging Config"
Cohesion: 0.29
Nodes (6): get_logger(), AgriBot structured logging configuration using structlog.  Provides JSON-forma, Configure structured logging for the entire application.      Args:         j, Get a structured logger by name.      Usage:         logger = get_logger(__na, setup_logging(), BoundLogger

### Community 50 - "Vision Schema Types"
Cohesion: 0.33
Nodes (4): Optional classifier-assisted image analysis for AgriBot.  Provides Top-K disease, PossibleCondition, Structured output schema for AgriBot image analysis pipeline.  Defines the canon, A predicted disease/condition with confidence score.

### Community 51 - "API-Key & Rate-Limit Middleware"
Cohesion: 0.33
Nodes (7): api_key_middleware(), _client_key(), rate_limit_middleware(), Identify a client for rate-limiting. Prefers X-Forwarded-For (first     hop), fa, Per-client token-bucket rate limit. Bypasses health/docs/static., Enforce API key if AGRIBOT_API_KEY is configured., Request

### Community 52 - "Classifier Predict Impl"
Cohesion: 0.47
Nodes (4): Path, Run Top-K classification on a crop image.          Args:             image_path:, Internal prediction implementation.          Uses ONNX Runtime for .onnx models, PossibleCondition

### Community 53 - "Image Processor Singleton"
Cohesion: 0.40
Nodes (4): get_image_processor(), Offline Image Processor for crop/plant photos.  Combines OCR (Tesseract) for t, Get or create the singleton ImageProcessor instance.      Thread-safe lazy ini, AgriBot Vision: Offline image processing for crop/plant diagnosis.

### Community 54 - "TTS API Endpoint"
Cohesion: 0.40
Nodes (6): Legacy TTS endpoint — use /v1/tts instead., Text-to-speech request., Synthesize text to speech (WAV audio stream)., text_to_speech(), TTSRequest, v1_tts()

### Community 55 - "Eval Runner Script"
Cohesion: 0.53
Nodes (5): _backend_alive(), _load_latest(), main(), End-to-end evaluation orchestrator.  Runs all five eval scripts in order, captur, _run_step()

### Community 56 - "Agent State Tests"
Cohesion: 0.33
Nodes (4): Tests for agent state structure., AgentState should have all required fields., A valid initial state should be constructible., TestAgentState

### Community 57 - "Health Check Endpoint"
Cohesion: 0.50
Nodes (4): health_check(), Legacy health endpoint — use /v1/health instead., System health, KG stats, manifest, and enabled modules., v1_health()

## Ambiguous Edges - Review These
- `Qwen2.5-1.5B-Instruct Q8_0 GGUF` → `Primary LLM Specification (docs/MODELS.md)`  [AMBIGUOUS]
  docs/MODELS.md · relation: conceptually_related_to
- `Cross-Encoder Reranker Specification` → `Two-Stage Hybrid Retrieval with RRF`  [AMBIGUOUS]
  docs/MODELS.md · relation: conceptually_related_to

## Knowledge Gaps
- **117 isolated node(s):** `Relation`, `name`, `private`, `version`, `type` (+112 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Qwen2.5-1.5B-Instruct Q8_0 GGUF` and `Primary LLM Specification (docs/MODELS.md)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Cross-Encoder Reranker Specification` and `Two-Stage Hybrid Retrieval with RRF`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `KnowledgeGraph` connect `Knowledge Graph CRUD` to `KG Seeding & Hybrid Retrieval Core`, `KG CRUD Tests`, `KG Search API`, `KG Seed Data Tests`, `FAISS/BM25 Index Builder`, `KG Entity Alias Lookup`, `Singleton Model Getters`, `Chat API (image/voice) Models`, `Chat API Request Handling`, `TTS API Endpoint`, `Entity Linker & Query Expansion`, `KG Alias Model`, `KG Schema (entities/relations)`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `build_agent_graph()` connect `LangGraph Agent Graph Wiring` to `KG Seeding & Hybrid Retrieval Core`, `Grounding Policy Module`, `Grounding Policy & Refusal Enforcement`, `Grade Router & Tests`, `Singleton Model Getters`, `Rerank Node`, `BanglaT5 Translator`, `Entity Linker & Query Expansion`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `preprocess_audio()` connect `Audio Preprocessing (ffmpeg)` to `Singleton Model Getters`, `Chat API (image/voice) Models`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `KnowledgeGraph` (e.g. with `EntityLinker` and `ChatRequest`) actually correct?**
  _`KnowledgeGraph` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `EntityLinker` (e.g. with `Entity` and `KnowledgeGraph`) actually correct?**
  _`EntityLinker` has 13 INFERRED edges - model-reasoned connections that need verification._