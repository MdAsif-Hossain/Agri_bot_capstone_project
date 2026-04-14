"""
AgriBot — Production FastAPI Backend.

Provides versioned REST API endpoints:
 - /v1/health          — system health + manifest
 - /v1/chat            — text query (+ voice_confirmed re-submission)
 - /v1/chat/voice      — voice query with confidence gating
 - /v1/chat/image      — image query (OCR baseline + optional classifier)
 - /v1/tts             — text-to-speech synthesis
 - /v1/kg/stats        — knowledge graph statistics
 - /v1/kg/search       — search KG entities

Legacy endpoints (/chat, /health, etc.) remain as thin wrappers.

Response schema uses nested blocks:
 - diagnostics: trace_id, timings_ms, mode_flags, warnings
 - voice?: transcript, asr_language, asr_confidence, needs_confirmation, ...
 - image?: pipeline_used, analysis_summary, limitations, possible_conditions
"""

import sys
import uuid
import time
import asyncio
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    APIRouter,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import structlog

# --- Ensure project root is on path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from agribot.logging_config import setup_logging, get_logger

setup_logging(json_output=True, log_level="INFO")
logger = get_logger("agribot.api")

# =============================================================================
# GLOBAL SERVICES + CONCURRENCY CONTROL
# =============================================================================
_services: dict = {}
_llm_semaphore: asyncio.Semaphore | None = None
_stt_semaphore: asyncio.Semaphore | None = None
_image_semaphore: asyncio.Semaphore | None = None


def _init_services() -> dict:
    """Load all models and build the agent pipeline."""
    from agribot.llm.engine import get_llm
    from agribot.ingestion.index_builder import IndexBundle
    from agribot.retrieval.hybrid import HybridRetriever
    from agribot.retrieval.reranker import Reranker
    from agribot.knowledge_graph.schema import KnowledgeGraph
    from agribot.knowledge_graph.entity_linker import EntityLinker
    from agribot.knowledge_graph.seed_data import seed_knowledge_graph
    from agribot.agent.graph import build_agent_graph
    from agribot.translation.bangla_t5 import get_translator
    from agribot.voice.stt import get_stt
    from agribot.voice.tts import get_tts
    from agribot.voice.audio_preprocess import check_ffmpeg

    svc = {}

    # 1. LLM
    logger.info("Loading LLM model...")
    svc["llm"] = get_llm(
        model_path=str(settings.MODEL_PATH),
        n_ctx=settings.LLM_N_CTX,
        n_gpu_layers=settings.LLM_N_GPU_LAYERS,
    )

    # 2. Indexes
    logger.info("Loading document indexes...")
    if not settings.INDEX_DIR.exists():
        raise RuntimeError(
            f"Index directory not found: {settings.INDEX_DIR}. "
            "Run `python ingest.py` first."
        )
    svc["index_bundle"] = IndexBundle.load(settings.INDEX_DIR)

    # 3. Embedding model
    logger.info("Loading embedding model...")
    from sentence_transformers import SentenceTransformer

    svc["embedding_model"] = SentenceTransformer(settings.EMBEDDING_MODEL)

    # 4. Retriever
    svc["retriever"] = HybridRetriever(
        index_bundle=svc["index_bundle"],
        embedding_model=svc["embedding_model"],
        dense_top_k=settings.DENSE_TOP_K,
        sparse_top_k=settings.SPARSE_TOP_K,
        dense_weight=settings.DENSE_WEIGHT,
        sparse_weight=settings.SPARSE_WEIGHT,
    )

    # 5. Reranker
    logger.info("Loading reranker...")
    svc["reranker"] = Reranker(
        threshold=settings.RERANK_THRESHOLD,
        top_n=settings.RERANK_TOP_N,
    )

    # 6. Knowledge Graph
    logger.info("Loading knowledge graph...")
    svc["kg"] = KnowledgeGraph(settings.KG_DB_PATH)
    seed_knowledge_graph(svc["kg"])
    svc["entity_linker"] = EntityLinker(svc["kg"])

    # 7. Translator
    logger.info("Loading BanglaT5 translator...")
    svc["translator"] = get_translator(device="cpu")

    # 8. Voice services
    logger.info("Initializing voice services...")
    svc["stt"] = get_stt(
        model_size=settings.WHISPER_MODEL_SIZE,
        device="cpu",
        beam_size=settings.WHISPER_BEAM_SIZE,
        vad_filter=settings.WHISPER_VAD_FILTER,
        min_silence_ms=settings.WHISPER_MIN_SILENCE_MS,
        language_hint=settings.WHISPER_LANGUAGE_HINT,
        task=settings.WHISPER_TASK,
        banglaspeech2text_enabled=settings.BANGLASPEECH2TEXT_ENABLED,
        banglaspeech2text_model_id=settings.BANGLASPEECH2TEXT_MODEL_ID,
        vosk_fallback_enabled=settings.VOSK_FALLBACK_ENABLED,
        vosk_bn_model_path=settings.VOSK_BN_MODEL_PATH,
    )
    svc["vosk_model_ready"] = Path(settings.VOSK_BN_MODEL_PATH).exists()
    svc["tts"] = get_tts(
        rate=settings.TTS_RATE, bengali_voice_name=settings.TTS_BENGALI_VOICE
    )
    svc["ffmpeg_available"] = check_ffmpeg()

    # 9. Optional image classifier
    if settings.IMAGE_CLASSIFIER_ENABLED and settings.IMAGE_CLASSIFIER_MODEL_PATH:
        try:
            from agribot.vision.classifier import get_classifier

            svc["classifier"] = get_classifier(
                model_path=settings.IMAGE_CLASSIFIER_MODEL_PATH,
                top_k=settings.IMAGE_CLASSIFIER_TOP_K,
                confidence_threshold=settings.IMAGE_CLASSIFIER_CONFIDENCE_THRESHOLD,
            )
        except Exception as e:
            logger.warning("Classifier init failed: %s; disabled", e)
            svc["classifier"] = None
    else:
        svc["classifier"] = None

    # 10. Agent graph (with config-driven retries + grounding policy)
    logger.info("Building agent pipeline...")
    svc["agent"] = build_agent_graph(
        llm=svc["llm"],
        retriever=svc["retriever"],
        reranker=svc["reranker"],
        entity_linker=svc["entity_linker"],
        translator=svc["translator"],
        max_tokens=settings.LLM_MAX_TOKENS,
        max_retries=settings.MAX_RETRIES,
        grounding_mode=settings.GROUNDING_MODE,
        on_verify_fail=settings.ON_VERIFY_FAIL,
    )

    svc["kg_stats"] = svc["kg"].get_stats()
    svc["chunk_count"] = len(svc["index_bundle"].chunks)

    # Load manifest if exists
    manifest_path = settings.INDEX_DIR / "manifest.json"
    if manifest_path.exists():
        import json

        svc["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        svc["manifest"] = None

    logger.info("All services initialized successfully")
    return svc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for FastAPI."""
    global _services, _llm_semaphore, _stt_semaphore, _image_semaphore
    logger.info("Starting AgriBot API server...")
    _services = _init_services()
    _llm_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM)
    _stt_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_STT)
    _image_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_IMAGE_ANALYSIS)
    yield
    if "kg" in _services:
        _services["kg"].close()
    logger.info("AgriBot API server shutdown")


# =============================================================================
# APP
# =============================================================================
app = FastAPI(
    title="AgriBot API",
    description=(
        "Offline Multimodal Agentic RAG System for Bilingual "
        "(Bengali–English) Agricultural Decision Support"
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# --- CORS (configurable, not wide-open by default) ---
_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# OPTIONAL API KEY MIDDLEWARE
# =============================================================================


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Enforce API key if AGRIBOT_API_KEY is configured."""
    if not settings.API_KEY:
        return await call_next(request)

    # Skip auth for health, docs, static files
    path = request.url.path
    if (
        path in ("/health", "/v1/health", "/docs", "/redoc", "/openapi.json")
        or path.startswith("/assets")
        or not path.startswith("/v1/")
    ):
        return await call_next(request)

    # Protected endpoints require API key
    if path.startswith("/v1/chat") or path.startswith("/v1/tts"):
        provided = request.headers.get("X-API-Key", "")
        if provided != settings.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

    return await call_next(request)


# =============================================================================
# NESTED RESPONSE MODELS
# =============================================================================


class DiagnosticsBlock(BaseModel):
    """Per-request diagnostics and traceability."""

    trace_id: str = Field(default="", description="Request trace ID")
    timings_ms: dict[str, float] = Field(
        default_factory=dict, description="Per-stage timings"
    )
    mode_flags: list[str] = Field(
        default_factory=list, description="Processing mode indicators"
    )
    warnings: list[str] = Field(default_factory=list, description="Processing warnings")


class VoiceBlock(BaseModel):
    """ASR diagnostics for voice-input requests."""

    transcript: str = Field(default="", description="Raw transcript from STT")
    asr_language: str = Field(default="", description="Detected language code")
    asr_confidence: float = Field(default=0.0, description="ASR confidence 0-1")
    asr_warnings: list[str] = Field(
        default_factory=list, description="ASR-specific warnings"
    )
    needs_confirmation: bool = Field(
        default=False, description="True if transcript needs user confirmation"
    )
    transcript_suspected: str = Field(
        default="", description="Suspected transcript for user review"
    )
    suggested_actions: list[str] = Field(
        default_factory=list, description="Actions user can take"
    )


class ImageBlock(BaseModel):
    """Image analysis diagnostics."""

    pipeline_used: str = Field(
        default="ocr_baseline",
        description="Analysis pipeline: ocr_baseline|classifier_assisted|ocr_fallback",
    )
    analysis_summary: dict = Field(
        default_factory=dict, description="Structured analysis result"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Analysis limitations"
    )
    possible_conditions: list[dict] = Field(
        default_factory=list, description="Top-K conditions [{label, confidence}]"
    )


class ChatResponseV1(BaseModel):
    """Chat response with bilingual answer, evidence, and nested diagnostics."""

    answer: str = Field(default="", description="English answer")
    answer_bn: str = Field(default="", description="Bengali translation")
    citations: list[str] = Field(default_factory=list, description="Source citations")
    kg_entities: list[dict] = Field(
        default_factory=list, description="Linked KG entities"
    )
    evidence_grade: str = Field(default="N/A", description="Evidence quality grade")
    is_verified: bool = Field(
        default=False, description="Whether answer passed verification"
    )
    verification_reason: str = Field(default="", description="Verification details")
    retry_count: int = Field(default=0, description="Number of retrieval retries")
    input_mode: str = Field(default="text", description="Input mode used")
    grounding_action: str = Field(default="pass", description="Grounding policy action")
    follow_up_suggestions: list[str] = Field(
        default_factory=list, description="Follow-up queries"
    )
    parsed_query: str = Field(default="", description="The query sent to RAG")
    # --- Nested blocks ---
    diagnostics: DiagnosticsBlock = Field(default_factory=DiagnosticsBlock)
    voice: Optional[VoiceBlock] = Field(
        default=None, description="Voice ASR diagnostics (only for voice input)"
    )
    image: Optional[ImageBlock] = Field(
        default=None, description="Image analysis diagnostics (only for image input)"
    )


# Backward-compatible alias
ChatResponse = ChatResponseV1


class ChatRequest(BaseModel):
    """Text-based chat request, also used for voice-confirmed re-submission."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query in English or Bengali",
    )
    input_mode: str = Field(
        default="text", description="Input mode: text|voice_confirmed"
    )
    trace_id: str = Field(
        default="", description="Original trace_id for voice-confirmed linkage"
    )


class HealthResponse(BaseModel):
    """System health status."""

    status: str
    chunk_count: int
    kg_entities: int
    kg_aliases: int
    kg_relations: int
    manifest: dict | None = Field(default=None, description="Index manifest summary")
    enabled_modules: dict = Field(default_factory=dict, description="Module status")
    grounding_mode: str = Field(default="strict", description="Current grounding mode")


class KGSearchResponse(BaseModel):
    """Knowledge graph search results."""

    entities: list[dict]


class TTSRequest(BaseModel):
    """Text-to-speech request."""

    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to synthesize"
    )
    language: str = Field(default="en", description="Language: 'en' or 'bn'")


# =============================================================================
# HELPER
# =============================================================================


def _build_initial_state(
    query: str, input_mode: str = "text", trace_id: str = ""
) -> dict:
    """Build the initial agent state for a query."""
    return {
        "query_original": query,
        "query_language": "",
        "query_normalized": "",
        "query_expanded": "",
        "kg_entities": [],
        "evidences": [],
        "evidence_texts": "",
        "evidence_grade": "",
        "answer": "",
        "answer_bn": "",
        "citations": [],
        "is_verified": False,
        "verification_reason": "",
        "retry_count": 0,
        "should_refuse": False,
        "input_mode": input_mode,
        "input_audio_path": "",
        "error": "",
        "trace_id": trace_id,
        "timings_ms": {},
        "grounding_action": "pass",
        "follow_up_suggestions": [],
    }


async def _run_agent(
    query: str,
    input_mode: str = "text",
    extra_diagnostics: dict | None = None,
    voice_block: VoiceBlock | None = None,
    image_block: ImageBlock | None = None,
    trace_id: str | None = None,
) -> ChatResponseV1:
    """Run the agent pipeline with concurrency control and timeout."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=trace_id)

    logger.info("Processing request", query=query[:100], input_mode=input_mode)

    # Acquire LLM semaphore with timeout
    try:
        if _llm_semaphore.locked():
            logger.info("LLM semaphore busy, queuing request")
        async with asyncio.timeout(settings.REQUEST_TIMEOUT_S):
            await _llm_semaphore.acquire()
    except asyncio.TimeoutError:
        logger.warning("Request timeout waiting for LLM semaphore")
        raise HTTPException(
            status_code=429,
            detail="System is busy processing other requests. Please try again.",
            headers={"Retry-After": "10"},
        )

    try:
        agent = _services["agent"]
        initial_state = _build_initial_state(query, input_mode, trace_id)

        # Run agent in thread pool
        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        result = await asyncio.wait_for(
            loop.run_in_executor(None, agent.invoke, initial_state),
            timeout=settings.REQUEST_TIMEOUT_S,
        )

        total_ms = (time.perf_counter() - start) * 1000
        timings = result.get("timings_ms", {})
        timings["total"] = round(total_ms, 1)

        # Merge extra diagnostics (from preprocessing steps)
        if extra_diagnostics:
            timings.update(extra_diagnostics.get("timings_ms", {}))

        # Build diagnostics block
        mode_flags = [input_mode]
        diag_warnings: list[str] = []

        if extra_diagnostics:
            mode_flags.extend(extra_diagnostics.get("mode_flags", []))
            diag_warnings.extend(extra_diagnostics.get("warnings", []))

        # Check for translation warning
        if not result.get("answer_bn") and result.get("answer"):
            diag_warnings.append("translation_unavailable")

        diagnostics = DiagnosticsBlock(
            trace_id=trace_id,
            timings_ms=timings,
            mode_flags=mode_flags,
            warnings=diag_warnings,
        )

        return ChatResponseV1(
            answer=result.get("answer", "An error occurred."),
            answer_bn=result.get("answer_bn", ""),
            citations=result.get("citations", []),
            kg_entities=result.get("kg_entities", []),
            evidence_grade=result.get("evidence_grade", "N/A"),
            is_verified=result.get("is_verified", False),
            verification_reason=result.get("verification_reason", ""),
            retry_count=result.get("retry_count", 0),
            input_mode=input_mode,
            grounding_action=result.get("grounding_action", "pass"),
            follow_up_suggestions=result.get("follow_up_suggestions", []),
            parsed_query=query,
            diagnostics=diagnostics,
            voice=voice_block,
            image=image_block,
        )

    except asyncio.TimeoutError:
        logger.error("Agent execution timed out", trace_id=trace_id)
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {settings.REQUEST_TIMEOUT_S}s. trace_id={trace_id}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent error: %s", e, exc_info=True, trace_id=trace_id)
        raise HTTPException(status_code=500, detail=f"Agent processing error: {e}")
    finally:
        _llm_semaphore.release()
        structlog.contextvars.unbind_contextvars("trace_id")


def _validate_upload_size(content: bytes, max_mb: int, file_type: str):
    """Validate upload file size."""
    max_bytes = max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{file_type} file too large ({len(content) / 1024 / 1024:.1f}MB). Max: {max_mb}MB",
        )


def _validate_content_type(
    content_type: str | None, allowed: list[str], file_type: str
):
    """Validate upload content type."""
    if content_type and not any(content_type.startswith(a) for a in allowed):
        raise HTTPException(
            status_code=415,
            detail=f"Invalid {file_type} content type: {content_type}. Allowed: {allowed}",
        )


# =============================================================================
# V1 ROUTER
# =============================================================================
v1 = APIRouter(prefix="/v1", tags=["v1"])


@v1.get("/health", response_model=HealthResponse)
async def v1_health():
    """System health, KG stats, manifest, and enabled modules."""
    stats = _services.get("kg_stats", {})
    classifier = _services.get("classifier")
    return HealthResponse(
        status="ok",
        chunk_count=_services.get("chunk_count", 0),
        kg_entities=stats.get("entities", 0),
        kg_aliases=stats.get("aliases", 0),
        kg_relations=stats.get("relations", 0),
        manifest=_services.get("manifest"),
        enabled_modules={
            "kg": True,
            "reranker": True,
            "translator": True,
            "stt": _services.get("stt") is not None,
            "ffmpeg": _services.get("ffmpeg_available", False),
            "banglaspeech2text": settings.BANGLASPEECH2TEXT_ENABLED,
            "vosk_fallback": settings.VOSK_FALLBACK_ENABLED,
            "vosk_model_ready": _services.get("vosk_model_ready", False),
            "tts": _services.get("tts") is not None,
            "image_classifier": classifier is not None and classifier.is_available
            if classifier
            else False,
            "vlm": False,  # Honest: VLM not implemented
        },
        grounding_mode=settings.GROUNDING_MODE,
    )


@v1.post("/chat", response_model=ChatResponseV1)
async def v1_chat(request: ChatRequest):
    """Process a text-based agricultural query (bilingual). Also handles voice-confirmed re-submission."""
    logger.info("Chat query", query=request.query[:100], input_mode=request.input_mode)
    return await _run_agent(
        request.query,
        input_mode=request.input_mode,
        trace_id=request.trace_id or None,
    )


@v1.post("/chat/voice", response_model=ChatResponseV1)
async def v1_chat_voice(
    audio: UploadFile = File(..., description="Audio file (WAV/MP3)"),
):
    """
    Process a voice-based query via Whisper transcription.

    Confidence gating:
      - If ASR confidence >= threshold → runs RAG and returns answer
      - If ASR confidence < threshold → returns needs_confirmation=true with transcript
    """
    trace_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=trace_id)

    stt = _services["stt"]

    content = await audio.read()
    _validate_upload_size(content, settings.AUDIO_MAX_MB, "Audio")
    _validate_content_type(audio.content_type, ["audio/"], "Audio")

    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    extra_timings: dict[str, float] = {}
    preprocess_warnings: list[str] = []

    try:
        # --- Step 1: Audio preprocessing (always) ---
        preprocess_start = time.perf_counter()
        try:
            from agribot.voice.audio_preprocess import preprocess_audio

            canonical_path, preprocess_info = preprocess_audio(
                tmp_path,
                max_duration_s=settings.VOICE_MAX_DURATION_SECONDS,
            )
            extra_timings["audio_preprocess"] = round(
                (time.perf_counter() - preprocess_start) * 1000, 1
            )
            preprocess_warnings.extend(preprocess_info.get("warnings", []))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.warning("Audio preprocessing failed: %s", e)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Audio decoding failed before transcription. "
                    "Please retry with a clear recording (WAV/WEBM/MP3) and ensure ffmpeg is installed."
                ),
            )

        # --- Step 2: STT transcription (with semaphore, then release before LLM) ---
        stt_start = time.perf_counter()
        try:
            async with asyncio.timeout(30):
                await _stt_semaphore.acquire()
        except asyncio.TimeoutError:
            raise HTTPException(status_code=429, detail="STT busy, try again.")

        try:
            loop = asyncio.get_event_loop()
            stt_result = await loop.run_in_executor(
                None, stt.transcribe, str(canonical_path)
            )

            # If language detection is uncertain, retry with forced Bengali.
            # This improves robustness for short/noisy Bengali queries.
            should_retry_bn = (
                stt_result.get("language") != "bn"
                or "uncertain_language" in stt_result.get("warnings", [])
                or "script_mismatch" in stt_result.get("warnings", [])
            )
            if should_retry_bn:
                stt_bn = await loop.run_in_executor(
                    None,
                    lambda: stt.transcribe(str(canonical_path), language="bn"),
                )
                if stt_bn.get("confidence", 0.0) >= stt_result.get("confidence", 0.0):
                    logger.info(
                        "Using Bengali-forced STT retry",
                        prev_lang=stt_result.get("language", ""),
                        prev_conf=stt_result.get("confidence", 0.0),
                        new_conf=stt_bn.get("confidence", 0.0),
                    )
                    stt_result = stt_bn
        finally:
            _stt_semaphore.release()

        extra_timings["stt"] = round((time.perf_counter() - stt_start) * 1000, 1)

        transcript = stt_result["text"]
        asr_confidence = stt_result["confidence"]
        asr_language = stt_result["language"]
        asr_warnings = stt_result.get("warnings", [])

        logger.info(
            "Voice transcription",
            language=asr_language,
            confidence=asr_confidence,
            text=transcript[:100],
            warnings=asr_warnings,
        )

        # Build voice diagnostics block
        voice_block = VoiceBlock(
            transcript=transcript,
            asr_language=asr_language,
            asr_confidence=asr_confidence,
            asr_warnings=asr_warnings,
        )

        # --- Step 3: Confidence gate ---
        needs_confirmation = (
            asr_confidence < settings.ASR_CONFIDENCE_THRESHOLD
            or "no_speech" in asr_warnings
            or "low_confidence" in asr_warnings
            or "uncertain_language" in asr_warnings
            or "script_mismatch" in asr_warnings
            or "repetitive_transcript" in asr_warnings
            or asr_language not in ("bn", "en")
            or not transcript.strip()
        )

        if needs_confirmation:
            logger.info(
                "Low confidence (%.2f < %.2f), requesting confirmation",
                asr_confidence,
                settings.ASR_CONFIDENCE_THRESHOLD,
            )
            voice_block.needs_confirmation = True
            voice_block.transcript_suspected = transcript
            voice_block.suggested_actions = [
                "Edit the transcript and confirm",
                "Record again with clearer speech",
                "Type your question instead",
            ]

            # Return without running RAG
            return ChatResponseV1(
                answer="",
                input_mode="voice",
                parsed_query=transcript,
                diagnostics=DiagnosticsBlock(
                    trace_id=trace_id,
                    timings_ms=extra_timings,
                    mode_flags=["voice", "needs_confirmation"],
                    warnings=preprocess_warnings + asr_warnings,
                ),
                voice=voice_block,
            )

        # --- Step 4: High confidence → run RAG ---
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        extra_diagnostics = {
            "timings_ms": extra_timings,
            "mode_flags": ["voice", "direct"],
            "warnings": preprocess_warnings + asr_warnings,
        }

        return await _run_agent(
            transcript,
            input_mode="voice",
            extra_diagnostics=extra_diagnostics,
            voice_block=voice_block,
            trace_id=trace_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Voice processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice processing error: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        if "canonical_path" in dir() and canonical_path != Path(tmp_path):
            Path(canonical_path).unlink(missing_ok=True)
        structlog.contextvars.unbind_contextvars("trace_id")


@v1.post("/chat/image", response_model=ChatResponseV1)
async def v1_chat_image(
    image: UploadFile = File(..., description="Crop/plant photo (JPG/PNG)"),
    query: str = Form(default="", description="Optional text query"),
):
    """Process an image-based query (OCR baseline + optional classifier)."""
    trace_id = str(uuid.uuid4())
    extra_timings: dict[str, float] = {}

    content = await image.read()
    _validate_upload_size(content, settings.IMAGE_MAX_MB, "Image")
    _validate_content_type(image.content_type, ["image/"], "Image")

    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Acquire image semaphore
        try:
            async with asyncio.timeout(15):
                await _image_semaphore.acquire()
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=429, detail="Image analysis busy, try again."
            )

        try:
            from agribot.vision.image_processor import get_image_processor

            analysis_start = time.perf_counter()

            processor = get_image_processor(
                vlm_model_path=settings.VLM_MODEL_PATH if settings.VLM_ENABLED else None
            )

            # Run structured analysis with optional classifier
            classifier = _services.get("classifier")
            loop = asyncio.get_event_loop()
            analysis_result = await loop.run_in_executor(
                None,
                lambda: processor.describe_image_structured(
                    tmp_path, classifier=classifier
                ),
            )

            extra_timings["image_analysis"] = round(
                (time.perf_counter() - analysis_start) * 1000, 1
            )
        finally:
            _image_semaphore.release()

        # Build combined query using structured result
        combined_query = analysis_result.build_query_text(query)

        # Build image block
        image_block = ImageBlock(
            pipeline_used=analysis_result.pipeline_used,
            analysis_summary=analysis_result.to_dict(),
            limitations=analysis_result.limitations,
            possible_conditions=[
                c.to_dict() for c in analysis_result.possible_conditions
            ],
        )

        logger.info(
            "Image query",
            pipeline=analysis_result.pipeline_used,
            description=combined_query[:100],
        )

        extra_diagnostics = {
            "timings_ms": extra_timings,
            "mode_flags": ["image", analysis_result.pipeline_used],
            "warnings": analysis_result.quality_flags,
        }

        return await _run_agent(
            combined_query,
            input_mode="image",
            extra_diagnostics=extra_diagnostics,
            image_block=image_block,
            trace_id=trace_id,
        )

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Image processing module not available.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image processing error: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@v1.post("/tts")
async def v1_tts(request: TTSRequest):
    """Synthesize text to speech (WAV audio stream)."""
    tts = _services.get("tts")
    if not tts:
        raise HTTPException(status_code=503, detail="TTS service not available")

    # Fail fast for Bengali when system Bengali voice is unavailable.
    # This avoids pyttsx3 hangs observed on some Windows setups.
    if request.language == "bn" and not tts.has_bengali_voice():
        raise HTTPException(
            status_code=422,
            detail=(
                "Bengali TTS voice is not installed on this system. "
                "Install a Bengali voice pack in Windows Speech settings."
            ),
        )

    try:
        audio_path = tts.save_audio_temp(request.text, language=request.language)
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise HTTPException(status_code=500, detail="TTS audio generation failed")

        def audio_stream():
            with open(audio_path, "rb") as f:
                yield from iter(lambda: f.read(8192), b"")
            audio_path.unlink(missing_ok=True)

        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=tts_output.wav"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("TTS error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")


@v1.get("/kg/stats")
async def v1_kg_stats():
    """Knowledge graph statistics."""
    kg = _services["kg"]
    return kg.get_stats()


@v1.get("/kg/search", response_model=KGSearchResponse)
async def v1_kg_search(q: str):
    """Search the knowledge graph for entities."""
    kg = _services["kg"]
    entities = kg.find_by_partial_alias(q)
    return KGSearchResponse(
        entities=[
            {
                "id": e.id,
                "canonical_bn": e.canonical_bn,
                "canonical_en": e.canonical_en,
                "entity_type": e.entity_type,
                "aliases": [a.alias_text for a in kg.get_aliases(e.id)],
            }
            for e in entities[:20]
        ]
    )


# --- Mount v1 router ---
app.include_router(v1)


# =============================================================================
# LEGACY ENDPOINTS (wrappers → v1, deprecated)
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["legacy"], deprecated=True)
async def health_check():
    """Legacy health endpoint — use /v1/health instead."""
    return await v1_health()


@app.post("/chat", response_model=ChatResponseV1, tags=["legacy"], deprecated=True)
async def chat(request: ChatRequest):
    """Legacy chat endpoint — use /v1/chat instead."""
    return await v1_chat(request)


@app.post(
    "/chat/voice", response_model=ChatResponseV1, tags=["legacy"], deprecated=True
)
async def chat_voice(audio: UploadFile = File(...)):
    """Legacy voice endpoint — use /v1/chat/voice instead."""
    return await v1_chat_voice(audio)


@app.post(
    "/chat/image", response_model=ChatResponseV1, tags=["legacy"], deprecated=True
)
async def chat_image(
    image: UploadFile = File(...),
    query: str = Form(default=""),
):
    """Legacy image endpoint — use /v1/chat/image instead."""
    return await v1_chat_image(image, query)


@app.post("/tts", tags=["legacy"], deprecated=True)
async def text_to_speech(request: TTSRequest):
    """Legacy TTS endpoint — use /v1/tts instead."""
    return await v1_tts(request)


@app.get("/kg/stats", tags=["legacy"], deprecated=True)
async def kg_stats():
    """Legacy KG stats — use /v1/kg/stats instead."""
    return await v1_kg_stats()


@app.get(
    "/kg/search", response_model=KGSearchResponse, tags=["legacy"], deprecated=True
)
async def kg_search(q: str):
    """Legacy KG search — use /v1/kg/search instead."""
    return await v1_kg_search(q)


# =============================================================================
# STATIC FILES — Serve React production build
# =============================================================================

_FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

if _FRONTEND_DIR.exists():
    app.mount(
        "/assets", StaticFiles(directory=str(_FRONTEND_DIR / "assets")), name="assets"
    )

    @app.get("/{full_path:path}", tags=["static"])
    async def serve_spa(full_path: str):
        """SPA catch-all: serve index.html for client-side routing."""
        file_path = _FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_FRONTEND_DIR / "index.html"))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
