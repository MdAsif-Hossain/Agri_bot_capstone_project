"""
AgriBot — Production FastAPI Backend.

Provides versioned REST API endpoints:
 - /v1/health          — system health + manifest
 - /v1/chat            — text query
 - /v1/chat/voice      — voice query (audio upload)
 - /v1/chat/image      — image query (photo upload)
 - /v1/tts             — text-to-speech synthesis
 - /v1/kg/stats        — knowledge graph statistics
 - /v1/kg/search       — search KG entities

Legacy endpoints (/chat, /health, etc.) remain as thin wrappers.
"""

import sys
import uuid
import time
import asyncio
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from functools import wraps

from fastapi import (
    FastAPI, UploadFile, File, Form,
    HTTPException, Request, Depends, APIRouter,
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
    svc["stt"] = get_stt(model_size=settings.WHISPER_MODEL_SIZE)
    svc["tts"] = get_tts(rate=settings.TTS_RATE, bengali_voice_name=settings.TTS_BENGALI_VOICE)

    # 9. Agent graph (with config-driven retries + grounding policy)
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
    global _services, _llm_semaphore
    logger.info("Starting AgriBot API server...")
    _services = _init_services()
    _llm_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM)
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
    version="1.0.0",
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
    if path in ("/health", "/v1/health", "/docs", "/redoc", "/openapi.json") \
       or path.startswith("/assets") or not path.startswith("/v1/"):
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
# MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Text-based chat request."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query in English or Bengali")


class ChatResponse(BaseModel):
    """Chat response with bilingual answer, evidence, and diagnostics."""
    answer: str = Field(..., description="English answer")
    answer_bn: str = Field(default="", description="Bengali translation")
    citations: list[str] = Field(default_factory=list, description="Source citations")
    kg_entities: list[dict] = Field(default_factory=list, description="Linked KG entities")
    evidence_grade: str = Field(default="N/A", description="Evidence quality grade")
    is_verified: bool = Field(default=False, description="Whether answer passed verification")
    verification_reason: str = Field(default="", description="Verification details")
    retry_count: int = Field(default=0, description="Number of retrieval retries")
    input_mode: str = Field(default="text", description="Input mode used")
    # --- New fields ---
    trace_id: str = Field(default="", description="Request trace ID for correlation")
    timings_ms: dict[str, float] = Field(default_factory=dict, description="Per-node timings in ms")
    grounding_action: str = Field(default="pass", description="Grounding policy action taken")
    follow_up_suggestions: list[str] = Field(default_factory=list, description="Suggested follow-up queries")


class HealthResponse(BaseModel):
    """System health status."""
    status: str
    chunk_count: int
    kg_entities: int
    kg_aliases: int
    kg_relations: int
    # --- New fields ---
    manifest: dict | None = Field(default=None, description="Index manifest summary")
    enabled_modules: dict = Field(default_factory=dict, description="Module status")
    grounding_mode: str = Field(default="strict", description="Current grounding mode")


class KGSearchResponse(BaseModel):
    """Knowledge graph search results."""
    entities: list[dict]


class TTSRequest(BaseModel):
    """Text-to-speech request."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    language: str = Field(default="en", description="Language: 'en' or 'bn'")


# =============================================================================
# HELPER
# =============================================================================

def _build_initial_state(query: str, input_mode: str = "text", trace_id: str = "") -> dict:
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
        # --- New fields ---
        "trace_id": trace_id,
        "timings_ms": {},
        "grounding_action": "pass",
        "follow_up_suggestions": [],
    }


async def _run_agent(query: str, input_mode: str = "text") -> ChatResponse:
    """Run the agent pipeline with concurrency control and timeout."""
    trace_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=trace_id)

    logger.info("Processing request", query=query[:100], input_mode=input_mode)

    # Acquire semaphore with timeout (concurrency limit)
    try:
        acquired = _llm_semaphore.locked()
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

        # Run agent in thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        result = await asyncio.wait_for(
            loop.run_in_executor(None, agent.invoke, initial_state),
            timeout=settings.REQUEST_TIMEOUT_S,
        )

        total_ms = (time.perf_counter() - start) * 1000
        timings = result.get("timings_ms", {})
        timings["total"] = round(total_ms, 1)

        return ChatResponse(
            answer=result.get("answer", "An error occurred."),
            answer_bn=result.get("answer_bn", ""),
            citations=result.get("citations", []),
            kg_entities=result.get("kg_entities", []),
            evidence_grade=result.get("evidence_grade", "N/A"),
            is_verified=result.get("is_verified", False),
            verification_reason=result.get("verification_reason", ""),
            retry_count=result.get("retry_count", 0),
            input_mode=input_mode,
            trace_id=trace_id,
            timings_ms=timings,
            grounding_action=result.get("grounding_action", "pass"),
            follow_up_suggestions=result.get("follow_up_suggestions", []),
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
            detail=f"{file_type} file too large ({len(content)/1024/1024:.1f}MB). Max: {max_mb}MB",
        )


def _validate_content_type(content_type: str | None, allowed: list[str], file_type: str):
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
            "tts": _services.get("tts") is not None,
            "vlm": settings.VLM_ENABLED,
        },
        grounding_mode=settings.GROUNDING_MODE,
    )


@v1.post("/chat", response_model=ChatResponse)
async def v1_chat(request: ChatRequest):
    """Process a text-based agricultural query (bilingual)."""
    logger.info("Chat query", query=request.query[:100])
    return await _run_agent(request.query, input_mode="text")


@v1.post("/chat/voice", response_model=ChatResponse)
async def v1_chat_voice(audio: UploadFile = File(..., description="Audio file (WAV/MP3)")):
    """Process a voice-based query via Whisper transcription."""
    stt = _services["stt"]

    content = await audio.read()
    _validate_upload_size(content, settings.AUDIO_MAX_MB, "Audio")
    _validate_content_type(audio.content_type, ["audio/"], "Audio")

    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = stt.transcribe(tmp_path)
        query = result["text"]
        logger.info("Voice transcription", language=result["language"], text=query[:100])

        if not query.strip():
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        return await _run_agent(query, input_mode="voice")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Voice processing error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice processing error: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@v1.post("/chat/image", response_model=ChatResponse)
async def v1_chat_image(
    image: UploadFile = File(..., description="Crop/plant photo (JPG/PNG)"),
    query: str = Form(default="", description="Optional text query"),
):
    """Process an image-based query (OCR + heuristic symptom analysis)."""
    content = await image.read()
    _validate_upload_size(content, settings.IMAGE_MAX_MB, "Image")
    _validate_content_type(image.content_type, ["image/"], "Image")

    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from agribot.vision.image_processor import get_image_processor
        processor = get_image_processor(
            vlm_model_path=settings.VLM_MODEL_PATH if settings.VLM_ENABLED else None
        )
        image_description = processor.describe_image(tmp_path)

        if query.strip():
            combined_query = f"{query}\n\nImage analysis: {image_description}"
        else:
            combined_query = f"Based on this crop image: {image_description}"

        logger.info("Image query", description=combined_query[:100])
        return await _run_agent(combined_query, input_mode="image")

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Image processing module not available.",
        )
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


@app.post("/chat", response_model=ChatResponse, tags=["legacy"], deprecated=True)
async def chat(request: ChatRequest):
    """Legacy chat endpoint — use /v1/chat instead."""
    return await v1_chat(request)


@app.post("/chat/voice", response_model=ChatResponse, tags=["legacy"], deprecated=True)
async def chat_voice(audio: UploadFile = File(...)):
    """Legacy voice endpoint — use /v1/chat/voice instead."""
    return await v1_chat_voice(audio)


@app.post("/chat/image", response_model=ChatResponse, tags=["legacy"], deprecated=True)
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


@app.get("/kg/search", response_model=KGSearchResponse, tags=["legacy"], deprecated=True)
async def kg_search(q: str):
    """Legacy KG search — use /v1/kg/search instead."""
    return await v1_kg_search(q)


# =============================================================================
# STATIC FILES — Serve React production build
# =============================================================================

_FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

if _FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIR / "assets")), name="assets")

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
