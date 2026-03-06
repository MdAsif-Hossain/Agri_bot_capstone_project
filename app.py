"""
AgriBot — Production Streamlit UI.

Chat-based agricultural advisory with:
- Bilingual output (English + Bengali)
- Voice input (Whisper ASR) and voice output (TTS)
- Source citations with expand/collapse
- Evidence transparency panel
- KG stats and system status in sidebar
"""

import sys
import logging
from pathlib import Path

import streamlit as st

# --- Ensure project root is on path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
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

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("agribot.ui")


# =============================================================================
# SERVICE INITIALIZATION (cached)
# =============================================================================

@st.cache_resource
def init_services():
    """Load all models and build the agent pipeline. Cached across reruns."""
    status = st.empty()

    # 1. LLM
    status.text("🤖 Loading LLM model...")
    llm = get_llm(
        model_path=str(settings.MODEL_PATH),
        n_ctx=settings.LLM_N_CTX,
        n_gpu_layers=settings.LLM_N_GPU_LAYERS,
    )

    # 2. Indexes
    status.text("📚 Loading document indexes...")
    if not settings.INDEX_DIR.exists():
        status.error(
            f"❌ Index directory not found: {settings.INDEX_DIR}\n\n"
            "Run `python ingest.py` first to build the indexes."
        )
        st.stop()

    index_bundle = IndexBundle.load(settings.INDEX_DIR)

    # 3. Embedding model (for queries)
    status.text("🧠 Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

    # 4. Retriever
    retriever = HybridRetriever(
        index_bundle=index_bundle,
        embedding_model=embedding_model,
        dense_top_k=settings.DENSE_TOP_K,
        sparse_top_k=settings.SPARSE_TOP_K,
        dense_weight=settings.DENSE_WEIGHT,
        sparse_weight=settings.SPARSE_WEIGHT,
    )

    # 5. Reranker
    status.text("⚖️ Loading reranker...")
    reranker = Reranker(
        threshold=settings.RERANK_THRESHOLD,
        top_n=settings.RERANK_TOP_N,
    )

    # 6. Knowledge Graph
    status.text("🌿 Loading knowledge graph...")
    kg = KnowledgeGraph(settings.KG_DB_PATH)
    seed_knowledge_graph(kg)
    entity_linker = EntityLinker(kg)

    # 7. BanglaT5 Translator
    status.text("🌐 Loading BanglaT5 translator...")
    translator = get_translator(device="cpu")

    # 8. Voice services (lazy-loaded, won't download models until first use)
    status.text("🎤 Initializing voice services...")
    stt = get_stt(model_size=settings.WHISPER_MODEL_SIZE)
    tts = get_tts(rate=settings.TTS_RATE, bengali_voice_name=settings.TTS_BENGALI_VOICE)

    # 9. Agent
    status.text("🔧 Building agent pipeline...")
    agent = build_agent_graph(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        entity_linker=entity_linker,
        translator=translator,
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    kg_stats = kg.get_stats()
    chunk_count = len(index_bundle.chunks)

    status.empty()
    return agent, kg_stats, chunk_count, stt, tts


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="🌾 AgriBot — Agricultural Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 12px;
    }
    .citation-box {
        background-color: #f0f7f0;
        border-left: 4px solid #2e7d32;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85em;
    }
    .evidence-header {
        color: #2e7d32;
        font-weight: bold;
        font-size: 0.9em;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: bold;
    }
    .badge-ok { background: #e8f5e9; color: #2e7d32; }
    .badge-warn { background: #fff3e0; color: #e65100; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD SERVICES
# =============================================================================
agent, kg_stats, chunk_count, stt, tts = init_services()


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🌾 AgriBot")
    st.markdown("*Offline Agricultural Advisory System*")
    st.divider()

    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📄 Chunks", chunk_count)
        st.metric("🔗 KG Entities", kg_stats["entities"])
    with col2:
        st.metric("📝 Aliases", kg_stats["aliases"])
        st.metric("🔀 Relations", kg_stats["relations"])

    st.divider()

    st.markdown("### ⚙️ About")
    st.markdown("""
    **AgriBot** uses:
    - 🔍 Hybrid retrieval (FAISS + BM25)
    - ⚖️ Cross-encoder reranking
    - 🌿 Dialect Knowledge Graph
    - 🔄 Self-correcting agent loop
    - ✅ Answer verification
    - 🎤 Voice input (Whisper ASR)
    - 🔊 Voice output (TTS)

    Model: `qwen3b.gguf` (RTX 3050)
    """)

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# CHAT INTERFACE
# =============================================================================
st.title("🌾 AgriBot — Agricultural Advisory")
st.caption("Ask questions about crops, diseases, pests, fertilizers, and more. Answers are cited from FAO/IRRI manuals.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_evidence_panel(data: dict) -> None:
    """Render the evidence/citations panel under an answer."""
    with st.expander("📋 Evidence & Citations", expanded=False):
        # Citations
        citations = data.get("citations", [])
        if citations:
            st.markdown("**Sources cited:**")
            for cit in citations:
                st.markdown(f'<div class="citation-box">📄 {cit}</div>', unsafe_allow_html=True)

        # KG entities
        kg_entities = data.get("kg_entities", [])
        if kg_entities:
            st.markdown("**Knowledge Graph entities linked:**")
            for ent in kg_entities:
                st.markdown(f"- **{ent['en']}** ({ent['bn']}) — _{ent['type']}_")

        # Agent trace
        col1, col2, col3 = st.columns(3)
        with col1:
            grade = data.get("evidence_grade", "N/A")
            color = "badge-ok" if grade == "SUFFICIENT" else "badge-warn"
            st.markdown(f'Evidence: <span class="status-badge {color}">{grade}</span>', unsafe_allow_html=True)
        with col2:
            verified = "✅ Yes" if data.get("is_verified") else "⚠️ No"
            st.markdown(f"Verified: {verified}")
        with col3:
            retries = data.get("retry_count", 0)
            st.markdown(f"Retries: {retries}")


# Display existing messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("answer_bn"):
            tab_en, tab_bn = st.tabs(["🇬🇧 English", "🇧🇩 বাংলা"])
            with tab_en:
                st.markdown(msg["content"])
            with tab_bn:
                st.markdown(msg["answer_bn"])
        else:
            st.markdown(msg["content"])

        # Show evidence panel for assistant messages
        if msg["role"] == "assistant" and msg.get("evidence_data"):
            _render_evidence_panel(msg["evidence_data"])

        # Voice output button for assistant messages
        if msg["role"] == "assistant":
            col_en_voice, col_bn_voice, _ = st.columns([1, 1, 4])
            with col_en_voice:
                if st.button("🔊 Read EN", key=f"tts_en_{idx}"):
                    with st.spinner("Speaking..."):
                        try:
                            audio_path = tts.save_audio_temp(msg["content"], language="en")
                            st.audio(str(audio_path), format="audio/wav")
                        except Exception as e:
                            st.warning(f"TTS error: {e}")
            with col_bn_voice:
                if msg.get("answer_bn") and st.button("🔊 Read BN", key=f"tts_bn_{idx}"):
                    with st.spinner("Speaking..."):
                        try:
                            audio_path = tts.save_audio_temp(msg["answer_bn"], language="bn")
                            st.audio(str(audio_path), format="audio/wav")
                        except Exception as e:
                            st.warning(f"TTS error: {e}")


# --- Voice Input ---
st.markdown("---")
voice_col, text_col = st.columns([1, 5])
with voice_col:
    st.markdown("**🎤 Voice Input**")
    audio_data = st.audio_input("Record your question", key="voice_input")

user_query = None
input_mode = "text"

# Process voice input
if audio_data is not None and "last_audio" not in st.session_state:
    st.session_state.last_audio = True
    with st.spinner("🎤 Transcribing your voice..."):
        try:
            import tempfile
            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data.getvalue())
                tmp_path = tmp.name

            result = stt.transcribe(tmp_path)
            user_query = result["text"]
            detected_lang = result["language"]
            input_mode = "voice"

            st.success(f"🎤 Transcribed ({detected_lang}): {user_query}")

            # Clean up temp file
            import os
            os.unlink(tmp_path)
        except Exception as e:
            st.error(f"❌ Voice transcription failed: {e}")
            logger.error("STT error: %s", e, exc_info=True)
else:
    # Reset audio state when no audio present
    if audio_data is None and "last_audio" in st.session_state:
        del st.session_state.last_audio

# --- Text Input ---
if text_query := st.chat_input("Ask about crops, diseases, pests, fertilizers..."):
    user_query = text_query
    input_mode = "text"

if user_query:
    # Display user message
    prefix = "🎤 " if input_mode == "voice" else ""
    st.session_state.messages.append({"role": "user", "content": f"{prefix}{user_query}"})
    with st.chat_message("user"):
        st.markdown(f"{prefix}{user_query}")

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and reasoning..."):
            # Build initial state
            initial_state = {
                "query_original": user_query,
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
            }

            try:
                result = agent.invoke(initial_state)

                answer = result.get("answer", "An error occurred.")
                answer_bn = result.get("answer_bn", "")
                evidence_data = {
                    "citations": result.get("citations", []),
                    "kg_entities": result.get("kg_entities", []),
                    "evidence_grade": result.get("evidence_grade", "N/A"),
                    "is_verified": result.get("is_verified", False),
                    "retry_count": result.get("retry_count", 0),
                }

            except Exception as e:
                logger.error("Agent error: %s", e, exc_info=True)
                answer = f"⚠️ An error occurred while processing your query: {str(e)}"
                answer_bn = ""
                evidence_data = {}

        # Display bilingual answer in tabs
        tab_en, tab_bn = st.tabs(["🇬🇧 English", "🇧🇩 বাংলা"])
        with tab_en:
            st.markdown(answer)
        with tab_bn:
            if answer_bn:
                st.markdown(answer_bn)
            else:
                st.info("Bengali translation not available.")

        # Show evidence panel
        if evidence_data:
            _render_evidence_panel(evidence_data)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "answer_bn": answer_bn,
            "evidence_data": evidence_data,
        })
