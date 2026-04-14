"""
AgriBot Central Configuration
All tunable parameters in one place.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field


class AgriConfig(BaseSettings):
    """Production configuration with sensible defaults for RTX 3050."""

    # --- Paths ---
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent)
    DATA_DIR: Optional[Path] = Field(default=None)
    PDF_DIR: Optional[Path] = Field(default=None)
    INDEX_DIR: Optional[Path] = Field(default=None)
    KG_DB_PATH: Optional[Path] = Field(default=None)
    MODEL_PATH: Optional[Path] = Field(default=None)

    # --- LLM ---
    LLM_N_CTX: int = 4096
    LLM_N_GPU_LAYERS: int = (
        15  # RTX 3050 4GB (Lowering from 20 to avoid OOM latency spikes)
    )
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.1  # Low for factual answers

    # --- Ingestion ---
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 120
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Retrieval ---
    DENSE_TOP_K: int = 5
    SPARSE_TOP_K: int = 5
    RERANK_TOP_N: int = 3
    RERANK_THRESHOLD: float = 0.10
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4

    # --- Agent ---
    MAX_RETRIES: int = 2
    EVIDENCE_GRADE_THRESHOLD: float = 0.5

    # --- Grounding Policy ---
    GROUNDING_MODE: str = "strict"  # "strict" | "lenient"
    ON_VERIFY_FAIL: str = "disclaimer"  # "disclaimer" | "cited_facts_only" | "refuse"

    # --- Voice / STT ---
    WHISPER_MODEL_SIZE: str = (
        "medium"  # Better latency/quality trade-off for RTX 3050 or CPU fallback
    )
    WHISPER_BEAM_SIZE: int = 8  # Beam search width (higher = better, slower)
    WHISPER_VAD_FILTER: bool = (
        False  # Disable VAD by default; short Bangla speech can be over-pruned
    )
    WHISPER_MIN_SILENCE_MS: int = 500  # Min silence for VAD segmentation (ms)
    WHISPER_LANGUAGE_HINT: Optional[str] = (
        "bn"  # Prefer Bengali decoding for Bangla voice input
    )
    WHISPER_TASK: str = "transcribe"  # "transcribe" or "translate"
    BANGLASPEECH2TEXT_ENABLED: bool = True  # Enable Bengali-specialized ASR backend
    BANGLASPEECH2TEXT_MODEL_ID: str = (
        "bangla-speech-processing/BanglaASR"  # Package key/name or HF model ID fallback
    )
    VOSK_FALLBACK_ENABLED: bool = (
        True  # Try Vosk Bengali fallback when Whisper output is unreliable
    )
    VOSK_BN_MODEL_PATH: str = ""  # Path to local Vosk Bengali model directory
    ASR_CONFIDENCE_THRESHOLD: float = 0.6  # Below this → needs_confirmation
    VOICE_MAX_DURATION_SECONDS: int = 60  # Max audio duration after decode
    TTS_RATE: int = 150  # TTS speaking rate (words per minute)
    TTS_BENGALI_VOICE: str = ""  # System Bengali voice name (empty = auto-detect)

    # --- Vision / Image ---
    VLM_ENABLED: bool = False  # Enable offline VLM captioning (placeholder)
    VLM_MODEL_PATH: Optional[str] = None  # Path to VLM GGUF model
    IMAGE_CLASSIFIER_ENABLED: bool = (
        False  # Enable optional classifier-assisted analysis
    )
    IMAGE_CLASSIFIER_MODEL_PATH: Optional[str] = None  # Path to classifier model
    IMAGE_CLASSIFIER_TOP_K: int = 3  # Top-K conditions to return
    IMAGE_CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.3  # Min confidence for conditions

    # --- Concurrency & Limits ---
    MAX_CONCURRENT_LLM: int = 1  # Max concurrent LLM generation requests
    MAX_CONCURRENT_STT: int = 2  # Max concurrent STT transcriptions
    MAX_CONCURRENT_IMAGE_ANALYSIS: int = 2  # Max concurrent image analyses
    REQUEST_TIMEOUT_S: int = 120  # Request timeout in seconds
    IMAGE_MAX_MB: int = 10  # Max image upload size in MB
    AUDIO_MAX_MB: int = 25  # Max audio upload size in MB

    # --- Security ---
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:8000"
    API_KEY: str = ""  # If set, require X-API-Key header

    # --- Noise Filtering ---
    HEADER_FOOTER_FREQ_THRESHOLD: float = 0.5  # Lines appearing in >50% of pages
    TOC_KEYWORDS: list[str] = [
        "table of contents",
        "contents",
        "index",
        "bibliography",
        "references",
        "appendix",
    ]
    MIN_CHUNK_LENGTH: int = 30  # Skip chunks shorter than this

    def model_post_init(self, __context):
        """Resolve relative paths after init."""
        if self.DATA_DIR is None:
            self.DATA_DIR = self.BASE_DIR / "data"
        if self.PDF_DIR is None:
            self.PDF_DIR = self.DATA_DIR / "pdfs"
        if self.INDEX_DIR is None:
            self.INDEX_DIR = self.DATA_DIR / "indexes"
        if self.KG_DB_PATH is None:
            self.KG_DB_PATH = self.DATA_DIR / "knowledge_graph.db"
        if self.MODEL_PATH is None:
            self.MODEL_PATH = (
                self.BASE_DIR / "models" / "qwen2.5-1.5b-instruct-q8_0.gguf"
            )
        if not self.VOSK_BN_MODEL_PATH:
            self.VOSK_BN_MODEL_PATH = str(self.BASE_DIR / "models" / "vosk-bn")

    model_config = ConfigDict(
        env_prefix="AGRIBOT_",
        env_file=".env",
        extra="ignore",
    )


# Singleton config instance
settings = AgriConfig()
