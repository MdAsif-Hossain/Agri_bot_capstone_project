"""
Speech-to-Text using faster-whisper (CTranslate2 Whisper).

Provides offline, GPU-accelerated speech recognition with automatic
language detection (Bengali / English).
"""

import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# Singleton
_stt_instance: "SpeechToText | None" = None
_stt_lock = Lock()


class SpeechToText:
    """
    Offline Speech-to-Text using faster-whisper.

    Supports automatic language detection and Bengali/English transcription.
    Models are downloaded once and cached locally for offline use.
    """

    SUPPORTED_SIZES = ("tiny", "base", "small", "medium", "large-v3")

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        """
        Initialize the STT engine.

        Args:
            model_size: Whisper model size (tiny/base/small/medium/large-v3).
                        'base' is recommended for RTX 3050 balance of speed/accuracy.
            device: 'cpu', 'cuda', or 'auto' (auto-detect GPU).
            compute_type: 'int8', 'float16', 'float32', or 'auto'.
        """
        if model_size not in self.SUPPORTED_SIZES:
            raise ValueError(
                f"Unsupported model size '{model_size}'. "
                f"Choose from: {self.SUPPORTED_SIZES}"
            )

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

        logger.info(
            "SpeechToText initialized (model=%s, device=%s, compute=%s)",
            model_size, device, compute_type,
        )

    def _ensure_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info("Loading faster-whisper model '%s'...", self.model_size)
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("faster-whisper model loaded successfully")

        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install it with: pip install faster-whisper"
            )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        beam_size: int = 5,
    ) -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            language: Force language ('bn' for Bengali, 'en' for English).
                      None = auto-detect.
            beam_size: Beam search width (higher = better quality, slower).

        Returns:
            dict with keys:
                - 'text': Full transcribed text
                - 'language': Detected/forced language code
                - 'language_probability': Confidence of language detection
                - 'segments': List of segment dicts with start/end/text
        """
        self._ensure_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing: %s (lang=%s)", audio_path.name, language or "auto")

        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,  # Voice Activity Detection to skip silence
        )

        # Collect segments
        segment_list = []
        full_text_parts = []

        for seg in segments:
            segment_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts)

        logger.info(
            "Transcription complete: lang=%s (%.1f%%), %d chars",
            info.language,
            info.language_probability * 100,
            len(full_text),
        )

        return {
            "text": full_text,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segment_list,
        }

    def transcribe_numpy(
        self,
        audio_array,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> dict:
        """
        Transcribe from a numpy audio array (e.g., from microphone recording).

        Args:
            audio_array: numpy array of audio samples (float32, mono)
            sample_rate: Sample rate of the audio (default 16kHz for Whisper)
            language: Force language or None for auto-detect

        Returns:
            Same dict format as transcribe()
        """
        self._ensure_model()

        import tempfile
        import numpy as np

        # Ensure float32 mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)

        # Save to temp WAV file (faster-whisper reads files)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_array, sample_rate)
            tmp_path = tmp.name

        try:
            return self.transcribe(tmp_path, language=language)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def get_stt(
    model_size: str = "base",
    device: str = "auto",
) -> SpeechToText:
    """
    Get or create the singleton STT instance.

    Thread-safe lazy initialization.
    """
    global _stt_instance

    if _stt_instance is not None:
        return _stt_instance

    with _stt_lock:
        if _stt_instance is not None:
            return _stt_instance
        _stt_instance = SpeechToText(model_size=model_size, device=device)

    return _stt_instance
