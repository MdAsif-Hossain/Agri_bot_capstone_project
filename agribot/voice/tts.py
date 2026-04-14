"""
Text-to-Speech using pyttsx3 (offline, system TTS voices).

Provides offline Bengali and English speech synthesis using OS-native
TTS engines (SAPI5 on Windows, espeak on Linux).
"""

import logging
import tempfile
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# Singleton
_tts_instance: "TextToSpeech | None" = None
_tts_lock = Lock()


class TextToSpeech:
    """
    Offline Text-to-Speech using pyttsx3.

    Uses system-installed voices. On Windows, Bengali voice packs
    can be installed via Settings → Time & Language → Speech.
    Falls back to default voice if Bengali is unavailable.
    """

    def __init__(self, rate: int = 150, bengali_voice_name: str = ""):
        """
        Initialize the TTS engine.

        Args:
            rate: Speaking rate (words per minute). Default 150.
            bengali_voice_name: Specific Bengali voice name to use.
                                Empty string = auto-detect.
        """
        self.rate = rate
        self.bengali_voice_name = bengali_voice_name
        self._engine = None
        self._voices = {}  # language -> voice_id mapping
        self._init_lock = Lock()

        logger.info("TextToSpeech initialized (rate=%d)", rate)

    def _ensure_engine(self):
        """Lazy-load the TTS engine on first use."""
        if self._engine is not None:
            return

        with self._init_lock:
            if self._engine is not None:
                return

            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)

                # Discover available voices
                voices = self._engine.getProperty("voices")
                self._available_voices = voices

                # Map languages to voice IDs
                default_voice_id = voices[0].id if voices else None

                for voice in voices:
                    voice_name = voice.name.lower()
                    voice_langs = [
                        lang.lower() if isinstance(lang, str) else str(lang).lower()
                        for lang in (voice.languages or [])
                    ]

                    # Detect Bengali voice
                    if (
                        "bengali" in voice_name
                        or "bangla" in voice_name
                        or "bn" in voice_name
                        or any(
                            "bn" in lang or "bengali" in lang for lang in voice_langs
                        )
                    ):
                        self._voices["bn"] = voice.id
                        logger.info("Found Bengali voice: %s", voice.name)

                    # Detect English voice
                    if "english" in voice_name or any(
                        "en" in lang for lang in voice_langs
                    ):
                        if "en" not in self._voices:
                            self._voices["en"] = voice.id

                # If user specified a Bengali voice name, try to find it
                if self.bengali_voice_name:
                    for voice in voices:
                        if self.bengali_voice_name.lower() in voice.name.lower():
                            self._voices["bn"] = voice.id
                            logger.info(
                                "Using user-specified Bengali voice: %s",
                                voice.name,
                            )
                            break

                # Ensure we have at least a default
                if "en" not in self._voices and default_voice_id:
                    self._voices["en"] = default_voice_id

                logger.info(
                    "TTS engine initialized. Available language voices: %s",
                    list(self._voices.keys()),
                )

            except ImportError:
                raise ImportError(
                    "pyttsx3 is not installed. Install it with: pip install pyttsx3"
                )

    def list_voices(self) -> list[dict]:
        """
        List all available system TTS voices.

        Returns:
            List of dicts with 'id', 'name', 'languages' keys.
        """
        self._ensure_engine()
        return [
            {
                "id": v.id,
                "name": v.name,
                "languages": v.languages,
            }
            for v in self._available_voices
        ]

    def has_bengali_voice(self) -> bool:
        """Check if a Bengali voice is available on the system."""
        self._ensure_engine()
        return "bn" in self._voices

    def speak(self, text: str, language: str = "en") -> None:
        """
        Speak text aloud using the system TTS engine.

        Args:
            text: Text to speak.
            language: 'bn' for Bengali, 'en' for English.
                      Falls back to English if requested language unavailable.
        """
        if not text.strip():
            return

        self._ensure_engine()

        # Select voice
        voice_id = self._voices.get(language) or self._voices.get("en")
        if voice_id:
            self._engine.setProperty("voice", voice_id)

        if language == "bn" and "bn" not in self._voices:
            logger.warning(
                "Bengali voice not available. Using default voice. "
                "Install a Bengali TTS voice pack for native pronunciation."
            )

        logger.info("Speaking (%s): %s...", language, text[:60])
        self._engine.say(text)
        self._engine.runAndWait()

    def save_audio(
        self,
        text: str,
        output_path: str | Path,
        language: str = "en",
    ) -> Path:
        """
        Save spoken text to a WAV audio file.

        Args:
            text: Text to synthesize.
            output_path: Where to save the WAV file.
            language: 'bn' or 'en'.

        Returns:
            Path to the saved audio file.
        """
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")

        self._ensure_engine()
        output_path = Path(output_path)

        # Select voice
        voice_id = self._voices.get(language) or self._voices.get("en")
        if voice_id:
            self._engine.setProperty("voice", voice_id)

        logger.info("Saving TTS audio to %s (%s)", output_path, language)
        self._engine.save_to_file(text, str(output_path))
        self._engine.runAndWait()

        return output_path

    def save_audio_temp(self, text: str, language: str = "en") -> Path:
        """
        Save spoken text to a temporary WAV file.

        Useful for Streamlit audio playback.

        Args:
            text: Text to synthesize.
            language: 'bn' or 'en'.

        Returns:
            Path to the temporary WAV file.
        """
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="agribot_tts_"
        )
        tmp.close()
        return self.save_audio(text, tmp.name, language)


def get_tts(rate: int = 150, bengali_voice_name: str = "") -> TextToSpeech:
    """
    Get or create the singleton TTS instance.

    Thread-safe lazy initialization.
    """
    global _tts_instance

    if _tts_instance is not None:
        return _tts_instance

    with _tts_lock:
        if _tts_instance is not None:
            return _tts_instance
        _tts_instance = TextToSpeech(rate=rate, bengali_voice_name=bengali_voice_name)

    return _tts_instance
