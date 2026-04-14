"""
Tests for the Voice I/O modules (STT and TTS).

All tests use mocking — no actual model downloads or audio processing required.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Speech-to-Text Tests
# =============================================================================


class TestSpeechToText:
    """Tests for the SpeechToText class."""

    def test_init_default_params(self):
        """SpeechToText initializes with default parameters."""
        from agribot.voice.stt import SpeechToText

        stt = SpeechToText()
        assert stt.model_size == "base"
        assert stt.device == "auto"
        assert stt.compute_type == "auto"
        assert stt._model is None

    def test_init_custom_params(self):
        """SpeechToText accepts custom model parameters."""
        from agribot.voice.stt import SpeechToText

        stt = SpeechToText(model_size="small", device="cpu", compute_type="int8")
        assert stt.model_size == "small"
        assert stt.device == "cpu"
        assert stt.compute_type == "int8"

    def test_invalid_model_size_raises(self):
        """SpeechToText rejects invalid model sizes."""
        from agribot.voice.stt import SpeechToText

        with pytest.raises(ValueError, match="Unsupported model size"):
            SpeechToText(model_size="nonexistent")

    def test_valid_model_sizes(self):
        """All documented model sizes are accepted."""
        from agribot.voice.stt import SpeechToText

        for size in ("tiny", "base", "small", "medium", "large-v3"):
            stt = SpeechToText(model_size=size)
            assert stt.model_size == size

    def test_transcribe_file_not_found(self):
        """Transcribing a nonexistent file raises FileNotFoundError."""
        from agribot.voice.stt import SpeechToText

        stt = SpeechToText()
        # Mock _ensure_model so we don't actually load Whisper
        stt._model = MagicMock()

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            stt.transcribe("/nonexistent/audio.wav")

    @patch("agribot.voice.stt.SpeechToText._ensure_model")
    def test_transcribe_returns_expected_format(self, mock_ensure):
        """Transcription result has the expected dict structure."""
        from agribot.voice.stt import SpeechToText

        stt = SpeechToText()

        # Create a mock segment
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = " Hello world "
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.0

        # Mock the transcription info
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([mock_segment], mock_info)

        # Create a temp file to transcribe
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            tmp_path = f.name

        try:
            result = stt.transcribe(tmp_path)
            assert "text" in result
            assert "language" in result
            assert "language_probability" in result
            assert "segments" in result
            assert result["text"] == "Hello world"
            assert result["language"] == "en"
            assert result["language_probability"] == 0.95
            assert len(result["segments"]) == 1
            assert result["segments"][0]["start"] == 0.0
            assert result["segments"][0]["end"] == 2.5
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("agribot.voice.stt.SpeechToText._ensure_model")
    def test_transcribe_with_forced_language(self, mock_ensure):
        """Forced language is passed through to the model."""
        from agribot.voice.stt import SpeechToText

        stt = SpeechToText()

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "টেস্ট"
        mock_segment.avg_logprob = -0.1
        mock_segment.no_speech_prob = 0.0

        mock_info = MagicMock()
        mock_info.language = "bn"
        mock_info.language_probability = 0.99

        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([mock_segment], mock_info)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            tmp_path = f.name

        try:
            stt.transcribe(tmp_path, language="bn")
            stt._model.transcribe.assert_called_once()
            call_kwargs = stt._model.transcribe.call_args
            assert call_kwargs[1]["language"] == "bn"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("agribot.voice.stt.SpeechToText._ensure_model")
    def test_structured_return_has_confidence(self, mock_ensure):
        """Transcription result includes confidence and warnings."""
        from agribot.voice.stt import SpeechToText

        stt = SpeechToText()

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.text = "test text"
        mock_segment.avg_logprob = -0.3
        mock_segment.no_speech_prob = 0.1

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([mock_segment], mock_info)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio data")
            tmp_path = f.name

        try:
            result = stt.transcribe(tmp_path)
            # Must have all required keys
            assert "confidence" in result
            assert "warnings" in result
            assert isinstance(result["confidence"], float)
            assert isinstance(result["warnings"], list)
            assert 0.0 <= result["confidence"] <= 1.0
            # With avg_logprob=-0.3, confidence = 1 + (-0.3) = 0.7
            assert abs(result["confidence"] - 0.7) < 0.05
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("agribot.voice.stt.SpeechToText._ensure_model")
    def test_low_confidence_emits_warning(self, mock_ensure):
        """Very low avg_logprob produces low_confidence warning."""
        from agribot.voice.stt import SpeechToText

        stt = SpeechToText()

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.text = "noisy garble"
        mock_segment.avg_logprob = -0.8  # Very low → conf = 0.2
        mock_segment.no_speech_prob = 0.2

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9

        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([mock_segment], mock_info)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake")
            tmp_path = f.name

        try:
            result = stt.transcribe(tmp_path)
            assert result["confidence"] < 0.4
            assert "low_confidence" in result["warnings"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("agribot.voice.stt.SpeechToText._ensure_model")
    def test_no_speech_emits_warning(self, mock_ensure):
        """Empty transcription produces no_speech warning."""
        from agribot.voice.stt import SpeechToText

        stt = SpeechToText()

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.5

        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([], mock_info)  # No segments

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake")
            tmp_path = f.name

        try:
            result = stt.transcribe(tmp_path)
            assert result["text"] == ""
            assert result["confidence"] == 0.0
            assert "no_speech" in result["warnings"]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_confidence_gating_threshold(self):
        """Low confidence below threshold should trigger confirmation."""
        # Test the confidence gating logic (simulated, no API call)
        threshold = 0.6

        high_conf_result = {"text": "clear speech", "confidence": 0.85, "warnings": []}
        low_conf_result = {
            "text": "garbled",
            "confidence": 0.3,
            "warnings": ["low_confidence"],
        }

        # High confidence → should proceed
        needs_confirm_high = (
            high_conf_result["confidence"] < threshold
            or "no_speech" in high_conf_result["warnings"]
            or "low_confidence" in high_conf_result["warnings"]
        )
        assert needs_confirm_high is False

        # Low confidence → should confirm
        needs_confirm_low = (
            low_conf_result["confidence"] < threshold
            or "no_speech" in low_conf_result["warnings"]
            or "low_confidence" in low_conf_result["warnings"]
        )
        assert needs_confirm_low is True


# =============================================================================
# Text-to-Speech Tests
# =============================================================================


class TestTextToSpeech:
    """Tests for the TextToSpeech class."""

    def test_init_default_params(self):
        """TextToSpeech initializes with default parameters."""
        from agribot.voice.tts import TextToSpeech

        tts = TextToSpeech()
        assert tts.rate == 150
        assert tts.bengali_voice_name == ""
        assert tts._engine is None

    def test_init_custom_rate(self):
        """TextToSpeech accepts a custom speaking rate."""
        from agribot.voice.tts import TextToSpeech

        tts = TextToSpeech(rate=200)
        assert tts.rate == 200

    def test_init_custom_bengali_voice(self):
        """TextToSpeech accepts a custom Bengali voice name."""
        from agribot.voice.tts import TextToSpeech

        tts = TextToSpeech(bengali_voice_name="TestBangla")
        assert tts.bengali_voice_name == "TestBangla"

    def test_speak_empty_text_noop(self):
        """Speaking empty text does nothing."""
        from agribot.voice.tts import TextToSpeech

        tts = TextToSpeech()
        tts.speak("")  # Should not raise
        tts.speak("   ")  # Should not raise
        assert tts._engine is None  # Engine never initialized

    def test_save_audio_rejects_empty(self):
        """save_audio raises ValueError for empty text."""
        from agribot.voice.tts import TextToSpeech

        tts = TextToSpeech()
        with pytest.raises(ValueError, match="Cannot synthesize empty text"):
            tts.save_audio("", "/tmp/out.wav")

    def test_list_voices_returns_list(self):
        """list_voices returns a list of voice dicts."""
        from agribot.voice.tts import TextToSpeech

        # Create mock voices
        mock_voice_en = MagicMock()
        mock_voice_en.id = "voice_en_1"
        mock_voice_en.name = "English Voice"
        mock_voice_en.languages = ["en"]

        mock_voice_bn = MagicMock()
        mock_voice_bn.id = "voice_bn_1"
        mock_voice_bn.name = "Bengali Voice"
        mock_voice_bn.languages = ["bn"]

        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = [mock_voice_en, mock_voice_bn]

        # Inject mock pyttsx3 into sys.modules
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = TextToSpeech()
            voices = tts.list_voices()

            assert isinstance(voices, list)
            assert len(voices) == 2
            assert voices[0]["name"] == "English Voice"
            assert voices[1]["name"] == "Bengali Voice"
        finally:
            del sys.modules["pyttsx3"]

    def test_has_bengali_voice_detection(self):
        """has_bengali_voice detects Bengali voice availability."""
        from agribot.voice.tts import TextToSpeech

        mock_voice_en = MagicMock()
        mock_voice_en.id = "voice_en"
        mock_voice_en.name = "English Voice"
        mock_voice_en.languages = ["en"]

        mock_voice_bn = MagicMock()
        mock_voice_bn.id = "voice_bn"
        mock_voice_bn.name = "Bengali Voice"
        mock_voice_bn.languages = ["bn"]

        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = [mock_voice_en, mock_voice_bn]

        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = TextToSpeech()
            assert tts.has_bengali_voice() is True
        finally:
            del sys.modules["pyttsx3"]

    def test_no_bengali_voice(self):
        """has_bengali_voice returns False when no Bengali voice installed."""
        from agribot.voice.tts import TextToSpeech

        mock_voice_en = MagicMock()
        mock_voice_en.id = "voice_en"
        mock_voice_en.name = "English Voice"
        mock_voice_en.languages = ["en"]

        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = [mock_voice_en]

        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine
        sys.modules["pyttsx3"] = mock_pyttsx3

        try:
            tts = TextToSpeech()
            assert tts.has_bengali_voice() is False
        finally:
            del sys.modules["pyttsx3"]


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletons:
    """Tests for the get_stt() and get_tts() singleton factories."""

    def test_get_stt_returns_instance(self):
        """get_stt returns a SpeechToText instance."""
        import agribot.voice.stt as stt_module

        # Reset singleton
        stt_module._stt_instance = None

        instance = stt_module.get_stt(model_size="tiny")
        assert isinstance(instance, stt_module.SpeechToText)
        assert instance.model_size == "tiny"

        # Cleanup
        stt_module._stt_instance = None

    def test_get_stt_returns_same_instance(self):
        """get_stt returns the same instance on subsequent calls."""
        import agribot.voice.stt as stt_module

        stt_module._stt_instance = None

        first = stt_module.get_stt()
        second = stt_module.get_stt()
        assert first is second

        stt_module._stt_instance = None

    def test_get_tts_returns_instance(self):
        """get_tts returns a TextToSpeech instance."""
        import agribot.voice.tts as tts_module

        tts_module._tts_instance = None

        instance = tts_module.get_tts(rate=120)
        assert isinstance(instance, tts_module.TextToSpeech)
        assert instance.rate == 120

        tts_module._tts_instance = None

    def test_get_tts_returns_same_instance(self):
        """get_tts returns the same instance on subsequent calls."""
        import agribot.voice.tts as tts_module

        tts_module._tts_instance = None

        first = tts_module.get_tts()
        second = tts_module.get_tts()
        assert first is second

        tts_module._tts_instance = None
