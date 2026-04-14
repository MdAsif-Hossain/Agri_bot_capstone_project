"""
Tests for audio preprocessing pipeline.

All tests use mocks — no ffmpeg or actual audio files required.
"""

import io
import wave
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers: Create minimal WAV files for testing
# ---------------------------------------------------------------------------


def _make_wav_bytes(
    n_channels: int = 1,
    sample_rate: int = 16000,
    sampwidth: int = 2,
    duration_s: float = 1.0,
) -> bytes:
    """Create a minimal WAV file as bytes."""
    n_frames = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        # Write silence (zeros)
        wf.writeframes(b"\x00" * (n_frames * n_channels * sampwidth))
    return buf.getvalue()


def _write_wav_file(path: Path, **kwargs) -> Path:
    """Write a WAV file to disk and return the path."""
    data = _make_wav_bytes(**kwargs)
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreprocessStdlib:
    """Test the stdlib (wave+audioop) fallback path."""

    @patch("agribot.voice.audio_preprocess._ffmpeg_available", return_value=False)
    def test_mono_16k_output(self, mock_ffmpeg, tmp_path):
        """Preprocessing produces mono 16kHz WAV."""
        from agribot.voice.audio_preprocess import preprocess_audio

        # Create stereo 44.1kHz test WAV
        src = tmp_path / "test_stereo.wav"
        _write_wav_file(src, n_channels=2, sample_rate=44100, duration_s=1.0)

        out_path, info = preprocess_audio(str(src), max_duration_s=60)
        try:
            assert out_path.exists()
            assert info["sample_rate"] == 16000
            assert info["channels"] == 1
            assert info["method"] == "stdlib"

            # Verify the output WAV properties
            with wave.open(str(out_path), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getframerate() == 16000
                assert wf.getsampwidth() == 2
        finally:
            out_path.unlink(missing_ok=True)

    @patch("agribot.voice.audio_preprocess._ffmpeg_available", return_value=False)
    def test_already_canonical_wav(self, mock_ffmpeg, tmp_path):
        """Already-canonical mono 16kHz WAV should still work."""
        from agribot.voice.audio_preprocess import preprocess_audio

        src = tmp_path / "canonical.wav"
        _write_wav_file(src, n_channels=1, sample_rate=16000, duration_s=2.0)

        out_path, info = preprocess_audio(str(src))
        try:
            assert abs(info["duration_s"] - 2.0) < 0.5
            assert info["sample_rate"] == 16000
        finally:
            out_path.unlink(missing_ok=True)

    @patch("agribot.voice.audio_preprocess._ffmpeg_available", return_value=False)
    def test_duration_enforcement(self, mock_ffmpeg, tmp_path):
        """Audio exceeding max duration should raise ValueError."""
        from agribot.voice.audio_preprocess import preprocess_audio

        src = tmp_path / "long.wav"
        _write_wav_file(src, duration_s=5.0)

        with pytest.raises(ValueError, match="too long"):
            preprocess_audio(str(src), max_duration_s=2)

    @patch("agribot.voice.audio_preprocess._ffmpeg_available", return_value=False)
    def test_non_wav_without_ffmpeg_raises(self, mock_ffmpeg, tmp_path):
        """Non-WAV files without ffmpeg should raise ValueError."""
        from agribot.voice.audio_preprocess import preprocess_audio

        src = tmp_path / "test.mp3"
        src.write_bytes(b"fake mp3 data")

        with pytest.raises(ValueError, match="ffmpeg"):
            preprocess_audio(str(src))

    def test_file_not_found(self):
        """Non-existent file should raise FileNotFoundError."""
        from agribot.voice.audio_preprocess import preprocess_audio

        with pytest.raises(FileNotFoundError):
            preprocess_audio("/nonexistent/path.wav")

    @patch("agribot.voice.audio_preprocess._ffmpeg_available", return_value=False)
    def test_short_audio_warning(self, mock_ffmpeg, tmp_path):
        """Very short audio should produce short_audio warning."""
        from agribot.voice.audio_preprocess import preprocess_audio

        src = tmp_path / "short.wav"
        _write_wav_file(src, duration_s=0.1)

        out_path, info = preprocess_audio(str(src))
        try:
            assert "short_audio" in info.get("warnings", [])
        finally:
            out_path.unlink(missing_ok=True)


class TestPreprocessFfmpeg:
    """Test the ffmpeg path (with mocked subprocess)."""

    @patch("agribot.voice.audio_preprocess._ffmpeg_available", return_value=True)
    @patch("subprocess.run")
    def test_ffmpeg_called_correctly(self, mock_run, mock_ffmpeg, tmp_path):
        """ffmpeg is called with correct arguments."""
        from agribot.voice.audio_preprocess import preprocess_audio

        src = tmp_path / "input.wav"
        _write_wav_file(src, duration_s=3.0)

        # Mock ffmpeg success; write a valid WAV to stdout path
        def side_effect(cmd, **kwargs):
            # Write a canonical WAV to the output path (last arg)
            out = Path(cmd[-1])
            _write_wav_file(out, n_channels=1, sample_rate=16000, duration_s=3.0)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        out_path, info = preprocess_audio(str(src))
        try:
            assert info["method"] == "ffmpeg"
            assert mock_run.called
            # Verify ffmpeg was called with -ac 1 and -ar 16000
            cmd_args = mock_run.call_args[0][0]
            assert "-ac" in cmd_args
            assert "1" in cmd_args
            assert "-ar" in cmd_args
            assert "16000" in cmd_args
        finally:
            out_path.unlink(missing_ok=True)


class TestCheckFfmpeg:
    """Test ffmpeg availability detection."""

    @patch("shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("os.environ.get", return_value="")
    def test_ffmpeg_found(self, mock_env, mock_which):
        from agribot.voice.audio_preprocess import _ffmpeg_available

        # Reset cache
        import agribot.voice.audio_preprocess as ap

        ap._ffmpeg_cache = None

        assert _ffmpeg_available() is True

    @patch("shutil.which", return_value=None)
    @patch("agribot.voice.audio_preprocess.Path.exists", return_value=False)
    @patch("os.environ.get", return_value="")
    def test_ffmpeg_not_found(self, mock_env, mock_exists, mock_which):
        from agribot.voice.audio_preprocess import _ffmpeg_available

        import agribot.voice.audio_preprocess as ap

        ap._ffmpeg_cache = None

        assert _ffmpeg_available() is False
