"""
Audio preprocessing for the AgriBot voice pipeline.

Canonicalizes uploaded audio to: WAV, mono, 16kHz, peak-normalized.
Two paths:
  1. ffmpeg (preferred): handles any format → canonical WAV
  2. Fallback: Python stdlib wave+audioop for WAV-only input

Also enforces maximum duration.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_audio(
    input_path: str | Path,
    max_duration_s: int = 60,
    target_sr: int = 16000,
) -> tuple[Path, dict]:
    """
    Canonicalize audio to mono 16kHz WAV.

    Args:
        input_path: Path to the uploaded audio file.
        max_duration_s: Maximum allowed duration in seconds.
        target_sr: Target sample rate (default 16000 for Whisper).

    Returns:
        Tuple of (path_to_canonical_wav, info_dict).
        info_dict keys: duration_s, sample_rate, channels, method, warnings.

    Raises:
        ValueError: If audio exceeds max duration or format not supported.
        FileNotFoundError: If input file doesn't exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    warnings: list[str] = []

    # Try ffmpeg first, then stdlib fallback
    if _ffmpeg_available():
        out_path, info = _preprocess_ffmpeg(input_path, target_sr)
        info["method"] = "ffmpeg"
    else:
        # Fallback: WAV only
        suffix = input_path.suffix.lower()
        if suffix not in (".wav", ".wave"):
            raise ValueError(
                f"Audio format '{suffix}' requires ffmpeg for conversion, "
                "but ffmpeg is not available. Please upload a WAV file, "
                "or install ffmpeg: https://ffmpeg.org/download.html"
            )
        out_path, info = _preprocess_stdlib(input_path, target_sr)
        info["method"] = "stdlib"
        warnings.append("ffmpeg_unavailable")

    # Enforce max duration
    if info["duration_s"] > max_duration_s:
        # Clean up temp file
        out_path.unlink(missing_ok=True)
        raise ValueError(
            f"Audio too long: {info['duration_s']:.1f}s "
            f"(max {max_duration_s}s). Please record a shorter message."
        )

    if info["duration_s"] < 0.3:
        warnings.append("short_audio")

    info["warnings"] = warnings
    return out_path, info


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return _ffmpeg_available()


# ---------------------------------------------------------------------------
# ffmpeg path
# ---------------------------------------------------------------------------

_ffmpeg_cache: Optional[bool] = None
_ffmpeg_bin: Optional[str] = None


def _resolve_ffmpeg_binary() -> Optional[str]:
    """Resolve an ffmpeg binary path from env, PATH, or common winget location."""
    # 1) Explicit override
    env_path = os.environ.get("AGRIBOT_FFMPEG_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path

    # 2) PATH lookup
    in_path = shutil.which("ffmpeg")
    if in_path:
        return in_path

    # 3) Common winget install location (Windows)
    winget_root = (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    )
    if winget_root.exists():
        candidates = sorted(
            winget_root.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"),
            key=lambda p: str(p),
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    return None


def _ffmpeg_available() -> bool:
    """Check and cache ffmpeg availability."""
    global _ffmpeg_cache, _ffmpeg_bin
    if _ffmpeg_cache is not None:
        return _ffmpeg_cache

    _ffmpeg_bin = _resolve_ffmpeg_binary()
    _ffmpeg_cache = _ffmpeg_bin is not None
    if not _ffmpeg_cache:
        logger.warning("ffmpeg not found; audio preprocessing limited to WAV input")
    else:
        logger.info("ffmpeg detected at: %s", _ffmpeg_bin)
    return _ffmpeg_cache


def _preprocess_ffmpeg(input_path: Path, target_sr: int) -> tuple[Path, dict]:
    """
    Use ffmpeg to convert any audio to canonical mono 16kHz WAV.

    Pipeline: input → mono → resample → peak normalize → WAV output
    """
    out_fd, out_path_str = tempfile.mkstemp(suffix=".wav", prefix="agribot_pp_")
    # Close the fd; ffmpeg will write to the path
    import os

    os.close(out_fd)
    out_path = Path(out_path_str)

    ffmpeg_bin = _resolve_ffmpeg_binary()
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",  # mono
        "-ar",
        str(target_sr),  # resample
        "-sample_fmt",
        "s16",  # 16-bit PCM
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",  # EBU R128 normalize
        "-f",
        "wav",
        str(out_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(
                "ffmpeg error: %s", result.stderr.decode(errors="replace")[-500:]
            )
            raise RuntimeError(
                f"ffmpeg preprocessing failed (code {result.returncode})"
            )
    except subprocess.TimeoutExpired:
        out_path.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg audio preprocessing timed out (30s)")

    # Read duration from output WAV
    info = _wav_info(out_path)
    return out_path, info


# ---------------------------------------------------------------------------
# stdlib fallback (WAV only)
# ---------------------------------------------------------------------------


def _preprocess_stdlib(input_path: Path, target_sr: int) -> tuple[Path, dict]:
    """
    Pure-Python WAV preprocessing using wave + audioop stdlib modules.

    Handles: mono conversion, basic resampling, peak normalization.
    Only supports WAV input.
    """
    import audioop

    with wave.open(str(input_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # Convert to mono if stereo
    if n_channels > 1:
        raw_data = audioop.tomono(raw_data, sampwidth, 0.5, 0.5)

    # Convert to 16-bit if not already
    if sampwidth != 2:
        raw_data = audioop.lin2lin(raw_data, sampwidth, 2)
        sampwidth = 2

    # Resample to target_sr if needed
    if framerate != target_sr:
        raw_data, _ = audioop.ratecv(raw_data, sampwidth, 1, framerate, target_sr, None)
        framerate = target_sr

    # Peak normalization (scale to ~90% of max to avoid clipping)
    max_val = audioop.max(raw_data, sampwidth)
    if max_val > 0:
        # Target peak: 90% of max representable value
        target_peak = int(32767 * 0.9)
        factor = target_peak / max_val
        if factor < 0.5 or factor > 5.0:
            # Only normalize if significantly off
            raw_data = audioop.mul(raw_data, sampwidth, min(factor, 5.0))

    # Write canonical WAV
    out_fd, out_path_str = tempfile.mkstemp(suffix=".wav", prefix="agribot_pp_")
    import os

    os.close(out_fd)
    out_path = Path(out_path_str)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(raw_data)

    n_samples = len(raw_data) // 2  # 16-bit = 2 bytes per sample
    duration_s = n_samples / target_sr

    return out_path, {
        "duration_s": round(duration_s, 2),
        "sample_rate": target_sr,
        "channels": 1,
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _wav_info(wav_path: Path) -> dict:
    """Read basic info from a WAV file."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration_s = n_frames / framerate if framerate > 0 else 0
            return {
                "duration_s": round(duration_s, 2),
                "sample_rate": framerate,
                "channels": n_channels,
            }
    except Exception:
        return {"duration_s": 0, "sample_rate": 0, "channels": 0}
