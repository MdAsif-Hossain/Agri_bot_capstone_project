"""
Speech-to-Text using faster-whisper (CTranslate2 Whisper).

Provides offline, GPU-accelerated speech recognition with automatic
language detection (Bengali / English).

Config-driven: beam_size, VAD, language hint, and confidence threshold
are all read from config.py settings.

Confidence mapping:
    conf = max(0.0, min(1.0, 1.0 + avg_logprob))
    where avg_logprob is the mean of per-segment avg_logprob values
    returned by faster-whisper.  avg_logprob typically ranges from
    -1.0 (random noise) to 0.0 (perfect), so 1 + x maps to [0, 1].
"""

import logging
import json
import re
import wave
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


def _script_stats(text: str) -> dict[str, int]:
    """Return rough script counts for Bengali, Devanagari, and Latin letters."""
    bengali = 0
    devanagari = 0
    latin = 0
    total_letters = 0
    for ch in text:
        cp = ord(ch)
        if 0x0980 <= cp <= 0x09FF:
            bengali += 1
            total_letters += 1
        elif 0x0900 <= cp <= 0x097F:
            devanagari += 1
            total_letters += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
            total_letters += 1
    return {
        "bengali": bengali,
        "devanagari": devanagari,
        "latin": latin,
        "letters": total_letters,
    }


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
        beam_size: int = 5,
        vad_filter: bool = True,
        min_silence_ms: int = 500,
        language_hint: Optional[str] = "bn",
        task: str = "transcribe",
        banglaspeech2text_enabled: bool = False,
        banglaspeech2text_model_id: Optional[str] = "",
        vosk_fallback_enabled: bool = False,
        vosk_bn_model_path: Optional[str] = "",
    ):
        """
        Initialize the STT engine.

        Args:
            model_size: Whisper model size (tiny/base/small/medium/large-v3).
            device: 'cpu', 'cuda', or 'auto' (auto-detect GPU).
            compute_type: 'int8', 'float16', 'float32', or 'auto'.
            beam_size: Beam search width (higher = better quality, slower).
            vad_filter: Enable Voice Activity Detection to skip silence.
            min_silence_ms: Min silence duration for VAD segmentation.
            language_hint: Force language code or None for auto-detect.
            task: 'transcribe' or 'translate'.
        """
        if model_size not in self.SUPPORTED_SIZES:
            raise ValueError(
                f"Unsupported model size '{model_size}'. "
                f"Choose from: {self.SUPPORTED_SIZES}"
            )

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.min_silence_ms = min_silence_ms
        self.language_hint = language_hint
        self.task = task
        self.banglaspeech2text_enabled = banglaspeech2text_enabled
        self.banglaspeech2text_model_id = (banglaspeech2text_model_id or "").strip()
        self.vosk_fallback_enabled = vosk_fallback_enabled
        self.vosk_bn_model_path = vosk_bn_model_path or ""
        self._model = None
        self._bangla_s2t = None
        self._bangla_s2t_backend = ""
        self._bangla_s2t_attempted = False
        self._vosk_model = None
        self._vosk_attempted = False

        logger.info(
            "SpeechToText initialized (model=%s, device=%s, beam=%d, vad=%s, lang=%s, b2t=%s, vosk_fallback=%s)",
            model_size,
            device,
            beam_size,
            vad_filter,
            language_hint or "auto",
            banglaspeech2text_enabled,
            vosk_fallback_enabled,
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

    def _ensure_bangla_s2t(self) -> bool:
        """Lazy-load optional BanglaSpeech2Text backend via official package API."""
        if not self.banglaspeech2text_enabled:
            return False

        if self._bangla_s2t is not None:
            return True

        if self._bangla_s2t_attempted:
            return False
        self._bangla_s2t_attempted = True

        model_key = self.banglaspeech2text_model_id or "base"

        try:
            from banglaspeech2text import Model, available_models

            try:
                import torch

                use_cuda = self.device == "cuda" or (
                    self.device == "auto" and torch.cuda.is_available()
                )
            except Exception:
                use_cuda = False

            device = "cuda:0" if use_cuda else "cpu"
            download_path = (
                Path(__file__).resolve().parents[2] / "models" / "banglaspeech2text"
            )
            download_path.mkdir(parents=True, exist_ok=True)

            model_choices = available_models()
            try:
                selected_model = model_choices[model_key]
            except Exception:
                logger.warning(
                    "BanglaSpeech2Text model '%s' not found in package index; falling back to 'base'",
                    model_key,
                )
                selected_model = model_choices["base"]

            if isinstance(selected_model, list):
                if not selected_model:
                    logger.warning(
                        "BanglaSpeech2Text model group '%s' is empty", model_key
                    )
                    return False
                selected_model = selected_model[0]

            logger.info("Loading BanglaSpeech2Text model '%s' on %s", model_key, device)
            self._bangla_s2t = Model(
                selected_model,
                download_path=str(download_path),
                device=device,
                verbose=False,
            )
            self._bangla_s2t.load()
            self._bangla_s2t_backend = "banglaspeech2text_package"
            logger.info("BanglaSpeech2Text model loaded")
            return True
        except Exception as e:
            logger.warning("Failed to load BanglaSpeech2Text package model: %s", e)

        try:
            from transformers import pipeline

            fallback_model_id = (
                model_key if "/" in model_key else "bangla-speech-processing/BanglaASR"
            )
            logger.info(
                "Loading BanglaSpeech2Text transformers fallback: %s", fallback_model_id
            )
            self._bangla_s2t = pipeline(
                "automatic-speech-recognition",
                model=fallback_model_id,
                device=-1,
            )
            self._bangla_s2t_backend = "transformers_fallback"
            logger.info("BanglaSpeech2Text transformers fallback loaded")
            return True
        except Exception as e:
            logger.warning("Failed to load BanglaSpeech2Text fallback pipeline: %s", e)
            self._bangla_s2t = None
            self._bangla_s2t_backend = ""
            return False

    def _transcribe_bangla_s2t(self, audio_path: Path) -> Optional[dict]:
        """Transcribe using optional BanglaSpeech2Text backend."""
        if not self._ensure_bangla_s2t() or self._bangla_s2t is None:
            return None

        try:
            if self._bangla_s2t_backend == "banglaspeech2text_package":
                out = self._bangla_s2t.recognize(str(audio_path))
            else:
                import numpy as np

                with wave.open(str(audio_path), "rb") as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    num_frames = wf.getnframes()
                    pcm_bytes = wf.readframes(num_frames)

                if sampwidth == 1:
                    audio_array = (
                        np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32)
                        - 128.0
                    ) / 128.0
                elif sampwidth == 2:
                    audio_array = (
                        np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                elif sampwidth == 4:
                    audio_array = (
                        np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.float32)
                        / 2147483648.0
                    )
                else:
                    logger.warning(
                        "BanglaSpeech2Text transcription failed: unsupported WAV sample width: %s",
                        sampwidth,
                    )
                    return None

                if channels > 1:
                    remainder = audio_array.size % channels
                    if remainder:
                        audio_array = audio_array[: audio_array.size - remainder]
                    if audio_array.size == 0:
                        return None
                    audio_array = audio_array.reshape(-1, channels).mean(axis=1)

                out = self._bangla_s2t(
                    {
                        "array": audio_array,
                        "sampling_rate": int(sample_rate),
                    }
                )
            if isinstance(out, dict):
                text = str(out.get("text", "")).strip()
            else:
                text = str(out).strip()

            stats = _script_stats(text)
            letters = max(1, stats["letters"])
            bengali_ratio = stats["bengali"] / letters
            compact_len = len(re.sub(r"\s+", "", text))

            confidence = 0.7 if text else 0.0
            if compact_len <= 4:
                confidence = min(confidence, 0.3)
            if stats["bengali"] == 0 and stats["latin"] > 0:
                confidence = min(confidence, 0.28)
            elif bengali_ratio < 0.35 and stats["latin"] > stats["bengali"]:
                confidence = min(confidence, 0.35)

            if bengali_ratio >= 0.6:
                language = "bn"
                language_probability = 0.98
            elif stats["latin"] > stats["bengali"]:
                language = "en"
                language_probability = 0.35
            else:
                language = "bn"
                language_probability = 0.6

            warnings = _build_warnings(
                full_text=text,
                confidence=confidence,
                segment_list=[],
                language_prob=language_probability,
            )
            if (
                stats["bengali"] == 0
                and stats["latin"] > 0
                and "script_mismatch" not in warnings
            ):
                warnings.append("script_mismatch")
            warnings.append("banglaspeech2text_used")

            return {
                "text": text,
                "language": language,
                "confidence": round(confidence, 3),
                "language_probability": language_probability,
                "segments": [],
                "warnings": warnings,
                "meta": {
                    "decode_language": "bn",
                    "backend": "banglaspeech2text",
                },
            }
        except Exception as e:
            logger.warning("BanglaSpeech2Text transcription failed: %s", e)
            return None

    def _ensure_vosk_model(self) -> bool:
        """Lazy-load Vosk Bengali model for fallback STT."""
        if not self.vosk_fallback_enabled:
            return False

        if self._vosk_model is not None:
            return True

        if self._vosk_attempted:
            return False
        self._vosk_attempted = True

        try:
            from vosk import Model  # type: ignore
        except ImportError:
            logger.warning("Vosk fallback requested but vosk package is not installed")
            return False

        model_path = self.vosk_bn_model_path.strip()
        if not model_path:
            logger.warning(
                "Vosk fallback requested but AGRIBOT_VOSK_BN_MODEL_PATH is empty"
            )
            return False

        path = Path(model_path)
        if not path.exists():
            logger.warning("Vosk Bengali model path not found: %s", path)
            return False

        try:
            logger.info("Loading Vosk Bengali fallback model from %s", path)
            self._vosk_model = Model(str(path))
            logger.info("Vosk Bengali fallback model loaded")
            return True
        except Exception as e:
            logger.warning("Failed to load Vosk Bengali fallback model: %s", e)
            self._vosk_model = None
            return False

    def _transcribe_vosk_bn(self, audio_path: Path) -> Optional[dict]:
        """Fallback Bengali STT using Vosk. Expects mono PCM WAV input."""
        if not self._ensure_vosk_model() or self._vosk_model is None:
            return None

        try:
            import vosk  # type: ignore

            with wave.open(str(audio_path), "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    logger.warning(
                        "Vosk fallback skipped due to incompatible WAV format (channels=%d, sampwidth=%d)",
                        wf.getnchannels(),
                        wf.getsampwidth(),
                    )
                    return None

                sample_rate = wf.getframerate()
                rec = vosk.KaldiRecognizer(self._vosk_model, sample_rate)
                rec.SetWords(False)

                parts: list[str] = []
                while True:
                    data = wf.readframes(4000)
                    if not data:
                        break
                    if rec.AcceptWaveform(data):
                        text = json.loads(rec.Result()).get("text", "").strip()
                        if text:
                            parts.append(text)

                final_text = json.loads(rec.FinalResult()).get("text", "").strip()
                if final_text:
                    parts.append(final_text)

            text = " ".join(parts).strip()
            confidence = 0.55 if text else 0.0
            warnings = _build_warnings(
                full_text=text,
                confidence=confidence,
                segment_list=[],
                language_prob=1.0,
            )
            warnings.append("vosk_fallback_used")

            return {
                "text": text,
                "language": "bn",
                "confidence": round(confidence, 3),
                "language_probability": 1.0,
                "segments": [],
                "warnings": warnings,
                "meta": {
                    "decode_language": "bn",
                    "backend": "vosk",
                },
            }
        except Exception as e:
            logger.warning("Vosk fallback transcription failed: %s", e)
            return None

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        beam_size: int | None = None,
    ) -> dict:
        """
        Transcribe an audio file to text with structured diagnostics.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            language: Force language override (or use instance default).
            beam_size: Beam size override (or use instance default).

        Returns:
            dict with keys:
                - 'text': Full transcribed text
                - 'language': Detected/forced language code
                - 'confidence': Float 0-1 mapped from avg_logprob
                - 'segments': List of segment dicts with start/end/text
                - 'warnings': List of warning strings
        """
        self._ensure_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use instance defaults unless overridden
        effective_lang = language if language is not None else self.language_hint
        effective_beam = beam_size if beam_size is not None else self.beam_size

        logger.info(
            "Transcribing: %s (lang=%s, beam=%d)",
            audio_path.name,
            effective_lang or "auto",
            effective_beam,
        )

        def _to_float(value, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _transcribe_once(
            vad_filter: bool,
            beam: int,
            decode_language: str | None,
            use_prompt: bool,
        ) -> dict:
            transcribe_kwargs = {
                "beam_size": beam,
                "vad_filter": vad_filter,
                "condition_on_previous_text": False,
                "task": self.task,
                "temperature": 0.0,
            }

            if use_prompt:
                transcribe_kwargs["initial_prompt"] = (
                    "কৃষি, ধান, পোকা, সার, রোগ, পাতা, গম, কৃষিবিদ্যা. "
                    "Agriculture, crops, farming, pests."
                )

            if decode_language is not None:
                transcribe_kwargs["language"] = decode_language

            if vad_filter:
                transcribe_kwargs["vad_parameters"] = dict(
                    min_silence_duration_ms=self.min_silence_ms,
                )

            segments_iter, info = self._model.transcribe(
                str(audio_path), **transcribe_kwargs
            )

            segment_list = []
            full_text_parts = []
            logprob_sum = 0.0
            logprob_count = 0

            for seg in segments_iter:
                text = str(getattr(seg, "text", "")).strip()
                if not text:
                    continue

                seg_logprob = _to_float(getattr(seg, "avg_logprob", -1.0), -1.0)
                seg_no_speech = _to_float(getattr(seg, "no_speech_prob", 0.0), 0.0)

                segment_list.append(
                    {
                        "start": _to_float(getattr(seg, "start", 0.0), 0.0),
                        "end": _to_float(getattr(seg, "end", 0.0), 0.0),
                        "text": text,
                        "avg_logprob": seg_logprob,
                        "no_speech_prob": seg_no_speech,
                    }
                )
                full_text_parts.append(text)
                logprob_sum += seg_logprob
                logprob_count += 1

            full_text = " ".join(full_text_parts)

            if logprob_count > 0:
                avg_logprob = logprob_sum / logprob_count
                confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
            else:
                confidence = 0.0

            language_probability = _to_float(
                getattr(info, "language_probability", 0.0), 0.0
            )
            warnings = _build_warnings(
                full_text=full_text,
                confidence=confidence,
                segment_list=segment_list,
                language_prob=language_probability,
            )

            if "repetitive_transcript" in warnings:
                confidence = min(confidence, 0.25)
            if "script_mismatch" in warnings:
                confidence = min(confidence, 0.20)
            if "uncertain_language" in warnings:
                confidence = min(confidence, 0.40)

            return {
                "text": full_text,
                "language": str(getattr(info, "language", decode_language or "")),
                "confidence": round(confidence, 3),
                "language_probability": language_probability,
                "segments": segment_list,
                "warnings": warnings,
                "meta": {
                    "vad_filter": vad_filter,
                    "beam_size": beam,
                    "decode_language": decode_language or "auto",
                    "use_prompt": use_prompt,
                },
            }

        def _quality_score(result: dict) -> float:
            score = float(result.get("confidence", 0.0))
            warnings = set(result.get("warnings", []))
            text = result.get("text", "").strip()
            low_text = text.lower()
            language = str(result.get("language", "")).lower()
            decode_language = str(
                result.get("meta", {}).get("decode_language", "auto")
            ).lower()
            backend = str(result.get("meta", {}).get("backend", "whisper")).lower()

            if result.get("text", "").strip():
                score += 0.2
            if "no_speech" in warnings:
                score -= 0.6
            if "low_confidence" in warnings:
                score -= 0.2
            if "noisy_audio" in warnings:
                score -= 0.15
            if "uncertain_language" in warnings:
                score -= 0.05

            # Prefer meaningful Bengali transcript over symbol-heavy gibberish.
            if text:
                stats = _script_stats(text)
                bengali_chars = stats["bengali"]
                devanagari_chars = stats["devanagari"]
                latin_chars = stats["latin"]
                punctuation_chars = sum(
                    1 for ch in text if not ch.isalnum() and not ch.isspace()
                )

                score += min(0.25, bengali_chars / max(20, len(text)))
                score -= min(0.20, punctuation_chars / max(10, len(text)))
                if latin_chars > bengali_chars * 0.7:
                    score -= 0.08

                # Strongly discourage Hindi/Devanagari output when decoding as Bengali.
                if decode_language == "bn" and devanagari_chars > max(3, bengali_chars):
                    score -= 0.95
                elif devanagari_chars > max(6, bengali_chars * 1.2):
                    score -= 0.60

                # Penalize implausible language IDs for this domain.
                if language not in {"bn", "en"}:
                    score -= 0.35
                if bengali_chars > latin_chars + 3 and language != "bn":
                    score -= 0.35
                if latin_chars > bengali_chars + 3 and language not in {"en", ""}:
                    score -= 0.15

                # Penalize repeated token gibberish (e.g., same short token repeated).
                tokens = [t for t in re.split(r"\s+|,|\.|;|:|\u0964", low_text) if t]
                if len(tokens) >= 3:
                    unique_ratio = len(set(tokens)) / len(tokens)
                    if unique_ratio < 0.5:
                        score -= 0.25
                if len(tokens) >= 6:
                    top_ratio = max(Counter(tokens).values()) / len(tokens)
                    if top_ratio >= 0.45:
                        score -= 0.45

                # Bonus for agricultural terms likely in user queries.
                agri_terms = {
                    "ধান",
                    "পোকা",
                    "রোগ",
                    "সার",
                    "পাতা",
                    "গাছ",
                    "ফসল",
                    "rice",
                    "pest",
                    "disease",
                    "fertilizer",
                    "crop",
                }
                if any(term in low_text for term in agri_terms):
                    score += 0.08

                # Prefer Bengali-specialized backend when script/language look sane.
                if backend == "banglaspeech2text":
                    if language == "bn" and devanagari_chars == 0:
                        score += 0.18
                    if "script_mismatch" in warnings:
                        score -= 0.5

            return score

        primary_result = _transcribe_once(
            vad_filter=self.vad_filter,
            beam=effective_beam,
            decode_language=effective_lang,
            use_prompt=False,
        )
        selected_result = primary_result

        candidates = [primary_result]

        should_retry_no_vad = self.vad_filter and (
            "no_speech" in primary_result["warnings"]
            or "noisy_audio" in primary_result["warnings"]
            or primary_result["confidence"] < 0.45
        )
        if should_retry_no_vad:
            retry_beam = max(int(effective_beam), 8)
            fallback_result = _transcribe_once(
                vad_filter=False,
                beam=retry_beam,
                decode_language=effective_lang,
                use_prompt=False,
            )
            candidates.append(fallback_result)

            if _quality_score(fallback_result) > _quality_score(primary_result):
                logger.info(
                    "Using fallback STT result (no VAD): primary_conf=%.3f, fallback_conf=%.3f",
                    primary_result["confidence"],
                    fallback_result["confidence"],
                )
                selected_result = fallback_result
            else:
                logger.info(
                    "Keeping primary STT result after fallback comparison: primary_conf=%.3f, fallback_conf=%.3f",
                    primary_result["confidence"],
                    fallback_result["confidence"],
                )

        should_retry_auto_lang = effective_lang is not None and (
            "no_speech" in selected_result["warnings"]
            or "low_confidence" in selected_result["warnings"]
            or selected_result["confidence"] < 0.35
        )
        if should_retry_auto_lang:
            auto_result = _transcribe_once(
                vad_filter=False,
                beam=max(int(effective_beam), 8),
                decode_language=None,
                use_prompt=False,
            )
            candidates.append(auto_result)
            if _quality_score(auto_result) > _quality_score(selected_result):
                logger.info(
                    "Using auto-language STT fallback: prev_conf=%.3f, auto_conf=%.3f",
                    selected_result["confidence"],
                    auto_result["confidence"],
                )
                selected_result = auto_result

        # Final attempt with domain prompt only when all prompt-free passes look weak.
        if _quality_score(selected_result) < 0.35:
            prompted_result = _transcribe_once(
                vad_filter=False,
                beam=max(int(effective_beam), 8),
                decode_language=effective_lang,
                use_prompt=True,
            )
            candidates.append(prompted_result)

        # Optional secondary backend for hard Bengali cases.
        should_try_vosk = (
            self.vosk_fallback_enabled
            and (effective_lang in (None, "bn"))
            and (
                "repetitive_transcript" in selected_result.get("warnings", [])
                or "script_mismatch" in selected_result.get("warnings", [])
                or "no_speech" in selected_result.get("warnings", [])
                or selected_result.get("confidence", 0.0) < 0.55
            )
        )
        if should_try_vosk:
            vosk_result = self._transcribe_vosk_bn(audio_path)
            if vosk_result is not None:
                candidates.append(vosk_result)
                if _quality_score(vosk_result) > _quality_score(selected_result):
                    logger.info(
                        "Using Vosk fallback transcript: whisper_conf=%.3f, vosk_conf=%.3f",
                        selected_result.get("confidence", 0.0),
                        vosk_result.get("confidence", 0.0),
                    )
                    selected_result = vosk_result

        # Optional BanglaSpeech2Text backend for Bengali-focused input.
        should_try_bangla_s2t = (
            self.banglaspeech2text_enabled
            and (effective_lang in (None, "bn"))
            and (
                "repetitive_transcript" in selected_result.get("warnings", [])
                or "script_mismatch" in selected_result.get("warnings", [])
                or "no_speech" in selected_result.get("warnings", [])
                or selected_result.get("confidence", 0.0) < 0.72
            )
        )
        if should_try_bangla_s2t:
            b2t_result = self._transcribe_bangla_s2t(audio_path)
            if b2t_result is not None:
                candidates.append(b2t_result)
                if _quality_score(b2t_result) > _quality_score(selected_result):
                    logger.info(
                        "Using BanglaSpeech2Text transcript: prev_conf=%.3f, b2t_conf=%.3f",
                        selected_result.get("confidence", 0.0),
                        b2t_result.get("confidence", 0.0),
                    )
                    selected_result = b2t_result

        # Choose best among all candidates with robust scoring.
        selected_result = max(candidates, key=_quality_score)

        logger.info(
            "Transcription complete: lang=%s (%.1f%%), conf=%.2f, %d chars, warnings=%s",
            selected_result["language"],
            selected_result["language_probability"] * 100,
            selected_result["confidence"],
            len(selected_result["text"]),
            selected_result["warnings"] or "none",
        )

        selected_result.pop("meta", None)
        return selected_result

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


def _build_warnings(
    full_text: str,
    confidence: float,
    segment_list: list[dict],
    language_prob: float,
) -> list[str]:
    """Build list of warning strings based on transcription metrics."""

    def _num(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    warnings: list[str] = []

    if not full_text.strip():
        warnings.append("no_speech")
        return warnings

    if confidence < 0.4:
        warnings.append("low_confidence")
    elif confidence < 0.6:
        warnings.append("moderate_confidence")

    # Check for high no-speech probability in segments
    high_nospeech = sum(
        1 for s in segment_list if _num(s.get("no_speech_prob", 0.0), 0.0) > 0.6
    )
    if high_nospeech > len(segment_list) * 0.5 and segment_list:
        warnings.append("noisy_audio")

    if language_prob < 0.7:
        warnings.append("uncertain_language")

    stats = _script_stats(full_text)
    if stats["devanagari"] > max(5, stats["bengali"] * 1.2):
        warnings.append("script_mismatch")

    # Repeated-token transcript is often ASR hallucination.
    tokens = [t for t in re.split(r"\s+|,|\.|;|:|\u0964", full_text.lower()) if t]
    if len(tokens) >= 6:
        top_ratio = max(Counter(tokens).values()) / len(tokens)
        if top_ratio >= 0.45:
            warnings.append("repetitive_transcript")
            if "low_confidence" not in warnings:
                warnings.append("low_confidence")

    # Character-level repetition detection for cases like "বেবেবেবে..."
    compact = re.sub(r"\s+", "", full_text.lower())
    if len(compact) >= 20:
        # Very low character diversity in a long transcript is suspicious.
        uniq_ratio = len(set(compact)) / len(compact)
        if uniq_ratio < 0.22:
            if "repetitive_transcript" not in warnings:
                warnings.append("repetitive_transcript")
            if "low_confidence" not in warnings:
                warnings.append("low_confidence")

        # Repeated 1-3 character motif over long runs.
        if re.search(r"(.{1,3})\1{8,}", compact):
            if "repetitive_transcript" not in warnings:
                warnings.append("repetitive_transcript")
            if "low_confidence" not in warnings:
                warnings.append("low_confidence")

    return warnings


def get_stt(
    model_size: str = "base",
    device: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
    min_silence_ms: int = 500,
    language_hint: Optional[str] = "bn",
    task: str = "transcribe",
    banglaspeech2text_enabled: bool = False,
    banglaspeech2text_model_id: Optional[str] = "",
    vosk_fallback_enabled: bool = False,
    vosk_bn_model_path: Optional[str] = "",
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
        _stt_instance = SpeechToText(
            model_size=model_size,
            device=device,
            beam_size=beam_size,
            vad_filter=vad_filter,
            min_silence_ms=min_silence_ms,
            language_hint=language_hint,
            task=task,
            banglaspeech2text_enabled=banglaspeech2text_enabled,
            banglaspeech2text_model_id=banglaspeech2text_model_id,
            vosk_fallback_enabled=vosk_fallback_enabled,
            vosk_bn_model_path=vosk_bn_model_path,
        )

    return _stt_instance
