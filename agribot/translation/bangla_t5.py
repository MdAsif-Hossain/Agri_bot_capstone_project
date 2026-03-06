"""
BUET BanglaT5 Translator.

Uses csebuetnlp/banglat5_nmt_en_bn for English→Bengali translation
and csebuetnlp/banglat5_nmt_bn_en for Bengali→English translation.
Includes the required bnunicodenormalizer preprocessing as recommended
by the BUET CSE NLP Group for optimal results.
"""

import logging
import re
from threading import Lock

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# Singleton instances
_translator: "BanglaTranslator | None" = None
_translator_lock = Lock()


class BanglaTranslator:
    """
    Bidirectional English↔Bengali translator using BUET BanglaT5.

    Models:
    - csebuetnlp/banglat5_nmt_en_bn (EN→BN, 247M params)
    - csebuetnlp/banglat5_nmt_bn_en (BN→EN, 247M params)
    """

    EN_TO_BN_MODEL = "csebuetnlp/banglat5_nmt_en_bn"
    BN_TO_EN_MODEL = "csebuetnlp/banglat5_nmt_bn_en"

    def __init__(self, device: str = "cpu"):
        """
        Initialize both translation directions.

        Args:
            device: Device to run on ('cpu' or 'cuda').
        """
        self.device = device

        # EN→BN model
        logger.info("Loading BanglaT5 EN→BN: %s", self.EN_TO_BN_MODEL)
        self.en_bn_tokenizer = AutoTokenizer.from_pretrained(self.EN_TO_BN_MODEL)
        self.en_bn_model = AutoModelForSeq2SeqLM.from_pretrained(self.EN_TO_BN_MODEL)
        self.en_bn_model.to(device)
        self.en_bn_model.eval()

        # BN→EN model
        logger.info("Loading BanglaT5 BN→EN: %s", self.BN_TO_EN_MODEL)
        self.bn_en_tokenizer = AutoTokenizer.from_pretrained(self.BN_TO_EN_MODEL)
        self.bn_en_model = AutoModelForSeq2SeqLM.from_pretrained(self.BN_TO_EN_MODEL)
        self.bn_en_model.to(device)
        self.bn_en_model.eval()

        logger.info("BanglaT5 translators loaded successfully")

        # Try to load normalizer
        self._normalizer = None
        try:
            from bnunicodenormalizer import Normalizer
            self._normalizer = Normalizer()
            logger.info("BN Unicode normalizer loaded")
        except ImportError:
            logger.warning("bnunicodenormalizer not installed; skipping normalization")

    def _normalize_bn(self, text: str) -> str:
        """Apply Bengali unicode normalization if available."""
        if self._normalizer is None:
            return text
        try:
            result = self._normalizer(text)
            return result["normalized"] if isinstance(result, dict) else str(result)
        except Exception:
            return text

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences for better translation quality.
        T5 models work best with shorter inputs.
        """
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        # Merge very short fragments with the previous sentence
        merged = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if merged and len(sent.split()) < 4:
                merged[-1] = merged[-1] + " " + sent
            else:
                merged.append(sent)
        return merged

    def translate_en_to_bn(
        self,
        text: str,
        max_length: int = 256,
        num_beams: int = 4,
    ) -> str:
        """
        Translate English text to Bengali.

        Handles long text by splitting into sentences and translating
        individually for better quality.

        Args:
            text: English text to translate
            max_length: Max tokens per sentence translation
            num_beams: Beam search width (higher = better quality, slower)

        Returns:
            Bengali translation
        """
        if not text.strip():
            return ""

        # Strip citation brackets before translation (preserve them)
        citations = re.findall(r"\[.*?\]", text)
        clean_text = re.sub(r"\[.*?\]", "", text).strip()

        if not clean_text:
            return text  # Only citations, return as-is

        # Split into manageable sentences
        sentences = self._split_sentences(clean_text)
        translated_parts = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Skip lines that are purely numbers, bullets, or whitespace
            if re.match(r"^[\d\.\)\-\s]+$", sentence):
                translated_parts.append(sentence)
                continue

            try:
                inputs = self.en_bn_tokenizer(
                    sentence,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True,
                ).to(self.device)

                generated = self.en_bn_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

                output = self.en_bn_tokenizer.decode(
                    generated[0], skip_special_tokens=True
                )

                # Apply BN normalization to output
                output = self._normalize_bn(output)
                translated_parts.append(output)

            except Exception as e:
                logger.warning("Translation failed for sentence: %s — %s", sentence[:50], e)
                translated_parts.append(sentence)  # Fallback to original

        translated = " ".join(translated_parts)

        # Append citations at the end if they existed
        if citations:
            translated += "\n" + " ".join(citations)

        return translated

    def translate_bn_to_en(
        self,
        text: str,
        max_length: int = 256,
        num_beams: int = 4,
    ) -> str:
        """
        Translate Bengali text to English.

        Useful for translating Bengali user queries before retrieval/LLM.

        Args:
            text: Bengali text to translate
            max_length: Max tokens for translation output
            num_beams: Beam search width

        Returns:
            English translation
        """
        if not text.strip():
            return ""

        # Normalize Bengali input
        text = self._normalize_bn(text)

        try:
            inputs = self.bn_en_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self.device)

            generated = self.bn_en_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

            output = self.bn_en_tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
            logger.info("BN→EN translation: '%s' → '%s'", text[:50], output[:50])
            return output.strip()

        except Exception as e:
            logger.error("BN→EN translation failed: %s", e)
            return text  # Fallback to original


def get_translator(device: str = "cpu") -> BanglaTranslator:
    """
    Get or create the singleton translator instance.

    Thread-safe lazy initialization.
    """
    global _translator

    if _translator is not None:
        return _translator

    with _translator_lock:
        if _translator is not None:
            return _translator
        _translator = BanglaTranslator(device=device)

    return _translator
