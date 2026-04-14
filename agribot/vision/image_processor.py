"""
Offline Image Processor for crop/plant photos.

Combines OCR (Tesseract) for text-on-image extraction with
rule-based symptom detection from visual features.
Designed to work fully offline for rural deployment.

When a VLM (Vision Language Model) is available, it uses that for
richer image captioning. Otherwise falls back to OCR + heuristic
color/pattern analysis.
"""

import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# Singleton
_processor_instance: "ImageProcessor | None" = None
_processor_lock = Lock()


class ImageProcessor:
    """
    Offline image analysis for agricultural crop photos.

    Capabilities:
    1. OCR extraction (text on labels, packaging, disease guides)
    2. Color-based symptom hints (yellowing, browning, spots)
    3. VLM captioning when available (e.g., via llama-cpp multimodal)
    """

    def __init__(self, vlm_model_path: str | None = None):
        """
        Initialize the image processor.

        Args:
            vlm_model_path: Optional path to a GGUF VLM model for captioning.
                            If None, uses OCR + heuristic analysis only.
        """
        self.vlm_model_path = vlm_model_path
        self._vlm = None

        logger.info(
            "ImageProcessor initialized (VLM: %s)",
            "enabled" if vlm_model_path else "OCR+heuristic only",
        )

    def describe_image(self, image_path: str | Path) -> str:
        """
        Analyze a crop/plant image and return a textual description.

        Uses OCR + heuristic analysis. VLM captioning is a placeholder
        and not currently functional.

        Args:
            image_path: Path to the image file (JPG, PNG, etc.)

        Returns:
            Text description of the image content/symptoms
        """
        result = self.describe_image_structured(image_path)
        return result.build_query_text()

    def describe_image_structured(self, image_path: str | Path, classifier=None):
        """
        Analyze a crop/plant image and return structured analysis.

        Args:
            image_path: Path to the image file.
            classifier: Optional CropClassifier instance for enhanced analysis.

        Returns:
            ImageAnalysisResult with structured fields.
        """
        from agribot.vision.schema import ImageAnalysisResult

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        result = ImageAnalysisResult()

        # 1. OCR for any text in the image
        ocr_text = self._ocr_extract(image_path)
        if ocr_text:
            result.ocr_text = ocr_text
            # Extract keywords from OCR text
            words = [w.strip().lower() for w in ocr_text.split() if len(w.strip()) > 3]
            result.keywords.extend(words[:10])

        # 2. Color/symptom heuristics
        symptom_hints = self._analyze_symptoms(image_path)
        if symptom_hints:
            result.symptom_hints = [h.strip() for h in symptom_hints.split(";")]
            # Extract keywords from symptom text
            for hint in result.symptom_hints:
                key_words = [w for w in hint.split() if len(w) > 4]
                result.keywords.extend(key_words[:3])

        # 3. Optional classifier-assisted path
        if (
            classifier is not None
            and hasattr(classifier, "is_available")
            and classifier.is_available
        ):
            try:
                conditions = classifier.predict(image_path)
                if conditions:
                    result.possible_conditions = conditions
                    result.pipeline_used = "classifier_assisted"
                    logger.info(
                        "Classifier predicted %d conditions (top: %s %.1f%%)",
                        len(conditions),
                        conditions[0].label,
                        conditions[0].confidence * 100,
                    )
                else:
                    result.pipeline_used = "ocr_baseline"
            except Exception as e:
                logger.warning("Classifier failed, using baseline: %s", e)
                result.pipeline_used = "ocr_fallback"
                result.limitations.append(f"Classifier error: {e}")
        else:
            result.pipeline_used = "ocr_baseline"

        # Deduplicate keywords
        result.keywords = list(dict.fromkeys(result.keywords))

        if not result.ocr_text and not result.symptom_hints:
            result.quality_flags.append("no_features_extracted")

        return result

    def _vlm_caption(self, image_path: Path) -> str:
        """
        Generate a caption using an offline Vision Language Model.

        Currently a placeholder for VLM integration (e.g., LLaVA, BakLLaVA
        via llama-cpp-python multimodal support).
        """
        if not self.vlm_model_path:
            return ""

        try:
            # Future: integrate llama-cpp-python with clip projection
            # from llama_cpp import Llama
            # from llama_cpp.llama_chat_format import Llava15ChatHandler
            logger.info("VLM captioning not yet fully integrated")
            return ""
        except Exception as e:
            logger.warning("VLM captioning failed: %s", e)
            return ""

    def _ocr_extract(self, image_path: Path) -> str:
        """Extract text from image using Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            # Try English + Bengali OCR
            text = pytesseract.image_to_string(img, lang="eng+ben")
            text = text.strip()

            if len(text) > 10:  # Only return meaningful text
                logger.info("OCR extracted %d chars from image", len(text))
                return text[:500]  # Limit length
            return ""

        except ImportError:
            logger.debug("pytesseract not available for image OCR")
            return ""
        except Exception as e:
            logger.warning("Image OCR failed: %s", e)
            return ""

    def _analyze_symptoms(self, image_path: Path) -> str:
        """
        Analyze image for agricultural symptom indicators using color analysis.

        Detects common visual patterns:
        - Yellowing (nitrogen deficiency, tungro, etc.)
        - Brown spots (blast, blight, leaf spot)
        - Wilting patterns
        - White powdery patches (powdery mildew)
        """
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(image_path).convert("RGB")
            # Resize for fast analysis
            img = img.resize((256, 256))
            pixels = np.array(img, dtype=np.float32)

            # Compute color channel statistics
            r_mean = pixels[:, :, 0].mean()
            g_mean = pixels[:, :, 1].mean()
            b_mean = pixels[:, :, 2].mean()

            # Compute HSV for better color analysis
            r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
            max_c = np.maximum(np.maximum(r, g), b)
            min_c = np.minimum(np.minimum(r, g), b)
            diff = max_c - min_c

            # Saturation
            saturation = np.where(max_c > 0, diff / max_c, 0)
            avg_saturation = saturation.mean()

            symptoms = []

            # Yellowing detection (high R, high G, low B relative to green)
            yellow_ratio = (r_mean + g_mean) / (2 * max(b_mean, 1))
            if yellow_ratio > 2.5 and g_mean > 100:
                symptoms.append(
                    "significant yellowing detected (possible nutrient deficiency or disease)"
                )

            # Brown spots (moderate R, low G, low B)
            brown_mask = (r > 100) & (g < 100) & (b < 80)
            brown_fraction = brown_mask.mean()
            if brown_fraction > 0.05:
                symptoms.append(
                    f"brown discoloration detected ({brown_fraction:.0%} of image area)"
                )

            # Predominantly green (healthy plant reference)
            if g_mean > r_mean and g_mean > b_mean and avg_saturation > 0.3:
                green_dominance = g_mean / max(r_mean, 1)
                if green_dominance > 1.3:
                    symptoms.append("predominantly green/healthy plant tissue visible")

            # White/pale patches (possible mildew)
            white_mask = (r > 200) & (g > 200) & (b > 200)
            white_fraction = white_mask.mean()
            if white_fraction > 0.1:
                symptoms.append(
                    f"white/pale patches detected ({white_fraction:.0%} of area, possible mildew)"
                )

            # Dark patches (possible rot or severe damage)
            dark_mask = max_c < 50
            dark_fraction = dark_mask.mean()
            if dark_fraction > 0.15:
                symptoms.append("dark/necrotic areas detected")

            if symptoms:
                return "; ".join(symptoms)
            return ""

        except ImportError:
            logger.debug("PIL/numpy not available for symptom analysis")
            return ""
        except Exception as e:
            logger.warning("Symptom analysis failed: %s", e)
            return ""


def get_image_processor(vlm_model_path: str | None = None) -> ImageProcessor:
    """
    Get or create the singleton ImageProcessor instance.

    Thread-safe lazy initialization.
    """
    global _processor_instance

    if _processor_instance is not None:
        return _processor_instance

    with _processor_lock:
        if _processor_instance is not None:
            return _processor_instance
        _processor_instance = ImageProcessor(vlm_model_path=vlm_model_path)

    return _processor_instance
