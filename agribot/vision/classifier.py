"""
Optional classifier-assisted image analysis for AgriBot.

Provides Top-K disease/condition predictions using a lightweight
image classification model. This is OPTIONAL and config-gated:
  - If the model file is missing or the feature is disabled, returns empty results.
  - Falls back silently and logs a warning on any error.

This module does NOT claim VLM/vision-language capability.
It performs simple image classification only.
"""

import logging
from pathlib import Path
from threading import Lock
from typing import Optional

from agribot.vision.schema import PossibleCondition

logger = logging.getLogger(__name__)

# Singleton
_classifier_instance: "Optional[CropClassifier]" = None
_classifier_lock = Lock()


class CropClassifier:
    """
    Lightweight crop disease classifier using MobileNetV3 + CBAM.

    Loads a model (ONNX format) at runtime if available.
    To adhere to edge-hardware constraints, this network is strictly CPU-bound.
    The CBAM (Convolutional Block Attention Module) spatial and channel attention
    layers help the network focus on disease lesions and ignore background noise/soil.
    """

    def __init__(
        self,
        model_path: str | Path,
        top_k: int = 3,
        confidence_threshold: float = 0.3,
    ):
        self.model_path = Path(model_path)
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._labels: list[str] = []
        self._available = False

        if not self.model_path.exists():
            logger.warning(
                "Classifier model not found at %s; classifier disabled",
                self.model_path,
            )
            return

        try:
            self._load_model()
            self._available = True
            logger.info(
                "CropClassifier loaded: %s (%d labels, top_k=%d)",
                self.model_path.name, len(self._labels), self.top_k,
            )
        except Exception as e:
            logger.warning("Failed to load classifier: %s; falling back", e)
            self._available = False

    def _load_model(self):
        """
        Load the classification model.

        Supported formats:
          - .onnx: uses onnxruntime
          - .pt/.pth: uses torchvision (if available)

        Stub implementation — replace with actual model loading.
        """
        suffix = self.model_path.suffix.lower()

        if suffix == ".onnx":
            try:
                import onnxruntime as ort
                self._model = ort.InferenceSession(
                    str(self.model_path),
                    providers=["CPUExecutionProvider"],
                )
                # Load labels from companion .txt file
                labels_path = self.model_path.with_suffix(".txt")
                if labels_path.exists():
                    self._labels = [
                        label.strip() for label in labels_path.read_text().strip().split("\n")
                        if label.strip()
                    ]
                else:
                    logger.warning("Labels file not found: %s", labels_path)
            except ImportError:
                raise ImportError(
                    "onnxruntime required for ONNX classifier. "
                    "Install: pip install onnxruntime"
                )
        else:
            logger.warning("Unsupported classifier format: %s", suffix)
            raise ValueError(f"Unsupported classifier format: {suffix}")

    @property
    def is_available(self) -> bool:
        """Check if classifier is loaded and ready."""
        return self._available and self._model is not None

    def predict(self, image_path: str | Path) -> list[PossibleCondition]:
        """
        Run Top-K classification on a crop image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of PossibleCondition sorted by descending confidence.
            Empty list if classifier not available or on error.
        """
        if not self.is_available:
            return []

        try:
            return self._predict_impl(image_path)
        except Exception as e:
            logger.warning("Classifier prediction error: %s; returning empty", e)
            return []

    def _predict_impl(self, image_path: str | Path) -> list[PossibleCondition]:
        """
        Internal prediction implementation.

        Uses ONNX Runtime for .onnx models (MobileNetV3 + CBAM architecture).
        """
        import numpy as np
        from PIL import Image

        # Preprocess image (standard ImageNet-style for MobileNetV3)
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        # CHW format, batch dimension
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

        # Run inference
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: arr})
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        # Top-K
        top_indices = probs.argsort()[::-1][:self.top_k]
        conditions = []
        for idx in top_indices:
            conf = float(probs[idx])
            if conf >= self.confidence_threshold and idx < len(self._labels):
                conditions.append(PossibleCondition(
                    label=self._labels[idx],
                    confidence=conf,
                ))

        return conditions


def get_classifier(
    model_path: Optional[str] = None,
    top_k: int = 3,
    confidence_threshold: float = 0.3,
) -> Optional[CropClassifier]:
    """
    Get or create the singleton classifier.

    Returns None if model_path is None or empty.
    """
    global _classifier_instance

    if not model_path:
        return None

    if _classifier_instance is not None:
        return _classifier_instance

    with _classifier_lock:
        if _classifier_instance is not None:
            return _classifier_instance
        _classifier_instance = CropClassifier(
            model_path=model_path,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

    return _classifier_instance
