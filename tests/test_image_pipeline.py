"""
Tests for the image analysis pipeline.

Tests baseline OCR+heuristic path and classifier-assisted path.
All tests use mocks — no actual models or images required.
"""

from unittest.mock import MagicMock, patch


class TestImageAnalysisResult:
    """Test the ImageAnalysisResult data class."""

    def test_to_dict(self):
        from agribot.vision.schema import ImageAnalysisResult

        result = ImageAnalysisResult(
            pipeline_used="ocr_baseline",
            ocr_text="some text",
            symptom_hints=["yellowing", "spots"],
            keywords=["leaf", "yellow"],
        )

        d = result.to_dict()
        assert d["pipeline_used"] == "ocr_baseline"
        assert d["ocr_text"] == "some text"
        assert len(d["symptom_hints"]) == 2
        assert d["possible_conditions"] == []

    def test_to_dict_with_conditions(self):
        from agribot.vision.schema import ImageAnalysisResult, PossibleCondition

        result = ImageAnalysisResult(
            pipeline_used="classifier_assisted",
            possible_conditions=[
                PossibleCondition(label="rice_blast", confidence=0.85),
                PossibleCondition(label="leaf_blight", confidence=0.12),
            ],
        )

        d = result.to_dict()
        assert d["pipeline_used"] == "classifier_assisted"
        assert len(d["possible_conditions"]) == 2
        assert d["possible_conditions"][0]["label"] == "rice_blast"
        assert d["possible_conditions"][0]["confidence"] == 0.85

    def test_build_query_text_empty(self):
        from agribot.vision.schema import ImageAnalysisResult

        result = ImageAnalysisResult()
        text = result.build_query_text()
        assert "unable to extract" in text.lower() or "uploaded" in text.lower()

    def test_build_query_text_with_data(self):
        from agribot.vision.schema import ImageAnalysisResult, PossibleCondition

        result = ImageAnalysisResult(
            ocr_text="Leaf Disease Guide",
            symptom_hints=["yellowing detected"],
            keywords=["yellowing", "leaf"],
            possible_conditions=[PossibleCondition("blast", 0.9)],
        )

        text = result.build_query_text("my rice is sick")
        assert "my rice is sick" in text
        assert "yellowing" in text
        assert "blast" in text

    def test_build_query_text_with_user_query(self):
        from agribot.vision.schema import ImageAnalysisResult

        result = ImageAnalysisResult(symptom_hints=["brown spots"])
        text = result.build_query_text("what disease is this?")
        assert "what disease is this?" in text
        assert "brown spots" in text


class TestImageProcessorStructured:
    """Test describe_image_structured (mock OCR + symptoms)."""

    @patch(
        "agribot.vision.image_processor.ImageProcessor._ocr_extract",
        return_value="Leaf Disease",
    )
    @patch(
        "agribot.vision.image_processor.ImageProcessor._analyze_symptoms",
        return_value="yellowing detected; brown spots",
    )
    def test_baseline_no_classifier(self, mock_symptoms, mock_ocr, tmp_path):
        from agribot.vision.image_processor import ImageProcessor

        # Create a fake image file
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image data")

        proc = ImageProcessor()
        result = proc.describe_image_structured(img_path, classifier=None)

        assert result.pipeline_used == "ocr_baseline"
        assert result.ocr_text == "Leaf Disease"
        assert len(result.symptom_hints) == 2
        assert result.possible_conditions == []

    @patch(
        "agribot.vision.image_processor.ImageProcessor._ocr_extract", return_value=""
    )
    @patch(
        "agribot.vision.image_processor.ImageProcessor._analyze_symptoms",
        return_value="",
    )
    def test_no_features_extracted(self, mock_symptoms, mock_ocr, tmp_path):
        from agribot.vision.image_processor import ImageProcessor

        img_path = tmp_path / "blank.jpg"
        img_path.write_bytes(b"fake image")

        proc = ImageProcessor()
        result = proc.describe_image_structured(img_path)

        assert "no_features_extracted" in result.quality_flags

    @patch(
        "agribot.vision.image_processor.ImageProcessor._ocr_extract",
        return_value="text",
    )
    @patch(
        "agribot.vision.image_processor.ImageProcessor._analyze_symptoms",
        return_value="spots",
    )
    def test_classifier_assisted_path(self, mock_symptoms, mock_ocr, tmp_path):
        from agribot.vision.image_processor import ImageProcessor
        from agribot.vision.schema import PossibleCondition

        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        # Mock classifier
        mock_classifier = MagicMock()
        mock_classifier.is_available = True
        mock_classifier.predict.return_value = [
            PossibleCondition("rice_blast", 0.9),
            PossibleCondition("leaf_blight", 0.3),
        ]

        proc = ImageProcessor()
        result = proc.describe_image_structured(img_path, classifier=mock_classifier)

        assert result.pipeline_used == "classifier_assisted"
        assert len(result.possible_conditions) == 2
        assert result.possible_conditions[0].label == "rice_blast"

    @patch(
        "agribot.vision.image_processor.ImageProcessor._ocr_extract",
        return_value="text",
    )
    @patch(
        "agribot.vision.image_processor.ImageProcessor._analyze_symptoms",
        return_value="spots",
    )
    def test_classifier_error_fallback(self, mock_symptoms, mock_ocr, tmp_path):
        from agribot.vision.image_processor import ImageProcessor

        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        mock_classifier = MagicMock()
        mock_classifier.is_available = True
        mock_classifier.predict.side_effect = RuntimeError("model error")

        proc = ImageProcessor()
        result = proc.describe_image_structured(img_path, classifier=mock_classifier)

        assert result.pipeline_used == "ocr_fallback"
        assert any("error" in limitation.lower() for limitation in result.limitations)

    @patch(
        "agribot.vision.image_processor.ImageProcessor._ocr_extract",
        return_value="text",
    )
    @patch(
        "agribot.vision.image_processor.ImageProcessor._analyze_symptoms",
        return_value="spots",
    )
    def test_classifier_unavailable_falls_back(self, mock_symptoms, mock_ocr, tmp_path):
        from agribot.vision.image_processor import ImageProcessor

        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake")

        mock_classifier = MagicMock()
        mock_classifier.is_available = False

        proc = ImageProcessor()
        result = proc.describe_image_structured(img_path, classifier=mock_classifier)

        assert result.pipeline_used == "ocr_baseline"


class TestClassifier:
    """Test the CropClassifier interface."""

    def test_missing_model_not_available(self, tmp_path):
        from agribot.vision.classifier import CropClassifier

        classifier = CropClassifier(
            model_path=tmp_path / "nonexistent_model.onnx",
        )
        assert not classifier.is_available
        assert classifier.predict(tmp_path / "img.jpg") == []

    def test_predict_returns_empty_when_unavailable(self, tmp_path):
        from agribot.vision.classifier import CropClassifier

        classifier = CropClassifier(model_path=tmp_path / "no.onnx")
        result = classifier.predict(tmp_path / "test.jpg")
        assert result == []
