"""
test_model.py
-------------
Unit tests for face mask detection system.
Run: python -m pytest tests/test_model.py -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPreprocessing:
    """Tests for image preprocessing functions."""

    def test_preprocess_face_shape(self):
        """Preprocessed output should be (1, 224, 224, 3)."""
        from src.utils import preprocess_face
        import cv2

        dummy_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_face(dummy_face)
        assert result.shape == (1, 224, 224, 3), f"Expected (1,224,224,3), got {result.shape}"

    def test_preprocess_face_normalization(self):
        """Pixel values should be in [0, 1] range."""
        from src.utils import preprocess_face
        import cv2

        dummy_face = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        result = preprocess_face(dummy_face)
        assert result.min() >= 0.0, "Min value should be >= 0"
        assert result.max() <= 1.0, "Max value should be <= 1"

    def test_preprocess_handles_small_face(self):
        """Should handle very small face regions without crashing."""
        from src.utils import preprocess_face
        import cv2

        small_face = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        result = preprocess_face(small_face)
        assert result.shape == (1, 224, 224, 3)


class TestModelArchitecture:
    """Tests for model construction."""

    def test_model_builds(self):
        """Model should build without errors."""
        from src.model import build_model
        model = build_model()
        assert model is not None

    def test_model_output_shape(self):
        """Model output should be (batch, 1) for binary classification."""
        from src.model import build_model
        import numpy as np

        model = build_model()
        dummy_input = np.random.rand(2, 224, 224, 3).astype("float32")
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (2, 1), f"Expected (2,1), got {output.shape}"

    def test_model_output_range(self):
        """Sigmoid output should be in [0, 1]."""
        from src.model import build_model
        import numpy as np

        model = build_model()
        dummy_input = np.random.rand(3, 224, 224, 3).astype("float32")
        output = model.predict(dummy_input, verbose=0)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_model_compiles(self):
        """Model should compile without errors."""
        from src.model import build_model, compile_model
        model = build_model()
        compiled = compile_model(model)
        assert compiled.optimizer is not None


class TestUtils:
    """Tests for utility functions."""

    def test_draw_prediction_returns_frame(self):
        """draw_prediction should return a numpy array."""
        from src.utils import draw_prediction
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_prediction(frame, 100, 100, 80, 80, "Mask", 0.95)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)

    def test_labels_correct(self):
        """Label values should match expected class names."""
        from src.utils import LABELS
        assert LABELS[0] == "No Mask"
        assert LABELS[1] == "Mask"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
