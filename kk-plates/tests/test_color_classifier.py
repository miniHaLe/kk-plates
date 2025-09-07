"""Tests for color classification."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch
from kkplates.classify.color_model import ColorClassifier


class TestColorClassifier:
    """Test suite for color classification."""
    
    @pytest.fixture
    def hsv_thresholds(self):
        """HSV thresholds for testing."""
        return {
            "red": {"h": [0, 10], "s": [80, 255], "v": [60, 255]},
            "yellow": {"h": [17, 35], "s": [80, 255], "v": [60, 255]},
            "normal": {"h": [0, 179], "s": [0, 70], "v": [120, 255]}
        }
    
    @pytest.fixture
    def classifier(self, hsv_thresholds):
        """Create classifier instance."""
        return ColorClassifier("dummy_model.onnx", hsv_thresholds)
    
    def create_solid_color_image(self, bgr_color, size=(100, 100)):
        """Create a solid color image for testing."""
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:, :] = bgr_color
        return img
    
    def test_init(self, classifier):
        """Test classifier initialization."""
        assert classifier.model_path == Path("dummy_model.onnx")
        assert len(classifier.hsv_thresholds) == 3
        assert classifier.session is None
    
    def test_classify_red_plate_hsv(self, classifier):
        """Test classification of red plate using HSV."""
        # Create a red image (BGR format)
        red_bgr = (0, 0, 200)  # Pure red
        img = self.create_solid_color_image(red_bgr)
        bbox = (10, 10, 90, 90)
        
        color, confidence = classifier.classify(img, bbox)
        
        assert color == "red"
        assert confidence > 0.98  # Should be very confident for solid color
    
    def test_classify_yellow_plate_hsv(self, classifier):
        """Test classification of yellow plate using HSV."""
        # Create a yellow image (BGR format)
        yellow_bgr = (0, 200, 200)  # Yellow
        img = self.create_solid_color_image(yellow_bgr)
        bbox = (10, 10, 90, 90)
        
        color, confidence = classifier.classify(img, bbox)
        
        assert color == "yellow"
        assert confidence > 0.98
    
    def test_classify_normal_plate_hsv(self, classifier):
        """Test classification of normal (white/gray) plate using HSV."""
        # Create a white/gray image (BGR format)
        gray_bgr = (200, 200, 200)  # Light gray
        img = self.create_solid_color_image(gray_bgr)
        bbox = (10, 10, 90, 90)
        
        color, confidence = classifier.classify(img, bbox)
        
        assert color == "normal"
        assert confidence > 0.98
    
    def test_classify_with_noise(self, classifier):
        """Test classification with noisy image."""
        # Create red image with noise
        red_bgr = (0, 0, 200)
        img = self.create_solid_color_image(red_bgr)
        
        # Add random noise
        noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        bbox = (10, 10, 90, 90)
        color, confidence = classifier.classify(img, bbox)
        
        # Should still classify as red, but with lower confidence
        assert color == "red"
        assert confidence > 0.5  # Lower threshold due to noise
    
    def test_classify_empty_bbox(self, classifier):
        """Test classification with empty bounding box."""
        img = self.create_solid_color_image((100, 100, 100))
        bbox = (50, 50, 50, 50)  # Zero-size bbox
        
        color, confidence = classifier.classify(img, bbox)
        
        assert color == "normal"  # Default
        assert confidence == 0.5  # Default confidence
    
    def test_classify_out_of_bounds_bbox(self, classifier):
        """Test classification with out-of-bounds bbox."""
        img = self.create_solid_color_image((200, 0, 0), size=(100, 100))
        bbox = (90, 90, 110, 110)  # Partially out of bounds
        
        # Should handle gracefully
        color, confidence = classifier.classify(img, bbox)
        assert color in ["red", "yellow", "normal"]
    
    @patch('onnxruntime.InferenceSession')
    def test_classify_with_cnn_fallback(self, mock_session, classifier):
        """Test CNN fallback when HSV is ambiguous."""
        # Mock ONNX session
        mock_instance = Mock()
        mock_session.return_value = mock_instance
        
        # Mock model inputs/outputs
        mock_input = Mock()
        mock_input.name = "input"
        mock_output = Mock()
        mock_output.name = "output"
        
        mock_instance.get_inputs.return_value = [mock_input]
        mock_instance.get_outputs.return_value = [mock_output]
        
        # Mock CNN prediction (high confidence for yellow)
        mock_instance.run.return_value = [np.array([[0.1, 0.85, 0.05]])]  # [red, yellow, normal]
        
        # Load model (will use mock)
        classifier.session = mock_instance
        
        # Create ambiguous color image (between red and yellow)
        ambiguous_bgr = (0, 150, 150)  # Orange-ish
        img = self.create_solid_color_image(ambiguous_bgr)
        bbox = (10, 10, 90, 90)
        
        color, confidence = classifier.classify(img, bbox)
        
        # Should use CNN result
        assert color == "yellow"
        assert confidence > 0.8
    
    def test_create_color_mask_red(self, classifier):
        """Test color mask creation for red."""
        # Create image with red and non-red regions
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :] = (0, 0, 200)  # Red top half
        img[50:, :] = (200, 200, 200)  # Gray bottom half
        
        mask = classifier.create_color_mask(img, "red")
        
        # Check mask shape
        assert mask.shape == (100, 100)
        
        # Top half should be mostly white (255)
        assert np.mean(mask[:50, :]) > 200
        
        # Bottom half should be mostly black (0)
        assert np.mean(mask[50:, :]) < 50
    
    def test_create_color_mask_invalid_color(self, classifier):
        """Test color mask with invalid color."""
        img = self.create_solid_color_image((100, 100, 100))
        mask = classifier.create_color_mask(img, "invalid_color")
        
        # Should return empty mask
        assert mask.shape == (100, 100)
        assert np.all(mask == 0)
    
    def test_hsv_wrap_around_for_red(self, classifier):
        """Test HSV wrap-around handling for red color."""
        # Create dark red (hue near 180)
        dark_red_bgr = (0, 0, 150)
        img = self.create_solid_color_image(dark_red_bgr)
        
        # Convert to HSV to check hue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[50, 50, 0]
        
        # If hue is near 0 or 180, test wrap-around
        if hue < 10 or hue > 170:
            bbox = (10, 10, 90, 90)
            color, confidence = classifier.classify(img, bbox)
            assert color == "red"
    
    def test_color_ratios_from_patches(self, classifier):
        """Test classification on multiple color patches."""
        # Create image with three color regions
        img = np.zeros((300, 100, 3), dtype=np.uint8)
        img[:100, :] = (0, 0, 200)  # Red
        img[100:200, :] = (0, 200, 200)  # Yellow
        img[200:, :] = (200, 200, 200)  # Normal
        
        results = []
        bboxes = [(10, 10, 90, 90), (10, 110, 90, 190), (10, 210, 90, 290)]
        expected = ["red", "yellow", "normal"]
        
        for bbox, expected_color in zip(bboxes, expected):
            color, confidence = classifier.classify(img, bbox)
            results.append((color, confidence))
            assert color == expected_color
            assert confidence > 0.98
    
    def test_thread_safety(self, classifier):
        """Test classifier can be used from multiple threads."""
        import threading
        
        img = self.create_solid_color_image((0, 0, 200))
        bbox = (10, 10, 90, 90)
        results = []
        
        def classify_thread():
            color, conf = classifier.classify(img, bbox)
            results.append((color, conf))
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=classify_thread)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All threads should get same result
        assert len(results) == 5
        assert all(r[0] == "red" for r in results)