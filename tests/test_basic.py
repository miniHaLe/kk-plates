"""
Basic tests for KichiKichi Conveyor Belt System
"""

import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dish_detection.dish_detector import DishDetector, DishDetection
from ocr_model.number_detector import ConveyorNumberDetector, NumberDetection
from tracking.conveyor_tracker import ConveyorTracker, ConveyorState
from config.config import config

class TestDishDetector(unittest.TestCase):
    """Test dish detection functionality"""
    
    def setUp(self):
        self.detector = DishDetector()
    
    def test_detector_initialization(self):
        """Test that detector initializes properly"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.confidence_threshold, 0.6)
    
    def test_color_classification(self):
        """Test color classification with sample images"""
        # Create test image (red-ish)
        red_image = np.zeros((100, 100, 3), dtype=np.uint8)
        red_image[:, :, 2] = 255  # Red channel
        
        dish_type = self.detector._classify_dish_color(red_image)
        # Should classify as red_dish or normal_dish depending on threshold
        self.assertIn(dish_type, ['red_dish', 'normal_dish'])
    
    def test_advertisement_detection(self):
        """Test advertisement dish detection"""
        # Create colorful test image
        colorful_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        is_ad = self.detector._is_advertisement_dish(colorful_image)
        self.assertIsInstance(is_ad, bool)

class TestNumberDetector(unittest.TestCase):
    """Test OCR number detection functionality"""
    
    def setUp(self):
        self.detector = ConveyorNumberDetector()
    
    def test_detector_initialization(self):
        """Test that OCR detector initializes properly"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.7)
    
    def test_number_extraction(self):
        """Test number extraction from text"""
        # Test valid numbers
        self.assertEqual(self.detector._extract_number("Phase 5"), 5)
        self.assertEqual(self.detector._extract_number("12"), 12)
        self.assertEqual(self.detector._extract_number("Stage 0"), 0)
        
        # Test invalid inputs
        self.assertIsNone(self.detector._extract_number("No numbers here"))
        self.assertIsNone(self.detector._extract_number("1000"))  # Out of range
    
    def test_preprocessing(self):
        """Test image preprocessing for OCR"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = self.detector._preprocess_for_ocr(test_image)
        
        # Should be grayscale
        self.assertEqual(len(processed.shape), 2)
        self.assertEqual(processed.shape, (100, 100))

class TestConveyorTracker(unittest.TestCase):
    """Test conveyor tracking functionality"""
    
    def setUp(self):
        self.tracker = ConveyorTracker()
    
    def test_tracker_initialization(self):
        """Test that tracker initializes properly"""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.state.current_stage, 1)
        self.assertEqual(self.tracker.state.current_phase, 0)
    
    def test_stage_advancement(self):
        """Test stage advancement logic"""
        initial_stage = self.tracker.state.current_stage
        self.tracker._advance_stage()
        self.assertEqual(self.tracker.state.current_stage, initial_stage + 1)
    
    def test_dish_counting(self):
        """Test dish counting functionality"""
        initial_count = self.tracker.state.total_dishes['normal_dish']
        
        # Create mock dish detection
        mock_detection = DishDetection(
            bbox=(10, 10, 50, 50),
            confidence=0.8,
            dish_type='normal_dish',
            center_point=(30, 30),
            timestamp=None
        )
        
        self.tracker._process_dish_detections([mock_detection])
        self.assertEqual(self.tracker.state.total_dishes['normal_dish'], initial_count + 1)
    
    def test_reset_functionality(self):
        """Test reset functionality"""
        # Add some dishes first
        self.tracker.state.total_dishes['normal_dish'] = 10
        
        # Reset
        self.tracker.reset_counts()
        
        # Check if reset
        self.assertEqual(self.tracker.state.total_dishes['normal_dish'], 0)
        self.assertEqual(self.tracker.state.current_stage, 1)
        self.assertEqual(self.tracker.state.current_phase, 0)

class TestConfiguration(unittest.TestCase):
    """Test configuration system"""
    
    def test_config_loading(self):
        """Test that configuration loads properly"""
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.conveyor)
        self.assertIsNotNone(config.camera)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.dashboard)
    
    def test_dish_classes(self):
        """Test dish class configuration"""
        expected_classes = ['normal_dish', 'red_dish', 'yellow_dish', 'advertisement_dish']
        for dish_class in expected_classes:
            self.assertIn(dish_class, config.model.dish_classes)
    
    def test_conveyor_parameters(self):
        """Test conveyor configuration parameters"""
        self.assertGreater(config.conveyor.max_phases_per_stage, 0)
        self.assertGreater(config.conveyor.total_stages, 0)
        self.assertGreaterEqual(config.conveyor.break_line_threshold, 0)
        self.assertLessEqual(config.conveyor.break_line_threshold, 1)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.dish_detector = DishDetector()
        self.number_detector = ConveyorNumberDetector()
        self.tracker = ConveyorTracker()
    
    def test_end_to_end_flow(self):
        """Test end-to-end processing flow"""
        # Create a simple test image
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection (will likely find nothing, but shouldn't crash)
        dish_detections = self.dish_detector.detect_dishes(test_frame)
        number_detections = self.number_detector.detect_numbers(test_frame)
        
        # Update tracker
        initial_stage = self.tracker.state.current_stage
        initial_phase = self.tracker.state.current_phase
        
        self.tracker.update_from_detections(dish_detections, number_detections)
        
        # System should still be running
        self.assertIsNotNone(self.tracker.state)
        self.assertGreaterEqual(self.tracker.state.current_stage, initial_stage)
        self.assertGreaterEqual(self.tracker.state.current_phase, initial_phase)

if __name__ == '__main__':
    # Create logs directory for testing
    os.makedirs('logs', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)
