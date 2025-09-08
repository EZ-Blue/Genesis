#!/usr/bin/env python3
"""
Simple unit tests for mocap integration package
Tests basic functionality with mocked dependencies
"""

import unittest
import tempfile
import os
import json
from unittest.mock import MagicMock
import sys
import pytest

# Mock dependencies before importing
sys.modules['genesis'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Configure torch mock
torch_mock = MagicMock()
torch_mock.zeros.return_value = [0.0] * 38
sys.modules['torch'] = torch_mock

# Import modules to test
from mocap_player_genesis import MocapGenesisPlayer


@pytest.mark.fast
@pytest.mark.unit
class TestMocapPlayerBasics(unittest.TestCase):
    """Basic tests for MocapGenesisPlayer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock JSON mocap data
        self.mock_mocap_data = {
            "metadata": {"total_frames": 2, "duration": 0.016667},
            "frames": [
                {
                    "frame": 1, "time": 0.008333,
                    "joints": {
                        "Skeleton": {"position": {"x": -50.0, "y": 1000.0, "z": 100.0}},
                        "RShoulder": {"rotation": {"x": 10.5, "y": -5.2, "z": 15.8}}
                    }
                },
                {
                    "frame": 2, "time": 0.016667, 
                    "joints": {
                        "Skeleton": {"position": {"x": -49.8, "y": 1001.2, "z": 99.8}},
                        "RShoulder": {"rotation": {"x": 11.0, "y": -5.0, "z": 16.2}}
                    }
                }
            ]
        }
        
        self.test_json_path = os.path.join(self.temp_dir, "test_mocap.json")
        with open(self.test_json_path, 'w') as f:
            json.dump(self.mock_mocap_data, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_joint_mapping_structure(self):
        """Test joint mapping dictionary structure"""
        player = MocapGenesisPlayer("mock_skeleton.xml", self.test_json_path)
        
        # Test joint mapping exists and has expected structure
        self.assertIsInstance(player.joint_mapping, dict)
        self.assertGreater(len(player.joint_mapping), 20)
        
        # Test specific joint mappings
        self.assertIn(("RShoulder", "x"), player.joint_mapping)
        self.assertIn(("RThigh", "x"), player.joint_mapping)
        
        # Test mapping values are strings
        for key, value in player.joint_mapping.items():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)
            self.assertIsInstance(value, str)
            self.assertIn(key[1], ['x', 'y', 'z'])
    
    def test_load_mocap_data(self):
        """Test mocap data loading"""
        player = MocapGenesisPlayer("mock_skeleton.xml", self.test_json_path)
        
        success = player.load_mocap_data()
        
        self.assertTrue(success)
        self.assertIsNotNone(player.mocap_data)
        self.assertIn('frames', player.mocap_data)
        self.assertEqual(len(player.mocap_data['frames']), 2)
    
    def test_coordinate_transformation(self):
        """Test coordinate system transformation"""
        player = MocapGenesisPlayer("mock_skeleton.xml", self.test_json_path)
        
        # Test hip joint transformation (should be inverted)
        result = player.transform_mocap_rotation_to_genesis("RThigh", "x", -30.0)
        self.assertEqual(result, 30.0)  # Should be inverted
        
        # Test joint with no transformation
        result = player.transform_mocap_rotation_to_genesis("UnknownJoint", "x", 45.0)
        self.assertEqual(result, 45.0)  # Should remain unchanged
    
    def test_joint_indices(self):
        """Test joint name to DOF index mapping"""
        player = MocapGenesisPlayer("mock_skeleton.xml", self.test_json_path)
        
        # Test known joint indices
        self.assertEqual(player.get_joint_index("hip_flexion_r"), 7)
        self.assertEqual(player.get_joint_index("knee_angle_r"), 16)
        self.assertEqual(player.get_joint_index("arm_flex_r"), 18)
        
        # Test unknown joint
        self.assertIsNone(player.get_joint_index("unknown_joint"))
    
    def test_euler_to_quaternion(self):
        """Test Euler to quaternion conversion"""
        player = MocapGenesisPlayer("mock_skeleton.xml", self.test_json_path)
        
        # Test zero rotation
        w, x, y, z = player.convert_euler_to_quaternion(0, 0, 0)
        self.assertAlmostEqual(w, 1.0, places=5)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)
        self.assertAlmostEqual(z, 0.0, places=5)
        
        # Test 90 degree X rotation
        w, x, y, z = player.convert_euler_to_quaternion(90, 0, 0)
        self.assertAlmostEqual(w, 0.7071, places=3)
        self.assertAlmostEqual(x, 0.7071, places=3)


def run_simple_tests():
    """Run basic working tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    print("Running basic mocap integration tests...")
    run_simple_tests()