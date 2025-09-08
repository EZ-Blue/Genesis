#!/usr/bin/env python3
"""
Realistic unit tests for mocap integration package
Tests with actual mocap data files and real functionality
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import MagicMock
import sys

# Mock dependencies before importing
sys.modules['genesis'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Configure torch mock
torch_mock = MagicMock()
torch_mock.zeros.return_value = [0.0] * 38
sys.modules['torch'] = torch_mock

# Import modules to test
from mocap_converter import MocapConverter
from mocap_player_genesis import MocapGenesisPlayer


class TestRealMocapData(unittest.TestCase):
    """Test with actual mocap data files"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_dir = Path(__file__).parent
        
        # Use actual LOCAL CSV file
        self.local_csv_path = self.base_dir / "mocap_data" / "Take 2025-08-19 03.01.18 PM_LOCAL.csv"
        self.local_json_path = self.base_dir / "mocap_data" / "Take 2025-08-19 03.01.18 PM_LOCAL_processed.json"
        
        # Mock skeleton path
        self.skeleton_xml = "skeleton/genesis_skeleton.xml"
    
    def test_local_csv_exists(self):
        """Test that the LOCAL CSV file exists"""
        self.assertTrue(self.local_csv_path.exists(), f"LOCAL CSV not found: {self.local_csv_path}")
    
    def test_local_json_exists(self):
        """Test that the processed JSON file exists"""
        self.assertTrue(self.local_json_path.exists(), f"LOCAL JSON not found: {self.local_json_path}")
    
    def test_mocap_converter_with_real_data(self):
        """Test MocapConverter with real LOCAL CSV data"""
        if not self.local_csv_path.exists():
            self.skipTest("LOCAL CSV file not available")
        
        converter = MocapConverter(str(self.local_csv_path))
        
        # Test loading CSV data
        converter.load_csv_data()
        self.assertIsNotNone(converter.frame_data)
        self.assertGreater(len(converter.frame_data), 0)
        
        # Test joint structure analysis
        converter.analyze_joint_structure()
        self.assertIsNotNone(converter.joint_info)
        self.assertGreater(len(converter.joint_info), 0)
        
        # Test frame data processing
        converter.process_frame_data()
        self.assertIsNotNone(converter.processed_data)
        self.assertEqual(len(converter.processed_data), len(converter.frame_data))
    
    def test_mocap_player_with_real_json(self):
        """Test MocapGenesisPlayer with real processed JSON"""
        if not self.local_json_path.exists():
            self.skipTest("LOCAL JSON file not available")
        
        player = MocapGenesisPlayer(self.skeleton_xml, str(self.local_json_path))
        
        # Test loading mocap data
        success = player.load_mocap_data()
        self.assertTrue(success)
        self.assertIsNotNone(player.mocap_data)
        
        # Test that we have frames
        frames = player.mocap_data['frames']
        self.assertGreater(len(frames), 0)
        
        # Test frame structure
        first_frame = frames[0]
        self.assertIn('frame', first_frame)
        self.assertIn('time', first_frame)
        self.assertIn('joints', first_frame)
    
    def test_joint_mapping_completeness(self):
        """Test that joint mapping covers all required DOFs"""
        player = MocapGenesisPlayer(self.skeleton_xml, "dummy.json")
        
        # Test joint mapping structure
        self.assertIsInstance(player.joint_mapping, dict)
        self.assertGreater(len(player.joint_mapping), 30)  # Should have many joint mappings
        
        # Test specific critical joint mappings
        critical_joints = [
            ("RThigh", "x"),     # Hip flexion
            ("LThigh", "x"),     # Hip flexion
            ("RShin", "x"),      # Knee
            ("LShin", "x"),      # Knee
            ("RShoulder", "x"),  # Shoulder
            ("LShoulder", "x"),  # Shoulder
        ]
        
        for joint_name, axis in critical_joints:
            self.assertIn((joint_name, axis), player.joint_mapping, 
                         f"Missing critical joint mapping: ({joint_name}, {axis})")
    
    def test_coordinate_transformations(self):
        """Test coordinate system transformations for key joints"""
        player = MocapGenesisPlayer(self.skeleton_xml, "dummy.json")
        
        # Test hip joint inversions (should be inverted)
        self.assertEqual(player.transform_mocap_rotation_to_genesis("RThigh", "x", -30.0), 30.0)
        self.assertEqual(player.transform_mocap_rotation_to_genesis("LThigh", "x", -30.0), 30.0)
        
        # Test knee joint inversions (should be inverted)
        self.assertEqual(player.transform_mocap_rotation_to_genesis("RShin", "x", -45.0), 45.0)
        self.assertEqual(player.transform_mocap_rotation_to_genesis("LShin", "x", -45.0), 45.0)
        
        # Test shoulder inversions (should be inverted)
        self.assertEqual(player.transform_mocap_rotation_to_genesis("RShoulder", "x", 10.0), -10.0)
        self.assertEqual(player.transform_mocap_rotation_to_genesis("RShoulder", "y", 10.0), -10.0)
        self.assertEqual(player.transform_mocap_rotation_to_genesis("RShoulder", "z", 10.0), -10.0)
    
    def test_joint_index_mapping(self):
        """Test joint name to DOF index mapping"""
        player = MocapGenesisPlayer(self.skeleton_xml, "dummy.json")
        
        # Test known joint indices from our Genesis skeleton
        expected_indices = {
            "hip_flexion_r": 7,
            "knee_angle_r": 16,
            "arm_flex_r": 18,
            "elbow_flex_r": 26,
            "hip_flexion_l": 10,
            "knee_angle_l": 17,
            "arm_flex_l": 21,
            "elbow_flex_l": 27,
        }
        
        for joint_name, expected_index in expected_indices.items():
            actual_index = player.get_joint_index(joint_name)
            self.assertEqual(actual_index, expected_index, 
                           f"Joint {joint_name}: expected index {expected_index}, got {actual_index}")
        
        # Test unknown joint returns None
        self.assertIsNone(player.get_joint_index("unknown_joint"))
    
    def test_euler_quaternion_conversion(self):
        """Test Euler to quaternion conversion"""
        player = MocapGenesisPlayer(self.skeleton_xml, "dummy.json")
        
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
        self.assertAlmostEqual(y, 0.0, places=3)
        self.assertAlmostEqual(z, 0.0, places=3)


class TestRealDataIntegration(unittest.TestCase):
    """Integration tests with real mocap data pipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.base_dir = Path(__file__).parent
        self.local_csv_path = self.base_dir / "mocap_data" / "Take 2025-08-19 03.01.18 PM_LOCAL.csv"
        self.temp_dir = tempfile.mkdtemp()
        self.skeleton_xml = "skeleton/genesis_skeleton.xml"
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_with_real_data(self):
        """Test complete pipeline: CSV -> processing -> JSON -> player"""
        if not self.local_csv_path.exists():
            self.skipTest("LOCAL CSV file not available")
        
        # Step 1: Convert real CSV to JSON
        converter = MocapConverter(str(self.local_csv_path))
        
        converter.load_csv_data()
        self.assertGreater(len(converter.frame_data), 100)  # Real data has many frames
        
        converter.analyze_joint_structure()
        self.assertGreater(len(converter.joint_info), 10)  # Real data has many joints
        
        converter.process_frame_data()
        self.assertEqual(len(converter.processed_data), len(converter.frame_data))
        
        # Save to temp file
        temp_json = os.path.join(self.temp_dir, "test_real_data.json")
        converter.save_processed_data(temp_json)
        self.assertTrue(os.path.exists(temp_json))
        
        # Step 2: Load with mocap player
        player = MocapGenesisPlayer(self.skeleton_xml, temp_json)
        success = player.load_mocap_data()
        self.assertTrue(success)
        
        # Step 3: Verify real joint data exists
        frames = player.mocap_data['frames']
        self.assertGreater(len(frames), 100)  # Real data should have many frames
        
        # Check that we have real joint data
        first_frame = frames[0]
        joints = first_frame['joints']
        self.assertGreater(len(joints), 0)  # Should have some joints from real data
        
        # Test that joint mappings work with real data
        for joint_name in list(joints.keys())[:5]:  # Test first 5 joints
            if 'rotation' in joints[joint_name]:
                for axis in ['x', 'y', 'z']:
                    if axis in joints[joint_name]['rotation']:
                        original_value = joints[joint_name]['rotation'][axis]
                        transformed = player.transform_mocap_rotation_to_genesis(joint_name, axis, original_value)
                        # Should return a numeric value
                        self.assertIsInstance(transformed, (int, float))


def run_real_tests():
    """Run tests with real mocap data"""
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(loader.loadTestsFromTestCase(TestRealMocapData))
    test_suite.addTests(loader.loadTestsFromTestCase(TestRealDataIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running mocap integration tests with real data...")
    success = run_real_tests()
    
    if success:
        print("\n✅ All real data tests passed!")
        exit(0)
    else:
        print("\n❌ Some real data tests failed!")
        exit(1)