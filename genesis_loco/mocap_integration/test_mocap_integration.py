#!/usr/bin/env python3
"""
Unit tests for mocap integration package
Tests mocap converter, player, and joint mapping functionality
"""

import unittest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Mock Genesis and torch dependencies before importing
sys.modules['genesis'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Configure torch mock to return reasonable values
torch_mock = MagicMock()
torch_mock.zeros.return_value = [0.0] * 38  # Mock qpos tensor
sys.modules['torch'] = torch_mock

# Import modules to test (after mocking dependencies)
sys.path.append(os.path.dirname(__file__))
from mocap_converter import MocapConverter
from mocap_player_genesis import MocapGenesisPlayer


class TestMocapConverter(unittest.TestCase):
    """Test cases for MocapConverter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.converter = MocapConverter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample CSV data
        self.sample_csv_data = [
            "Frame,Time,Skeleton:Skeleton.position.x,Skeleton:Skeleton.position.y,Skeleton:Skeleton.position.z,"
            "Skeleton:RShoulder.rotation.x,Skeleton:RShoulder.rotation.y,Skeleton:RShoulder.rotation.z",
            "1,0.008333,-50.0,1000.0,100.0,10.5,-5.2,15.8",
            "2,0.016667,-49.8,1001.2,99.8,11.0,-5.0,16.2",
            "3,0.025000,-49.6,1002.4,99.6,11.5,-4.8,16.6"
        ]
        
        self.test_csv_path = os.path.join(self.temp_dir, "test_mocap.csv")
        with open(self.test_csv_path, 'w') as f:
            f.write('\n'.join(self.sample_csv_data))
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_parse_csv_header(self):
        """Test CSV header parsing"""
        with open(self.test_csv_path, 'r') as f:
            lines = f.readlines()
        
        parsed = self.converter.parse_csv_header(lines[0].strip())
        
        self.assertIsInstance(parsed, dict)
        self.assertIn('Frame', parsed)
        self.assertIn('Time', parsed)
        self.assertIn('Skeleton:Skeleton.position.x', parsed)
        self.assertIn('Skeleton:RShoulder.rotation.x', parsed)
    
    def test_convert_csv_to_json(self):
        """Test full CSV to JSON conversion"""
        output_path = os.path.join(self.temp_dir, "test_output.json")
        
        # Run conversion
        success = self.converter.convert_csv_to_json(self.test_csv_path, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify JSON structure
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('metadata', data)
        self.assertIn('frames', data)
        self.assertEqual(len(data['frames']), 3)
        
        # Check frame structure
        frame = data['frames'][0]
        self.assertIn('frame', frame)
        self.assertIn('time', frame)
        self.assertIn('joints', frame)
        
        # Check joints structure
        joints = frame['joints']
        self.assertIn('Skeleton', joints)
        self.assertIn('RShoulder', joints)
        
        # Check position data
        skeleton = joints['Skeleton']
        self.assertIn('position', skeleton)
        self.assertEqual(skeleton['position']['x'], -50.0)
        self.assertEqual(skeleton['position']['y'], 1000.0)
        
        # Check rotation data
        rshoulder = joints['RShoulder']
        self.assertIn('rotation', rshoulder)
        self.assertEqual(rshoulder['rotation']['x'], 10.5)


class TestMocapGenesisPlayer(unittest.TestCase):
    """Test cases for MocapGenesisPlayer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock JSON mocap data
        self.mock_mocap_data = {
            "metadata": {
                "total_frames": 2,
                "duration": 0.016667,
                "frame_rate": 120.0,
                "joints_count": 2
            },
            "frames": [
                {
                    "frame": 1,
                    "time": 0.008333,
                    "joints": {
                        "Skeleton": {
                            "position": {"x": -50.0, "y": 1000.0, "z": 100.0}
                        },
                        "RShoulder": {
                            "rotation": {"x": 10.5, "y": -5.2, "z": 15.8}
                        },
                        "RThigh": {
                            "rotation": {"x": -30.0, "y": 5.0, "z": 2.0}
                        }
                    }
                },
                {
                    "frame": 2,
                    "time": 0.016667,
                    "joints": {
                        "Skeleton": {
                            "position": {"x": -49.8, "y": 1001.2, "z": 99.8}
                        },
                        "RShoulder": {
                            "rotation": {"x": 11.0, "y": -5.0, "z": 16.2}
                        },
                        "RThigh": {
                            "rotation": {"x": -29.5, "y": 5.2, "z": 2.1}
                        }
                    }
                }
            ]
        }
        
        self.test_json_path = os.path.join(self.temp_dir, "test_mocap.json")
        with open(self.test_json_path, 'w') as f:
            json.dump(self.mock_mocap_data, f)
        
        # Mock skeleton XML path (not actually used in unit tests)
        self.skeleton_xml = "mock_skeleton.xml"
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_joint_mapping_structure(self):
        """Test joint mapping dictionary structure"""
        player = MocapGenesisPlayer(self.skeleton_xml, self.test_json_path)
        
        # Test joint mapping exists and has expected structure
        self.assertIsInstance(player.joint_mapping, dict)
        self.assertGreater(len(player.joint_mapping), 20)  # Should have many joints
        
        # Test specific joint mappings
        self.assertIn(("RShoulder", "x"), player.joint_mapping)
        self.assertIn(("RThigh", "x"), player.joint_mapping)
        self.assertIn(("RHand", "y"), player.joint_mapping)
        
        # Test mapping values are strings
        for key, value in player.joint_mapping.items():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)  # (joint_name, axis)
            self.assertIsInstance(value, str)
            self.assertIn(key[1], ['x', 'y', 'z'])  # Valid axis
    
    def test_load_mocap_data(self):
        """Test mocap data loading"""
        player = MocapGenesisPlayer(self.skeleton_xml, self.test_json_path)
        
        success = player.load_mocap_data()
        
        self.assertTrue(success)
        self.assertIsNotNone(player.mocap_data)
        self.assertIn('frames', player.mocap_data)
        self.assertEqual(len(player.mocap_data['frames']), 2)
    
    def test_transform_mocap_rotation_to_genesis(self):
        """Test coordinate system transformation"""
        player = MocapGenesisPlayer(self.skeleton_xml, self.test_json_path)
        
        # Test hip joint transformation (should be inverted)
        transformed = player.transform_mocap_rotation_to_genesis("RThigh", "x", -30.0)
        self.assertEqual(transformed, 30.0)  # Should be inverted
        
        # Test shoulder joint transformation
        transformed = player.transform_mocap_rotation_to_genesis("RShoulder", "x", 10.5)
        self.assertEqual(transformed, -10.5)  # Should be inverted
        
        # Test joint with no transformation
        transformed = player.transform_mocap_rotation_to_genesis("UnknownJoint", "x", 45.0)
        self.assertEqual(transformed, 45.0)  # Should remain unchanged
    
    def test_get_joint_index(self):
        """Test joint name to DOF index mapping"""
        player = MocapGenesisPlayer(self.skeleton_xml, self.test_json_path)
        
        # Test known joint indices
        self.assertEqual(player.get_joint_index("hip_flexion_r"), 7)
        self.assertEqual(player.get_joint_index("knee_angle_r"), 16)
        self.assertEqual(player.get_joint_index("arm_flex_r"), 18)
        
        # Test unknown joint
        self.assertIsNone(player.get_joint_index("unknown_joint"))
    
    def test_convert_euler_to_quaternion(self):
        """Test Euler to quaternion conversion"""
        player = MocapGenesisPlayer(self.skeleton_xml, self.test_json_path)
        
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


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete mocap pipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create more comprehensive CSV test data
        self.integration_csv_data = [
            "Frame,Time,Skeleton:Skeleton.position.x,Skeleton:Skeleton.position.y,Skeleton:Skeleton.position.z,"
            "Skeleton:RThigh.rotation.x,Skeleton:RThigh.rotation.y,Skeleton:RThigh.rotation.z,"
            "Skeleton:RShoulder.rotation.x,Skeleton:RShoulder.rotation.y,Skeleton:RShoulder.rotation.z",
            "1,0.008333,-50.0,1000.0,100.0,-30.0,5.0,2.0,10.5,-5.2,15.8",
            "2,0.016667,-49.8,1001.2,99.8,-29.5,5.2,2.1,11.0,-5.0,16.2",
            "3,0.025000,-49.6,1002.4,99.6,-29.0,5.4,2.2,11.5,-4.8,16.6"
        ]
        
        self.test_csv_path = os.path.join(self.temp_dir, "integration_test.csv")
        with open(self.test_csv_path, 'w') as f:
            f.write('\n'.join(self.integration_csv_data))
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test complete CSV to JSON to mocap player pipeline"""
        # Step 1: Convert CSV to JSON
        converter = MocapConverter()
        json_path = os.path.join(self.temp_dir, "integration_output.json")
        
        success = converter.convert_csv_to_json(self.test_csv_path, json_path)
        self.assertTrue(success)
        
        # Step 2: Load with mocap player
        player = MocapGenesisPlayer("mock_skeleton.xml", json_path)
        success = player.load_mocap_data()
        self.assertTrue(success)
        
        # Step 3: Test joint mapping for loaded data
        frames = player.mocap_data['frames']
        self.assertEqual(len(frames), 3)
        
        # Test specific joint data exists
        first_frame = frames[0]
        joints = first_frame['joints']
        
        self.assertIn('RThigh', joints)
        self.assertIn('RShoulder', joints)
        
        # Test coordinate transformation is applied correctly
        rthigh_x = joints['RThigh']['rotation']['x']  # -30.0 in CSV
        transformed = player.transform_mocap_rotation_to_genesis("RThigh", "x", rthigh_x)
        self.assertEqual(transformed, 30.0)  # Should be inverted


class TestMocapDataValidation(unittest.TestCase):
    """Test cases for mocap data validation and error handling"""
    
    def test_invalid_csv_format(self):
        """Test handling of invalid CSV format"""
        temp_dir = tempfile.mkdtemp()
        invalid_csv = os.path.join(temp_dir, "invalid.csv")
        
        # Create invalid CSV (missing required columns)
        with open(invalid_csv, 'w') as f:
            f.write("WrongColumn1,WrongColumn2\n")
            f.write("1,2\n")
        
        converter = MocapConverter()
        output_path = os.path.join(temp_dir, "output.json")
        
        # Should handle gracefully without crashing
        try:
            success = converter.convert_csv_to_json(invalid_csv, output_path)
            # May succeed or fail gracefully, but shouldn't crash
        except Exception as e:
            # If it throws an exception, it should be informative
            self.assertIsInstance(e, (ValueError, KeyError, FileNotFoundError))
        
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_empty_mocap_data(self):
        """Test handling of empty mocap data"""
        temp_dir = tempfile.mkdtemp()
        empty_json = os.path.join(temp_dir, "empty.json")
        
        # Create empty but valid JSON structure
        empty_data = {"metadata": {}, "frames": []}
        with open(empty_json, 'w') as f:
            json.dump(empty_data, f)
        
        player = MocapGenesisPlayer("mock_skeleton.xml", empty_json)
        success = player.load_mocap_data()
        
        self.assertTrue(success)
        self.assertEqual(len(player.mocap_data['frames']), 0)
        
        import shutil
        shutil.rmtree(temp_dir)


def run_tests():
    """Run all unit tests"""
    # Create test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(loader.loadTestsFromTestCase(TestMocapConverter))
    test_suite.addTests(loader.loadTestsFromTestCase(TestMocapGenesisPlayer))
    test_suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    test_suite.addTests(loader.loadTestsFromTestCase(TestMocapDataValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running mocap integration unit tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)