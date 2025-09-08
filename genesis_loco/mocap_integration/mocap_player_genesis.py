#!/usr/bin/env python3
"""
Genesis Mocap Player - Working Version
Plays motion capture data using Genesis physics simulation with the skeleton model
"""

import genesis as gs
import torch
import numpy as np
import json
import math
from pathlib import Path
import time

class MocapGenesisPlayer:
    def __init__(self, skeleton_xml_path: str, mocap_json_path: str):
        """Initialize the Genesis mocap player"""
        self.skeleton_xml_path = skeleton_xml_path
        self.mocap_json_path = mocap_json_path
        self.mocap_data = None
        self.scene = None
        self.robot = None
        
        # Complete mapping from mocap joints to Genesis skeleton joints
        # Each Genesis joint has only 1 DOF, map best axis from mocap data
        self.joint_mapping = {
            # Spine/torso - map to lumbar joints (without Skeleton: prefix!)
            ("Ab", "x"): "lumbar_extension",      # Forward/back bend
            ("Ab", "y"): "lumbar_bending",     # Left/right bend
            ("Ab", "z"): "lumbar_rotation",     # Rotation/twist

            # Right leg - complete leg chain (without Skeleton: prefix!)
            ("RThigh", "x"): "hip_flexion_r",     # Hip flex/extend (primary walking motion)
            ("RThigh", "y"): "hip_adduction_r",   # Hip abduct/adduct (side-to-side)
            ("RThigh", "z"): "hip_rotation_r",    # Hip internal/external rotation
            ("RShin", "x"): "knee_angle_r",       # Right knee flex/extend (using X-axis)
            ("RFoot", "z"): "ankle_angle_r",      # Ankle dorsi/plantar flex
            ("RFoot", "y"): "subtalar_angle_r",   # Ankle inversion/eversion
            ("RToe", "x"): "mtp_angle_r",         # Toe flex/extend
            
            # Left leg - mirror right leg (without Skeleton: prefix!)
            ("LThigh", "x"): "hip_flexion_l",     # Hip flex/extend
            ("LThigh", "y"): "hip_adduction_l",   # Hip abduct/adduct
            ("LThigh", "z"): "hip_rotation_l",    # Hip internal/external rotation
            ("LShin", "x"): "knee_angle_l",       # Left knee flex/extend (using X-axis for consistency)
            ("LFoot", "z"): "ankle_angle_l",      # Ankle dorsi/plantar flex
            ("LFoot", "y"): "subtalar_angle_l",   # Ankle inversion/eversion
            ("LToe", "x"): "mtp_angle_l",         # Toe flex/extend
            
            # Right arm - complete arm chain (without Skeleton: prefix!)
            ("RShoulder", "x"): "arm_flex_r",     # Shoulder flex/extend
            ("RShoulder", "z"): "arm_add_r",      # Shoulder abduct/adduct
            ("RShoulder", "y"): "arm_rot_r",      # Shoulder internal/external rotation
            ("RUArm", "y"): "elbow_flex_r",       # Elbow flex/extend (forearm controls elbow)
            ("RUArm", "x"): "pro_sup_r",          # Upper arm pronation/supination
            ("RHand", "y"): "wrist_flex_r",       # Wrist flex/extend (try Y-axis)
            ("RHand", "z"): "wrist_dev_r",        # Wrist radial/ulnar deviation (Z-axis)

            # Left arm - mirror right arm (without Skeleton: prefix!)
            ("LShoulder", "x"): "arm_flex_l",     # Shoulder flex/extend
            ("LShoulder", "z"): "arm_add_l",      # Shoulder abduct/adduct
            ("LShoulder", "y"): "arm_rot_l",      # Shoulder internal/external rotation
            ("LFArm", "y"): "elbow_flex_l",       # Elbow flex/extend (forearm controls elbow)
            ("LFArm", "x"): "pro_sup_l",          # Upper arm pronation/supination
            ("LHand", "y"): "wrist_flex_l",       # Wrist flex/extend (try Y-axis)
            ("LHand", "z"): "wrist_dev_l",        # Wrist radial/ulnar deviation (Z-axis)
        }
        
    def load_mocap_data(self):
        """Load the processed mocap data"""
        print("Loading mocap data...")
        with open(self.mocap_json_path, 'r') as f:
            self.mocap_data = json.load(f)
        
        frames = self.mocap_data['frames']
        if frames:
            print(f"Loaded {len(frames)} frames ({frames[-1]['time']:.2f}s)")
        else:
            print("Loaded 0 frames (empty dataset)")
        return True
        
    def setup_genesis(self):
        """Initialize Genesis scene and load the skeleton"""
        print("Setting up Genesis scene...")
        
        # Initialize Genesis
        gs.init()
        
        # Create scene - kinematic mode to prevent ground penetration
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1/120.0,  # 120 FPS to match mocap
                gravity=(0, 0, -9.81),  # Gravity enabled
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.8),
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
            ),
            show_viewer=True,
        )
        
        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane())
        
        # Load skeleton from XML
        print(f"Loading skeleton from {self.skeleton_xml_path}")
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=self.skeleton_xml_path),
        )
        
        # Build scene  
        self.scene.build()
        
        print(f"Robot loaded with {self.robot.n_dofs} DOFs")
        print(f"Robot has {self.robot.n_joints} joints and {self.robot.n_links} links")
        
        return True
        
    def convert_euler_to_quaternion(self, euler_x, euler_y, euler_z):
        """Convert Euler angles (degrees) to quaternion (w, x, y, z)"""
        # Convert degrees to radians
        roll = math.radians(euler_x)   # X rotation
        pitch = math.radians(euler_y)  # Y rotation  
        yaw = math.radians(euler_z)    # Z rotation
        
        # Compute quaternion (w, x, y, z)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return w, x, y, z
        
    def transform_mocap_rotation_to_genesis(self, mocap_joint, axis, angle_deg):
        """Fix coordinate conventions between LOCAL mocap and Genesis"""
        
        # Fix coordinate system differences between mocap and Genesis
        coordinate_fixes = {
            # Hip joints - invert to match Genesis coordinate system
            "RThigh": {"x": -1.0},  # Hip flexion: mocap -73° -> Genesis +73°
            "LThigh": {"x": -1.0},  # Hip flexion: mocap -76° -> Genesis +76°
            
            # Knee joints - invert to match natural knee flexion direction
            "RShin": {"x": -1.0},   # Right knee: mocap +107° -> Genesis -107° (bend backward)
            "LShin": {"x": -1.0},   # Left knee: use X-axis with inversion for consistency
            
            # Torso/spine - may need coordinate adjustment
            "Ab": {"x": 1.0},       # Keep spine extension as is
            
            # Arm joints - check for coordinate system differences
            "RShoulder": {"x": -1.0, "y": -1.0, "z": -1.0},  # Right shoulder - invert all axes
            "LShoulder": {"x": -1.0, "y": -1.0, "z": -1.0},  # Left shoulder - invert all axes
            "RUArm": {"x": -1.0, "y": 1.0, "z": -1.0},    # Right upper arm - don't invert Y (elbow axis)
            "LUArm": {"x": -1.0, "y": -1.0, "z": -1.0},   # Left upper arm - invert all axes
            "RFArm": {"x": -1.0, "y": -1.0, "z": -1.0},   # Right forearm - invert all axes
            "LFArm": {"x": -1.0, "y": -1.0, "z": -1.0},   # Left forearm - invert all axes
            "RHand": {"y": -1.0, "z": -1.0},              # Right hand - invert Y (now wrist_flex), invert Z (wrist_dev)
            "LHand": {"y": -1.0, "z": -1.0},              # Left hand - invert Y (now wrist_flex), invert Z (wrist_dev)
        }
        
        # Apply coordinate system fixes
        if mocap_joint in coordinate_fixes and axis in coordinate_fixes[mocap_joint]:
            correction_factor = coordinate_fixes[mocap_joint][axis]
            angle_deg = angle_deg * correction_factor
        
        # No additional offsets - using pure mocap data
        neutral_position_offsets = {}
        
        joint_axis_key = (mocap_joint, axis)
        if joint_axis_key in neutral_position_offsets:
            angle_deg += neutral_position_offsets[joint_axis_key]
        
        return angle_deg
    
    def mocap_frame_to_genesis_qpos(self, frame_data):
        """Convert a mocap frame to Genesis qpos tensor"""
        joints = frame_data['joints']
        
        # Initialize qpos tensor (matching robot n_qs, not n_dofs!)
        qpos = torch.zeros(self.robot.n_qs)
        
        # Handle root (freejoint) - DOFs 0-6: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        if "Skeleton" in joints:
            skeleton_data = joints["Skeleton"]
            
            # Position (convert from mm to m and adjust coordinate system)
            if 'position' in skeleton_data:
                pos = skeleton_data['position']
                qpos[0] = pos['x'] / 1000.0  # mm to m, X forward
                qpos[1] = pos['z'] / 1000.0  # mm to m, Z is up in mocap -> Y right in Genesis
                qpos[2] = pos['y'] / 1000.0 + 0.2  # mm to m, Y up in mocap -> Z up in Genesis (small offset for kinematic mode)

            # Use the same upright orientation as the joint tester
            qpos[3] = 0.7071  # quat_w (cos(45°))
            qpos[4] = 0.7071  # quat_x (sin(45°)) - 90° rotation around X-axis 
            qpos[5] = 0.0     # quat_y
            qpos[6] = 0.0     # quat_z
        
        # Set skeleton to proper standing position first (before applying mocap data)
        joint_to_dof = {
            "hip_flexion_r": 7, "hip_adduction_r": 8, "hip_rotation_r": 9,
            "hip_flexion_l": 10, "hip_adduction_l": 11, "hip_rotation_l": 12,
            "lumbar_extension": 13, "lumbar_bending": 14, "lumbar_rotation": 15,
            "knee_angle_r": 15, "knee_angle_l": 16,
            "arm_flex_r": 18, "arm_add_r": 19, "arm_rot_r": 20,
            "arm_flex_l": 21, "arm_add_l": 22, "arm_rot_l": 23,
            "ankle_angle_r": 24, "ankle_angle_l": 25,
            "elbow_flex_r": 26, "elbow_flex_l": 27,
            "subtalar_angle_r": 28, "subtalar_angle_l": 29,
            "pro_sup_r": 30, "pro_sup_l": 31,
            "mtp_angle_r": 32, "mtp_angle_l": 33,
            "wrist_flex_r": 34, "wrist_dev_r": 35,
            "wrist_flex_l": 36, "wrist_dev_l": 37,
        }
        self.set_standing_pose(qpos, joint_to_dof)
        
        # Create a dictionary to store joint angles by joint name
        joint_angles = {}
        
        
        # Process mapped joints
        mapped_count = 0
        for (mocap_joint, axis), genesis_joint in self.joint_mapping.items():
            if mocap_joint in joints and 'rotation' in joints[mocap_joint]:
                rot = joints[mocap_joint]['rotation']
                
                # Get the rotation value for the specified axis
                if axis in rot:
                    angle_deg = rot[axis]
                    
                    # Use raw mocap data without heavy scaling or clipping
                    # Apply minimal coordinate transformation if needed
                    transformed_angle = self.transform_mocap_rotation_to_genesis(mocap_joint, axis, angle_deg)
                    
                    # Convert directly to radians without aggressive scaling
                    angle_rad = math.radians(transformed_angle)
                    
                    # Apply reasonable joint limits (allow full human range of motion)
                    angle_rad = np.clip(angle_rad, -3.0, 3.0)  # ~170 degrees each way
                    joint_angles[genesis_joint] = angle_rad
                    mapped_count += 1
                    
        
        print(f"Successfully mapped {mapped_count} joints")
        
        # Set joint angles using Genesis joint names
        for joint_name, angle in joint_angles.items():
            try:
                joint_idx = self.get_joint_index(joint_name)
                if joint_idx is not None:
                    qpos[joint_idx] = angle
            except Exception as e:
                print(f"Warning: Could not set joint {joint_name}: {e}")
        
                
        return qpos
    
    def get_joint_scale_factor(self, mocap_joint, axis, genesis_joint):
        """Get the scaling factor for converting mocap angles to Genesis angles"""
        # Define scaling factors based on joint type and axis
        joint_scales = {
            # All joints set to 1.0 - use full mocap angles
            "lumbar_extension": 1.0,
            "lumbar_bending": 1.0,
            "lumbar_rotation": 1.0,
            
            # Hip joints
            "hip_flexion_r": 1.0, "hip_flexion_l": 1.0,
            "hip_adduction_r": 1.0, "hip_adduction_l": 1.0,
            "hip_rotation_r": 1.0, "hip_rotation_l": 1.0,
            
            # Knee
            "knee_angle_r": 1.0, "knee_angle_l": 1.0,
            
            # Ankle
            "ankle_angle_r": 1.0, "ankle_angle_l": 1.0,
            "subtalar_angle_r": 1.0, "subtalar_angle_l": 1.0,
            
            # Toes
            "mtp_angle_r": 1.0, "mtp_angle_l": 1.0,
            
            # Shoulder
            "arm_flex_r": 1.0, "arm_flex_l": 1.0,
            "arm_add_r": 1.0, "arm_add_l": 1.0,
            "arm_rot_r": 1.0, "arm_rot_l": 1.0,
            
            # Elbow
            "elbow_flex_r": 1.0, "elbow_flex_l": 1.0,
            
            # Forearm/wrist
            "pro_sup_r": 1.0, "pro_sup_l": 1.0,
            "wrist_flex_r": 1.0, "wrist_flex_l": 1.0,
            "wrist_dev_r": 1.0, "wrist_dev_l": 1.0,
        }
        
        return joint_scales.get(genesis_joint, 0.01)  # Default scale factor
    
    def get_joint_index(self, joint_name):
        """Get the DOF index for a Genesis joint name"""
        # Use the manually mapped DOF indices you discovered
        joint_to_dof = {
            "hip_flexion_r": 7,
            "hip_adduction_r": 8, 
            "hip_rotation_r": 9,
            "hip_flexion_l": 10,
            "hip_adduction_l": 11,
            "hip_rotation_l": 12,
            "lumbar_extension": 13,
            "lumbar_bending": 14,
            "lumbar_rotation": 15,
            "knee_angle_r": 16,
            "knee_angle_l": 17,
            "arm_flex_r": 18,
            "arm_add_r": 19,
            "arm_rot_r": 20,  # Fixed from "arm_rotation_r"
            "arm_flex_l": 21,
            "arm_add_l": 22,
            "arm_rot_l": 23,  # Fixed from "arm_rotation_l"
            "ankle_angle_r": 24,
            "ankle_angle_l": 25,
            "elbow_flex_r": 26,
            "elbow_flex_l": 27,
            "subtalar_angle_r": 28,
            "subtalar_angle_l": 29,
            "pro_sup_r": 30,
            "pro_sup_l": 31,
            "mtp_angle_r": 32,
            "mtp_angle_l": 33,
            "wrist_flex_r": 34,
            "wrist_dev_r": 35,
            "wrist_flex_l": 36,
            "wrist_dev_l": 37,
        }
        
        return joint_to_dof.get(joint_name, None)
    
    def set_standing_pose(self, qpos, joint_to_dof):
        """Set skeleton to a proper neutral standing pose"""
        # Reset all joint angles to neutral/zero (standing position)
        for joint_name, dof_idx in joint_to_dof.items():
            qpos[dof_idx] = 0.0
        
        
    def play_mocap(self, frame_rate=30, start_frame=0, end_frame=None, loop=False):
        """Play the mocap animation"""
        if not self.mocap_data:
            print("No mocap data loaded!")
            return
            
        frames = self.mocap_data['frames']
        if end_frame is None:
            end_frame = len(frames)
            
        print(f"Playing frames {start_frame} to {end_frame} at {frame_rate} FPS")
        
        # Calculate frame skip for desired playback rate
        frame_skip = max(1, int(120 / frame_rate))  # Mocap is 120fps
        
        try:
            while True:
                for i in range(start_frame, min(end_frame, len(frames)), frame_skip):
                    frame = frames[i]
                    
                    # Convert mocap frame to Genesis pose
                    qpos = self.mocap_frame_to_genesis_qpos(frame)
                    
                    # Set robot pose (kinematic mode)
                    self.robot.set_qpos(qpos)
                    
                    # Step scene for visualization only (no physics with gravity=0)
                    self.scene.step()
                    
                    # Control playback speed
                    time.sleep(1.0 / frame_rate)
                    
                    if i % 30 == 0:
                        print(f"Frame {i}/{len(frames)} (t={frame['time']:.2f}s)", end='\r')
                    
                    # Check if viewer is still alive
                    if not self.scene.viewer.is_alive():
                        print("\nViewer closed")
                        return
                
                if not loop:
                    break
                    
                print(f"\nLoop complete! Restarting...")
                
        except KeyboardInterrupt:
            print("\nPlayback interrupted by user")
        except Exception as e:
            print(f"\nError during playback: {e}")
            import traceback
            traceback.print_exc()
            
        print("\nPlayback complete!")
        
    def interactive_viewer(self):
        """Start interactive viewer with initial pose"""
        print("Starting interactive viewer...")
        print("Viewer controls:")
        print("- Mouse: Rotate camera")
        print("- Mouse wheel: Zoom")
        print("- Arrow keys: Move camera")
        print("- 'R': Reset camera")
        print("- Close window to quit")
        
        # Set initial pose from first frame
        if self.mocap_data and self.mocap_data['frames']:
            qpos = self.mocap_frame_to_genesis_qpos(self.mocap_data['frames'][0])
            self.robot.set_qpos(qpos)
            self.scene.step()
        
        # Keep viewer open
        try:
            while self.scene.viewer.is_alive():
                self.scene.step()
                time.sleep(0.016)  # ~60 FPS
        except KeyboardInterrupt:
            print("Viewer closed by user")


def main():
    """Main function with command line argument support"""
    import argparse
    import glob
    import os
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Play mocap animation with Genesis skeleton')
    parser.add_argument('--json_path', type=str, help='Path to specific JSON mocap file')
    parser.add_argument('--list_files', action='store_true', help='List available mocap JSON files')
    parser.add_argument('--interactive', action='store_true', help='Interactively select from available files')
    
    args = parser.parse_args()
    
    # File paths
    skeleton_xml = "/home/choonspin/intuitive_autonomy/integration/Genesis/genesis_loco/skeleton/genesis_skeleton_torque_box_feet.xml"
    
    # Find all available JSON files
    json_pattern = "mocap_data/*_processed.json"
    available_files = glob.glob(json_pattern)
    available_files.sort()
    
    # If no files in csv directory, also check mocapdata directory as fallback
    if not available_files:
        json_pattern_fallback = "/home/choonspin/intuitive_autonomy/integration/Genesis/genesis_loco/mocapdata/drive-download-20250829T005222Z-1-001/*_processed.json"
        available_files = glob.glob(json_pattern_fallback)
        available_files.sort()
    
    # Handle list files option
    if args.list_files:
        print("Available mocap JSON files:")
        for i, file_path in enumerate(available_files, 1):
            filename = os.path.basename(file_path)
            print(f"{i}. {filename}")
        return
    
    # Handle interactive selection
    if args.interactive:
        if not available_files:
            print("No processed JSON files found!")
            return
            
        print("Available mocap JSON files:")
        for i, file_path in enumerate(available_files, 1):
            filename = os.path.basename(file_path)
            print(f"{i}. {filename}")
        
        try:
            choice = int(input(f"\nSelect file (1-{len(available_files)}): ")) - 1
            if 0 <= choice < len(available_files):
                mocap_json = available_files[choice]
                print(f"Selected: {os.path.basename(mocap_json)}")
            else:
                print("Invalid selection. Using default file.")
                mocap_json = available_files[0] if available_files else None
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Using default file.")
            mocap_json = available_files[0] if available_files else None
    
    # Handle direct path argument
    elif args.json_path:
        mocap_json = args.json_path
        if not os.path.exists(mocap_json):
            print(f"Error: JSON file not found: {mocap_json}")
            return
    
    # Default file selection
    else:
        if available_files:
            mocap_json = available_files[0]  # Use first available file
            print(f"Using default file: {os.path.basename(mocap_json)}")
        else:
            print("No processed JSON files found! Please convert some mocap data first.")
            return
    
    # Check if files exist
    if not Path(skeleton_xml).exists():
        print(f"Skeleton XML not found: {skeleton_xml}")
        return
        
    if not Path(mocap_json).exists():
        print(f"Mocap JSON not found: {mocap_json}")
        print("Please run: python mocap_converter_fixed.py")
        return
        
    try:
        # Create player
        player = MocapGenesisPlayer(skeleton_xml, mocap_json)
        
        # Load data and setup Genesis
        player.load_mocap_data()
        player.setup_genesis()
        
        # Choose playback mode
        print("\nChoose playback mode:")
        print("1. Auto-play animation (30 FPS)")
        print("2. Auto-play animation with loop")
        print("3. Interactive viewer (static pose)")
        
        choice = "2"  # Default to auto-play for testing
        
        if choice == "1":
            # Auto-play the animation once
            print("Starting auto-play...")
            player.play_mocap(frame_rate=30, start_frame=100, end_frame=None, loop=False)  # Skip first 100 frames
            
            # Keep viewer open after playback
            input("Press Enter to close viewer...")
            
        elif choice == "2":
            # Auto-play with loop
            print("Starting looped playback...")
            player.play_mocap(frame_rate=30, start_frame=0, end_frame=None, loop=True)  # ~3 seconds looped
            
        elif choice == "3":
            # Interactive mode
            player.interactive_viewer()
            
        else:
            print("Invalid choice. Starting interactive viewer...")
            player.interactive_viewer()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()