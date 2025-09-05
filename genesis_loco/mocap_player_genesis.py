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
            ("RShin", "x"): "knee_angle_r",       # Knee flex/extend (X-axis has largest value: 5.30°)
            ("RFoot", "x"): "ankle_angle_r",      # Ankle dorsi/plantar flex
            ("RFoot", "y"): "subtalar_angle_r",   # Ankle inversion/eversion
            ("RToe", "x"): "mtp_angle_r",         # Toe flex/extend
            
            # Left leg - mirror right leg (without Skeleton: prefix!)
            ("LThigh", "x"): "hip_flexion_l",     # Hip flex/extend
            ("LThigh", "y"): "hip_adduction_l",   # Hip abduct/adduct
            ("LThigh", "z"): "hip_rotation_l",    # Hip internal/external rotation
            ("LShin", "y"): "knee_angle_l",       # Knee flex/extend (Y-axis has largest value: 10.63°)
            ("LFoot", "x"): "ankle_angle_l",      # Ankle dorsi/plantar flex
            ("LFoot", "y"): "subtalar_angle_l",   # Ankle inversion/eversion
            ("LToe", "x"): "mtp_angle_l",         # Toe flex/extend
            
            # Right arm - complete arm chain (without Skeleton: prefix!)
            ("RShoulder", "x"): "arm_flex_r",     # Shoulder flex/extend
            ("RShoulder", "z"): "arm_add_r",      # Shoulder abduct/adduct
            ("RShoulder", "y"): "arm_rot_r",      # Shoulder internal/external rotation
            ("RUArm", "x"): "elbow_flex_r",       # Elbow flex/extend
            ("RFArm", "y"): "pro_sup_r",          # Forearm pronation/supination
            ("RHand", "z"): "wrist_flex_r",       # Wrist flex/extend
            ("RHand", "x"): "wrist_dev_r",        # Wrist radial/ulnar deviation

            # Left arm - mirror right arm (without Skeleton: prefix!)
            ("LShoulder", "x"): "arm_flex_l",     # Shoulder flex/extend
            ("LShoulder", "z"): "arm_add_l",      # Shoulder abduct/adduct
            ("LShoulder", "y"): "arm_rot_l",      # Shoulder internal/external rotation
            ("LUArm", "x"): "elbow_flex_l",       # Elbow flex/extend
            ("LFArm", "y"): "pro_sup_l",          # Forearm pronation/supination
            ("LHand", "z"): "wrist_flex_l",       # Wrist flex/extend
            ("LHand", "x"): "wrist_dev_l",        # Wrist radial/ulnar deviation
        }
        
    def load_mocap_data(self):
        """Load the processed mocap data"""
        print("Loading mocap data...")
        with open(self.mocap_json_path, 'r') as f:
            self.mocap_data = json.load(f)
        
        frames = self.mocap_data['frames']
        print(f"Loaded {len(frames)} frames ({frames[-1]['time']:.2f}s)")
        return True
        
    def setup_genesis(self):
        """Initialize Genesis scene and load the skeleton"""
        print("Setting up Genesis scene...")
        
        # Initialize Genesis
        gs.init()
        
        # Create scene - kinematic mode with no physics
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1/120.0,  # 120 FPS to match mocap
                gravity=(0, 0, 0),  # No gravity
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
        """Simple inversion fixes for specific problematic joints"""
        
        # Apply joint-specific inversions for natural human motion
        joint_inversions = {
            # Hip flexion - invert to fix backward legs
            "RThigh": {"x": -1.0},  # Hip flexion
            "LThigh": {"x": -1.0},  # Hip flexion
            
            # Knee joints - use the axis with largest values
            "RShin": {"x": -1.0},  # Knee flexion (X-axis: 5.30°)
            "LShin": {"y": -1.0},  # Knee flexion (Y-axis: 10.63° - much larger!)

            # Wrist/hand rotations - invert Z-axis to point hands in -Z direction
            "RHand": {"z": -1.0, "y": -1.0},  # Wrist rotations (Z-axis for -Z pointing)
            "LHand": {"z": -1.0, "y": -1.0},  # Wrist rotations (Z-axis for -Z pointing)

            # Arms - may need inversion for natural positioning
            "RUArm": {"z": -1.0},  # Arm rotation
            "LUArm": {"z": -1.0},  # Arm rotation
            "RFArm": {"y": -1.0},  # Forearm rotation  
            "LFArm": {"y": -1.0},  # Forearm rotation
        }
        
        # Apply joint-specific inversions
        if mocap_joint in joint_inversions and axis in joint_inversions[mocap_joint]:
            inversion_factor = joint_inversions[mocap_joint][axis]
            angle_deg = angle_deg * inversion_factor
        
        # Apply rest pose offsets to make joints upright in neutral position
        rest_pose_offsets = {
            # Add offset to make knees bend naturally in standing position (like the spine)
            ("RShin", "x"): -90.0,  # Offset to make right knee upright
            ("LShin", "y"): -90.0,  # Offset to make left knee upright
            
            # Add offset to make wrists point down (-Z direction) instead of backwards
            ("RHand", "z"): 90.0,  # Offset to make right wrist point down
            ("LHand", "z"): 90.0,  # Offset to make left wrist point down
        }
        
        joint_axis_key = (mocap_joint, axis)
        if joint_axis_key in rest_pose_offsets:
            angle_deg += rest_pose_offsets[joint_axis_key]
        
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
                qpos[2] = pos['y'] / 1000.0 + 0.1  # mm to m, Y up in mocap -> Z up in Genesis +offset

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
        
        # Debug: Print available joints to see what we have
        all_joints = list(joints.keys())
        print(f"Available joints in frame: {all_joints[:10]}...")  # First 10 joints
        
        # Check if we have knee joints specifically
        knee_joints = [j for j in all_joints if 'Shin' in j]
        if knee_joints:
            print(f"Found knee joints: {knee_joints}")
        else:
            print("No knee joints found with 'Shin' in name")
        
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
                    
                    # Debug first few successful mappings and knee joints specifically
                    if mapped_count <= 3 or 'shin' in mocap_joint.lower():
                        print(f"✓ Mapped {mocap_joint}:{axis} -> {genesis_joint} = {angle_deg:.1f}° -> {angle_rad:.3f} rad")
        
        print(f"Successfully mapped {mapped_count} joints")
        
        # Set joint angles using Genesis joint names
        for joint_name, angle in joint_angles.items():
            try:
                # Find the index of this joint in the robot's DOF list
                joint_idx = self.get_joint_index(joint_name)
                if joint_idx is not None:
                    qpos[joint_idx] = angle
                    # Debug knee joints specifically
                    if 'knee' in joint_name:
                        print(f"🔧 Setting {joint_name} (DOF {joint_idx}) = {angle:.3f} rad ({angle*180/3.14159:.1f}°)")
            except Exception as e:
                print(f"Warning: Could not set joint {joint_name}: {e}")
        
        # Dynamic knee calculation with enhanced debugging
        # Get current hip flexion angles (in radians) 
        hip_flexion_r = joint_angles.get('hip_flexion_r', 0.0)
        hip_flexion_l = joint_angles.get('hip_flexion_l', 0.0)
        
        # Get mocap knee angles that were applied earlier
        mocap_knee_r = joint_angles.get('knee_angle_r', 0.0)
        mocap_knee_l = joint_angles.get('knee_angle_l', 0.0)
        
        # Calculate dynamic knee angles based on hip flexion
        # More hip flexion (sitting) = more knee bend (more negative)
        # Less hip flexion (standing) = less knee bend (closer to 0)
        knee_r_dynamic = max(-1.5, min(-0.2, -abs(hip_flexion_r) - 0.5))
        knee_l_dynamic = max(-1.5, min(-0.2, -abs(hip_flexion_l) - 0.5))
        
        # Debug: Compare mocap vs dynamic values
        print(f"🦵 Knee Analysis:")
        print(f"   Hip flexion: R={hip_flexion_r:.3f} ({hip_flexion_r*180/3.14159:.1f}°), L={hip_flexion_l:.3f} ({hip_flexion_l*180/3.14159:.1f}°)")
        print(f"   Mocap knees: R={mocap_knee_r:.3f} ({mocap_knee_r*180/3.14159:.1f}°), L={mocap_knee_l:.3f} ({mocap_knee_l*180/3.14159:.1f}°)")
        print(f"   Dynamic knees: R={knee_r_dynamic:.3f} ({knee_r_dynamic*180/3.14159:.1f}°), L={knee_l_dynamic:.3f} ({knee_l_dynamic*180/3.14159:.1f}°)")
        
        # Check what values are currently in qpos for knees
        current_knee_r = qpos[16] if len(qpos) > 16 else 0.0
        current_knee_l = qpos[17] if len(qpos) > 17 else 0.0
        print(f"   Current qpos[16]={current_knee_r:.3f} ({current_knee_r*180/3.14159:.1f}°), qpos[17]={current_knee_l:.3f} ({current_knee_l*180/3.14159:.1f}°)")
        
        # Apply dynamic knee calculation (temporarily to debug)
        qpos[16] = knee_r_dynamic  # Right knee 
        qpos[17] = knee_l_dynamic  # Left knee
        
        print(f"   Applied dynamic: qpos[16]={knee_r_dynamic:.3f}, qpos[17]={knee_l_dynamic:.3f}")
        print(f"   DOF 16 & 17 should be knee_angle_r & knee_angle_l")
        print()
                
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
        
        # Apply small adjustments for natural standing pose
        # Slight knee bend for stability
        qpos[joint_to_dof["knee_angle_r"]] = 0.05  # ~3 degrees
        qpos[joint_to_dof["knee_angle_l"]] = 0.05  # ~3 degrees
        
        # Arms hanging naturally at sides
        qpos[joint_to_dof["arm_flex_r"]] = -0.1   # Slight extension
        qpos[joint_to_dof["arm_flex_l"]] = -0.1   # Slight extension
        
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


def find_mocap_files(base_dir, file_types=['json', 'bvh', 'csv']):
    """Find all mocap files recursively in the base directory"""
    import os
    import glob
    from datetime import datetime
    
    files = []
    
    # Search patterns for different file types
    patterns = {
        'json': '**/*_processed.json',
        'bvh': '**/*.bvh', 
        'csv': '**/*.csv'
    }
    
    for file_type in file_types:
        if file_type in patterns:
            pattern = os.path.join(base_dir, patterns[file_type])
            matches = glob.glob(pattern, recursive=True)
            
            for match in matches:
                try:
                    stat = os.stat(match)
                    files.append({
                        'path': match,
                        'name': os.path.basename(match),
                        'type': file_type.upper(),
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime)
                    })
                except OSError:
                    continue
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x['modified'], reverse=True)
    return files

def format_file_size(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}TB"

def filter_files_interactive(files):
    """Interactive filtering and search of files"""
    print("\nFile filtering options:")
    print("1. Show all files")
    print("2. JSON files only")  
    print("3. BVH files only")
    print("4. CSV files only")
    print("5. Search by name")
    
    try:
        filter_choice = input("\nSelect filter (1-5, default=1): ").strip()
        if not filter_choice:
            filter_choice = "1"
            
        if filter_choice == "1":
            return files
        elif filter_choice == "2":
            return [f for f in files if f['type'] == 'JSON']
        elif filter_choice == "3":
            return [f for f in files if f['type'] == 'BVH']
        elif filter_choice == "4":
            return [f for f in files if f['type'] == 'CSV']
        elif filter_choice == "5":
            search_term = input("Enter search term: ").strip().lower()
            if search_term:
                return [f for f in files if search_term in f['name'].lower()]
            else:
                return files
        else:
            print("Invalid choice, showing all files.")
            return files
            
    except (ValueError, KeyboardInterrupt):
        return files

def display_files_with_details(files, start_idx=0, page_size=20):
    """Display files with detailed information in pages"""
    if not files:
        print("No files found!")
        return 0, False
        
    end_idx = min(start_idx + page_size, len(files))
    
    print(f"\nShowing files {start_idx + 1}-{end_idx} of {len(files)}:")
    print("-" * 80)
    print(f"{'#':<3} {'Type':<4} {'Size':<8} {'Modified':<19} {'Name'}")
    print("-" * 80)
    
    for i in range(start_idx, end_idx):
        file_info = files[i]
        print(f"{i+1:<3} {file_info['type']:<4} {format_file_size(file_info['size']):<8} "
              f"{file_info['modified'].strftime('%Y-%m-%d %H:%M'):<19} {file_info['name']}")
    
    has_more = end_idx < len(files)
    return end_idx, has_more

def select_file_interactive(files):
    """Enhanced interactive file selection with pagination and search"""
    if not files:
        print("No files available!")
        return None
        
    # Apply filters if desired
    filtered_files = filter_files_interactive(files)
    
    if not filtered_files:
        print("No files match the filter criteria!")
        return None
    
    # Pagination for large file lists
    page_size = 20
    current_idx = 0
    
    while True:
        end_idx, has_more = display_files_with_details(filtered_files, current_idx, page_size)
        
        print("\nOptions:")
        print("- Enter file number to select")
        if has_more:
            print("- 'n' for next page")
        if current_idx > 0:
            print("- 'p' for previous page")
        print("- 'q' to quit")
        print("- 'r' to refresh/re-filter")
        
        try:
            choice = input(f"\nYour choice: ").strip().lower()
            
            if choice == 'q':
                return None
            elif choice == 'n' and has_more:
                current_idx = end_idx
            elif choice == 'p' and current_idx > 0:
                current_idx = max(0, current_idx - page_size)
            elif choice == 'r':
                filtered_files = filter_files_interactive(files)
                current_idx = 0
            else:
                # Try to parse as file number
                file_num = int(choice) - 1
                if 0 <= file_num < len(filtered_files):
                    selected = filtered_files[file_num]
                    print(f"Selected: {selected['name']}")
                    return selected['path']
                else:
                    print(f"Invalid selection. Please enter 1-{len(filtered_files)}")
                    
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled.")
            continue

def main():
    """Main function with enhanced command line argument support"""
    import argparse
    import os
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Play mocap animation with Genesis skeleton')
    parser.add_argument('--json_path', type=str, help='Path to specific JSON mocap file')
    parser.add_argument('--list_files', action='store_true', help='List available mocap files with details')
    parser.add_argument('--interactive', action='store_true', help='Interactively browse and select from available files')
    parser.add_argument('--filter', choices=['json', 'bvh', 'csv'], help='Filter files by type when listing')
    parser.add_argument('--search', type=str, help='Search for files containing this term')
    
    args = parser.parse_args()
    
    # File paths
    skeleton_xml = "/home/choonspin/intuitive_autonomy/integration/Genesis/genesis_loco/skeleton/genesis_skeleton_torque_box_feet.xml"
    mocap_base_dir = "/home/choonspin/intuitive_autonomy/integration/Genesis/genesis_loco/mocapdata/"
    
    # Find all mocap files
    print("Scanning for mocap files...")
    all_files = find_mocap_files(mocap_base_dir)
    
    if not all_files:
        print("No mocap files found in the directory!")
        return
    
    # Handle list files option
    if args.list_files:
        filtered_files = all_files
        
        # Apply command line filters
        if args.filter:
            filtered_files = [f for f in filtered_files if f['type'].lower() == args.filter]
            
        if args.search:
            search_term = args.search.lower()
            filtered_files = [f for f in filtered_files if search_term in f['name'].lower()]
        
        if filtered_files:
            display_files_with_details(filtered_files)
        else:
            print("No files match the specified criteria.")
        return
    
    # Handle interactive selection
    if args.interactive:
        mocap_json = select_file_interactive(all_files)
        if not mocap_json:
            print("No file selected. Exiting.")
            return
    
    # Handle direct path argument
    elif args.json_path:
        mocap_json = args.json_path
        if not os.path.exists(mocap_json):
            print(f"Error: JSON file not found: {mocap_json}")
            return
    
    # Default file selection - use most recent JSON file
    else:
        json_files = [f for f in all_files if f['type'] == 'JSON']
        if json_files:
            mocap_json = json_files[0]['path']  # Most recent JSON file
            print(f"Using most recent JSON file: {os.path.basename(mocap_json)}")
        else:
            print("No processed JSON files found! Please convert some mocap data first.")
            print("Available files:")
            display_files_with_details(all_files[:10])  # Show first 10 files
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