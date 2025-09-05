#!/usr/bin/env python3
"""
Motion Capture Data Converter - Fixed Version
Converts quaternion rotations to Euler angles and preserves hierarchical structure
"""

import csv
import numpy as np
import math
import json
from typing import Dict, List, Tuple, Optional

class MocapConverter:
    def __init__(self, csv_file_path: str):
        """Initialize the mocap converter with CSV file path"""
        self.csv_path = csv_file_path
        self.joint_hierarchy = {}
        self.joint_info = {}
        self.processed_data = {}
        self.frame_data = []
        
    def load_csv_data(self):
        """Load and parse the CSV mocap data"""
        print("Loading CSV data...")
        
        # Read CSV file manually to handle variable columns
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        
        # Find header rows
        type_row = None
        name_row = None  
        parent_row = None
        component_row = None
        frame_row = None
        
        for i, line in enumerate(lines):
            if len(line) >= 2:
                if line[1].strip() == 'Type':
                    type_row = i
                elif line[1].strip() == 'Name':
                    name_row = i
                elif line[1].strip() == 'Parent':
                    parent_row = i
                elif line[1].strip() == '' and parent_row is not None and component_row is None:
                    component_row = i
            
            if len(line) >= 1 and line[0].strip() == 'Frame':
                frame_row = i
                break
        
        if frame_row is None:
            raise ValueError("Could not find frame data in CSV")
            
        print(f"Found headers - Type: {type_row}, Name: {name_row}, Parent: {parent_row}, Component: {component_row}, Frame: {frame_row}")
            
        # Extract header information
        if type_row is not None:
            self.types = lines[type_row][2:]
        else:
            self.types = []
            
        if name_row is not None:
            self.names = lines[name_row][2:]
        else:
            self.names = []
            
        if parent_row is not None:
            self.parents = lines[parent_row][2:]
        else:
            self.parents = []
            
        # Use the Frame row to get the actual component names (X,Y,Z,W,X,Y,Z,etc.)
        if frame_row is not None:
            self.components = lines[frame_row][2:]  # Skip 'Frame' and 'Time (Seconds)'
        else:
            self.components = []
        
        # Extract frame data
        self.frame_data = []
        for i in range(frame_row + 1, len(lines)):
            if lines[i] and len(lines[i]) > 2:  # Skip empty lines
                try:
                    frame_num = int(lines[i][0])
                    time = float(lines[i][1])
                    data = [float(x) if x.strip() != '' else 0.0 for x in lines[i][2:]]
                    self.frame_data.append([frame_num, time] + data)
                except (ValueError, IndexError):
                    continue  # Skip malformed lines
        
        print(f"Loaded {len(self.frame_data)} frames of data")
        print(f"Found {len(self.names)} data channels")
        
    def analyze_joint_structure(self):
        """Analyze the joint structure and build hierarchy"""
        print("Analyzing joint structure...")
        
        # Group columns by joint name
        joint_columns = {}
        
        for i, (joint_type, joint_name, parent, component) in enumerate(zip(
            self.types, self.names, self.parents, self.components)):
            
            if not joint_name or joint_name.strip() == '':
                continue
            
            joint_name = str(joint_name).strip()
            joint_type = str(joint_type).strip()
            parent = str(parent).strip()
            component = str(component).strip()
                
            # Clean up joint name
            if ':' in joint_name:
                joint_name = joint_name.split(':')[1]
            
            # Skip non-bone markers for now
            if joint_type not in ['Bone']:
                continue
            
            if joint_name not in joint_columns:
                joint_columns[joint_name] = {
                    'type': joint_type,
                    'parent': parent.split(':')[1] if ':' in parent else parent,
                    'rotation_cols': [],
                    'position_cols': [],
                    'components': []
                }
            
            # Track all components for this joint
            joint_columns[joint_name]['components'].append((i, component))
        
        # Now organize components by rotation/position based on patterns
        for joint_name, info in joint_columns.items():
            components = info['components']
            
            # Sort by index to maintain order
            components.sort(key=lambda x: x[0])
            
            # Look for rotation pattern (X,Y,Z,W) followed by position pattern (X,Y,Z)
            if len(components) >= 4:
                # First 4 should be rotation (X,Y,Z,W)
                rot_components = components[:4]
                if [comp[1] for comp in rot_components] == ['X', 'Y', 'Z', 'W']:
                    info['rotation_cols'] = [comp[0] for comp in rot_components]
                
                # Next 3 should be position (X,Y,Z)
                if len(components) >= 7:
                    pos_components = components[4:7]
                    if [comp[1] for comp in pos_components] == ['X', 'Y', 'Z']:
                        info['position_cols'] = [comp[0] for comp in pos_components]
        
        self.joint_info = joint_columns
        self.build_hierarchy()
        
        print(f"Found {len(self.joint_info)} joints:")
        for joint_name, info in self.joint_info.items():
            print(f"  {joint_name}: parent={info['parent']}, type={info['type']}, rot_cols={len(info['rotation_cols'])}, pos_cols={len(info['position_cols'])}")
    
    def build_hierarchy(self):
        """Build the parent-child hierarchy"""
        self.joint_hierarchy = {}
        
        for joint_name, info in self.joint_info.items():
            parent = info['parent']
            if parent and parent != 'Root' and parent in self.joint_info:
                if parent not in self.joint_hierarchy:
                    self.joint_hierarchy[parent] = []
                self.joint_hierarchy[parent].append(joint_name)
    
    @staticmethod
    def quaternion_to_euler(x, y, z, w, order='xyz'):
        """
        Convert quaternion to Euler angles
        
        Args:
            x, y, z, w: Quaternion components
            order: Rotation order (default: 'xyz')
            
        Returns:
            Tuple of (rx, ry, rz) in radians
        """
        # Normalize quaternion
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return (0, 0, 0)
        
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to Euler angles (XYZ order)
        if order.lower() == 'xyz':
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
            else:
                pitch = math.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return (roll, pitch, yaw)
        
        else:
            raise ValueError(f"Rotation order {order} not implemented")
    
    @staticmethod
    def radians_to_degrees(radians):
        """Convert radians to degrees"""
        return math.degrees(radians)
    
    def process_frame_data(self, euler_degrees=True):
        """Process all frame data converting quaternions to Euler angles"""
        print("Processing frame data...")
        
        processed_frames = []
        
        for frame_row in self.frame_data:
            frame_num = int(frame_row[0])
            time = float(frame_row[1])
            data_values = frame_row[2:]
            
            frame_data = {
                'frame': frame_num,
                'time': time,
                'joints': {}
            }
            
            # Process each joint
            for joint_name, info in self.joint_info.items():
                joint_data = {
                    'name': joint_name,
                    'parent': info['parent'],
                    'type': info['type']
                }
                
                # Process rotation data (quaternion -> euler)
                if info['rotation_cols'] and len(info['rotation_cols']) == 4:
                    rot_cols = info['rotation_cols']
                    try:
                        qx = float(data_values[rot_cols[0]]) if rot_cols[0] < len(data_values) else 0.0
                        qy = float(data_values[rot_cols[1]]) if rot_cols[1] < len(data_values) else 0.0
                        qz = float(data_values[rot_cols[2]]) if rot_cols[2] < len(data_values) else 0.0
                        qw = float(data_values[rot_cols[3]]) if rot_cols[3] < len(data_values) else 1.0
                        
                        # Convert to Euler angles
                        euler_rad = self.quaternion_to_euler(qx, qy, qz, qw)
                        
                        if euler_degrees:
                            euler_angles = tuple(self.radians_to_degrees(rad) for rad in euler_rad)
                        else:
                            euler_angles = euler_rad
                        
                        joint_data['rotation'] = {
                            'x': euler_angles[0],
                            'y': euler_angles[1], 
                            'z': euler_angles[2]
                        }
                        
                        # Also store original quaternion
                        joint_data['quaternion'] = {'x': qx, 'y': qy, 'z': qz, 'w': qw}
                    except (ValueError, IndexError):
                        joint_data['rotation'] = {'x': 0, 'y': 0, 'z': 0}
                        joint_data['quaternion'] = {'x': 0, 'y': 0, 'z': 0, 'w': 1}
                
                # Process position data
                if info['position_cols'] and len(info['position_cols']) >= 3:
                    pos_cols = info['position_cols'][:3]  # Take first 3 for X,Y,Z
                    try:
                        joint_data['position'] = {
                            'x': float(data_values[pos_cols[0]]) if pos_cols[0] < len(data_values) else 0.0,
                            'y': float(data_values[pos_cols[1]]) if pos_cols[1] < len(data_values) else 0.0,
                            'z': float(data_values[pos_cols[2]]) if pos_cols[2] < len(data_values) else 0.0
                        }
                    except (ValueError, IndexError):
                        joint_data['position'] = {'x': 0, 'y': 0, 'z': 0}
                
                frame_data['joints'][joint_name] = joint_data
            
            processed_frames.append(frame_data)
        
        self.processed_data = processed_frames
        print(f"Processed {len(processed_frames)} frames")
    
    def get_joint_hierarchy_dict(self):
        """Return the joint hierarchy as a dictionary"""
        return self.joint_hierarchy
    
    def get_joint_list_by_hierarchy(self):
        """Return joints organized by hierarchy level"""
        hierarchy_levels = {}
        
        def get_level(joint_name, level=0):
            if joint_name in hierarchy_levels:
                return hierarchy_levels[joint_name]
            
            if joint_name not in self.joint_info:
                hierarchy_levels[joint_name] = level
                return level
            
            parent = self.joint_info[joint_name]['parent']
            if parent == 'Root' or parent not in self.joint_info:
                hierarchy_levels[joint_name] = level
                return level
            
            parent_level = get_level(parent, level + 1)
            hierarchy_levels[joint_name] = parent_level + 1
            return parent_level + 1
        
        # Calculate levels for all joints
        for joint_name in self.joint_info:
            get_level(joint_name)
        
        # Group by level
        levels = {}
        for joint, level in hierarchy_levels.items():
            if level not in levels:
                levels[level] = []
            levels[level].append(joint)
        
        return levels
    
    def save_processed_data(self, output_path: str, format_type='json'):
        """Save processed data to file"""
        if format_type.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump({
                    'hierarchy': self.joint_hierarchy,
                    'joint_info': self.joint_info,
                    'frames': self.processed_data
                }, f, indent=2)
        
        print(f"Saved processed data to {output_path}")
    
    def print_summary(self):
        """Print a summary of the processed data"""
        if not self.processed_data:
            print("No processed data available")
            return
        
        print("\n=== MOCAP DATA SUMMARY ===")
        print(f"Total frames: {len(self.processed_data)}")
        print(f"Total joints: {len(self.joint_info)}")
        print(f"Time range: {self.processed_data[0]['time']:.3f} - {self.processed_data[-1]['time']:.3f} seconds")
        
        print("\n=== JOINT HIERARCHY ===")
        hierarchy_levels = self.get_joint_list_by_hierarchy()
        for level in sorted(hierarchy_levels.keys()):
            print(f"Level {level}: {', '.join(hierarchy_levels[level])}")
        
        print("\n=== SAMPLE FRAME DATA (Frame 0) ===")
        if self.processed_data:
            sample_frame = self.processed_data[0]
            for joint_name, joint_data in list(sample_frame['joints'].items())[:5]:  # Show first 5 joints
                print(f"Joint: {joint_name}")
                print(f"  Parent: {joint_data.get('parent', 'None')}")
                if 'rotation' in joint_data:
                    rot = joint_data['rotation']
                    print(f"  Rotation (deg): X={rot['x']:.2f}, Y={rot['y']:.2f}, Z={rot['z']:.2f}")
                if 'position' in joint_data:
                    pos = joint_data['position']
                    print(f"  Position (mm): X={pos['x']:.2f}, Y={pos['y']:.2f}, Z={pos['z']:.2f}")
                print()


def find_csv_files(base_dir):
    """Find all CSV files recursively in the base directory"""
    import os
    import glob
    from datetime import datetime
    
    files = []
    pattern = os.path.join(base_dir, '**/*.csv')
    matches = glob.glob(pattern, recursive=True)
    
    for match in matches:
        try:
            stat = os.stat(match)
            files.append({
                'path': match,
                'name': os.path.basename(match),
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

def display_csv_files(files, start_idx=0, page_size=20):
    """Display CSV files with details"""
    if not files:
        print("No CSV files found!")
        return 0, False
        
    end_idx = min(start_idx + page_size, len(files))
    
    print(f"\nShowing files {start_idx + 1}-{end_idx} of {len(files)}:")
    print("-" * 80)
    print(f"{'#':<3} {'Size':<8} {'Modified':<19} {'Name'}")
    print("-" * 80)
    
    for i in range(start_idx, end_idx):
        file_info = files[i]
        print(f"{i+1:<3} {format_file_size(file_info['size']):<8} "
              f"{file_info['modified'].strftime('%Y-%m-%d %H:%M'):<19} {file_info['name']}")
    
    has_more = end_idx < len(files)
    return end_idx, has_more

def select_csv_interactive(files):
    """Interactive CSV file selection with pagination"""
    if not files:
        print("No CSV files available!")
        return None
    
    # Search functionality
    print("\nOptions:")
    print("1. Show all files")
    print("2. Search by name")
    
    try:
        choice = input("\nSelect option (1-2, default=1): ").strip()
        if choice == "2":
            search_term = input("Enter search term: ").strip().lower()
            if search_term:
                files = [f for f in files if search_term in f['name'].lower()]
                if not files:
                    print("No files match the search term!")
                    return None
    except (ValueError, KeyboardInterrupt):
        pass
    
    # Pagination for large file lists
    page_size = 20
    current_idx = 0
    
    while True:
        end_idx, has_more = display_csv_files(files, current_idx, page_size)
        
        print("\nOptions:")
        print("- Enter file number to select")
        if has_more:
            print("- 'n' for next page")
        if current_idx > 0:
            print("- 'p' for previous page")
        print("- 'q' to quit")
        
        try:
            choice = input(f"\nYour choice: ").strip().lower()
            
            if choice == 'q':
                return None
            elif choice == 'n' and has_more:
                current_idx = end_idx
            elif choice == 'p' and current_idx > 0:
                current_idx = max(0, current_idx - page_size)
            else:
                # Try to parse as file number
                file_num = int(choice) - 1
                if 0 <= file_num < len(files):
                    selected = files[file_num]
                    print(f"Selected: {selected['name']}")
                    return selected['path']
                else:
                    print(f"Invalid selection. Please enter 1-{len(files)}")
                    
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled.")
            continue

def main():
    """Main function with enhanced command line argument support"""
    import argparse
    import os
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert mocap CSV to processed JSON format')
    parser.add_argument('--csv_path', type=str, help='Path to specific CSV file to convert')
    parser.add_argument('--list_files', action='store_true', help='List available CSV files')
    parser.add_argument('--interactive', action='store_true', help='Interactively select from available CSV files')
    parser.add_argument('--search', type=str, help='Search for CSV files containing this term')
    
    args = parser.parse_args()
    
    # Base directory for mocap data
    mocap_base_dir = "/home/choonspin/intuitive_autonomy/integration/Genesis/genesis_loco/mocapdata/"
    
    # Find all CSV files
    print("Scanning for CSV files...")
    csv_files = find_csv_files(mocap_base_dir)
    
    if not csv_files:
        print("No CSV files found in the directory!")
        return
    
    # Handle list files option
    if args.list_files:
        filtered_files = csv_files
        
        # Apply search filter if provided
        if args.search:
            search_term = args.search.lower()
            filtered_files = [f for f in filtered_files if search_term in f['name'].lower()]
        
        if filtered_files:
            display_csv_files(filtered_files)
        else:
            print("No files match the specified criteria.")
        return
    
    # Handle interactive selection
    if args.interactive:
        csv_file = select_csv_interactive(csv_files)
        if not csv_file:
            print("No file selected. Exiting.")
            return
    
    # Handle direct path argument
    elif args.csv_path:
        csv_file = args.csv_path
        if not os.path.exists(csv_file):
            print(f"Error: CSV file not found: {csv_file}")
            return
    
    # Default file selection - use most recent CSV file
    else:
        if csv_files:
            csv_file = csv_files[0]['path']  # Most recent CSV file
            print(f"Using most recent CSV file: {os.path.basename(csv_file)}")
        else:
            print("No CSV files found!")
            return
    
    try:
        # Create converter instance
        converter = MocapConverter(csv_file)
        
        # Process the data step by step
        converter.load_csv_data()
        converter.analyze_joint_structure()
        converter.process_frame_data(euler_degrees=True)
        
        # Print summary
        converter.print_summary()
        
        # Generate output filename from input filename (same base name)
        output_file = csv_file.replace('.csv', '_processed.json')
        
        # Save processed data
        converter.save_processed_data(output_file)
        
        print(f"\nConversion complete! Check {output_file} for the processed data.")
        
    except Exception as e:
        print(f"Error processing mocap data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()