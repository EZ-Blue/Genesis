#!/usr/bin/env python3
"""
Simple Joint Tester - Test one DOF at a time without physics interference
"""

import genesis as gs
import torch
import numpy as np
import time
from pathlib import Path

def test_joint_mapping():
    """Simple joint mapping test"""
    skeleton_xml = "/home/choonspin/intuitive_autonomy/integration/Genesis/genesis_loco/skeleton/genesis_skeleton_torque_box_feet.xml"
    
    if not Path(skeleton_xml).exists():
        print(f"Skeleton XML not found: {skeleton_xml}")
        return
    
    # Initialize Genesis
    gs.init()
    
    # Create scene with no physics
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1/60.0,
            gravity=(0, 0, 0),
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
    
    # Add ground and robot
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=skeleton_xml))
    
    # Build scene
    scene.build()
    
    print(f"Robot loaded with {robot.n_dofs} DOFs, {robot.n_qs} qpos elements")
    
    # Get initial pose
    qpos = robot.get_qpos().clone()
    
    # Set upright root position with correct orientation
    qpos[0] = 0.0   # x position
    qpos[1] = 0.0   # y position  
    qpos[2] = 1.0   # z position (1m up)
    # Rotate 90 degrees around Z-axis to face forward (X-axis forward in Genesis)
    qpos[3] = 0.7071   # quat_w (cos(45°))
    qpos[4] = 0.7071      # quat_x
    qpos[5] = 0.0      # quat_y  
    qpos[6] = 0.0  # quat_z (sin(45°)) - 90° rotation around Z
    qpos[7:] = 0.0  # All joint angles to zero
    
    robot.set_qpos(qpos)
    
    print("\n=== JOINT TESTER ===")
    print("Commands:")
    print("  test <dof> [angle_deg]  - Test DOF with angle (default 30°)")
    print("  reset                   - Reset all joints to zero")
    print("  quit                    - Exit")
    print(f"DOF range: 0-{len(qpos)-1}")
    print("DOFs 0-2: Root position (x, y, z)")
    print("DOFs 3-6: Root orientation (quat_w, quat_x, quat_y, quat_z)") 
    print("DOFs 7-37: Joint angles")
    
    test_angle = 30.0  # degrees
    
    try:
        while scene.viewer.is_alive():
            try:
                print(f"\nCommands: test <dof> [angle], reset, quit (test angle = {test_angle}°)")
                cmd = input("Enter command: ").strip()
                
                if cmd.lower() in ['quit', 'q', 'exit']:
                    break
                    
                elif cmd.lower() in ['reset', 'r']:
                    # Reset to proper upright root position and orientation
                    qpos[0] = 0.0     # x position
                    qpos[1] = 0.0     # y position  
                    qpos[2] = 1.0     # z position (1m up)
                    qpos[3] = 0.7071  # quat_w (cos(45°))
                    qpos[4] = 0.7071  # quat_x
                    qpos[5] = 0.0     # quat_y  
                    qpos[6] = 0.0     # quat_z
                    qpos[7:] = 0.0    # Reset all joint angles to zero
                    robot.set_qpos(qpos)
                    scene.step()  # Update visualization
                    print("Reset to upright position and all joints to zero")
                    
                elif cmd.startswith('test '):
                    parts = cmd.split()
                    dof_idx = int(parts[1])
                    
                    if len(parts) > 2:
                        angle_deg = float(parts[2])
                    else:
                        angle_deg = test_angle
                    
                    if 0 <= dof_idx < len(qpos):
                        # Reset all joints first (keep root position/orientation)
                        if dof_idx >= 7:
                            qpos[7:] = 0.0  # Reset only joint angles
                        
                        # Set the specific DOF
                        if dof_idx <= 2:  # Position DOFs
                            qpos[dof_idx] = angle_deg / 100.0  # Convert to meters for position
                        elif dof_idx <= 6:  # Orientation DOFs (quaternion)
                            if dof_idx == 3:  # quat_w should be close to 1 for small rotations
                                qpos[dof_idx] = max(0.1, np.cos(np.radians(angle_deg)/2))
                            else:  # quat_x, quat_y, quat_z
                                qpos[dof_idx] = np.sin(np.radians(angle_deg)/2)
                        else:  # Joint angles
                            qpos[dof_idx] = np.radians(angle_deg)
                            
                        robot.set_qpos(qpos)
                        scene.step()  # Update visualization immediately
                        
                        if dof_idx <= 2:
                            print(f"Set DOF {dof_idx} (position) to {angle_deg/100.0:.2f}m")
                        elif dof_idx <= 6:
                            print(f"Set DOF {dof_idx} (quaternion) to {angle_deg}°")
                        else:
                            print(f"Set DOF {dof_idx} (joint) to {angle_deg}°")
                    else:
                        print(f"DOF {dof_idx} out of range (0-{len(qpos)-1})")
                        
                elif cmd.startswith('angle '):
                    test_angle = float(cmd.split()[1])
                    print(f"Test angle set to {test_angle}°")
                    
                else:
                    print("Unknown command. Use: test <dof>, reset, quit")
                    
            except (ValueError, IndexError) as e:
                print(f"Invalid command: {e}")
            except KeyboardInterrupt:
                break
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    test_joint_mapping()