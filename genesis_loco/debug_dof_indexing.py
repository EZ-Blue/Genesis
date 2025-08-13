#!/usr/bin/env python3
"""
Debug DOF Indexing Issues

Diagnose why joint observations are all zeros despite having valid DOF indices.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import genesis as gs

def main():
    print("🔍 Debug DOF Indexing Issues")
    print("=" * 50)
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Import and create environment
    from environments.skeleton_humanoid_refactored import SkeletonHumanoidEnv
    
    env = SkeletonHumanoidEnv(
        num_envs=1,  # Single env for easier debugging
        episode_length_s=5.0,
        dt=0.02,
        use_box_feet=True,
        show_viewer=False
    )
    
    print(f"Environment created: {env.num_envs} envs, {env.num_actions} actions")
    
    # Test 1: Check all DOF values
    print("\n🧪 TEST 1: All DOF Values")
    print("-" * 40)
    
    # Reset to get a known state
    env.reset()
    
    # Get all DOF positions and velocities
    all_dof_pos = env.robot.get_dofs_position()
    all_dof_vel = env.robot.get_dofs_velocity()
    
    print(f"Total DOFs: {env.robot.n_dofs}")
    print(f"All DOF positions shape: {all_dof_pos.shape}")
    print(f"All DOF velocities shape: {all_dof_vel.shape}")
    
    # Show first environment's DOF values
    env_0_pos = all_dof_pos[0].cpu().numpy()
    env_0_vel = all_dof_vel[0].cpu().numpy()
    
    print(f"\nEnvironment 0 DOF positions:")
    for i, pos in enumerate(env_0_pos):
        if abs(pos) > 1e-6:  # Only show non-zero values
            print(f"  DOF {i:2d}: {pos:.6f}")
    
    print(f"\nEnvironment 0 DOF velocities:")
    for i, vel in enumerate(env_0_vel):
        if abs(vel) > 1e-6:  # Only show non-zero values
            print(f"  DOF {i:2d}: {vel:.6f}")
    
    # Test 2: Check motor DOF indices
    print("\n🧪 TEST 2: Motor DOF Index Validation")
    print("-" * 40)
    
    print(f"Number of motor DOFs: {len(env.motors_dof_idx)}")
    print(f"Motor DOF indices: {env.motors_dof_idx}")
    
    # Check if indices are valid
    max_idx = max(env.motors_dof_idx) if env.motors_dof_idx else -1
    min_idx = min(env.motors_dof_idx) if env.motors_dof_idx else -1
    
    print(f"Index range: {min_idx} to {max_idx}")
    print(f"Total robot DOFs: {env.robot.n_dofs}")
    
    if max_idx >= env.robot.n_dofs:
        print(f"❌ CRITICAL: Index {max_idx} >= total DOFs {env.robot.n_dofs}")
    else:
        print("✅ All indices are within valid range")
    
    # Test 3: Check motor DOF values specifically
    print("\n🧪 TEST 3: Motor DOF Values")
    print("-" * 40)
    
    motor_positions = env_0_pos[env.motors_dof_idx]
    motor_velocities = env_0_vel[env.motors_dof_idx]
    
    print(f"Motor DOF positions: {motor_positions}")
    print(f"Motor DOF velocities: {motor_velocities}")
    
    non_zero_pos = sum(1 for x in motor_positions if abs(x) > 1e-6)
    non_zero_vel = sum(1 for x in motor_velocities if abs(x) > 1e-6)
    
    print(f"Non-zero motor positions: {non_zero_pos}/{len(motor_positions)}")
    print(f"Non-zero motor velocities: {non_zero_vel}/{len(motor_velocities)}")
    
    # Test 4: Apply actions and check changes
    print("\n🧪 TEST 4: Action Application Test")
    print("-" * 40)
    
    # Store initial positions
    initial_pos = env.robot.get_dofs_position()[0].clone()
    
    # Apply non-zero actions
    test_actions = torch.ones((1, env.num_actions), device=env.device) * 0.1
    print(f"Applying test actions: {test_actions[0][:5].cpu().numpy()}...")
    
    # Step the environment
    obs, rewards, dones, info = env.step(test_actions)
    
    # Check if positions changed
    final_pos = env.robot.get_dofs_position()[0]
    pos_change = torch.abs(final_pos - initial_pos)
    
    print(f"Position changes > 1e-6:")
    for i, change in enumerate(pos_change):
        if change > 1e-6:
            print(f"  DOF {i:2d}: {change.item():.6f}")
    
    # Check motor DOF changes specifically
    motor_changes = pos_change[env.motors_dof_idx]
    motor_changes_count = sum(1 for x in motor_changes if x > 1e-6)
    print(f"Motor DOF changes: {motor_changes_count}/{len(env.motors_dof_idx)}")
    
    # Test 5: Check joint name to DOF mapping
    print("\n🧪 TEST 5: Joint Name to DOF Mapping")
    print("-" * 40)
    
    print("Sample joint mappings:")
    sample_joints = ["lumbar_extension", "hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
    
    for joint_name in sample_joints:
        if joint_name in env.joint_to_motor_idx:
            dof_idx = env.joint_to_motor_idx[joint_name]
            try:
                # Try to get the joint object
                joint_obj = env.robot.get_joint(joint_name)
                actual_dof_start = joint_obj.dof_start
                
                current_pos = env_0_pos[dof_idx]
                print(f"  {joint_name}: mapped_idx={dof_idx}, actual_dof_start={actual_dof_start}, pos={current_pos:.6f}")
                
                if dof_idx != actual_dof_start:
                    print(f"    ❌ MISMATCH: mapped != actual")
                    
            except Exception as e:
                print(f"  {joint_name}: ERROR - {e}")
        else:
            print(f"  {joint_name}: NOT FOUND in joint_to_motor_idx")
    
    # Test 6: Check observation construction
    print("\n🧪 TEST 6: Observation Construction Debug")
    print("-" * 40)
    
    # Get the actual observation
    obs = env._get_observations()
    obs_0 = obs[0].cpu().numpy()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: 5 (root) + {len(env.motors_dof_idx)} (joint_pos) + 6 (root_vel) + {len(env.motors_dof_idx)} (joint_vel)")
    expected_size = 5 + len(env.motors_dof_idx) + 6 + len(env.motors_dof_idx)
    print(f"Expected size: {expected_size}, Actual size: {obs.shape[1]}")
    
    # Break down observation components
    root_z = obs_0[0]
    root_quat = obs_0[1:5]
    joint_pos = obs_0[5:5+len(env.motors_dof_idx)]
    root_vel = obs_0[5+len(env.motors_dof_idx):11+len(env.motors_dof_idx)]
    joint_vel = obs_0[11+len(env.motors_dof_idx):11+2*len(env.motors_dof_idx)]
    
    print(f"Root z: {root_z:.6f}")
    print(f"Root quat: {root_quat}")
    print(f"Joint positions: {joint_pos}")
    print(f"Root velocities: {root_vel}")
    print(f"Joint velocities: {joint_vel}")
    
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS COMPLETE")
    print("Check above for issues with DOF indexing and action application")

if __name__ == "__main__":
    main()