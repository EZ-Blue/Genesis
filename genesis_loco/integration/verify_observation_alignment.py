#!/usr/bin/env python3
"""
Simple Debug Script to Verify Observation Alignment

This script loads the Genesis skeleton model and expert trajectory data,
then prints detailed observations with indices and joint names to verify alignment.
"""

import torch
import numpy as np
import sys
import os

# Import paths
sys.path.append('/home/ez/Documents/Genesis/genesis_loco')
sys.path.append('/home/ez/Documents/loco-mujoco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge


def main():
    print("🔍 OBSERVATION ALIGNMENT VERIFICATION")
    print("=" * 60)
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create Genesis environment
    print("Creating Genesis SkeletonHumanoidEnv...")
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=5.0,
        dt=0.02,
        use_box_feet=True,
        show_viewer=False
    )
    
    # Create data bridge and load trajectory
    print("Loading expert trajectory data...")
    data_bridge = LocoMujocoDataBridge(env)
    success = data_bridge.load_trajectory("walk")
    
    if not success:
        print("❌ Failed to load trajectory!")
        return
    
    print(f"✅ Loaded trajectory: {data_bridge.trajectory_length} timesteps")
    
    # Get Genesis policy observation
    print("\n" + "=" * 60)
    print("GENESIS POLICY OBSERVATIONS")
    print("=" * 60)
    
    obs, _ = env.reset()
    policy_obs = obs[0]  # First environment
    
    print(f"Policy observation shape: {policy_obs.shape}")
    print(f"Policy observation full values:")
    for i, val in enumerate(policy_obs):
        print(f"  [{i:2d}]: {val.item():.6f}")
    
    # Parse policy observation components
    print(f"\nPolicy observation components:")
    idx = 0
    
    # q_root_no_xy (5D): [z, quat_w, quat_x, quat_y, quat_z]
    print(f"  q_root_no_xy [{idx}:{idx+5}]: {policy_obs[idx:idx+5].tolist()}")
    print(f"    - Root Z: {policy_obs[idx].item():.6f}")
    print(f"    - Root quat: [{policy_obs[idx+1].item():.4f}, {policy_obs[idx+2].item():.4f}, {policy_obs[idx+3].item():.4f}, {policy_obs[idx+4].item():.4f}]")
    idx += 5
    
    # Joint positions (27D)
    print(f"  joint_pos [{idx}:{idx+27}]: First 5 = {policy_obs[idx:idx+5].tolist()}")
    idx += 27
    
    # dq_root (6D): [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
    print(f"  dq_root [{idx}:{idx+6}]: {policy_obs[idx:idx+6].tolist()}")
    print(f"    - Linear vel: {policy_obs[idx:idx+3].tolist()}")
    print(f"    - Angular vel: {policy_obs[idx+3:idx+6].tolist()}")
    idx += 6
    
    # Joint velocities (27D)
    print(f"  joint_vel [{idx}:{idx+27}]: First 5 = {policy_obs[idx:idx+5].tolist()}")
    
    # Get expert observation at timestep 0
    print("\n" + "=" * 60)
    print("EXPERT TRAJECTORY OBSERVATIONS")
    print("=" * 60)
    
    timestep = 0
    converted_state = data_bridge.get_trajectory_state(timestep)
    
    if converted_state is None:
        print("❌ Failed to get expert trajectory state!")
        return
    
    # Apply expert state to environment
    env_ids = torch.tensor([0], device=env.device)
    data_bridge.apply_trajectory_state(converted_state, env_ids)
    
    # Get expert observation
    expert_obs = env._get_observations()[0]  # First environment
    
    print(f"Expert observation shape: {expert_obs.shape}")
    print(f"Expert observation full values:")
    for i, val in enumerate(expert_obs):
        print(f"  [{i:2d}]: {val.item():.6f}")
    
    # Parse expert observation components
    print(f"\nExpert observation components:")
    idx = 0
    
    # q_root_no_xy (5D)
    print(f"  q_root_no_xy [{idx}:{idx+5}]: {expert_obs[idx:idx+5].tolist()}")
    print(f"    - Root Z: {expert_obs[idx].item():.6f}")
    print(f"    - Root quat: [{expert_obs[idx+1].item():.4f}, {expert_obs[idx+2].item():.4f}, {expert_obs[idx+3].item():.4f}, {expert_obs[idx+4].item():.4f}]")
    idx += 5
    
    # Joint positions (27D) with joint names
    print(f"  joint_pos [{idx}:{idx+27}]:")
    joint_pos_expert = expert_obs[idx:idx+27]
    
    # Print joint positions with their names
    for i, joint_val in enumerate(joint_pos_expert):
        if i < len(env.motors_dof_idx):
            motor_dof_idx = env.motors_dof_idx[i]
            # Find joint name for this DOF index
            joint_name = "unknown"
            for name, dof_idx in env.joint_to_motor_idx.items():
                if dof_idx == motor_dof_idx:
                    joint_name = name
                    break
            print(f"    [{idx+i:2d}] {joint_name}: {joint_val.item():.6f}")
        else:
            print(f"    [{idx+i:2d}] joint_{i}: {joint_val.item():.6f}")
    
    idx += 27
    
    # dq_root (6D)
    print(f"  dq_root [{idx}:{idx+6}]: {expert_obs[idx:idx+6].tolist()}")
    print(f"    - Linear vel: {expert_obs[idx:idx+3].tolist()}")
    print(f"    - Angular vel: {expert_obs[idx+3:idx+6].tolist()}")
    idx += 6
    
    # Joint velocities (27D) with joint names
    print(f"  joint_vel [{idx}:{idx+27}]:")
    joint_vel_expert = expert_obs[idx:idx+27]
    
    # Print first 5 joint velocities with names
    for i in range(min(5, len(joint_vel_expert))):
        if i < len(env.motors_dof_idx):
            motor_dof_idx = env.motors_dof_idx[i]
            joint_name = "unknown"
            for name, dof_idx in env.joint_to_motor_idx.items():
                if dof_idx == motor_dof_idx:
                    joint_name = name
                    break
            print(f"    [{idx+i:2d}] {joint_name}_vel: {joint_vel_expert[i].item():.6f}")
        else:
            print(f"    [{idx+i:2d}] joint_{i}_vel: {joint_vel_expert[i].item():.6f}")
    
    print("    ... (remaining velocities)")
    
    # Compare policy vs expert
    print("\n" + "=" * 60)
    print("POLICY vs EXPERT COMPARISON")
    print("=" * 60)
    
    print(f"Shape comparison:")
    print(f"  - Policy shape: {policy_obs.shape}")
    print(f"  - Expert shape: {expert_obs.shape}")
    print(f"  - Shapes match: {policy_obs.shape == expert_obs.shape}")
    
    # Check value differences
    abs_diff = torch.abs(policy_obs - expert_obs)
    print(f"\nValue comparison:")
    print(f"  - Mean absolute difference: {abs_diff.mean().item():.6f}")
    print(f"  - Max absolute difference: {abs_diff.max().item():.6f}")
    print(f"  - Policy value range: [{policy_obs.min().item():.4f}, {policy_obs.max().item():.4f}]")
    print(f"  - Expert value range: [{expert_obs.min().item():.4f}, {expert_obs.max().item():.4f}]")
    
    # Show largest differences
    print(f"\nLargest differences (index: policy_val -> expert_val):")
    sorted_indices = torch.argsort(abs_diff, descending=True)[:10]
    for i in sorted_indices:
        idx_val = i.item()
        print(f"  [{idx_val:2d}]: {policy_obs[idx_val].item():.6f} -> {expert_obs[idx_val].item():.6f} (diff: {abs_diff[idx_val].item():.6f})")
    
    # Joint mapping verification
    print("\n" + "=" * 60)
    print("JOINT MAPPING VERIFICATION")  
    print("=" * 60)
    
    print(f"Genesis joint configuration:")
    print(f"  - Total joints: {len(env.joint_names)}")
    print(f"  - Controllable motors: {len(env.motors_dof_idx)}")
    print(f"  - Box feet enabled: {env.use_box_feet}")
    
    print(f"\nMotor DOF index to joint name mapping:")
    for i, motor_dof_idx in enumerate(env.motors_dof_idx):
        joint_name = "unknown"
        for name, dof_idx in env.joint_to_motor_idx.items():
            if dof_idx == motor_dof_idx:
                joint_name = name
                break
        print(f"  Motor[{i:2d}] -> DOF[{motor_dof_idx:2d}] -> {joint_name}")
    
    print("\n✅ Observation alignment verification complete!")
    print("Check the output above to ensure policy and expert observations have:")
    print("  1. Same component structure and ordering")
    print("  2. Reasonable value differences (expert should have non-zero motion)")
    print("  3. Proper joint name mappings")


if __name__ == "__main__":
    main()