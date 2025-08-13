#!/usr/bin/env python3
"""
Debug Refactored Environment - Test All Critical Issues

Simple diagnostics to identify why observations are zero and episode length isn't incrementing.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import genesis as gs

def main():
    print("🔍 Debug Refactored Environment")
    print("=" * 50)
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Import and create environment
    from environments.skeleton_humanoid_refactored import SkeletonHumanoidEnv
    
    env = SkeletonHumanoidEnv(
        num_envs=2,  # Small for debugging
        episode_length_s=5.0,
        dt=0.02,
        use_box_feet=True,
        show_viewer=True
    )
    
    print(f"Environment created: {env.num_envs} envs, {env.num_actions} actions")
    
    # Test 1: Check initial robot state after scene build
    print("\n🧪 TEST 1: Initial Robot State After Scene Build")
    print("-" * 40)
    
    # Get raw robot state directly from Genesis
    raw_pos = env.robot.get_pos()
    raw_quat = env.robot.get_quat() 
    raw_dof_pos = env.robot.get_dofs_position()
    raw_dof_vel = env.robot.get_dofs_velocity()
    
    print(f"Raw root pos: {raw_pos[0].cpu().numpy()}")
    print(f"Raw root quat: {raw_quat[0].cpu().numpy()}")
    print(f"Raw DOF pos shape: {raw_dof_pos.shape}, sample: {raw_dof_pos[0][:5].cpu().numpy()}")
    print(f"Raw DOF vel shape: {raw_dof_vel.shape}, sample: {raw_dof_vel[0][:5].cpu().numpy()}")
    
    # Check if quaternion is invalid
    quat_norm = torch.norm(raw_quat, dim=1)
    print(f"Quaternion norms: {quat_norm.cpu().numpy()}")
    if torch.any(quat_norm < 0.1):
        print("❌ CRITICAL: Invalid quaternion detected (norm too small)")
    else:
        print("✅ Quaternions appear valid")
    
    # Test 2: Check DOF mapping
    print("\n🧪 TEST 2: DOF Index Mapping Verification")
    print("-" * 40)
    
    print(f"Total robot DOFs: {env.robot.n_dofs}")
    print(f"Action DOFs: {len(env.motors_dof_idx)}")
    print(f"DOF indices: {env.motors_dof_idx[:10]}...")  # First 10
    
    # Check if DOF indices are valid
    max_dof_idx = max(env.motors_dof_idx) if env.motors_dof_idx else -1
    if max_dof_idx >= env.robot.n_dofs:
        print(f"❌ CRITICAL: DOF index {max_dof_idx} >= total DOFs {env.robot.n_dofs}")
    else:
        print("✅ DOF indices appear valid")
    
    # Test specific joint mapping
    test_joints = ["lumbar_extension", "hip_flexion_r", "knee_angle_r"]
    for joint_name in test_joints:
        if joint_name in env.joint_to_motor_idx:
            dof_idx = env.joint_to_motor_idx[joint_name]
            try:
                joint_obj = env.robot.get_joint(joint_name)
                actual_dof_start = joint_obj.dof_start
                print(f"Joint {joint_name}: mapped={dof_idx}, actual={actual_dof_start}")
                if dof_idx != actual_dof_start:
                    print(f"  ❌ MISMATCH detected!")
            except:
                print(f"Joint {joint_name}: Genesis lookup failed")
    
    # Test 3: Reset and check state updates
    print("\n🧪 TEST 3: Reset and State Update")
    print("-" * 40)
    
    obs, info = env.reset()
    print(f"After reset - obs shape: {obs.shape}")
    print(f"Environment buffers:")
    print(f"  root_pos: {env.root_pos[0].cpu().numpy()}")
    print(f"  root_quat: {env.root_quat[0].cpu().numpy()}")
    print(f"  episode_length: {env.episode_length_buf[0].item()}")
    
    # Compare with raw Genesis data
    raw_pos_after = env.robot.get_pos()
    raw_quat_after = env.robot.get_quat()
    print(f"Raw Genesis after reset:")
    print(f"  pos: {raw_pos_after[0].cpu().numpy()}")
    print(f"  quat: {raw_quat_after[0].cpu().numpy()}")
    
    # Check if buffers match Genesis
    pos_match = torch.allclose(env.root_pos[0], raw_pos_after[0], atol=1e-4)
    quat_match = torch.allclose(env.root_quat[0], raw_quat_after[0], atol=1e-4)
    print(f"Buffer-Genesis match: pos={pos_match}, quat={quat_match}")
    
    # Test 4: Step and check updates
    print("\n🧪 TEST 4: Step and Update Verification")
    print("-" * 40)
    
    # Store before step
    episode_len_before = env.episode_length_buf[0].item()
    pos_before = env.root_pos[0].clone()
    
    # Step with zero actions
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, rewards, dones, info = env.step(actions)
    
    # Check updates
    episode_len_after = env.episode_length_buf[0].item()
    pos_after = env.root_pos[0]
    
    print(f"Episode length: {episode_len_before} -> {episode_len_after}")
    print(f"Position change: {torch.norm(pos_after - pos_before).item():.6f}")
    print(f"Rewards: {rewards[0].item():.6f}")
    print(f"Done: {dones[0].item()}")
    print(f"Reset buffer: {env.reset_buf[0].item()}")
    
    if episode_len_after == episode_len_before:
        print("❌ CRITICAL: Episode length not incrementing!")
    else:
        print("✅ Episode length incrementing correctly")
    
    # Test 5: Termination conditions
    print("\n🧪 TEST 5: Termination Condition Check")
    print("-" * 40)
    
    # Manually check termination logic
    height = env.root_pos[0, 2].item()
    height_ok = 0.8 <= height <= 1.1
    print(f"Height: {height:.3f}m, valid: {height_ok}")
    
    # Check orientation
    from genesis.utils.geom import quat_to_xyz
    root_euler = quat_to_xyz(env.root_quat[0:1])
    roll, pitch = root_euler[0, 0].item(), root_euler[0, 1].item()
    roll_ok = abs(roll) <= torch.pi/4
    pitch_ok = abs(pitch) <= torch.pi/4
    print(f"Roll: {roll:.3f}rad ({roll*180/torch.pi:.1f}°), valid: {roll_ok}")
    print(f"Pitch: {pitch:.3f}rad ({pitch*180/torch.pi:.1f}°), valid: {pitch_ok}")
    
    # Check episode length termination
    episode_len_ok = env.episode_length_buf[0].item() < env.max_episode_length
    print(f"Episode length: {env.episode_length_buf[0].item()}/{env.max_episode_length}, valid: {episode_len_ok}")
    
    should_terminate = not (height_ok and roll_ok and pitch_ok and episode_len_ok)
    actually_terminated = env.reset_buf[0].item()
    print(f"Should terminate: {should_terminate}, Actually terminated: {actually_terminated}")
    
    # Test 6: Observation construction
    print("\n🧪 TEST 6: Observation Construction")
    print("-" * 40)
    
    # Manually construct observation to see where zeros come from
    print(f"Root pos z: {env.root_pos[0, 2].item():.6f}")
    print(f"Root quat: {env.root_quat[0].cpu().numpy()}")
    print(f"DOF pos sample: {env.dof_pos[0, :5].cpu().numpy()}")
    print(f"DOF vel sample: {env.dof_vel[0, :5].cpu().numpy()}")
    print(f"Root lin vel: {env.root_lin_vel[0].cpu().numpy()}")
    print(f"Root ang vel: {env.root_ang_vel[0].cpu().numpy()}")
    
    # Check if observation matches expectation
    expected_nonzero = env.root_pos[0, 2].item()  # Should be ~0.975
    actual_obs_first = obs[0, 0].item()
    print(f"Expected first obs (height): {expected_nonzero:.6f}")
    print(f"Actual first obs: {actual_obs_first:.6f}")
    
    if abs(actual_obs_first) < 1e-6:
        print("❌ CRITICAL: Height not making it to observations!")
    else:
        print("✅ Height correctly in observations")
    
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS COMPLETE")
    print("Check above for ❌ CRITICAL issues")

if __name__ == "__main__":
    main()