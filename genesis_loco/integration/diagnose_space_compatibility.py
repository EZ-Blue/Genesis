#!/usr/bin/env python3
"""
Observation and Action Space Compatibility Diagnostic

Compare expert dataset vs Genesis skeleton model to identify space mismatches.
"""

import torch
import numpy as np
import sys
import os
from typing import Dict, Any

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
import genesis as gs


def safe_init_genesis():
    """Safely initialize Genesis"""
    try:
        gs.init(backend=gs.gpu)
        return True, "Genesis initialized"
    except Exception as e:
        if "already initialized" in str(e):
            return True, "Genesis already initialized"
        else:
            return False, f"Genesis initialization failed: {e}"


def analyze_expert_dataset(data_bridge: LocoMujocoDataBridge, num_samples: int = 5):
    """Analyze expert dataset structure and values"""
    print("🔍 EXPERT DATASET ANALYSIS")
    print("=" * 50)
    
    if data_bridge.loco_trajectory is None:
        print("❌ No trajectory loaded")
        return None
    
    traj = data_bridge.loco_trajectory
    traj_length = traj.data.qpos.shape[0]
    
    print(f"Trajectory Info:")
    print(f"   Length: {traj_length} timesteps")
    print(f"   Frequency: {traj.info.frequency} Hz")
    print(f"   Joint names: {len(traj.info.joint_names)} joints")
    print(f"   Joint names: {traj.info.joint_names}")
    
    # Analyze data shapes
    print(f"\nData Shapes:")
    print(f"   qpos shape: {traj.data.qpos.shape}")
    print(f"   qvel shape: {traj.data.qvel.shape}")
    
    # Sample random timesteps
    np.random.seed(42)  # Reproducible results
    sample_indices = np.random.choice(traj_length, size=min(num_samples, traj_length), replace=False)
    sample_indices = sorted(sample_indices)
    
    print(f"\nSampling {len(sample_indices)} random timesteps: {sample_indices}")
    
    expert_data = {}
    for i, timestep in enumerate(sample_indices):
        qpos = traj.data.qpos[timestep]
        qvel = traj.data.qvel[timestep]
        
        print(f"\n--- Timestep {timestep} ---")
        print(f"qpos ({len(qpos)}): min={qpos.min():.4f}, max={qpos.max():.4f}, mean={qpos.mean():.4f}")
        print(f"qvel ({len(qvel)}): min={qvel.min():.4f}, max={qvel.max():.4f}, mean={qvel.mean():.4f}")
        
        # Root state (first 7 elements typically)
        if len(qpos) >= 7:
            root_pos = qpos[:3]
            root_quat = qpos[3:7]
            print(f"Root pos: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")
            print(f"Root quat: [{root_quat[0]:.3f}, {root_quat[1]:.3f}, {root_quat[2]:.3f}, {root_quat[3]:.3f}]")
        
        # Joint positions (after root state)
        if len(qpos) > 7:
            joint_positions = qpos[7:]
            print(f"Joint positions ({len(joint_positions)}): min={joint_positions.min():.4f}, max={joint_positions.max():.4f}")
            print(f"First 10 joints: {joint_positions[:10]}")
        
        expert_data[timestep] = {
            'qpos': qpos,
            'qvel': qvel,
            'joint_names': traj.info.joint_names
        }
    
    return expert_data


def analyze_genesis_skeleton(env: SkeletonHumanoidEnv, num_samples: int = 5):
    """Analyze Genesis skeleton model structure and values"""
    print("\n🤖 GENESIS SKELETON ANALYSIS")
    print("=" * 50)
    
    print(f"Environment Info:")
    print(f"   Num envs: {env.num_envs}")
    print(f"   Num observations: {env.num_observations}")
    print(f"   Num actions: {env.num_actions}")
    print(f"   Total robot DOFs: {env.robot.n_dofs}")
    print(f"   Motor DOF indices: {env.motors_dof_idx}")
    print(f"   Joint names: {len(env.joint_names)} joints")
    print(f"   Joint names: {sorted(list(env.joint_names))}")
    
    # Reset and collect samples
    obs, _ = env.reset()
    
    genesis_data = {}
    print(f"\nSampling {num_samples} environment steps:")
    
    for i in range(num_samples):
        # Random actions
        actions = torch.randn((env.num_envs, env.num_actions), device=env.device) * 0.1
        
        # Step environment
        obs, rewards, dones, info = env.step(actions)
        
        # Extract state information
        root_pos = env.root_pos[0].cpu().numpy()
        root_quat = env.root_quat[0].cpu().numpy() 
        root_lin_vel = env.root_lin_vel[0].cpu().numpy()
        root_ang_vel = env.root_ang_vel[0].cpu().numpy()
        dof_pos = env.dof_pos[0].cpu().numpy()
        dof_vel = env.dof_vel[0].cpu().numpy()
        
        print(f"\n--- Step {i+1} ---")
        print(f"Observation shape: {obs.shape}")
        print(f"Observation: min={obs.min().item():.4f}, max={obs.max().item():.4f}, mean={obs.mean().item():.4f}")
        print(f"Actions shape: {actions.shape}")
        print(f"Actions: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")
        print(f"Rewards: {rewards[0].item():.4f}")
        
        print(f"Root pos: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")
        print(f"Root quat: [{root_quat[0]:.3f}, {root_quat[1]:.3f}, {root_quat[2]:.3f}, {root_quat[3]:.3f}]")
        print(f"DOF positions ({len(dof_pos)}): min={dof_pos.min():.4f}, max={dof_pos.max():.4f}")
        print(f"Controlled DOFs ({len(env.motors_dof_idx)}): {dof_pos[env.motors_dof_idx][:10]}")  # First 10
        
        genesis_data[i] = {
            'obs': obs[0].cpu().numpy(),
            'actions': actions[0].cpu().numpy(),
            'root_pos': root_pos,
            'root_quat': root_quat,
            'dof_pos': dof_pos,
            'dof_vel': dof_vel,
            'rewards': rewards[0].item()
        }
    
    return genesis_data


def compare_spaces(expert_data: Dict, genesis_data: Dict, data_bridge: LocoMujocoDataBridge):
    """Compare expert and genesis observation/action spaces"""
    print("\n🔄 SPACE COMPATIBILITY COMPARISON")
    print("=" * 50)
    
    if not expert_data or not genesis_data:
        print("❌ Missing data for comparison")
        return
    
    # Get sample data
    expert_sample = next(iter(expert_data.values()))
    genesis_sample = next(iter(genesis_data.values()))
    
    print("📊 DIMENSIONAL COMPARISON:")
    print(f"   Expert qpos dimension: {len(expert_sample['qpos'])}")
    print(f"   Expert qvel dimension: {len(expert_sample['qvel'])}")
    print(f"   Genesis observation dimension: {len(genesis_sample['obs'])}")
    print(f"   Genesis action dimension: {len(genesis_sample['actions'])}")
    print(f"   Genesis total DOFs: {len(genesis_sample['dof_pos'])}")
    print(f"   Genesis controlled DOFs: {len(data_bridge.genesis_env.motors_dof_idx)}")
    
    print("\n🎯 JOINT MAPPING ANALYSIS:")
    expert_joints = set(expert_sample['joint_names'])
    genesis_joints = set(data_bridge.joint_names)
    
    matched_joints = expert_joints.intersection(genesis_joints)
    expert_only = expert_joints - genesis_joints
    genesis_only = genesis_joints - expert_joints
    
    print(f"   Expert joints: {len(expert_joints)}")
    print(f"   Genesis joints: {len(genesis_joints)}")
    print(f"   Matched joints: {len(matched_joints)} ({len(matched_joints)/len(expert_joints)*100:.1f}%)")
    
    if expert_only:
        print(f"   Expert-only joints: {sorted(list(expert_only))}")
    if genesis_only:
        print(f"   Genesis-only joints: {sorted(list(genesis_only))}")
    
    print("\n📈 VALUE RANGE COMPARISON:")
    
    # Compare position ranges
    expert_qpos_range = (expert_sample['qpos'].min(), expert_sample['qpos'].max())
    genesis_dof_range = (genesis_sample['dof_pos'].min(), genesis_sample['dof_pos'].max())
    
    print(f"   Expert qpos range: [{expert_qpos_range[0]:.4f}, {expert_qpos_range[1]:.4f}]")
    print(f"   Genesis DOF range: [{genesis_dof_range[0]:.4f}, {genesis_dof_range[1]:.4f}]")
    
    # Compare velocity ranges
    expert_qvel_range = (expert_sample['qvel'].min(), expert_sample['qvel'].max())
    genesis_dof_vel_range = (genesis_sample['dof_vel'].min(), genesis_sample['dof_vel'].max())
    
    print(f"   Expert qvel range: [{expert_qvel_range[0]:.4f}, {expert_qvel_range[1]:.4f}]")
    print(f"   Genesis DOF vel range: [{genesis_dof_vel_range[0]:.4f}, {genesis_dof_vel_range[1]:.4f}]")
    
    print("\n🔍 DATA BRIDGE CONVERSION TEST:")
    
    # Test data bridge conversion
    try:
        expert_timestep = next(iter(expert_data.keys()))
        converted_state = data_bridge.get_trajectory_state(expert_timestep)
        
        if converted_state:
            print("   ✅ Data bridge conversion successful")
            print(f"   Converted DOF pos range: [{converted_state['dof_pos'].min().item():.4f}, {converted_state['dof_pos'].max().item():.4f}]")
            print(f"   Converted root pos: [{converted_state['root_pos'][0]:.3f}, {converted_state['root_pos'][1]:.3f}, {converted_state['root_pos'][2]:.3f}]")
            
            # Apply converted state to environment
            env_ids = torch.tensor([0], device=data_bridge.device)
            data_bridge.apply_trajectory_state(converted_state, env_ids)
            obs_after_conversion = data_bridge.genesis_env._get_observations()
            
            print(f"   Observation after conversion: min={obs_after_conversion.min().item():.4f}, max={obs_after_conversion.max().item():.4f}")
            
        else:
            print("   ❌ Data bridge conversion failed")
            
    except Exception as e:
        print(f"   ❌ Data bridge test error: {e}")
    
    # Identify potential issues
    print("\n⚠️  POTENTIAL ISSUES IDENTIFIED:")
    issues = []
    
    if len(matched_joints) < len(expert_joints) * 0.8:
        issues.append(f"Low joint matching rate: {len(matched_joints)}/{len(expert_joints)}")
    
    if abs(expert_qpos_range[1] - expert_qpos_range[0]) > abs(genesis_dof_range[1] - genesis_dof_range[0]) * 5:
        issues.append("Expert and Genesis position ranges very different")
    
    if len(genesis_sample['obs']) != data_bridge.genesis_env.num_observations:
        issues.append("Observation dimension mismatch")
    
    if not issues:
        print("   ✅ No major compatibility issues detected")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")


def main():
    """Main diagnostic function"""
    print("🔬 OBSERVATION & ACTION SPACE COMPATIBILITY DIAGNOSTIC")
    print("=" * 70)
    
    try:
        # Initialize Genesis
        success, message = safe_init_genesis()
        if not success:
            raise RuntimeError(message)
        print(f"✅ {message}")
        
        # Create environment
        print("\n1. Creating Genesis skeleton environment...")
        env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=5.0,
            dt=0.01,
            show_viewer=False,
            use_box_feet=True
        )
        print("   ✅ Environment created")
        
        # Create data bridge and load trajectory
        print("\n2. Loading expert trajectory...")
        data_bridge = LocoMujocoDataBridge(env)
        success = data_bridge.load_trajectory("walk")
        
        if not success:
            print("   ❌ Failed to load trajectory")
            return
        print("   ✅ Expert trajectory loaded")
        
        # Analyze expert dataset
        expert_data = analyze_expert_dataset(data_bridge, num_samples=3)
        
        # Analyze Genesis skeleton
        genesis_data = analyze_genesis_skeleton(env, num_samples=3)
        
        # Compare spaces
        compare_spaces(expert_data, genesis_data, data_bridge)
        
        print("\n" + "=" * 70)
        print("✅ DIAGNOSTIC COMPLETE")
        print("Review the analysis above to identify compatibility issues.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()