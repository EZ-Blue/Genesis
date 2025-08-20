#!/usr/bin/env python3
"""
GAIL Training Diagnosis - Deep analysis of why discriminator keeps getting overconfident
"""

import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')
sys.path.append('/home/ez/Documents/loco-mujoco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge


def analyze_observation_distributions():
    """Deep statistical analysis of expert vs policy observation distributions"""
    print("🔍 DEEP OBSERVATION DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create environment
    env = SkeletonHumanoidEnv(
        num_envs=32,  # Use multiple envs for policy diversity
        episode_length_s=5.0,
        dt=0.02,
        use_box_feet=True,
        show_viewer=False
    )
    
    # Load expert data
    data_bridge = LocoMujocoDataBridge(env)
    success = data_bridge.load_trajectory("walk")
    
    if not success:
        print("❌ Failed to load trajectory!")
        return
    
    print("Loading expert observations...")
    
    # Get expert observations (sample 5000 for efficiency)
    expert_obs_list = []
    trajectory_length = data_bridge.trajectory_length
    sample_indices = np.linspace(0, trajectory_length-1, 5000, dtype=int)
    
    env_ids = torch.tensor([0], device=env.device)
    
    for i in sample_indices[:1000]:  # Limit for speed
        state_data = data_bridge.get_trajectory_state(i)
        if state_data is None:
            continue
            
        data_bridge.apply_trajectory_state(state_data, env_ids)
        obs = env._get_observations()
        expert_obs_list.append(obs[0].cpu().numpy())
    
    expert_obs = np.array(expert_obs_list)
    print(f"Expert observations: {expert_obs.shape}")
    
    # Get policy observations (random policy)
    print("Generating policy observations...")
    policy_obs_list = []
    
    obs, _ = env.reset()
    
    for _ in range(1000):
        # Random actions (simulates early training policy)
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.1
        obs, _, _, _ = env.step(actions)
        
        # Sample some observations
        for env_idx in range(min(4, env.num_envs)):
            policy_obs_list.append(obs[env_idx].cpu().numpy())
    
    policy_obs = np.array(policy_obs_list[:1000])  # Match expert sample size
    print(f"Policy observations: {policy_obs.shape}")
    
    # Statistical analysis
    print(f"\n📊 STATISTICAL COMPARISON:")
    
    # Overall statistics
    print(f"Expert mean: {expert_obs.mean():.6f}")
    print(f"Policy mean: {policy_obs.mean():.6f}")
    print(f"Expert std: {expert_obs.std():.6f}")
    print(f"Policy std: {policy_obs.std():.6f}")
    
    # Per-component analysis
    components = {
        'q_root_no_xy': (0, 5),
        'joint_pos': (5, 32), 
        'dq_root': (32, 38),
        'joint_vel': (38, 65)
    }
    
    print(f"\n📋 COMPONENT-WISE ANALYSIS:")
    
    concerning_components = []
    
    for comp_name, (start, end) in components.items():
        expert_comp = expert_obs[:, start:end]
        policy_comp = policy_obs[:, start:end]
        
        # Statistical measures
        expert_mean = expert_comp.mean(axis=0)
        policy_mean = policy_comp.mean(axis=0)
        expert_std = expert_comp.std(axis=0)
        policy_std = policy_comp.std(axis=0)
        
        # Compute differences
        mean_diff = np.abs(expert_mean - policy_mean).mean()
        std_ratio = (policy_std / (expert_std + 1e-8)).mean()
        
        # Range differences
        expert_range = expert_comp.max(axis=0) - expert_comp.min(axis=0)
        policy_range = policy_comp.max(axis=0) - policy_comp.min(axis=0)
        range_ratio = (policy_range / (expert_range + 1e-8)).mean()
        
        print(f"  {comp_name}:")
        print(f"    Mean diff: {mean_diff:.6f}")
        print(f"    Std ratio (policy/expert): {std_ratio:.3f}")
        print(f"    Range ratio (policy/expert): {range_ratio:.3f}")
        
        # Flag concerning differences
        if mean_diff > 0.1 or std_ratio > 2.0 or std_ratio < 0.5 or range_ratio > 2.0 or range_ratio < 0.5:
            concerning_components.append(comp_name)
            print(f"    ⚠️  CONCERNING DIFFERENCE DETECTED!")
    
    # Distribution overlap analysis
    print(f"\n🔄 DISTRIBUTION OVERLAP ANALYSIS:")
    
    for comp_name, (start, end) in components.items():
        expert_comp = expert_obs[:, start:end].flatten()
        policy_comp = policy_obs[:, start:end].flatten()
        
        # Calculate overlap using histogram intersection
        bins = np.linspace(
            min(expert_comp.min(), policy_comp.min()),
            max(expert_comp.max(), policy_comp.max()),
            50
        )
        
        expert_hist, _ = np.histogram(expert_comp, bins=bins, density=True)
        policy_hist, _ = np.histogram(policy_comp, bins=bins, density=True)
        
        # Histogram intersection (overlap measure)
        overlap = np.sum(np.minimum(expert_hist, policy_hist)) / np.sum(np.maximum(expert_hist, policy_hist))
        
        print(f"  {comp_name} overlap: {overlap:.3f}")
        
        if overlap < 0.3:
            print(f"    ⚠️  LOW OVERLAP - DISCRIMINATOR CAN EASILY DISTINGUISH!")
    
    # Value range analysis
    print(f"\n📏 VALUE RANGE ANALYSIS:")
    
    expert_min, expert_max = expert_obs.min(), expert_obs.max()
    policy_min, policy_max = policy_obs.min(), policy_obs.max()
    
    print(f"Expert range: [{expert_min:.3f}, {expert_max:.3f}]")
    print(f"Policy range: [{policy_min:.3f}, {policy_max:.3f}]")
    
    # Check for out-of-distribution values
    policy_below_expert = (policy_obs < expert_min).sum()
    policy_above_expert = (policy_obs > expert_max).sum()
    total_policy_vals = policy_obs.size
    
    oob_percentage = ((policy_below_expert + policy_above_expert) / total_policy_vals) * 100
    
    print(f"Policy values outside expert range: {oob_percentage:.1f}%")
    
    if oob_percentage > 10:
        print(f"    ⚠️  SIGNIFICANT OUT-OF-DISTRIBUTION VALUES!")
    
    # Joint-specific analysis (first 5 joints)
    print(f"\n🦴 JOINT-SPECIFIC ANALYSIS (first 5 joints):")
    
    joint_pos_expert = expert_obs[:, 5:10]  # First 5 joint positions
    joint_pos_policy = policy_obs[:, 5:10]
    
    for joint_idx in range(5):
        expert_joint = joint_pos_expert[:, joint_idx]
        policy_joint = joint_pos_policy[:, joint_idx]
        
        # Statistical difference
        ks_stat = np.abs(np.sort(expert_joint) - np.sort(policy_joint)).max()
        
        print(f"  Joint {joint_idx}:")
        print(f"    Expert: mean={expert_joint.mean():.3f}, std={expert_joint.std():.3f}")
        print(f"    Policy: mean={policy_joint.mean():.3f}, std={policy_joint.std():.3f}")
        print(f"    KS statistic: {ks_stat:.3f}")
        
        if ks_stat > 0.3:
            print(f"    ⚠️  LARGE DISTRIBUTION DIFFERENCE!")
    
    print(f"\n🎯 SUMMARY:")
    if concerning_components:
        print(f"❌ Components with concerning differences: {concerning_components}")
        print(f"   These are likely causing discriminator overconfidence.")
    else:
        print(f"✅ No major statistical differences found.")
        print(f"   Issue may be in discriminator architecture or training.")
    
    return {
        'expert_obs': expert_obs,
        'policy_obs': policy_obs,
        'concerning_components': concerning_components
    }


def analyze_discriminator_gradients():
    """Analyze what the discriminator is learning to distinguish"""
    print(f"\n🧠 DISCRIMINATOR LEARNING ANALYSIS")
    print("=" * 50)
    
    # This would require loading a trained discriminator and computing gradients
    # For now, provide guidance on what to look for
    
    print("To analyze discriminator learning:")
    print("1. Load a trained GAIL discriminator")
    print("2. Compute gradients of discriminator w.r.t. observations")
    print("3. Identify which observation components have highest gradients")
    print("4. Those components are what the discriminator uses to distinguish")
    
    print("\nCommon culprits:")
    print("- Root position/velocity ranges")
    print("- Joint position distributions")
    print("- Observation value scaling differences")
    print("- Temporal correlation differences")


def suggest_fixes(analysis_results):
    """Suggest specific fixes based on analysis"""
    print(f"\n🛠️ SUGGESTED FIXES:")
    
    concerning = analysis_results.get('concerning_components', [])
    
    if 'q_root_no_xy' in concerning:
        print("1. ROOT STATE ISSUE:")
        print("   - Add root position/velocity normalization")
        print("   - Ensure root state initialization matches expert")
        print("   - Check quaternion normalization")
    
    if 'joint_pos' in concerning:
        print("2. JOINT POSITION ISSUE:")
        print("   - Add joint position normalization")
        print("   - Check joint limits match LocoMujoco")
        print("   - Verify PD controller gains")
    
    if 'dq_root' in concerning:
        print("3. ROOT VELOCITY ISSUE:")
        print("   - Normalize root velocities")
        print("   - Check physics timestep differences")
        print("   - Verify contact dynamics")
    
    if 'joint_vel' in concerning:
        print("4. JOINT VELOCITY ISSUE:")
        print("   - Add joint velocity scaling")
        print("   - Check damping parameters")
        print("   - Verify action smoothing")
    
    print("\n5. GENERAL SOLUTIONS:")
    print("   - Add observation normalization layer")
    print("   - Use smaller discriminator (reduce capacity)")
    print("   - Add noise to observations during training")
    print("   - Implement domain randomization")
    print("   - Use spectral normalization in discriminator")


def main():
    print("🚨 GAIL TRAINING DIAGNOSIS")
    print("=" * 70)
    print("Analyzing why discriminator keeps getting overconfident...")
    
    analysis_results = analyze_observation_distributions()
    
    if analysis_results:
        analyze_discriminator_gradients()
        suggest_fixes(analysis_results)
    
    print(f"\n✅ Diagnosis complete!")
    print(f"Focus on fixing the concerning components identified above.")


if __name__ == "__main__":
    main()