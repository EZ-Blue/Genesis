"""
Debug Action Range and Model Output Scale Issues

This script investigates the mismatch between training data ranges 
and model outputs that might be causing the skeleton to collapse.
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from integration.data_bridge import LocoMujocoDataBridge
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.single_trajectory_behavior_cloning import SingleTrajectoryMLP
import genesis as gs

def debug_training_data_ranges():
    """Check the ranges of target joint positions in training data"""
    print("🔍 DEBUGGING TRAINING DATA RANGES")
    print("=" * 50)
    
    # Create environment and data bridge
    gs.init(backend=gs.gpu)
    env = SkeletonHumanoidEnv(num_envs=1, episode_length_s=8.0, dt=0.01, use_box_feet=True)
    data_bridge = LocoMujocoDataBridge(env)
    data_bridge.load_trajectory('walk')
    
    print(f"✅ Loaded trajectory: {data_bridge.trajectory_length} timesteps")
    
    # Sample multiple trajectory states to get range statistics
    dof_positions = []
    sample_indices = np.linspace(0, data_bridge.trajectory_length-1, 100, dtype=int)
    
    for i in sample_indices:
        state = data_bridge.get_trajectory_state(i)
        if state and 'dof_pos' in state:
            dof_positions.append(state['dof_pos'].cpu().numpy())
    
    if len(dof_positions) > 0:
        dof_positions = np.array(dof_positions)  # Shape: (100, num_joints)
        
        print(f"📊 Training Data Joint Position Analysis:")
        print(f"   Shape: {dof_positions.shape}")
        print(f"   Global Range: [{dof_positions.min():.6f}, {dof_positions.max():.6f}]")
        print(f"   Global Mean: {dof_positions.mean():.6f}")
        print(f"   Global Std: {dof_positions.std():.6f}")
        print(f"   Mean Magnitude: {np.abs(dof_positions).mean():.6f}")
        
        # Per-joint analysis
        print(f"\n📋 Per-Joint Statistics (first 10 joints):")
        for j in range(min(10, dof_positions.shape[1])):
            joint_values = dof_positions[:, j]
            print(f"   Joint {j}: range=[{joint_values.min():.4f}, {joint_values.max():.4f}], "
                  f"mean={joint_values.mean():.4f}, std={joint_values.std():.4f}")
                  
        return dof_positions
    else:
        print("❌ Failed to collect training data")
        return None

def debug_model_outputs(model_path, sample_obs=None):
    """Check the ranges of trained model outputs"""
    print(f"\n🧠 DEBUGGING MODEL OUTPUTS")
    print("=" * 50)
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim'] 
        hidden_dims = checkpoint.get('hidden_dims', [512, 256])
        
        model = SingleTrajectoryMLP(obs_dim, action_dim, hidden_dims, dropout_rate=0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded: {obs_dim}→{action_dim}, hidden={hidden_dims}")
        
        # Generate test observations (if not provided)
        if sample_obs is None:
            # Use random observations in typical range
            sample_obs = torch.randn(10, obs_dim) * 0.5  # Typical obs range
            
        with torch.no_grad():
            predicted_actions = model(sample_obs)
            
        predicted_actions = predicted_actions.cpu().numpy()
        
        print(f"📊 Model Output Analysis:")
        print(f"   Shape: {predicted_actions.shape}")
        print(f"   Range: [{predicted_actions.min():.6f}, {predicted_actions.max():.6f}]")
        print(f"   Mean: {predicted_actions.mean():.6f}")
        print(f"   Std: {predicted_actions.std():.6f}")
        print(f"   Mean Magnitude: {np.abs(predicted_actions).mean():.6f}")
        
        # Per-action analysis
        print(f"\n📋 Per-Action Statistics (first 10 actions):")
        for a in range(min(10, predicted_actions.shape[1])):
            action_values = predicted_actions[:, a]
            print(f"   Action {a}: range=[{action_values.min():.4f}, {action_values.max():.4f}], "
                  f"mean={action_values.mean():.4f}, std={action_values.std():.4f}")
                  
        return predicted_actions
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def debug_environment_action_limits():
    """Check what action ranges the environment expects"""
    print(f"\n🏃 DEBUGGING ENVIRONMENT ACTION LIMITS")
    print("=" * 50)
    
    # gs.init(backend=gs.gpu)
    env = SkeletonHumanoidEnv(num_envs=1, episode_length_s=8.0, dt=0.01, use_box_feet=True)
    
    print(f"✅ Environment created:")
    print(f"   Number of actions: {env.num_actions}")
    print(f"   Motor DOF indices: {env.motors_dof_idx}")
    
    # Check what happens with different action ranges
    test_actions = [
        torch.zeros(env.num_actions),              # Zero actions
        torch.ones(env.num_actions) * 0.1,        # Small actions  
        torch.ones(env.num_actions) * 0.5,        # Medium actions
        torch.ones(env.num_actions) * 1.0,        # Large actions
        torch.ones(env.num_actions) * 2.0,        # Very large actions
    ]
    
    print(f"\n📋 Testing Action Ranges:")
    for i, actions in enumerate(test_actions):
        print(f"   Test {i}: magnitude={torch.norm(actions).item():.4f}, "
              f"range=[{actions.min():.4f}, {actions.max():.4f}]")

def main():
    """Run all debug checks"""
    print("🚀 ACTION RANGE DEBUG ANALYSIS")
    print("=" * 70)
    
    # 1. Check training data ranges
    training_data = debug_training_data_ranges()
    
    # 2. Check model output ranges (you'll need to update this path)
    model_path = "/home/ez/Documents/Genesis/genesis_loco/final_single_trajectory_seg0-500_20250823_161840.pth"  # Update this path
    model_outputs = debug_model_outputs(model_path)
    
    # 3. Check environment expectations
    debug_environment_action_limits()
    
    # 4. Compare ranges
    if training_data is not None and model_outputs is not None:
        print(f"\n🔍 RANGE COMPARISON")
        print("=" * 50)
        
        training_range = (training_data.min(), training_data.max())
        model_range = (model_outputs.min(), model_outputs.max())
        
        print(f"Training data range: [{training_range[0]:.6f}, {training_range[1]:.6f}]")
        print(f"Model output range:  [{model_range[0]:.6f}, {model_range[1]:.6f}]")
        
        range_ratio = (model_range[1] - model_range[0]) / (training_range[1] - training_range[0])
        print(f"Range ratio (model/training): {range_ratio:.6f}")
        
        if range_ratio < 0.1:
            print("⚠️  WARNING: Model outputs are much smaller than training targets!")
            print("   This could cause the skeleton to collapse due to insufficient control.")
        elif range_ratio > 10.0:
            print("⚠️  WARNING: Model outputs are much larger than training targets!")
            print("   This could cause instability or oscillations.")
        else:
            print("✅ Model output range seems reasonable compared to training data.")

if __name__ == "__main__":
    main()