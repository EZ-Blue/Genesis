"""
Force Trained BC Actions

Test trained behavior cloning model by directly forcing joint positions instead
of using PD control. This bypasses physics control issues and shows if the model
is predicting correct joint trajectories.

The script applies model predictions directly as joint positions and steps
the physics to see the resulting motion.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import glob
from typing import Dict, List

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv


class SingleTrajectoryMLP(nn.Module):
    """
    Recreate the model architecture (must match training script)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        # Hidden layers with dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


def find_latest_model(pattern: str = "*single_trajectory*.pth") -> str:
    """Find the most recent model file"""
    model_files = glob.glob(pattern)
    if not model_files:
        raise FileNotFoundError(f"No model files found matching pattern: {pattern}")
    
    # Sort by modification time, newest first
    latest_file = max(model_files, key=os.path.getmtime)
    print(f"📂 Found latest model: {latest_file}")
    return latest_file


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load trained model and configuration"""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Segment: {checkpoint.get('start_timestep', 'unknown')}-{checkpoint.get('end_timestep', 'unknown')}")
    
    # Create model with same architecture as training
    model = SingleTrajectoryMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[512, 256],  # Match training script
        dropout_rate=0.0  # No dropout during inference
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")
    
    return model, obs_dim, action_dim


def run_forced_action_evaluation(model: nn.Module, max_steps: int = 1000) -> None:
    """
    Run model evaluation by directly forcing joint positions instead of PD control
    """
    print(f"🔧 Starting FORCED ACTION evaluation...")
    print(f"   Max steps: {max_steps}")
    print("   Mode: Direct joint position application (bypassing PD control)")
    print("   Press Ctrl+C to stop early")
    
    # Create environment (must match training configuration)
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=20.0,  # Longer episodes for evaluation
        dt=0.01,  # Match training dt
        show_viewer=True,
        use_box_feet=True,
        obs_history_length=1
    )
    
    # Initialize environment
    obs, _ = env.reset()
    
    episode_length = 0
    episode_count = 0
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            for step in range(max_steps):
                # Get predicted joint positions from model
                predicted_joint_positions = model(obs[0])  # Single environment
                
                # FORCE the joint positions directly instead of using PD control
                # This bypasses the physics control and shows if predictions are correct
                
                # Prepare joint positions for all environments (we only have 1)
                joint_positions = predicted_joint_positions.unsqueeze(0)  # [1, action_dim]
                
                # Apply joint positions directly to robot
                env_ids = torch.tensor([0], device=env.device)
                env.robot.set_dofs_position(
                    joint_positions,
                    dofs_idx_local=env.motors_dof_idx,
                    envs_idx=env_ids,
                    zero_velocity=False  # Keep some velocity for natural motion
                )
                
                # Step physics without PD control actions
                env.scene.step()
                
                # Update robot state buffers manually (since we bypassed step())
                env._update_robot_state()
                
                # Get new observations manually
                obs = env._get_observations()
                
                # Get rewards manually (optional - just for logging)
                # rewards = env._get_rewards()
                # episode_reward = rewards[0].item() if rewards is not None else 0.0
                
                # Check termination manually using environment's method
                done = env._check_termination()
                
                episode_length += 1
                
                # Handle episode termination
                if done[0] if done is not None else False:
                    episode_count += 1
                    elapsed = time.time() - start_time
                    print(f"Episode {episode_count}: Length={episode_length}, "
                          f"Time={elapsed:.1f}s")
                    
                    # Reset if needed
                    if episode_count < 5:  # Allow a few episodes
                        obs, _ = env.reset()
                    
                    episode_length = 0
                    start_time = time.time()
                
                # Progress logging
                if step % 100 == 0:
                    print(f"Step {step}: Joint pos range [{predicted_joint_positions.min():.3f}, {predicted_joint_positions.max():.3f}]")
                
                # Small delay for smooth visualization
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print(f"\n⏹️  Evaluation stopped by user")
        
    print(f"✅ Forced action evaluation completed!")
    print(f"   Total episodes: {episode_count}")
    print(f"   Final episode length: {episode_length}")
    print(f"   This shows model predictions without PD control interference")


def compare_pd_vs_forced(model: nn.Module, comparison_steps: int = 200) -> None:
    """
    Compare PD control vs forced actions side by side
    """
    print(f"\n🔄 Running PD vs FORCED comparison...")
    
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=20.0,
        dt=0.01,
        show_viewer=True,
        use_box_feet=True,
        obs_history_length=1
    )
    
    obs, _ = env.reset()
    initial_obs = obs.clone()
    
    print(f"Phase 1: PD Control (first {comparison_steps} steps)")
    
    # Phase 1: PD Control
    with torch.no_grad():
        for step in range(comparison_steps):
            action = model(obs[0])
            obs, reward, done, info = env.step(action.unsqueeze(0))
            time.sleep(0.02)
            
            if done[0]:
                break
    
    print(f"Phase 1 complete. Resetting for Phase 2...")
    time.sleep(2)
    
    # Reset to same initial state
    env.reset()
    obs = initial_obs
    
    print(f"Phase 2: FORCED Actions (next {comparison_steps} steps)")
    
    # Phase 2: Forced actions
    with torch.no_grad():
        for step in range(comparison_steps):
            predicted_joint_positions = model(obs[0])
            
            # Force joint positions
            joint_positions = predicted_joint_positions.unsqueeze(0)
            env_ids = torch.tensor([0], device=env.device)
            env.robot.set_dofs_position(
                joint_positions,
                dofs_idx_local=env.motors_dof_idx,
                envs_idx=env_ids,
                zero_velocity=False
            )
            
            # Manual physics step and state update
            env.scene.step()
            env._update_robot_state()
            obs = env._get_observations()
            
            time.sleep(0.02)
    
    print("✅ Comparison complete!")
    print("   You should see clear differences between PD control and forced actions")


def main():
    """Main evaluation function"""
    print("🔧 FORCED ACTION BEHAVIOR CLONING EVALUATION")
    print("=" * 60)
    print("This script tests if BC model predictions are correct by forcing")
    print("joint positions directly, bypassing PD control physics issues.")
    print("=" * 60)
    
    # Initialize Genesis
    gs.init(backend=gs.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get model path
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return 1
    else:
        try:
            model_path = find_latest_model()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("Usage: python force_trained_bc_actions.py [model_path.pth]")
            return 1
    
    try:
        # Load model
        model, obs_dim, action_dim = load_model(model_path, device)
        
        print("=" * 60)
        print("Choose evaluation mode:")
        print("1. Forced actions only")
        print("2. PD vs Forced comparison")
        
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "2":
            # Run comparison
            compare_pd_vs_forced(model, comparison_steps=300)
        else:
            # Run forced actions only
            run_forced_action_evaluation(model, max_steps=1500)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())