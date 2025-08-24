"""
Visualize Trained RL Walking Policy

Efficient visualization script for skeleton humanoid RL model.
Loads saved model and runs it in real-time with viewer.
"""

import torch
import argparse
import os
import sys
import time
import numpy as np

# Add parent directory to path for imports
sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.ppo_policy import PPOActorCritic


def load_model(checkpoint_path: str, device: str = "cuda") -> tuple:
    """Load trained model and configuration"""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Fix for PyTorch 2.6 weights_only default change
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"⚠️  Warning: Using weights_only=False for compatibility with older checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    obs_dim = config['obs_dim']
    action_dim = config['action_dim']
    
    # Initialize policy with same architecture as training
    policy = PPOActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=[512, 256, 128],
        activation='tanh',
        init_std=0.3,
        learnable_std=True,
        use_running_norm=True
    ).to(device)
    
    # Load trained weights
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()  # Set to evaluation mode
    
    print(f"✅ Model loaded successfully!")
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Total training timesteps: {checkpoint.get('training_metrics', {}).get('total_timesteps', 'Unknown')}")
    
    return policy, config


def create_eval_env(config: dict, show_viewer: bool = True) -> SkeletonHumanoidEnv:
    """Create environment for evaluation"""
    print("🏃 Creating evaluation environment...")
    
    # Use single environment for visualization
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=config.get('episode_length_s', 20.0),  # Longer episodes for visualization
        use_box_feet=True,  # Always use box feet for stability
        show_viewer=show_viewer,
        dt=0.02  # 50 Hz for smooth visualization
    )
    
    return env


def run_inference_loop(policy: PPOActorCritic, env: SkeletonHumanoidEnv, max_steps: int = 10000):
    """Run inference loop with the trained policy"""
    print("🚀 Starting inference loop...")
    print("   Press Ctrl+C to stop")
    
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    
    try:
        with torch.no_grad():
            for step in range(max_steps):
                # Get action from policy (use mean for deterministic evaluation)
                action_dist, value = policy(obs)
                action = action_dist.mean  # Use mean instead of sampling for consistent behavior
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward[0].item()
                episode_length += 1
                
                # Handle episode termination/reset
                if done[0]:
                    episode_count += 1
                    print(f"Episode {episode_count}: Length={episode_length}, Reward={episode_reward:.2f}")
                    episode_reward = 0.0
                    episode_length = 0
                    
                # Add small delay for smooth visualization
                time.sleep(0.01)  # ~100 FPS display
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopping inference...")
        print(f"Completed {episode_count} episodes")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Visualize trained RL walking policy")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                      help="Path to model checkpoint (.pth file)")
    parser.add_argument("--no-viewer", action="store_true",
                      help="Run without viewer (for performance testing)")
    parser.add_argument("--max-steps", type=int, default=10000,
                      help="Maximum number of steps to run")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("🦴 Skeleton Humanoid RL Walking Visualization")
    print("=" * 50)
    
    # Initialize Genesis
    gs.init(backend=gs.cuda if args.device == "cuda" else gs.cpu)
    
    # Load trained model
    try:
        policy, config = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create evaluation environment
    try:
        env = create_eval_env(config, show_viewer=not args.no_viewer)
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return
    
    print("=" * 50)
    
    # Run inference loop
    run_inference_loop(policy, env, args.max_steps)
    
    print("✅ Visualization completed!")


if __name__ == "__main__":
    main()