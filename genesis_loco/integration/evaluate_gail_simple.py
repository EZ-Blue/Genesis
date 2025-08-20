"""
Simple GAIL Policy Evaluation Script

Quick script to load and evaluate a trained GAIL policy without visualization complexity.
Good for testing if your trained model works.
"""

import torch
import time
import os
import sys

# Add genesis_loco to path
sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.ppo_policy import PPOActorCritic


def load_and_evaluate_policy(policy_path: str, num_steps: int = 500, show_viewer: bool = True):
    """
    Load a trained policy and evaluate it
    
    Args:
        policy_path: Path to the saved policy checkpoint
        num_steps: Number of steps to evaluate
        show_viewer: Whether to show the Genesis viewer
    """
    print(f"🧪 Simple GAIL Policy Evaluation")
    print(f"Policy: {policy_path}")
    print(f"Steps: {num_steps}")
    print(f"Viewer: {show_viewer}")
    
    # Check if policy file exists
    if not os.path.exists(policy_path):
        print(f"❌ Policy file not found: {policy_path}")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        # Initialize Genesis
        print("\n1. Initializing Genesis...")
        gs.init(backend=gs.gpu)
        
        # Create environment (small number of envs for testing)
        print("2. Creating environment...")
        env = SkeletonHumanoidEnv(
            num_envs=1,  # Single environment for simplicity
            episode_length_s=10.0,
            dt=0.02,
            use_box_feet=True,
            show_viewer=show_viewer
        )
        print(f"   Environment created: {env.num_observations} obs, {env.num_actions} actions")
        
        # Load policy
        print("3. Loading policy...")
        checkpoint = torch.load(policy_path, map_location=device)
        
        # Create policy network
        policy = PPOActorCritic(
            obs_dim=env.num_observations,
            action_dim=env.num_actions,
            hidden_layers=[512, 256],
            activation='tanh',
            init_std=0.125,
            learnable_std=False,
            use_running_norm=True
        )
        
        # Load weights
        policy.load_state_dict(checkpoint['policy_state'])
        policy.to(device)
        policy.eval()
        print(f"   Policy loaded: {sum(p.numel() for p in policy.parameters())} parameters")
        
        # Run evaluation
        print("4. Running evaluation...")
        obs, _ = env.reset()
        
        total_reward = 0.0
        step_count = 0
        episode_count = 0
        start_time = time.time()
        
        print("   Starting policy rollout...")
        if show_viewer:
            print("   Genesis viewer should be visible")
        
        for step in range(num_steps):
            # Get action from policy (deterministic)
            with torch.no_grad():
                action_dist, value = policy(obs)
                action = action_dist.mean  # Use mean for deterministic evaluation
            
            # Step environment
            obs, reward, reset_buf, info = env.step(action)
            
            total_reward += reward.mean().item()
            step_count += 1
            
            # Check for episode reset
            if reset_buf.any():
                episode_count += 1
                avg_reward = total_reward / step_count if step_count > 0 else 0.0
                print(f"   Episode {episode_count} completed: {step_count} steps, avg reward: {avg_reward:.3f}")
                total_reward = 0.0
                step_count = 0
            
            # Print progress
            if (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"   Step {step + 1}/{num_steps} ({steps_per_sec:.1f} steps/sec)")
            
            # Small delay for visualization
            if show_viewer:
                time.sleep(0.01)
        
        # Final statistics
        total_time = time.time() - start_time
        final_avg_reward = total_reward / max(step_count, 1)
        
        print(f"\n✅ Evaluation completed!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Steps per second: {num_steps / total_time:.1f}")
        print(f"   Episodes completed: {episode_count}")
        print(f"   Final average reward: {final_avg_reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest policy checkpoint in a directory"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for policy files
    policy_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('gail_policy_') and file.endswith('.pth'):
            policy_files.append(os.path.join(checkpoint_dir, file))
    
    if not policy_files:
        raise FileNotFoundError(f"No policy checkpoints found in: {checkpoint_dir}")
    
    # Return the final checkpoint if it exists, otherwise the latest numbered one
    final_checkpoint = os.path.join(checkpoint_dir, 'gail_policy_final.pth')
    if os.path.exists(final_checkpoint):
        return final_checkpoint
    
    # Sort by modification time and return newest
    policy_files.sort(key=os.path.getmtime, reverse=True)
    return policy_files[0]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple GAIL policy evaluation')
    parser.add_argument('--policy_path', type=str, help='Path to policy checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory containing checkpoints (will find latest)')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of evaluation steps')
    parser.add_argument('--no_viewer', action='store_true', help='Disable Genesis viewer')
    
    args = parser.parse_args()
    
    # Determine policy path
    if args.policy_path:
        policy_path = args.policy_path
    elif args.checkpoint_dir:
        try:
            policy_path = find_latest_checkpoint(args.checkpoint_dir)
            print(f"Found latest checkpoint: {policy_path}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            exit(1)
    else:
        print("❌ Must provide either --policy_path or --checkpoint_dir")
        exit(1)
    
    # Run evaluation
    success = load_and_evaluate_policy(
        policy_path=policy_path,
        num_steps=args.num_steps,
        show_viewer=not args.no_viewer
    )
    
    exit(0 if success else 1)