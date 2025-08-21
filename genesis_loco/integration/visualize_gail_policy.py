"""
GAIL Policy Visualization Script

Load and visualize trained GAIL walking policies in Genesis viewer.
Supports comparison with expert trajectories and multiple rendering options.
"""

import torch
import numpy as np
import time
import argparse
import os
from typing import Optional

# Genesis imports
import genesis as gs

# Local imports
import sys
sys.path.append('/home/ez/Documents/Genesis/genesis_loco')
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
from integration.ppo_policy import PPOActorCritic
from integration.gail_trainer import GAILConfig


class GAILPolicyVisualizer:
    """
    Visualizer for trained GAIL policies
    
    Loads saved models and renders walking behavior in Genesis viewer
    """
    
    def __init__(self, 
                 policy_path: str,
                 num_envs: int = 1,
                 episode_length_s: float = 10.0,
                 show_viewer: bool = True):
        
        self.policy_path = policy_path
        self.num_envs = num_envs
        self.episode_length_s = episode_length_s
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🎬 GAIL Policy Visualizer")
        print(f"   Policy: {policy_path}")
        print(f"   Environments: {num_envs}")
        print(f"   Device: {self.device}")
        
        # Initialize Genesis
        print("Initializing Genesis...")
        gs.init(backend=gs.gpu)
        
        # Create environment
        print("Creating environment...")
        self.env = SkeletonHumanoidEnv(
            num_envs=num_envs,
            episode_length_s=episode_length_s,
            dt=0.02,
            use_box_feet=True,
            show_viewer=show_viewer
        )
        
        # Load policy
        print("Loading policy...")
        self.policy = self._load_policy()
        print(f"   ✅ Policy loaded: {sum(p.numel() for p in self.policy.parameters())} parameters")
        
        # Initialize data bridge for expert comparison
        self.data_bridge = None
        self.expert_observations = None
    
    def _load_policy(self) -> PPOActorCritic:
        """Load trained policy from checkpoint"""
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")
        
        checkpoint = torch.load(self.policy_path, map_location=self.device)
        
        # Create policy network (using standard config)
        policy = PPOActorCritic(
            obs_dim=self.env.num_observations,
            action_dim=self.env.num_actions,
            # hidden_layers=[512, 256],
            hidden_layers=[256, 128],
            activation='tanh',
            init_std=0.125,
            learnable_std=False,
            use_running_norm=True
        )
        
        # Load weights
        policy.load_state_dict(checkpoint['policy_state'])
        policy.to(self.device)
        policy.eval()
        
        return policy
    
    def load_expert_trajectory(self, trajectory_name: str = "walk") -> bool:
        """Load expert trajectory for comparison"""
        print(f"Loading expert trajectory: {trajectory_name}...")
        
        try:
            self.data_bridge = LocoMujocoDataBridge(self.env)
            success = self.data_bridge.load_trajectory(trajectory_name)
            
            if not success:
                print(f"   ❌ Failed to load trajectory: {trajectory_name}")
                return False
            
            print(f"   ✅ Expert trajectory loaded: {self.data_bridge.trajectory_length} timesteps")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading expert trajectory: {e}")
            return False
    
    def run_policy(self, 
                   num_steps: Optional[int] = None,
                   deterministic: bool = True,
                   render_fps: int = 30) -> dict:
        """
        Run policy and render in viewer
        
        Args:
            num_steps: Number of steps to run (None = infinite)
            deterministic: Use deterministic actions (mean of policy)
            render_fps: Target rendering FPS
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"\n🏃 Running policy visualization")
        print(f"   Steps: {'∞' if num_steps is None else num_steps}")
        print(f"   Deterministic: {deterministic}")
        print(f"   Target FPS: {render_fps}")
        print("   Press 'q' to quit, 'r' to reset")
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Set deterministic mode
        if deterministic and hasattr(self.policy, 'log_std'):
            self.policy.log_std.data.fill_(-10.0)  # Very low std for deterministic actions
        
        step_count = 0
        episode_count = 0
        total_reward = 0.0
        episode_rewards = []
        
        start_time = time.time()
        last_render_time = start_time
        target_dt = 1.0 / render_fps
        
        try:
            while num_steps is None or step_count < num_steps:
                # Get action from policy
                with torch.no_grad():
                    if deterministic:
                        action_dist, _ = self.policy(obs)
                        action = action_dist.mean
                    else:
                        action, _, _, _ = self.policy.get_action_and_value(obs)
                
                # Step environment
                obs, reward, reset_buf, info = self.env.step(action)
                
                total_reward += reward.mean().item()
                step_count += 1
                
                # Handle resets
                if reset_buf.any():
                    episode_rewards.append(total_reward / step_count if step_count > 0 else 0.0)
                    episode_count += 1
                    total_reward = 0.0
                    
                    print(f"   Episode {episode_count} completed, avg reward: {episode_rewards[-1]:.3f}")
                
                # Control rendering rate
                current_time = time.time()
                elapsed_since_render = current_time - last_render_time
                
                if elapsed_since_render >= target_dt:
                    # Genesis handles its own rendering when viewer is enabled
                    last_render_time = current_time
                    
                    # Small sleep to prevent overwhelming the GPU
                    time.sleep(0.001)
                
                # Print progress periodically
                if step_count % 100 == 0:
                    elapsed_time = current_time - start_time
                    steps_per_sec = step_count / elapsed_time if elapsed_time > 0 else 0
                    print(f"   Step {step_count}, {steps_per_sec:.1f} steps/sec, avg reward: {total_reward/100:.3f}")
                    total_reward = 0.0
        
        except KeyboardInterrupt:
            print("\n   Visualization interrupted by user")
        
        # Calculate final metrics
        final_time = time.time() - start_time
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        
        metrics = {
            'total_steps': step_count,
            'total_episodes': episode_count,
            'total_time': final_time,
            'steps_per_second': step_count / final_time if final_time > 0 else 0,
            'average_episode_reward': avg_reward,
            'episode_rewards': episode_rewards
        }
        
        print(f"\n📊 Visualization Summary:")
        print(f"   Total steps: {metrics['total_steps']}")
        print(f"   Total episodes: {metrics['total_episodes']}")
        print(f"   Runtime: {metrics['total_time']:.1f}s")
        print(f"   Steps/sec: {metrics['steps_per_second']:.1f}")
        print(f"   Avg episode reward: {metrics['average_episode_reward']:.3f}")
        
        return metrics
    
    def compare_with_expert(self, 
                           trajectory_name: str = "walk",
                           num_steps: int = 500,
                           side_by_side: bool = True):
        """
        Compare policy with expert trajectory
        
        Args:
            trajectory_name: Expert trajectory to load
            num_steps: Number of steps to compare
            side_by_side: Show policy and expert side by side (requires 2+ envs)
        """
        if not self.load_expert_trajectory(trajectory_name):
            print("❌ Cannot compare without expert trajectory")
            return
        
        if side_by_side and self.num_envs < 2:
            print("⚠️  Side-by-side comparison requires at least 2 environments")
            side_by_side = False
        
        print(f"\n🔄 Comparing policy with expert trajectory")
        print(f"   Side-by-side: {side_by_side}")
        
        # Reset environment
        obs, _ = self.env.reset()
        
        if side_by_side:
            # Environment 0: Policy, Environment 1: Expert
            env_0_ids = torch.tensor([0], device=self.device)
            env_1_ids = torch.tensor([1], device=self.device)
            
            print("   Environment 0: Policy actions")
            print("   Environment 1: Expert trajectory")
        
        for step in range(num_steps):
            if side_by_side:
                # Get policy action for env 0
                with torch.no_grad():
                    action_dist, _ = self.policy(obs)
                    policy_action = action_dist.mean
                
                # Create mixed actions: policy for env 0, zeros for env 1 (will be overridden)
                actions = torch.zeros((self.num_envs, self.env.num_actions), device=self.device)
                actions[0] = policy_action[0]
                
                # Apply expert state to env 1
                if step < self.data_bridge.trajectory_length:
                    expert_state = self.data_bridge.get_trajectory_state(step)
                    if expert_state is not None:
                        self.data_bridge.apply_trajectory_state(expert_state, env_1_ids)
            else:
                # Just run policy
                with torch.no_grad():
                    action_dist, _ = self.policy(obs)
                    actions = action_dist.mean
            
            # Step environment
            obs, reward, reset_buf, info = self.env.step(actions)
            
            # Small delay for visualization
            time.sleep(0.02)
            
            if step % 100 == 0:
                print(f"   Step {step}/{num_steps}")
        
        print("✅ Comparison complete")
    
    def save_recording(self, 
                      output_path: str,
                      num_steps: int = 1000,
                      deterministic: bool = True):
        """
        Save a recording of the policy
        
        Args:
            output_path: Path to save recording
            num_steps: Number of steps to record
            deterministic: Use deterministic actions
        """
        print(f"\n📹 Recording policy to: {output_path}")
        
        # Note: Genesis recording functionality would be used here
        # For now, we'll run the policy and let the user manually record
        print("   Start your screen recording software now...")
        print("   Press Enter when ready to begin demonstration...")
        input()
        
        metrics = self.run_policy(
            num_steps=num_steps, 
            deterministic=deterministic,
            render_fps=30
        )
        
        print(f"✅ Recording session completed")
        print("   Stop your screen recording software")
        
        return metrics


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description='Visualize trained GAIL walking policy')
    
    parser.add_argument('--policy_path', type=str, required=True, 
                       help='Path to saved policy checkpoint')
    parser.add_argument('--num_envs', type=int, default=1,
                       help='Number of environments (use 2+ for side-by-side comparison)')
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of steps to run (None = infinite)')
    parser.add_argument('--episode_length', type=float, default=10.0,
                       help='Episode length in seconds')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--render_fps', type=int, default=30,
                       help='Target rendering FPS')
    
    # Comparison options
    parser.add_argument('--compare_expert', action='store_true',
                       help='Compare with expert trajectory')
    parser.add_argument('--trajectory', type=str, default='walk',
                       help='Expert trajectory to use for comparison')
    parser.add_argument('--side_by_side', action='store_true',
                       help='Show policy and expert side by side')
    
    # Recording options
    parser.add_argument('--record', action='store_true',
                       help='Record the policy demonstration')
    parser.add_argument('--output_path', type=str, default='./gail_policy_demo.mp4',
                       help='Output path for recording')
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = GAILPolicyVisualizer(
            policy_path=args.policy_path,
            num_envs=args.num_envs,
            episode_length_s=args.episode_length,
            show_viewer=True
        )
        
        if args.compare_expert:
            # Compare with expert trajectory
            visualizer.compare_with_expert(
                trajectory_name=args.trajectory,
                num_steps=args.num_steps or 500,
                side_by_side=args.side_by_side
            )
        elif args.record:
            # Record demonstration
            visualizer.save_recording(
                output_path=args.output_path,
                num_steps=args.num_steps or 1000,
                deterministic=args.deterministic
            )
        else:
            # Standard visualization
            visualizer.run_policy(
                num_steps=args.num_steps,
                deterministic=args.deterministic,
                render_fps=args.render_fps
            )
    
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())