"""
Test Trained Walking Agent

Load a trained walking model and visualize the performance.
"""

import torch
import numpy as np
import time
import sys
import os
from typing import Optional

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_policy import SkeletonPolicyNetwork
from simple_trainer import create_default_config


class WalkingTester:
    """
    Test trained walking agents with visualization
    """
    
    def __init__(self, model_path: str, config_override: Optional[dict] = None):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🧪 Loading trained walking model: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Override config if provided (e.g., to change visualization settings)
        if config_override:
            self.config.update(config_override)
        
        print(f"   Model from iteration: {checkpoint.get('iteration', 'unknown')}")
        print(f"   Model reward: {checkpoint.get('reward', 'unknown')}")
        
        # Setup environment
        self._setup_environment()
        
        # Load policy
        self._load_policy(checkpoint)
        
        print("✅ Walking tester ready!")
    
    def _setup_environment(self):
        """Setup Genesis environment for testing"""
        import genesis as gs
        gs.init(backend=gs.gpu)
        
        # Import skeleton environment
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        # Force enable viewer for testing
        test_config = self.config.copy()
        test_config['show_viewer'] = True
        test_config['num_envs'] = 1  # Single environment for testing
        
        self.env = SkeletonHumanoidEnv(
            num_envs=test_config['num_envs'],
            episode_length_s=test_config['episode_length_s'],
            dt=test_config['dt'],
            show_viewer=test_config['show_viewer']
        )
        
        print(f"   ✓ Test environment: {self.env.num_observations} obs, {self.env.num_actions} actions")
    
    def _load_policy(self, checkpoint):
        """Load trained policy network"""
        self.policy = SkeletonPolicyNetwork(
            obs_dim=self.env.num_observations,
            action_dim=self.env.num_actions,
            hidden_layers=self.config['policy']['hidden_layers'],
            activation=self.config['policy']['activation'],
            use_obs_norm=True
        ).to(self.device)
        
        # Load trained weights
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()  # Set to evaluation mode
        
        param_count = sum(p.numel() for p in self.policy.parameters())
        print(f"   ✓ Policy loaded: {param_count} parameters")
    
    def test_walking(self, num_episodes: int = 5, max_steps_per_episode: int = 1000):
        """
        Test the trained walking agent
        
        Args:
            num_episodes: Number of test episodes
            max_steps_per_episode: Maximum steps per episode
        """
        print(f"\n🚶‍♂️ Testing Walking Agent")
        print(f"   Episodes: {num_episodes}")
        print(f"   Max steps per episode: {max_steps_per_episode}")
        print("=" * 50)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            print(f"\n📺 Episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            print("   Press Enter to start episode (or 'q' to quit)...")
            user_input = input().strip().lower()
            if user_input == 'q':
                break
            
            # Run episode
            for step in range(max_steps_per_episode):
                # Sample action from trained policy
                with torch.no_grad():
                    actions, _, _ = self.policy.sample_actions(obs)
                
                # Environment step
                obs, rewards, dones, _ = self.env.step(actions)
                
                episode_reward += rewards[0].item()
                episode_length += 1
                
                # Check if episode ended
                if dones[0]:
                    print(f"   Episode ended at step {step + 1}")
                    break
                
                # Small delay for better visualization
                time.sleep(0.02)  # 50Hz visualization
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"   ✓ Episode reward: {episode_reward:.3f}")
            print(f"   ✓ Episode length: {episode_length} steps")
            print(f"   ✓ Average reward per step: {episode_reward/episode_length:.4f}")
        
        # Summary statistics
        if episode_rewards:
            print(f"\n📊 Test Results Summary:")
            print(f"   Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
            print(f"   Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
            print(f"   Best episode: {max(episode_rewards):.3f}")
            print(f"   Longest episode: {max(episode_lengths)} steps")
    
    def interactive_test(self):
        """
        Interactive testing mode with controls
        """
        print(f"\n🎮 Interactive Walking Test Mode")
        print("Controls:")
        print("  Enter - Run one episode")
        print("  'r' - Reset current episode")  
        print("  'q' - Quit")
        print("=" * 50)
        
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        while True:
            print(f"\nCurrent episode - Reward: {episode_reward:.3f}, Length: {episode_length} steps")
            command = input("Command (Enter/r/q): ").strip().lower()
            
            if command == 'q':
                break
            elif command == 'r':
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                print("Episode reset!")
                continue
            
            # Run single step or full episode
            for step in range(100):  # Run 100 steps at a time
                with torch.no_grad():
                    actions, _, _ = self.policy.sample_actions(obs)
                
                obs, rewards, dones, _ = self.env.step(actions)
                episode_reward += rewards[0].item()
                episode_length += 1
                
                if dones[0]:
                    print(f"Episode ended! Final reward: {episode_reward:.3f}, Length: {episode_length}")
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_length = 0
                    break
                
                time.sleep(0.02)


def main():
    """Main testing function"""
    print("🧪 Genesis Walking Agent Tester")
    print("=" * 40)
    
    # Find available models
    print("Looking for trained models...")
    
    model_files = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('walking_training_'):
            best_model_path = os.path.join(item, 'best_model.pt')
            if os.path.exists(best_model_path):
                model_files.append(best_model_path)
    
    if not model_files:
        print("❌ No trained models found!")
        print("Run train_walking.py first to train a model.")
        return
    
    print(f"Found {len(model_files)} trained models:")
    for i, model_path in enumerate(model_files):
        # Extract timestamp from path
        timestamp = model_path.split('_')[2].split('/')[0]
        print(f"  {i+1}. {model_path} (trained: {timestamp})")
    
    # Select model
    if len(model_files) == 1:
        selected_model = model_files[0]
        print(f"Using: {selected_model}")
    else:
        choice = input(f"Select model (1-{len(model_files)}): ").strip()
        try:
            selected_model = model_files[int(choice) - 1]
        except (ValueError, IndexError):
            selected_model = model_files[0]
            print(f"Invalid choice, using: {selected_model}")
    
    # Test mode selection
    print("\nSelect test mode:")
    print("1. Automatic Testing (5 episodes)")
    print("2. Interactive Mode (manual control)")
    
    mode = input("Enter choice (1/2): ").strip()
    
    # Initialize tester
    tester = WalkingTester(selected_model)
    
    if mode == "2":
        tester.interactive_test()
    else:
        tester.test_walking(num_episodes=5)
    
    print("\n🎯 Testing complete!")


if __name__ == "__main__":
    main()