#!/usr/bin/env python3
"""
Test Trained Imitation Learning Model

Load and evaluate a trained walking/running/squatting model.
"""

import torch
import sys
import os
import time
from typing import Dict

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.simple_policy import SkeletonPolicyNetwork
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


class ModelTester:
    """Test trained imitation learning models"""
    
    def __init__(self, model_path: str, show_viewer: bool = True):
        self.model_path = model_path
        self.show_viewer = show_viewer
        
        print(f"🧪 Loading and Testing Trained Model")
        print(f"   Model: {model_path}")
        print(f"   Viewer: {show_viewer}")
        print("=" * 60)
        
        # Initialize Genesis
        success, message = safe_init_genesis()
        if not success:
            raise RuntimeError(message)
        print(f"   ✅ {message}")
        
        self.device = gs.device
        
        # Load model
        self._load_model()
        
        # Setup environment
        self._setup_environment()
        
    def _load_model(self):
        """Load the trained model"""
        print("1. Loading trained model...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model info
        self.behavior = checkpoint.get('behavior', 'unknown')
        self.config = checkpoint.get('config', {})
        self.iteration = checkpoint.get('iteration', 0)
        self.reward = checkpoint.get('reward', 0.0)
        
        print(f"   ✅ Model loaded:")
        print(f"      - Behavior: {self.behavior}")
        print(f"      - Training iteration: {self.iteration}")
        print(f"      - Best reward: {self.reward:.3f}")
        
        # Get model dimensions from config or guess
        obs_dim = 65  # Default skeleton observation dim
        action_dim = 27  # Default skeleton action dim (with box feet)
        
        # Create policy network
        policy_config = self.config.get('policy', {})
        self.policy = SkeletonPolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layers=policy_config.get('hidden_layers', [512, 256, 128]),
            activation=policy_config.get('activation', 'tanh'),
            use_obs_norm=policy_config.get('use_obs_norm', True)
        )
        
        # Load policy weights
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        self.policy.to(self.device)
        
        print(f"   ✅ Policy network loaded")
        
    def _setup_environment(self):
        """Setup test environment"""
        print("2. Setting up test environment...")
        
        # Use single environment for testing
        self.env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=30.0,  # Longer episodes for testing
            dt=0.01,
            show_viewer=self.show_viewer,
            use_box_feet=True
        )
        
        print(f"   ✅ Environment ready:")
        print(f"      - Observations: {self.env.num_observations}")
        print(f"      - Actions: {self.env.num_actions}")
        print(f"      - Episode length: 30.0s")
        
    def test_model(self, num_episodes: int = 5):
        """Test the model for multiple episodes"""
        print(f"\n🚀 Testing Model - {num_episodes} Episodes")
        print("=" * 60)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            obs, _ = self.env.reset()
            
            episode_reward = 0.0
            episode_length = 0
            start_time = time.time()
            
            done = False
            while not done:
                # Get action from policy
                with torch.no_grad():
                    actions, _, _ = self.policy.sample_actions(obs)
                
                # Step environment
                obs, rewards, dones, info = self.env.step(actions)
                
                episode_reward += rewards[0].item()
                episode_length += 1
                
                done = dones[0].item()
                
                # Safety check for infinite episodes
                if episode_length >= 3000:  # 30 seconds at 100Hz
                    print("   ⏰ Episode reached maximum length")
                    break
            
            episode_time = time.time() - start_time
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"   Reward: {episode_reward:8.2f}")
            print(f"   Length: {episode_length:4d} steps ({episode_length/100:.1f}s)")
            print(f"   Real time: {episode_time:.1f}s")
            
            if done and episode_length < 2000:
                print(f"   ⚠️ Early termination at {episode_length} steps")
        
        # Summary statistics
        print(f"\n📈 Test Results Summary:")
        print(f"   Average reward: {sum(episode_rewards)/len(episode_rewards):8.2f}")
        print(f"   Average length: {sum(episode_lengths)/len(episode_lengths):8.1f} steps")
        print(f"   Best reward: {max(episode_rewards):8.2f}")
        print(f"   Worst reward: {min(episode_rewards):8.2f}")
        
        # Performance assessment
        avg_length = sum(episode_lengths) / len(episode_lengths)
        if avg_length > 1500:  # > 15 seconds
            print("   🎉 EXCELLENT: Model maintains balance for long periods!")
        elif avg_length > 500:   # > 5 seconds  
            print("   👍 GOOD: Model shows decent stability")
        else:
            print("   📉 NEEDS WORK: Model falls quickly")
            
        return episode_rewards, episode_lengths
    
    def interactive_test(self):
        """Interactive testing mode"""
        print(f"\n🎮 Interactive Testing Mode")
        print("Press Enter to run episode, 'q' to quit")
        print("=" * 60)
        
        episode_num = 1
        while True:
            user_input = input(f"\nEpisode {episode_num} (Enter to run, 'q' to quit): ").strip()
            
            if user_input.lower() == 'q':
                break
                
            # Run single episode
            self.test_model(num_episodes=1)
            episode_num += 1
        
        print("👋 Interactive testing ended")


def main():
    """Main testing function"""
    print("🤖 Trained Model Tester")
    print("=" * 50)
    
    # Get model path
    model_path = input("Enter path to trained model (or press Enter for default): ").strip()
    
    if not model_path:
        # Look for recent training directories
        import glob
        recent_dirs = glob.glob("imitation_*_*/best_model.pt")
        if recent_dirs:
            model_path = recent_dirs[-1]  # Most recent
            print(f"Using most recent model: {model_path}")
        else:
            print("❌ No model path provided and no recent training found")
            print("Please provide path to best_model.pt file")
            return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Viewer option
    viewer_choice = input("Enable viewer? (y/n, default=y): ").strip().lower()
    show_viewer = viewer_choice != 'n'
    
    try:
        # Create tester
        tester = ModelTester(model_path, show_viewer=show_viewer)
        
        # Test mode selection
        print("\nTest modes:")
        print("1. Quick test (5 episodes)")
        print("2. Extended test (20 episodes)")
        print("3. Interactive mode")
        
        mode = input("Select mode (1/2/3, default=1): ").strip()
        
        if mode == "2":
            tester.test_model(num_episodes=20)
        elif mode == "3":
            tester.interactive_test()
        else:
            tester.test_model(num_episodes=5)
            
        print(f"\n✅ Testing completed!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()