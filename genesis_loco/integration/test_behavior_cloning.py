#!/usr/bin/env python3
"""
Test Script for Trained Behavior Cloning Model

Loads a trained behavior cloning model and tests it in the Genesis environment
to see how well it learned to imitate the expert locomotion behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import glob
from typing import Dict, List, Tuple

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.behavior_cloning_trainer import BehaviorCloningMLP
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


class BehaviorCloningTester:
    """
    Test a trained behavior cloning model in the Genesis environment
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🧪 Behavior Cloning Model Tester")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        # Initialize components
        self._setup_genesis()
        self._load_model()
        self._setup_environment()
        
        # Test results
        self.test_metrics = {}
        
    def _setup_genesis(self):
        """Initialize Genesis physics"""
        success, message = safe_init_genesis()
        if not success:
            raise RuntimeError(message)
        print(f"✅ {message}")
    
    def _load_model(self):
        """Load the trained behavior cloning model"""
        print(f"📥 Loading model from {self.model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model dimensions
        self.obs_dim = checkpoint['obs_dim']
        self.action_dim = checkpoint['action_dim']
        self.behavior = checkpoint.get('behavior', 'unknown')
        
        # Create and load model
        self.model = BehaviorCloningMLP(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=[256, 128]
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set to evaluation mode
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded: {total_params:,} parameters")
        print(f"   Behavior: {self.behavior}")
        print(f"   Input: {self.obs_dim} observations -> Output: {self.action_dim} positions")
    
    def _setup_environment(self):
        """Setup Genesis skeleton environment"""
        self.env = SkeletonHumanoidEnv(
            num_envs=1,  # Single environment for testing
            episode_length_s=30.0,  # Longer episodes for evaluation
            dt=0.01,
            show_viewer=True,  # Enable visualization
            use_box_feet=True,
            obs_history_length=3  # Match training configuration
        )
        
        print(f"✅ Environment: obs_dim={self.env.num_observations}, action_dim={self.env.num_actions}")
        
        # Check dimension compatibility
        if self.obs_dim != self.env.num_observations:
            print(f"⚠️  Warning: Model obs_dim ({self.obs_dim}) != env obs_dim ({self.env.num_observations})")
        if self.action_dim != self.env.num_actions:
            print(f"⚠️  Warning: Model action_dim ({self.action_dim}) != env action_dim ({self.env.num_actions})")
    
    def _get_observation(self) -> torch.Tensor:
        """Get current observation from environment"""
        # Get raw observation from environment
        obs, _ = self.env.get_observations()
        
        # Convert to tensor and ensure correct shape
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs[0]  # Take first (and only) environment
        else:
            obs_tensor = torch.tensor(obs[0], dtype=torch.float32)
        
        # Ensure correct size
        if obs_tensor.shape[0] > self.obs_dim:
            obs_tensor = obs_tensor[:self.obs_dim]
        elif obs_tensor.shape[0] < self.obs_dim:
            padding = torch.zeros(self.obs_dim - obs_tensor.shape[0])
            obs_tensor = torch.cat([obs_tensor, padding])
        
        return obs_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def run_test_episode(self, max_steps: int = 3000, record_data: bool = True) -> Dict:
        """Run a single test episode"""
        print(f"\n🏃 Running test episode ({max_steps} steps)")
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Episode data
        episode_data = {
            'step': [],
            'reward': [],
            'height': [],
            'forward_pos': [],
            'actions': [],
            'observations': []
        }
        
        total_reward = 0.0
        step_count = 0
        
        for step in range(max_steps):
            # Get current observation
            obs_tensor = self._get_observation()
            
            # Predict target positions using trained model
            with torch.no_grad():
                predicted_positions = self.model(obs_tensor)
                actions = predicted_positions[0].cpu().numpy()  # Remove batch dimension
            
            # Take action in environment
            actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(self.env.device)
            obs, rewards, dones, info = self.env.step(actions_tensor)
            
            # Record data
            if record_data:
                root_pos = self.env.root_pos[0].cpu().numpy()
                episode_data['step'].append(step)
                episode_data['reward'].append(rewards[0].item())
                episode_data['height'].append(root_pos[2])
                episode_data['forward_pos'].append(root_pos[0])
                episode_data['actions'].append(actions.copy())
                episode_data['observations'].append(obs_tensor[0].cpu().numpy())
            
            total_reward += rewards[0].item()
            step_count += 1
            
            # Print progress
            if step % 500 == 0:
                root_pos = self.env.root_pos[0].cpu().numpy()
                print(f"   Step {step:4d}: Reward={rewards[0].item():.3f}, "
                      f"Height={root_pos[2]:.3f}, Forward={root_pos[0]:.3f}")
            
            # Check for early termination
            if dones[0]:
                print(f"   Episode ended at step {step} (robot fell or episode limit reached)")
                break
        
        # Episode summary
        final_pos = self.env.root_pos[0].cpu().numpy()
        
        episode_summary = {
            'total_steps': step_count,
            'total_reward': total_reward,
            'avg_reward': total_reward / step_count if step_count > 0 else 0,
            'final_height': final_pos[2],
            'final_forward': final_pos[0],
            'distance_traveled': abs(final_pos[0]),
            'stayed_upright': final_pos[2] > 0.5,
            'data': episode_data
        }
        
        print(f"\n📊 Episode Summary:")
        print(f"   Total steps: {episode_summary['total_steps']}")
        print(f"   Total reward: {episode_summary['total_reward']:.2f}")
        print(f"   Average reward: {episode_summary['avg_reward']:.4f}")
        print(f"   Distance traveled: {episode_summary['distance_traveled']:.2f}m")
        print(f"   Final height: {episode_summary['final_height']:.3f}m")
        print(f"   Stayed upright: {'✅' if episode_summary['stayed_upright'] else '❌'}")
        
        return episode_summary
    
    def run_multiple_tests(self, num_episodes: int = 5) -> List[Dict]:
        """Run multiple test episodes"""
        print(f"\n🔄 Running {num_episodes} test episodes")
        
        results = []
        for i in range(num_episodes):
            print(f"\n--- Episode {i+1}/{num_episodes} ---")
            result = self.run_test_episode(record_data=(i == 0))  # Only record data for first episode
            results.append(result)
        
        # Aggregate statistics
        total_steps = [r['total_steps'] for r in results]
        distances = [r['distance_traveled'] for r in results]
        avg_rewards = [r['avg_reward'] for r in results]
        stayed_upright = [r['stayed_upright'] for r in results]
        
        print(f"\n📈 Aggregate Results ({num_episodes} episodes):")
        print(f"   Average steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
        print(f"   Average distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}m")
        print(f"   Average reward: {np.mean(avg_rewards):.4f} ± {np.std(avg_rewards):.4f}")
        print(f"   Success rate: {100*np.mean(stayed_upright):.1f}% stayed upright")
        
        return results
    
    def plot_episode_data(self, episode_data: Dict, save_path: str = None):
        """Plot episode performance data"""
        if not episode_data['step']:
            print("No episode data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward over time
        axes[0, 0].plot(episode_data['step'], episode_data['reward'])
        axes[0, 0].set_title('Reward Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height over time
        axes[0, 1].plot(episode_data['step'], episode_data['height'])
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Fall threshold')
        axes[0, 1].set_title('Robot Height Over Time')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Height (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Forward position over time
        axes[1, 0].plot(episode_data['step'], episode_data['forward_pos'])
        axes[1, 0].set_title('Forward Position Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('X Position (m)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Action magnitude over time
        if episode_data['actions']:
            action_magnitudes = [np.linalg.norm(actions) for actions in episode_data['actions']]
            axes[1, 1].plot(episode_data['step'], action_magnitudes)
            axes[1, 1].set_title('Action Magnitude Over Time')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Action Norm')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()


def find_latest_model(pattern: str = "behavior_cloning_*.pth") -> str:
    """Find the most recently saved model"""
    model_files = glob.glob(pattern)
    if not model_files:
        raise FileNotFoundError(f"No model files found matching pattern: {pattern}")
    
    # Sort by modification time, get latest
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model


def main():
    """Main testing function"""
    
    print("🧪 Genesis Behavior Cloning Model Tester")
    print("=" * 50)
    
    # Find model to test
    try:
        latest_model = find_latest_model()
        print(f"📁 Found latest model: {latest_model}")
        
        use_latest = input(f"Test this model? (y/n): ").strip().lower()
        if use_latest != 'y':
            model_path = input("Enter path to model file: ").strip()
        else:
            model_path = latest_model
            
    except FileNotFoundError:
        print("❌ No trained models found!")
        model_path = input("Enter path to model file: ").strip()
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Test configuration
    print("\nTest configuration:")
    print("1. Quick test (1 episode, 1000 steps)")
    print("2. Standard test (3 episodes, 3000 steps)")
    print("3. Comprehensive test (5 episodes, 5000 steps)")
    
    test_choice = input("Select test type (1/2/3): ").strip()
    
    if test_choice == "1":
        num_episodes, max_steps = 1, 1000
        print("⚡ Quick test")
    elif test_choice == "3":
        num_episodes, max_steps = 5, 5000
        print("🔬 Comprehensive test")
    else:
        num_episodes, max_steps = 3, 3000
        print("🎯 Standard test")
    
    print(f"   Episodes: {num_episodes}")
    print(f"   Max steps per episode: {max_steps}")
    
    input("\nPress Enter to start testing...")
    
    try:
        # Initialize tester
        tester = BehaviorCloningTester(model_path)
        
        # Run tests
        if num_episodes == 1:
            result = tester.run_test_episode(max_steps=max_steps, record_data=True)
            
            # Plot results
            plot_choice = input("\nPlot episode data? (y/n): ").strip().lower()
            if plot_choice == 'y':
                tester.plot_episode_data(result['data'])
        else:
            results = tester.run_multiple_tests(num_episodes)
        
        print(f"\n🎉 Testing complete!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()