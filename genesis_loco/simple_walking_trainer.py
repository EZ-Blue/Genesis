#!/usr/bin/env python3
"""
Simple Walking Training Script

Fixed implementation for imitation learning with skeleton humanoid.
Addresses the zero reward and episode length issues.
"""

import torch
import numpy as np
import time
import sys
import os
from typing import Dict, Any

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
import genesis as gs

class SimpleWalkingTrainer:
    """
    Minimal working trainer for skeleton walking imitation learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        print("🚀 Initializing Simple Walking Trainer")
        
        # Initialize Genesis first to establish device context
        print("   Initializing Genesis...")
        gs.init(backend=gs.gpu)
        
        # Now get the device from Genesis (after initialization)
        self.device = gs.device
        print(f"   Using device: {self.device}")
        
        self._setup_environment()
        self._setup_data_bridge()
        
    def _setup_environment(self):
        """Setup skeleton humanoid environment"""
        print("   Setting up environment...")
        
        # Genesis already initialized in __init__
        
        # Create environment with proper configuration
        self.env = SkeletonHumanoidEnv(
            num_envs=self.config.get('num_envs', 4),
            episode_length_s=self.config.get('episode_length_s', 5.0),
            dt=self.config.get('dt', 0.02),
            show_viewer=self.config.get('show_viewer', True),
            use_trajectory_control=False,  # Use torque control for training
            use_box_feet=True  # Enable stable ground contact
        )
        
        print(f"     ✓ Environment created:")
        print(f"       - Environments: {self.env.num_envs}")
        print(f"       - Episode length: {self.config.get('episode_length_s', 5.0)}s")
        print(f"       - Action dim: {self.env.num_actions}")
        print(f"       - Observation dim: {self.env.num_observations}")
        
    def _setup_data_bridge(self):
        """Setup LocoMujoco data bridge for expert trajectories"""
        print("   Setting up data bridge...")
        
        try:
            self.data_bridge = LocoMujocoDataBridge(self.env)
            
            # Load walking trajectory
            success, _ = self.data_bridge.load_trajectory("walk")
            if not success:
                print("     ⚠️ Failed to load trajectory, using simple rewards only")
                self.use_expert_data = False
            else:
                print("     ✓ Expert walking trajectory loaded")
                self.use_expert_data = True
                
        except Exception as e:
            print(f"     ⚠️ Data bridge setup failed: {e}")
            print("     Continuing with simple reward training only")
            self.use_expert_data = False
    
    def test_environment_basic(self):
        """Test basic environment functionality"""
        print("\n🧪 Testing Environment Basics")
        print("=" * 50)
        
        # Reset environment
        print("Testing environment reset...")
        obs = self.env.reset()
        print(f"✓ Reset successful - obs: {obs}")
        
        # Test random actions
        print("Testing random actions...")
        for step in range(10):
            # Generate random actions in reasonable range
            actions = torch.randn((self.env.num_envs, self.env.num_actions), device=self.device) * 0.1
            
            obs, rewards, dones, info = self.env.step(actions)
            
            print(f"  Step {step+1:2d}: reward={rewards[0].item():.3f}, done={dones[0].item()}, "
                  f"episode_length={self.env.episode_length_buf[0].item()}")
            
            if dones[0]:
                print("    Environment reset due to termination")
                break
        
        print(f"\n📊 Test Results:")
        print(f"   Final episode lengths: {self.env.episode_length_buf}")
        print(f"   Final rewards: {rewards}")
        print(f"   Terminations: {dones.sum().item()}/{self.env.num_envs}")
        
    def simple_policy_test(self, num_steps: int = 100):
        """Test with a simple standing policy"""
        print(f"\n🤖 Testing Simple Standing Policy ({num_steps} steps)")
        print("=" * 50)
        
        obs = self.env.reset()
        
        total_reward = torch.zeros(self.env.num_envs, device=self.device)
        episode_lengths = []
        
        for step in range(num_steps):
            # Simple policy: small actions to try to stand/balance
            actions = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device)
            
            # Add small stabilizing torques (PD-like behavior)
            if hasattr(self.env, 'root_pos'):
                height_error = 1.0 - self.env.root_pos[:, 2]  # Target 1m height
                actions[:, :3] = height_error.unsqueeze(1) * 0.1  # Small corrective actions
            
            obs, rewards, dones, info = self.env.step(actions)
            total_reward += rewards
            
            # Log progress
            if step % 20 == 0:
                avg_height = self.env.root_pos[:, 2].mean().item()
                avg_reward = rewards.mean().item()
                avg_length = self.env.episode_length_buf.float().mean().item()
                print(f"  Step {step:3d}: Height={avg_height:.3f}m, Reward={avg_reward:.3f}, "
                      f"Episode Length={avg_length:.1f}")
            
            # Track completed episodes
            if torch.any(dones):
                completed_envs = dones.nonzero(as_tuple=False).flatten()
                for env_id in completed_envs:
                    length = self.env.episode_length_buf[env_id].item()
                    episode_lengths.append(length)
                    print(f"    Env {env_id}: Episode completed with length {length}")
        
        print(f"\n📊 Simple Policy Results:")
        print(f"   Average total reward: {total_reward.mean().item():.3f}")
        print(f"   Average episode length: {np.mean(episode_lengths) if episode_lengths else 'No completions'}")
        print(f"   Completed episodes: {len(episode_lengths)}")
        print(f"   Final average height: {self.env.root_pos[:, 2].mean().item():.3f}m")
        
    def run_diagnostics(self):
        """Run comprehensive diagnostics"""
        print("\n🔍 Running Training Diagnostics")
        print("=" * 60)
        
        # Test 1: Basic environment functionality
        self.test_environment_basic()
        
        # Test 2: Simple policy
        self.simple_policy_test(100)
        
        # Test 3: Action space validation
        print(f"\n🎯 Action Space Validation:")
        print(f"   Environment action dim: {self.env.num_actions}")
        print(f"   Skeleton actions: {self.env.num_skeleton_actions}")
        print(f"   Action mapping: {len(self.env.action_to_joint_idx) if hasattr(self.env, 'action_to_joint_idx') else 'Not available'}")
        
        # Test 4: Reward system validation
        print(f"\n🎁 Reward System Validation:")
        if hasattr(self.env, 'reward_functions'):
            print(f"   Registered reward functions: {list(self.env.reward_functions.keys())}")
            print(f"   Reward config: {self.env.reward_cfg}")
        else:
            print("   ⚠️ No reward functions registered!")
            
        print("\n✅ Diagnostics complete!")

def main():
    """Main function"""
    print("🚶‍♂️ Simple Walking Trainer Diagnostics")
    
    # Simple configuration for testing
    config = {
        'num_envs': 4,
        'episode_length_s': 5.0,
        'dt': 0.02,
        'show_viewer': True,
    }
    
    try:
        trainer = SimpleWalkingTrainer(config)
        trainer.run_diagnostics()
        
        print("\n🎉 If you see rewards and episode lengths > 0, the training setup is working!")
        print("Next step: Implement proper PPO + AMP training loop")
        
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()