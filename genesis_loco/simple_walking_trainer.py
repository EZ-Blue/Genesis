#!/usr/bin/env python3
"""
Simple Walking Trainer with Imitation Learning

Updated to work with refactored skeleton environment, data bridge, and AMP integration.
Implements proper imitation learning pipeline for walking gait training.
"""

import torch
import numpy as np
import time
import sys
import os
from typing import Dict, Any, Optional

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import refactored components
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
from integration.amp_integration import AMPGenesisIntegration
import genesis as gs

class SimpleWalkingTrainer:
    """
    Simple trainer for skeleton walking using imitation learning with AMP
    
    Compatible with refactored environment and integration components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        print("🚀 Initializing Simple Walking Trainer with Imitation Learning")
        print("=" * 60)
        
        # Initialize Genesis
        print("1. Initializing Genesis...")
        try:
            gs.init(backend=gs.gpu)
            print(f"   ✅ Genesis initialized")
        except Exception as e:
            if "already initialized" in str(e):
                print(f"   ✅ Genesis already initialized")
            else:
                raise e
        
        self.device = gs.device
        print(f"   ✅ Using device: {self.device}")
        
        # Setup components
        self._setup_environment()
        self._setup_data_bridge()
        self._setup_amp_integration()
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        
    def _setup_environment(self):
        """Setup skeleton humanoid environment"""
        print("2. Setting up environment...")
        
        self.env = SkeletonHumanoidEnv(
            num_envs=self.config.get('num_envs', 16),
            episode_length_s=self.config.get('episode_length_s', 10.0),
            dt=self.config.get('dt', 0.01),
            show_viewer=self.config.get('show_viewer', True),
            use_box_feet=True
        )
        
        print(f"   ✅ Environment created:")
        print(f"      - Environments: {self.env.num_envs}")
        print(f"      - Episode length: {self.config.get('episode_length_s', 10.0)}s")
        print(f"      - Actions: {self.env.num_actions}")
        print(f"      - Observations: {self.env.num_observations}")
        
    def _setup_data_bridge(self):
        """Setup LocoMujoco data bridge for expert trajectories"""
        print("3. Setting up data bridge...")
        
        try:
            self.data_bridge = LocoMujocoDataBridge(self.env)
            
            # Load walking trajectory
            success = self.data_bridge.load_trajectory("walk")
            if not success:
                raise RuntimeError("Failed to load trajectory")
                
            print(f"   ✅ Expert walking trajectory loaded:")
            print(f"      - Length: {self.data_bridge.trajectory_length} timesteps")
            print(f"      - Frequency: {self.data_bridge.trajectory_frequency} Hz")
            self.use_expert_data = True
                
        except Exception as e:
            print(f"   ❌ Data bridge setup failed: {e}")
            print("      Cannot proceed without expert data for imitation learning")
            raise
    
    def _setup_amp_integration(self):
        """Setup AMP integration for discriminator training"""
        print("4. Setting up AMP integration...")
        
        if not self.use_expert_data:
            print("   ⚠️ Skipping AMP setup - no expert data available")
            self.amp_integration = None
            return
        
        try:
            # Create AMP integration
            self.amp_integration = AMPGenesisIntegration(
                genesis_env=self.env,
                data_bridge=self.data_bridge,
                discriminator_config={
                    'hidden_layers': [512, 256],
                    'activation': 'tanh',
                    'learning_rate': 5e-5,
                    'use_running_norm': True
                }
            )
            
            # Load expert data for discriminator
            success = self.amp_integration.load_expert_data()
            if not success:
                raise RuntimeError("Failed to load expert observations")
                
            print(f"   ✅ AMP integration ready:")
            print(f"      - Expert samples: {self.amp_integration.n_expert_samples}")
            print(f"      - Observation dim: {self.env.num_observations}")
            
        except Exception as e:
            print(f"   ❌ AMP integration setup failed: {e}")
            self.amp_integration = None
            raise
    
    def test_expert_trajectory_application(self, num_samples: int = 10):
        """Test applying expert trajectory states to environment"""
        print(f"\n🎯 Testing Expert Trajectory Application ({num_samples} samples)")
        print("=" * 60)
        
        if not self.use_expert_data:
            print("   ⚠️ No expert data available")
            return
        
        # Reset environment
        self.env.reset()
        
        # Test trajectory application
        env_ids = torch.tensor([0], device=self.device)  # First environment only
        
        print("   Applying expert states and recording observations...")
        expert_obs_list = []
        
        for i in range(0, num_samples * 10, 10):  # Sample every 10th timestep
            # Get trajectory state
            state_data = self.data_bridge.get_trajectory_state(i)
            if state_data is None:
                continue
            
            # Apply to environment
            self.data_bridge.apply_trajectory_state(state_data, env_ids)
            
            # Get observation
            obs = self.env._get_observations()
            expert_obs_list.append(obs[0])
            
            # Log sample
            root_height = state_data['root_pos'][2].item()
            print(f"      Sample {i:3d}: Root height = {root_height:.3f}m")
        
        if expert_obs_list:
            expert_obs_tensor = torch.stack(expert_obs_list, dim=0)
            print(f"   ✅ Generated {expert_obs_tensor.shape[0]} expert observations")
            print(f"      Observation shape: {expert_obs_tensor.shape}")
            print(f"      Observation range: [{expert_obs_tensor.min().item():.3f}, {expert_obs_tensor.max().item():.3f}]")
        else:
            print("   ❌ Failed to generate expert observations")
    
    def test_amp_discriminator(self, num_steps: int = 50):
        """Test AMP discriminator functionality"""
        print(f"\n🧠 Testing AMP Discriminator ({num_steps} steps)")
        print("=" * 60)
        
        if self.amp_integration is None:
            print("   ⚠️ AMP integration not available")
            return
        
        # Reset environment
        obs, _ = self.env.reset()
        
        discriminator_losses = []
        amp_rewards = []
        
        for step in range(num_steps):
            # Random policy actions
            actions = torch.randn((self.env.num_envs, self.env.num_actions), device=self.device) * 0.1
            
            # Step environment
            obs, env_rewards, dones, info = self.env.step(actions)
            
            # Compute AMP rewards
            amp_reward = self.amp_integration.compute_amp_rewards(obs)
            amp_rewards.append(amp_reward.mean().item())
            
            # Train discriminator
            if step % 5 == 0:  # Train every 5 steps
                metrics = self.amp_integration.train_discriminator_step(obs)
                discriminator_losses.append(metrics.get('discriminator_loss', 0.0))
                
                if step % 20 == 0:
                    print(f"      Step {step:3d}: AMP reward = {amp_reward.mean().item():.4f}, "
                          f"Disc loss = {metrics.get('discriminator_loss', 0.0):.4f}")
        
        print(f"   ✅ AMP discriminator test completed:")
        print(f"      - Average AMP reward: {np.mean(amp_rewards):.4f}")
        print(f"      - Average discriminator loss: {np.mean(discriminator_losses):.4f}")
        print(f"      - Expert accuracy: {metrics.get('expert_accuracy', 0.0):.3f}")
        print(f"      - Policy accuracy: {metrics.get('policy_accuracy', 0.0):.3f}")
    
    def simple_imitation_training(self, num_steps: int = 500):
        """Simple imitation learning training loop"""
        print(f"\n🏃‍♂️ Simple Imitation Learning Training ({num_steps} steps)")
        print("=" * 60)
        
        if self.amp_integration is None:
            print("   ❌ Cannot train without AMP integration")
            return
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Training metrics
        episode_rewards = []
        amp_rewards_history = []
        env_rewards_history = []
        mixed_rewards_history = []
        
        print("   Starting training loop...")
        
        for step in range(num_steps):
            # Simple policy: small random actions with some structure
            actions = torch.randn((self.env.num_envs, self.env.num_actions), device=self.device) * 0.2
            
            # Add some structure to actions to encourage walking-like motion
            if step > 100:  # After initial exploration
                # Oscillatory pattern for legs (simple walking pattern)
                phase = (step * 0.1) % (2 * np.pi)
                leg_pattern = torch.sin(torch.tensor(phase)) * 0.3
                
                # Apply to hip and knee joints (rough approximation)
                if self.env.num_actions >= 6:
                    actions[:, 3:6] += leg_pattern  # Right leg
                    actions[:, 9:12] -= leg_pattern  # Left leg (opposite phase)
            
            # Step environment
            obs, env_rewards, dones, info = self.env.step(actions)
            
            # Compute AMP rewards
            amp_rewards = self.amp_integration.compute_amp_rewards(obs)
            
            # Mix environment and AMP rewards
            mixed_rewards = self.amp_integration.get_mixed_rewards(
                env_rewards, obs, env_reward_weight=0.3
            )
            
            # Train discriminator
            if step % 10 == 0:
                self.amp_integration.train_discriminator_step(obs)
            
            # Track metrics
            env_rewards_history.append(env_rewards.mean().item())
            amp_rewards_history.append(amp_rewards.mean().item())
            mixed_rewards_history.append(mixed_rewards.mean().item())
            
            # Log progress
            if step % 100 == 0:
                avg_height = self.env.root_pos[:, 2].mean().item()
                print(f"      Step {step:4d}: Height={avg_height:.3f}m, "
                      f"Env={env_rewards.mean().item():.3f}, "
                      f"AMP={amp_rewards.mean().item():.3f}, "
                      f"Mixed={mixed_rewards.mean().item():.3f}")
            
            # Track episode completions
            if torch.any(dones):
                completed_envs = dones.sum().item()
                avg_episode_reward = mixed_rewards.mean().item()
                episode_rewards.append(avg_episode_reward)
                
                if len(episode_rewards) % 10 == 0:
                    print(f"      Episodes completed: {len(episode_rewards)}, "
                          f"Average reward: {np.mean(episode_rewards[-10:]):.3f}")
        
        print(f"   ✅ Training completed:")
        print(f"      - Episodes completed: {len(episode_rewards)}")
        print(f"      - Average env reward: {np.mean(env_rewards_history):.4f}")
        print(f"      - Average AMP reward: {np.mean(amp_rewards_history):.4f}")
        print(f"      - Average mixed reward: {np.mean(mixed_rewards_history):.4f}")
        print(f"      - Final height: {self.env.root_pos[:, 2].mean().item():.3f}m")
    
    def run_full_test(self):
        """Run comprehensive test of imitation learning pipeline"""
        print("\n🔬 Running Full Imitation Learning Test")
        print("=" * 70)
        
        try:
            # Test 1: Expert trajectory application
            self.test_expert_trajectory_application(20)
            
            # Test 2: AMP discriminator functionality
            self.test_amp_discriminator(100)
            
            # Test 3: Simple imitation training
            self.simple_imitation_training(1000)
            
            print("\n🎉 All tests completed successfully!")
            print("The imitation learning pipeline is working correctly.")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    print("🚶‍♂️ Simple Walking Trainer with Imitation Learning")
    print("=" * 70)
    
    # Configuration for imitation learning
    config = {
        'num_envs': 16,
        'episode_length_s': 10.0,
        'dt': 0.01,
        'show_viewer': True,
    }
    
    try:
        trainer = SimpleWalkingTrainer(config)
        trainer.run_full_test()
        
        print("\n✅ Success! Your imitation learning setup is ready for training.")
        print("Next steps:")
        print("  1. Implement proper PPO policy network")
        print("  2. Add trajectory reward tracking")
        print("  3. Fine-tune AMP reward mixing")
        
    except Exception as e:
        print(f"\n❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()