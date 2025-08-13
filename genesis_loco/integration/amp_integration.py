"""
Refactored AMP Integration - Simple and Efficient

Clean, minimal AMP implementation compatible with refactored skeleton environment
and data bridge. Removes complexity while maintaining core functionality.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import random

from amp_discriminator import AMPDiscriminator, AMPTrainer
from data_bridge import LocoMujocoDataBridge


class AMPGenesisIntegration:
    """
    Simple AMP integration for Genesis environments
    
    Compatible with skeleton_humanoid_refactored.py and data_bridge_refactored.py
    """
    
    def __init__(self,
                 genesis_env,
                 data_bridge: LocoMujocoDataBridge,
                 discriminator_config: Dict = None):
        
        self.genesis_env = genesis_env
        self.data_bridge = data_bridge
        self.device = genesis_env.device
        
        # Default discriminator configuration
        if discriminator_config is None:
            discriminator_config = {
                'hidden_layers': [512, 256],
                'activation': 'tanh',
                'learning_rate': 5e-5,
                'use_running_norm': True
            }
        
        # Initialize discriminator
        obs_dim = self.genesis_env.num_observations
        print(f"   Initializing discriminator with {obs_dim} observation dimensions")
        self.discriminator = AMPDiscriminator(
            input_dim=obs_dim,
            hidden_layers=discriminator_config['hidden_layers'],
            activation=discriminator_config['activation'],
            use_running_norm=discriminator_config['use_running_norm']
        )
        
        # Initialize trainer
        self.trainer = AMPTrainer(
            discriminator=self.discriminator,
            learning_rate=discriminator_config['learning_rate'],
            device=self.device
        )
        
        # Expert data
        self.expert_observations = None
        self.expert_batch_size = 256
        
        print(f"✅ AMP Integration initialized:")
        print(f"   - Observation dimension: {obs_dim}")
        print(f"   - Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")
    
    def load_expert_data(self) -> bool:
        """
        Load expert trajectory data and generate observations
        
        Returns:
            bool: Success status
        """
        print("Loading expert trajectory data...")
        
        if self.data_bridge.loco_trajectory is None:
            print("❌ No trajectory loaded in data bridge")
            return False
        
        # Generate expert observations efficiently
        expert_obs_list = []
        trajectory_length = self.data_bridge.trajectory_length
        
        # Sample every 10th timestep to reduce memory usage
        sample_interval = 10
        n_samples = min(1000, trajectory_length // sample_interval)
        
        print(f"   Generating expert observations from {n_samples} samples...")
        
        # Use first environment for expert observation generation
        env_ids = torch.tensor([0], device=self.device)
        
        for i in range(0, n_samples * sample_interval, sample_interval):
            # Get trajectory state at timestep
            state_data = self.data_bridge.get_trajectory_state(i)
            
            if state_data is None:
                continue
            
            # Apply state to Genesis environment
            self.data_bridge.apply_trajectory_state(state_data, env_ids)
            
            # Get observation from environment
            obs = self.genesis_env._get_observations()
            expert_obs_list.append(obs[0])  # First environment only
        
        if not expert_obs_list:
            print("❌ Failed to generate expert observations")
            return False
        
        # Stack all expert observations
        self.expert_observations = torch.stack(expert_obs_list, dim=0)
        
        print(f"   ✅ Generated {self.expert_observations.shape[0]} expert observations")
        print(f"   ✅ Expert observation shape: {self.expert_observations.shape}")
        
        return True
    
    def sample_expert_batch(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Sample random batch of expert observations
        
        Args:
            batch_size: Number of observations to sample
            
        Returns:
            torch.Tensor: Expert observation batch [batch_size, obs_dim]
        """
        if self.expert_observations is None:
            raise RuntimeError("Expert data not loaded. Call load_expert_data() first.")
        
        if batch_size is None:
            batch_size = self.expert_batch_size
        
        n_expert = self.expert_observations.shape[0]
        
        # Random sampling with replacement
        indices = torch.randint(0, n_expert, (batch_size,), device=self.device)
        return self.expert_observations[indices]
    
    def compute_amp_rewards(self, policy_observations: torch.Tensor) -> torch.Tensor:
        """
        Compute AMP rewards for policy observations
        
        Args:
            policy_observations: Policy observations [num_envs, obs_dim]
            
        Returns:
            torch.Tensor: AMP rewards [num_envs]
        """
        return self.trainer.compute_amp_rewards(policy_observations)
    
    def train_discriminator_step(self, policy_observations: torch.Tensor) -> Dict[str, float]:
        """
        Single discriminator training step
        
        Args:
            policy_observations: Current policy observations [num_envs, obs_dim]
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Sample expert batch matching policy batch size
        batch_size = policy_observations.shape[0]
        expert_batch = self.sample_expert_batch(batch_size)
        
        # Train discriminator
        return self.trainer.train_discriminator(expert_batch, policy_observations)
    
    def get_mixed_rewards(self, 
                         env_rewards: torch.Tensor,
                         policy_observations: torch.Tensor,
                         env_reward_weight: float = 0.5) -> torch.Tensor:
        """
        Compute mixed rewards combining environment and AMP rewards
        
        Args:
            env_rewards: Environment task rewards [num_envs]
            policy_observations: Policy observations [num_envs, obs_dim]  
            env_reward_weight: Weight for environment reward (0.0-1.0)
            
        Returns:
            torch.Tensor: Mixed rewards [num_envs]
        """
        amp_rewards = self.compute_amp_rewards(policy_observations)
        
        return (env_reward_weight * env_rewards + 
                (1 - env_reward_weight) * amp_rewards)
    
    def save_discriminator(self, path: str):
        """Save discriminator state"""
        self.trainer.save_discriminator(path)
    
    def load_discriminator(self, path: str):
        """Load discriminator state"""
        self.trainer.load_discriminator(path)
    
    @property
    def expert_data_loaded(self) -> bool:
        """Check if expert data is loaded"""
        return self.expert_observations is not None
    
    @property
    def n_expert_samples(self) -> int:
        """Number of expert observation samples"""
        if self.expert_observations is None:
            return 0
        return self.expert_observations.shape[0]


def test_amp_integration():
    """
    Test AMP integration with refactored components
    """
    print("🧪 Testing Refactored AMP Integration")
    print("=" * 50)
    
    try:
        import genesis as gs
        gs.init(backend=gs.gpu)
        
        # Import refactored components
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        # Create environment
        print("1. Creating Genesis environment...")
        env = SkeletonHumanoidEnv(
            num_envs=4,
            episode_length_s=5.0,
            dt=0.02,
            use_box_feet=True,
            show_viewer=False
        )
        print(f"   ✅ Environment: {env.num_envs} envs, {env.num_observations} obs")
        
        # Create data bridge
        print("2. Creating data bridge...")
        data_bridge = LocoMujocoDataBridge(env)
        success = data_bridge.load_trajectory("walk")
        if not success:
            print("   ❌ Failed to load trajectory")
            return
        print(f"   ✅ Trajectory loaded: {data_bridge.trajectory_length} timesteps")
        
        # Create AMP integration
        print("3. Creating AMP integration...")
        amp_integration = AMPGenesisIntegration(env, data_bridge)
        
        # Load expert data
        print("4. Loading expert data...")
        success = amp_integration.load_expert_data()
        if not success:
            print("   ❌ Failed to load expert data")
            return
        print(f"   ✅ Expert data: {amp_integration.n_expert_samples} samples")
        
        # Test discriminator functionality
        print("5. Testing discriminator...")
        
        # Get policy observations
        env.reset()
        obs, _, _, _ = env.step(torch.zeros((env.num_envs, env.num_actions), device=env.device))
        print(f"   Environment observation shape: {obs.shape}")
        print(f"   Environment num_observations: {env.num_observations}")
        
        # Compute AMP rewards
        amp_rewards = amp_integration.compute_amp_rewards(obs)
        print(f"   ✅ AMP rewards computed: mean={amp_rewards.mean().item():.4f}")
        
        # Test discriminator training
        metrics = amp_integration.train_discriminator_step(obs)
        print(f"   ✅ Discriminator trained: loss={metrics.get('discriminator_loss', 0):.4f}")
        
        # Test mixed rewards
        env_rewards = torch.ones(env.num_envs, device=env.device) * 0.1
        mixed_rewards = amp_integration.get_mixed_rewards(env_rewards, obs, env_reward_weight=0.5)
        print(f"   ✅ Mixed rewards: mean={mixed_rewards.mean().item():.4f}")
        
        print("\n🎉 All tests passed! AMP integration working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_amp_integration()