"""
AMP Integration with Genesis Environment

Connects the AMP discriminator with Genesis physics and LocoMujoco data bridge
for complete imitation learning pipeline.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import random
import sys
import os

# Fix import paths for direct execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from amp_discriminator import AMPDiscriminator, AMPTrainer
from data_bridge import LocoMujocoDataBridge


class AMPGenesisIntegration:
    """
    Complete AMP integration for Genesis environments
    
    Combines:
    - LocoMujoco trajectory data (via DataBridge)
    - AMP discriminator (PyTorch)
    - Genesis physics simulation
    - Expert data sampling for training
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
        
        # Expert trajectory data (loaded from data bridge)
        self.expert_trajectory = None
        self.expert_observations = None
        
        # Sampling configuration
        self.expert_batch_size = 256
        self.current_expert_idx = 0
        
        print(f"✓ AMP Integration initialized")
        print(f"  - Observation dimension: {obs_dim}")
        print(f"  - Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")
    
    def load_expert_data(self) -> bool:
        """
        Load and prepare expert trajectory data for discriminator training
        
        Returns:
            Success status
        """
        print("Loading expert trajectory data...")
        
        # Get converted trajectory data from data bridge
        success, trajectory_data = self.data_bridge.convert_to_genesis_format()
        if not success:
            print("✗ Failed to get trajectory data from data bridge")
            return False
        
        self.expert_trajectory = trajectory_data
        
        # Generate expert observations by applying trajectory to Genesis
        expert_obs_list = []
        n_timesteps = min(1000, trajectory_data['info']['timesteps'])  # Limit for memory
        
        print(f"  - Generating expert observations from {n_timesteps} timesteps...")
        
        # Create temporary environment state for observation generation
        for t in range(0, n_timesteps, 10):  # Sample every 10th timestep
            # Apply trajectory state to Genesis
            dof_pos = trajectory_data['dof_pos'][t:t+1]  # [1, num_dofs]
            root_pos = trajectory_data['root_pos'][t:t+1]  # [1, 3]  
            root_quat = trajectory_data['root_quat'][t:t+1]  # [1, 4]
            
            # Set Genesis state
            env_ids = torch.tensor([0], device=self.device)
            self.genesis_env.robot.set_dofs_position(dof_pos, envs_idx=env_ids, zero_velocity=True)
            self.genesis_env.robot.set_pos(root_pos, envs_idx=env_ids, zero_velocity=True)
            self.genesis_env.robot.set_quat(root_quat, envs_idx=env_ids, zero_velocity=True)
            
            # Update environment state and get observation
            self.genesis_env._update_robot_state()
            obs = self.genesis_env._get_observations()  # [1, obs_dim]
            expert_obs_list.append(obs[0])  # Just the first (and only) environment
        
        # Combine all expert observations
        self.expert_observations = torch.stack(expert_obs_list, dim=0)  # [n_samples, obs_dim]
        
        print(f"  ✓ Generated {self.expert_observations.shape[0]} expert observations")
        print(f"  ✓ Expert observation shape: {self.expert_observations.shape}")
        
        return True
    
    def sample_expert_batch(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Sample a batch of expert observations for discriminator training
        
        Args:
            batch_size: Number of expert observations to sample
            
        Returns:
            Expert observation batch [batch_size, obs_dim]
        """
        if self.expert_observations is None:
            raise RuntimeError("Expert data not loaded. Call load_expert_data() first.")
        
        if batch_size is None:
            batch_size = self.expert_batch_size
        
        n_expert = self.expert_observations.shape[0]
        
        # Random sampling with replacement
        indices = torch.randint(0, n_expert, (batch_size,), device=self.device)
        expert_batch = self.expert_observations[indices]
        
        return expert_batch
    
    def compute_amp_rewards(self, policy_observations: torch.Tensor) -> torch.Tensor:
        """
        Compute AMP rewards for policy observations
        
        Args:
            policy_observations: Policy observations [num_envs, obs_dim]
            
        Returns:
            AMP rewards [num_envs]
        """
        return self.trainer.compute_amp_rewards(policy_observations)
    
    def train_discriminator_step(self, policy_observations: torch.Tensor) -> Dict[str, float]:
        """
        Single discriminator training step using policy and expert data
        
        Args:
            policy_observations: Current policy observations [num_envs, obs_dim]
            
        Returns:
            Training metrics
        """
        # Sample expert batch matching policy batch size
        batch_size = policy_observations.shape[0]
        expert_batch = self.sample_expert_batch(batch_size)
        
        # Train discriminator
        metrics = self.trainer.train_discriminator(expert_batch, policy_observations)
        
        return metrics
    
    def get_mixed_rewards(self, 
                         env_rewards: torch.Tensor,
                         policy_observations: torch.Tensor,
                         env_reward_weight: float = 0.5) -> torch.Tensor:
        """
        Compute mixed rewards combining environment and AMP discriminator rewards
        
        Args:
            env_rewards: Environment task rewards [num_envs]
            policy_observations: Policy observations [num_envs, obs_dim]  
            env_reward_weight: Weight for environment reward (0.0 = pure AMP, 1.0 = pure env)
            
        Returns:
            Mixed rewards [num_envs]
        """
        amp_rewards = self.compute_amp_rewards(policy_observations)
        
        mixed_rewards = (env_reward_weight * env_rewards + 
                        (1 - env_reward_weight) * amp_rewards)
        
        return mixed_rewards
    
    def save_discriminator(self, path: str):
        """Save discriminator state"""
        self.trainer.save_discriminator(path)
    
    def load_discriminator(self, path: str):
        """Load discriminator state"""
        self.trainer.load_discriminator(path)


def test_amp_genesis_integration():
    """
    Test complete AMP integration with Genesis environment
    
    This test verifies:
    1. Integration setup with data bridge
    2. Expert data loading and processing
    3. AMP reward computation with real Genesis observations
    4. Discriminator training with Genesis data
    """
    print("=" * 60)
    print("TESTING AMP-GENESIS INTEGRATION")
    print("=" * 60)
    
    try:
        # Setup Genesis environment and data bridge (reuse from data_bridge.py)
        import genesis as gs
        gs.init(backend=gs.gpu)
        
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        # Create Genesis environment
        print("1. Creating Genesis environment...")
        skeleton_env = SkeletonHumanoidEnv(
            num_envs=4,  # Small batch for testing
            episode_length_s=5.0,
            dt=0.02,
            show_viewer=False
        )
        print(f"   ✓ Environment: {skeleton_env.num_envs} envs, {skeleton_env.num_observations} obs dim")
        
        # Create data bridge and load trajectory
        print("2. Setting up data bridge...")
        bridge = LocoMujocoDataBridge(skeleton_env)
        
        success, _ = bridge.load_trajectory("walk")
        if not success:
            print("   ✗ Failed to load trajectory")
            return False
            
        success, _ = bridge.build_joint_mapping()
        if not success:
            print("   ✗ Failed to build joint mapping") 
            return False
            
        print("   ✓ Data bridge ready")
        
        # Create AMP integration
        print("3. Creating AMP integration...")
        amp_integration = AMPGenesisIntegration(
            genesis_env=skeleton_env,
            data_bridge=bridge,
            discriminator_config={
                'hidden_layers': [256, 128],  # Smaller for testing
                'activation': 'tanh',  # Add missing activation
                'learning_rate': 1e-4,
                'use_running_norm': True
            }
        )
        print("   ✓ AMP integration created")
        
        # Load expert data
        print("4. Loading expert trajectory data...")
        success = amp_integration.load_expert_data()
        if not success:
            print("   ✗ Failed to load expert data")
            return False
        print("   ✓ Expert data loaded")
        
        # Test policy observations and rewards
        print("5. Testing AMP rewards with Genesis observations...")
        
        # Reset environment to get real observations
        obs, _ = skeleton_env.reset()
        print(f"   - Policy observations shape: {obs.shape}")
        
        # Compute AMP rewards
        amp_rewards = amp_integration.compute_amp_rewards(obs)
        print(f"   - AMP rewards shape: {amp_rewards.shape}")
        print(f"   - AMP rewards range: [{amp_rewards.min().item():.3f}, {amp_rewards.max().item():.3f}]")
        print(f"   - AMP rewards mean: {amp_rewards.mean().item():.3f}")
        
        # Test mixed rewards
        env_rewards = torch.ones_like(amp_rewards) * 0.5  # Mock environment rewards
        mixed_rewards = amp_integration.get_mixed_rewards(env_rewards, obs, env_reward_weight=0.7)
        print(f"   - Mixed rewards mean: {mixed_rewards.mean().item():.3f}")
        
        # Test discriminator training
        print("6. Testing discriminator training...")
        initial_metrics = amp_integration.train_discriminator_step(obs)
        print(f"   - Initial discriminator loss: {initial_metrics['discriminator_loss']:.4f}")
        print(f"   - Expert accuracy: {initial_metrics['expert_accuracy']:.3f}")
        print(f"   - Policy accuracy: {initial_metrics['policy_accuracy']:.3f}")
        
        # Train for a few more steps
        for step in range(5):
            metrics = amp_integration.train_discriminator_step(obs)
        
        final_metrics = metrics
        print(f"   - Final discriminator loss: {final_metrics['discriminator_loss']:.4f}")
        print(f"   - Final expert accuracy: {final_metrics['expert_accuracy']:.3f}")
        print(f"   - Final policy accuracy: {final_metrics['policy_accuracy']:.3f}")
        
        print("\n" + "=" * 60)
        print("✅ AMP-GENESIS INTEGRATION TEST SUCCESS!")
        print("✅ Complete pipeline working end-to-end")
        print("✅ Ready for full imitation learning training")
        print("=" * 60)
        
        return True, amp_integration
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    # Run integration test
    success, integration = test_amp_genesis_integration()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Integrate with PPO training loop")
        print("2. Implement full episode collection with AMP rewards")
        print("3. Add trajectory following tasks (optional)")
        print("4. Scale up training with larger batch sizes")
        print("5. Test on different LocoMujoco datasets (run, jump, etc.)")
    else:
        print("\n🔧 Fix Integration Issues:")
        print("1. Ensure all previous steps completed successfully")
        print("2. Check Genesis environment setup")
        print("3. Verify trajectory data format compatibility")
        print("4. Debug discriminator network initialization")