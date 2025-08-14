"""
Simple Imitation Learning Trainer

Complete training loop combining:
- Genesis skeleton environment
- LocoMujoco expert data
- PPO policy
- AMP discriminator
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import sys
import os

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_policy import SkeletonPolicyNetwork, PPOTrainer
from amp_integration import AMPGenesisIntegration
from data_bridge import LocoMujocoDataBridge


class TrajectoryBuffer:
    """
    Simple buffer for collecting episode trajectories
    """
    
    def __init__(self, num_envs: int, max_episode_length: int, obs_dim: int, action_dim: int, device: torch.device):
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Buffers
        self.observations = torch.zeros((max_episode_length, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((max_episode_length, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((max_episode_length, num_envs), device=device)
        self.values = torch.zeros((max_episode_length, num_envs), device=device)
        self.log_probs = torch.zeros((max_episode_length, num_envs), device=device)
        self.dones = torch.zeros((max_episode_length, num_envs), dtype=torch.bool, device=device)
        
        self.step = 0
        
    def add(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
            values: torch.Tensor, log_probs: torch.Tensor, dones: torch.Tensor):
        """Add step data to buffer"""
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.dones[self.step] = dones
        
        self.step += 1
    
    def compute_gae(self, next_values: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        gae = 0
        for t in reversed(range(self.step)):
            if t == self.step - 1:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = next_values
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
        
        return advantages, returns
    
    def get_batch(self, advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get flattened batch for training"""
        # Flatten time and environment dimensions
        batch = {
            'observations': self.observations[:self.step].reshape(-1, self.obs_dim),
            'actions': self.actions[:self.step].reshape(-1, self.action_dim),
            'old_log_probs': self.log_probs[:self.step].reshape(-1),
            'advantages': advantages[:self.step].reshape(-1),
            'returns': returns[:self.step].reshape(-1)
        }
        
        return batch
    
    def clear(self):
        """Clear buffer for next episode collection"""
        self.step = 0


class SimpleImitationTrainer:
    """
    Complete imitation learning trainer for Genesis skeleton
    
    Combines all components:
    - Genesis physics simulation
    - LocoMujoco expert trajectories  
    - PPO policy learning
    - AMP discriminator rewards
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🚀 Initializing Simple Imitation Learning Trainer")
        print(f"   Device: {self.device}")
        
        # Initialize components
        self._setup_environment()
        self._setup_amp_integration()
        self._setup_policy()
        self._setup_training_buffer()
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.discriminator_accuracies = []
        
        print("✅ Trainer initialization complete!")
    
    def _setup_environment(self):
        """Setup Genesis skeleton environment"""
        print("   Setting up Genesis environment...")
        
        import genesis as gs
        
        # Initialize Genesis if not already done
        try:
            gs.init(backend=gs.gpu)
        except Exception as e:
            if "already initialized" not in str(e):
                raise e
        
        # Import skeleton environment
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        self.env = SkeletonHumanoidEnv(
            num_envs=self.config['num_envs'],
            episode_length_s=self.config['episode_length_s'],
            dt=self.config['dt'],
            show_viewer=self.config.get('show_viewer', False)
        )
        
        self.obs_dim = self.env.num_observations
        self.action_dim = self.env.num_actions
        
        print(f"     ✓ {self.config['num_envs']} environments")
        print(f"     ✓ Observation dim: {self.obs_dim}")
        print(f"     ✓ Action dim: {self.action_dim}")
    
    def _setup_amp_integration(self):
        """Setup AMP discriminator and expert data"""
        print("   Setting up AMP integration...")
        
        # Create data bridge
        self.data_bridge = LocoMujocoDataBridge(self.env)
        
        # Load expert trajectory
        success = self.data_bridge.load_trajectory("walk")
        if not success:
            raise RuntimeError("Failed to load expert trajectory")
        
        # Create AMP integration
        self.amp_integration = AMPGenesisIntegration(
            genesis_env=self.env,
            data_bridge=self.data_bridge,
            discriminator_config=self.config['discriminator']
        )
        
        # Load expert data
        success = self.amp_integration.load_expert_data()
        if not success:
            raise RuntimeError("Failed to load expert data")
        
        print("     ✓ Expert trajectory loaded")
        print("     ✓ AMP discriminator ready")
    
    def _setup_policy(self):
        """Setup PPO policy network"""
        print("   Setting up policy network...")
        
        self.policy = SkeletonPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=self.config['policy']['hidden_layers'],
            activation=self.config['policy']['activation'],
            use_obs_norm=True
        )
        
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            learning_rate=self.config['policy']['learning_rate'],
            clip_epsilon=self.config['policy']['clip_epsilon'],
            device=self.device
        )
        
        param_count = sum(p.numel() for p in self.policy.parameters())
        print(f"     ✓ Policy network: {param_count} parameters")
    
    def _setup_training_buffer(self):
        """Setup trajectory collection buffer"""
        max_steps = int(self.config['episode_length_s'] / self.config['dt'])
        
        self.buffer = TrajectoryBuffer(
            num_envs=self.config['num_envs'],
            max_episode_length=max_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        print(f"     ✓ Buffer: {max_steps} max steps per episode")
    
    def collect_trajectories(self) -> Dict[str, float]:
        """
        Collect trajectories using current policy
        
        Returns:
            Collection metrics
        """
        self.policy.eval()
        self.buffer.clear()
        
        # Reset environment
        obs, _ = self.env.reset()
        episode_rewards = torch.zeros(self.config['num_envs'], device=self.device)
        episode_lengths = torch.zeros(self.config['num_envs'], device=self.device)
        
        # Collect steps
        for step in range(self.config['max_episode_steps']):
            # Sample actions from policy
            with torch.no_grad():
                actions, log_probs, values = self.policy.sample_actions(obs)
            
            # Environment step
            next_obs, env_rewards, dones, _ = self.env.step(actions)
            
            # Compute AMP rewards
            amp_rewards = self.amp_integration.compute_amp_rewards(obs)
            
            # Mix environment and AMP rewards
            mixed_rewards = (self.config['env_reward_weight'] * env_rewards + 
                           (1 - self.config['env_reward_weight']) * amp_rewards)
            
            # Store in buffer
            self.buffer.add(obs, actions, mixed_rewards, values, log_probs, dones)
            
            # Update metrics
            episode_rewards += mixed_rewards
            episode_lengths += 1
            
            # Handle episode termination
            if dones.any():
                finished_episodes = dones.nonzero(as_tuple=False).squeeze(-1)
                for env_idx in finished_episodes:
                    self.episode_rewards.append(episode_rewards[env_idx].item())
                    self.episode_lengths.append(episode_lengths[env_idx].item())
                
                episode_rewards[dones] = 0
                episode_lengths[dones] = 0
            
            obs = next_obs
            
            # Continue until max steps (removed early stopping for simplicity)
        
        # Compute final values for GAE
        with torch.no_grad():
            _, next_values = self.policy(obs)
        
        # Compute advantages and returns
        advantages, returns = self.buffer.compute_gae(next_values)
        
        # Prepare batch
        batch = self.buffer.get_batch(advantages, returns)
        
        # Collection metrics
        metrics = {
            'episode_reward_mean': np.mean(self.episode_rewards[-self.config['num_envs']:]) if self.episode_rewards else 0.0,
            'episode_length_mean': np.mean(self.episode_lengths[-self.config['num_envs']:]) if self.episode_lengths else 0.0,
            'amp_reward_mean': amp_rewards.mean().item(),
            'env_reward_mean': env_rewards.mean().item()
        }
        
        return batch, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step: update policy and discriminator
        
        Returns:
            Training metrics
        """
        # Update policy with PPO
        ppo_metrics = self.ppo_trainer.update_policy(
            obs=batch['observations'],
            actions=batch['actions'],
            old_log_probs=batch['old_log_probs'],
            advantages=batch['advantages'],
            returns=batch['returns']
        )
        
        # Update discriminator
        disc_metrics = self.amp_integration.train_discriminator_step(batch['observations'])
        
        # Combine metrics
        combined_metrics = {**ppo_metrics}
        combined_metrics.update({
            f"disc_{k}": v for k, v in disc_metrics.items()
        })
        
        return combined_metrics
    
    def train(self, num_iterations: int):
        """
        Main training loop
        
        Args:
            num_iterations: Number of training iterations
        """
        print(f"\n🎯 Starting training for {num_iterations} iterations")
        print("=" * 60)
        
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Collect trajectories
            batch, collection_metrics = self.collect_trajectories()
            
            # Training step
            training_metrics = self.train_step(batch)
            
            # Compute iteration time
            iteration_time = time.time() - start_time
            
            # Log progress
            if iteration % self.config['log_interval'] == 0:
                print(f"\nIteration {iteration:4d}/{num_iterations}")
                print(f"Time: {iteration_time:.2f}s")
                print(f"Episode Reward: {collection_metrics['episode_reward_mean']:.3f}")
                print(f"Episode Length: {collection_metrics['episode_length_mean']:.1f}")
                print(f"AMP Reward: {collection_metrics['amp_reward_mean']:.3f}")
                print(f"Policy Loss: {training_metrics['policy_loss']:.4f}")
                print(f"Disc Expert Acc: {training_metrics['disc_expert_accuracy']:.3f}")
                print(f"Disc Policy Acc: {training_metrics['disc_policy_accuracy']:.3f}")
                print("-" * 40)
        
        print("\n✅ Training completed!")


def create_default_config() -> Dict:
    """Create default training configuration"""
    return {
        # Environment
        'num_envs': 16,
        'episode_length_s': 10.0,
        'dt': 0.02,
        'show_viewer': False,
        
        # Training
        'max_episode_steps': 250,  # 5 seconds at 50Hz
        'min_episode_steps': 100,
        'env_reward_weight': 0.3,  # 30% env reward, 70% AMP reward
        'log_interval': 5,
        
        # Policy network
        'policy': {
            'hidden_layers': [512, 256],
            'activation': 'tanh',
            'learning_rate': 3e-4,
            'clip_epsilon': 0.2
        },
        
        # Discriminator
        'discriminator': {
            'hidden_layers': [256, 128],
            'activation': 'tanh',
            'learning_rate': 1e-4,
            'use_running_norm': True
        }
    }


def test_simple_training():
    """Test the complete training pipeline with a few iterations"""
    print("=" * 60)
    print("TESTING SIMPLE IMITATION LEARNING PIPELINE")
    print("=" * 60)
    
    try:
        # Create test configuration
        config = create_default_config()
        config['num_envs'] = 4  # Small for testing
        config['log_interval'] = 1
        
        # Create trainer
        trainer = SimpleImitationTrainer(config)
        
        # Run a few training iterations
        print("\n🧪 Running test training...")
        trainer.train(num_iterations=3)
        
        print("\n" + "=" * 60)
        print("✅ SIMPLE TRAINING TEST SUCCESS!")
        print("✅ Complete pipeline working end-to-end")
        print("✅ Ready for full-scale imitation learning")
        print("=" * 60)
        
        return True, trainer
        
    except Exception as e:
        print(f"\n❌ TRAINING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


if __name__ == "__main__":
    # Test the training pipeline
    success, trainer = test_simple_training()
    
    if success:
        print("\n🎯 Ready for Full Training!")
        print("To run full training:")
        print("1. Increase num_envs (64-256)")
        print("2. Increase num_iterations (1000+)")
        print("3. Enable show_viewer=True to watch learning")
        print("4. Experiment with reward mixing (env_reward_weight)")
    else:
        print("\n🔧 Fix issues and retry")