"""
GAIL Training Loop - PyTorch Implementation adapted from LocoMujoco

Complete GAIL training implementation integrating:
- PPO policy training
- GAIL discriminator training  
- Expert trajectory data loading
- Genesis environment integration
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from dataclasses import dataclass

from gail_discriminator import GAILDiscriminator, GAILTrainer
from ppo_policy import PPOActorCritic, PPOTrainer
from data_bridge import LocoMujocoDataBridge


@dataclass
class GAILConfig:
    """
    GAIL training configuration
    Adapted from LocoMujoco's configuration structure
    """
    # Environment parameters
    num_envs: int = 2048
    num_steps: int = 14
    episode_length_s: float = 5.0
    dt: float = 0.01
    
    # Training parameters
    total_timesteps: int = 75_000_000
    update_epochs: int = 4
    n_disc_epochs: int = 10  # LocoMujoco default: train discriminator thoroughly each update
    disc_minibatch_size: int = 2048
    num_minibatches: int = 32
    train_disc_interval: int = 1  # Train discriminator every nth update (1=every update like LocoMujoco)
    
    # PPO parameters
    lr: float = 1e-4  # LocoMujoco default policy learning rate
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2  # LocoMujoco default clip epsilon
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    weight_decay: float = 0.0
    
    # Discriminator parameters
    disc_lr: float = 5e-5  # LocoMujoco default discriminator learning rate
    disc_ent_coef: float = 0.01  # Entropy regularization to prevent discriminator overconfidence
    
    # Dynamic training schedule parameters
    use_dynamic_schedule: bool = False  # Enable adaptive discriminator training
    disc_expert_threshold_high: float = 0.93  # Reduce disc training when expert accuracy > this
    disc_expert_threshold_low: float = 0.85   # Increase disc training when expert accuracy < this
    disc_policy_threshold_low: float = 0.25   # Alert when policy accuracy < this
    n_disc_epochs_min: int = 3   # Minimum discriminator epochs when reducing
    n_disc_epochs_max: int = 15  # Maximum discriminator epochs when increasing
    
    # Network architecture
    hidden_layers: List[int] = None
    activation: str = 'tanh'
    init_std: float = 0.125
    learnable_std: bool = False
    
    # Reward mixing
    proportion_env_reward: float = 0.0  # 0.0 = pure GAIL, 1.0 = pure environment
    
    # Logging
    log_interval: int = 10
    save_interval: int = 1000
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [512, 256]
        
        # Compute derived parameters (following LocoMujoco)
        self.num_updates = self.total_timesteps // (self.num_steps * self.num_envs)
        self.minibatch_size = (self.num_envs * self.num_steps) // self.num_minibatches


class GAILGenesisTrainer:
    """
    GAIL Trainer for Genesis Environments
    
    Integrates PPO policy training with GAIL discriminator following LocoMujoco's approach.
    """
    
    def __init__(self,
                 genesis_env,
                 data_bridge: LocoMujocoDataBridge,
                 config: GAILConfig,
                 device: torch.device = None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.genesis_env = genesis_env
        self.data_bridge = data_bridge
        self.config = config
        self.device = device
        
        # Initialize policy
        self.policy = PPOActorCritic(
            obs_dim=genesis_env.num_observations,
            action_dim=genesis_env.num_actions,
            hidden_layers=config.hidden_layers,
            activation=config.activation,
            init_std=config.init_std,
            learnable_std=config.learnable_std,
            use_running_norm=True
        )
        
        # Initialize discriminator
        self.discriminator = GAILDiscriminator(
            input_dim=genesis_env.num_observations,
            hidden_layers=config.hidden_layers,
            activation=config.activation,
            use_running_norm=True
        )
        
        # Initialize trainers
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            learning_rate=config.lr,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_eps=config.clip_eps,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            weight_decay=config.weight_decay,
            device=device
        )
        
        self.gail_trainer = GAILTrainer(
            discriminator=self.discriminator,
            learning_rate=config.disc_lr,
            entropy_coef=config.disc_ent_coef,
            device=device
        )
        
        # Expert data storage
        self.expert_observations = None
        
        # Training metrics
        self.training_metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'discriminator_losses': [],
            'discriminator_output_policy': [],
            'discriminator_output_expert': [],
            'timesteps': []
        }
        
        print(f"✅ GAIL Trainer initialized:")
        print(f"   - Policy parameters: {sum(p.numel() for p in self.policy.parameters())}")
        print(f"   - Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")
        print(f"   - Device: {device}")
    
    def load_expert_data(self) -> bool:
        """
        Load expert trajectory data using cached physics-based observations
        
        Returns:
            bool: Success status
        """
        print("Loading expert trajectory data...")
        
        if self.data_bridge.loco_trajectory is None:
            print("❌ No trajectory loaded in data bridge")
            return False
        
        trajectory_length = self.data_bridge.trajectory_length
        
        # Use cached expert observations with physics integration (all timesteps)
        self.expert_observations = self.data_bridge.get_expert_observations_cached(
            dataset_name="walk",
            num_timesteps=None,  # Use all timesteps by default
            start_timestep=0,    # Start from beginning  
            step_interval=1,     # Every timestep
            force_reload=False   # Use cache if available
        )
        
        if self.expert_observations is None:
            print("❌ Failed to load expert observations")
            return False
        
        print(f"   ✅ Expert observation shape: {self.expert_observations.shape}")
        return True
    
    def sample_expert_batch(self, batch_size: int) -> torch.Tensor:
        """Sample random batch of expert observations"""
        if self.expert_observations is None:
            raise RuntimeError("Expert data not loaded. Call load_expert_data() first.")
        
        n_expert = self.expert_observations.shape[0]
        indices = torch.randint(0, n_expert, (batch_size,), device=self.device)
        return self.expert_observations[indices]
    
    def get_adaptive_disc_epochs(self, expert_output: float, policy_output: float) -> int:
        """
        Aggressive adaptive discriminator epochs to prevent overconfidence
        
        Args:
            expert_output: Current expert discriminator output (0-1)
            policy_output: Current policy discriminator output (0-1)
            
        Returns:
            int: Number of discriminator epochs to use
        """
        if not self.config.use_dynamic_schedule:
            return self.config.n_disc_epochs
        
        # Calculate discriminator confidence gap - ideal is expert=1, policy=0, but we want balance
        confidence_gap = expert_output + (1 - policy_output) - 1.0  # How far from balanced (should be ~0)
        
        # Be much more aggressive in reducing discriminator training
        if expert_output > 0.85 or confidence_gap > 0.4:  # Very overconfident
            return 1  # Minimal training
        elif expert_output > 0.75 or confidence_gap > 0.25:  # Moderately overconfident  
            return max(1, self.config.n_disc_epochs // 3)  # Heavily reduced
        elif expert_output > 0.65 or confidence_gap > 0.15:  # Slightly overconfident
            return max(2, self.config.n_disc_epochs // 2)  # Moderately reduced
        else:
            return self.config.n_disc_epochs  # Normal training
    
    def collect_trajectories(self) -> Tuple[torch.Tensor, ...]:
        """
        Collect trajectories using current policy
        
        Following LocoMujoco's trajectory collection approach
        """
        observations = torch.zeros((self.config.num_envs, self.config.num_steps, self.genesis_env.num_observations), device=self.device)
        actions = torch.zeros((self.config.num_envs, self.config.num_steps, self.genesis_env.num_actions), device=self.device)
        log_probs = torch.zeros((self.config.num_envs, self.config.num_steps), device=self.device)
        values = torch.zeros((self.config.num_envs, self.config.num_steps), device=self.device)
        rewards = torch.zeros((self.config.num_envs, self.config.num_steps), device=self.device)
        dones = torch.zeros((self.config.num_envs, self.config.num_steps), device=self.device)
        
        # Get initial observation
        obs = self.genesis_env._get_observations()
        
        for step in range(self.config.num_steps):
            observations[:, step] = obs
            
            # Sample action
            with torch.no_grad():
                action, log_prob, _, value = self.policy.get_action_and_value(obs)
            
            actions[:, step] = action
            log_probs[:, step] = log_prob
            values[:, step] = value
            
            # Step environment
            obs, env_reward, reset_buf, info = self.genesis_env.step(action)
            done = reset_buf
            
            # Compute GAIL rewards
            gail_reward = self.gail_trainer.compute_gail_rewards(obs)
            
            # Mix rewards (following LocoMujoco)
            mixed_reward = (self.config.proportion_env_reward * env_reward + 
                           (1 - self.config.proportion_env_reward) * gail_reward)
            
            rewards[:, step] = mixed_reward
            dones[:, step] = done.float()
        
        # Compute next value for GAE
        with torch.no_grad():
            _, _, _, next_value = self.policy.get_action_and_value(obs)
        
        return observations, actions, log_probs, values, rewards, dones, next_value
    
    def train_step(self, update_num: int) -> Dict[str, float]:
        """
        Single GAIL training step
        
        Following LocoMujoco's training procedure:
        1. Collect trajectories
        2. Train discriminator
        3. Compute GAE
        4. Update PPO policy
        """
        # Collect trajectories
        observations, actions, log_probs, values, rewards, dones, next_value = self.collect_trajectories()
        
        # Train discriminator with adaptive frequency and epochs
        disc_should_train = update_num % self.config.train_disc_interval == 0
        
        # Reduce training frequency when discriminator is overconfident
        if update_num > 0 and self.gail_trainer.last_metrics:
            expert_output = self.gail_trainer.last_metrics.get('discriminator_output_expert', 0.5)
            policy_output = self.gail_trainer.last_metrics.get('discriminator_output_policy', 0.5)
            
            # Skip discriminator training more often when it's overconfident
            if expert_output > 0.9:
                disc_should_train = update_num % (self.config.train_disc_interval * 4) == 0
            elif expert_output > 0.8:
                disc_should_train = update_num % (self.config.train_disc_interval * 2) == 0
        
        if disc_should_train:
            # Get adaptive number of epochs based on current performance
            if update_num > 0 and self.gail_trainer.last_metrics:
                expert_output = self.gail_trainer.last_metrics.get('discriminator_output_expert', 0.5)
                policy_output = self.gail_trainer.last_metrics.get('discriminator_output_policy', 0.5)
                adaptive_epochs = self.get_adaptive_disc_epochs(expert_output, policy_output)
            else:
                adaptive_epochs = self.config.n_disc_epochs
            
            # Store for monitoring
            self._last_adaptive_epochs = adaptive_epochs
            
            # Log adaptive changes occasionally
            if adaptive_epochs != self.config.n_disc_epochs and update_num % 20 == 0:
                print(f"   🔄 Adaptive epochs: {adaptive_epochs} (base: {self.config.n_disc_epochs})")
                
            disc_metrics_list = []
            for _ in range(adaptive_epochs):
                # Sample policy and expert batches
                policy_batch = observations.reshape(-1, observations.shape[-1])
                policy_indices = torch.randint(0, policy_batch.shape[0], (self.config.disc_minibatch_size,), device=self.device)
                policy_sample = policy_batch[policy_indices]
                
                expert_sample = self.sample_expert_batch(self.config.disc_minibatch_size)
                
                # Train discriminator
                disc_metrics = self.gail_trainer.train_discriminator(expert_sample, policy_sample)
                disc_metrics_list.append(disc_metrics)
            
            # Average discriminator metrics
            avg_disc_metrics = {}
            for key in disc_metrics_list[0].keys():
                avg_disc_metrics[key] = np.mean([m[key] for m in disc_metrics_list])
        else:
            # Skip discriminator training, use last known metrics
            avg_disc_metrics = self.gail_trainer.last_metrics.copy() if self.gail_trainer.last_metrics else {
                'discriminator_loss': 0.0,
                'policy_accuracy': 0.5,
                'expert_accuracy': 0.5,
                'discriminator_output_policy': 0.5,
                'discriminator_output_expert': 0.5
            }
        
        # Compute GAE
        advantages, returns = self.ppo_trainer.compute_gae(rewards, values, dones, next_value)
        
        # Update PPO policy
        ppo_metrics = self.ppo_trainer.update_policy(
            observations, actions, log_probs, advantages, returns, values,
            num_epochs=self.config.update_epochs,
            batch_size=self.config.minibatch_size
        )
        
        # Combine metrics
        combined_metrics = {
            'mean_episode_return': rewards.sum(dim=1).mean().item(),
            'mean_episode_length': self.config.num_steps,  # Fixed length episodes
            **ppo_metrics,
            **avg_disc_metrics,
            'timestep': (update_num + 1) * self.config.num_envs * self.config.num_steps
        }
        
        # Add adaptive training info for monitoring
        if hasattr(self, '_last_adaptive_epochs'):
            combined_metrics['adaptive_disc_epochs'] = self._last_adaptive_epochs
        
        return combined_metrics
    
    def train(self, save_dir: str = "./gail_outputs") -> Dict[str, List]:
        """
        Main GAIL training loop
        
        Following LocoMujoco's training structure
        """
        print(f"🚀 Starting GAIL training for {self.config.total_timesteps:,} timesteps")
        print(f"   Updates: {self.config.num_updates}")
        print(f"   Environments: {self.config.num_envs}")
        print(f"   Steps per update: {self.config.num_steps}")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load expert data
        if not self.load_expert_data():
            raise RuntimeError("Failed to load expert data")
        
        # Reset environment
        self.genesis_env.reset()
        
        start_time = time.time()
        
        for update in range(self.config.num_updates):
            # Training step
            metrics = self.train_step(update)
            
            # Store metrics
            for key, value in metrics.items():
                if key not in self.training_metrics:
                    self.training_metrics[key] = []
                self.training_metrics[key].append(value)
            
            # Logging
            if (update + 1) % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                timesteps_done = metrics['timestep']
                timesteps_per_sec = timesteps_done / elapsed_time
                
                print(f"Update {update + 1}/{self.config.num_updates} | "
                      f"Timesteps: {timesteps_done:,} | "
                      f"Return: {metrics['mean_episode_return']:.3f} | "
                      f"Policy Loss: {metrics['policy_loss']:.4f} | "
                      f"Disc Loss: {metrics['discriminator_loss']:.4f} | "
                      f"Disc Out Policy: {metrics['discriminator_output_policy']:.3f} | "
                      f"Disc Out Expert: {metrics['discriminator_output_expert']:.3f} | "
                      f"Steps/sec: {timesteps_per_sec:.0f}")
            
            # Saving
            if (update + 1) % self.config.save_interval == 0:
                self.save_checkpoint(save_dir, update + 1)
        
        # Final save
        self.save_checkpoint(save_dir, self.config.num_updates, final=True)
        
        print(f"\n🎉 Training completed in {time.time() - start_time:.1f}s")
        return self.training_metrics
    
    def save_checkpoint(self, save_dir: str, update: int, final: bool = False):
        """Save training checkpoint"""
        suffix = "final" if final else f"update_{update}"
        
        # Save policy
        policy_path = os.path.join(save_dir, f"gail_policy_{suffix}.pth")
        self.ppo_trainer.save_policy(policy_path)
        
        # Save discriminator
        disc_path = os.path.join(save_dir, f"gail_discriminator_{suffix}.pth")
        self.gail_trainer.save_discriminator(disc_path)
        
        # Save metrics
        metrics_path = os.path.join(save_dir, f"gail_metrics_{suffix}.pth")
        torch.save(self.training_metrics, metrics_path)
        
        if final:
            print(f"✅ Final checkpoint saved to {save_dir}")
        else:
            print(f"   Checkpoint saved at update {update}")
    
    def load_checkpoint(self, policy_path: str, discriminator_path: str):
        """Load training checkpoint"""
        self.ppo_trainer.load_policy(policy_path)
        self.gail_trainer.load_discriminator(discriminator_path)
        print(f"✅ Checkpoint loaded from {policy_path} and {discriminator_path}")


def test_gail_trainer():
    """Test GAIL trainer with skeleton environment"""
    print("🧪 Testing GAIL Trainer")
    print("=" * 50)
    
    try:
        import genesis as gs
        gs.init(backend=gs.gpu)
        
        # Import environment components
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        # Create environment
        print("1. Creating Genesis environment...")
        env = SkeletonHumanoidEnv(
            num_envs=64,  # Smaller for testing
            episode_length_s=5.0,
            dt=0.01,
            use_box_feet=True,
            show_viewer=False
        )
        print(f"   ✅ Environment: {env.num_envs} envs, {env.num_observations} obs, {env.num_actions} actions")
        
        # Create data bridge
        print("2. Creating data bridge...")
        data_bridge = LocoMujocoDataBridge(env)
        success = data_bridge.load_trajectory("walk")
        if not success:
            print("   ❌ Failed to load trajectory")
            return False
        print(f"   ✅ Trajectory loaded: {data_bridge.trajectory_length} timesteps")
        
        # Create config
        print("3. Creating GAIL config...")
        config = GAILConfig(
            num_envs=64,
            num_steps=10,
            total_timesteps=100_000,  # Small for testing
            n_disc_epochs=2,
            disc_minibatch_size=128,
            num_minibatches=8,
            log_interval=5
        )
        print(f"   ✅ Config created: {config.num_updates} updates")
        
        # Create GAIL trainer
        print("4. Creating GAIL trainer...")
        trainer = GAILGenesisTrainer(env, data_bridge, config)
        
        # Test expert data loading
        print("5. Testing expert data loading...")
        success = trainer.load_expert_data()
        if not success:
            print("   ❌ Failed to load expert data")
            return False
        print(f"   ✅ Expert data loaded: {trainer.expert_observations.shape[0]} samples")
        
        # Test single training step
        print("6. Testing single training step...")
        metrics = trainer.train_step(0)
        print(f"   ✅ Training step completed:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.4f}")
            else:
                print(f"      {key}: {value}")
        
        print("\n🎉 All GAIL trainer tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_gail_trainer()