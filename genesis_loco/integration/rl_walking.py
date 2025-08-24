"""
Simple Reinforcement Learning Script for Skeleton Humanoid Walking

Uses PPO to train the skeleton_humanoid model to walk forward.
The environment is configured with forward motion rewards only.
Policy outputs target joint positions for PD control.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import genesis as gs

# Add parent directory to path for imports
import sys
sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.ppo_policy import PPOActorCritic


class PPOTrainer:
    """Simple PPO trainer for skeleton humanoid walking"""
    
    def __init__(self,
                 num_envs: int = 256,
                 episode_length_s: float = 5.0,
                 num_steps_per_env: int = 24,  # Short rollouts like Genesis Go2
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 use_box_feet: bool = True,
                 device: str = "cuda"):
        
        self.num_envs = num_envs
        self.episode_length_s = episode_length_s
        self.num_steps_per_env = num_steps_per_env  # Short rollout length
        self.device = device
        
        # PPO hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # Initialize environment
        print("🏃 Initializing Skeleton Humanoid Environment...")
        self.env = SkeletonHumanoidEnv(
            num_envs=num_envs,
            episode_length_s=episode_length_s,
            use_box_feet=use_box_feet,
            show_viewer=False,
            dt=0.01  #
        )
        
        # Get environment dimensions
        obs = self.env.reset()
        self.obs_dim = self.env.num_observations
        self.action_dim = self.env.num_actions
        
        print(f"   Observation dim: {self.obs_dim}")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Number of envs: {num_envs}")
        
        # Initialize policy
        print("🧠 Initializing PPO Policy...")
        self.policy = PPOActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=[512, 256, 128],
            activation='tanh',
            init_std=0.3,
            learnable_std=True,
            use_running_norm=True
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Training tracking
        self.training_metrics = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'total_timesteps': 0
        }
        
        print("✅ PPO Trainer initialized successfully!")
        
        # Initial reset to start environments
        self.env.reset()
    
    def collect_rollout(self) -> Dict[str, torch.Tensor]:
        """Collect short rollout data from environment (Genesis Go2 style)"""
        # Get current observations (don't reset - continue from where we are)
        obs, _ = self.env.get_observations()
        
        # Storage for short rollout
        rollout_steps = self.num_steps_per_env
        observations = torch.zeros((rollout_steps, self.num_envs, self.obs_dim), device=self.device)
        actions = torch.zeros((rollout_steps, self.num_envs, self.action_dim), device=self.device)
        rewards = torch.zeros((rollout_steps, self.num_envs), device=self.device)
        dones = torch.zeros((rollout_steps, self.num_envs), device=self.device)
        values = torch.zeros((rollout_steps, self.num_envs), device=self.device)
        log_probs = torch.zeros((rollout_steps, self.num_envs), device=self.device)
        
        # Collect short rollout - minimize CPU-GPU transfers
        for step in range(rollout_steps):
            observations[step] = obs
            
            with torch.no_grad():
                action_dist, value = self.policy(obs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            actions[step] = action
            values[step] = value.squeeze(-1)
            log_probs[step] = log_prob
            
            # Step environment - keep everything on GPU
            obs, reward, done, info = self.env.step(action)
            
            # Ensure tensors stay on GPU
            rewards[step] = reward.to(self.device) if not reward.is_cuda else reward
            dones[step] = done.float().to(self.device) if not done.is_cuda else done.float()
        
        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        return {
            'observations': observations.reshape(-1, self.obs_dim),
            'actions': actions.reshape(-1, self.action_dim),
            'old_log_probs': log_probs.reshape(-1),
            'advantages': advantages.reshape(-1),
            'returns': returns.reshape(-1),
            'episode_rewards': rewards.sum(dim=0)
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation - Vectorized GPU version"""
        T, N = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Append zeros for final timestep
        next_values = torch.cat([values[1:], torch.zeros(1, N, device=self.device)], dim=0)
        next_dones = torch.cat([dones[1:], torch.ones(1, N, device=self.device)], dim=0)
        
        # Compute deltas vectorized
        deltas = rewards + self.gamma * next_values * (1.0 - next_dones) - values
        
        # Compute GAE using scan (still needs loop but minimized)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - next_dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def update_policy(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using PPO"""
        observations = rollout_data['observations']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['old_log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Multiple epochs of optimization
        for _ in range(self.ppo_epochs):
            # Mini-batch training - keep indices on GPU
            indices = torch.randperm(observations.shape[0], device=self.device)
            for start in range(0, observations.shape[0], self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                action_dist, values = self.policy(batch_obs)
                
                # Policy loss
                log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(-1), batch_returns)
                
                # Entropy loss
                entropy_loss = -action_dist.entropy().mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
    
    def train(self, total_timesteps: int = 1000000, save_interval: int = 100000):
        """Main training loop"""
        print(f"🚀 Starting PPO training for {total_timesteps} timesteps...")
        
        # Use short rollouts instead of full episodes
        rollout_timesteps = self.num_envs * self.num_steps_per_env
        num_rollouts = total_timesteps // rollout_timesteps
        
        print(f"   Rollout timesteps: {rollout_timesteps} (vs {self.num_envs * self.env.max_episode_length} for full episodes)")
        print(f"   Number of rollouts: {num_rollouts}")
        
        start_time = time.time()
        
        for rollout in range(num_rollouts):
            # Collect rollout
            rollout_data = self.collect_rollout()
            
            # Update policy
            training_metrics = self.update_policy(rollout_data)
            
            # Track metrics
            self.training_metrics['episode_rewards'].append(rollout_data['episode_rewards'].mean().item())
            self.training_metrics['policy_losses'].append(training_metrics['policy_loss'])
            self.training_metrics['value_losses'].append(training_metrics['value_loss'])
            self.training_metrics['entropy_losses'].append(training_metrics['entropy_loss'])
            self.training_metrics['total_timesteps'] += rollout_timesteps
            
            # Logging
            if rollout % 10 == 0:
                avg_reward = np.mean(self.training_metrics['episode_rewards'][-10:])
                elapsed_time = time.time() - start_time
                timesteps_per_sec = self.training_metrics['total_timesteps'] / elapsed_time
                
                print(f"Rollout {rollout:4d} | "
                      f"Timesteps: {self.training_metrics['total_timesteps']:8d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Policy Loss: {training_metrics['policy_loss']:6.4f} | "
                      f"Value Loss: {training_metrics['value_loss']:6.4f} | "
                      f"FPS: {timesteps_per_sec:6.0f}")
            
            # Save model
            save_rollout_interval = max(1, save_interval // rollout_timesteps)
            if rollout % save_rollout_interval == 0 and rollout > 0:
                self.save_model(f"rl_walking_checkpoint_{self.training_metrics['total_timesteps']}")
        
        print("✅ Training completed!")
        self.save_model("rl_walking_final")
        self.plot_training_curves()
    
    def save_model(self, filename: str):
        """Save model and training metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"/home/ez/Documents/Genesis/genesis_loco/{filename}_{timestamp}.pth"
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_metrics': self.training_metrics,
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'num_envs': self.num_envs,
                'episode_length_s': self.episode_length_s
            }
        }, save_path)
        print(f"💾 Model saved to: {save_path}")
    
    def plot_training_curves(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode rewards
        axes[0, 0].plot(self.training_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Rollout')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True)
        
        # Policy loss
        axes[0, 1].plot(self.training_metrics['policy_losses'])
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].set_xlabel('Rollout')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Value loss
        axes[1, 0].plot(self.training_metrics['value_losses'])
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].set_xlabel('Rollout')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Entropy loss
        axes[1, 1].plot(self.training_metrics['entropy_losses'])
        axes[1, 1].set_title('Entropy Loss')
        axes[1, 1].set_xlabel('Rollout')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"/home/ez/Documents/Genesis/genesis_loco/rl_walking_training_{timestamp}.png", dpi=300)
        print(f"📊 Training curves saved")


def main():
    """Main training function"""
    # Training configuration - Genesis Go2 style
    config = {
        'num_envs': 2048,          # Increased parallel environments (Genesis uses 4096)
        'episode_length_s': 10.0,   # Episode length in seconds (for termination)
        'num_steps_per_env': 24,   # Short rollout steps (like Genesis Go2)
        'lr': 3e-4,                # Learning rate
        'total_timesteps': 10000000, # Total training timesteps
        'use_box_feet': True,      # Use box feet for stability
    }
    
    print("🦴 Skeleton Humanoid RL Walking Training")
    print("=" * 50)
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 50)
    
    # Extract total_timesteps for training
    total_timesteps = config.pop('total_timesteps')

    gs.init(backend=gs.cuda)
    
    # Initialize trainer
    trainer = PPOTrainer(**config)
    
    # Start training
    trainer.train(total_timesteps=total_timesteps)


if __name__ == "__main__":
    main()