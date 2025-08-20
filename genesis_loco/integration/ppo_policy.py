"""
PPO Policy Network - PyTorch Implementation adapted from LocoMujoco

PyTorch implementation of PPO policy compatible with Genesis environments.
Follows LocoMujoco's ActorCritic architecture and PPO training approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Tuple, Optional
import numpy as np


class RunningMeanStd(nn.Module):
    """
    Running mean and standard deviation normalization
    Shared with GAIL discriminator implementation
    """
    
    def __init__(self, input_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        
        # Buffers for persistent statistics
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('var', torch.ones(input_dim))
        self.register_buffer('count', torch.zeros(1))
        
    def forward(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """Normalize input and optionally update statistics"""
        if update_stats and self.training:
            self._update_stats(x)
            
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)
    
    def _update_stats(self, x: torch.Tensor):
        """Update running statistics"""
        batch_size = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        
        # Online update
        new_count = self.count + batch_size
        delta = batch_mean - self.mean
        
        # Update mean and variance
        self.mean += delta * batch_size / new_count
        self.var = (self.var * self.count + batch_var * batch_size + 
                   delta.pow(2) * self.count * batch_size / new_count) / new_count
        self.count.copy_(new_count)


class PPOActorCritic(nn.Module):
    """
    PPO Actor-Critic Network
    
    Adapted from LocoMujoco's ActorCritic implementation.
    Supports separate observation indices for actor and critic.
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 hidden_layers: list = [512, 256],
                 activation: str = 'tanh',
                 init_std: float = 0.125,
                 learnable_std: bool = False,
                 use_running_norm: bool = True,
                 actor_obs_indices: Optional[torch.Tensor] = None,
                 critic_obs_indices: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learnable_std = learnable_std
        self.use_running_norm = use_running_norm
        
        # Observation indices for actor and critic (following LocoMujoco)
        if actor_obs_indices is None:
            actor_obs_indices = torch.arange(obs_dim)
        if critic_obs_indices is None:
            critic_obs_indices = torch.arange(obs_dim)
            
        self.register_buffer('actor_obs_indices', actor_obs_indices)
        self.register_buffer('critic_obs_indices', critic_obs_indices)
        
        actor_input_dim = len(actor_obs_indices)
        critic_input_dim = len(critic_obs_indices)
        
        # Input normalization
        if use_running_norm:
            self.actor_norm = RunningMeanStd(actor_input_dim)
            self.critic_norm = RunningMeanStd(critic_input_dim)
        
        # Actor network
        actor_layers = []
        prev_dim = actor_input_dim
        for hidden_dim in hidden_layers:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh() if activation == 'tanh' else nn.ReLU()
            ])
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        prev_dim = critic_input_dim
        for hidden_dim in hidden_layers:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh() if activation == 'tanh' else nn.ReLU()
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action standard deviation
        if learnable_std:
            self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_std))
        else:
            self.register_buffer('log_std', torch.ones(action_dim) * np.log(init_std))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor, update_stats: bool = True) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass through actor-critic
        
        Returns:
            action_distribution: Normal distribution over actions
            value: State value estimate
        """
        # Extract observations for actor and critic
        actor_obs = obs[:, self.actor_obs_indices]
        critic_obs = obs[:, self.critic_obs_indices]
        
        # Apply normalization if enabled
        if self.use_running_norm:
            actor_obs = self.actor_norm(actor_obs, update_stats=update_stats)
            critic_obs = self.critic_norm(critic_obs, update_stats=update_stats)
        
        # Actor forward pass
        action_mean = self.actor(actor_obs)
        action_std = torch.exp(self.log_std)
        action_distribution = Normal(action_mean, action_std)
        
        # Critic forward pass
        value = self.critic(critic_obs).squeeze(-1)
        
        return action_distribution, value
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value
        
        Args:
            obs: Observations
            action: Optional pre-computed action
            
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value estimate
        """
        action_dist, value = self.forward(obs)
        
        if action is None:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action).sum(-1)
        entropy = action_dist.entropy().sum(-1)
        
        return action, log_prob, entropy, value


class PPOTrainer:
    """
    PPO Policy Trainer
    
    Handles PPO training following LocoMujoco's approach:
    - Clipped policy objective
    - Value function loss with clipping
    - Entropy regularization
    - GAE for advantage estimation
    """
    
    def __init__(self,
                 policy: PPOActorCritic,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.0,
                 max_grad_norm: float = 0.5,
                 weight_decay: float = 0.0,
                 device: torch.device = None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = policy.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.AdamW(
            policy.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            eps=1e-5
        )
        
        # Metrics tracking
        self.last_metrics = {}
    
    def compute_gae(self, 
                   rewards: torch.Tensor,
                   values: torch.Tensor,
                   dones: torch.Tensor,
                   next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Following LocoMujoco's GAE implementation with reverse scan.
        """
        batch_size, num_steps = rewards.shape
        
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(batch_size, device=self.device)
        
        # Compute GAE in reverse order (following LocoMujoco)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[:, t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[:, t]
                next_value_t = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value_t * next_non_terminal - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[:, t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self,
                     observations: torch.Tensor,
                     actions: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages: torch.Tensor,
                     returns: torch.Tensor,
                     old_values: torch.Tensor,
                     num_epochs: int = 4,
                     batch_size: int = 2048) -> Dict[str, float]:
        """
        Update PPO policy using multiple epochs of minibatch updates
        
        Following LocoMujoco's PPO update procedure.
        """
        # Flatten data for minibatch training
        num_envs, num_steps = observations.shape[:2]
        total_samples = num_envs * num_steps
        
        observations = observations.reshape(total_samples, -1)
        actions = actions.reshape(total_samples, -1)
        old_log_probs = old_log_probs.reshape(total_samples)
        advantages = advantages.reshape(total_samples)
        returns = returns.reshape(total_samples)
        old_values = old_values.reshape(total_samples)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        update_count = 0
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(total_samples, device=self.device)
            
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Policy loss (clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.clip_eps, self.clip_eps
                )
                value_losses = F.mse_loss(new_values, batch_returns, reduction='none')
                value_losses_clipped = F.mse_loss(value_pred_clipped, batch_returns, reduction='none')
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_coef * value_loss - 
                             self.entropy_coef * entropy_loss)
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                update_count += 1
        
        # Average metrics
        self.last_metrics = {
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy_loss / update_count
        }
        
        return self.last_metrics
    
    def save_policy(self, path: str):
        """Save policy state"""
        torch.save({
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': self.last_metrics
        }, path)
    
    def load_policy(self, path: str):
        """Load policy state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.last_metrics = checkpoint.get('metrics', {})


def test_ppo_policy():
    """Test PPO policy implementation"""
    print("🧪 Testing PPO Policy")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters matching skeleton environment
    obs_dim = 65
    action_dim = 23
    num_envs = 4
    num_steps = 10
    
    try:
        # Create policy
        print("1. Creating PPO policy...")
        policy = PPOActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_layers=[512, 256],
            activation='tanh',
            init_std=0.125,
            learnable_std=False,
            use_running_norm=True
        )
        print(f"   ✅ Policy created: {sum(p.numel() for p in policy.parameters())} parameters")
        
        # Create trainer
        print("2. Creating PPO trainer...")
        trainer = PPOTrainer(policy, learning_rate=1e-4, device=device)
        print("   ✅ Trainer created")
        
        # Test forward pass
        print("3. Testing forward pass...")
        obs = torch.randn(num_envs, obs_dim, device=device)
        action_dist, value = policy(obs)
        print(f"   ✅ Forward pass: action_dist mean shape {action_dist.mean.shape}, value shape {value.shape}")
        
        # Test action sampling
        print("4. Testing action sampling...")
        action, log_prob, entropy, value = policy.get_action_and_value(obs)
        print(f"   ✅ Action sampling: action shape {action.shape}, log_prob shape {log_prob.shape}")
        
        # Test GAE computation
        print("5. Testing GAE computation...")
        rewards = torch.randn(num_envs, num_steps, device=device)
        values = torch.randn(num_envs, num_steps, device=device)
        dones = torch.zeros(num_envs, num_steps, device=device)
        next_value = torch.randn(num_envs, device=device)
        
        advantages, returns = trainer.compute_gae(rewards, values, dones, next_value)
        print(f"   ✅ GAE computation: advantages shape {advantages.shape}, returns shape {returns.shape}")
        
        # Test policy update
        print("6. Testing policy update...")
        observations = torch.randn(num_envs, num_steps, obs_dim, device=device)
        actions = torch.randn(num_envs, num_steps, action_dim, device=device)
        old_log_probs = torch.randn(num_envs, num_steps, device=device)
        old_values = torch.randn(num_envs, num_steps, device=device)
        
        metrics = trainer.update_policy(
            observations, actions, old_log_probs, advantages, returns, old_values,
            num_epochs=2, batch_size=16
        )
        print(f"   ✅ Policy update completed:")
        for key, value in metrics.items():
            print(f"      {key}: {value:.4f}")
        
        print("\n🎉 All PPO policy tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_ppo_policy()