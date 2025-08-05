"""
Simple PPO Policy Network for Genesis Skeleton Imitation Learning

A clean, minimal implementation matching LocoMujoco's policy architecture
but adapted for PyTorch and Genesis environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict


class RunningMeanStd(nn.Module):
    """Running mean and std for observation normalization (same as discriminator)"""
    
    def __init__(self, input_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('var', torch.ones(input_dim))
        self.register_buffer('count', torch.zeros(1))
        
    def forward(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        if update_stats and self.training:
            self._update_stats(x)
            
        normalized = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return normalized
    
    def _update_stats(self, x: torch.Tensor):
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        
        new_count = self.count + batch_size
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_size / new_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_size
        m_ab = ((self.mean - batch_mean) ** 2) * self.count * batch_size / new_count
        new_var = (m_a + m_b + m_ab) / new_count
        
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(new_count)


class SkeletonPolicyNetwork(nn.Module):
    """
    Actor-Critic network for Genesis skeleton model
    
    Architecture matching LocoMujoco:
    - Shared observation processing with running normalization
    - Separate actor and critic heads
    - Continuous action output with learned std
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 hidden_layers: list = [512, 256],
                 activation: str = 'tanh',
                 use_obs_norm: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_obs_norm = use_obs_norm
        
        # Observation normalization
        if self.use_obs_norm:
            self.obs_norm = RunningMeanStd(obs_dim)
        
        # Shared feature extraction
        shared_layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_layers:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                shared_layers.append(nn.Tanh())
            elif activation == 'relu':
                shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*shared_layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))  # Learnable log std
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Small initialization for actor head
        nn.init.xavier_uniform_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
    
    def forward(self, obs: torch.Tensor, update_obs_stats: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network
        
        Args:
            obs: Observations [batch_size, obs_dim]
            update_obs_stats: Whether to update observation normalization stats
            
        Returns:
            action_mean: Mean actions [batch_size, action_dim]
            values: Value estimates [batch_size]
        """
        x = obs
        
        # Normalize observations
        if self.use_obs_norm:
            x = self.obs_norm(x, update_stats=update_obs_stats)
        
        # Shared feature extraction
        features = self.shared_net(x)
        
        # Actor output (action mean)
        action_mean = self.actor_mean(features)
        
        # Critic output (state value)
        values = self.critic(features).squeeze(-1)
        
        return action_mean, values
    
    def get_action_distribution(self, obs: torch.Tensor) -> Normal:
        """Get action distribution for sampling and log prob computation"""
        action_mean, _ = self.forward(obs)
        action_std = torch.exp(self.actor_logstd)
        
        return Normal(action_mean, action_std)
    
    def sample_actions(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from policy
        
        Returns:
            actions: Sampled actions [batch_size, action_dim]
            log_probs: Log probabilities [batch_size]
            values: Value estimates [batch_size]
        """
        action_dist = self.get_action_distribution(obs)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        
        _, values = self.forward(obs)
        
        return actions, log_probs, values
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training
        
        Returns:
            log_probs: Log probabilities of given actions
            values: Value estimates
            entropy: Action distribution entropy
        """
        action_dist = self.get_action_distribution(obs)
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)
        
        _, values = self.forward(obs)
        
        return log_probs, values, entropy


class PPOTrainer:
    """
    Simple PPO trainer for skeleton policy
    
    Implements the core PPO algorithm with:
    - Clipped policy objective
    - Value function learning
    - Entropy regularization
    """
    
    def __init__(self,
                 policy: SkeletonPolicyNetwork,
                 learning_rate: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: torch.device = torch.device('cuda')):
        
        self.policy = policy.to(device)
        self.device = device
        
        # PPO hyperparameters
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training metrics
        self.training_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0
        }
    
    def update_policy(self, 
                     obs: torch.Tensor,
                     actions: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages: torch.Tensor,
                     returns: torch.Tensor) -> Dict[str, float]:
        """
        PPO policy update
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Actions taken [batch_size, action_dim]
            old_log_probs: Log probs from behavior policy [batch_size]
            advantages: GAE advantages [batch_size]
            returns: Discounted returns [batch_size]
            
        Returns:
            Training metrics
        """
        self.policy.train()
        
        # Evaluate current policy on the data
        log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)
        
        # Policy loss (PPO clipped objective)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy loss (for exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coeff * value_loss + 
                     self.entropy_coeff * entropy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Compute additional metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
            
            self.training_stats.update({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'total_loss': total_loss.item(),
                'approx_kl': approx_kl.item(),
                'clip_fraction': clip_fraction.item()
            })
        
        return self.training_stats


def test_skeleton_policy():
    """
    Test the skeleton policy network
    
    Verifies:
    1. Network forward pass
    2. Action sampling
    3. PPO training step
    """
    print("=" * 60)
    print("TESTING SKELETON POLICY NETWORK")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters (matching skeleton environment)
    batch_size = 64
    obs_dim = 55      # Skeleton observation dimension
    action_dim = 27   # Skeleton action dimension (without foot joints)
    
    print(f"\nTest setup:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Observation dimension: {obs_dim}")
    print(f"  - Action dimension: {action_dim}")
    
    # Create policy network
    print("\n1. Creating policy network...")
    policy = SkeletonPolicyNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=[256, 128],  # Smaller for testing
        activation='tanh',
        use_obs_norm=True
    )
    
    trainer = PPOTrainer(policy, learning_rate=3e-4, device=device)
    print(f"   ✓ Policy created with {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    test_obs = torch.randn(batch_size, obs_dim, device=device)
    action_mean, values = policy(test_obs)
    
    print(f"   ✓ Input observations: {test_obs.shape}")
    print(f"   ✓ Action mean: {action_mean.shape}")
    print(f"   ✓ Values: {values.shape}")
    print(f"   ✓ Action range: [{action_mean.min().item():.3f}, {action_mean.max().item():.3f}]")
    print(f"   ✓ Value range: [{values.min().item():.3f}, {values.max().item():.3f}]")
    
    # Test action sampling
    print("\n3. Testing action sampling...")
    actions, log_probs, sampled_values = policy.sample_actions(test_obs)
    
    print(f"   ✓ Sampled actions: {actions.shape}")
    print(f"   ✓ Log probabilities: {log_probs.shape}")
    print(f"   ✓ Action std: {torch.exp(policy.actor_logstd).mean().item():.3f}")
    
    # Test PPO training step
    print("\n4. Testing PPO training step...")
    
    # Create mock training data
    old_log_probs = log_probs.detach()
    advantages = torch.randn(batch_size, device=device)
    returns = values.detach() + advantages
    
    # Training step
    metrics = trainer.update_policy(test_obs, actions, old_log_probs, advantages, returns)
    
    print(f"   ✓ Policy loss: {metrics['policy_loss']:.4f}")
    print(f"   ✓ Value loss: {metrics['value_loss']:.4f}")
    print(f"   ✓ Entropy loss: {metrics['entropy_loss']:.4f}")
    print(f"   ✓ Total loss: {metrics['total_loss']:.4f}")
    print(f"   ✓ KL divergence: {metrics['approx_kl']:.4f}")
    
    # Test observation normalization
    print("\n5. Testing observation normalization...")
    initial_mean = policy.obs_norm.mean.clone()
    
    # Process different distribution
    different_obs = torch.randn(batch_size, obs_dim, device=device) * 2.0 + 1.0
    _ = policy(different_obs)
    
    updated_mean = policy.obs_norm.mean
    print(f"   ✓ Observation stats updated: {not torch.allclose(initial_mean, updated_mean)}")
    print(f"   ✓ Mean shift: {(updated_mean - initial_mean).abs().mean().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ SKELETON POLICY TEST SUCCESS!")
    print("✅ Ready to integrate with Genesis environment")
    print("✅ Next: Create training loop with AMP rewards")
    print("=" * 60)
    
    return trainer


if __name__ == "__main__":
    # Run policy test
    trainer = test_skeleton_policy()
    
    print("\n🎯 Next Steps:")
    print("1. Integrate policy with Genesis skeleton environment")
    print("2. Create episode collection system")
    print("3. Implement training loop with AMP discriminator")
    print("4. Test end-to-end imitation learning")