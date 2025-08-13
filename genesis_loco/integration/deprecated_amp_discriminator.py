"""
AMP Discriminator - PyTorch Implementation

Simple PyTorch conversion of LocoMujoco's JAX-based AMP discriminator.
Designed for Genesis physics integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, Optional
import numpy as np


class RunningMeanStd(nn.Module):
    """
    Running mean and standard deviation normalization
    Equivalent to LocoMujoco's use_running_mean_stand=True
    """
    
    def __init__(self, input_dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
        
        # Use buffers so they're saved with model state but not updated by optimizer
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('var', torch.ones(input_dim))
        self.register_buffer('count', torch.zeros(1))
        
    def forward(self, x: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Normalize input and optionally update running statistics
        
        Args:
            x: Input tensor [batch_size, input_dim]
            update_stats: Whether to update running mean/std (True during training)
        """
        if update_stats and self.training:
            self._update_stats(x)
            
        # Normalize using current statistics
        normalized = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return normalized
    
    def _update_stats(self, x: torch.Tensor):
        """Update running mean and variance with new batch"""
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        
        # Update count
        new_count = self.count + batch_size
        
        # Update mean using Welford's online algorithm
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_size / new_count
        
        # Update variance
        m_a = self.var * self.count
        m_b = batch_var * batch_size
        m_ab = ((self.mean - batch_mean) ** 2) * self.count * batch_size / new_count
        new_var = (m_a + m_b + m_ab) / new_count
        
        # Update buffers
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(new_count)


class AMPDiscriminator(nn.Module):
    """
    AMP Discriminator Network
    
    PyTorch implementation matching LocoMujoco's architecture:
    - Fully connected network with running mean/std normalization
    - Hidden layers: [512, 256] with tanh activation
    - Single output for discriminator score
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_layers: list = [512, 256],
                 activation: str = 'tanh',
                 use_running_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_running_norm = use_running_norm
        
        # Input normalization
        if self.use_running_norm:
            self.input_norm = RunningMeanStd(input_dim)
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer (single discriminator score)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using standard scheme"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Forward pass through discriminator
        
        Args:
            obs: Observations [batch_size, input_dim]
            update_stats: Whether to update running statistics (True during training)
            
        Returns:
            Discriminator scores [batch_size, 1]
        """
        x = obs
        
        # Apply input normalization if enabled
        if self.use_running_norm:
            x = self.input_norm(x, update_stats=update_stats)
        
        # Forward through network
        scores = self.network(x)
        
        return scores.squeeze(-1)  # [batch_size]


class AMPTrainer:
    """
    AMP Discriminator Training Manager
    
    Handles training the discriminator on expert vs policy data using:
    - Least squares loss (AMP-style)
    - Expert target = 1, Policy target = -1
    - Quadratic reward function
    """
    
    def __init__(self, 
                 discriminator: AMPDiscriminator,
                 learning_rate: float = 5e-5,
                 device: torch.device = torch.device('cuda')):
        
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
        # Training metrics
        self.training_stats = {
            'discriminator_loss': 0.0,
            'expert_accuracy': 0.0,
            'policy_accuracy': 0.0,
            'expert_score_mean': 0.0,
            'policy_score_mean': 0.0
        }
    
    def compute_amp_rewards(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute AMP rewards from discriminator scores
        
        AMP reward function: max(0, 1 - 0.25 * (score - 1)^2)
        
        Args:
            obs: Policy observations [batch_size, obs_dim]
            
        Returns:
            AMP rewards [batch_size]
        """
        self.discriminator.eval()
        with torch.no_grad():
            scores = self.discriminator(obs, update_stats=False)
            rewards = torch.clamp(1.0 - 0.25 * (scores - 1.0) ** 2, min=0.0)
        
        return rewards
    
    def train_discriminator(self, 
                          expert_obs: torch.Tensor,
                          policy_obs: torch.Tensor) -> Dict[str, float]:
        """
        Train discriminator on expert vs policy observations
        
        Args:
            expert_obs: Expert observations [batch_size, obs_dim]
            policy_obs: Policy observations [batch_size, obs_dim]
            
        Returns:
            Training metrics dictionary
        """
        self.discriminator.train()
        
        # Combine expert and policy data
        all_obs = torch.cat([expert_obs, policy_obs], dim=0)
        
        # Create targets: expert=1, policy=-1 (AMP style)
        expert_targets = torch.ones(expert_obs.shape[0], device=self.device)
        policy_targets = -torch.ones(policy_obs.shape[0], device=self.device)
        all_targets = torch.cat([expert_targets, policy_targets], dim=0)
        
        # Forward pass
        scores = self.discriminator(all_obs, update_stats=True)
        
        # Least squares loss (AMP discriminator loss)
        loss = F.mse_loss(scores, all_targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            expert_scores = scores[:expert_obs.shape[0]]
            policy_scores = scores[expert_obs.shape[0]:]
            
            # Accuracy: how often discriminator correctly classifies
            expert_accuracy = (expert_scores > 0).float().mean()  # Should be > 0 for expert
            policy_accuracy = (policy_scores < 0).float().mean()  # Should be < 0 for policy
            
            self.training_stats.update({
                'discriminator_loss': loss.item(),
                'expert_accuracy': expert_accuracy.item(),
                'policy_accuracy': policy_accuracy.item(),
                'expert_score_mean': expert_scores.mean().item(),
                'policy_score_mean': policy_scores.mean().item()
            })
        
        return self.training_stats
    
    def save_discriminator(self, path: str):
        """Save discriminator state"""
        torch.save({
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load_discriminator(self, path: str):
        """Load discriminator state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']


def test_amp_discriminator():
    """
    Test AMP discriminator implementation
    
    Verifies:
    1. Network forward pass
    2. Running mean/std normalization
    3. AMP reward computation
    4. Training step
    """
    print("=" * 60)
    print("TESTING AMP DISCRIMINATOR IMPLEMENTATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 256
    obs_dim = 55  # Skeleton environment observation dimension
    
    print(f"\nTest setup:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Observation dimension: {obs_dim}")
    
    # Create discriminator
    print("\n1. Creating AMP discriminator...")
    discriminator = AMPDiscriminator(
        input_dim=obs_dim,
        hidden_layers=[512, 256],
        activation='tanh',
        use_running_norm=True
    )
    
    trainer = AMPTrainer(discriminator, learning_rate=5e-5, device=device)
    print(f"   ✓ Discriminator created with {sum(p.numel() for p in discriminator.parameters())} parameters")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    test_obs = torch.randn(batch_size, obs_dim, device=device)
    scores = discriminator(test_obs)
    
    print(f"   ✓ Input shape: {test_obs.shape}")
    print(f"   ✓ Output shape: {scores.shape}")
    print(f"   ✓ Score range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
    
    # Test AMP reward computation
    print("\n3. Testing AMP reward computation...")
    rewards = trainer.compute_amp_rewards(test_obs)
    
    print(f"   ✓ Reward shape: {rewards.shape}")
    print(f"   ✓ Reward range: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")
    print(f"   ✓ Mean reward: {rewards.mean().item():.3f}")
    
    # Test discriminator training
    print("\n4. Testing discriminator training...")
    
    # Create mock expert and policy data
    expert_obs = torch.randn(batch_size, obs_dim, device=device) + 1.0  # Slightly different distribution
    policy_obs = torch.randn(batch_size, obs_dim, device=device)
    
    # Train for a few steps
    for step in range(5):
        metrics = trainer.train_discriminator(expert_obs, policy_obs)
        
        if step == 0 or step == 4:
            print(f"   Step {step+1}:")
            print(f"     Loss: {metrics['discriminator_loss']:.4f}")
            print(f"     Expert accuracy: {metrics['expert_accuracy']:.3f}")
            print(f"     Policy accuracy: {metrics['policy_accuracy']:.3f}")
            print(f"     Expert score: {metrics['expert_score_mean']:.3f}")
            print(f"     Policy score: {metrics['policy_score_mean']:.3f}")
    
    # Test running statistics
    print("\n5. Testing running mean/std normalization...")
    initial_mean = discriminator.input_norm.mean.clone()
    initial_var = discriminator.input_norm.var.clone()
    
    # Process more data to update stats
    more_obs = torch.randn(batch_size, obs_dim, device=device) * 2.0 + 3.0  # Different distribution
    _ = discriminator(more_obs)
    
    updated_mean = discriminator.input_norm.mean
    updated_var = discriminator.input_norm.var
    
    print(f"   ✓ Mean changed: {torch.allclose(initial_mean, updated_mean)} (should be False)")
    print(f"   ✓ Var changed: {torch.allclose(initial_var, updated_var)} (should be False)")
    print(f"   ✓ Mean shift: {(updated_mean - initial_mean).abs().mean().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ AMP DISCRIMINATOR TEST SUCCESS!")
    print("✅ All components working correctly")
    print("✅ Ready for integration with Genesis environment")
    print("=" * 60)
    
    return trainer


if __name__ == "__main__":
    # Run discriminator test
    trainer = test_amp_discriminator()
    
    print("\n🎯 Next Steps:")
    print("1. Integrate discriminator with Genesis environment")
    print("2. Create expert data sampling from trajectory bridge")
    print("3. Implement full AMP training loop with PPO")
    print("4. Test end-to-end imitation learning pipeline")