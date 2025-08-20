"""
GAIL Discriminator - PyTorch Implementation adapted from LocoMujoco

PyTorch implementation of GAIL discriminator compatible with Genesis environments.
Follows LocoMujoco's discriminator architecture and training approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional
import numpy as np


class RunningMeanStd(nn.Module):
    """
    Running mean and standard deviation normalization
    Adapted from LocoMujoco's normalization approach
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
        """Update running statistics using efficient online algorithm"""
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


class GAILDiscriminator(nn.Module):
    """
    GAIL Discriminator Network
    
    Adapted from LocoMujoco's FullyConnectedNet discriminator architecture.
    Uses binary classification to distinguish expert from policy observations.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_layers: list = [512, 256],
                 activation: str = 'tanh',
                 use_running_norm: bool = True):
        super().__init__()
        
        self.use_running_norm = use_running_norm
        
        # Input normalization (following LocoMujoco)
        if use_running_norm:
            self.input_norm = RunningMeanStd(input_dim)
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh() if activation == 'tanh' else nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output layer - single logit for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """Forward pass through discriminator"""
        x = obs
        
        # Apply normalization if enabled
        if self.use_running_norm:
            x = self.input_norm(x, update_stats=update_stats)
        
        # Forward through network
        return self.network(x).squeeze(-1)


class GAILTrainer:
    """
    GAIL Discriminator Trainer
    
    Handles discriminator training following LocoMujoco's approach:
    - Binary cross-entropy loss
    - Bernoulli entropy regularization
    - Expert=1, Policy=0 labeling
    """
    
    def __init__(self, 
                 discriminator: GAILDiscriminator,
                 learning_rate: float = 5e-5,
                 entropy_coef: float = 0.0,
                 device: torch.device = None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.discriminator = discriminator.to(device)
        self.device = device
        self.entropy_coef = entropy_coef
        self.optimizer = optim.AdamW(discriminator.parameters(), lr=learning_rate, eps=1e-5)
        self.max_grad_norm = learning_rate  # Use learning rate for grad clipping like LocoMujoco
        
        # Metrics tracking
        self.last_metrics = {}
    
    def compute_gail_rewards(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute GAIL rewards from discriminator scores
        
        Following LocoMujoco: reward = -log(1 - sigmoid(logits) + eps)
        Higher reward when discriminator thinks observation is from expert
        """
        self.discriminator.eval()
        with torch.no_grad():
            logits = self.discriminator(obs, update_stats=False)
            # Convert logits to probabilities (probability of being expert)
            probs = torch.sigmoid(logits)
            # GAIL reward: -log(1 - p_expert + eps)
            # Higher when discriminator thinks it's expert-like
            rewards = -torch.log(1 - probs + 1e-6)
        return rewards
    
    def train_discriminator(self, 
                          expert_obs: torch.Tensor,
                          policy_obs: torch.Tensor) -> Dict[str, float]:
        """
        Single discriminator training step
        
        Following LocoMujoco's approach:
        - Expert observations labeled as 1
        - Policy observations labeled as 0
        - Binary cross-entropy loss with entropy regularization
        
        Args:
            expert_obs: Expert observations [batch_size, obs_dim]
            policy_obs: Policy observations [batch_size, obs_dim]
            
        Returns:
            Training metrics
        """
        self.discriminator.train()
        
        # Combine data and create targets (following LocoMujoco)
        all_obs = torch.cat([policy_obs, expert_obs], dim=0)  # Policy first, then expert
        policy_targets = torch.zeros(policy_obs.shape[0], device=self.device)  # Policy = 0
        expert_targets = torch.ones(expert_obs.shape[0], device=self.device)   # Expert = 1
        all_targets = torch.cat([policy_targets, expert_targets], dim=0)
        
        # Forward pass
        logits = self.discriminator(all_obs, update_stats=True)
        
        # Binary cross-entropy loss (following LocoMujoco)
        log_p = F.logsigmoid(logits)
        log_not_p = F.logsigmoid(-logits)
        bce_loss = torch.mean(-all_targets * log_p - (1. - all_targets) * log_not_p)
        
        # Bernoulli entropy regularization (following LocoMujoco)
        if self.entropy_coef > 0:
            discrim_prob = torch.sigmoid(logits)
            bernoulli_ent = self.entropy_coef * torch.mean(
                (1. - discrim_prob) * logits - F.logsigmoid(logits)
            )
            total_loss = bce_loss - bernoulli_ent
        else:
            total_loss = bce_loss
        
        # Optimization step with gradient clipping (following LocoMujoco)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            policy_logits = logits[:policy_obs.shape[0]]
            expert_logits = logits[policy_obs.shape[0]:]
            
            policy_probs = torch.sigmoid(policy_logits)
            expert_probs = torch.sigmoid(expert_logits)
            
            self.last_metrics = {
                'discriminator_loss': total_loss.item(),
                'policy_accuracy': (policy_probs < 0.5).float().mean().item(),  # Should be classified as 0
                'expert_accuracy': (expert_probs > 0.5).float().mean().item(),  # Should be classified as 1
                'discriminator_output_policy': policy_probs.mean().item(),
                'discriminator_output_expert': expert_probs.mean().item()
            }
        
        return self.last_metrics
    
    def save_discriminator(self, path: str):
        """Save discriminator state"""
        torch.save({
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': self.last_metrics
        }, path)
    
    def load_discriminator(self, path: str):
        """Load discriminator state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.last_metrics = checkpoint.get('metrics', {})


def test_gail_discriminator():
    """Test GAIL discriminator implementation"""
    print("🧪 Testing GAIL Discriminator")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters matching skeleton environment
    obs_dim = 65
    batch_size = 32
    
    try:
        # Create discriminator
        print("1. Creating GAIL discriminator...")
        discriminator = GAILDiscriminator(
            input_dim=obs_dim,
            hidden_layers=[512, 256],
            activation='tanh',
            use_running_norm=True
        )
        print(f"   ✅ Discriminator created: {sum(p.numel() for p in discriminator.parameters())} parameters")
        
        # Create trainer
        print("2. Creating GAIL trainer...")
        trainer = GAILTrainer(discriminator, learning_rate=5e-5, entropy_coef=0.0, device=device)
        print("   ✅ Trainer created")
        
        # Test forward pass
        print("3. Testing forward pass...")
        expert_obs = torch.randn(batch_size, obs_dim, device=device)
        policy_obs = torch.randn(batch_size, obs_dim, device=device)
        
        logits = discriminator(expert_obs)
        print(f"   ✅ Forward pass: output shape {logits.shape}")
        
        # Test GAIL rewards
        print("4. Testing GAIL rewards...")
        rewards = trainer.compute_gail_rewards(policy_obs)
        print(f"   ✅ GAIL rewards: mean={rewards.mean().item():.4f}, range=[{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
        
        # Test training step
        print("5. Testing training step...")
        metrics = trainer.train_discriminator(expert_obs, policy_obs)
        print(f"   ✅ Training step completed:")
        for key, value in metrics.items():
            print(f"      {key}: {value:.4f}")
        
        print("\n🎉 All GAIL discriminator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_gail_discriminator()