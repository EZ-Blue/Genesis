"""
Refactored AMP Discriminator - Simple and Efficient

Clean, minimal PyTorch implementation compatible with refactored integration.
Maintains core AMP functionality while removing unnecessary complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional


class RunningMeanStd(nn.Module):
    """
    Efficient running mean and standard deviation normalization
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


class AMPDiscriminator(nn.Module):
    """
    Simple AMP Discriminator Network
    
    Efficient implementation matching LocoMujoco's architecture with minimal overhead.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_layers: list = [512, 256],
                 activation: str = 'tanh',
                 use_running_norm: bool = True):
        super().__init__()
        
        self.use_running_norm = use_running_norm
        
        # Input normalization
        if use_running_norm:
            self.input_norm = RunningMeanStd(input_dim)
        
        # Build network efficiently
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh() if activation == 'tanh' else nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output layer
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


class AMPTrainer:
    """
    Simple AMP Discriminator Trainer
    
    Efficient training manager with minimal overhead.
    """
    
    def __init__(self, 
                 discriminator: AMPDiscriminator,
                 learning_rate: float = 5e-5,
                 device: torch.device = None):
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.discriminator = discriminator.to(device)
        self.device = device
        self.optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
        
        # Simple metrics tracking
        self.last_metrics = {}
    
    def compute_amp_rewards(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute AMP rewards from discriminator scores
        
        AMP reward: max(0, 1 - 0.25 * (score - 1)^2)
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
        Single discriminator training step
        
        Args:
            expert_obs: Expert observations [batch_size, obs_dim]
            policy_obs: Policy observations [batch_size, obs_dim]
            
        Returns:
            Training metrics
        """
        self.discriminator.train()
        
        # Combine data and create targets
        all_obs = torch.cat([expert_obs, policy_obs], dim=0)
        expert_targets = torch.ones(expert_obs.shape[0], device=self.device)
        policy_targets = -torch.ones(policy_obs.shape[0], device=self.device)
        all_targets = torch.cat([expert_targets, policy_targets], dim=0)
        
        # Forward pass
        scores = self.discriminator(all_obs, update_stats=True)
        
        # Loss and optimization
        loss = F.mse_loss(scores, all_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Compute simple metrics
        with torch.no_grad():
            expert_scores = scores[:expert_obs.shape[0]]
            policy_scores = scores[expert_obs.shape[0]:]
            
            self.last_metrics = {
                'discriminator_loss': loss.item(),
                'expert_accuracy': (expert_scores > 0).float().mean().item(),
                'policy_accuracy': (policy_scores < 0).float().mean().item(),
                'expert_score_mean': expert_scores.mean().item(),
                'policy_score_mean': policy_scores.mean().item()
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


def test_amp_discriminator():
    """Test refactored AMP discriminator"""
    print("🧪 Testing Refactored AMP Discriminator")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    obs_dim = 65  # Match skeleton environment
    batch_size = 32
    
    try:
        # Create discriminator
        print("1. Creating discriminator...")
        discriminator = AMPDiscriminator(
            input_dim=obs_dim,
            hidden_layers=[512, 256],
            activation='tanh',
            use_running_norm=True
        )
        print(f"   ✅ Discriminator created: {sum(p.numel() for p in discriminator.parameters())} parameters")
        
        # Create trainer
        print("2. Creating trainer...")
        trainer = AMPTrainer(discriminator, learning_rate=5e-5, device=device)
        print("   ✅ Trainer created")
        
        # Test forward pass
        print("3. Testing forward pass...")
        expert_obs = torch.randn(batch_size, obs_dim, device=device)
        policy_obs = torch.randn(batch_size, obs_dim, device=device)
        
        scores = discriminator(expert_obs)
        print(f"   ✅ Forward pass: output shape {scores.shape}")
        
        # Test AMP rewards
        print("4. Testing AMP rewards...")
        rewards = trainer.compute_amp_rewards(policy_obs)
        print(f"   ✅ AMP rewards: mean={rewards.mean().item():.4f}, range=[{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
        
        # Test training step
        print("5. Testing training step...")
        metrics = trainer.train_discriminator(expert_obs, policy_obs)
        print(f"   ✅ Training step completed:")
        for key, value in metrics.items():
            print(f"      {key}: {value:.4f}")
        
        # Test save/load
        print("6. Testing save/load...")
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        trainer.save_discriminator(temp_path)
        trainer.load_discriminator(temp_path)
        os.unlink(temp_path)
        print("   ✅ Save/load successful")
        
        print("\n🎉 All discriminator tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_amp_discriminator()