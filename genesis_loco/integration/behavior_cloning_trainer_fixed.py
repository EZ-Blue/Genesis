#!/usr/bin/env python3
"""
FIXED Behavior Cloning Trainer for Genesis Skeleton Humanoid

Addresses critical issues identified in diagnostic:
1. Action range collapse - added action scaling and proper loss function
2. Over-smoothing - added action variation regularization
3. Model overfitting - reduced capacity and added stronger regularization
4. Data sampling - improved expert data utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Tuple, Dict, List
from datetime import datetime

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
import genesis as gs


def safe_init_genesis():
    """Safely initialize Genesis"""
    try:
        gs.init(backend=gs.gpu)
        return True, "Genesis initialized"
    except Exception as e:
        if "already initialized" in str(e):
            return True, "Genesis already initialized"
        else:
            return False, f"Genesis initialization failed: {e}"


class FixedBehaviorCloningMLP(nn.Module):
    """
    FIXED MLP for behavior cloning addressing action range collapse
    - Reduced capacity to prevent overfitting
    - Better initialization for action prediction
    - Action scaling layers
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [128, 64],  # REDUCED capacity
                 action_std_init: float = 0.5):  # Action scaling
        super().__init__()
        
        self.action_dim = action_dim
        
        layers = []
        prev_dim = obs_dim
        
        # Smaller hidden layers to prevent overfitting
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())  # ELU can help with range issues
            layers.append(nn.Dropout(0.2))  # Stronger dropout
            prev_dim = hidden_dim
        
        # Output layer with special initialization
        output_layer = nn.Linear(prev_dim, action_dim)
        
        self.network = nn.Sequential(*layers)
        self.output_layer = output_layer
        
        # Action scaling parameters (learnable)
        self.action_scale = nn.Parameter(torch.ones(action_dim) * action_std_init)
        self.action_bias = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights for better action prediction
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to encourage proper action ranges"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier normal for better gradient flow
                nn.init.xavier_normal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        
        # Special initialization for output layer
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)  # Smaller gain
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass with action scaling"""
        hidden = self.network(obs)
        raw_actions = self.output_layer(hidden)
        
        # Apply learnable scaling and bias
        scaled_actions = raw_actions * torch.abs(self.action_scale) + self.action_bias
        
        return scaled_actions


class FixedBehaviorCloningTrainer:
    """
    FIXED Trainer addressing all identified issues
    """
    
    def __init__(self, behavior: str = "walk", device: str = "auto"):
        self.behavior = behavior
        
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🤖 FIXED Behavior Cloning Trainer for {behavior.upper()}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        # Initialize components
        self._setup_genesis()
        self._setup_environment()
        self._setup_data_bridge()
        self._analyze_expert_data()  # NEW: Analyze data before model setup
        self._setup_model()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.action_range_losses = []  # NEW: Track action range loss
        
    def _setup_genesis(self):
        """Initialize Genesis physics"""
        success, message = safe_init_genesis()
        if not success:
            raise RuntimeError(message)
        print(f"✅ {message}")
    
    def _setup_environment(self):
        """Setup Genesis skeleton environment"""
        self.env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=30.0,
            dt=0.01,
            show_viewer=False,
            use_box_feet=True,
            obs_history_length=3
        )
        
        self.obs_dim = self.env.num_observations
        self.action_dim = self.env.num_actions
        
        print(f"✅ Environment: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
    
    def _setup_data_bridge(self):
        """Setup LocoMujoco data bridge"""
        self.data_bridge = LocoMujocoDataBridge(self.env)
        success = self.data_bridge.load_trajectory(self.behavior)
        if not success:
            raise RuntimeError(f"Failed to load {self.behavior} trajectory")
        
        self.trajectory_length = self.data_bridge.trajectory_length
        print(f"✅ Expert trajectory: {self.trajectory_length} timesteps")
    
    def _analyze_expert_data(self):
        """Analyze expert data to inform model design"""
        print("🔍 Analyzing expert data for model design...")
        
        # Sample expert actions to compute statistics
        expert_actions = []
        timesteps = np.linspace(0, self.trajectory_length-1, 1000, dtype=int)
        
        for timestep in timesteps:
            state = self.data_bridge.get_trajectory_state(timestep)
            if state and 'dof_pos' in state:
                dof_pos = state['dof_pos']
                if hasattr(dof_pos, 'cpu'):
                    dof_pos = dof_pos.cpu().numpy()
                expert_actions.append(dof_pos)
        
        expert_actions = np.array(expert_actions)
        
        # Compute action statistics for model initialization
        self.action_mean = torch.tensor(np.mean(expert_actions, axis=0), 
                                      dtype=torch.float32, device=self.device)
        self.action_std = torch.tensor(np.std(expert_actions, axis=0), 
                                     dtype=torch.float32, device=self.device)
        self.action_range = torch.tensor(np.max(expert_actions, axis=0) - np.min(expert_actions, axis=0),
                                       dtype=torch.float32, device=self.device)
        
        print(f"   📊 Expert action statistics:")
        print(f"      Mean: {self.action_mean.mean().item():.6f}")
        print(f"      Std: {self.action_std.mean().item():.6f}")
        print(f"      Range: {self.action_range.mean().item():.6f}")
        
        # Identify active vs inactive joints
        self.active_joints = self.action_std > 0.01  # Joints with significant movement
        print(f"   📊 Active joints: {self.active_joints.sum().item()}/{len(self.active_joints)}")
    
    def _setup_model(self):
        """Setup FIXED behavior cloning model"""
        # Use expert data statistics for better initialization
        action_std_init = self.action_std.mean().item()
        
        self.model = FixedBehaviorCloningMLP(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=[128, 64],  # REDUCED capacity
            action_std_init=max(0.1, action_std_init)  # Prevent too small initialization
        ).to(self.device)
        
        # Initialize action scaling with expert statistics
        with torch.no_grad():
            self.model.action_scale.data = self.action_std + 0.01  # Prevent zeros
            self.model.action_bias.data = self.action_mean
        
        # Optimizer with stronger regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=5e-4,  # REDUCED learning rate
            weight_decay=1e-3  # INCREASED weight decay
        )
        
        # More aggressive learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.7
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        param_ratio = total_params / self.action_dim
        print(f"✅ FIXED model: {total_params:,} parameters (ratio: {param_ratio:.1f})")
    
    def _generate_training_data(self, num_samples: int = 20000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate training data with FIXED sampling strategy"""
        print(f"📊 Generating {num_samples:,} training samples with FIXED sampling...")
        
        # Use ALL physics-based expert observations (no subsampling)
        expert_observations = self.data_bridge.get_expert_observations_cached(
            dataset_name=self.behavior,
            num_timesteps=None,  # Use ALL timesteps
            start_timestep=0,
            step_interval=1,  # FIXED: Every timestep, not sparse sampling
            force_reload=False
        )
        
        if expert_observations is None:
            raise RuntimeError("Failed to load physics-based expert observations!")
        
        print(f"✅ Loaded {expert_observations.shape[0]} physics-based expert observations")
        
        # Generate corresponding target positions from ALL trajectory states
        observations = []
        actions = []
        
        print(f"📊 Collecting target positions from full trajectory...")
        
        # Use sequential sampling for better temporal coverage
        total_available = min(expert_observations.shape[0], self.trajectory_length)
        if num_samples > total_available:
            print(f"⚠️  Requested {num_samples} samples but only {total_available} available")
            num_samples = total_available
        
        # Sample with better distribution
        if num_samples < total_available:
            # Stratified sampling across trajectory
            indices = np.linspace(0, total_available-1, num_samples, dtype=int)
        else:
            # Use all available
            indices = np.arange(total_available)
        
        for i, idx in enumerate(indices):
            if i % 2000 == 0:
                print(f"   Progress: {i:,}/{num_samples:,} ({100*i/num_samples:.1f}%)")
            
            # Get expert state for target positions
            current_state = self.data_bridge.get_trajectory_state(idx)
            if current_state is None:
                continue
            
            try:
                # Extract target joint positions
                target_positions = self._extract_target_positions(current_state)
                if target_positions is None:
                    continue
                
                # Use physics-based observation
                obs = expert_observations[idx].cpu().numpy()
                
                observations.append(obs)
                actions.append(target_positions)
                
            except Exception as e:
                if i < 10:
                    print(f"   Debug error at index {idx}: {e}")
                continue
        
        if len(observations) == 0:
            raise RuntimeError("No valid training samples generated!")
        
        # Convert to tensors
        obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        pos_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        
        print(f"✅ Generated {len(observations):,} valid samples")
        print(f"   Observation shape: {obs_tensor.shape}")
        print(f"   Target position shape: {pos_tensor.shape}")
        
        # Analyze generated action distribution
        action_ranges = pos_tensor.max(dim=0)[0] - pos_tensor.min(dim=0)[0]
        print(f"   📊 Generated action ranges - mean: {action_ranges.mean().item():.6f}")
        
        return obs_tensor, pos_tensor
    
    def _extract_target_positions(self, current_state: Dict) -> np.ndarray:
        """Extract expert target joint positions"""
        if current_state is None:
            return None
            
        try:
            def to_numpy(data):
                if hasattr(data, 'cpu'):
                    return data.cpu().numpy()
                else:
                    return np.array(data)
            
            if 'dof_pos' in current_state:
                dof_pos = to_numpy(current_state['dof_pos'])
                target_positions = dof_pos
            else:
                return None
            
            return target_positions.astype(np.float32)
            
        except Exception as e:
            print(f"   Error in _extract_target_positions: {e}")
            return None
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """FIXED loss function addressing action range collapse"""
        
        # 1. Primary MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # 2. Action range preservation loss (CRITICAL FIX)
        pred_ranges = predictions.max(dim=0)[0] - predictions.min(dim=0)[0]
        target_ranges = targets.max(dim=0)[0] - targets.min(dim=0)[0]
        
        # Penalize range collapse - encourage predictions to use full range
        range_loss = F.mse_loss(pred_ranges, target_ranges)
        
        # 3. Active joint focus (weight active joints more)
        active_weight = self.active_joints.float()
        weighted_mse = torch.mean(((predictions - targets) ** 2) * active_weight.unsqueeze(0))
        
        # 4. Action smoothness regularization (but not too much!)
        if predictions.shape[0] > 1:
            pred_diff = torch.diff(predictions, dim=0)
            target_diff = torch.diff(targets, dim=0)
            smoothness_loss = F.mse_loss(pred_diff, target_diff)
        else:
            smoothness_loss = torch.tensor(0.0, device=self.device)
        
        # Combine losses with weights
        total_loss = (mse_loss + 
                     2.0 * range_loss +  # High weight on range preservation
                     1.5 * weighted_mse + 
                     0.1 * smoothness_loss)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'range_loss': range_loss,
            'weighted_mse': weighted_mse,
            'smoothness_loss': smoothness_loss
        }
    
    def train(self, num_epochs: int = 1000, batch_size: int = 512, 
              train_samples: int = 20000, val_samples: int = 4000):
        """Train the FIXED behavior cloning model"""
        
        print(f"\n🎯 Starting FIXED Behavior Cloning Training")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Training samples: {train_samples:,}")
        print(f"   Validation samples: {val_samples:,}")
        print("=" * 60)
        
        # Generate training and validation data
        train_obs, train_actions = self._generate_training_data(train_samples)
        val_obs, val_actions = self._generate_training_data(val_samples)
        
        # Training setup
        best_val_loss = float('inf')
        patience = 50  # REDUCED patience
        epochs_no_improve = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            train_range_losses = []
            
            # Shuffle training data
            indices = torch.randperm(train_obs.size(0))
            train_obs_shuffled = train_obs[indices]
            train_actions_shuffled = train_actions[indices]
            
            # Mini-batch training
            for i in range(0, train_obs.size(0), batch_size):
                batch_obs = train_obs_shuffled[i:i+batch_size]
                batch_actions = train_actions_shuffled[i:i+batch_size]
                
                # Forward pass
                predicted_actions = self.model(batch_obs)
                
                # FIXED loss computation
                loss_dict = self._compute_loss(predicted_actions, batch_actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                # Gradient clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_losses.append(loss_dict['total_loss'].item())
                train_range_losses.append(loss_dict['range_loss'].item())
            
            avg_train_loss = np.mean(train_losses)
            avg_range_loss = np.mean(train_range_losses)
            
            self.train_losses.append(avg_train_loss)
            self.action_range_losses.append(avg_range_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_obs)
                val_loss_dict = self._compute_loss(val_predictions, val_actions)
                val_loss = val_loss_dict['total_loss'].item()
                self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Progress logging with FIXED metrics
            if epoch % 5 == 0 or epoch < 10:
                elapsed = time.time() - start_time
                
                # Compute action range metrics
                with torch.no_grad():
                    pred_ranges = val_predictions.max(dim=0)[0] - val_predictions.min(dim=0)[0]
                    target_ranges = val_actions.max(dim=0)[0] - val_actions.min(dim=0)[0]
                    range_ratio = (pred_ranges.mean() / (target_ranges.mean() + 1e-8)).item()
                
                print(f"Epoch {epoch:4d}/{num_epochs}: "
                      f"Train: {avg_train_loss:.6f}, "
                      f"Val: {val_loss:.6f}, "
                      f"Range: {avg_range_loss:.6f}, "
                      f"Ratio: {range_ratio:.3f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping with improvement tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                self._save_model("best_model_fixed")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n⏰ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break
        
        total_time = time.time() - start_time
        print(f"\n✅ FIXED training completed in {total_time:.1f}s")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        
        # Plot training curves
        self._plot_training_curves()
        
        # Save final model
        self._save_model("final_model_fixed")
        
        # FINAL DIAGNOSTIC
        self._final_diagnostic(val_obs, val_actions)
    
    def _final_diagnostic(self, val_obs: torch.Tensor, val_actions: torch.Tensor):
        """Final diagnostic of trained model"""
        print(f"\n🔍 FINAL MODEL DIAGNOSTIC")
        print("=" * 40)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(val_obs)
            
            # Action range analysis
            pred_ranges = predictions.max(dim=0)[0] - predictions.min(dim=0)[0]
            target_ranges = val_actions.max(dim=0)[0] - val_actions.min(dim=0)[0]
            
            print(f"📊 Action range comparison:")
            print(f"   Target range mean: {target_ranges.mean().item():.6f}")
            print(f"   Predicted range mean: {pred_ranges.mean().item():.6f}")
            print(f"   Range ratio: {(pred_ranges.mean() / (target_ranges.mean() + 1e-8)).item():.3f}")
            
            # Check for collapsed joints
            collapsed_joints = (pred_ranges < 0.01).sum().item()
            print(f"   Collapsed joints (<0.01 range): {collapsed_joints}/{len(pred_ranges)}")
            
            if collapsed_joints < len(pred_ranges) * 0.5:
                print("   ✅ Action range collapse FIXED!")
            else:
                print("   ⚠️  Some action range issues remain")
    
    def _save_model(self, name: str):
        """Save model weights"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"behavior_cloning_{self.behavior}_{name}_{timestamp}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'behavior': self.behavior,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'action_range': self.action_range
        }, save_path)
        print(f"💾 FIXED model saved: {save_path}")
    
    def _plot_training_curves(self):
        """Plot FIXED training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0,0].plot(self.train_losses, label='Training Loss', alpha=0.7)
        axes[0,0].plot(self.val_losses, label='Validation Loss', alpha=0.7)
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Total Loss')
        axes[0,0].set_title('Training Progress')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')
        
        # Range loss (NEW)
        axes[0,1].plot(self.action_range_losses, color='red', alpha=0.7)
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Range Loss')
        axes[0,1].set_title('Action Range Preservation')
        axes[0,1].grid(True, alpha=0.3)
        
        # Learning rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lr_history = [group['lr'] for group in self.optimizer.param_groups]
        else:
            lr_history = [self.optimizer.param_groups[0]['lr']] * len(self.train_losses)
        
        axes[1,0].plot(lr_history[:len(self.train_losses)], color='green', alpha=0.7)
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Learning Rate')
        axes[1,0].set_title('Learning Rate Schedule')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')
        
        # Loss components (placeholder)
        axes[1,1].text(0.5, 0.5, 'Model Fixed:\n✅ Action Range\n✅ Reduced Capacity\n✅ Better Sampling', 
                      ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('FIXES Applied')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"behavior_cloning_{self.behavior}_FIXED_curves_{timestamp}.png", dpi=150)
        print(f"📊 FIXED training curves saved")
        plt.show()


def main():
    """Main function for FIXED trainer"""
    print("🤖 FIXED Genesis Behavior Cloning Trainer")
    print("=" * 50)
    print("FIXES Applied:")
    print("✅ Action range collapse prevention")
    print("✅ Reduced model capacity")  
    print("✅ Better data sampling")
    print("✅ Action range preservation loss")
    print("✅ Stronger regularization")
    
    try:
        trainer = FixedBehaviorCloningTrainer(behavior="walk")
        trainer.train(
            num_epochs=1000,
            train_samples=20000,
            val_samples=4000
        )
        
        print(f"\n🎉 FIXED behavior cloning training complete!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()