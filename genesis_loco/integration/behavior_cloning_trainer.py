#!/usr/bin/env python3
"""
Simple Behavior Cloning Trainer for Genesis Skeleton Humanoid

Direct imitation learning approach that trains an MLP to predict expert actions
given current observations. Much simpler than PPO+AMP and often more effective
for locomotion tasks.
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


class BehaviorCloningMLP(nn.Module):
    """
    Simple MLP for behavior cloning with position control
    Maps observations -> target joint positions (for PD controller)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Prevent overfitting
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for locomotion
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better locomotion learning"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass: obs -> predicted target joint positions"""
        return self.network(obs)


class BehaviorCloningTrainer:
    """
    Trains a policy using behavior cloning on expert trajectory data
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
        
        print(f"🤖 Behavior Cloning Trainer for {behavior.upper()}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        # Initialize components
        self._setup_genesis()
        self._setup_environment()
        self._setup_data_bridge()
        self._setup_model()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        
    def _setup_genesis(self):
        """Initialize Genesis physics"""
        success, message = safe_init_genesis()
        if not success:
            raise RuntimeError(message)
        print(f"✅ {message}")
    
    def _setup_environment(self):
        """Setup Genesis skeleton environment with observation history for temporal context"""
        self.env = SkeletonHumanoidEnv(
            num_envs=1,  # Single environment for data collection
            episode_length_s=30.0,
            dt=0.01,
            show_viewer=False,
            use_box_feet=True,
            obs_history_length=3  # Add temporal context like LocoMujoco
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
        
        # CRITICAL: Get Genesis DOF mapping for controlled joints
        self.motors_dof_idx = self.env.motors_dof_idx  # Controlled DOF indices in Genesis
        self.joint_to_motor_idx = self.env.joint_to_motor_idx  # Joint name -> DOF index mapping
        
        # Convert to numpy for consistent indexing in data processing
        if hasattr(self.motors_dof_idx, 'cpu'):
            self.motors_dof_idx_np = self.motors_dof_idx.cpu().numpy()
        else:
            self.motors_dof_idx_np = np.array(self.motors_dof_idx)
        
        print(f"✅ Expert trajectory: {self.trajectory_length} timesteps")
        print(f"   Controlled DOFs: {len(self.motors_dof_idx)} joints at indices {self.motors_dof_idx[:5]}...")
        print(f"   Joint mapping available: {len(self.joint_to_motor_idx)} joint names")
        print(f"   Motor DOF indices (numpy): {self.motors_dof_idx_np[:5]}... (showing first 5)")
        
        # Debug: Test a sample state conversion
        sample_state = self.data_bridge.get_trajectory_state(0)
        if sample_state and 'dof_pos' in sample_state:
            sample_dof_pos = sample_state['dof_pos']
            if hasattr(sample_dof_pos, 'cpu'):
                sample_dof_pos = sample_dof_pos.cpu().numpy()
            print(f"   Debug: Sample DOF array shape: {sample_dof_pos.shape}")
            
            # Updated: data_bridge now returns data already in motor DOF order
            # No need to extract with motors_dof_idx - the data is already in the right order
            print(f"   Debug: DOF data already in motor order, shape: {sample_dof_pos.shape}")
            print(f"   Debug: Expected action dim: {self.action_dim}")
            assert sample_dof_pos.shape[0] == self.action_dim, \
                f"Motor DOF data mismatch: {sample_dof_pos.shape[0]} != {self.action_dim}"
    
    def _setup_model(self):
        """Setup behavior cloning model"""
        self.model = BehaviorCloningMLP(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=[256, 128]  # Moderate size for efficiency
        ).to(self.device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=20, factor=0.5
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Behavior cloning model: {total_params:,} parameters")
    
    def _generate_training_data(self, num_samples: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate observation-action pairs from expert trajectory"""
        print(f"📊 Generating {num_samples:,} training samples...")
        
        observations = []
        actions = []
        
        # Sample random timesteps from trajectory
        timesteps = np.random.randint(0, self.trajectory_length - 10, num_samples)
        
        # Debug: Check first sample to understand data format
        first_state = self.data_bridge.get_trajectory_state(0)
        print(f"   Debug: First state keys: {list(first_state.keys()) if first_state else 'None'}")
        if first_state:
            for key, value in first_state.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"   Debug: {key} shape: {np.array(value).shape}")
        
        for i, timestep in enumerate(timesteps):
            if i % 1000 == 0:
                print(f"   Progress: {i:,}/{num_samples:,} ({100*i/num_samples:.1f}%)")
            
            # Get expert state at this timestep
            current_state = self.data_bridge.get_trajectory_state(timestep)
            if current_state is None:
                continue
            
            # Get next state for action (what expert did)
            next_state = self.data_bridge.get_trajectory_state(timestep + 1)
            if next_state is None:
                continue
            
            # Convert states to observations and actions
            try:
                # Create observation from current state
                current_obs = self._state_to_observation(current_state)
                if current_obs is None:
                    continue
                
                # Extract target joint positions from current state
                target_positions = self._extract_target_positions(current_state)
                if target_positions is None:
                    continue
                
                observations.append(current_obs)
                actions.append(target_positions)  # Now contains target positions, not actions
                
            except Exception as e:
                if i < 10:  # Only print first few errors for debugging
                    print(f"   Debug error at timestep {timestep}: {e}")
                continue
        
        if len(observations) == 0:
            print("❌ No valid samples found. Checking data bridge format...")
            # Additional debugging
            for i in range(min(5, self.trajectory_length)):
                state = self.data_bridge.get_trajectory_state(i)
                print(f"   Sample {i}: {state}")
            raise RuntimeError("No valid training samples generated!")
        
        # Convert to tensors
        obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
        pos_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        
        print(f"✅ Generated {len(observations):,} valid samples")
        print(f"   Observation shape: {obs_tensor.shape}")
        print(f"   Target position shape: {pos_tensor.shape}")
        
        return obs_tensor, pos_tensor
    
    def _state_to_observation(self, state: Dict) -> np.ndarray:
        """Convert trajectory state to environment observation format"""
        if state is None:
            return None
            
        try:
            obs_parts = []
            
            # Extract available data based on actual keys in state
            available_keys = state.keys()
            
            # Helper function to safely convert tensors to numpy
            def to_numpy(data):
                if hasattr(data, 'cpu'):  # PyTorch tensor
                    return data.cpu().numpy()
                else:  # Already numpy or list
                    return np.array(data)
            
            # Root z position (height) - key for stability
            if 'root_pos' in state:
                root_pos = to_numpy(state['root_pos'])
                obs_parts.append([root_pos[2]])  # Just height for now
            else:
                obs_parts.append([0.975])  # Default height
            
            # Root orientation (quaternion)
            if 'root_quat' in state:
                root_quat = to_numpy(state['root_quat'])
                obs_parts.append(root_quat[:4])
            else:
                obs_parts.append([1.0, 0.0, 0.0, 0.0])  # Default quaternion
            
            # Joint positions (controlled joints only) - UPDATED: data_bridge already returns motor DOF order
            if 'dof_pos' in state:
                dof_pos = to_numpy(state['dof_pos'])
                # data_bridge now returns data already in motor DOF order, no extraction needed
                controlled_joint_pos = dof_pos  # Already in correct order
                obs_parts.append(controlled_joint_pos)
            else:
                obs_parts.append(np.zeros(self.action_dim))
            
            # Root velocities
            if 'root_lin_vel' in state:
                root_vel = to_numpy(state['root_lin_vel'])
                obs_parts.append(root_vel[:3])
            else:
                obs_parts.append([0.0, 0.0, 0.0])
            
            if 'root_ang_vel' in state:
                root_ang_vel = to_numpy(state['root_ang_vel'])
                obs_parts.append(root_ang_vel[:3])
            else:
                obs_parts.append([0.0, 0.0, 0.0])
            
            # Joint velocities (controlled joints only) - UPDATED: data_bridge already returns motor DOF order
            if 'dof_vel' in state:
                dof_vel = to_numpy(state['dof_vel'])
                # data_bridge now returns data already in motor DOF order, no extraction needed
                controlled_joint_vel = dof_vel  # Already in correct order
                obs_parts.append(controlled_joint_vel)
            else:
                obs_parts.append(np.zeros(self.action_dim))
            
            # Flatten all parts
            obs = np.concatenate([np.array(part).flatten() for part in obs_parts])
            
            # Ensure correct observation size
            if len(obs) < self.obs_dim:
                obs = np.pad(obs, (0, self.obs_dim - len(obs)))
            elif len(obs) > self.obs_dim:
                obs = obs[:self.obs_dim]
            
            return obs.astype(np.float32)
            
        except Exception as e:
            print(f"   Error in _state_to_observation: {e}")
            return None
    
    def _extract_target_positions(self, current_state: Dict) -> np.ndarray:
        """Extract expert target joint positions (what we want to achieve)"""
        if current_state is None:
            return None
            
        try:
            # Helper function to safely convert tensors to numpy
            def to_numpy(data):
                if hasattr(data, 'cpu'):  # PyTorch tensor
                    return data.cpu().numpy()
                else:  # Already numpy or list
                    return np.array(data)
            
            # Extract controlled joint positions directly from expert trajectory
            # UPDATED: data_bridge now returns data already in motor DOF order
            if 'dof_pos' in current_state:
                dof_pos = to_numpy(current_state['dof_pos'])
                # data_bridge now returns data already in motor DOF order, no extraction needed
                target_positions = dof_pos  # Already in correct order
            else:
                return None
            
            # No clipping - let the PD controller handle the range
            return target_positions.astype(np.float32)
            
        except Exception as e:
            print(f"   Error in _extract_target_positions: {e}")
            return None
    
    def train(self, num_epochs: int = 500, batch_size: int = 256, 
              train_samples: int = 20000, val_samples: int = 2000):
        """Train the behavior cloning model"""
        
        print(f"\n🎯 Starting Behavior Cloning Training")
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
        patience = 200
        epochs_no_improve = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            # Shuffle training data
            indices = torch.randperm(train_obs.size(0))
            train_obs_shuffled = train_obs[indices]
            train_actions_shuffled = train_actions[indices]
            
            # Mini-batch training
            for i in range(0, train_obs.size(0), batch_size):
                batch_obs = train_obs_shuffled[i:i+batch_size]
                batch_actions = train_actions_shuffled[i:i+batch_size]
                
                # Forward pass
                predicted_positions = self.model(batch_obs)
                loss = F.mse_loss(predicted_positions, batch_actions)  # batch_actions now contains target positions
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_obs)
                val_loss = F.mse_loss(val_predictions, val_actions).item()
                self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Progress logging
            if epoch % 10 == 0 or epoch < 10:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d}/{num_epochs}: "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                      f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model
                self._save_model("best_model")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n⏰ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.1f}s")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        
        # Plot training curves
        self._plot_training_curves()
        
        # Save final model
        self._save_model("final_model")
    
    def _save_model(self, name: str):
        """Save model weights"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"behavior_cloning_{self.behavior}_{name}_{timestamp}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'behavior': self.behavior
        }, save_path)
        print(f"💾 Model saved: {save_path}")
    
    def _plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Behavior Cloning Training - {self.behavior.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for loss curves
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"behavior_cloning_{self.behavior}_training_curves_{timestamp}.png", dpi=150)
        print(f"📊 Training curves saved")
        plt.show()


def main():
    """Main training function"""
    
    print("🤖 Genesis Behavior Cloning Trainer")
    print("=" * 50)
    
    # Behavior selection
    print("Available behaviors:")
    print("1. walk - Natural human walking")
    print("2. run - Running/jogging motion")
    
    choice = input("Select behavior (1/2 or walk/run): ").strip().lower()
    
    behavior_map = {"1": "walk", "2": "run"}
    if choice in behavior_map:
        behavior = behavior_map[choice]
    elif choice in ["walk", "run"]:
        behavior = choice
    else:
        print("Invalid choice, defaulting to 'walk'")
        behavior = "walk"
    
    print(f"\n🎯 Selected behavior: {behavior.upper()}")
    
    # Training configuration
    print("\nTraining scale:")
    print("1. Quick test (500 epochs, 5K samples)")
    print("2. Standard (1000 epochs, 20K samples)")
    print("3. Extensive (2000 epochs, 50K samples)")
    
    scale_choice = input("Select scale (1/2/3): ").strip()
    
    if scale_choice == "1":
        epochs, train_samples = 500, 5000
        print("⚡ Quick test configuration")
    elif scale_choice == "3":
        epochs, train_samples = 2000, 50000
        print("🚀 Extensive training configuration")
    else:
        epochs, train_samples = 1000, 20000
        print("🎯 Standard training configuration")
    
    print(f"   Epochs: {epochs}")
    print(f"   Training samples: {train_samples:,}")
    
    input("\nPress Enter to start training...")
    
    try:
        # Initialize trainer and run training
        trainer = BehaviorCloningTrainer(behavior=behavior)
        trainer.train(
            num_epochs=epochs,
            train_samples=train_samples,
            val_samples=train_samples // 10  # 10% for validation
        )
        
        print(f"\n🎉 Behavior cloning training complete!")
        print(f"📁 Models and plots saved in current directory")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()