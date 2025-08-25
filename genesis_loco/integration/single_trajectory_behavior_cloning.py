#!/usr/bin/env python3
"""
Single Trajectory Behavior Cloning

Simple and efficient behavior cloning focused on learning from a single walking 
trajectory segment. This approach can be more effective than training on mixed
trajectory data as it learns consistent, coherent motion patterns.

Key advantages:
- Consistent motion patterns (no mixing different walking styles)
- Better temporal coherence 
- Faster training convergence
- Easier to debug and analyze

Usage:
1. First run visualize_expert_trajectory.py to identify good segments
2. Use this script to train on a specific segment
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

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
import genesis as gs


def safe_init_genesis():
    """Safely initialize Genesis"""
    try:
        gs.init(backend=gs.gpu)
        return True
    except Exception as e:
        if "already initialized" in str(e):
            return True
        else:
            return False


class SingleTrajectoryMLP(nn.Module):
    """
    Simple MLP optimized for single trajectory learning
    - Smaller capacity to prevent overfitting on small dataset
    - Better initialization for consistent motion patterns
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        # Hidden layers with dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())  # ELU for smooth activations
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize for smooth motion prediction
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for smooth motion prediction"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class SingleTrajectoryTrainer:
    """
    Single trajectory behavior cloning trainer
    """
    
    def __init__(self, start_timestep: int, end_timestep: int, 
                 device: str = "auto", trajectory_name: str = "walk"):
        
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.trajectory_name = trajectory_name

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
        
        print(f"🎯 Single Trajectory Behavior Cloning")
        print(f"   Trajectory: {trajectory_name}")
        print(f"   Segment: timesteps {start_timestep} to {end_timestep}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        # Initialize components
        self._setup_genesis()
        self._setup_environment()
        self._setup_data_bridge()
        self._verify_joint_ordering()
        self._analyze_trajectory_segment()
        self._setup_model()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
    
    def _setup_genesis(self):
        """Initialize Genesis"""
        if not safe_init_genesis():
            raise RuntimeError("Failed to initialize Genesis")
        print("✅ Genesis initialized")
    
    def _setup_environment(self):
        """Setup Genesis environment"""
        self.env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=10.0,
            dt=0.01,
            show_viewer=True,
            use_box_feet=True,
            obs_history_length=1  # Simple observation for single trajectory
        )
        
        self.obs_dim = self.env.num_observations
        self.action_dim = self.env.num_actions
        
        print(f"✅ Environment: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
    
    def _setup_data_bridge(self):
        """Setup data bridge"""
        self.data_bridge = LocoMujocoDataBridge(self.env)
        success = self.data_bridge.load_trajectory(self.trajectory_name)
        if not success:
            raise RuntimeError(f"Failed to load {self.trajectory_name} trajectory")
        
        self.trajectory_length = self.data_bridge.trajectory_length
        print(f"✅ Trajectory loaded: {self.trajectory_length} total timesteps")
    
    def _verify_joint_ordering(self):
        """Verify joint ordering consistency between data bridge and environment"""
        print(f"🔍 Verifying joint ordering consistency...")
        
        # Get environment's joint information
        env_motors_dof_idx = self.env.motors_dof_idx
        env_joint_names = list(self.env.joint_names)
        
        # Get data bridge's joint information  
        bridge_motors_dof_idx = self.data_bridge.motors_dof_idx
        bridge_joint_names = list(self.data_bridge.joint_names)
        
        print(f"   Environment action_dim: {self.env.num_actions}")
        print(f"   Data bridge dof_pos dim: {len(bridge_motors_dof_idx)}")
        
        # The data bridge is designed to output dof_pos in the same order as environment expects
        # since it uses the same motors_dof_idx mapping, so they should always match
        if env_motors_dof_idx == bridge_motors_dof_idx:
            print(f"✅ Joint orderings match perfectly!")
        else:
            print(f"⚠️  Joint ordering mismatch detected!")
            print(f"   This suggests a bug in data_bridge initialization")
            print(f"   Environment motors_dof_idx: {env_motors_dof_idx}")
            print(f"   Data bridge motors_dof_idx: {bridge_motors_dof_idx}")
    
    def _analyze_trajectory_segment(self):
        """Analyze the specific trajectory segment"""
        # Validate segment
        if self.end_timestep > self.trajectory_length:
            self.end_timestep = self.trajectory_length
            print(f"⚠️  End timestep adjusted to {self.end_timestep}")
        
        if self.start_timestep >= self.end_timestep:
            raise ValueError("Invalid segment: start >= end")
        
        self.segment_length = self.end_timestep - self.start_timestep
        segment_duration = self.segment_length / self.data_bridge.trajectory_frequency
        
        print(f"✅ Trajectory segment analyzed:")
        print(f"   Length: {self.segment_length} timesteps")
        print(f"   Duration: {segment_duration:.2f} seconds")
        print(f"   Frequency: {self.data_bridge.trajectory_frequency} Hz")
        
        # Analyze segment characteristics
        start_state = self.data_bridge.get_trajectory_state(self.start_timestep)
        end_state = self.data_bridge.get_trajectory_state(self.end_timestep - 1)
        
        if start_state and end_state:
            start_pos = start_state['root_pos']
            end_pos = end_state['root_pos']
            
            if hasattr(start_pos, 'cpu'):
                start_pos = start_pos.cpu().numpy()
                end_pos = end_pos.cpu().numpy()
            
            distance = end_pos[0] - start_pos[0]
            avg_speed = distance / segment_duration
            
            print(f"   Distance covered: {distance:.3f}m")
            print(f"   Average speed: {avg_speed:.3f} m/s")
    
    def _setup_model(self):
        """Setup model optimized for single trajectory"""
        # Smaller model for single trajectory to prevent overfitting
        self.model = SingleTrajectoryMLP(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=[512, 256],  
            # dropout_rate=0.1
        ).to(self.device)
        
        # Optimizer with conservative settings
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=3e-4,  # Lower learning rate for stability
            # weight_decay=1e-3
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=15, factor=0.7
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        param_ratio = total_params / self.action_dim
        print(f"✅ Model: {total_params:,} parameters (ratio: {param_ratio:.1f})")
    
    # def _collect_segment_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Collect observations and actions from trajectory segment"""
    #     print(f"📊 Collecting data from segment...")
        
    #     observations = []
    #     actions = []
        
    #     # Collect data from entire segment
    #     env_ids = torch.tensor([0], device=self.device)
        
    #     for step in range(self.segment_length):
    #         timestep = self.start_timestep + step
            
    #         if step % 200 == 0:
    #             progress = (step / self.segment_length) * 100
    #             print(f"   Progress: {progress:.1f}%")
            
    #         # Get trajectory state
    #         state = self.data_bridge.get_trajectory_state(timestep)
    #         if state is None:
    #             continue
            
    #         try:
    #             # Apply state to environment to get observation
    #             self.data_bridge.apply_trajectory_state(state, env_ids)
                
    #             # Get current observation
    #             obs = self.env._get_observations()
    #             if obs is not None:
    #                 current_obs = obs[0].cpu().numpy()
                    
    #                 # Get target action (joint positions from expert)
    #                 target_action = self._extract_target_action(state)
    #                 if target_action is not None:
    #                     observations.append(current_obs)
    #                     actions.append(target_action)
                
    #         except Exception as e:
    #             if step < 10:  # Only print first few errors
    #                 print(f"   Error at timestep {timestep}: {e}")
    #             continue
        
    #     if len(observations) == 0:
    #         raise RuntimeError("No valid data collected from segment")
        
    #     # Convert to tensors
    #     obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)
    #     action_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        
    #     print(f"✅ Collected {len(observations)} samples from segment")
    #     print(f"   Observation shape: {obs_tensor.shape}")
    #     print(f"   Action shape: {action_tensor.shape}")
        
    #     # CRITICAL VERIFICATION: Ensure dimensions match model expectations
    #     print(f"🔍 Dimension verification:")
    #     print(f"   Environment obs_dim: {self.obs_dim}")
    #     print(f"   Environment action_dim: {self.action_dim}")
    #     print(f"   Collected obs_dim: {obs_tensor.shape[1]}")
    #     print(f"   Collected action_dim: {action_tensor.shape[1]}")
        
    #     if obs_tensor.shape[1] != self.obs_dim:
    #         raise ValueError(f"Observation dimension mismatch: collected {obs_tensor.shape[1]} != expected {self.obs_dim}")
    #     if action_tensor.shape[1] != self.action_dim:
    #         raise ValueError(f"Action dimension mismatch: collected {action_tensor.shape[1]} != expected {self.action_dim}")
        
    #     print(f"✅ All dimensions match correctly!")
        
    #     return obs_tensor, action_tensor

    def _collect_segment_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect (current_obs, next_target_positions) pairs"""
        print("📊 Collecting observation-action pairs...")

        # Pre-allocate numpy arrays for efficiency
        expected_samples = self.segment_length - 1
        observations = np.zeros((expected_samples, self.obs_dim), dtype=np.float32)
        actions = np.zeros((expected_samples, self.action_dim), dtype=np.float32)
        
        env_ids = torch.tensor([0], device=self.device)
        valid_samples = 0

        # Collect data from trajectory segment
        for step in range(self.segment_length - 1):  # -1 for next timestep access
            
            if step % 100 == 0:
                progress = (step / (self.segment_length - 1)) * 100
                print(f"   Collection Progress: {progress:.1f}%")
                
            current_timestep = self.start_timestep + step
            next_timestep = current_timestep + 1

            # Get current state and apply it
            current_state = self.data_bridge.get_trajectory_state(current_timestep)
            if current_state is None:
                continue

            # Apply state with physics stepping to get exact positions
            self.data_bridge.apply_trajectory_state(current_state, env_ids)
            current_obs = self.env._get_observations()[0]

            # TORQUE-BASED TRAINING: Compute expert torques instead of positions
            target_action = self.data_bridge.compute_expert_torques(current_timestep)
            if target_action is None:
                continue
                
            # OLD POSITION-BASED TRAINING (commented out):
            # next_state = self.data_bridge.get_trajectory_state(next_timestep)
            # if next_state is None:
            #     continue
            # target_action = next_state['dof_pos']

            # Store in pre-allocated arrays
            observations[valid_samples] = current_obs.cpu().numpy()
            actions[valid_samples] = target_action.cpu().numpy()
            valid_samples += 1

        # Trim to actual valid samples and convert to tensors
        if valid_samples == 0:
            raise RuntimeError("No valid data collected from segment")
            
        observations = observations[:valid_samples]
        actions = actions[:valid_samples]
        
        print(f"✅ Collected {valid_samples} valid samples from segment")
        
        return torch.from_numpy(observations).to(self.device), torch.from_numpy(actions).to(self.device)

    # def _apply_state_without_physics(self, state_data, env_ids):
    #   """Apply trajectory state without physics stepping"""
    #   # Set positions directly
    #   dof_pos = state_data['dof_pos'].unsqueeze(0).repeat(len(env_ids), 1)
    #   root_pos = state_data['root_pos'].unsqueeze(0).repeat(len(env_ids), 1)
    #   root_quat = state_data['root_quat'].unsqueeze(0).repeat(len(env_ids), 1)

    #   self.env.robot.set_dofs_position(dof_pos, dofs_idx_local=self.env.motors_dof_idx, envs_idx=env_ids)
    #   self.env.robot.set_pos(root_pos, envs_idx=env_ids)
    #   self.env.robot.set_quat(root_quat, envs_idx=env_ids)

    #   # Update state buffers without physics stepping
    #   self.env._update_robot_state()
    
    def _extract_target_action(self, state: Dict) -> np.ndarray:
        """Extract target joint positions for controllable joints only"""
        if 'dof_pos' not in state:
            return None
        
        dof_pos = state['dof_pos']
        if hasattr(dof_pos, 'cpu'):
            dof_pos = dof_pos.cpu().numpy()
        
        # The dof_pos from data_bridge already contains only controllable joint positions
        # in the correct order (mapped using motors_dof_idx), so we can use it directly
        # This gives us exactly 27 joint positions matching our action space
        return dof_pos.astype(np.float32)
    
    def train(self, num_epochs: int = 800, batch_size: int = 128, 
              validation_split: float = 0.2):
        """Train on single trajectory segment"""
        
        print(f"\n🎯 Starting Single Trajectory Training")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Validation split: {validation_split}")
        print("=" * 60)
        
        # Collect all data from segment
        all_obs, all_actions = self._collect_segment_data()
        
        # Split into train/validation
        n_samples = all_obs.shape[0]
        # n_val = int(n_samples * validation_split)
        n_val = 0
        n_train = n_samples - n_val
        
        # Random split
        # indices = torch.randperm(n_samples)

        # No random split, model learns full sequential walking
        train_indices = torch.arange(n_train)
        # val_indices = torch.arange(n_train, n_samples)
        
        train_obs = all_obs[train_indices]
        train_actions = all_actions[train_indices]
        # val_obs = all_obs[val_indices]
        # val_actions = all_actions[val_indices]
        
        print(f"   Training samples: {n_train}")
        print(f"   Validation samples: {n_val}")
        
        # Training loop
        # best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience = 100  # Increased patience for single trajectory
        epochs_no_improve = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            # Shuffle training data
            # train_perm = torch.randperm(n_train)
            train_obs_shuffled = train_obs#[train_perm]
            train_actions_shuffled = train_actions#[train_perm]
            
            # Mini-batch training
            for i in range(0, n_train, batch_size):
                batch_obs = train_obs_shuffled[i:i+batch_size]
                batch_actions = train_actions_shuffled[i:i+batch_size]
                
                # Forward pass
                predicted_actions = self.model(batch_obs)
                loss = F.mse_loss(predicted_actions, batch_actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            # self.model.eval()
            # with torch.no_grad():
            #     val_predictions = self.model(val_obs)
            #     val_loss = F.mse_loss(val_predictions, val_actions).item()
            #     self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            # self.scheduler.step(val_loss)
            self.scheduler.step(avg_train_loss)
            
            # Progress logging
            if epoch % 20 == 0 or epoch < 10:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d}/{num_epochs}: "
                      f"Train: {avg_train_loss:.6f}, "
                    #   f"Val: {val_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                      f"Time: {elapsed:.1f}s")
            
            # # Early stopping
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     epochs_no_improve = 0
            #     # Save best model
            #     self._save_model("best_single_trajectory")
            # else:
            #     epochs_no_improve += 1
            #     if epochs_no_improve >= patience:
            #         print(f"\n⏰ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            #         break

            # Save best model based on training loss
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                self._save_model("best_overfit")
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.1f}s")
        # print(f"   Best validation loss: {best_val_loss:.6f}")
        
        # Plot training curves
        self._plot_training_curves()
        
        # Save final model
        self._save_model("final_single_trajectory")
        
        # return best_val_loss
        return best_train_loss
    
    def _save_model(self, name: str):
        """Save model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_seg{self.start_timestep}-{self.end_timestep}_{timestamp}.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'start_timestep': self.start_timestep,
            'end_timestep': self.end_timestep,
            'segment_length': self.segment_length,
            'trajectory_name': self.trajectory_name
        }, filename)
        
        print(f"💾 Model saved: {filename}")
    
    def _plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        # plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Single Trajectory Training - Segment {self.start_timestep}-{self.end_timestep}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"single_trajectory_training_seg{self.start_timestep}-{self.end_timestep}_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📊 Training curves saved: {filename}")
        plt.show()


def main():
    """Main function"""
    print("🎯 SINGLE TRAJECTORY BEHAVIOR CLONING")
    print("=" * 60)
    print("Train a behavior cloning model on a specific trajectory segment")
    print("for better consistency and faster convergence.")
    
    # Get segment parameters
    print("\nFirst, run visualize_expert_trajectory.py to identify good segments!")
    print("Then specify the segment you want to train on:")
    
    try:
        start_timestep = int(input("Start timestep: "))
        end_timestep = int(input("End timestep: "))
        
        if start_timestep >= end_timestep:
            print("❌ Invalid segment: start must be < end")
            return 1
        
        segment_length = end_timestep - start_timestep
        if segment_length < 100:
            print("⚠️  Warning: Very short segment, training may not be effective")
        
        print(f"✅ Training segment: {start_timestep} to {end_timestep} ({segment_length} timesteps)")
        
        # Training configuration
        print("\nTraining configuration:")
        print("1. Quick test (200 epochs)")
        print("2. Standard (800 epochs)")
        print("3. Extended (1500 epochs)")
        
        config_choice = input("Select configuration (1/2/3): ").strip()
        
        if config_choice == "1":
            epochs = 200
            print("⚡ Quick test configuration")
        elif config_choice == "3":
            epochs = 1500
            print("🚀 Extended training configuration")
        else:
            epochs = 800
            print("🎯 Standard training configuration")
        
        print(f"   Epochs: {epochs}")
        
        input("\nPress Enter to start training...")
        
        # Create trainer and train
        trainer = SingleTrajectoryTrainer(start_timestep, end_timestep)
        best_loss = trainer.train(num_epochs=epochs)
        
        print(f"\n🎉 Single trajectory training complete!")
        # print(f"   Best validation loss: {best_loss:.6f}")
        print(f"   Best training loss: {best_loss:.6f}")
        print(f"   Segment: {start_timestep}-{end_timestep}")
        
    except ValueError:
        print("❌ Invalid input format")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())