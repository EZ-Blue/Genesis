#!/usr/bin/env python3
"""
Behavior Cloning Model vs Expert Trajectory Comparison

This script loads expert walking data and a trained behavior cloning model,
then compares the model's predicted next joint states against the actual 
expert trajectory next states for a selected segment.

Tests whether the behavior cloning model correctly learned the temporal
relationship: current_observation → next_joint_positions
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge


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
    """Recreate behavior cloning model architecture"""
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class BCTrajectoryComparator:
    """Compare behavior cloning model predictions with expert trajectory"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔍 BC Trajectory Comparator")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        # Initialize components
        self._setup_genesis()
        self._load_model()
        self._setup_environment()
        self._setup_data_bridge()
    
    def _setup_genesis(self):
        """Initialize Genesis"""
        if not safe_init_genesis():
            raise RuntimeError("Failed to initialize Genesis")
        print("✅ Genesis initialized")
    
    def _load_model(self):
        """Load trained behavior cloning model"""
        print(f"📥 Loading model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.obs_dim = checkpoint['obs_dim']
        self.action_dim = checkpoint['action_dim']
        
        # Get model training info if available
        if 'start_timestep' in checkpoint:
            self.model_start_timestep = checkpoint['start_timestep']
            self.model_end_timestep = checkpoint['end_timestep']
            self.model_segment_length = checkpoint['segment_length']
            print(f"   📊 Model trained on segment: {self.model_start_timestep}-{self.model_end_timestep}")
        
        # Infer model architecture
        try:
            first_weight = checkpoint['model_state_dict']['network.0.weight']
            first_hidden_dim = first_weight.shape[0]
            
            if first_hidden_dim == 256:
                hidden_dims = [256, 128]
            elif first_hidden_dim == 128:
                hidden_dims = [128, 64]
            else:
                hidden_dims = [256, 128]  # Default
        except:
            hidden_dims = [256, 128]
        
        # Create and load model
        self.model = SingleTrajectoryMLP(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            dropout_rate=0.0  # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded: {self.obs_dim} → {self.action_dim}")
    
    def _setup_environment(self):
        """Setup Genesis environment"""
        self.env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=30.0,
            dt=0.01,
            show_viewer=False,  # No viewer for testing
            use_box_feet=True,
            obs_history_length=1
        )
        
        print(f"✅ Environment: {self.env.num_observations} obs, {self.env.num_actions} actions")
        
        # Verify dimensions
        if self.obs_dim != self.env.num_observations:
            print(f"⚠️  Model obs_dim ({self.obs_dim}) != env obs_dim ({self.env.num_observations})")
        if self.action_dim != self.env.num_actions:
            print(f"⚠️  Model action_dim ({self.action_dim}) != env action_dim ({self.env.num_actions})")
    
    def _setup_data_bridge(self):
        """Setup data bridge for trajectory access"""
        self.data_bridge = LocoMujocoDataBridge(self.env)
        success = self.data_bridge.load_trajectory("walk")
        if not success:
            raise RuntimeError("Failed to load expert trajectory")
        
        print(f"✅ Expert trajectory loaded: {self.data_bridge.trajectory_length} timesteps")
    
    def select_test_segment(self) -> Tuple[int, int]:
        """Allow user to select test segment"""
        total_length = self.data_bridge.trajectory_length
        
        print(f"\n📋 SEGMENT SELECTION")
        print(f"   Total trajectory length: {total_length} timesteps")
        
        # Show model training segment if available
        if hasattr(self, 'model_start_timestep'):
            print(f"   Model was trained on: {self.model_start_timestep}-{self.model_end_timestep}")
            print(f"   Recommendation: Test on training segment for best performance")
        
        print(f"\nSelect test segment:")
        
        # Get user input
        try:
            start_timestep = int(input(f"Start timestep (0-{total_length-1}): "))
            end_timestep = int(input(f"End timestep ({start_timestep+1}-{total_length}): "))
            
            # Validate
            if start_timestep < 0 or start_timestep >= total_length:
                raise ValueError("Invalid start timestep")
            if end_timestep <= start_timestep or end_timestep > total_length:
                raise ValueError("Invalid end timestep")
            
            segment_length = end_timestep - start_timestep
            print(f"✅ Selected segment: {start_timestep}-{end_timestep} ({segment_length} timesteps)")
            
            return start_timestep, end_timestep
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            return None, None
    
    def compare_segment(self, start_timestep: int, end_timestep: int) -> Dict:
        """Compare model predictions vs expert for selected segment"""
        print(f"\n🔍 COMPARING MODEL vs EXPERT")
        print(f"   Segment: {start_timestep} to {end_timestep}")
        print("=" * 60)
        
        env_ids = torch.tensor([0], device=self.device)
        
        # Storage for comparison results
        comparison_data = {
            'timesteps': [],
            'model_predictions': [],
            'expert_targets': [],
            'prediction_errors': [],
            'mae_per_joint': [],
            'mse_per_joint': []
        }
        
        total_mae = 0.0
        total_mse = 0.0
        num_comparisons = 0
        
        print(f"Processing {end_timestep - start_timestep - 1} timestep comparisons...")
        
        for step in range(start_timestep, end_timestep - 1):  # -1 for next timestep
            current_timestep = step
            next_timestep = step + 1
            
            # Get current expert state and apply with physics
            current_state = self.data_bridge.get_trajectory_state(current_timestep)
            if current_state is None:
                continue
            
            self.data_bridge.apply_trajectory_state(current_state, env_ids)
            current_obs = self.env._get_observations()[0]
            
            # Get expert next state target with physics
            next_state = self.data_bridge.get_trajectory_state(next_timestep)
            if next_state is None:
                continue
            
            self.data_bridge.apply_trajectory_state(next_state, env_ids)
            expert_target = self.env.robot.get_dofs_position(
                dofs_idx_local=self.env.motors_dof_idx
            )[0]
            
            # Model prediction
            with torch.no_grad():
                obs_tensor = current_obs.unsqueeze(0).to(self.device)
                model_prediction = self.model(obs_tensor)[0]
            
            # Compute errors
            prediction_error = torch.abs(model_prediction - expert_target)
            mae = prediction_error.mean().item()
            mse = ((model_prediction - expert_target) ** 2).mean().item()
            
            # Store results
            comparison_data['timesteps'].append(current_timestep)
            comparison_data['model_predictions'].append(model_prediction.cpu().numpy())
            comparison_data['expert_targets'].append(expert_target.cpu().numpy())
            comparison_data['prediction_errors'].append(prediction_error.cpu().numpy())
            comparison_data['mae_per_joint'].append(prediction_error.cpu().numpy())
            comparison_data['mse_per_joint'].append(((model_prediction - expert_target) ** 2).cpu().numpy())
            
            total_mae += mae
            total_mse += mse
            num_comparisons += 1
            
            # Progress update
            if (step - start_timestep) % 50 == 0:
                progress = ((step - start_timestep) / (end_timestep - start_timestep - 1)) * 100
                print(f"   Progress: {progress:.1f}% | Current MAE: {mae:.6f}")
        
        # Compute overall metrics
        avg_mae = total_mae / num_comparisons if num_comparisons > 0 else 0
        avg_mse = total_mse / num_comparisons if num_comparisons > 0 else 0
        rmse = np.sqrt(avg_mse)
        
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   Comparisons made: {num_comparisons}")
        print(f"   Mean Absolute Error (MAE): {avg_mae:.6f}")
        print(f"   Mean Squared Error (MSE): {avg_mse:.6f}")
        print(f"   Root Mean Squared Error (RMSE): {rmse:.6f}")
        
        # Per-joint analysis
        if comparison_data['mae_per_joint']:
            mae_per_joint = np.array(comparison_data['mae_per_joint'])
            avg_mae_per_joint = mae_per_joint.mean(axis=0)
            
            print(f"\n📈 PER-JOINT ANALYSIS (first 10 joints):")
            for i in range(min(10, len(avg_mae_per_joint))):
                print(f"   Joint {i:2d}: MAE = {avg_mae_per_joint[i]:.6f}")
        
        # Quality assessment
        print(f"\n🏆 PREDICTION QUALITY ASSESSMENT:")
        if avg_mae < 0.01:
            print(f"   ✅ EXCELLENT: Very low prediction error")
        elif avg_mae < 0.05:
            print(f"   ✅ GOOD: Low prediction error")
        elif avg_mae < 0.1:
            print(f"   ⚠️  MODERATE: Moderate prediction error")
        else:
            print(f"   ❌ POOR: High prediction error")
        
        comparison_data['avg_mae'] = avg_mae
        comparison_data['avg_mse'] = avg_mse
        comparison_data['rmse'] = rmse
        
        return comparison_data
    
    def plot_comparison_results(self, comparison_data: Dict, save_path: str = None):
        """Plot comparison results"""
        if not comparison_data['timesteps']:
            print("No data to plot")
            return
        
        try:
            print(f"\n📊 Generating comparison plots...")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'BC Model vs Expert Trajectory Comparison\nModel: {os.path.basename(self.model_path)}')
            
            timesteps = comparison_data['timesteps']
            predictions = np.array(comparison_data['model_predictions'])
            targets = np.array(comparison_data['expert_targets'])
            errors = np.array(comparison_data['prediction_errors'])
            
            # Plot 1: Sample joint predictions vs targets
            joint_idx = 0  # First joint
            axes[0, 0].plot(timesteps, predictions[:, joint_idx], label='Model Prediction', alpha=0.8)
            axes[0, 0].plot(timesteps, targets[:, joint_idx], label='Expert Target', alpha=0.8)
            axes[0, 0].set_title(f'Joint {joint_idx} Predictions vs Targets')
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('Joint Position (rad)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Prediction errors over time
            mean_errors = errors.mean(axis=1)
            axes[0, 1].plot(timesteps, mean_errors)
            axes[0, 1].set_title('Mean Prediction Error Over Time')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Mean Absolute Error')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Error distribution histogram
            axes[1, 0].hist(mean_errors, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(comparison_data['avg_mae'], color='red', linestyle='--', 
                              label=f'Mean MAE: {comparison_data["avg_mae"]:.6f}')
            axes[1, 0].set_title('Error Distribution')
            axes[1, 0].set_xlabel('Mean Absolute Error')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Per-joint average errors
            mae_per_joint = np.array(comparison_data['mae_per_joint']).mean(axis=0)
            joint_indices = range(min(15, len(mae_per_joint)))  # Show first 15 joints
            axes[1, 1].bar(joint_indices, mae_per_joint[:len(joint_indices)])
            axes[1, 1].set_title('Average Error Per Joint (First 15)')
            axes[1, 1].set_xlabel('Joint Index')
            axes[1, 1].set_ylabel('Mean Absolute Error')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path is None:
                timestamp = os.path.basename(self.model_path).replace('.pth', '')
                save_path = f"bc_trajectory_comparison_{timestamp}.png"
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Plotting failed: {e}")


def find_latest_bc_model() -> str:
    """Find the most recent behavior cloning model"""
    patterns = ["*single_trajectory*.pth", "*behavior_cloning*.pth", "best_*.pth", "*.pth"]
    
    for pattern in patterns:
        model_files = glob.glob(pattern)
        if model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            return latest_model
    
    raise FileNotFoundError("No behavior cloning model files found")


def main():
    """Main function"""
    print("🔍 BEHAVIOR CLONING TRAJECTORY COMPARISON")
    print("=" * 60)
    print("Compare BC model predictions with expert trajectory next states")
    
    # Find model to test
    try:
        latest_model = find_latest_bc_model()
        print(f"📁 Found latest model: {latest_model}")
        
        use_latest = input(f"Test this model? (y/n): ").strip().lower()
        if use_latest != 'y':
            model_path = input("Enter path to model file: ").strip()
        else:
            model_path = latest_model
            
    except FileNotFoundError:
        print("❌ No trained models found!")
        model_path = input("Enter path to model file: ").strip()
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return 1
    
    try:
        # Create comparator
        comparator = BCTrajectoryComparator(model_path)
        
        # Select test segment
        start_timestep, end_timestep = comparator.select_test_segment()
        if start_timestep is None:
            return 1
        
        # Run comparison
        comparison_data = comparator.compare_segment(start_timestep, end_timestep)
        
        # Generate plots
        plot_choice = input("\nGenerate comparison plots? (y/n): ").strip().lower()
        if plot_choice == 'y':
            comparator.plot_comparison_results(comparison_data)
        
        print(f"\n🎉 Comparison complete!")
        
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())