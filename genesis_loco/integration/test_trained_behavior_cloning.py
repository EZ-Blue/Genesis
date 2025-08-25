#!/usr/bin/env python3
"""
Test Trained Behavior Cloning Model

Loads a trained behavior cloning model and visualizes it controlling the Genesis
skeleton humanoid in real-time. The model predicts target joint positions which
are applied through the PD controller in skeleton_humanoid.py.

This script demonstrates:
1. Loading trained model (.pth file)
2. Real-time policy execution in Genesis
3. Observation-to-action loop
4. Visual assessment of learned walking behavior
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import glob
from typing import Dict, List
# import keyboard  # Removed - requires root on Linux

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv


def safe_init_genesis():
    """Safely initialize Genesis"""
    try:
        gs.init(backend=gs.gpu)
        return True
    except Exception as e:
        if "already initialized" in str(e):
            return True
        else:
            print(f"Genesis initialization failed: {e}")
            return False


class SingleTrajectoryMLP(nn.Module):
    """
    Recreate the model architecture (must match training script)
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
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class BehaviorCloningTester:
    """Test trained behavior cloning model in Genesis"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        
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
        
        print(f"🤖 Behavior Cloning Model Tester")
        print(f"   Model: {os.path.basename(model_path)}")
        print(f"   Device: {self.device}")
        print("=" * 60)
        
        # Initialize components
        self._setup_genesis()
        self._load_model()
        self._setup_environment()
        
        # Test metrics
        self.test_metrics = {}
    
    def _setup_genesis(self):
        """Initialize Genesis"""
        if not safe_init_genesis():
            raise RuntimeError("Failed to initialize Genesis")
        print("✅ Genesis initialized")
    
    def _load_model(self):
        """Load trained model"""
        print(f"📥 Loading model from {self.model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model info
        self.obs_dim = checkpoint['obs_dim']
        self.action_dim = checkpoint['action_dim']
        
        # Check for single trajectory specific info
        if 'start_timestep' in checkpoint:
            self.start_timestep = checkpoint['start_timestep']
            self.end_timestep = checkpoint['end_timestep']
            self.segment_length = checkpoint['segment_length']
            self.trajectory_name = checkpoint.get('trajectory_name', 'unknown')
            print(f"   📊 Single trajectory model detected:")
            print(f"      Segment: {self.start_timestep}-{self.end_timestep}")
            print(f"      Length: {self.segment_length} timesteps")
            print(f"      Trajectory: {self.trajectory_name}")
        
        # Create model (try to infer architecture from state dict)
        try:
            # Check first hidden layer size to infer architecture
            first_weight = checkpoint['model_state_dict']['network.0.weight']
            first_hidden_dim = first_weight.shape[0]
            
            # Try to infer full architecture
            if first_hidden_dim == 256:
                hidden_dims = [256, 128]
            elif first_hidden_dim == 512:
                hidden_dims = [512, 256]
            elif first_hidden_dim == 128:
                hidden_dims = [128, 64]
            elif first_hidden_dim == 96:
                hidden_dims = [96, 48]
            else:
                hidden_dims = [256, 128]  # Default fallback
            
            print(f"   🧠 Inferred architecture: {hidden_dims}")
            
        except:
            hidden_dims = [256, 128]  # Default
            print(f"   🧠 Using default architecture: {hidden_dims}")
        
        # Create model
        self.model = SingleTrajectoryMLP(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            dropout_rate=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set to evaluation mode
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded: {total_params:,} parameters")
        print(f"   Input: {self.obs_dim} observations")
        print(f"   Output: {self.action_dim} joint positions")
    
    def _setup_environment(self):
        """Setup Genesis environment"""
        self.env = SkeletonHumanoidEnv(
            num_envs=1,  # Single environment for testing
            episode_length_s=30.0,  # Long episodes for evaluation
            dt=0.01,
            show_viewer=True,  # Enable visualization
            use_box_feet=True,
            obs_history_length=1,  # Match training configuration
            sim_options=gs.options.SimOptions(
                dt=0.1, 
                # substeps=2,
                # gravity=(0.0,0.0,0.0)
            ),
        )

        
        print(f"✅ Environment created:")
        print(f"   Observations: {self.env.num_observations}")
        print(f"   Actions: {self.env.num_actions}")
        
        # Verify dimensions match model
        if self.obs_dim != self.env.num_observations:
            print(f"⚠️  Warning: Model obs_dim ({self.obs_dim}) != env obs_dim ({self.env.num_observations})")
        if self.action_dim != self.env.num_actions:
            print(f"⚠️  Warning: Model action_dim ({self.action_dim}) != env action_dim ({self.env.num_actions})")
        
        # Setup data bridge for initial state positioning
        self._setup_data_bridge()
        
        # Verify joint ordering consistency
        self._verify_joint_ordering()
    
    def _setup_data_bridge(self):
        """Setup data bridge for trajectory access"""
        from integration.data_bridge import LocoMujocoDataBridge
        
        try:
            self.data_bridge = LocoMujocoDataBridge(self.env)
            success = self.data_bridge.load_trajectory("walk")
            if success:
                print(f"✅ Data bridge loaded for initial positioning")
            else:
                print(f"⚠️  Failed to load trajectory data for initial positioning")
                self.data_bridge = None
        except Exception as e:
            print(f"⚠️  Data bridge setup failed: {e}")
            self.data_bridge = None
    
    def _set_initial_position_from_expert(self):
        """Set initial position to match expert trajectory start"""
        if not self.data_bridge:
            print(f"⚠️  No data bridge available, using default initial position")
            return False
        
        # Check if this is a single trajectory model with specific start timestep
        if hasattr(self, 'start_timestep'):
            start_timestep = self.start_timestep
            print(f"📍 Setting initial position from expert timestep {start_timestep}")
        else:
            # Use beginning of trajectory for general models
            start_timestep = 0
            print(f"📍 Setting initial position from trajectory start (timestep 0)")
        
        try:
            # Get expert state at start of training segment
            expert_state = self.data_bridge.get_trajectory_state(start_timestep)
            if expert_state is None:
                print(f"⚠️  Could not get expert state at timestep {start_timestep}")
                return False
            
            # Apply expert state to environment
            env_ids = torch.tensor([0], device=self.env.device)
            self.data_bridge.apply_trajectory_state(expert_state, env_ids)
            
            # Get the applied position for confirmation
            root_pos = self.env.root_pos[0].cpu().numpy()
            print(f"✅ Initial position set to expert state:")
            print(f"   Position: [{root_pos[0]:+.3f}, {root_pos[1]:+.3f}, {root_pos[2]:+.3f}]m")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to set initial position from expert: {e}")
            return False
    
    def _verify_joint_ordering(self):
        """Verify joint ordering consistency between trained model and environment"""
        if not self.data_bridge:
            print(f"⚠️  No data bridge available, skipping joint ordering verification")
            return
            
        print(f"🔍 Verifying model/environment compatibility...")
        
        # Check dimensions match
        print(f"   Environment action_dim: {self.env.num_actions}")
        print(f"   Model action_dim: {self.action_dim}")
        print(f"   Environment obs_dim: {self.env.num_observations}")
        print(f"   Model obs_dim: {self.obs_dim}")
        
        if self.action_dim == self.env.num_actions:
            print(f"✅ Action dimensions match perfectly!")
        else:
            print(f"⚠️  Action dimension mismatch! Model may not work correctly.")
            
        if self.obs_dim == self.env.num_observations:
            print(f"✅ Observation dimensions match perfectly!")
        else:
            print(f"⚠️  Observation dimension mismatch! Model may not work correctly.")
    
    def _get_observation(self) -> torch.Tensor:
        """Get current observation from environment"""
        obs = self.env._get_observations()
        
        # Convert to tensor and ensure correct shape
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs[0]  # Take first (and only) environment
        else:
            obs_tensor = torch.tensor(obs[0], dtype=torch.float32)
        
        # Handle dimension mismatch
        if obs_tensor.shape[0] > self.obs_dim:
            obs_tensor = obs_tensor[:self.obs_dim]
            print(f"⚠️  Truncated observation from {obs_tensor.shape[0]} to {self.obs_dim}")
        elif obs_tensor.shape[0] < self.obs_dim:
            padding = torch.zeros(self.obs_dim - obs_tensor.shape[0])
            obs_tensor = torch.cat([obs_tensor, padding])
            print(f"⚠️  Padded observation from {obs_tensor.shape[0]} to {self.obs_dim}")
        
        return obs_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def test_walking(self, max_steps: int = 3000, record_metrics: bool = True):
        """Test the model's walking ability"""
        print(f"\n🚶 TESTING WALKING BEHAVIOR")
        print("=" * 60)
        print(f"Duration: {max_steps} steps ({max_steps * self.env.dt:.1f} seconds)")
        print(f"Watch the Genesis viewer to assess walking quality")
        print(f"Press Ctrl+C to stop early")
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Set initial position to match expert trajectory start
        expert_init_success = self._set_initial_position_from_expert()
        if not expert_init_success:
            print(f"   Using default reset position")
        
        # Metrics tracking
        if record_metrics:
            episode_data = {
                'steps': [],
                'rewards': [],
                'positions': [],
                'heights': [],
                'actions': [],
                'action_magnitudes': [],
                'velocities': []
            }
        
        total_reward = 0.0
        step_count = 0
        start_time = time.time()
        
        print(f"🏃 Starting walking test...")
        
        try:
            for step in range(max_steps):
                # Get current observation
                obs_tensor = self._get_observation()
                
                # Predict target joint positions using trained model
                with torch.no_grad():
                    predicted_positions = self.model(obs_tensor)
                    actions = predicted_positions[0]  # Remove batch dimension, keep on device
                
                # Ensure actions are on the correct device (same as environment)
                actions = actions.to(self.env.device)
                
                # Apply actions through environment's PD controller
                # The environment's step function calls _apply_actions which uses control_dofs_position
                obs, rewards, dones, info = self.env.step(actions.unsqueeze(0))

                # Manual stepping control (press Enter to continue)
                if step % 1 == 0:  # Every step - change to higher number for less frequent pauses
                    input(f"Step {step}: Press Enter to continue (Ctrl+C to exit)...")
                
                # Record metrics
                if record_metrics:
                    root_pos = self.env.root_pos[0].cpu().numpy()
                    root_vel = self.env.root_lin_vel[0].cpu().numpy()
                    
                    episode_data['steps'].append(step)
                    episode_data['rewards'].append(rewards[0].item())
                    episode_data['positions'].append(root_pos[0])  # Forward position
                    episode_data['heights'].append(root_pos[2])    # Height
                    episode_data['actions'].append(actions.cpu().numpy().copy())  # Move to CPU for storage
                    episode_data['action_magnitudes'].append(torch.norm(actions).item())
                    episode_data['velocities'].append(root_vel[0])  # Forward velocity
                
                total_reward += rewards[0].item()
                step_count += 1
                
                # Progress updates
                if step % 500 == 0:
                    elapsed = time.time() - start_time
                    root_pos = self.env.root_pos[0].cpu().numpy()
                    avg_reward = total_reward / step_count if step_count > 0 else 0
                    
                    print(f"   Step {step:4d}: "
                          f"Pos={root_pos[0]:+6.2f}m, "
                          f"Height={root_pos[2]:.3f}m, "
                          f"Reward={avg_reward:.4f}, "
                          f"Time={elapsed:.1f}s")
                
                # Check for early termination (falling)
                if dones[0]:
                    print(f"   💥 Episode ended at step {step} (robot fell or episode limit)")
                    break
        
        except KeyboardInterrupt:
            print(f"\n🛑 Test stopped by user at step {step}")
        
        # Final analysis
        total_time = time.time() - start_time
        final_pos = self.env.root_pos[0].cpu().numpy()
        
        print(f"\n📊 WALKING TEST RESULTS:")
        print("=" * 60)
        print(f"Total steps: {step_count}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Final position: {final_pos[0]:+.3f}m forward, {final_pos[2]:.3f}m height")
        print(f"Distance traveled: {abs(final_pos[0]):.3f}m")
        print(f"Average speed: {abs(final_pos[0]) / (step_count * self.env.dt):.3f} m/s")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward: {total_reward / step_count:.4f}")
        
        # Walking quality assessment
        stayed_upright = final_pos[2] > 0.5
        moved_forward = abs(final_pos[0]) > 0.5
        
        print(f"\n🏆 WALKING QUALITY ASSESSMENT:")
        print(f"   Stayed upright: {'✅' if stayed_upright else '❌'} (height > 0.5m)")
        print(f"   Moved forward: {'✅' if moved_forward else '❌'} (distance > 0.5m)")
        
        if stayed_upright and moved_forward:
            print(f"   🎉 SUCCESS: Model learned to walk!")
        elif stayed_upright:
            print(f"   🚶 PARTIAL: Model stays upright but doesn't walk well")
        else:
            print(f"   ❌ FAILED: Model falls over")
        
        if record_metrics:
            self.test_metrics = episode_data
            return episode_data
        
        return None
    
    def analyze_action_patterns(self, episode_data: Dict):
        """Analyze action patterns from recorded data"""
        if not episode_data or len(episode_data['actions']) == 0:
            print("No action data to analyze")
            return
        
        print(f"\n🔍 ACTION PATTERN ANALYSIS:")
        print("=" * 60)
        
        actions = np.array(episode_data['actions'])
        action_magnitudes = np.array(episode_data['action_magnitudes'])
        
        print(f"Action statistics:")
        print(f"   Shape: {actions.shape}")
        print(f"   Range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"   Mean magnitude: {action_magnitudes.mean():.6f}")
        print(f"   Std magnitude: {action_magnitudes.std():.6f}")
        
        # Check for problematic patterns
        if action_magnitudes.std() < 0.01:
            print(f"   ⚠️  Very low action variation - possible action collapse")
        
        if actions.max() - actions.min() < 0.1:
            print(f"   ⚠️  Very small action range - possible range collapse")
        
        # Joint-specific analysis
        print(f"\n📊 Per-joint action statistics (first 10 joints):")
        for i in range(min(10, actions.shape[1])):
            joint_actions = actions[:, i]
            print(f"   Joint {i:2d}: range [{joint_actions.min():+6.3f}, {joint_actions.max():+6.3f}], "
                  f"std {joint_actions.std():.4f}")
    
    def plot_walking_results(self, episode_data: Dict, save_path: str = None):
        """Plot walking test results"""
        if not episode_data:
            print("No data to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            print(f"\n📊 Generating walking analysis plots...")
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Behavior Cloning Walking Test - {os.path.basename(self.model_path)}')
            
            steps = episode_data['steps']
            time_axis = np.array(steps) * self.env.dt
            
            # Forward position
            axes[0, 0].plot(time_axis, episode_data['positions'])
            axes[0, 0].set_title('Forward Position')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Position (m)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Height
            axes[0, 1].plot(time_axis, episode_data['heights'])
            axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Fall threshold')
            axes[0, 1].set_title('Root Height')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Height (m)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Forward velocity
            axes[0, 2].plot(time_axis, episode_data['velocities'])
            axes[0, 2].set_title('Forward Velocity')
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].set_ylabel('Velocity (m/s)')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Rewards
            axes[1, 0].plot(time_axis, episode_data['rewards'])
            axes[1, 0].set_title('Rewards')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Action magnitudes
            axes[1, 1].plot(time_axis, episode_data['action_magnitudes'])
            axes[1, 1].set_title('Action Magnitude')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('||Actions||')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Summary stats
            final_pos = episode_data['positions'][-1] if episode_data['positions'] else 0
            final_height = episode_data['heights'][-1] if episode_data['heights'] else 0
            avg_reward = np.mean(episode_data['rewards']) if episode_data['rewards'] else 0
            
            axes[1, 2].text(0.1, 0.8, f"Final Position: {final_pos:.2f}m", fontsize=12)
            axes[1, 2].text(0.1, 0.7, f"Final Height: {final_height:.3f}m", fontsize=12)
            axes[1, 2].text(0.1, 0.6, f"Avg Reward: {avg_reward:.4f}", fontsize=12)
            axes[1, 2].text(0.1, 0.5, f"Steps: {len(steps)}", fontsize=12)
            axes[1, 2].text(0.1, 0.4, f"Duration: {len(steps) * self.env.dt:.1f}s", fontsize=12)
            
            success = final_height > 0.5 and abs(final_pos) > 0.5
            axes[1, 2].text(0.1, 0.2, f"Success: {'✅' if success else '❌'}", fontsize=14, weight='bold')
            
            axes[1, 2].set_title('Test Summary')
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = f"walking_test_{timestamp}.png"
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
            plt.show()
            
        except ImportError:
            print("⚠️  Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"⚠️  Plotting failed: {e}")


def find_latest_model(pattern: str = "*single_trajectory*.pth") -> str:
    """Find the most recent model file"""
    model_files = glob.glob(pattern)
    if not model_files:
        # Try other patterns
        patterns = ["*behavior_cloning*.pth", "*.pth"]
        for pat in patterns:
            model_files = glob.glob(pat)
            if model_files:
                break
    
    if not model_files:
        raise FileNotFoundError(f"No model files found")
    
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model


def main():
    """Main testing function"""
    print("🤖 BEHAVIOR CLONING MODEL TESTER")
    print("=" * 60)
    print("Test your trained behavior cloning model in Genesis!")
    
    # Find model to test
    try:
        latest_model = find_latest_model()
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
    
    # Test configuration
    print("\nTest configuration:")
    print("1. Quick test (1000 steps, ~10s)")
    print("2. Standard test (3000 steps, ~30s)")
    print("3. Long test (6000 steps, ~60s)")
    
    test_choice = input("Select test duration (1/2/3): ").strip()
    
    if test_choice == "1":
        max_steps = 1000
        print("⚡ Quick test")
    elif test_choice == "3":
        max_steps = 6000
        print("🔬 Long test")
    else:
        max_steps = 3000
        print("🎯 Standard test")
    
    print(f"   Steps: {max_steps}")
    print(f"   Duration: ~{max_steps * 0.01:.0f} seconds")
    
    input("\nPress Enter to start testing...")
    
    try:
        # Create tester and run test
        tester = BehaviorCloningTester(model_path)
        episode_data = tester.test_walking(max_steps=max_steps, record_metrics=True)
        
        # Analysis
        if episode_data:
            tester.analyze_action_patterns(episode_data)
            
            # Plotting
            plot_choice = input("\nGenerate analysis plots? (y/n): ").strip().lower()
            if plot_choice == 'y':
                tester.plot_walking_results(episode_data)
        
        print(f"\n🎉 Testing complete!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())