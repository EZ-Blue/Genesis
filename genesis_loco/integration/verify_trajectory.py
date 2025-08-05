"""
Trajectory Following Verification Script

Forces the Genesis skeleton to follow the exact LocoMujoco walking trajectory
using control_dofs_position to verify joint mapping and control accuracy.
"""

import torch
import numpy as np
import time
import sys
import os
from typing import Tuple

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_bridge import LocoMujocoDataBridge
from amp_integration import AMPGenesisIntegration


class TrajectoryFollower:
    """
    Forces Genesis skeleton to follow LocoMujoco expert trajectory exactly
    
    Uses control_dofs_position for precise trajectory following to validate:
    1. Joint name mapping accuracy
    2. DOF ordering correctness  
    3. Control system functionality
    4. Trajectory data integrity
    """
    
    def __init__(self, show_viewer: bool = True, dataset_name: str = "walk"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        
        print("🎯 Trajectory Following Verification")
        print(f"   Dataset: {dataset_name}")
        print(f"   Device: {self.device}")
        print(f"   Visualization: {show_viewer}")
        
        # Setup environment and data
        self._setup_environment(show_viewer)
        self._setup_data_bridge()
        self._load_trajectory_data()
        
        print("✅ Trajectory follower ready!")
    
    def _setup_environment(self, show_viewer: bool):
        """Setup Genesis skeleton environment"""
        print("   Setting up Genesis environment...")
        
        import genesis as gs
        gs.init(backend=gs.gpu)
        
        # Import skeleton environment
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        self.env = SkeletonHumanoidEnv(
            num_envs=1,  # Single environment for verification
            episode_length_s=30.0,  # Long episodes for full trajectory
            dt=0.01,  # High frequency for smooth following (100Hz)
            show_viewer=show_viewer
        )
        
        # Enable PD control for position following
        self.env.setup_pd_control()  # This sets appropriate PD gains
        
        print(f"     ✓ Environment ready: {self.env.num_dofs} DOFs")
    
    def _setup_data_bridge(self):
        """Setup data bridge for trajectory loading"""
        print("   Setting up data bridge...")
        
        self.data_bridge = LocoMujocoDataBridge(self.env)
        
        # Load expert trajectory
        success, _ = self.data_bridge.load_trajectory(self.dataset_name)
        if not success:
            raise RuntimeError(f"Failed to load trajectory: {self.dataset_name}")
        
        # Build joint mapping
        success, mapping_info = self.data_bridge.build_joint_mapping()
        if not success:
            raise RuntimeError("Failed to build joint mapping")
        
        self.mapping_info = mapping_info
        print(f"     ✓ Joint mapping: {mapping_info['match_percentage']:.1f}% match rate")
    
    def _load_trajectory_data(self):
        """Load and prepare trajectory data"""
        print("   Loading trajectory data...")
        
        success, self.trajectory_data = self.data_bridge.convert_to_genesis_format()
        if not success:
            raise RuntimeError("Failed to convert trajectory data")
        
        self.n_timesteps = self.trajectory_data['info']['timesteps']
        self.frequency = self.trajectory_data['info']['frequency']
        
        print(f"     ✓ Trajectory loaded: {self.n_timesteps} timesteps at {self.frequency}Hz")
    
    def follow_trajectory(self, 
                         start_timestep: int = 0,
                         num_timesteps: int = None,
                         playback_speed: float = 1.0,
                         loop: bool = True) -> bool:
        """
        Follow the trajectory using position control
        
        Args:
            start_timestep: Starting timestep in trajectory
            num_timesteps: Number of timesteps to follow (None = all)
            playback_speed: Speed multiplier (1.0 = normal speed)
            loop: Whether to loop the trajectory
            
        Returns:
            Success status
        """
        if num_timesteps is None:
            num_timesteps = min(5000, self.n_timesteps - start_timestep)  # Limit for reasonable demo
        
        print(f"\n🚶‍♂️ Following trajectory:")
        print(f"   Start timestep: {start_timestep}")
        print(f"   Duration: {num_timesteps} steps ({num_timesteps/self.frequency:.1f}s)")
        print(f"   Playback speed: {playback_speed}x")
        print(f"   Loop: {loop}")
        print("=" * 50)
        
        # Reset environment to trajectory start position
        self._reset_to_trajectory_position(start_timestep)
        
        try:
            step_count = 0
            trajectory_idx = start_timestep
            
            while True:
                # Get current trajectory targets
                target_dof_pos = self.trajectory_data['dof_pos'][trajectory_idx:trajectory_idx+1]  # [1, num_dofs]
                target_root_pos = self.trajectory_data['root_pos'][trajectory_idx:trajectory_idx+1]  # [1, 3]
                
                # Apply position control
                self.env.robot.control_dofs_position(target_dof_pos)
                
                # Step simulation
                self.env.scene.step()
                
                # Update environment state for monitoring
                self.env._update_robot_state()
                
                # Progress tracking
                step_count += 1
                trajectory_idx += 1
                
                # Print progress periodically
                if step_count % 500 == 0:
                    progress = (step_count / num_timesteps) * 100
                    current_time = trajectory_idx / self.frequency
                    print(f"   Progress: {step_count}/{num_timesteps} ({progress:.1f}%) - Time: {current_time:.1f}s")
                    
                    # Check tracking accuracy
                    self._check_tracking_accuracy(trajectory_idx - 1)
                
                # Handle trajectory end
                if trajectory_idx >= start_timestep + num_timesteps:
                    if loop:
                        print("   🔄 Looping trajectory...")
                        trajectory_idx = start_timestep
                        self._reset_to_trajectory_position(start_timestep)
                    else:
                        break
                
                # Respect playback speed
                if playback_speed < 2.0:  # Only add delay for reasonable speeds
                    time.sleep((1.0 / self.frequency) / playback_speed)
            
            print("✅ Trajectory following completed successfully!")
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️ Trajectory following stopped by user")
            return True
            
        except Exception as e:
            print(f"\n❌ Trajectory following failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _reset_to_trajectory_position(self, timestep: int):
        """Reset Genesis environment to specific trajectory position"""
        # Get trajectory state at timestep
        dof_pos = self.trajectory_data['dof_pos'][timestep:timestep+1]
        root_pos = self.trajectory_data['root_pos'][timestep:timestep+1]
        root_quat = self.trajectory_data['root_quat'][timestep:timestep+1]
        
        # Apply to Genesis
        env_ids = torch.tensor([0], device=self.device)
        self.env.robot.set_dofs_position(dof_pos, envs_idx=env_ids, zero_velocity=True)
        self.env.robot.set_pos(root_pos, envs_idx=env_ids, zero_velocity=True)
        self.env.robot.set_quat(root_quat, envs_idx=env_ids, zero_velocity=True)
        
        print(f"   Reset to timestep {timestep} ({timestep/self.frequency:.1f}s)")
    
    def _check_tracking_accuracy(self, timestep: int):
        """Check how accurately Genesis is following the trajectory"""
        # Get target state
        target_dof_pos = self.trajectory_data['dof_pos'][timestep]  # [num_dofs]
        target_root_pos = self.trajectory_data['root_pos'][timestep]  # [3]
        
        # Get current Genesis state
        current_dof_pos = self.env.dof_pos[0]  # [num_dofs]
        current_root_pos = self.env.root_pos[0]  # [3]
        
        # Compute tracking errors
        dof_error = torch.mean(torch.abs(current_dof_pos - target_dof_pos)).item()
        root_error = torch.mean(torch.abs(current_root_pos - target_root_pos)).item()
        
        print(f"     Tracking - DOF error: {dof_error:.4f}, Root error: {root_error:.4f}")
        
        # Warning for large errors
        if dof_error > 0.1:
            print(f"     ⚠️ Large DOF tracking error detected!")
        if root_error > 0.05:
            print(f"     ⚠️ Large root tracking error detected!")
    
    def verify_joint_mapping(self) -> bool:
        """
        Verify joint mapping by testing individual joint movements
        """
        print(f"\n🔍 Verifying Joint Mapping:")
        print("   Testing individual joint control...")
        
        # Reset to neutral pose
        self._reset_to_trajectory_position(1000)  # Use a mid-trajectory pose
        
        # Test each mapped joint
        successful_joints = []
        failed_joints = []
        
        for loco_joint, genesis_joint in self.data_bridge.joint_mapping.items():
            print(f"\n   Testing: {loco_joint} → {genesis_joint}")
            
            try:
                # Get Genesis DOF index
                genesis_idx = self.env.dof_names.index(genesis_joint)
                
                # Create test positions
                neutral_pos = torch.zeros(1, self.env.num_dofs, device=self.device)
                test_pos = neutral_pos.clone()
                
                # Apply small test movement to this joint
                test_pos[0, genesis_idx] = 0.2  # 0.2 radian movement
                
                # Apply and observe
                self.env.robot.control_dofs_position(test_pos)
                
                # Step simulation multiple times to reach target
                for _ in range(50):
                    self.env.scene.step()
                    time.sleep(0.01)
                
                # Check if movement occurred
                self.env._update_robot_state()
                current_pos = self.env.dof_pos[0, genesis_idx].item()
                
                if abs(current_pos - 0.2) < 0.1:  # Reasonable tolerance
                    print(f"     ✅ Success - Joint moved to {current_pos:.3f}")
                    successful_joints.append(loco_joint)
                else:
                    print(f"     ❌ Failed - Expected 0.2, got {current_pos:.3f}")
                    failed_joints.append(loco_joint)
                
                # Return to neutral
                self.env.robot.control_dofs_position(neutral_pos)
                for _ in range(20):
                    self.env.scene.step()
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"     ❌ Error testing joint: {e}")
                failed_joints.append(loco_joint)
        
        # Summary
        success_rate = len(successful_joints) / len(self.data_bridge.joint_mapping)
        print(f"\n📊 Joint Mapping Verification Results:")
        print(f"   Successful: {len(successful_joints)}/{len(self.data_bridge.joint_mapping)} ({success_rate*100:.1f}%)")
        print(f"   Failed: {len(failed_joints)}")
        
        if failed_joints:
            print(f"   Failed joints: {failed_joints}")
        
        return success_rate > 0.9  # 90% success rate threshold
    
    def interactive_mode(self):
        """
        Interactive mode for trajectory verification
        """
        print(f"\n🎮 Interactive Trajectory Verification")
        print("Commands:")
        print("   'f' - Follow full trajectory")
        print("   's XXXX' - Start from specific timestep")
        print("   'l' - Loop current segment")
        print("   'v' - Verify joint mapping")
        print("   'r' - Reset to trajectory start")
        print("   'q' - Quit")
        print("=" * 50)
        
        current_start = 0
        current_length = 2000
        
        while True:
            print(f"\nCurrent segment: {current_start} to {current_start + current_length}")
            command = input("Command: ").strip().lower()
            
            if command == 'q':
                break
            elif command == 'f':
                self.follow_trajectory(start_timestep=0, num_timesteps=5000, loop=False)
            elif command.startswith('s '):
                try:
                    start_timestep = int(command.split()[1])
                    current_start = max(0, min(start_timestep, self.n_timesteps - 1000))
                    print(f"Set start timestep to {current_start}")
                except:
                    print("Invalid timestep. Use: s 1000")
            elif command == 'l':
                self.follow_trajectory(start_timestep=current_start, num_timesteps=current_length, loop=True)
            elif command == 'v':
                self.verify_joint_mapping()
            elif command == 'r':
                self._reset_to_trajectory_position(current_start)
            else:
                print("Unknown command")


def main():
    """Main verification function"""
    print("🔍 Genesis Trajectory Following Verification")
    print("=" * 50)
    
    # Configuration
    print("Select verification mode:")
    print("1. Automatic verification (follow trajectory once)")
    print("2. Interactive mode (manual control)")
    print("3. Joint mapping test only")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    # Dataset selection
    datasets = ["walk", "run", "jump"]  # Common LocoMujoco datasets
    print(f"\nAvailable datasets: {datasets}")
    dataset = input(f"Select dataset (default: walk): ").strip() or "walk"
    
    try:
        # Initialize trajectory follower
        follower = TrajectoryFollower(show_viewer=True, dataset_name=dataset)
        
        if choice == "1":
            # Automatic verification
            print(f"\n🤖 Running automatic verification...")
            
            # First verify joint mapping
            mapping_success = follower.verify_joint_mapping()
            
            if mapping_success:
                # Then follow trajectory
                follower.follow_trajectory(start_timestep=1000, num_timesteps=2000, loop=False)
            else:
                print("⚠️ Joint mapping verification failed. Check joint names and mappings.")
        
        elif choice == "2":
            # Interactive mode
            follower.interactive_mode()
        
        else:
            # Joint mapping test only
            follower.verify_joint_mapping()
        
        print("\n✅ Verification completed!")
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()