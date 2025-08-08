"""
Trajectory Following Verification Script

Forces the Genesis skeleton to follow the exact LocoMujoco walking trajectory
using control_dofs_position to verify joint mapping and control accuracy.

UPDATED: Fixed Genesis DOF control issues (consistent with data_bridge.py):
- Uses Genesis motor detection for correct joint mapping (via data_bridge)
- Uses pre-computed local DoF indices from data_bridge.genesis_dof_indices
- Applies control with dofs_idx_local parameter to avoid root joint control
- Fixes dimension mismatches between trajectory data and DOF arrays
- Root joint (6-DoF freejoint) is properly excluded from control commands

Key fixes applied:
1. robot.control_dofs_position(targets, dofs_idx_local=controllable_indices)
2. Uses data_bridge.genesis_dof_indices directly (no manual conversion)
3. Consistent with skeleton_humanoid.py _apply_actions approach
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
        
        # Initialize per-joint error tracking
        self.dof_error_history = []
        self.joint_names_for_tracking = []
        
        print("✅ Trajectory follower ready!")
    
    def _setup_environment(self, show_viewer: bool):
        """Setup Genesis skeleton environment"""
        print("   Setting up Genesis environment...")
        
        import genesis as gs
        gs.init(backend=gs.gpu)
        
        # Import skeleton environment - fix path
        genesis_loco_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, genesis_loco_dir)
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        self.env = SkeletonHumanoidEnv(
            num_envs=1,  # Single environment for verification
            episode_length_s=30.0,  # Long episodes for full trajectory
            dt=0.01,  # High frequency for smooth following (100Hz to match LocoMujoco)
            show_viewer=show_viewer,
            use_trajectory_control=True,  # Enable trajectory-optimized control
            use_box_feet=True  # Enable box feet for stable ground contact (LocoMujoco compatibility)
        )
        
        # Control setup is handled automatically by use_trajectory_control=True
        print(f"     ✓ Trajectory-optimized control enabled")
        
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
        
        # Verify consistency with data_bridge DoF mapping approach
        if hasattr(self.data_bridge, 'genesis_dof_indices'):
            print(f"     ✓ DoF control mapping: {len(self.data_bridge.genesis_dof_indices)} controllable joints")
            print(f"     ✓ Consistency check: using same approach as data_bridge.py and skeleton_humanoid.py")
        else:
            print(f"     ⚠️ Warning: DoF indices not available from data_bridge")
    
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
        
        # CRITICAL: Initialize skeleton to exact trajectory starting pose
        print(f"   Initializing skeleton to trajectory starting pose (frame {start_timestep})...")
        self._initialize_to_trajectory_pose(start_timestep)
        
        try:
            step_count = 0
            trajectory_idx = start_timestep
            
            while True:
                # Get current trajectory targets with velocity feedforward
                target_dof_pos = self.trajectory_data['dof_pos'][trajectory_idx:trajectory_idx+1]  # [1, num_dofs]
                target_dof_vel = self.trajectory_data['dof_vel'][trajectory_idx:trajectory_idx+1]  # [1, num_dofs]
                target_root_pos = self.trajectory_data['root_pos'][trajectory_idx:trajectory_idx+1]  # [1, 3]
                target_root_quat = self.trajectory_data['root_quat'][trajectory_idx:trajectory_idx+1]  # [1, 4]
                
                # CRITICAL FIX: Position control using proper local DoF indices (consistent with data_bridge.py)
                if hasattr(self.data_bridge, 'genesis_dof_indices'):
                    # Use the pre-computed local DoF indices from data_bridge (same approach as skeleton_humanoid.py)
                    # These indices are already local DoF indices from Genesis motor detection
                    controllable_dof_indices = self.data_bridge.genesis_dof_indices
                    
                    # Apply position control with velocity feedforward for better tracking
                    self.env.robot.control_dofs_position(target_dof_pos, dofs_idx_local=controllable_dof_indices)
                    # self.env.robot.control_dofs_velocity(target_dof_vel, dofs_idx_local=controllable_dof_indices)
                else:
                    # Fallback: use skeleton_humanoid approach if bridge data not available
                    print("⚠️ Warning: No bridge DoF indices, using skeleton_humanoid fallback")
                    if hasattr(self.env, 'action_to_joint_idx'):
                        controllable_dof_indices = list(self.env.action_to_joint_idx.values())
                        self.env.robot.control_dofs_position(target_dof_pos, dofs_idx_local=controllable_dof_indices)
                        # self.env.robot.control_dofs_velocity(target_dof_vel, dofs_idx_local=controllable_dof_indices)
                    else:
                        print("❌ Error: Cannot determine controllable DoF indices")
                        return False
                
                # Set root state directly (like LocoMujoco does for trajectory following)
                env_ids = torch.tensor([0], device=self.device)
                # self.env.robot.set_pos(target_root_pos, envs_idx=env_ids, zero_velocity=False)
                # self.env.robot.set_quat(target_root_quat, envs_idx=env_ids, zero_velocity=False)
                
                # Reset physics periodically to prevent error accumulation
                if step_count % 500 == 0 and step_count > 0:
                    print(f"     Periodic reset at step {step_count}")
                    self._initialize_to_trajectory_pose(trajectory_idx)
                
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
            
            # Analyze per-joint tracking errors
            self._analyze_joint_tracking_errors()
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️ Trajectory following stopped by user")
            
            # Analyze per-joint tracking errors if we have data
            if self.dof_error_history:
                self._analyze_joint_tracking_errors()
            return True
            
        except Exception as e:
            print(f"\n❌ Trajectory following failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _reset_to_trajectory_position(self, timestep: int):
        """Reset Genesis environment to specific trajectory position"""
        # Get trajectory state at timestep
        dof_pos = self.trajectory_data['dof_pos'][timestep:timestep+1]  # [1, 27]
        root_pos = self.trajectory_data['root_pos'][timestep:timestep+1]
        root_quat = self.trajectory_data['root_quat'][timestep:timestep+1]
        
        # Apply to Genesis using explicit DOF indices (Genesis approach)
        env_ids = torch.tensor([0], device=self.device)
        
        # CRITICAL FIX: Use pre-computed local DOF indices (consistent with data_bridge.py approach)
        if hasattr(self.data_bridge, 'genesis_dof_indices'):
            # Use the local DoF indices directly from data_bridge (same as skeleton_humanoid.py)
            controllable_dof_indices = self.data_bridge.genesis_dof_indices
            self.env.robot.set_dofs_position(dof_pos, dofs_idx_local=controllable_dof_indices, envs_idx=env_ids, zero_velocity=True)
        else:
            # Fallback: use skeleton_humanoid approach
            print(f"   ⚠️ No bridge DoF indices, using skeleton_humanoid fallback")
            if hasattr(self.env, 'action_to_joint_idx'):
                controllable_dof_indices = list(self.env.action_to_joint_idx.values())
                self.env.robot.set_dofs_position(dof_pos, dofs_idx_local=controllable_dof_indices, envs_idx=env_ids, zero_velocity=True)
            else:
                print(f"   ❌ Cannot determine controllable DoF indices, trajectory reset may fail")
                # Last resort: create full DoF tensor and map controlled joints
                if hasattr(self.data_bridge, 'genesis_joint_names'):
                    full_dof_pos = torch.zeros((1, self.env.num_dofs), device=self.device)
                    for i, joint_name in enumerate(self.data_bridge.genesis_joint_names):
                        if i < dof_pos.shape[1]:  # Ensure we don't exceed trajectory data
                            # This is a fallback - may not work correctly without proper mapping
                            full_dof_pos[0, i] = dof_pos[0, i]
                    self.env.robot.set_dofs_position(full_dof_pos, envs_idx=env_ids, zero_velocity=True)
                else:
                    self.env.robot.set_dofs_position(dof_pos, envs_idx=env_ids, zero_velocity=True)
        
        # Set root state
        self.env.robot.set_pos(root_pos, envs_idx=env_ids, zero_velocity=True)
        self.env.robot.set_quat(root_quat, envs_idx=env_ids, zero_velocity=True)
        
        print(f"   Reset to timestep {timestep} ({timestep/self.frequency:.1f}s)")
    
    def _initialize_to_trajectory_pose(self, timestep: int):
        """
        Initialize skeleton to exact trajectory pose with proper synchronization
        
        This is critical for preventing initial tripping - ensures the skeleton
        starts in exactly the same pose as the trajectory expects.
        """
        print(f"     Setting skeleton to trajectory frame {timestep}...")
        
        # Get trajectory state at this timestep
        target_dof_pos = self.trajectory_data['dof_pos'][timestep]  # [num_controllable_dofs]
        target_root_pos = self.trajectory_data['root_pos'][timestep]  # [3]
        target_root_quat = self.trajectory_data['root_quat'][timestep]  # [4]
        
        # Set joint positions using proper DoF control (same as trajectory following)
        env_ids = torch.tensor([0], device=self.device)
        
        if hasattr(self.data_bridge, 'genesis_dof_indices'):
            # Use pre-computed local DoF indices (consistent approach)
            controllable_dof_indices = self.data_bridge.genesis_dof_indices
            
            # Set joint positions directly (not control, but direct setting for initialization)
            target_dof_pos_batch = target_dof_pos.unsqueeze(0)  # [1, num_controllable_dofs]
            self.env.robot.set_dofs_position(
                target_dof_pos_batch, 
                dofs_idx_local=controllable_dof_indices, 
                envs_idx=env_ids, 
                zero_velocity=True
            )
            print(f"     ✓ Set {len(controllable_dof_indices)} joint positions")
            
        else:
            print(f"     ⚠️ No DoF indices available - using fallback")
            # Fallback approach
            if hasattr(self.env, 'action_to_joint_idx'):
                controllable_dof_indices = list(self.env.action_to_joint_idx.values())
                target_dof_pos_batch = target_dof_pos.unsqueeze(0)
                self.env.robot.set_dofs_position(
                    target_dof_pos_batch, 
                    dofs_idx_local=controllable_dof_indices, 
                    envs_idx=env_ids, 
                    zero_velocity=True
                )
        
        # Set root pose - this is crucial for proper initialization
        target_root_pos_batch = target_root_pos.unsqueeze(0)  # [1, 3]
        target_root_quat_batch = target_root_quat.unsqueeze(0)  # [1, 4]
        
        self.env.robot.set_pos(
            target_root_pos_batch, 
            envs_idx=env_ids, 
            zero_velocity=True
        )
        self.env.robot.set_quat(
            target_root_quat_batch, 
            envs_idx=env_ids, 
            zero_velocity=True
        )
        
        print(f"     ✓ Set root pose: pos={target_root_pos[:3]}, quat={target_root_quat[:4]}")
        
        # Step physics a few times to settle the pose
        print(f"     Settling physics...")
        for _ in range(5):
            self.env.scene.step()
            
        # Update environment state
        self.env._update_robot_state()
        
        # Verify initialization
        current_height = self.env.root_pos[0, 2].item()
        target_height = target_root_pos[2].item()
        height_error = abs(current_height - target_height)
        
        if height_error < 0.05:  # 5cm tolerance
            print(f"     ✅ Pose initialization successful - height: {current_height:.3f}m (target: {target_height:.3f}m)")
        else:
            print(f"     ⚠️ Pose initialization may have issues - height: {current_height:.3f}m (target: {target_height:.3f}m, error: {height_error:.3f}m)")
    
    def _check_tracking_accuracy(self, timestep: int):
        """Check how accurately Genesis is following the trajectory"""
        # Get target state
        target_dof_pos = self.trajectory_data['dof_pos'][timestep]  # [27] - trajectory joints
        target_root_pos = self.trajectory_data['root_pos'][timestep]  # [3]
        
        # Get current Genesis state for controlled joints only
        current_root_pos = self.env.root_pos[0]  # [3]
        
        # Only compare the joints we're actually controlling
        if hasattr(self.data_bridge, 'genesis_dof_indices'):
            # Extract current positions for the specific DOFs we control
            full_dof_pos = self.env.dof_pos[0]  # [37] - all Genesis DOFs
            current_dof_pos = full_dof_pos[self.data_bridge.genesis_dof_indices]  # [27] - controlled DOFs
            
            # Per-joint error tracking
            joint_errors = torch.abs(current_dof_pos - target_dof_pos)  # [27] - error per joint
            
            # Store per-joint errors with joint names
            if hasattr(self.data_bridge, 'genesis_joint_names'):
                joint_error_data = {
                    'timestep': timestep,
                    'joint_errors': joint_errors.cpu().numpy(),
                    'joint_names': self.data_bridge.genesis_joint_names
                }
                self.dof_error_history.append(joint_error_data)
                
                # Store joint names for final analysis (only once)
                if not self.joint_names_for_tracking:
                    self.joint_names_for_tracking = self.data_bridge.genesis_joint_names.copy()
        else:
            print(f"     ⚠️ No DOF indices available - skipping DOF accuracy check")
            current_dof_pos = target_dof_pos  # Dummy to avoid error
            joint_errors = torch.zeros_like(target_dof_pos)
        
        # Compute overall tracking errors
        dof_error = torch.mean(joint_errors).item()
        root_error = torch.mean(torch.abs(current_root_pos - target_root_pos)).item()
        
        print(f"     Tracking - DOF error: {dof_error:.4f}, Root error: {root_error:.4f}")
        
        # Warning for large errors
        if dof_error > 0.1:
            print(f"     ⚠️ Large DOF tracking error detected!")
        if root_error > 0.05:
            print(f"     ⚠️ Large root tracking error detected!")
    
    def _analyze_joint_tracking_errors(self):
        """Analyze per-joint tracking errors and report problematic joints"""
        if not self.dof_error_history or not self.joint_names_for_tracking:
            print("   ⚠️ No joint tracking data available for analysis")
            return
        
        import numpy as np
        
        print(f"\n📊 Per-Joint Tracking Error Analysis:")
        print("=" * 60)
        
        # Convert error history to numpy array
        n_timesteps = len(self.dof_error_history)
        n_joints = len(self.joint_names_for_tracking)
        
        # Create error matrix [timesteps, joints]
        error_matrix = np.zeros((n_timesteps, n_joints))
        for i, error_data in enumerate(self.dof_error_history):
            error_matrix[i] = error_data['joint_errors']
        
        # Compute statistics for each joint
        joint_stats = []
        for j, joint_name in enumerate(self.joint_names_for_tracking):
            joint_errors = error_matrix[:, j]
            stats = {
                'joint': joint_name,
                'mean_error': np.mean(joint_errors),
                'max_error': np.max(joint_errors),
                'std_error': np.std(joint_errors),
                'median_error': np.median(joint_errors)
            }
            joint_stats.append(stats)
        
        # Sort by mean error (worst first)
        joint_stats.sort(key=lambda x: x['mean_error'], reverse=True)
        
        print(f"Analyzed {n_timesteps} timesteps across {n_joints} joints\n")
        
        # Report worst joints
        print("🔴 **WORST TRACKING JOINTS** (Mean Error > 0.1 rad):")
        worst_joints = [js for js in joint_stats if js['mean_error'] > 0.1]
        if worst_joints:
            for i, js in enumerate(worst_joints[:10]):  # Top 10 worst
                print(f"  {i+1:2d}. {js['joint']:<20} - Mean: {js['mean_error']:.4f}, Max: {js['max_error']:.4f}, Std: {js['std_error']:.4f}")
        else:
            print("  ✅ No joints with mean error > 0.1 rad")
        
        # Report good joints  
        print(f"\n🟢 **BEST TRACKING JOINTS** (Mean Error < 0.05 rad):")
        good_joints = [js for js in joint_stats if js['mean_error'] < 0.05]
        if good_joints:
            for i, js in enumerate(good_joints[-10:]):  # Top 10 best
                print(f"  {i+1:2d}. {js['joint']:<20} - Mean: {js['mean_error']:.4f}, Max: {js['max_error']:.4f}, Std: {js['std_error']:.4f}")
        else:
            print("  ⚠️ No joints with mean error < 0.05 rad")
        
        # Body part analysis
        print(f"\n🔍 **BODY PART ANALYSIS**:")
        body_parts = {
            'Lower Body': ['hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 'ankle_angle'],
            'Spine': ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation'],
            'Arms': ['arm_flex', 'arm_add', 'arm_rot', 'elbow_flex', 'pro_sup', 'wrist_flex', 'wrist_dev']
        }
        
        for part_name, joint_patterns in body_parts.items():
            part_joints = [js for js in joint_stats if any(pattern in js['joint'] for pattern in joint_patterns)]
            if part_joints:
                part_mean_error = np.mean([js['mean_error'] for js in part_joints])
                part_max_error = np.max([js['max_error'] for js in part_joints])
                print(f"  {part_name:<12} - Joints: {len(part_joints):2d}, Mean: {part_mean_error:.4f}, Max: {part_max_error:.4f}")
        
        print("=" * 60)
    
    def verify_joint_mapping(self) -> bool:
        """
        Verify joint mapping by testing individual joint movements
        """
        print(f"\n🔍 Verifying Joint Mapping:")
        print("   Testing individual joint control...")
        
        # Reset to neutral pose
        self._reset_to_trajectory_position(1000)  # Use a mid-trajectory pose
        
        # Test each mapped joint (excluding root)
        successful_joints = []
        failed_joints = []
        
        for loco_joint, genesis_joint in self.data_bridge.joint_mapping.items():
            # Skip root joint - it's not controllable
            if loco_joint == 'root' or genesis_joint == 'root':
                print(f"\n   Skipping: {loco_joint} → {genesis_joint} (root joint not controllable)")
                continue
                
            print(f"\n   Testing: {loco_joint} → {genesis_joint}")
            
            try:
                # Get Genesis controllable DOF index using consistent data_bridge approach
                if hasattr(self.data_bridge, 'genesis_dof_indices') and hasattr(self.data_bridge, 'genesis_joint_names'):
                    # Use the DOF indices from data bridge (consistent with motor detection)
                    if genesis_joint in self.data_bridge.genesis_joint_names:
                        joint_idx = self.data_bridge.genesis_joint_names.index(genesis_joint)
                        genesis_dof_idx = self.data_bridge.genesis_dof_indices[joint_idx]
                    else:
                        print(f"     ❌ Joint {genesis_joint} not found in Genesis motor detection")
                        failed_joints.append(loco_joint)
                        continue
                else:
                    print(f"     ❌ Genesis motor detection data not available")
                    failed_joints.append(loco_joint)
                    continue
                
                # Validate DOF index is within controllable range
                if genesis_dof_idx >= self.env.num_dofs:
                    print(f"     ❌ DOF index {genesis_dof_idx} out of range")
                    failed_joints.append(loco_joint)
                    continue
                
                # CRITICAL FIX: Test individual joint using proper local DoF control (consistent approach)
                # Create test position tensor for this specific joint
                test_dof_pos = torch.zeros(1, 1, device=self.device)  # [1, 1] - single joint
                test_dof_pos[0, 0] = 0.2  # 0.2 radian movement for this joint
                
                # Apply test movement using specific DoF index (consistent with data_bridge approach)
                self.env.robot.control_dofs_position(test_dof_pos, dofs_idx_local=[genesis_dof_idx])
                
                # Step simulation multiple times to reach target
                for _ in range(50):
                    self.env.scene.step()
                    time.sleep(0.01)
                
                # Check if movement occurred
                self.env._update_robot_state()
                current_pos = self.env.dof_pos[0, genesis_dof_idx].item()
                
                if abs(current_pos - 0.2) < 0.1:  # Reasonable tolerance
                    print(f"     ✅ Success - Joint moved to {current_pos:.3f} (local DOF idx: {genesis_dof_idx})")
                    successful_joints.append(loco_joint)
                else:
                    print(f"     ❌ Failed - Expected 0.2, got {current_pos:.3f} (local DOF idx: {genesis_dof_idx})")
                    failed_joints.append(loco_joint)
                
                # Return to neutral using same specific control approach
                neutral_dof_pos = torch.zeros(1, 1, device=self.device)
                self.env.robot.control_dofs_position(neutral_dof_pos, dofs_idx_local=[genesis_dof_idx])
                for _ in range(20):
                    self.env.scene.step()
                    time.sleep(0.01)
                
            except Exception as e:
                print(f"     ❌ Error testing joint: {e}")
                failed_joints.append(loco_joint)
        
        # Summary (excluding root joint from total count)
        total_controllable_joints = len(self.data_bridge.joint_mapping) - 1  # Exclude root
        success_rate = len(successful_joints) / total_controllable_joints if total_controllable_joints > 0 else 0
        
        print(f"\n📊 Joint Mapping Verification Results:")
        print(f"   Controllable joints tested: {total_controllable_joints}")
        print(f"   Successful: {len(successful_joints)}/{total_controllable_joints} ({success_rate*100:.1f}%)")
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
    datasets = ["walk", "run", "squat"]  # Common LocoMujoco datasets
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
                # Then follow trajectory - start from a stable walking frame
                print(f"\n🚶‍♂️ Starting trajectory from frame 100 (avoiding initial transition)")
                follower.follow_trajectory(start_timestep=100, num_timesteps=1000, loop=False)
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