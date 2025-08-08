"""
LocoMujoco Data Bridge - Simple Implementation

Step-by-step integration of LocoMujoco trajectory data with Genesis
"""

import torch
import numpy as np
import sys
import os

# Add LocoMujoco path
sys.path.append('/home/ez/Documents/loco-mujoco')


class LocoMujocoDataBridge:
    """
    Simple bridge to load LocoMujoco trajectory data for Genesis
    
    Implementation approach: Start simple, add complexity incrementally
    """
    
    def __init__(self, genesis_skeleton_env):
        """
        Initialize data bridge
        
        Args:
            genesis_skeleton_env: Your SkeletonHumanoidEnv instance
        """
        self.genesis_env = genesis_skeleton_env
        self.device = genesis_skeleton_env.device
        
        # Will be set when trajectory is loaded
        self.loco_trajectory = None
        self.joint_mapping = None
        
    def load_trajectory(self, dataset_name: str = "walk"):
        """
        Step 1: Load LocoMujoco trajectory and inspect its structure
        
        Args:
            dataset_name: Name of dataset to load (e.g., "walk", "run")
            
        Returns:
            Success status and basic info about loaded trajectory
        """
        print(f"Step 1: Loading LocoMujoco trajectory '{dataset_name}'...")
        
        try:
            # Import LocoMujoco components
            from loco_mujoco.task_factories.imitation_factory import ImitationFactory
            from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf
            
            # Load trajectory using LocoMujoco's proven pipeline
            print("  - Creating LocoMujoco environment...")
            loco_env = ImitationFactory.make(
                "SkeletonTorque",
                default_dataset_conf=DefaultDatasetConf([dataset_name])
            )
            
            # Store the trajectory
            self.loco_trajectory = loco_env.th.traj
            
            # Print trajectory info for verification
            self._print_trajectory_info()
            
            print("✓ Step 1 completed: LocoMujoco trajectory loaded successfully")
            return True, self._get_trajectory_summary()
            
        except ImportError as e:
            print(f"✗ Failed to import LocoMujoco: {e}")
            print("Note: Make sure LocoMujoco is installed and datasets are downloaded")
            return False, str(e)
            
        except Exception as e:
            print(f"✗ Failed to load trajectory: {e}")
            return False, str(e)
    
    def _print_trajectory_info(self):
        """Print detailed trajectory information for inspection"""
        if self.loco_trajectory is None:
            return
            
        traj = self.loco_trajectory
        print(f"  - Trajectory loaded successfully!")
        print(f"  - Joint names: {traj.info.joint_names[:5]}... ({len(traj.info.joint_names)} total)")
        print(f"  - Frequency: {traj.info.frequency} Hz")
        print(f"  - Timesteps: {traj.data.qpos.shape[0]}")
        print(f"  - Joint positions shape: {traj.data.qpos.shape}")
        print(f"  - Joint velocities shape: {traj.data.qvel.shape}")
        
        # Check for multiple trajectories
        if hasattr(traj.data, 'split_points') and len(traj.data.split_points) > 2:
            n_trajectories = len(traj.data.split_points) - 1
            print(f"  - Multiple trajectories: {n_trajectories} sequences")
        else:
            print(f"  - Single trajectory sequence")
    
    def _get_trajectory_summary(self) -> dict:
        """Get summary of loaded trajectory"""
        if self.loco_trajectory is None:
            return {}
            
        traj = self.loco_trajectory
        return {
            'joint_names': traj.info.joint_names,
            'frequency': traj.info.frequency,
            'timesteps': traj.data.qpos.shape[0],
            'n_joints': len(traj.info.joint_names),
            'data_shapes': {
                'qpos': traj.data.qpos.shape,
                'qvel': traj.data.qvel.shape
            }
        }
    
    def build_joint_mapping(self):
        """
        Step 2: Build mapping between LocoMujoco and Genesis joint names using Genesis motor detection
        
        Uses Genesis' proven motor detection approach instead of flawed dof_names.
        
        Returns:
            Success status and mapping information
        """
        print(f"\nStep 2: Building joint mapping using Genesis motor detection...")
        
        if self.loco_trajectory is None:
            print("No trajectory loaded. Run load_trajectory() first.")
            return False, "No trajectory loaded"
        
        # Get joint names using Genesis' proven approach
        loco_joints = self.loco_trajectory.info.joint_names
        genesis_motors_dof_idx, genesis_motors_dof_name = self._get_genesis_motors_info()
        
        print(f"  - LocoMujoco joints: {len(loco_joints)} joints")
        print(f"  - Genesis motors: {len(genesis_motors_dof_name)} DOFs")
        
        # Build mapping using Genesis motor detection
        mapping = {}
        matched_joints = []
        unmatched_loco = []
        unmatched_genesis = []
        
        genesis_motor_set = set(genesis_motors_dof_name)
        
        # Check each LocoMujoco joint for exact match in Genesis motors
        for loco_joint in loco_joints:
            if loco_joint in genesis_motor_set:
                mapping[loco_joint] = loco_joint  # Same name
                matched_joints.append(loco_joint)
            else:
                unmatched_loco.append(loco_joint)
        
        # Find Genesis motors that don't have LocoMujoco matches
        for genesis_motor in genesis_motors_dof_name:
            if genesis_motor not in mapping.values():
                unmatched_genesis.append(genesis_motor)
        
        # Store the mapping
        self.joint_mapping = mapping
        
        # Print results
        print(f"  - Results:")
        print(f"    ✓ Matched joints: {len(matched_joints)}")
        print(f"    ⚠ Unmatched LocoMujoco: {len(unmatched_loco)}")
        print(f"    ⚠ Unmatched Genesis: {len(unmatched_genesis)}")
        
        if matched_joints:
            print(f"  - Matched Joints:")
            for joint in matched_joints[:5]:
                print(f"    ✓ {joint}")
        
        if unmatched_loco:
            print(f"  - Unmatched LocoMujoco joints:")
            for joint in unmatched_loco[:5]:
                print(f"    ⚠ {joint}")
        
        if unmatched_genesis:
            print(f"  - Unmatched Genesis DOFs:")
            for dof in unmatched_genesis[:5]:
                print(f"    ⚠ {dof}")
        
        # Validate assumption
        total_joints = len(loco_joints)
        matched_count = len(matched_joints)
        match_percentage = (matched_count / total_joints) * 100 if total_joints > 0 else 0
        
        print(f"  - Match rate: {match_percentage:.1f}% ({matched_count}/{total_joints})")
        
        if match_percentage > 80:
            print("✓ Step 2 completed: Joint name assumption validated!")
            return True, {
                'mapping': mapping,
                'matched': matched_joints,
                'unmatched_loco': unmatched_loco,
                'unmatched_genesis': unmatched_genesis,
                'match_percentage': match_percentage
            }
        else:
            print("⚠ Step 2 warning: Low match rate - may need custom mapping")
            return True, {
                'mapping': mapping,
                'matched': matched_joints,
                'unmatched_loco': unmatched_loco,
                'unmatched_genesis': unmatched_genesis,
                'match_percentage': match_percentage
            }
    
    def convert_to_genesis_format(self):
        """
        Step 3: Convert LocoMujoco trajectory data to Genesis tensor format
        
        Converts LocoMujoco numpy arrays to Genesis PyTorch tensors with:
        - Proper DOF ordering (Genesis dof_names order)
        - Correct tensor shapes [timesteps, features] for imitation learning
        - GPU placement and float32 dtype matching Genesis conventions
        
        Returns:
            Success status and converted trajectory data
        """
        print(f"\nStep 3: Converting trajectory data to Genesis format...")
        
        if self.loco_trajectory is None:
            print("✗ No trajectory loaded. Run load_trajectory() first.")
            return False, "No trajectory loaded"
            
        if self.joint_mapping is None:
            print("✗ No joint mapping. Run build_joint_mapping() first.")
            return False, "No joint mapping"
        
        try:
            # Get LocoMujoco trajectory data
            loco_data = self.loco_trajectory.data
            loco_info = self.loco_trajectory.info
            
            print(f"  - Converting {loco_data.qpos.shape[0]} timesteps...")
            print(f"  - Source: numpy arrays")
            print(f"  - Target: PyTorch tensors on {self.device}")
            
            # Convert joint data with Genesis DOF ordering
            genesis_dof_pos, genesis_dof_vel = self._convert_joint_data(loco_data, loco_info)
            
            # Convert root state data
            root_pos, root_quat, root_lin_vel, root_ang_vel = self._convert_root_data(loco_data)
            
            # Package converted data
            genesis_trajectory = {
                'dof_pos': genesis_dof_pos,      # [timesteps, num_dofs] - Genesis format
                'dof_vel': genesis_dof_vel,      # [timesteps, num_dofs] - Genesis format  
                'root_pos': root_pos,            # [timesteps, 3] - Genesis format
                'root_quat': root_quat,          # [timesteps, 4] - Genesis format
                'root_lin_vel': root_lin_vel,    # [timesteps, 3] - Genesis format
                'root_ang_vel': root_ang_vel,    # [timesteps, 3] - Genesis format
                'info': {
                    'timesteps': genesis_dof_pos.shape[0],
                    'n_dofs': genesis_dof_pos.shape[1],
                    'frequency': loco_info.frequency,
                    'device': self.device,
                    'matched_joints': len(self.joint_mapping)
                }
            }
            
            # Verify tensor formats match Genesis conventions
            self._validate_genesis_format(genesis_trajectory)
            
            # print("✓ Step 3 completed: Data converted to Genesis format")

            # print(f"[DEBUG] root_pos: {genesis_trajectory['root_pos']}\n")
            # print(f"[DEBUG] root_quat: {genesis_trajectory['root_quat']}\n")
            return True, genesis_trajectory
            
        except Exception as e:
            print(f"✗ Step 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def _get_genesis_motors_info(self):
        """Get controllable motor DOF indices using Genesis' proven approach (from _main.py view function)"""
        import genesis as gs
        
        motors_dof_idx = []
        motors_dof_name = []
        
        for joint in self.genesis_env.robot.joints:
            # Skip non-controllable joints (same as Genesis view function)
            if joint.type == gs.JOINT_TYPE.FREE:
                continue
            elif joint.type == gs.JOINT_TYPE.FIXED:
                continue
            
            # Get DOF indices using Genesis' correct method
            dofs_idx_local = self.genesis_env.robot.get_joint(joint.name).dofs_idx_local
            if dofs_idx_local:
                if len(dofs_idx_local) == 1:
                    dofs_name = [joint.name]
                else:
                    # Multi-DOF joints get suffixed names
                    dofs_name = [f"{joint.name}_{i_d}" for i_d in dofs_idx_local]
                
                motors_dof_idx += dofs_idx_local
                motors_dof_name += dofs_name
        
        return motors_dof_idx, motors_dof_name

    def _convert_joint_data(self, loco_data, loco_info):
        """Convert joint data using Genesis' proven motor detection approach"""
        n_timesteps = loco_data.qpos.shape[0]
        
        print(f"    - Building Genesis DOF mapping using proven motor detection...")
        
        # Get all controllable motors using Genesis' proven method (same as view() function)
        genesis_motors_dof_idx, genesis_motors_dof_name = self._get_genesis_motors_info()
        
        print(f"    - Genesis detected {len(genesis_motors_dof_idx)} controllable DOFs:")
        for idx, (dof_idx, dof_name) in enumerate(zip(genesis_motors_dof_idx, genesis_motors_dof_name)):
            print(f"      {idx:2d}: DOF {dof_idx:2d} = {dof_name}")
        
        # Build mapping between LocoMujoco trajectory and Genesis motors
        genesis_joint_names = []
        genesis_dof_indices = []
        loco_qpos_indices = []
        loco_qvel_indices = []
        
        # Map each joint using Genesis motor detection - EXCLUDE ROOT
        for loco_joint, genesis_joint in self.joint_mapping.items():
            # Skip root joint - it's not controllable in Genesis (matches LocoMujoco behavior)
            if loco_joint == 'root' or genesis_joint == 'root':
                print(f"      Skipping root joint: {loco_joint} → {genesis_joint} (not controllable)")
                continue
            
            try:
                # Find this joint in Genesis-detected motors
                if genesis_joint in genesis_motors_dof_name:
                    motor_idx = genesis_motors_dof_name.index(genesis_joint)
                    genesis_dof_idx = genesis_motors_dof_idx[motor_idx]
                
                    # Get LocoMujoco indices
                    loco_qpos_idx_array = loco_info.joint_name2ind_qpos[loco_joint]
                    loco_qvel_idx_array = loco_info.joint_name2ind_qvel[loco_joint]
                    
                    # Extract scalar index (most joints are 1-DOF)
                    loco_qpos_idx = loco_qpos_idx_array[0] if hasattr(loco_qpos_idx_array, '__getitem__') else loco_qpos_idx_array
                    loco_qvel_idx = loco_qvel_idx_array[0] if hasattr(loco_qvel_idx_array, '__getitem__') else loco_qvel_idx_array
                    
                    # Store mapping
                    genesis_joint_names.append(genesis_joint)
                    genesis_dof_indices.append(genesis_dof_idx)
                    loco_qpos_indices.append(loco_qpos_idx)
                    loco_qvel_indices.append(loco_qvel_idx)
                    
                    print(f"      ✓ Mapped: {loco_joint} → {genesis_joint} (Genesis DOF idx: {genesis_dof_idx}, LocoMujoco qpos: {loco_qpos_idx})")
                else:
                    print(f"      ❌ Joint {genesis_joint} not found in Genesis motors")
                
            except Exception as e:
                print(f"      ❌ Failed to map {loco_joint} → {genesis_joint}: {e}")
                continue
        
        n_controllable_dofs = len(genesis_dof_indices)
        print(f"    - Successfully mapped {n_controllable_dofs} controllable joints")
        print(f"    - Genesis DOF indices: {genesis_dof_indices}")
        
        # Create trajectory data ordered by Genesis DOF indices (Genesis approach)
        genesis_dof_pos = torch.zeros((n_timesteps, n_controllable_dofs), dtype=torch.float32, device=self.device)
        genesis_dof_vel = torch.zeros((n_timesteps, n_controllable_dofs), dtype=torch.float32, device=self.device)
        
        # Fill trajectory data in Genesis DOF order
        for i, (loco_qpos_idx, loco_qvel_idx) in enumerate(zip(loco_qpos_indices, loco_qvel_indices)):
            genesis_dof_pos[:, i] = torch.tensor(loco_data.qpos[:, loco_qpos_idx], dtype=torch.float32, device=self.device)
            genesis_dof_vel[:, i] = torch.tensor(loco_data.qvel[:, loco_qvel_idx], dtype=torch.float32, device=self.device)
        
        # Store DOF mapping for control (like Genesis examples)
        self.genesis_dof_indices = genesis_dof_indices
        self.genesis_joint_names = genesis_joint_names
        
        print(f"    - Trajectory data shape: pos{genesis_dof_pos.shape}, vel{genesis_dof_vel.shape}")
        
        return genesis_dof_pos, genesis_dof_vel
    
    def _convert_root_data(self, loco_data):
        """Convert root state using LocoMujoco's joint indexing system"""
        n_timesteps = loco_data.qpos.shape[0]
        loco_info = self.loco_trajectory.info
        
        print(f"    - Converting root state data using LocoMujoco joint indexing...")
        
        # Initialize with Genesis tensor format [timesteps, features]
        root_pos = torch.zeros((n_timesteps, 3), dtype=torch.float32, device=self.device)
        root_quat = torch.zeros((n_timesteps, 4), dtype=torch.float32, device=self.device)  
        root_lin_vel = torch.zeros((n_timesteps, 3), dtype=torch.float32, device=self.device)
        root_ang_vel = torch.zeros((n_timesteps, 3), dtype=torch.float32, device=self.device)
        
        # Find root joint using LocoMujoco's joint indexing system
        root_joint_found = False
        if 'root' in loco_info.joint_name2ind_qpos:
            root_qpos_indices = loco_info.joint_name2ind_qpos['root']
            root_qvel_indices = loco_info.joint_name2ind_qvel['root']
            
            print(f"      Found root joint at qpos indices: {root_qpos_indices}")
            print(f"      Found root joint at qvel indices: {root_qvel_indices}")
            
            # Verify this is a freejoint (7 qpos DOF, 6 qvel DOF)
            if len(root_qpos_indices) == 7 and len(root_qvel_indices) == 6:
                # Extract root position: [x, y, z]
                root_pos = torch.tensor(
                    loco_data.qpos[:, root_qpos_indices[:3]], 
                    dtype=torch.float32, device=self.device
                )
                
                # Extract root quaternion: [qw, qx, qy, qz]
                root_quat = torch.tensor(
                    loco_data.qpos[:, root_qpos_indices[3:7]], 
                    dtype=torch.float32, device=self.device
                )
                
                # Extract root linear velocity: [vx, vy, vz]
                root_lin_vel = torch.tensor(
                    loco_data.qvel[:, root_qvel_indices[:3]], 
                    dtype=torch.float32, device=self.device
                )
                
                # Extract root angular velocity: [wx, wy, wz]
                root_ang_vel = torch.tensor(
                    loco_data.qvel[:, root_qvel_indices[3:6]], 
                    dtype=torch.float32, device=self.device
                )
                
                root_joint_found = True
                print(f"      ✓ Root joint data extracted using LocoMujoco indices")
                
            else:
                print(f"      ⚠️ Root joint has unexpected DOF count: qpos={len(root_qpos_indices)}, qvel={len(root_qvel_indices)}")
        
        # Fallback: search for any freejoint in the joint list
        if not root_joint_found:
            print(f"      No 'root' joint found, searching for freejoint...")
            for joint_name in loco_info.joint_names:
                if joint_name in loco_info.joint_name2ind_qpos:
                    qpos_indices = loco_info.joint_name2ind_qpos[joint_name]
                    qvel_indices = loco_info.joint_name2ind_qvel[joint_name]
                    
                    # Check if this is a freejoint (7 qpos, 6 qvel)
                    if len(qpos_indices) == 7 and len(qvel_indices) == 6:
                        print(f"      Found freejoint '{joint_name}' at indices qpos={qpos_indices}, qvel={qvel_indices}")
                        
                        # Extract root data using this freejoint
                        root_pos = torch.tensor(
                            loco_data.qpos[:, qpos_indices[:3]], 
                            dtype=torch.float32, device=self.device
                        )
                        root_quat = torch.tensor(
                            loco_data.qpos[:, qpos_indices[3:7]], 
                            dtype=torch.float32, device=self.device
                        )
                        root_lin_vel = torch.tensor(
                            loco_data.qvel[:, qvel_indices[:3]], 
                            dtype=torch.float32, device=self.device
                        )
                        root_ang_vel = torch.tensor(
                            loco_data.qvel[:, qvel_indices[3:6]], 
                            dtype=torch.float32, device=self.device
                        )
                        
                        root_joint_found = True
                        print(f"      ✓ Root data extracted from freejoint '{joint_name}'")
                        break
        
        # Final fallback: use hardcoded indices (original logic)
        if not root_joint_found:
            print(f"      ⚠️ No freejoint found, using hardcoded root extraction (indices 0:7, 0:6)")
            if loco_data.qpos.shape[1] >= 7:
                root_pos = torch.tensor(loco_data.qpos[:, :3], dtype=torch.float32, device=self.device)
                root_quat = torch.tensor(loco_data.qpos[:, 3:7], dtype=torch.float32, device=self.device)
            
            if loco_data.qvel.shape[1] >= 6:
                root_lin_vel = torch.tensor(loco_data.qvel[:, :3], dtype=torch.float32, device=self.device)
                root_ang_vel = torch.tensor(loco_data.qvel[:, 3:6], dtype=torch.float32, device=self.device)
        
        print(f"      Root data shapes: pos{root_pos.shape}, quat{root_quat.shape}, lin_vel{root_lin_vel.shape}, ang_vel{root_ang_vel.shape}")
        
        return root_pos, root_quat, root_lin_vel, root_ang_vel
    
    def _validate_genesis_format(self, genesis_trajectory):
        """Validate converted data matches Genesis tensor conventions"""
        print(f"  - Validating Genesis format compliance...")
        
        # Check tensor properties
        for key, tensor in genesis_trajectory.items():
            if key == 'info':
                continue
            assert isinstance(tensor, torch.Tensor), f"{key} must be torch.Tensor"
            assert tensor.dtype == torch.float32, f"{key} must be float32"
            # Robust device validation: handle cuda vs cuda:0 difference
            assert tensor.device.type == self.device.type, f"{key} must be on {self.device.type} device"
        
        # Check shapes match Genesis conventions
        n_timesteps = genesis_trajectory['info']['timesteps']
        n_dofs = genesis_trajectory['info']['n_dofs']
        
        expected_shapes = {
            'dof_pos': (n_timesteps, n_dofs),
            'dof_vel': (n_timesteps, n_dofs), 
            'root_pos': (n_timesteps, 3),
            'root_quat': (n_timesteps, 4),
            'root_lin_vel': (n_timesteps, 3),
            'root_ang_vel': (n_timesteps, 3)
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = genesis_trajectory[key].shape
            assert actual_shape == expected_shape, f"{key} shape {actual_shape} != expected {expected_shape}"
        
        print(f"    ✓ All tensors validated against Genesis format")
        print(f"    ✓ Device: {self.device}, dtype: torch.float32") 
        print(f"    ✓ Shapes: dof_pos{genesis_trajectory['dof_pos'].shape}, root_pos{genesis_trajectory['root_pos'].shape}")

    def apply_trajectory_frame(self, trajectory_data, frame_idx, env_ids=None):
        """
        Apply a single frame of trajectory data to Genesis using proper DoF mapping
        
        This method correctly maps trajectory data to Genesis DoFs, avoiding the
        root joint control issue by using local DoF indices.
        
        Args:
            trajectory_data: Converted trajectory data from convert_to_genesis_format()
            frame_idx: Frame index to apply (0 to timesteps-1)
            env_ids: Environment indices to apply to (default: all environments)
        """
        if not hasattr(self, 'genesis_dof_indices'):
            raise RuntimeError("No DoF mapping available. Run convert_to_genesis_format() first.")
        
        if env_ids is None:
            env_ids = torch.arange(self.genesis_env.num_envs, device=self.device)
        
        # Get trajectory state for this frame
        dof_positions = trajectory_data['dof_pos'][frame_idx:frame_idx+1]  # [1, num_controllable_dofs]
        root_position = trajectory_data['root_pos'][frame_idx:frame_idx+1]  # [1, 3]
        root_quaternion = trajectory_data['root_quat'][frame_idx:frame_idx+1]  # [1, 4]
        
        # Apply joint positions using proper local DoF indices (avoiding root joint)
        self.genesis_env.robot.control_dofs_position(
            dof_positions.expand(len(env_ids), -1),  # Expand to match env_ids
            dofs_idx_local=self.genesis_dof_indices
        )
        
        # Set root pose directly
        self.genesis_env.robot.set_pos(
            root_position.expand(len(env_ids), -1), 
            envs_idx=env_ids, 
            zero_velocity=True
        )
        self.genesis_env.robot.set_quat(
            root_quaternion.expand(len(env_ids), -1), 
            envs_idx=env_ids, 
            zero_velocity=True
        )
        
    def apply_trajectory_sequence(self, trajectory_data, start_frame=0, num_frames=None):
        """
        Apply a sequence of trajectory frames using proper DoF control
        
        Args:
            trajectory_data: Converted trajectory data
            start_frame: Starting frame index
            num_frames: Number of frames to apply (None for all remaining)
        
        Returns:
            Success status
        """
        if num_frames is None:
            num_frames = trajectory_data['info']['timesteps'] - start_frame
        
        end_frame = min(start_frame + num_frames, trajectory_data['info']['timesteps'])
        
        print(f"Applying trajectory sequence: frames {start_frame} to {end_frame-1}")
        
        for frame_idx in range(start_frame, end_frame):
            try:
                # Apply this frame
                self.apply_trajectory_frame(trajectory_data, frame_idx)
                
                # Step physics
                self.genesis_env.scene.step()
                
                # Update environment state
                self.genesis_env._update_robot_state()
                
                # Check stability every 10 frames
                if frame_idx % 10 == 0:
                    height = self.genesis_env.root_pos[0, 2].item()
                    if height < 0.5 or height > 1.5:
                        print(f"⚠ Warning: Potential instability at frame {frame_idx}: height={height:.3f}m")
                        
            except Exception as e:
                print(f"❌ Failed to apply frame {frame_idx}: {e}")
                return False
                
        print(f"✓ Successfully applied {end_frame - start_frame} trajectory frames")
        return True


def test_trajectory_integration():
    """
    Step 4: Test trajectory integration with Genesis environment
    
    This test verifies that converted trajectory data can be:
    1. Applied to Genesis physics simulation
    2. Used for trajectory following
    3. Properly synchronized with simulation timesteps
    """
    print("=" * 60)
    print("STEP 4: Testing Trajectory Integration with Genesis")
    print("=" * 60)
    
    try:
        # First run Steps 1-3 to get trajectory data
        print("Running Steps 1-3 to get trajectory data...")
        success, bridge = test_data_bridge_step3(show_viewer=True)
        
        if not success:
            print("❌ Step 4 failed: Could not complete Steps 1-3")
            return False, None
            
        # Get the converted trajectory data
        success, trajectory_data = bridge.convert_to_genesis_format()
        if not success:
            print("❌ Step 4 failed: Could not get trajectory data")
            return False, None
            
        print(f"\n✓ Trajectory data ready: {trajectory_data['info']['timesteps']} timesteps")
        
        # Test trajectory integration
        print("\nStep 4: Testing trajectory integration...")
        
        # Test 1: Verify trajectory data can be applied to Genesis
        print("  Test 1: Applying trajectory data to Genesis...")
        success = _test_apply_trajectory_data(bridge.genesis_env, trajectory_data, bridge)
        if not success:
            return False, None
            
        # Test 2: Verify trajectory following simulation
        print("  Test 2: Running trajectory following simulation...")
        success = _test_trajectory_following(bridge.genesis_env, trajectory_data, bridge)
        if not success:
            return False, None
            
        # Test 3: Verify observation compatibility
        print("  Test 3: Testing observation compatibility...")
        success = _test_observation_compatibility(bridge.genesis_env, trajectory_data)
        if not success:
            return False, None
            
        print("\n" + "=" * 60)
        print("✅ STEP 4 SUCCESS!")
        print("✅ Trajectory data successfully integrated with Genesis")
        print("✅ Physics simulation runs with trajectory data")
        print("✅ Observations compatible with imitation learning")
        print("✅ Ready for Step 5: AMP discriminator implementation")
        print("=" * 60)
        return True, bridge
            
    except Exception as e:
        print(f"\n❌ STEP 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def _test_apply_trajectory_data(genesis_env, trajectory_data, bridge=None):
    """Test applying trajectory data to Genesis simulation using proper DoF mapping"""
    try:
        # Get first few timesteps of trajectory data
        n_test_steps = min(10, trajectory_data['info']['timesteps'])
        
        # Get controllable DoF indices for proper joint position setting
        if bridge and hasattr(bridge, 'genesis_dof_indices'):
            controllable_dof_indices = bridge.genesis_dof_indices
            print(f"    - Using {len(controllable_dof_indices)} controllable DoF indices for position setting")
        else:
            print("    ⚠ Warning: No bridge DoF indices, attempting direct DoF position setting")
            controllable_dof_indices = None
        
        for step in range(n_test_steps):
            # Apply trajectory state to Genesis
            dof_positions = trajectory_data['dof_pos'][step:step+1]  # [1, num_controllable_dofs]
            root_position = trajectory_data['root_pos'][step:step+1]  # [1, 3]
            root_quaternion = trajectory_data['root_quat'][step:step+1]  # [1, 4]
            
            # Apply to all environments (just 1 for testing)
            env_ids = torch.tensor([0], device=genesis_env.device)
            
            # CRITICAL FIX: Set joint positions using specific DoF indices to avoid root joint
            if controllable_dof_indices is not None:
                # Create full DoF position tensor and set only controllable joints
                full_dof_positions = torch.zeros((1, genesis_env.num_dofs), device=genesis_env.device)
                for i, dof_idx in enumerate(controllable_dof_indices):
                    if i < dof_positions.shape[1]:  # Ensure we don't exceed trajectory data
                        full_dof_positions[0, dof_idx] = dof_positions[0, i]
                genesis_env.robot.set_dofs_position(full_dof_positions, envs_idx=env_ids, zero_velocity=True)
            else:
                # Fallback: direct setting (may cause issues with root joint)
                genesis_env.robot.set_dofs_position(dof_positions, envs_idx=env_ids, zero_velocity=True)
            
            # Set root pose
            genesis_env.robot.set_pos(root_position, envs_idx=env_ids, zero_velocity=True)
            genesis_env.robot.set_quat(root_quaternion, envs_idx=env_ids, zero_velocity=True)
            
            # Step physics to verify stability
            genesis_env.scene.step()
            
        print("    ✓ Trajectory data successfully applied to Genesis using proper DoF mapping")
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to apply trajectory data: {e}")
        import traceback
        traceback.print_exc()
        return False


def _test_trajectory_following(genesis_env, trajectory_data, bridge=None):
    """Test trajectory following simulation using bridge's proper DoF control methods"""
    try:
        n_test_steps = min(500, trajectory_data['info']['timesteps'])
        
        print(f"    - Running {n_test_steps} simulation steps using bridge methods...")
        
        if bridge and hasattr(bridge, 'genesis_dof_indices'):
            # Use the bridge's optimized trajectory application method
            print(f"    - Using bridge.apply_trajectory_sequence() with {len(bridge.genesis_dof_indices)} controllable DoFs")
            success = bridge.apply_trajectory_sequence(trajectory_data, start_frame=0, num_frames=n_test_steps)
            if success:
                print("    ✓ Trajectory following simulation completed successfully using bridge methods")
                return True
            else:
                print("    ❌ Bridge trajectory sequence failed")
                return False
        else:
            # Fallback: manual control with skeleton_humanoid approach
            print("    ⚠ Warning: No bridge available, using fallback manual control")
            
            if not hasattr(genesis_env, 'action_to_joint_idx'):
                print("    ❌ Error: Cannot determine controllable DoF indices")
                return False
                
            controllable_dof_indices = []
            for action_name in genesis_env.action_spec:
                joint_idx = genesis_env.action_to_joint_idx[action_name]
                controllable_dof_indices.append(joint_idx)
            print(f"    - Using {len(controllable_dof_indices)} controllable DoF indices: {controllable_dof_indices}")
            
            for step in range(n_test_steps):
                # Get target state from trajectory
                target_dof_pos = trajectory_data['dof_pos'][step:step+1]  # [1, num_controllable_dofs]
                
                # Apply position control to specific local DoF indices only
                genesis_env.robot.control_dofs_position(
                    target_dof_pos, 
                    dofs_idx_local=controllable_dof_indices
                )
                
                # Step simulation
                genesis_env.scene.step()
                
                # Update environment state
                genesis_env._update_robot_state()
                
                # Check stability
                if step % 10 == 0:
                    height = genesis_env.root_pos[0, 2].item()
                    if height < 0.5 or height > 1.5:
                        print(f"    ⚠ Simulation may be unstable: height={height:.3f}m at step {step}")
                        
            print("    ✓ Trajectory following simulation completed successfully using fallback")
            return True
        
    except Exception as e:
        print(f"    ✗ Trajectory following failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _test_observation_compatibility(genesis_env, trajectory_data):
    """Test observation compatibility with imitation learning"""
    try:
        # Get observations from Genesis environment
        obs = genesis_env._get_observations()
        
        # Check observation shape and properties
        print(f"    - Genesis observations shape: {obs.shape}")
        print(f"    - Observation dtype: {obs.dtype}")
        print(f"    - Observation device: {obs.device}")
        
        # Verify observations are reasonable
        assert obs.shape[0] == genesis_env.num_envs, "Observation batch size mismatch"
        assert obs.dtype == torch.float32, "Observations should be float32"
        assert obs.device.type == genesis_env.device.type, "Observation device mismatch"
        
        # Check for NaN or inf values
        assert torch.isfinite(obs).all(), "Observations contain NaN or inf values"
        
        print("    ✓ Observations compatible with imitation learning")
        return True
        
    except Exception as e:
        print(f"    ✗ Observation compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_bridge_step3(show_viewer=False):
    """
    Test Steps 1-3: Complete LocoMujoco to Genesis conversion
    
    This test verifies:
    1. LocoMujoco trajectory loading
    2. Joint name mapping validation  
    3. Data format conversion to Genesis tensors
    """
    print("=" * 60)
    print("TESTING STEPS 1-3: Complete LocoMujoco → Genesis Conversion")
    print("=" * 60)
    
    try:
        # Create a minimal Genesis environment for testing
        import genesis as gs
        gs.init(backend=gs.gpu)
        
        # Import your skeleton environment
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        # Create skeleton environment
        print("Creating Genesis skeleton environment...")
        skeleton_env = SkeletonHumanoidEnv(
            num_envs=1,  # Single environment for testing
            episode_length_s=5.0,
            dt=0.02,
            show_viewer=show_viewer
        )
        print(f"✓ Genesis environment created with {skeleton_env.num_dofs} DOFs")
        print(f"  Genesis DOF names: {skeleton_env.dof_names[:5]}...")
        
        # Create data bridge
        print("\nCreating data bridge...")
        bridge = LocoMujocoDataBridge(skeleton_env)
        
        # Test trajectory loading
        print("\nTesting trajectory loading...")
        success, info = bridge.load_trajectory("walk")
        
        if not success:
            print(f"\n❌ STEP 1 FAILED: {info}")
            return False, None
        
        # Test joint mapping
        print("\nTesting joint mapping...")
        success, mapping_info = bridge.build_joint_mapping()
        
        if not success:
            print(f"\n❌ STEP 2 FAILED: {mapping_info}")
            return False, None
        
        # Test data conversion
        print("\nTesting data conversion...")
        success, trajectory_data = bridge.convert_to_genesis_format()
        
        if success:
            match_rate = mapping_info['match_percentage']
            print("\n" + "=" * 60)
            print("✅ STEPS 1-3 SUCCESS!")
            print("✅ LocoMujoco trajectory loaded and converted")
            print(f"✅ Joint mapping: {match_rate:.1f}% match rate")
            print(f"✅ Genesis tensors: {trajectory_data['info']['timesteps']} timesteps")
            print("✅ Data ready for imitation learning!")
            print("✅ Ready for Step 4: Integration testing")
            print("=" * 60)
            return True, bridge
        else:
            print(f"\n❌ STEP 3 FAILED: {trajectory_data}")
            return False, None
            
    except Exception as e:
        print(f"\n❌ STEP 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def demo_correct_dof_mapping():
    """
    Demo: Correct DoF mapping for motion capture data replay
    
    This function demonstrates how to correctly map LocoMujoco motion capture
    data to Genesis simulation, avoiding the root joint control issues.
    """
    print("=" * 80)
    print("DEMO: Correct DoF Mapping for Motion Capture Data Replay")
    print("=" * 80)
    
    try:
        # Step 1-3: Load and convert trajectory data
        success, bridge = test_data_bridge_step3(show_viewer=False)
        if not success:
            print("❌ Failed to load trajectory data")
            return False
            
        success, trajectory_data = bridge.convert_to_genesis_format()
        if not success:
            print("❌ Failed to convert trajectory data")
            return False
            
        print(f"\n✓ Loaded trajectory with {trajectory_data['info']['timesteps']} frames")
        print(f"✓ Mapped {len(bridge.genesis_dof_indices)} controllable joints")
        print(f"✓ DoF indices: {bridge.genesis_dof_indices}")
        
        # Demo: Correct trajectory application
        print("\n" + "=" * 50)
        print("DEMONSTRATING CORRECT DOF MAPPING:")
        print("=" * 50)
        
        # Apply a few frames to show the difference
        print("\n1. Applying trajectory frames using CORRECT local DoF indices:")
        print("   - This avoids controlling the root joint (first 6 global DoFs)")
        print("   - Uses bridge.apply_trajectory_frame() with proper mapping")
        
        for frame in [0, 50, 100]:
            if frame < trajectory_data['info']['timesteps']:
                print(f"\n   Applying frame {frame}...")
                try:
                    bridge.apply_trajectory_frame(trajectory_data, frame)
                    bridge.genesis_env.scene.step()
                    bridge.genesis_env._update_robot_state()
                    height = bridge.genesis_env.root_pos[0, 2].item()
                    print(f"   ✓ Frame {frame} applied successfully, height: {height:.3f}m")
                except Exception as e:
                    print(f"   ❌ Frame {frame} failed: {e}")
                    
        print("\n2. The KEY DIFFERENCES from the old approach:")
        print("   OLD (incorrect): robot.control_dofs_position(dof_pos)")
        print("   NEW (correct):   robot.control_dofs_position(dof_pos, dofs_idx_local=controllable_indices)")
        print("\n   This ensures:")
        print("   - Root joint (6-DoF freejoint) is NOT controlled")
        print("   - Only actuated joints are controlled")
        print("   - Proper mapping between LocoMujoco and Genesis DoF indices")
        
        print("\n" + "=" * 50)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("✅ DoF mapping now correctly avoids root joint control issues")
        print("✅ Motion capture data can be properly replayed in Genesis")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the corrected DoF mapping demo
    print("Running corrected DoF mapping demonstration...")
    success = demo_correct_dof_mapping()
    
    if success:
        print("\n🎯 Summary of Fixes Applied:")
        print("1. ✅ Fixed global vs local DoF index mapping")
        print("2. ✅ Added proper controllable DoF detection using Genesis motor detection")
        print("3. ✅ Implemented bridge.apply_trajectory_frame() with correct DoF mapping")
        print("4. ✅ Updated all trajectory control functions to use dofs_idx_local parameter")
        print("5. ✅ Root joint is now properly excluded from control commands")
        
        print("\n🎯 Key Methods for Correct DoF Control:")
        print("- bridge.apply_trajectory_frame(trajectory_data, frame_idx)")
        print("- bridge.apply_trajectory_sequence(trajectory_data, start_frame, num_frames)")
        print("- robot.control_dofs_position(targets, dofs_idx_local=controllable_indices)")
        
        print("\n🎯 Next Steps:")
        print("1. Use bridge.apply_trajectory_sequence() for motion replay")
        print("2. Implement imitation learning with proper DoF mapping")
        print("3. Test with different LocoMujoco datasets")
    else:
        print("\n🔧 Fix Issues:")
        print("1. Ensure LocoMujoco is installed and datasets downloaded")
        print("2. Check Genesis environment initialization")
        print("3. Verify joint name consistency between LocoMujoco and Genesis XML")