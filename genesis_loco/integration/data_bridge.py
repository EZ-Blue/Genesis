"""
Refactored LocoMujoco Data Bridge - Simple and Efficient

Clean, minimal implementation that works with the refactored skeleton environment.
Uses direct joint mapping without complex motor detection.

ENHANCED: BVH Integration Support
- Direct NPZ trajectory file loading
- Preprocessed BVH data compatibility  
- Automatic file vs behavior detection

BVH Preprocessing Dependency:
- Location: /home/choonspin/intuitive_autonomy/loco-mujoco/preprocess_scripts/bvh_general_pipeline.py
- Usage: python bvh_general_pipeline.py --input motion.bvh --output motion.npz
- Result: NPZ files can be loaded directly by this bridge
"""

import torch
import numpy as np
import sys
import os
import pickle
from pathlib import Path

# Add LocoMujoco path
sys.path.append('/home/ez/Documents/loco-mujoco')


class LocoMujocoDataBridge:
    """
    Simple, efficient bridge for LocoMujoco trajectory data with Genesis
    
    Compatible with skeleton_humanoid_refactored.py environment.
    Implements trajectory segmentation for AMP training.
    """
    
    def __init__(self, genesis_skeleton_env):
        """
        Initialize data bridge
        
        Args:
            genesis_skeleton_env: SkeletonHumanoidEnv instance (refactored)
        """
        self.genesis_env = genesis_skeleton_env
        self.device = genesis_skeleton_env.device
        
        # Use environment's existing joint mapping
        self.motors_dof_idx = genesis_skeleton_env.motors_dof_idx
        self.joint_names = genesis_skeleton_env.joint_names
        
        # Trajectory data
        self.loco_trajectory = None
        
        # Trajectory segmentation parameters
        self.segment_length = 300  # 3 seconds at 100Hz (covers full gait cycle)
        self.segment_overlap = 50  # 0.5 second overlap between segments
        self.segments = []  # Cached segmented trajectories
        
        # Expert observation caching
        self.cache_dir = Path("./expert_obs_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cached_expert_observations = None
        
    def load_trajectory(self, dataset_name: str = "walk"):
        """
        Load LocoMujoco trajectory using proven pipeline or NPZ file
        
        Args:
            dataset_name: Dataset to load (e.g., "walk", "run") or path to NPZ file
            
        Returns:
            bool: Success status
        """
        print(f"Loading trajectory '{dataset_name}'...")
        
        # Check if it's a file path to NPZ trajectory
        if dataset_name.endswith('.npz') or os.path.exists(dataset_name):
            return self._load_npz_trajectory(dataset_name)
        
        try:
            # Import LocoMujoco components
            from loco_mujoco.task_factories.imitation_factory import ImitationFactory
            from loco_mujoco.task_factories.dataset_confs import DefaultDatasetConf
            
            # Load trajectory using LocoMujoco's pipeline
            loco_env = ImitationFactory.make(
                "SkeletonTorque",
                default_dataset_conf=DefaultDatasetConf([dataset_name])
            )
            
            self.loco_trajectory = loco_env.th.traj
            
            # Validate compatibility
            self._validate_trajectory()
            
            # Create trajectory segments for AMP training
            # self._create_segments()  # DISABLED: Using raw trajectory data instead
            
            print(f"✅ Trajectory loaded: {self.loco_trajectory.data.qpos.shape[0]} timesteps")
            # print(f"✅ Created {len(self.segments)} trajectory segments for training")  # DISABLED
            return True
            
        except Exception as e:
            print(f"❌ Failed to load trajectory: {e}")
            return False
    
    def _load_npz_trajectory(self, npz_path: str):
        """
        Load trajectory from NPZ file (preprocessed BVH data)
        
        Args:
            npz_path: Path to NPZ trajectory file
            
        Returns:
            bool: Success status
        """
        print(f"Loading NPZ trajectory file: {npz_path}")
        
        try:
            from loco_mujoco.trajectory import Trajectory
            
            # Load trajectory from NPZ file
            self.loco_trajectory = Trajectory.load(npz_path)
            
            # Validate compatibility
            self._validate_trajectory()
            
            # Create trajectory segments for AMP training
            self._create_segments()
            
            print(f"✅ NPZ trajectory loaded: {self.loco_trajectory.data.qpos.shape[0]} timesteps")
            print(f"✅ Created {len(self.segments)} trajectory segments for training")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load NPZ trajectory: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_trajectory(self):
        """Validate trajectory compatibility with Genesis environment"""
        if self.loco_trajectory is None:
            return
            
        traj = self.loco_trajectory
        loco_joints = set(traj.info.joint_names)
        genesis_joints = set(self.joint_names)
        
        # Check joint compatibility
        matched = loco_joints.intersection(genesis_joints)
        missing = loco_joints - genesis_joints
        
        print(f"Joint compatibility: {len(matched)}/{len(loco_joints)} matched")
        
        if missing:
            print(f"Missing Genesis joints: {list(missing)[:5]}...")
            
        if len(matched) < len(loco_joints) * 0.8:
            print("⚠️  Warning: Low joint match rate - some trajectory data may be ignored")
    
    # def _create_segments(self):
    #     """Create trajectory segments with root position normalization"""
    #     if self.loco_trajectory is None:
    #         return
    #     
    #     traj = self.loco_trajectory
    #     total_length = traj.data.qpos.shape[0]
    #     
    #     # Clear existing segments
    #     self.segments = []
    #     
    #     # Create overlapping segments
    #     segment_start = 0
    #     while segment_start + self.segment_length <= total_length:
    #         segment_end = segment_start + self.segment_length
    #         
    #         # Extract segment data and make writable copies
    #         segment_qpos = np.array(traj.data.qpos[segment_start:segment_end])
    #         segment_qvel = np.array(traj.data.qvel[segment_start:segment_end])
    #         
    #         # Normalize root position: subtract initial position, keep at origin
    #         initial_root_pos = segment_qpos[0, :3].copy()
    #         segment_qpos[:, :3] -= initial_root_pos  # All root positions relative to start
    #         segment_qpos[0, :3] = [0.0, 0.0, 0.975]  # Start at Genesis default height
    #         
    #         # Store segment
    #         segment_info = {
    #             'qpos': segment_qpos,
    #             'qvel': segment_qvel,
    #             'start_idx': segment_start,
    #             'end_idx': segment_end,
    #             'length': self.segment_length,
    #             'joint_names': traj.info.joint_names
    #         }
    #         
    #         self.segments.append(segment_info)
    #         
    #         # Move to next segment with overlap
    #         segment_start += (self.segment_length - self.segment_overlap)
    #     
    #     print(f"   Created segments: {len(self.segments)} segments of {self.segment_length} timesteps each")  # DISABLED: Using raw trajectory data
    
    def get_trajectory_state(self, timestep: int):
        """
        Get trajectory state at specific timestep formatted for Genesis
        Uses raw trajectory data without segmentation.
        
        Args:
            timestep: Trajectory timestep index
            
        Returns:
            dict: State data formatted for Genesis environment
        """
        if self.loco_trajectory is None:
            return None
        
        traj = self.loco_trajectory
        max_timestep = traj.data.qpos.shape[0]
        
        # Clamp timestep to valid range
        timestep = max(0, min(timestep, max_timestep - 1))
        
        # Extract raw trajectory state
        qpos = traj.data.qpos[timestep]
        qvel = traj.data.qvel[timestep]
        
        # Convert to Genesis format
        genesis_state = self._convert_state_to_genesis(qpos, qvel, traj.info.joint_names)
        
        return genesis_state
    
    def get_trajectory_batch(self, start_timestep: int, batch_size: int):
        """
        Get batch of trajectory states for training
        
        Args:
            start_timestep: Starting timestep
            batch_size: Number of timesteps to extract
            
        Returns:
            dict: Batch of trajectory states formatted for Genesis
        """
        if self.loco_trajectory is None:
            return None
            
        traj = self.loco_trajectory
        max_timestep = traj.data.qpos.shape[0]
        
        # Ensure we don't exceed trajectory bounds
        end_timestep = min(start_timestep + batch_size, max_timestep)
        actual_batch_size = end_timestep - start_timestep
        
        if actual_batch_size <= 0:
            return None
        
        # Extract batch data
        qpos_batch = traj.data.qpos[start_timestep:end_timestep]
        qvel_batch = traj.data.qvel[start_timestep:end_timestep]
        
        # Convert batch to Genesis format
        genesis_batch = self._convert_batch_to_genesis(
            qpos_batch, qvel_batch, traj.info.joint_names
        )
        
        return genesis_batch
    
    def _convert_state_to_genesis(self, loco_qpos, loco_qvel, loco_joint_names):
        """Convert single LocoMujoco state to Genesis format using skeleton_humanoid.py configuration"""
        
        # Use skeleton_humanoid.py's motors_dof_idx order directly
        n_controllable = len(self.motors_dof_idx)
        genesis_dof_pos = torch.zeros(n_controllable, device=self.device)
        genesis_dof_vel = torch.zeros(n_controllable, device=self.device)
        
        # Map using skeleton_humanoid.py's joint order and motor indices
        for motor_idx, global_dof_idx in enumerate(self.motors_dof_idx):
            # Find which joint corresponds to this motor DOF index
            joint_name = None
            for name, dof_idx in self.genesis_env.joint_to_motor_idx.items():
                if dof_idx == global_dof_idx:
                    joint_name = name
                    break
            
            if joint_name and joint_name in loco_joint_names:
                # CRITICAL FIX: Use LocoMujoco's proper joint indexing system 
                # 
                # Problem: The refactored version was using loco_joint_names.index(joint_name) 
                # which assumes qpos data is ordered the same as joint_names. This is WRONG.
                # 
                # Solution: Use LocoMujoco's joint_name2ind_qpos mapping which gives the 
                # actual index where each joint's data is stored in the qpos array.
                # 
                # Why this matters: LocoMujoco stores joint data in an internal order that 
                # doesn't match joint name lists. Using the wrong index means we were 
                # assigning the wrong joint positions to Genesis joints, causing completely 
                # incorrect trajectory following.
                #
                # This fix restores the same logic as the working deprecated version.
                if hasattr(self.loco_trajectory, 'info') and hasattr(self.loco_trajectory.info, 'joint_name2ind_qpos'):

                    if joint_name in self.loco_trajectory.info.joint_name2ind_qpos:
                        loco_qpos_idx_array = self.loco_trajectory.info.joint_name2ind_qpos[joint_name]
                        loco_qvel_idx_array = self.loco_trajectory.info.joint_name2ind_qvel[joint_name]
                        
                        # Extract scalar index (most joints are 1-DOF)
                        loco_qpos_idx = loco_qpos_idx_array[0] if hasattr(loco_qpos_idx_array, '__getitem__') else loco_qpos_idx_array
                        loco_qvel_idx = loco_qvel_idx_array[0] if hasattr(loco_qvel_idx_array, '__getitem__') else loco_qvel_idx_array
                        
                        genesis_dof_pos[motor_idx] = float(loco_qpos[loco_qpos_idx])
                        genesis_dof_vel[motor_idx] = float(loco_qvel[loco_qvel_idx])
                else:
                    # Fallback to simple index lookup (may be incorrect for some datasets)
                    loco_idx = loco_joint_names.index(joint_name)
                    genesis_dof_pos[motor_idx] = float(loco_qpos[loco_idx])
                    genesis_dof_vel[motor_idx] = float(loco_qvel[loco_idx])
        
        # Extract root state (first 7 elements: pos + quat)
        root_pos = torch.tensor(loco_qpos[:3], device=self.device, dtype=torch.float32)
        root_quat = torch.tensor(loco_qpos[3:7], device=self.device, dtype=torch.float32)
        root_lin_vel = torch.tensor(loco_qvel[:3], device=self.device, dtype=torch.float32)
        root_ang_vel = torch.tensor(loco_qvel[3:6], device=self.device, dtype=torch.float32)
        
        return {
            'dof_pos': genesis_dof_pos,
            'dof_vel': genesis_dof_vel,
            'root_pos': root_pos,
            'root_quat': root_quat,
            'root_lin_vel': root_lin_vel,
            'root_ang_vel': root_ang_vel
        }
    
    def _convert_batch_to_genesis(self, loco_qpos_batch, loco_qvel_batch, loco_joint_names):
        """Convert batch of LocoMujoco states to Genesis format using skeleton_humanoid.py configuration"""
        
        batch_size = loco_qpos_batch.shape[0]
        
        # Use skeleton_humanoid.py's motors_dof_idx order directly
        n_controllable = len(self.motors_dof_idx)
        genesis_dof_pos = torch.zeros((batch_size, n_controllable), device=self.device)
        genesis_dof_vel = torch.zeros((batch_size, n_controllable), device=self.device)
        
        # Map using skeleton_humanoid.py's joint order and motor indices
        for motor_idx, global_dof_idx in enumerate(self.motors_dof_idx):
            # Find which joint corresponds to this motor DOF index
            joint_name = None
            for name, dof_idx in self.genesis_env.joint_to_motor_idx.items():
                if dof_idx == global_dof_idx:
                    joint_name = name
                    break
            
            if joint_name and joint_name in loco_joint_names:
                # CRITICAL FIX: Use LocoMujoco's proper joint indexing system (same fix as above)
                if hasattr(self.loco_trajectory, 'info') and hasattr(self.loco_trajectory.info, 'joint_name2ind_qpos'):
                    if joint_name in self.loco_trajectory.info.joint_name2ind_qpos:
                        loco_qpos_idx_array = self.loco_trajectory.info.joint_name2ind_qpos[joint_name]
                        loco_qvel_idx_array = self.loco_trajectory.info.joint_name2ind_qvel[joint_name]
                        
                        # Extract scalar index (most joints are 1-DOF)
                        loco_qpos_idx = loco_qpos_idx_array[0] if hasattr(loco_qpos_idx_array, '__getitem__') else loco_qpos_idx_array
                        loco_qvel_idx = loco_qvel_idx_array[0] if hasattr(loco_qvel_idx_array, '__getitem__') else loco_qvel_idx_array
                        
                        genesis_dof_pos[:, motor_idx] = torch.tensor(
                            loco_qpos_batch[:, loco_qpos_idx], device=self.device, dtype=torch.float32
                        )
                        genesis_dof_vel[:, motor_idx] = torch.tensor(
                            loco_qvel_batch[:, loco_qvel_idx], device=self.device, dtype=torch.float32
                        )
                else:
                    # Fallback to simple index lookup
                    loco_idx = loco_joint_names.index(joint_name)
                    genesis_dof_pos[:, motor_idx] = torch.tensor(
                        loco_qpos_batch[:, loco_idx], device=self.device, dtype=torch.float32
                    )
                    genesis_dof_vel[:, motor_idx] = torch.tensor(
                        loco_qvel_batch[:, loco_idx], device=self.device, dtype=torch.float32
                    )
        
        # Extract root states
        root_pos = torch.tensor(loco_qpos_batch[:, :3], device=self.device, dtype=torch.float32)
        root_quat = torch.tensor(loco_qpos_batch[:, 3:7], device=self.device, dtype=torch.float32)
        root_lin_vel = torch.tensor(loco_qvel_batch[:, :3], device=self.device, dtype=torch.float32)
        root_ang_vel = torch.tensor(loco_qvel_batch[:, 3:6], device=self.device, dtype=torch.float32)
        
        return {
            'dof_pos': genesis_dof_pos,
            'dof_vel': genesis_dof_vel,
            'root_pos': root_pos,
            'root_quat': root_quat,
            'root_lin_vel': root_lin_vel,
            'root_ang_vel': root_ang_vel
        }
    
    def apply_trajectory_state(self, state_data, env_ids=None):
        """
        Apply trajectory state to Genesis environment with physics integration
        for smooth, continuous motion (resolves GAIL discriminator issues)
        
        Args:
            state_data: State data from get_trajectory_state()
            env_ids: Environment indices to apply to (None for all)
        """
        if env_ids is None:
            env_ids = torch.arange(self.genesis_env.num_envs, device=self.device)
        
        num_envs = len(env_ids)
        
        # Prepare state tensors for multiple environments
        dof_pos = state_data['dof_pos'].unsqueeze(0).repeat(num_envs, 1)
        root_pos = state_data['root_pos'].unsqueeze(0).repeat(num_envs, 1)
        root_quat = state_data['root_quat'].unsqueeze(0).repeat(num_envs, 1)
        
        # Apply to Genesis robot using motor DOF indices (consistent with skeleton_humanoid.py)
        self.genesis_env.robot.set_dofs_position(
            dof_pos, 
            dofs_idx_local=self.motors_dof_idx, 
            envs_idx=env_ids, 
            zero_velocity=False  # Keep velocity for physics integration
        )
        self.genesis_env.robot.set_pos(root_pos, envs_idx=env_ids)
        self.genesis_env.robot.set_quat(root_quat, envs_idx=env_ids)

        # print(f"DATA BRIDGE BEFORE STEP: {self.genesis_env.robot.get_dofs_position(dofs_idx_local=self.genesis_env.motors_dof_idx)}")
        
        # CRITICAL: Physics integration for smooth motion (like verify_trajectory.py)
        self.genesis_env.scene.step()

        # print(f"DATA BRIDGE AFTER STEP: {self.genesis_env.robot.get_dofs_position(dofs_idx_local=self.genesis_env.motors_dof_idx)}")
        
        # Update environment state buffers
        self.genesis_env._update_robot_state()
    
    def get_expert_observations_cached(self, 
                                     dataset_name: str = "walk",
                                     num_timesteps: int = None,
                                     start_timestep: int = 0,
                                     step_interval: int = 1,
                                     force_reload: bool = False) -> torch.Tensor:
        """
        Get expert observations with caching for fast reloading
        
        Args:
            dataset_name: Dataset name for cache filename
            num_timesteps: Number of timesteps to collect (None = all available)
            start_timestep: Starting timestep
            step_interval: Collect every N timesteps  
            force_reload: Force regeneration even if cache exists
            
        Returns:
            expert_observations: [num_collected, obs_dim] tensor
        """
        # Use all timesteps if not specified
        if num_timesteps is None:
            num_timesteps = self.trajectory_length - start_timestep
        
        # Generate cache key from parameters
        cache_key = f"{dataset_name}_n{num_timesteps}_s{start_timestep}_i{step_interval}"
        cache_file = self.cache_dir / f"expert_obs_{cache_key}.pt"
        
        # Try loading from cache first
        if cache_file.exists() and not force_reload:
            print(f"📦 Loading cached expert observations: {cache_file.name}")
            try:
                cached_data = torch.load(cache_file, map_location=self.device)
                self.cached_expert_observations = cached_data['observations']
                print(f"✅ Loaded {self.cached_expert_observations.shape[0]} cached observations")
                return self.cached_expert_observations
            except Exception as e:
                print(f"⚠️  Cache loading failed ({e}), regenerating...")
        
        # Generate expert observations with physics integration
        print(f"🔄 Generating expert observations (will be cached for future use)")
        expert_observations = self._collect_physics_expert_observations(
            start_timestep=start_timestep,
            num_timesteps=num_timesteps,
            step_interval=step_interval
        )
        
        if expert_observations is not None:
            # Save to cache
            print(f"💾 Caching expert observations: {cache_file.name}")
            cache_data = {
                'observations': expert_observations,
                'dataset_name': dataset_name,
                'num_timesteps': num_timesteps,
                'start_timestep': start_timestep,
                'step_interval': step_interval,
                'obs_shape': expert_observations.shape,
                'device': str(self.device)
            }
            try:
                torch.save(cache_data, cache_file)
                print(f"✅ Expert observations cached successfully")
            except Exception as e:
                print(f"⚠️  Cache saving failed: {e}")
            
            self.cached_expert_observations = expert_observations
        
        return expert_observations
    
    def _collect_physics_expert_observations(self,
                                           start_timestep: int = 0,
                                           num_timesteps: int = None,
                                           step_interval: int = 1) -> torch.Tensor:
        """
        Internal method to collect expert observations with physics integration
        """
        if self.loco_trajectory is None:
            print("❌ No trajectory loaded")
            return None
        
        # Use all timesteps if not specified
        if num_timesteps is None:
            num_timesteps = self.trajectory_length - start_timestep
        
        # Limit to available trajectory length
        actual_num_timesteps = min(num_timesteps, self.trajectory_length - start_timestep)
        
        print(f"📊 Collecting physics-based expert observations:")
        print(f"   Start: timestep {start_timestep}")
        print(f"   Duration: {actual_num_timesteps} timesteps")
        print(f"   Step interval: {step_interval}")
        print(f"   Physics integration: ENABLED")
        
        expert_obs_list = []
        env_ids = torch.tensor([0], device=self.device)
        
        # Initialize to starting pose
        initial_state = self.get_trajectory_state(start_timestep)
        if initial_state is not None:
            self.apply_trajectory_state(initial_state, env_ids)
            # Settle physics
            for _ in range(3):
                self.genesis_env.scene.step()
            self.genesis_env._update_robot_state()
        
        # Collect with progress tracking
        for step in range(actual_num_timesteps):
            trajectory_idx = start_timestep + step
            
            if trajectory_idx >= self.trajectory_length:
                break
            
            # Apply trajectory state with physics
            state_data = self.get_trajectory_state(trajectory_idx)
            if state_data is not None:
                self.apply_trajectory_state(state_data, env_ids)
                
                # Collect at specified interval
                if step % step_interval == 0:
                    obs = self.genesis_env._get_observations()
                    expert_obs_list.append(obs[0])
            
            # Progress updates
            if step % 1000 == 0:
                progress = (step / actual_num_timesteps) * 100
                print(f"   Progress: {progress:.1f}% ({step}/{actual_num_timesteps})")
        
        if not expert_obs_list:
            print("❌ Failed to collect expert observations")
            return None
        
        expert_observations = torch.stack(expert_obs_list, dim=0)
        print(f"✅ Generated {expert_observations.shape[0]} physics-based expert observations")
        
        return expert_observations
    
    def clear_cache(self, dataset_name: str = None):
        """
        Clear cached expert observations
        
        Args:
            dataset_name: Clear specific dataset cache (None = clear all)
        """
        if dataset_name:
            pattern = f"expert_obs_{dataset_name}_*.pt"
        else:
            pattern = "expert_obs_*.pt"
        
        cache_files = list(self.cache_dir.glob(pattern))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                print(f"🗑️  Cleared cache: {cache_file.name}")
            except Exception as e:
                print(f"⚠️  Failed to clear {cache_file.name}: {e}")
        
        if cache_files:
            print(f"✅ Cleared {len(cache_files)} cache files")
        else:
            print(f"ℹ️  No cache files found matching pattern: {pattern}")
    
    def compute_expert_torques(self, current_timestep: int) -> torch.Tensor:
        """
        Compute torques using PD control to achieve trajectory positions
        
        Uses the same PD gains as defined in skeleton_torque.xml to compute
        what torques would be needed to follow the trajectory.
        """
        current_state = self.get_trajectory_state(current_timestep)
        next_state = self.get_trajectory_state(current_timestep + 1)
        
        if current_state is None or next_state is None:
            return None
            
        current_pos = current_state['dof_pos']
        target_pos = next_state['dof_pos']
        
        # Get only the motor DOF positions (match the action space)
        if len(current_pos) != len(self.motors_dof_idx):
            # Extract motor positions from full DOF vector
            motor_current_pos = current_pos[self.motors_dof_idx]
            motor_target_pos = target_pos[self.motors_dof_idx] 
        else:
            motor_current_pos = current_pos
            motor_target_pos = target_pos
            
        # Approximate velocities using finite differences
        dt = 1.0 / self.trajectory_frequency
        current_vel = torch.zeros_like(motor_current_pos)  # Assume zero for simplicity
        target_vel = (motor_target_pos - motor_current_pos) / dt
        
        # PD gains mapped to actual joint names from environment
        # Use the joint_names from genesis_env to get correct ordering
        joint_pd_map = {
            # From skeleton_torque.xml
            "hip_flexion_r": (10.0, 5.0), "hip_adduction_r": (10.0, 5.0), "hip_rotation_r": (20.0, 5.0),
            "knee_angle_r": (1.0, 1.0), "ankle_angle_r": (10.0, 1.0),
            "hip_flexion_l": (10.0, 5.0), "hip_adduction_l": (10.0, 5.0), "hip_rotation_l": (20.0, 5.0), 
            "knee_angle_l": (1.0, 1.0), "ankle_angle_l": (10.0, 1.0),
            "lumbar_extension": (10.0, 5.0), "lumbar_bending": (10.0, 5.0), "lumbar_rotation": (20.0, 5.0),
            "arm_flex_r": (1.0, 1.0), "arm_add_r": (1.0, 1.0), "arm_rot_r": (1.0, 1.0),
            "elbow_flex_r": (0.0, 1.0), "pro_sup_r": (0.0, 1.0), "wrist_flex_r": (0.0, 1.0), "wrist_dev_r": (0.0, 1.0),
            "arm_flex_l": (1.0, 1.0), "arm_add_l": (1.0, 1.0), "arm_rot_l": (1.0, 1.0),
            "elbow_flex_l": (0.0, 1.0), "pro_sup_l": (0.0, 1.0), "wrist_flex_l": (0.0, 1.0), "wrist_dev_l": (0.0, 1.0),
        }
        
        # Build gains in the correct order using joint_names from environment
        kp_gains = []
        kv_gains = []
        
        for joint_name in self.genesis_env.joint_names:  # Use actual joint ordering
            if joint_name in joint_pd_map:
                kp, kv = joint_pd_map[joint_name]
                kp_gains.append(kp)
                kv_gains.append(kv)
            else:
                # Fallback for unknown joints
                kp_gains.append(1.0)
                kv_gains.append(1.0)
                print(f"⚠️  Unknown joint {joint_name}, using default gains")
        
        kp_gains = torch.tensor(kp_gains, device=self.device)
        kv_gains = torch.tensor(kv_gains, device=self.device)
        
        # Ensure gains match motor count
        if len(kp_gains) != len(self.motors_dof_idx):
            print(f"⚠️  Warning: PD gains length ({len(kp_gains)}) != motor count ({len(self.motors_dof_idx)})")
            # Fallback to default gains
            kp_gains = torch.ones(len(self.motors_dof_idx), device=self.device) * 10.0
            kv_gains = torch.ones(len(self.motors_dof_idx), device=self.device) * 2.0
            
        # PD torque calculation
        pos_error = motor_target_pos - motor_current_pos
        vel_error = target_vel - current_vel
        
        torques = kp_gains * pos_error + kv_gains * vel_error
        
        return torques

    @property
    def trajectory_length(self):
        """Get trajectory length in timesteps"""
        if self.loco_trajectory is None:
            return 0
        return self.loco_trajectory.data.qpos.shape[0]
    
    @property
    def trajectory_frequency(self):
        """Get trajectory frequency in Hz"""
        if self.loco_trajectory is None:
            return 0
        return self.loco_trajectory.info.frequency