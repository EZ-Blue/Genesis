"""
Skeleton Humanoid Environment for Genesis

Implementation of LocoMujoco's SkeletonTorque environment adapted for Genesis physics.
Based on: /home/ez/Documents/loco-mujoco/loco_mujoco/environments/humanoids/skeletons.py
"""

import torch
import numpy as np
from typing import Dict, Tuple, Any, List
import warnings

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.genesis_loco_env import GenesisLocoBaseEnv


class SkeletonHumanoidEnv(GenesisLocoBaseEnv):
    """
    Skeleton Humanoid Environment using torque control
    
    Matches LocoMujoco's SkeletonTorque environment structure:
    - 31 total actions
    - 27 actions (after removing foot joints with box feet)
    - 59 observations (root + joint states)
    - Torque control with direct force application
    """
    
    def __init__(self, 
                 num_envs: int = 1024,
                 episode_length_s: float = 10.0,
                 dt: float = 0.02,
                 use_box_feet: bool = False,
                 disable_arms: bool = False,
                 show_viewer: bool = False,
                 use_trajectory_control: bool = False,
                 **kwargs):
        
        self.use_box_feet = use_box_feet
        self.disable_arms = disable_arms
        self.use_trajectory_control = use_trajectory_control
        
        # Initialize base environment - select XML based on box_feet setting
        if use_box_feet:
            robot_file = "/home/ez/Documents/Genesis/genesis_loco/skeleton/genesis_skeleton_torque_box_feet.xml"
            print(f"Using LocoMujoco-style box feet for stable ground contact")
        else:
            robot_file = "/home/ez/Documents/Genesis/genesis_loco/skeleton/revised_genesis_skeleton.xml"
            print(f"Using standard foot collision meshes")
            
        super().__init__(
            num_envs=num_envs,
            robot_file=robot_file,
            dt=dt,
            episode_length_s=episode_length_s,
            obs_cfg=self._get_obs_config(),
            reward_cfg=self._get_reward_config(),
            show_viewer=show_viewer,
            **kwargs
        )
        
        # Initialize skeleton-specific components
        self._init_skeleton_components()
    
    def _get_obs_config(self) -> Dict[str, Any]:
        """Observation configuration matching LocoMujoco skeleton"""
        return {
            "include_root_pos_z": True,
            "include_root_quat": True,
            "include_root_vel": True,
            "include_joint_pos": True,
            "include_joint_vel": True,
        }
    
    def _get_reward_config(self) -> Dict[str, Any]:
        """Reward configuration for skeleton locomotion"""
        return {
            "trajectory_tracking": 1.0,
            "upright_orientation": 0.2,
            "energy_efficiency": -0.01,
            "root_height": 0.1,
        }
    
    def _init_skeleton_components(self):
        """Initialize skeleton-specific parameters and buffers"""
        # Get action specification matching LocoMujoco
        self._setup_action_spec()
        
        # Setup control mode based on use case
        if self.use_trajectory_control:
            self.setup_balanced_pd_control()  # Use balanced gains instead
        else:
            # self._setup_torque_control()
            self.setup_pd_control()
        
        # Box feet are handled at XML level - no runtime addition needed
        if self.use_box_feet:
            print(f"    ✅ Box feet enabled via XML configuration")
        
        # Initialize buffers
        self._init_skeleton_buffers()
        
        # Register reward functions for base class
        self._register_reward_functions()
    
    def _get_motors_info(self):
        """Get controllable motor DOF indices and names using Genesis' approach"""
        import genesis as gs
        
        motors_dof_idx = []
        motors_dof_name = []
        
        for joint in self.robot.joints:
            # Skip non-controllable joints (same as Genesis view function)
            if joint.type == gs.JOINT_TYPE.FREE:
                continue
            elif joint.type == gs.JOINT_TYPE.FIXED:
                continue
            
            # Get DOF indices using Genesis' correct method
            dofs_idx_local = self.robot.get_joint(joint.name).dofs_idx_local
            if dofs_idx_local:
                if len(dofs_idx_local) == 1:
                    dofs_name = [joint.name]
                else:
                    # Multi-DOF joints get suffixed names
                    dofs_name = [f"{joint.name}_{i_d}" for i_d in dofs_idx_local]
                
                motors_dof_idx += dofs_idx_local
                motors_dof_name += dofs_name
        
        return motors_dof_idx, motors_dof_name

    def _setup_action_spec(self):
        """Setup action specification using Genesis' proven motor detection approach"""
        
        # Get all controllable motors using Genesis' method
        motors_dof_idx, motors_dof_name = self._get_motors_info()
        
        print(f"Genesis detected {len(motors_dof_idx)} controllable DOFs:")
        for idx, (dof_idx, dof_name) in enumerate(zip(motors_dof_idx, motors_dof_name)):
            print(f"  {idx:2d}: DOF {dof_idx:2d} = {dof_name}")
        
        # LocoMujoco action name to joint name mapping
        action_to_joint_mapping = {
            # Lumbar spine
            "mot_lumbar_ext": "lumbar_extension",
            "mot_lumbar_bend": "lumbar_bending", 
            "mot_lumbar_rot": "lumbar_rotation",
            
            # Right leg
            "mot_hip_flexion_r": "hip_flexion_r",
            "mot_hip_adduction_r": "hip_adduction_r",
            "mot_hip_rotation_r": "hip_rotation_r",
            "mot_knee_angle_r": "knee_angle_r", 
            "mot_ankle_angle_r": "ankle_angle_r",
            "mot_subtalar_angle_r": "subtalar_angle_r",
            "mot_mtp_angle_r": "mtp_angle_r",
            
            # Left leg  
            "mot_hip_flexion_l": "hip_flexion_l",
            "mot_hip_adduction_l": "hip_adduction_l",
            "mot_hip_rotation_l": "hip_rotation_l",
            "mot_knee_angle_l": "knee_angle_l",
            "mot_ankle_angle_l": "ankle_angle_l", 
            "mot_subtalar_angle_l": "subtalar_angle_l",
            "mot_mtp_angle_l": "mtp_angle_l",
            
            # Right arm
            "mot_shoulder_flex_r": "arm_flex_r",
            "mot_shoulder_add_r": "arm_add_r",
            "mot_shoulder_rot_r": "arm_rot_r",
            "mot_elbow_flex_r": "elbow_flex_r",
            "mot_pro_sup_r": "pro_sup_r",
            "mot_wrist_flex_r": "wrist_flex_r",
            "mot_wrist_dev_r": "wrist_dev_r",
            
            # Left arm
            "mot_shoulder_flex_l": "arm_flex_l",
            "mot_shoulder_add_l": "arm_add_l", 
            "mot_shoulder_rot_l": "arm_rot_l",
            "mot_elbow_flex_l": "elbow_flex_l",
            "mot_pro_sup_l": "pro_sup_l",
            "mot_wrist_flex_l": "wrist_flex_l",
            "mot_wrist_dev_l": "wrist_dev_l",
        }
        
        # Build action spec using Genesis-detected motors
        self.action_spec = []
        self.action_to_joint_idx = {}
        
        print(f"\nMapping LocoMujoco actions to Genesis motors:")
        for action_name, joint_name in action_to_joint_mapping.items():
            # Find this joint in the motors detected by Genesis
            if joint_name in motors_dof_name:
                motor_idx = motors_dof_name.index(joint_name)
                dof_idx = motors_dof_idx[motor_idx]
                
                # Apply filtering rules
                if self.use_box_feet and action_name in ["mot_subtalar_angle_l", "mot_mtp_angle_l", 
                                                        "mot_subtalar_angle_r", "mot_mtp_angle_r"]:
                    print(f"  {action_name}: FILTERED (box feet enabled)")
                    continue
                
                self.action_to_joint_idx[action_name] = dof_idx
                self.action_spec.append(action_name)
                print(f"  {action_name}: {joint_name} -> DOF {dof_idx} ✅")
            else:
                print(f"  {action_name}: {joint_name} -> NOT FOUND ❌")
        
        self.num_skeleton_actions = len(self.action_spec)
        
        print(f"\nFinal action specification:")
        print(f"  Total actions: {self.num_skeleton_actions}")
        print(f"  Action names: {self.action_spec}")
        print(f"  DOF mapping: {self.action_to_joint_idx}")
    
    def _setup_torque_control(self):
        """Setup pure torque control using control_dofs_force"""
        print("Setting up pure torque control with control_dofs_force...")
        
        # Explicitly set PD gains to zero to ensure no interference
        # kp_values = torch.zeros(self.num_dofs, device=self.device)
        # kv_values = torch.zeros(self.num_dofs, device=self.device)
        # self.robot.set_dofs_kp(kp_values)
        # self.robot.set_dofs_kv(kv_values)
        # print(f"Set all PD gains to zero: kp={kp_values[0]}, kv={kv_values[0]}")
        
        # Set torque limits
        torque_limit = 1000.0   # revise torque limit to be more reasonable
        limits = torch.ones(self.num_dofs, device=self.device) * torque_limit
        self.robot.set_dofs_force_range(-limits, limits)
        
        print(f"Pure torque control enabled with limits: ±{torque_limit} N⋅m")
    
    def setup_pd_control(self):
        """Setup PD gains matching LocoMujoco skeleton_torque configuration"""
        print("Setting up LocoMujoco-matching PD control gains...")
        
        # Initialize with default values (will be overridden for specific joints)
        kp_values = torch.ones(self.num_dofs, device=self.device) * 100.0  # Default kp
        kv_values = torch.ones(self.num_dofs, device=self.device) * 2.0    # Default kv
        
        # LocoMujoco PD gains from training configuration
        loco_pd_gains = {
            # Lumbar joints
            "lumbar_extension": (300, 6),
            "lumbar_bending": (160, 5), 
            "lumbar_rotation": (100, 5),
            
            # Leg joints (right)
            "hip_flexion_r": (200, 5),
            "hip_adduction_r": (200, 5),
            "hip_rotation_r": (200, 5),
            "knee_angle_r": (300, 6),
            "ankle_angle_r": (40, 2),
            "subtalar_angle_r": (40, 2),
            "mtp_angle_r": (40, 2),
            
            # Leg joints (left)
            "hip_flexion_l": (200, 5),
            "hip_adduction_l": (200, 5),
            "hip_rotation_l": (200, 5),
            "knee_angle_l": (300, 6),
            "ankle_angle_l": (40, 2),
            "subtalar_angle_l": (40, 2),
            "mtp_angle_l": (40, 2),
            
            # Arm joints (right)
            "arm_flex_r": (100, 2),
            "arm_add_r": (100, 2),
            "arm_rot_r": (100, 2),
            "elbow_flex_r": (100, 2),
            "pro_sup_r": (50, 2),
            "wrist_flex_r": (50, 2),
            "wrist_dev_r": (50, 2),
            
            # Arm joints (left)
            "arm_flex_l": (100, 2),
            "arm_add_l": (100, 2),
            "arm_rot_l": (100, 2),
            "elbow_flex_l": (100, 2),
            "pro_sup_l": (50, 2),
            "wrist_flex_l": (50, 2),
            "wrist_dev_l": (50, 2),
        }
        
        # Apply LocoMujoco PD gains to corresponding Genesis joints
        applied_count = 0
        for joint_name, (kp, kv) in loco_pd_gains.items():
            try:
                joint_obj = self.robot.get_joint(joint_name)
                dof_idx = joint_obj.dof_idx_local
                if dof_idx < self.num_dofs:
                    kp_values[dof_idx] = float(kp) 
                    kv_values[dof_idx] = float(kv) 
                    applied_count += 1
                    print(f"    Applied LocoMujoco gains: {joint_name} (DOF {dof_idx}): kp={kp}, kv={kv}")
            except Exception as e:
                print(f"    Warning: Could not set gains for {joint_name}: {e}")
        
        print(f"    Successfully applied LocoMujoco gains to {applied_count}/{len(loco_pd_gains)} joints")
        
        # Apply PD gains to robot
        self.robot.set_dofs_kp(kp_values)
        self.robot.set_dofs_kv(kv_values)
        
        # Verify gains were set correctly
        actual_kp = self.robot.get_dofs_kp()
        actual_kv = self.robot.get_dofs_kv()
        
        print(f"    LocoMujoco PD gains applied:")
        print(f"    - kp: min={actual_kp.min():.1f}, max={actual_kp.max():.1f}, mean={actual_kp.mean():.1f}")
        print(f"    - kv: min={actual_kv.min():.1f}, max={actual_kv.max():.1f}, mean={actual_kv.mean():.1f}")
        
        # Validation
        if actual_kp.mean() > 50 and actual_kv.mean() > 1:
            print("    ✅ LocoMujoco-matching PD gains successfully applied")
        else:
            print("    ⚠️  Warning: PD gains may not have been applied correctly")
    
    def setup_trajectory_pd_control(self):
        """Setup trajectory-optimized PD gains for smooth position tracking"""
        print("Setting up trajectory-optimized PD control gains...")
        
        # CRITICAL: Use much lower, smooth PD gains optimized for trajectory following
        kp_values = torch.zeros(self.num_dofs, device=self.device)
        kv_values = torch.zeros(self.num_dofs, device=self.device)
        
        # Trajectory-optimized PD gains (much lower than LocoMujoco training gains)
        trajectory_pd_gains = {
            # Spine joints - low gains for smooth motion
            "lumbar_extension": (50, 8),
            "lumbar_bending": (50, 8),
            "lumbar_rotation": (30, 6),
            
            # Leg joints - moderate gains for stability
            "hip_flexion_r": (80, 10),
            "hip_adduction_r": (80, 10),
            "hip_rotation_r": (60, 8),
            "knee_angle_r": (100, 12),
            "ankle_angle_r": (60, 8),
            "subtalar_angle_r": (20, 4),
            "mtp_angle_r": (20, 4),
            
            "hip_flexion_l": (80, 10),
            "hip_adduction_l": (80, 10), 
            "hip_rotation_l": (60, 8),
            "knee_angle_l": (100, 12),
            "ankle_angle_l": (60, 8),
            "subtalar_angle_l": (20, 4),
            "mtp_angle_l": (20, 4),
            
            # Arm joints - very low gains for smooth motion
            "arm_flex_r": (30, 6),
            "arm_add_r": (30, 6),
            "arm_rot_r": (25, 5),
            "elbow_flex_r": (40, 6),
            "pro_sup_r": (20, 4),
            "wrist_flex_r": (15, 3),
            "wrist_dev_r": (15, 3),
            
            "arm_flex_l": (30, 6),
            "arm_add_l": (30, 6),
            "arm_rot_l": (25, 5),
            "elbow_flex_l": (40, 6),
            "pro_sup_l": (20, 4),
            "wrist_flex_l": (15, 3),
            "wrist_dev_l": (15, 3),
        }
        
        # Apply trajectory-optimized gains
        applied_count = 0
        for joint_name, (kp, kv) in trajectory_pd_gains.items():
            try:
                joint_obj = self.robot.get_joint(joint_name)
                dof_idx = joint_obj.dofs_idx_local[0] if joint_obj.dofs_idx_local else None
                if dof_idx is not None and dof_idx < self.num_dofs:
                    kp_values[dof_idx] = float(kp)
                    kv_values[dof_idx] = float(kv)
                    applied_count += 1
                    print(f"    Applied trajectory gains: {joint_name} (DOF {dof_idx}): kp={kp}, kv={kv}")
            except Exception as e:
                print(f"    Warning: Could not set trajectory gains for {joint_name}: {e}")
        
        print(f"    Successfully applied trajectory gains to {applied_count}/{len(trajectory_pd_gains)} joints")
        
        # Apply gains to robot
        self.robot.set_dofs_kp(kp_values)
        self.robot.set_dofs_kv(kv_values)
        
        # Verify gains
        actual_kp = self.robot.get_dofs_kp()
        actual_kv = self.robot.get_dofs_kv()
        
        print(f"    Trajectory-optimized PD gains applied:")
        print(f"    - kp: min={actual_kp.min():.1f}, max={actual_kp.max():.1f}, mean={actual_kp.mean():.1f}")
        print(f"    - kv: min={actual_kv.min():.1f}, max={actual_kv.max():.1f}, mean={actual_kv.mean():.1f}")
        
        print("    ✅ Trajectory-optimized control enabled")
    
    def setup_balanced_pd_control(self):
        """Balanced PD gains for stable trajectory following"""
        print("Setting up balanced PD control gains...")
        
        kp_values = torch.ones(self.num_dofs, device=self.device) * 80.0
        kv_values = torch.ones(self.num_dofs, device=self.device) * 8.0
        
        # Balanced gains - compromise between LocoMujoco (high) and trajectory (low) values
        gains = {
            # Spine - moderate for posture
            "lumbar_extension": (120, 10), "lumbar_bending": (100, 9), "lumbar_rotation": (80, 8),
            # Legs - higher for stability 
            "hip_flexion_r": (140, 12), "hip_adduction_r": (120, 10), "hip_rotation_r": (100, 9),
            "hip_flexion_l": (140, 12), "hip_adduction_l": (120, 10), "hip_rotation_l": (100, 9),
            "knee_angle_r": (180, 14), "knee_angle_l": (180, 14),
            "ankle_angle_r": (100, 10), "ankle_angle_l": (100, 10),
            "subtalar_angle_r": (60, 6), "subtalar_angle_l": (60, 6),
            "mtp_angle_r": (40, 5), "mtp_angle_l": (40, 5),
            # Arms - lower for natural motion
            "arm_flex_r": (60, 7), "arm_add_r": (50, 6), "arm_rot_r": (40, 5), "elbow_flex_r": (70, 7),
            "arm_flex_l": (60, 7), "arm_add_l": (50, 6), "arm_rot_l": (40, 5), "elbow_flex_l": (70, 7),
            "pro_sup_r": (30, 4), "wrist_flex_r": (25, 3), "wrist_dev_r": (25, 3),
            "pro_sup_l": (30, 4), "wrist_flex_l": (25, 3), "wrist_dev_l": (25, 3),
        }
        
        # Apply gains
        applied = 0
        for joint_name, (kp, kv) in gains.items():
            try:
                joint_obj = self.robot.get_joint(joint_name)
                dof_idx = joint_obj.dofs_idx_local[0] if hasattr(joint_obj, 'dofs_idx_local') else joint_obj.dof_idx_local
                if dof_idx < self.num_dofs:
                    kp_values[dof_idx] = float(kp)
                    kv_values[dof_idx] = float(kv)
                    applied += 1
            except:
                continue
        
        self.robot.set_dofs_kp(kp_values)
        self.robot.set_dofs_kv(kv_values)
        
        actual_kp = self.robot.get_dofs_kp()
        actual_kv = self.robot.get_dofs_kv()
        print(f"    Applied to {applied}/{len(gains)} joints")
        print(f"    kp: {actual_kp.min():.1f}-{actual_kp.max():.1f} (avg: {actual_kp.mean():.1f})")
        print(f"    kv: {actual_kv.min():.1f}-{actual_kv.max():.1f} (avg: {actual_kv.mean():.1f})")
        print("    ✅ Balanced PD gains applied")
    
    def _register_reward_functions(self):
        """Register reward functions with the base class reward system"""
        # Ensure reward_functions dict exists (from base class _init_reward_functions)
        if not hasattr(self, 'reward_functions'):
            self.reward_functions = {}
        
        # Register skeleton-specific reward functions
        self.reward_functions.update({
            'upright_orientation': self._reward_upright_orientation,
            'energy_efficiency': self._reward_energy_efficiency, 
            'root_height': self._reward_root_height,
        })
        print(f"    ✅ Registered {len(self.reward_functions)} reward functions")
    

    def _init_skeleton_buffers(self):
        """Initialize skeleton-specific state buffers"""
        # Previous actions for observations and smoothness
        self.prev_actions = torch.zeros((self.num_envs, self.num_skeleton_actions), 
                                       device=self.device)
        
        # Energy consumption tracking
        self.energy_consumption = torch.zeros((self.num_envs,), device=self.device)
        
        # Target velocity for locomotion
        self.target_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_velocity[:, 0] = 1.0  # Default forward velocity
    
    def _apply_actions(self, actions: torch.Tensor):
        """Apply torque actions directly to joints using correct Genesis DOF control"""
        
        # CRITICAL FIX: Create torque tensor with correct dimensions for controllable DOFs only
        num_controllable_dofs = len(self.action_to_joint_idx)
        torques = torch.zeros((self.num_envs, num_controllable_dofs), device=self.device)
        
        # CRITICAL FIX: Get DOF indices for explicit control (matching motor detection)
        controllable_dof_indices = []
        
        # Direct mapping using pre-computed indices (in action order)
        for i, action_name in enumerate(self.action_spec):
            if i < actions.shape[-1]:  # Ensure we don't exceed action dimensions
                joint_idx = self.action_to_joint_idx[action_name]
                torques[:, i] = actions[:, i]  # FIXED: Use action index, not joint index
                controllable_dof_indices.append(joint_idx)
        
        # CRITICAL FIX: Apply torques to specific DOFs only using explicit indices
        # This avoids the double indexing problem by using local DOF indices
        self.robot.control_dofs_force(torques, dofs_idx_local=controllable_dof_indices)
        
        # Track energy consumption (using explicit DOF selection)
        if hasattr(self, 'dof_vel'):
            controlled_dof_vel = self.dof_vel[:, controllable_dof_indices] if len(controllable_dof_indices) > 0 else self.dof_vel[:, :num_controllable_dofs]
            power = torch.sum(torch.abs(torques * controlled_dof_vel), dim=1)
            self.energy_consumption += power * self.dt
    
    
    def _get_observations(self) -> torch.Tensor:
        """Get observations matching LocoMujoco skeleton structure"""
        obs_components = []
        
        # Root pose (no x,y position, only z + quaternion) - 5D
        obs_components.append(self.root_pos[:, 2:3])  # z position only
        obs_components.append(self.root_quat)         # quaternion (4D)
        
        # Joint positions - simplified to match available DOFs
        num_joint_obs = min(22, self.num_dofs)  # LocoMujoco has 22 joint positions
        joint_pos = self.dof_pos[:, :num_joint_obs]
        obs_components.append(joint_pos)
        
        # Root velocity (full 6D) 
        obs_components.append(self.root_lin_vel)      # 3D
        obs_components.append(self.root_ang_vel)      # 3D
        
        # Joint velocities
        joint_vel = self.dof_vel[:, :num_joint_obs]
        obs_components.append(joint_vel)
        
        obs = torch.cat(obs_components, dim=-1)
        
        # Update buffers
        if self.obs_buf is None:
            self.obs_buf = torch.zeros_like(obs)
        self.obs_buf[:] = obs
        
        self.extras["observations"]["policy"] = obs
        return obs
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step environment with action history tracking and proper episode management"""
        # Store for next observation
        self._current_actions = actions.clone()
        
        # Episode length is handled by parent class
        
        # Call parent step
        obs, rewards, dones, info = super().step(actions)
        
        # Update action history for skeleton-specific observations
        self.prev_actions[:] = actions[:, :self.num_skeleton_actions]
        
        # Add episode info to extras
        info.update({
            'episode_length': self.episode_length_buf.clone(),
            'episode_reward': rewards.clone()  
        })
        
        return obs, rewards, dones, info
    
    def _check_termination(self) -> torch.Tensor:
        """Check skeleton-specific termination conditions"""
        done = super()._check_termination()
        
        # Height limits (LocoMujoco: 0.8-1.1m)
        height_violation = (self.root_pos[:, 2] < 0.8) | (self.root_pos[:, 2] > 1.1)
        done = done | height_violation
        
        # Orientation limits (45 degree roll/pitch)
        root_euler = quat_to_xyz(self.root_quat)  # Returns roll, pitch, yaw in radians
        extreme_tilt = (torch.abs(root_euler[:, 0]) > torch.pi/4) | (torch.abs(root_euler[:, 1]) > torch.pi/4)
        done = done | extreme_tilt
        
        return done
    
    def _reset_robot_pose(self, env_ids: torch.Tensor):
        """Reset to upright standing pose"""
        num_reset = len(env_ids)
        
        # Zero joint positions
        default_pose = torch.zeros((num_reset, self.num_dofs), device=self.device)
        
        # Standing position
        default_root_pos = torch.tensor([0.0, 0.0, 0.975], device=self.device).repeat(num_reset, 1)
        default_root_quat = torch.tensor([0.7071067811865475, 0.7071067811865475, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        
        # Apply reset
        self.robot.set_dofs_position(default_pose, envs_idx=env_ids, zero_velocity=True)
        self.robot.set_pos(default_root_pos, envs_idx=env_ids, zero_velocity=True)
        self.robot.set_quat(default_root_quat, envs_idx=env_ids, zero_velocity=True)
        
        # Reset buffers
        self.energy_consumption[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.target_velocity[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
    
    # Reward functions
    def _reward_upright_orientation(self) -> torch.Tensor:
        """Reward staying upright"""
        root_euler = quat_to_xyz(self.root_quat)  # Returns roll, pitch, yaw in radians
        orientation_error = torch.abs(root_euler[:, 0]) + torch.abs(root_euler[:, 1])
        return torch.exp(-orientation_error * 2.0)
    
    def _reward_energy_efficiency(self) -> torch.Tensor:
        """Penalize energy consumption"""
        return torch.sum(torch.abs(self.prev_actions), dim=1)
    
    def _reward_root_height(self) -> torch.Tensor:
        """Reward proper standing height"""
        height_error = torch.abs(self.root_pos[:, 2] - 1.0)
        return torch.exp(-height_error * 5.0)
    
    @property
    def num_observations(self) -> int:
        """Calculate observation space size"""
        # Root: 5 (z + quat) + Joints: 22 + Root vel: 6 + Joint vel: 22 = 55
        return 5 + min(22, self.num_dofs) + 6 + min(22, self.num_dofs)
    
    @property 
    def num_actions(self) -> int:
        """Action space size"""
        return self.num_skeleton_actions