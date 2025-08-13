"""
Consolidated Skeleton Humanoid Environment for Genesis

Refactored implementation following Go2Env pattern - single comprehensive environment
for LocoMujoco skeleton imitation learning adapted for Genesis physics.
Maintains exact values and configurations from current skeleton_humanoid.py and genesis_loco_env.py
"""

import torch
import numpy as np
import math
import warnings
from typing import Dict, Tuple, Any, List, Optional

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat

from loco_mujoco.trajectory import TrajectoryHandler, Trajectory, TrajState


class SkeletonHumanoidEnv:
    """
    Consolidated Skeleton Humanoid Environment using Genesis physics
    
    Matches LocoMujoco's SkeletonTorque environment structure:
    - 31 total actions (27 with box feet)
    - 59 observations (root + joint states)  
    - Torque control with direct force application
    """
    
    def __init__(self, 
                 num_envs: int = 1024,
                 episode_length_s: float = 10.0,
                 dt: float = 0.01,
                 use_box_feet: bool = False,
                 disable_arms: bool = False,
                 show_viewer: bool = False,
                 **kwargs):
        
        self.num_envs = num_envs
        self.use_box_feet = use_box_feet
        self.disable_arms = disable_arms
        self.dt = dt
        self.episode_length_s = episode_length_s
        self.max_episode_length = math.ceil(episode_length_s / dt)
        self.device = gs.device
        
        # Configurations (matching current implementation)
        self.reward_cfg = self._get_reward_config()
        
        # Initialize Genesis scene
        self._init_genesis_scene(show_viewer)
        
        # Load robot
        self._load_robot()

        # Default joints for SkeletonTorque Model
        self.joint_names = {
            "lumbar_extension", "lumbar_bending", "lumbar_rotation",
            
            # Right leg
            "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
            
            # Left leg  
            "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l",
            
            # Right arm
            "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r",
            
            # Left arm
            "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l",
        }
        
        # Build scene
        self.scene.build(n_envs=num_envs)
        
        # Initialize skeleton-specific components
        self._init_skeleton_components()
        
        # Initialize buffers
        self._init_buffers()
        
        # Initialize reward functions
        self._init_reward_functions()
        
        # Trajectory handler (optional)
        self.th = None
        
        print(f"✅ SkeletonHumanoidEnv initialized:")
        print(f"   - Environments: {self.num_envs}")
        print(f"   - Actions: {self.num_actions}")
        print(f"   - Observations: {self.num_observations}")
        print(f"   - Episode length: {self.episode_length_s}s")
    
    def _get_reward_config(self) -> Dict[str, Any]:
        """Reward configuration for skeleton locomotion (from current implementation)"""
        return {
            "trajectory_tracking": 1.0,
            "upright_orientation": 0.2,
            "energy_efficiency": -0.01,
            "root_height": 0.1,
        }

    def _init_genesis_scene(self, show_viewer: bool = False):
        """Initialize Genesis scene with appropriate settings (from current base env)"""
        self.scene = gs.Scene(
            show_FPS=False,
            sim_options=gs.options.SimOptions(
                dt=self.dt, 
                substeps=2,
                # gravity=(0.0,0.0,0.0)
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=True,
            ),
            show_viewer=show_viewer,
        )
        
        # Add ground plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    def _load_robot(self):
        """Load skeleton robot into scene (from current implementation)"""
        # Select robot file based on box_feet setting (exact same logic)
        if self.use_box_feet:
            robot_file = "/home/ez/Documents/Genesis/genesis_loco/skeleton/genesis_skeleton_torque_box_feet.xml"
            print(f"Using LocoMujoco-style box feet for stable ground contact")
        else:
            robot_file = "/home/ez/Documents/Genesis/genesis_loco/skeleton/revised_genesis_skeleton.xml"
            print(f"Using standard foot collision meshes")
        
        # Load robot (from base env)
        self.robot_file = robot_file
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=self.robot_file))

    def _init_skeleton_components(self):
        """Initialize skeleton-specific parameters and buffers (from current implementation)"""
        # Get action specification matching LocoMujoco
        self._setup_action_spec()
        
        # Setup control mode
        self.setup_pd_control()
        
        # Box feet are handled at XML level - no runtime addition needed
        if self.use_box_feet:
            print(f"    ✅ Box feet enabled via XML configuration")

    def _setup_action_spec(self):
        """Setup action specification using Genesis motor detection"""
        if self.use_box_feet:
            excluded_joints = {"subtalar_angle_l", "mtp_angle_l", "subtalar_angle_r", "mtp_angle_r"}
            self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.joint_names if name not in excluded_joints]
            self.joint_to_motor_idx = {name: self.robot.get_joint(name).dof_start for name in self.joint_names if name not in excluded_joints}
        else:
            self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.joint_names]
            self.joint_to_motor_idx = {name: self.robot.get_joint(name).dof_start for name in self.joint_names}

        self.num_actions = len(self.motors_dof_idx)
        print(f"Total Actoins: {self.num_actions}\n")
        print(f"List of Motor DoF Indices: {self.motors_dof_idx}\n")
        print(f"Joint to Motor Idx Mapping: {self.joint_to_motor_idx}\n")

    def setup_pd_control(self):
        """Setup PD gains matching LocoMujoco skeleton_torque configuration"""
        print("Setting up LocoMujoco-matching PD control gains...")
        
        # Initialize with default values for ALL robot DOFs
        kp_values = torch.ones(self.robot.n_dofs, device=self.device) * 100.0  # Default kp
        kv_values = torch.ones(self.robot.n_dofs, device=self.device) * 2.0    # Default kv
        
        # LocoMujoco PD gains from training configuration
        loco_pd_gains = {
            # Lumbar joints
            "lumbar_extension": (300, 6), "lumbar_bending": (160, 5), "lumbar_rotation": (100, 5),
            
            # Leg joints (right)
            "hip_flexion_r": (200, 5), "hip_adduction_r": (200, 5), "hip_rotation_r": (200, 5), "knee_angle_r": (300, 6), 
            "ankle_angle_r": (40, 2), "subtalar_angle_r": (40, 2), "mtp_angle_r": (40, 2),
            
            # Leg joints (left)
            "hip_flexion_l": (200, 5), "hip_adduction_l": (200, 5), "hip_rotation_l": (200, 5), "knee_angle_l": (300, 6),
            "ankle_angle_l": (40, 2), "subtalar_angle_l": (40, 2), "mtp_angle_l": (40, 2),
            
            # Arm joints (right)
            "arm_flex_r": (100, 2), "arm_add_r": (100, 2), "arm_rot_r": (100, 2), "elbow_flex_r": (100, 2), "pro_sup_r": (50, 2),
            "wrist_flex_r": (50, 2), "wrist_dev_r": (50, 2),
            
            # Arm joints (left)
            "arm_flex_l": (100, 2), "arm_add_l": (100, 2), "arm_rot_l": (100, 2), "elbow_flex_l": (100, 2), "pro_sup_l": (50, 2),
            "wrist_flex_l": (50, 2), "wrist_dev_l": (50, 2),
        }
        
        # Apply LocoMujoco PD gains to corresponding Genesis joints
        applied_count = 0
        for joint_name, (kp, kv) in loco_pd_gains.items():
            if joint_name in self.joint_to_motor_idx:
                dof_idx = self.joint_to_motor_idx[joint_name]
                # Update the tensors at the specific DOF index
                kp_values[dof_idx] = float(kp)
                kv_values[dof_idx] = float(kv)
                applied_count += 1
                print(f"    Applied LocoMujoco gains: {joint_name} (DOF {dof_idx}): kp={kp}, kv={kv}")
            else:
                print(f"    Warning: Joint {joint_name} not found in action mapping")
        
        print(f"    Successfully applied LocoMujoco gains to {applied_count}/{len(loco_pd_gains)} joints")
        
        # Set all gains at once (this is the correct Genesis API usage)
        self.robot.set_dofs_kp(kp_values)
        self.robot.set_dofs_kv(kv_values)
        
        # Verify gains were set correctly
        actual_kp = self.robot.get_dofs_kp()
        actual_kv = self.robot.get_dofs_kv()
        
        print(f"    LocoMujoco PD gains applied:")
        # print(f"    - kp: min={actual_kp.min():.1f}, max={actual_kp.max():.1f}, mean={actual_kp.mean():.1f}")
        # print(f"    - kv: min={actual_kv.min():.1f}, max={actual_kv.max():.1f}, mean={actual_kv.mean():.1f}")

    def _init_buffers(self):
        """Initialize state buffers for environments (from base env with skeleton additions)"""
        # Episode management (from base env)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Robot state buffers (from base env)
        self.dof_pos = torch.zeros((self.num_envs, self.robot.n_dofs), device=self.device, dtype=torch.float32)
        self.dof_vel = torch.zeros((self.num_envs, self.robot.n_dofs), device=self.device, dtype=torch.float32)
        self.root_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.root_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.root_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.root_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        
        # Observation and reward buffers (from base env)
        self.obs_buf = None  # Will be initialized based on observation space
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        # Trajectory tracking buffers (from base env)
        self.traj_idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.traj_time = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        # Skeleton-specific buffers (from current implementation)
        # Previous actions for observations and smoothness
        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), 
                                       device=self.device)
        
        # Energy consumption tracking
        self.energy_consumption = torch.zeros((self.num_envs,), device=self.device)
        
        # Target velocity for locomotion
        self.target_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_velocity[:, 0] = 1.0  # Default forward velocity
        
        # Extras for logging (from base env)
        self.extras = {"observations": {}}

    def _init_reward_functions(self):
        """Initialize reward functions based on configuration (from base env)"""
        self.reward_functions = {}
        self.episode_sums = {}
        
        for reward_name, reward_scale in self.reward_cfg.items():
            if hasattr(self, f"_reward_{reward_name}"):
                self.reward_functions[reward_name] = getattr(self, f"_reward_{reward_name}")
                self.episode_sums[reward_name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            else:
                warnings.warn(f"Reward function '_reward_{reward_name}' not found, skipping.")

    @property
    def num_observations(self) -> int:
        """Calculate observation space size"""
        # Root: 5 (z + quat) + Controlled joints: num_actions + Root vel: 6 + Controlled joint vel: num_actions
        return 5 + self.num_actions + 6 + self.num_actions

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment forward one timestep (consolidated from base env and skeleton)
        """
        # Store for next observation (from skeleton)
        self._current_actions = actions.clone()
        
        # Apply actions (skeleton implementation)
        self._apply_actions(actions)
        
        # Step Genesis physics (from base env)
        self.scene.step()
        
        # Update robot state from simulation (from base env)
        self._update_robot_state()
        
        # Update episode length (from base env)
        self.episode_length_buf += 1
        
        # Update trajectory time if using trajectories (from base env)
        if self.th is not None:
            self.traj_time += self.dt
        
        # Check for episode termination (skeleton implementation)
        self.reset_buf = self._check_termination()
        
        # Reset environments that are done (from base env)
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        # Compute rewards (from base env)
        self._compute_rewards()
        
        # Get observations (skeleton implementation)
        obs = self._get_observations()
        
        # Update action history for skeleton-specific observations (from skeleton)
        self.prev_actions[:] = actions[:, :self.num_actions]
        
        # Add episode info to extras (from skeleton)
        self.extras.update({
            'episode_length': self.episode_length_buf.clone(),
            'episode_reward': self.rew_buf.clone()  
        })
        
        return obs, self.rew_buf, self.reset_buf, self.extras

    def _apply_actions(self, actions: torch.Tensor):
        """Apply torque actions directly to joints using simplified DOF mapping"""
        
        # Apply actions directly to the motors using pre-computed DOF indices
        self.robot.control_dofs_force(actions, dofs_idx_local=self.motors_dof_idx)
        
        # Track energy consumption
        if hasattr(self, 'dof_vel') and hasattr(self, 'energy_consumption'):
            # Get velocities for controlled DOFs only
            controlled_dof_vel = self.dof_vel[:, self.motors_dof_idx]
            power = torch.sum(torch.abs(actions * controlled_dof_vel), dim=1)
            self.energy_consumption += power * self.dt

    def _update_robot_state(self):
        """Update robot state buffers from Genesis simulation (from base env)"""
        # Update joint states
        self.dof_pos[:] = self.robot.get_dofs_position()
        self.dof_vel[:] = self.robot.get_dofs_velocity()
        
        # Update root states
        self.root_pos[:] = self.robot.get_pos()
        self.root_quat[:] = self.robot.get_quat()
        self.root_lin_vel[:] = self.robot.get_vel()
        self.root_ang_vel[:] = self.robot.get_ang()

    def _check_termination(self) -> torch.Tensor:
        """Check skeleton-specific termination conditions"""
        # Default termination: episode length exceeded
        done = self.episode_length_buf >= self.max_episode_length
        
        # Height limits (LocoMujoco: 0.8-1.1m)
        height_violation = (self.root_pos[:, 2] < 0.8) | (self.root_pos[:, 2] > 1.1)
        done = done | height_violation
        
        # Orientation limits relative to skeleton's default orientation
        # Default quat is [0.7071067811865475, 0.7071067811865475, 0.0, 0.0] (90° roll in skeleton frame)
        # Check deviation from this default rather than absolute orientation
        default_quat = torch.tensor([0.7071067811865475, 0.7071067811865475, 0.0, 0.0], device=self.device)
        
        # For now, disable orientation termination to avoid immediate resets
        # TODO: Implement proper relative orientation checking
        # extreme_tilt = False
        
        return done

    def _compute_rewards(self):
        """Compute rewards for current step (from base env)"""
        self.rew_buf[:] = 0.0
        
        for reward_name, reward_func in self.reward_functions.items():
            if reward_name in self.reward_cfg:
                reward_scale = self.reward_cfg[reward_name]
                reward_value = reward_func()
                
                self.rew_buf += reward_scale * reward_value
                self.episode_sums[reward_name] += reward_scale * reward_value

    def _get_observations(self) -> torch.Tensor:
        """Get observations matching LocoMujoco skeleton structure (exact same as current)"""
        obs_components = [
          # Root state (5D)
          self.root_pos[:, 2:3],  # z position
          self.root_quat,         # quaternion

          # Controlled joint positions (27D) - use your motor indices
          self.dof_pos[:, self.motors_dof_idx],

          # Root velocity (6D)
          self.root_lin_vel,
          self.root_ang_vel,

          # Controlled joint velocities (27D) - use your motor indices  
          self.dof_vel[:, self.motors_dof_idx],
        ]
        
        obs = torch.cat(obs_components, dim=-1)
        
        # Update buffers
        if self.obs_buf is None:
            self.obs_buf = torch.zeros_like(obs)
        self.obs_buf[:] = obs
        
        self.extras["observations"]["policy"] = obs
        return obs

    def reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments (from base env with skeleton pose)"""
        if len(env_ids) == 0:
            return
            
        # Reset episode length (from base env)
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        
        # Reset trajectory tracking (from base env)
        if self.th is not None:
            self.traj_idx[env_ids] = 0
            self.traj_time[env_ids] = 0.0
        
        # Reset robot to default pose (skeleton implementation)
        self._reset_robot_pose(env_ids)
        
        # Log episode rewards (from base env)
        self._log_episode_rewards(env_ids)

    def _reset_robot_pose(self, env_ids: torch.Tensor):
        """Reset to upright standing pose (exact same as current)"""
        num_reset = len(env_ids)
        
        # Zero joint positions
        default_pose = torch.zeros((num_reset, self.robot.n_dofs), device=self.device)
        
        # Standing position (exact same values from XML)
        default_root_pos = torch.tensor([0.0, 0.0, 0.975], device=self.device).repeat(num_reset, 1)
        default_root_quat = torch.tensor([0.7071067811865475, 0.7071067811865475, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        
        # Apply reset
        self.robot.set_dofs_position(default_pose, envs_idx=env_ids, zero_velocity=True)
        self.robot.set_pos(default_root_pos, envs_idx=env_ids, zero_velocity=True)
        self.robot.set_quat(default_root_quat, envs_idx=env_ids, zero_velocity=True)
        
        # CRITICAL FIX: Update state buffers after setting robot pose
        self._update_robot_state()
        
        # Reset buffers (from skeleton)
        self.energy_consumption[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.target_velocity[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=self.device)

    def _log_episode_rewards(self, env_ids: torch.Tensor):
        """Log episode reward summaries (from base env)"""
        if not hasattr(self, 'extras'):
            self.extras = {}
        if 'episode' not in self.extras:
            self.extras['episode'] = {}
            
        for reward_name, reward_sum in self.episode_sums.items():
            mean_reward = torch.mean(reward_sum[env_ids]).item()
            self.extras['episode'][f'rew_{reward_name}'] = mean_reward / self.episode_length_s
            reward_sum[env_ids] = 0.0

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset all environments (from base env)"""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        obs = self._get_observations()
        return obs, self.extras

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        """Get current observations (from base env)"""
        obs = self._get_observations()
        return obs, self.extras

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        """Get privileged observations (default: None) (from base env)"""
        return None

    # Trajectory Integration (from base env)
    def load_trajectory(self, 
                        traj: Trajectory = None,
                        traj_path: str = None,
                        warn: bool = True) -> None:
        """
        Loads trajectories. If there were trajectories loaded already, this function overrides the latter.
        (exact same as base env)
        """
        if self.th is not None and warn:
            warnings.warn("New trajectories loaded, which overrides the old ones.", RuntimeWarning)

        # Create a mock model object with necessary attributes for TrajectoryHandler
        mock_model = type('MockModel', (), {
            'nq': self.robot.n_dofs,
            'nv': self.robot.n_dofs,
            'dt': self.dt,
            'joint_names': self.joint_names
        })()
        
        th_params = {}  # No th_params in this simplified version
        self.th = TrajectoryHandler(model=mock_model, warn=warn, traj_path=traj_path,
                                    traj=traj, control_dt=self.dt, **th_params)

    # Reward Functions (exact same as current skeleton implementation)
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

    def _reward_trajectory_tracking(self) -> torch.Tensor:
        """Reward for tracking trajectory (from base env)"""
        if self.th is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            
        target_state = self._get_trajectory_target_state()
        if target_state is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        # Compute position tracking error
        if hasattr(target_state, 'qpos') and target_state.qpos is not None:
            target_qpos = torch.tensor(target_state.qpos, device=self.device, dtype=torch.float32)
            if len(target_qpos.shape) == 1:
                target_qpos = target_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            
            pos_error = torch.norm(self.dof_pos - target_qpos, dim=-1)
            return torch.exp(-pos_error)
        
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

    def _apply_trajectory_state(self, traj_state: TrajState, env_ids: torch.Tensor):
        """Apply trajectory state to Genesis physics simulation (from base env)"""
        if traj_state is None:
            return
            
        # Apply joint positions and velocities
        if hasattr(traj_state, 'qpos') and traj_state.qpos is not None:
            qpos_tensor = torch.tensor(traj_state.qpos, device=self.device, dtype=torch.float32)
            if len(qpos_tensor.shape) == 1:
                qpos_tensor = qpos_tensor.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.set_dofs_position(qpos_tensor, envs_idx=env_ids)
            
        if hasattr(traj_state, 'qvel') and traj_state.qvel is not None:
            qvel_tensor = torch.tensor(traj_state.qvel, device=self.device, dtype=torch.float32)
            if len(qvel_tensor.shape) == 1:
                qvel_tensor = qvel_tensor.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.set_dofs_velocity(qvel_tensor, envs_idx=env_ids)
            
        # Apply root pose if available
        if hasattr(traj_state, 'root_pos') and traj_state.root_pos is not None:
            root_pos_tensor = torch.tensor(traj_state.root_pos, device=self.device, dtype=torch.float32)
            if len(root_pos_tensor.shape) == 1:
                root_pos_tensor = root_pos_tensor.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.set_pos(root_pos_tensor, envs_idx=env_ids)
            
        if hasattr(traj_state, 'root_quat') and traj_state.root_quat is not None:
            root_quat_tensor = torch.tensor(traj_state.root_quat, device=self.device, dtype=torch.float32)
            if len(root_quat_tensor.shape) == 1:
                root_quat_tensor = root_quat_tensor.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.set_quat(root_quat_tensor, envs_idx=env_ids)

    def _get_trajectory_target_state(self) -> Optional[TrajState]:
        """Get target state from trajectory at current time (from base env)"""
        if self.th is None:
            return None
            
        # Get current trajectory state
        current_traj_state = self.th.get_current_traj_data()
        return current_traj_state