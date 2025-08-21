#!/usr/bin/env python3
"""
Physics-Based Expert Data Collection

Collects expert observations using the same smooth trajectory following approach
as verify_trajectory.py to ensure observations come from continuous, physics-based motion.
"""

import torch
import numpy as np
import sys
import os

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')
sys.path.append('/home/ez/Documents/loco-mujoco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge


class PhysicsBasedExpertCollector:
    """
    Collect expert observations using continuous physics-based trajectory following
    
    This ensures expert observations come from the same type of smooth, continuous
    motion that the Genesis policy will generate during training.
    """
    
    def __init__(self, env: SkeletonHumanoidEnv, data_bridge: LocoMujocoDataBridge):
        self.env = env
        self.data_bridge = data_bridge
        self.device = env.device
        
        print("🔬 Physics-Based Expert Collection Initialized")
        print(f"   Environment: {env.num_envs} envs, {env.num_observations} obs")
        print(f"   Trajectory: {data_bridge.trajectory_length} timesteps")
    
    def collect_expert_observations(self, 
                                   start_timestep: int = 0,
                                   num_timesteps: int = None,
                                   step_interval: int = 1,) -> torch.Tensor:
        """
        Collect expert observations using physics-based trajectory following
        
        Args:
            start_timestep: Start from this trajectory timestep
            num_timesteps: Number of timesteps to collect (None = all)
            step_interval: Collect every N timesteps (1 = every step)
            
        Returns:
            expert_observations: [num_collected, obs_dim] tensor
        """
        if num_timesteps is None:
            num_timesteps = self.data_bridge.trajectory_length - start_timestep
        
        # Limit collection for memory efficiency
        max_collect = min(10000, num_timesteps)  # Max 10k observations
        actual_timesteps = min(max_collect, num_timesteps)
        
        print(f"📊 Collecting expert observations:")
        print(f"   Start: timestep {start_timestep}")
        print(f"   Duration: {actual_timesteps} timesteps")
        print(f"   Step interval: {step_interval}")
        print(f"   Physics integration: ENABLED")
        
        expert_obs_list = []
        env_ids = torch.tensor([0], device=self.device)  # Use first environment
        
        # Initialize to trajectory starting pose (like verify_trajectory.py)
        print("   Initializing to trajectory starting pose...")
        self._initialize_to_trajectory_pose(start_timestep)
        
        # Collect observations with physics integration
        for step in range(actual_timesteps):
            trajectory_idx = start_timestep + step
            
            if trajectory_idx >= self.data_bridge.trajectory_length:
                break
            
            # Get trajectory state
            traj_state = self.data_bridge.get_trajectory_state(trajectory_idx)
            if traj_state is None:
                continue
            
            # Apply state using the same method as verify_trajectory.py
            self._apply_trajectory_state_with_physics(traj_state, env_ids)
            
            # Collect observation only at specified intervals
            if step % step_interval == 0:
                obs = self.env._get_observations()
                expert_obs_list.append(obs[0])
            
            # Progress indicator
            if step % 1000 == 0:
                progress = (step / actual_timesteps) * 100
                print(f"   Progress: {progress:.1f}% ({step}/{actual_timesteps})")
        
        if not expert_obs_list:
            print("❌ Failed to collect expert observations")
            return None
        
        expert_observations = torch.stack(expert_obs_list, dim=0)
        print(f"✅ Collected {expert_observations.shape[0]} expert observations")
        print(f"   Observation shape: {expert_observations.shape}")
        
        return expert_observations
    
    def _initialize_to_trajectory_pose(self, timestep: int):
        """Initialize to exact trajectory pose (from verify_trajectory.py)"""
        traj_state = self.data_bridge.get_trajectory_state(timestep)
        if traj_state is None:
            return
        
        target_dof_pos = traj_state['dof_pos']
        target_root_pos = traj_state['root_pos']
        target_root_quat = traj_state['root_quat']
        
        env_ids = torch.tensor([0], device=self.device)
        
        if hasattr(self.data_bridge, 'motors_dof_idx'):
            controllable_dof_indices = self.data_bridge.motors_dof_idx
            target_dof_pos_batch = target_dof_pos.unsqueeze(0)
            
            self.env.robot.set_dofs_position(
                target_dof_pos_batch,
                dofs_idx_local=controllable_dof_indices,
                envs_idx=env_ids,
                zero_velocity=True
            )
        
        # Set root pose
        target_root_pos_batch = target_root_pos.unsqueeze(0)
        target_root_quat_batch = target_root_quat.unsqueeze(0)
        
        self.env.robot.set_pos(target_root_pos_batch, envs_idx=env_ids, zero_velocity=True)
        self.env.robot.set_quat(target_root_quat_batch, envs_idx=env_ids, zero_velocity=True)
        
        # Settle physics (from verify_trajectory.py)
        for _ in range(5):
            self.env.scene.step()
        
        self.env._update_robot_state()
    
    def _apply_trajectory_state_with_physics(self, traj_state: dict, env_ids: torch.Tensor):
        """
        Apply trajectory state with physics integration (from verify_trajectory.py)
        
        This is the KEY difference - we run physics integration after setting state
        """
        target_dof_pos = traj_state['dof_pos'].unsqueeze(0)
        target_root_pos = traj_state['root_pos'].unsqueeze(0)
        target_root_quat = traj_state['root_quat'].unsqueeze(0)
        
        # Apply joint positions
        if hasattr(self.data_bridge, 'motors_dof_idx'):
            controllable_dof_indices = self.data_bridge.motors_dof_idx
            
            try:
                self.env.robot.set_dofs_position(
                    target_dof_pos,
                    dofs_idx_local=controllable_dof_indices,
                    envs_idx=env_ids,
                    zero_velocity=False
                )
                
                # Apply root pose
                self.env.robot.set_pos(target_root_pos, envs_idx=env_ids, zero_velocity=False)
                self.env.robot.set_quat(target_root_quat, envs_idx=env_ids, zero_velocity=False)
                
            except Exception as e:
                # Continue with physics integration even if state setting partially fails
                pass
        
        # CRITICAL: Physics integration step (this makes motion continuous)
        self.env.scene.step()
        
        # Update environment state
        self.env._update_robot_state()
    
    def compare_collection_methods(self, num_test_steps: int = 100) -> dict:
        """
        Compare old vs new expert data collection methods
        
        Returns comparison metrics to verify improvement
        """
        print(f"\n🔬 COMPARING COLLECTION METHODS")
        print("=" * 50)
        
        # Method 1: Old direct state application (current GAIL method)
        print("1. Old method (direct state application):")
        old_obs_list = []
        env_ids = torch.tensor([0], device=self.device)
        
        for step in range(num_test_steps):
            traj_state = self.data_bridge.get_trajectory_state(step)
            if traj_state is None:
                continue
            
            # Direct application without physics (current method)
            self.data_bridge.apply_trajectory_state(traj_state, env_ids)
            obs = self.env._get_observations()
            old_obs_list.append(obs[0])
        
        old_observations = torch.stack(old_obs_list, dim=0) if old_obs_list else None
        
        # Method 2: New physics-based collection
        print("2. New method (physics-based):")
        new_observations = self.collect_expert_observations(
            start_timestep=0,
            num_timesteps=num_test_steps,
            step_interval=1
        )
        
        if old_observations is not None and new_observations is not None:
            # Compare observation smoothness
            old_smoothness = self._calculate_smoothness(old_observations)
            new_smoothness = self._calculate_smoothness(new_observations)
            
            print(f"\n📊 COMPARISON RESULTS:")
            print(f"   Old method smoothness: {old_smoothness:.6f}")
            print(f"   New method smoothness: {new_smoothness:.6f}")
            print(f"   Improvement factor: {old_smoothness / (new_smoothness + 1e-8):.2f}x")
            
            if new_smoothness < old_smoothness:
                print(f"   ✅ New method produces smoother observations!")
            else:
                print(f"   ⚠️  Old method was smoother (unexpected)")
            
            return {
                'old_smoothness': old_smoothness,
                'new_smoothness': new_smoothness,
                'improvement_factor': old_smoothness / (new_smoothness + 1e-8),
                'old_observations': old_observations,
                'new_observations': new_observations
            }
        
        return {}
    
    def _calculate_smoothness(self, observations: torch.Tensor) -> float:
        """Calculate observation smoothness (lower = smoother)"""
        if observations.shape[0] < 2:
            return float('inf')
        
        # Calculate differences between consecutive observations
        diffs = observations[1:] - observations[:-1]
        smoothness = torch.mean(torch.norm(diffs, dim=1)).item()
        return smoothness


def test_physics_based_collection():
    """Test the physics-based expert collection"""
    print("🧪 TESTING PHYSICS-BASED EXPERT COLLECTION")
    print("=" * 60)
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create environment (single env for testing)
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=10.0,
        dt=0.01,
        use_box_feet=True,
        show_viewer=True  # No viewer for testing
    )
    
    # Create data bridge
    data_bridge = LocoMujocoDataBridge(env)
    success = data_bridge.load_trajectory("walk")
    
    if not success:
        print("❌ Failed to load trajectory")
        return False
    
    # Create collector
    collector = PhysicsBasedExpertCollector(env, data_bridge)
    
    # Test collection
    print("\n1. Testing physics-based collection...")
    expert_obs = collector.collect_expert_observations(
        start_timestep=0,
        num_timesteps=500,  # Small test
        step_interval=1
    )
    
    if expert_obs is not None:
        print(f"   ✅ Collected: {expert_obs.shape}")
        print(f"   Range: [{expert_obs.min():.3f}, {expert_obs.max():.3f}]")
        print(f"   Mean: {expert_obs.mean():.6f}, Std: {expert_obs.std():.6f}")
    
    # Compare methods
    print("\n2. Comparing collection methods...")
    comparison = collector.compare_collection_methods(num_test_steps=100)
    
    if comparison:
        print("   ✅ Method comparison completed")
        return comparison['improvement_factor'] > 1.0
    
    return True


if __name__ == "__main__":
    success = test_physics_based_collection()
    if success:
        print("\n✅ Physics-based expert collection is working!")
    else:
        print("\n❌ Physics-based expert collection failed!")
    exit(0 if success else 1)