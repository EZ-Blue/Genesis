#!/usr/bin/env python3
"""
Diagnose Trajectory Application - Verify expert trajectory states apply correctly to Genesis

This script creates a Genesis skeleton environment, loads the LocoMujoco expert walking trajectory,
applies the trajectory states to the Genesis model, and visualizes the result to verify correctness.
"""

import torch
import numpy as np
import sys
import os
import time

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')
sys.path.append('/home/ez/Documents/loco-mujoco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge


def diagnose_trajectory_application():
    """Test trajectory application and visualize the result"""
    print("🔍 TRAJECTORY APPLICATION DIAGNOSIS")
    print("=" * 60)
    
    # Initialize Genesis with viewer for visualization
    print("1. Initializing Genesis with visualization...")
    gs.init(backend=gs.gpu)
    
    # Create environment EXACTLY like training (single env for visualization)
    print("2. Creating skeleton environment...")
    env = SkeletonHumanoidEnv(
        num_envs=1,  # Single environment for clear visualization
        episode_length_s=10.0,  # Longer for observation
        dt=0.01,  # Match training
        use_box_feet=True,
        show_viewer=True  # ENABLE VIEWER for diagnosis
    )
    
    print(f"   ✅ Environment created:")
    print(f"      - Observations: {env.num_observations}")
    print(f"      - Actions: {env.num_actions}")
    print(f"      - DOF count: {env.robot.n_dofs}")
    print(f"      - Controlled DOFs: {len(env.motors_dof_idx)}")
    
    # Create data bridge EXACTLY like training
    print("3. Creating data bridge...")
    data_bridge = LocoMujocoDataBridge(env)
    success = data_bridge.load_trajectory("walk")
    
    if not success:
        print("❌ Failed to load trajectory!")
        return False
    
    print(f"   ✅ Trajectory loaded:")
    print(f"      - Length: {data_bridge.trajectory_length} timesteps")
    print(f"      - Duration: {data_bridge.trajectory_length * 0.01:.1f}s (assuming 0.01s timesteps)")
    
    # Test single trajectory application
    print("4. Testing single trajectory state application...")
    
    env_ids = torch.tensor([0], device=env.device)
    
    # Test a few different timesteps
    test_timesteps = [0, 100, 500, 1000, min(2000, data_bridge.trajectory_length-1)]
    
    for i, timestep in enumerate(test_timesteps):
        if timestep >= data_bridge.trajectory_length:
            continue
            
        print(f"   Testing timestep {timestep}...")
        
        # Get trajectory state
        state_data = data_bridge.get_trajectory_state(timestep)
        if state_data is None:
            print(f"   ❌ Failed to get state for timestep {timestep}")
            continue
        
        print(f"      State data keys: {list(state_data.keys())}")
        print(f"      DOF pos shape: {state_data['dof_pos'].shape}")
        print(f"      DOF vel shape: {state_data['dof_vel'].shape}")
        print(f"      Root pos: {state_data['root_pos']}")
        print(f"      Root quat: {state_data['root_quat']}")
        
        # Apply to Genesis environment
        try:
            data_bridge.apply_trajectory_state(state_data, env_ids)
            
            # Get observations to verify
            obs = env._get_observations()
            print(f"      ✅ Applied successfully, obs shape: {obs.shape}")
            print(f"      Root height (obs[2]): {obs[0, 2].item():.3f}")
            print(f"      Joint pos range: [{obs[0, 5:32].min().item():.3f}, {obs[0, 5:32].max().item():.3f}]")
            
        except Exception as e:
            print(f"      ❌ Failed to apply state: {e}")
    
    print(f"\n5. Visualizing trajectory playback...")
    print(f"   🎬 Playing expert walking trajectory in Genesis viewer...")
    print(f"   📺 Watch the Genesis viewer window to verify motion looks correct!")
    
    # Play trajectory for visual inspection
    playback_duration = min(5.0, data_bridge.trajectory_length * 0.01)  # Max 5 seconds
    playback_timesteps = int(playback_duration / 0.01)  # Match env dt
    trajectory_step_size = max(1, data_bridge.trajectory_length // playback_timesteps)
    
    print(f"   Playback parameters:")
    print(f"      - Duration: {playback_duration:.1f}s")
    print(f"      - Timesteps: {playback_timesteps}")
    print(f"      - Trajectory step size: {trajectory_step_size}")
    
    # Reset to initial pose
    env.reset()
    
    start_time = time.time()
    
    for step in range(playback_timesteps):
        # Calculate which trajectory timestep to use
        traj_timestep = (step * trajectory_step_size) % data_bridge.trajectory_length
        
        # Get and apply trajectory state
        state_data = data_bridge.get_trajectory_state(traj_timestep)
        if state_data is not None:
            data_bridge.apply_trajectory_state(state_data, env_ids)
        
        # Step simulation to update viewer
        env.scene.step()
        
        # Control playback speed
        time.sleep(0.02)  # 50 FPS playback
        
        # Progress indication
        if step % 25 == 0:
            progress = (step / playback_timesteps) * 100
            print(f"      Progress: {progress:.0f}% (step {step}/{playback_timesteps})")
    
    elapsed_time = time.time() - start_time
    print(f"   ✅ Playback completed in {elapsed_time:.1f}s")
    
    # Detailed state comparison
    print(f"\n6. Detailed state analysis...")
    
    # Compare first and middle trajectory states
    state_0 = data_bridge.get_trajectory_state(0)
    state_mid = data_bridge.get_trajectory_state(data_bridge.trajectory_length // 2)
    
    if state_0 and state_mid:
        print(f"   Initial state (t=0):")
        print(f"      Root pos: {state_0['root_pos']}")
        print(f"      Root quat: {state_0['root_quat']}")
        print(f"      Joint pos range: [{state_0['dof_pos'].min().item():.3f}, {state_0['dof_pos'].max().item():.3f}]")
        
        print(f"   Middle state (t={data_bridge.trajectory_length // 2}):")
        print(f"      Root pos: {state_mid['root_pos']}")
        print(f"      Root quat: {state_mid['root_quat']}")
        print(f"      Joint pos range: [{state_mid['dof_pos'].min().item():.3f}, {state_mid['dof_pos'].max().item():.3f}]")
        
        # Check for reasonable motion
        root_displacement = torch.norm(state_mid['root_pos'] - state_0['root_pos']).item()
        joint_change = torch.norm(state_mid['dof_pos'] - state_0['dof_pos']).item()
        
        print(f"   Motion analysis:")
        print(f"      Root displacement: {root_displacement:.3f}m")
        print(f"      Joint configuration change: {joint_change:.3f}rad")
        
        if root_displacement > 0.1:  # Should move during walking
            print(f"      ✅ Root is moving (good for walking)")
        else:
            print(f"      ⚠️  Root barely moving (unexpected for walking)")
        
        if joint_change > 0.5:  # Joints should change significantly
            print(f"      ✅ Joints are changing significantly")
        else:
            print(f"      ⚠️  Joints changing very little")
    
    # Test observation consistency
    print(f"\n7. Testing observation consistency...")
    
    # Apply same state twice and check observations are identical
    test_timestep = 100
    state_data = data_bridge.get_trajectory_state(test_timestep)
    
    if state_data:
        # First application
        data_bridge.apply_trajectory_state(state_data, env_ids)
        obs1 = env._get_observations()
        
        # Second application
        data_bridge.apply_trajectory_state(state_data, env_ids)
        obs2 = env._get_observations()
        
        # Check consistency
        obs_diff = torch.norm(obs1 - obs2).item()
        print(f"   Observation consistency test:")
        print(f"      Difference between identical applications: {obs_diff:.10f}")
        
        if obs_diff < 1e-6:
            print(f"      ✅ Observations are consistent")
        else:
            print(f"      ⚠️  Observations differ (potential issue)")
    
    # Final assessment
    print(f"\n🎯 TRAJECTORY APPLICATION ASSESSMENT:")
    print(f"   1. Check the Genesis viewer window:")
    print(f"      - Does the skeleton look like a human?")
    print(f"      - Does the motion look like walking?")
    print(f"      - Are there any unnatural poses or movements?")
    print(f"      - Is the character moving forward smoothly?")
    print(f"   ")
    print(f"   2. If the motion looks wrong:")
    print(f"      - Joint ordering might be mismatched")
    print(f"      - Coordinate frames might be different")
    print(f"      - Units or scaling might be wrong")
    print(f"      - Skeleton structure differences")
    print(f"   ")
    print(f"   3. If the motion looks correct:")
    print(f"      - Expert trajectory application is working")
    print(f"      - Distribution mismatch issue is elsewhere")
    print(f"      - Focus on discriminator overconfidence solutions")
    
    print(f"\n✅ Trajectory application diagnosis complete!")
    print(f"📺 Keep the viewer window open to inspect the motion manually")
    
    # Keep viewer open for manual inspection
    input("Press Enter to close the viewer and exit...")
    
    return True


def main():
    print("🧪 TRAJECTORY APPLICATION DIAGNOSIS")
    print("=" * 70)
    print("This script will visualize the expert trajectory applied to Genesis skeleton model")
    print("Look for natural walking motion in the Genesis viewer window")
    
    try:
        success = diagnose_trajectory_application()
        
        if success:
            print(f"\n✅ Diagnosis completed!")
            print(f"Based on the visualization, assess whether trajectory application is correct.")
        else:
            print(f"\n❌ Diagnosis failed!")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())