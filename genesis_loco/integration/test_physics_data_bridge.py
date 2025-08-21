#!/usr/bin/env python3
"""
Simple Test: Physics-Based Data Bridge

Quick test to verify the updated apply_trajectory_state method works correctly
with the skeleton humanoid environment and produces smooth motion.
"""

import torch
import time
import sys
import os

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge


def test_physics_data_bridge():
    """Test the updated physics-based data bridge"""
    print("🧪 TESTING PHYSICS-BASED DATA BRIDGE")
    print("=" * 50)
    
    # Initialize Genesis with viewer
    print("1. Initializing Genesis with viewer...")
    gs.init(backend=gs.gpu)
    
    # Create skeleton environment
    print("2. Creating skeleton environment...")
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=10.0,
        dt=0.01,
        use_box_feet=True,
        show_viewer=True  # ENABLE VIEWER
    )
    print(f"   ✅ Environment created: {env.num_observations} obs, {env.num_actions} actions")
    
    # Create data bridge
    print("3. Loading trajectory data...")
    data_bridge = LocoMujocoDataBridge(env)
    success = data_bridge.load_trajectory("walk")
    
    if not success:
        print("❌ Failed to load trajectory")
        return False
    
    print(f"   ✅ Trajectory loaded: {data_bridge.trajectory_length} timesteps")
    
    # Test trajectory application
    print("4. Testing physics-based trajectory application...")
    print("   📺 Watch the Genesis viewer for smooth walking motion!")
    
    env_ids = torch.tensor([0], device=env.device)
    
    # Test parameters
    start_timestep = 100  # Skip initial frames
    test_duration = 500   # 5 seconds of motion
    
    print(f"   Testing {test_duration} timesteps starting from {start_timestep}")
    print("   Expected: Smooth, continuous walking motion")
    
    # Apply trajectory states and visualize
    for step in range(test_duration):
        trajectory_idx = start_timestep + step
        
        if trajectory_idx >= data_bridge.trajectory_length:
            break
        
        # Get trajectory state
        state_data = data_bridge.get_trajectory_state(trajectory_idx)
        if state_data is None:
            continue
        
        # Apply with physics (NEW METHOD)
        data_bridge.apply_trajectory_state(state_data, env_ids)
        
        # Small delay for visual inspection
        time.sleep(0.02)  # 50 FPS playback
        
        # Progress indication
        if step % 100 == 0:
            progress = (step / test_duration) * 100
            current_height = env.root_pos[0, 2].item()
            print(f"   Step {step}/{test_duration} ({progress:.0f}%) - Height: {current_height:.3f}m")
    
    print("   ✅ Physics-based trajectory application completed!")
    
    # Test observation collection
    print("5. Testing expert observation collection...")
    
    expert_obs_list = []
    collection_steps = 100
    
    for step in range(collection_steps):
        trajectory_idx = start_timestep + step
        
        if trajectory_idx >= data_bridge.trajectory_length:
            break
        
        # Apply trajectory state (with physics)
        state_data = data_bridge.get_trajectory_state(trajectory_idx)
        if state_data is not None:
            data_bridge.apply_trajectory_state(state_data, env_ids)
            
            # Collect observation
            obs = env._get_observations()
            expert_obs_list.append(obs[0])
    
    if expert_obs_list:
        expert_observations = torch.stack(expert_obs_list, dim=0)
        print(f"   ✅ Collected {expert_observations.shape[0]} expert observations")
        print(f"   Observation range: [{expert_observations.min():.3f}, {expert_observations.max():.3f}]")
        print(f"   Mean: {expert_observations.mean():.6f}, Std: {expert_observations.std():.6f}")
    else:
        print("   ❌ Failed to collect observations")
        return False
    
    # Final assessment
    print("\n🎯 ASSESSMENT:")
    print("   1. Did you see smooth, continuous walking motion in the viewer?")
    print("   2. Was there no jerky or discontinuous movement?")
    print("   3. Did the character move forward naturally?")
    print("")
    print("   If YES to all: ✅ Physics-based data bridge is working correctly!")
    print("   If NO to any: ❌ There may still be issues with the implementation")
    
    # Keep viewer open
    input("\nPress Enter to close the test...")
    
    return True


def main():
    """Main test function"""
    print("🔬 PHYSICS-BASED DATA BRIDGE TEST")
    print("=" * 60)
    print("This test verifies the updated data bridge produces smooth trajectory motion")
    
    try:
        success = test_physics_data_bridge()
        
        if success:
            print("\n✅ Test completed successfully!")
            print("Your GAIL training should now use smooth expert observations.")
        else:
            print("\n❌ Test failed!")
            return 1
    
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())