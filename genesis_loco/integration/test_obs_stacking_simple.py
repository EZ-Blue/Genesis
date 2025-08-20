#!/usr/bin/env python3
"""
Simple Observation Stacking Test

Minimal test to verify observation stacking works correctly.
"""

import torch
import sys
import os

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.skeleton_humanoid import SkeletonHumanoidEnv
import genesis as gs


def test_basic_stacking():
    """Test basic observation stacking functionality"""
    print("🧪 Basic Observation Stacking Test")
    print("=" * 40)
    
    # Initialize Genesis
    try:
        gs.init(backend=gs.gpu)
        print("✅ Genesis initialized")
    except Exception as e:
        if "already initialized" in str(e):
            print("✅ Genesis already initialized")
        else:
            raise e
    
    # Test with minimal environment
    print("\n📊 Testing with history_length = 3")
    
    env = SkeletonHumanoidEnv(
        num_envs=4,  # Very small for testing
        episode_length_s=2.0,
        dt=0.01,
        show_viewer=False,
        use_box_feet=True,
        obs_history_length=3
    )
    
    print(f"✅ Environment created")
    print(f"   Observation dimension: {env.num_observations}")
    print(f"   Action dimension: {env.num_actions}")
    print(f"   History length: {env.obs_history_length}")
    
    # Expected dimensions
    base_obs_size = 5 + env.num_actions + 6 + env.num_actions  # Should be 65
    expected_total = base_obs_size * 3  # 195 with history
    
    print(f"   Expected base obs: {base_obs_size}")
    print(f"   Expected total obs: {expected_total}")
    
    assert env.num_observations == expected_total, f"Dimension mismatch: {env.num_observations} != {expected_total}"
    print(f"✅ Observation dimensions correct")
    
    # Test reset and step
    obs, _ = env.reset()
    print(f"✅ Reset successful, obs shape: {obs.shape}")
    
    # Test a few steps
    for i in range(3):
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        obs, rewards, dones, _ = env.step(actions)
        print(f"✅ Step {i+1} successful, obs shape: {obs.shape}")
    
    print(f"\n🎉 Basic observation stacking test passed!")
    return True


if __name__ == "__main__":
    try:
        success = test_basic_stacking()
        if success:
            print(f"\n🚀 Observation stacking implementation verified!")
            print(f"💡 Ready for training with temporal context")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"Check CUDA memory or restart Python session")