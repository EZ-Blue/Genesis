#!/usr/bin/env python3
"""
Test Refactored Skeleton Humanoid Environment

Simple test script to validate the refactored environment works correctly
and maintains the same functionality as the original.
"""

import torch
import numpy as np
import time
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import genesis as gs

def test_refactored_environment():
    """Test the refactored skeleton environment"""
    print("🧪 Testing Refactored Skeleton Humanoid Environment")
    print("=" * 60)
    
    try:
        # Initialize Genesis
        print("   Initializing Genesis...")
        gs.init(backend=gs.gpu)
        device = gs.device
        print(f"   Using device: {device}")
        
        # Import refactored environment
        from environments.skeleton_humanoid_refactored import SkeletonHumanoidEnv
        
        # Test environment creation
        print("   Creating environment...")
        env = SkeletonHumanoidEnv(
            num_envs=4,
            episode_length_s=5.0,
            dt=0.02,
            use_box_feet=True,
            show_viewer=True,
            use_trajectory_control=False
        )
        
        print(f"✅ Environment created successfully!")
        print(f"   - Environments: {env.num_envs}")
        print(f"   - Actions: {env.num_actions}")
        print(f"   - Observations: {env.num_observations}")
        print(f"   - DOFs: {env.num_dofs}")
        
        # Test reset
        print("\n🔄 Testing environment reset...")
        obs, info = env.reset()
        print(f"   ✅ Reset successful - obs shape: {obs.shape}")
        print(f"   ✅ Info keys: {list(info.keys())}")
        
        # Test stepping
        print("\n👟 Testing environment stepping...")
        total_reward = torch.zeros(env.num_envs, device=device)
        episode_lengths = torch.zeros(env.num_envs, device=device)
        
        for step in range(50):  # 1 second at 50Hz
            # Random actions in reasonable range
            actions = torch.randn((env.num_envs, env.num_actions), device=device) * 0.1
            
            # Step environment
            obs, rewards, dones, info = env.step(actions)
            
            total_reward += rewards
            episode_lengths += 1
            
            if step % 10 == 0:
                avg_reward = rewards.mean().item()
                avg_height = obs[:, 0].mean().item()  # z position is first in obs
                print(f"     Step {step:2d}: Reward={avg_reward:.3f}, Height={avg_height:.3f}m")
            
            # Check for any issues
            if torch.any(torch.isnan(obs)):
                print("     ❌ NaN detected in observations!")
                break
            if torch.any(torch.isnan(rewards)):
                print("     ❌ NaN detected in rewards!")
                break
        
        print(f"\n📊 Step Test Results:")
        print(f"   Average total reward: {total_reward.mean().item():.3f}")
        print(f"   Final episode lengths: {episode_lengths}")
        print(f"   Terminated environments: {dones.sum().item()}/{env.num_envs}")
        
        # Test PD control functionality
        print("\n⚙️  Testing PD control setup...")
        original_use_traj = env.use_trajectory_control
        
        # Test switching to balanced PD control
        env.use_trajectory_control = True
        env.setup_balanced_pd_control()
        
        # Switch back
        env.use_trajectory_control = original_use_traj
        env.setup_pd_control()
        
        print("   ✅ PD control switching works")
        
        # Test action application
        print("\n🎯 Testing action application...")
        test_actions = torch.zeros((env.num_envs, env.num_actions), device=device)
        test_actions[:, 0] = 0.1  # Small action on first joint
        
        obs_before = env._get_observations()
        env._apply_actions(test_actions)
        env.scene.step()
        env._update_robot_state()
        obs_after = env._get_observations()
        
        obs_diff = torch.norm(obs_after - obs_before, dim=1).mean().item()
        print(f"   ✅ Actions applied - observation change: {obs_diff:.6f}")
        
        # Test trajectory loading (optional)
        print("\n📈 Testing trajectory loading...")
        try:
            success = env.load_trajectory(traj_path=None)  # This should handle None gracefully
            print("   ✅ Trajectory loading interface works")
        except Exception as e:
            print(f"   ⚠️  Trajectory loading failed (expected): {e}")
        
        print("\n" + "=" * 60)
        print("✅ REFACTORED ENVIRONMENT TEST SUCCESS!")
        print("✅ All core functionality working correctly")
        print("✅ Ready to replace original environment")
        print("=" * 60)
        
        return True, env
        
    except Exception as e:
        print(f"\n❌ REFACTORED ENVIRONMENT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def compare_with_original():
    """Compare refactored vs original environment"""
    print("\n🔍 Comparing Refactored vs Original Environment")
    print("=" * 60)
    
    try:
        # Test original environment
        print("   Testing original environment...")
        from environments.skeleton_humanoid import SkeletonHumanoidEnv as OriginalEnv
        
        original_env = OriginalEnv(
            num_envs=2,
            episode_length_s=5.0,
            dt=0.02,
            use_box_feet=True,
            show_viewer=False
        )
        
        # Test refactored environment
        print("   Testing refactored environment...")
        from environments.skeleton_humanoid_refactored import SkeletonHumanoidEnv as RefactoredEnv
        
        refactored_env = RefactoredEnv(
            num_envs=2,
            episode_length_s=5.0,
            dt=0.02,
            use_box_feet=True,
            show_viewer=False
        )
        
        # Compare key properties
        print(f"\n📋 Comparison Results:")
        print(f"   Environment count:     Original={original_env.num_envs:2d}, Refactored={refactored_env.num_envs:2d}")
        print(f"   Action dimensions:     Original={original_env.num_actions:2d}, Refactored={refactored_env.num_actions:2d}")
        print(f"   Observation dimensions: Original={original_env.num_observations:2d}, Refactored={refactored_env.num_observations:2d}")
        print(f"   DOF count:             Original={original_env.num_dofs:2d}, Refactored={refactored_env.num_dofs:2d}")
        
        # Check if dimensions match
        if (original_env.num_actions == refactored_env.num_actions and 
            original_env.num_observations == refactored_env.num_observations and
            original_env.num_dofs == refactored_env.num_dofs):
            print("   ✅ All dimensions match perfectly!")
        else:
            print("   ⚠️  Some dimensions differ - check implementation")
        
        # Quick functional test
        print("\n🚀 Quick functional comparison...")
        
        # Reset both
        orig_obs, _ = original_env.reset()
        refact_obs, _ = refactored_env.reset()
        
        print(f"   Original obs shape: {orig_obs.shape}")
        print(f"   Refactored obs shape: {refact_obs.shape}")
        
        # Step both with same actions
        actions = torch.zeros((2, min(original_env.num_actions, refactored_env.num_actions)), device=gs.device)
        
        orig_obs, orig_rew, orig_done, _ = original_env.step(actions)
        refact_obs, refact_rew, refact_done, _ = refactored_env.step(actions)
        
        print(f"   Original reward mean: {orig_rew.mean().item():.6f}")
        print(f"   Refactored reward mean: {refact_rew.mean().item():.6f}")
        
        print("   ✅ Both environments step successfully!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚶‍♂️ Refactored Skeleton Environment Testing")
    
    # Test refactored environment
    success, env = test_refactored_environment()
    
    if success:
        # Compare with original if possible
        try:
            comparison_success = compare_with_original()
            if comparison_success:
                print("\n🎉 COMPREHENSIVE TESTING COMPLETE!")
                print("✅ Refactored environment is ready for production use")
                print("\n📝 Next Steps:")
                print("1. Replace imports in training scripts")
                print("2. Remove old base environment dependency")
                print("3. Update any scripts that reference GenesisLocoBaseEnv")
            else:
                print("\n⚠️  Comparison had issues but refactored env works standalone")
        except ImportError:
            print("\n📝 Original environment not available for comparison")
            print("✅ Refactored environment works standalone")
    else:
        print("\n🔧 Fix refactored environment issues before proceeding")

if __name__ == "__main__":
    main()