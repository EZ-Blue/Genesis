"""
Test Torque Computation

Quick test to verify the torque-based approach works correctly
before retraining the behavior cloning model.
"""

import torch
import sys
sys.path.append('.')

from integration.data_bridge import LocoMujocoDataBridge
from environments.skeleton_humanoid import SkeletonHumanoidEnv
import genesis as gs

def test_torque_computation():
    """Test that torque computation works and produces reasonable values"""
    print("🔧 TESTING TORQUE COMPUTATION")
    print("=" * 50)
    
    # Initialize Genesis and environment
    gs.init(backend=gs.gpu)
    env = SkeletonHumanoidEnv(
        num_envs=1, 
        episode_length_s=8.0, 
        dt=0.01, 
        use_box_feet=True
    )
    
    # Create data bridge and load trajectory
    data_bridge = LocoMujocoDataBridge(env)
    success = data_bridge.load_trajectory('walk')
    
    if not success:
        print("❌ Failed to load trajectory")
        return False
        
    print(f"✅ Loaded trajectory: {data_bridge.trajectory_length} timesteps")
    print(f"   Motor count: {len(env.motors_dof_idx)}")
    
    # Test torque computation on several timesteps
    print(f"\n📊 Testing torque computation:")
    
    test_timesteps = [100, 1000, 5000, 10000]
    torque_stats = []
    
    for timestep in test_timesteps:
        if timestep >= data_bridge.trajectory_length - 1:
            continue
            
        torques = data_bridge.compute_expert_torques(timestep)
        
        if torques is not None:
            torque_range = (torques.min().item(), torques.max().item())
            torque_magnitude = torch.norm(torques).item()
            
            torque_stats.append({
                'timestep': timestep,
                'range': torque_range,
                'magnitude': torque_magnitude,
                'mean': torques.mean().item(),
                'std': torques.std().item()
            })
            
            print(f"   Timestep {timestep}: range=[{torque_range[0]:.3f}, {torque_range[1]:.3f}], "
                  f"magnitude={torque_magnitude:.3f}")
        else:
            print(f"   Timestep {timestep}: ❌ Failed to compute torques")
    
    if len(torque_stats) > 0:
        # Overall statistics
        all_ranges = [stat['range'] for stat in torque_stats]
        all_magnitudes = [stat['magnitude'] for stat in torque_stats]
        
        min_torque = min([r[0] for r in all_ranges])
        max_torque = max([r[1] for r in all_ranges])
        avg_magnitude = sum(all_magnitudes) / len(all_magnitudes)
        
        print(f"\n📈 Overall Torque Statistics:")
        print(f"   Global range: [{min_torque:.3f}, {max_torque:.3f}]")
        print(f"   Average magnitude: {avg_magnitude:.3f}")
        print(f"   Expected model output range: ~[-0.8, 0.7] (from debug)")
        
        # Check if scaling makes sense
        scaling_factor = 50.0  # From _apply_actions
        scaled_range = (min_torque / scaling_factor, max_torque / scaling_factor)
        print(f"   With 50x scaling: model should output [{scaled_range[0]:.3f}, {scaled_range[1]:.3f}]")
        
        if abs(scaled_range[0]) <= 1.0 and abs(scaled_range[1]) <= 1.0:
            print("   ✅ Scaling looks reasonable for model outputs!")
        else:
            print("   ⚠️  Scaling might need adjustment")
            
        return True
    else:
        print("❌ No successful torque computations")
        return False

def test_environment_torque_application():
    """Test that the environment can apply torques properly"""
    print(f"\n🏃 TESTING ENVIRONMENT TORQUE APPLICATION")
    print("=" * 50)
    
    # Create environment (reuse from above if possible)
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=8.0, 
        dt=0.01,
        use_box_feet=True
    )
    
    # Test with different torque magnitudes
    test_torques = [
        torch.zeros(env.num_actions),                    # Zero torques
        torch.ones(env.num_actions) * 10.0,             # Small torques
        torch.randn(env.num_actions) * 25.0,            # Random torques  
        torch.ones(env.num_actions) * 50.0,             # Large torques
    ]
    
    print(f"   Number of actions: {env.num_actions}")
    
    for i, torques in enumerate(test_torques):
        try:
            # Apply torques through the environment
            obs, rewards, dones, info = env.step(torques.unsqueeze(0))
            
            torque_magnitude = torch.norm(torques).item()
            print(f"   Test {i}: torque magnitude={torque_magnitude:.1f} → ✅ Applied successfully")
            
        except Exception as e:
            print(f"   Test {i}: torque magnitude={torch.norm(torques).item():.1f} → ❌ Failed: {e}")
            
    return True

def main():
    """Run all tests"""
    print("🚀 TORQUE-BASED CONTROL TESTS")
    print("=" * 70)
    
    success1 = test_torque_computation()
    success2 = test_environment_torque_application()
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   You can now retrain the behavior cloning model with torque targets.")
        print(f"   Run: python integration/single_trajectory_behavior_cloning.py")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        print(f"   Check the torque computation and environment setup.")

if __name__ == "__main__":
    main()