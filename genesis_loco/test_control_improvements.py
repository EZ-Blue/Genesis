"""
Test Control Improvements

Verify that trajectory-optimized PD gains improve tracking performance.
"""

import torch
import sys
import os
import time

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
genesis_loco_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, genesis_loco_dir)

def test_control_improvements():
    """Test improved control parameters"""
    print("🎯 Testing Control Improvements")
    print("=" * 50)
    
    # Setup Genesis
    import genesis as gs
    gs.init(backend=gs.gpu)
    from environments.skeleton_humanoid import SkeletonHumanoidEnv
    
    print("\n1. Testing Original Control (high PD gains)")
    env_original = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=5.0,
        dt=0.019,  # 100Hz
        show_viewer=False,
        use_trajectory_control=False  # Original high gains
    )
    env_original.setup_pd_control()  # Apply original gains
    
    test_control_response(env_original, "Original Control")
    
    print("\n2. Testing Trajectory-Optimized Control (low PD gains)")
    env_optimized = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=5.0,
        dt=0.019,  # 100Hz  
        show_viewer=False,
        use_trajectory_control=True  # Trajectory-optimized gains
    )
    
    test_control_response(env_optimized, "Trajectory-Optimized Control")

def test_control_response(env, control_name):
    """Test control response for a given environment"""
    print(f"\n--- {control_name} ---")
    
    # Reset and stabilize
    obs, _ = env.reset()
    for _ in range(20):
        env.step(torch.zeros(1, env.num_actions, device=env.device))
    
    # Test position control on first controllable DOF
    initial_pos = env.dof_pos[0, 0].item()
    target_pos = 0.1  # Small target movement
    
    print(f"  Initial pos: {initial_pos:.4f}")
    print(f"  Target pos:  {target_pos:.4f}")
    
    # Create position command
    target_positions = torch.zeros(1, env.num_dofs, device=env.device)
    target_positions[0, 0] = target_pos
    
    # Apply position control and track response
    errors = []
    for step in range(50):  # 0.5 seconds at 100Hz
        env.robot.control_dofs_position(target_positions)
        env.scene.step()
        env._update_robot_state()
        
        current_pos = env.dof_pos[0, 0].item()
        error = abs(current_pos - target_pos)
        errors.append(error)
        
        if step % 10 == 0:
            print(f"    Step {step:2d}: pos={current_pos:.4f}, error={error:.4f}")
    
    final_error = errors[-1]
    avg_error = sum(errors) / len(errors)
    
    print(f"  Final error: {final_error:.4f}")
    print(f"  Average error: {avg_error:.4f}")
    
    # Assessment
    if final_error < 0.01:
        print("  ✅ Excellent control response")
    elif final_error < 0.05:
        print("  ✅ Good control response") 
    elif final_error < 0.1:
        print("  ⚠️ Moderate control response")
    else:
        print("  ❌ Poor control response")
    
    return final_error, avg_error

if __name__ == "__main__":
    try:
        test_control_improvements()
        
        print(f"\n🎯 Summary:")
        print("If trajectory-optimized control shows better performance:")
        print("  - Lower final errors")
        print("  - Smoother response (less oscillation)")
        print("  - Better trajectory tracking expected")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()