"""
Debug Control Parameters

Quick diagnostic to understand control issues in trajectory following.
"""

import torch
import sys
import os

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
genesis_loco_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, genesis_loco_dir)

def analyze_control_parameters():
    """Analyze Genesis vs LocoMujoco control setup"""
    print("🔧 Control Parameters Analysis")
    print("=" * 50)
    
    # Setup Genesis environment
    import genesis as gs
    gs.init(backend=gs.gpu)
    from environments.skeleton_humanoid import SkeletonHumanoidEnv
    
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=5.0,
        dt=0.019,  # 50Hz
        show_viewer=False,
        use_trajectory_control=True
    )
    
    print(f"Genesis Control Setup:")
    print(f"  - Simulation dt: {env.dt}")
    print(f"  - Simulation frequency: {1/env.dt} Hz")
    print(f"  - Controllable DOFs: {env.num_dofs}")

    env.setup_pd_control()
    
    # Check PD gains
    kp_values = env.robot.get_dofs_kp()
    kv_values = env.robot.get_dofs_kv()
    
    print(f"\nPD Control Gains:")
    print(f"  - Position gains (kp): min={kp_values.min():.1f}, max={kp_values.max():.1f}, mean={kp_values.mean():.1f}")
    print(f"  - Velocity gains (kv): min={kv_values.min():.1f}, max={kv_values.max():.1f}, mean={kv_values.mean():.1f}")
    
    # Check joint limits
    joint_lower = env.robot.get_dofs_limit()[0]
    joint_upper = env.robot.get_dofs_limit()[1]
    
    print(f"\nJoint Limits:")
    print(f"  - Lower limits: min={joint_lower.min():.2f}, max={joint_lower.max():.2f}")
    print(f"  - Upper limits: min={joint_upper.min():.2f}, max={joint_upper.max():.2f}")
    print(f"  - Joint ranges: min={(joint_upper-joint_lower).min():.2f}, max={(joint_upper-joint_lower).max():.2f}")
    
    # Test single step control response
    print(f"\nControl Response Test:")
    
    # Reset environment
    obs, _ = env.reset()
    initial_pos = env.dof_pos[0].clone()
    
    # Apply small position command
    target_pos = torch.zeros(1, env.num_dofs, device=env.device)
    target_pos[0, 0] = 0.1  # Small movement on first DOF
    
    print(f"  - Initial position DOF[0]: {initial_pos[0]:.4f}")
    print(f"  - Target position DOF[0]: {target_pos[0, 0]:.4f}")
    
    # Step with position control
    for step in range(50):  # 1 second at 50Hz
        env.robot.control_dofs_position(target_pos)
        env.scene.step()
        env._update_robot_state()
        
        if step % 10 == 0:
            current_pos = env.dof_pos[0, 0].item()
            error = abs(current_pos - target_pos[0, 0].item())
            print(f"    Step {step:2d}: pos={current_pos:.4f}, error={error:.4f}")
    
    final_pos = env.dof_pos[0, 0].item()
    final_error = abs(final_pos - target_pos[0, 0].item())
    
    print(f"  - Final position DOF[0]: {final_pos:.4f}")
    print(f"  - Final error: {final_error:.4f}")
    
    if final_error < 0.01:
        print("  ✅ Control response: GOOD")
    elif final_error < 0.05:
        print("  ⚠️ Control response: MODERATE")
    else:
        print("  ❌ Control response: POOR")
    
    return env

def compare_locomujoco_control():
    """Compare with LocoMujoco control parameters"""
    print(f"\nLocoMujoco Control Setup (for reference):")
    print(f"  - Simulation dt: 0.01s (100Hz)")
    print(f"  - Control frequency: 100Hz")
    print(f"  - PD gains: Varies by joint type")
    print(f"  - Control method: Direct motor torques (in SkeletonTorque)")
    
    print(f"\nKey Differences:")
    print(f"  1. Frequency: LocoMujoco 100Hz vs Genesis 50Hz")
    print(f"  2. Control: LocoMujoco uses torque control, Genesis position control")
    print(f"  3. PD gains: May need tuning for trajectory following")

def suggest_fixes():
    """Suggest potential fixes for tracking errors"""
    print(f"\n🔧 Suggested Fixes:")
    print(f"1. **Increase Control Frequency**:")
    print(f"   - Change dt=0.01 (100Hz) to match LocoMujoco")
    print(f"   - Higher frequency = better trajectory tracking")
    
    print(f"\n2. **Tune PD Gains**:")
    print(f"   - Lower kp gains for less aggressive control")
    print(f"   - Higher kv gains for better damping")
    print(f"   - Different gains for leg vs arm joints")
    
    print(f"\n3. **Trajectory Following Strategy**:")
    print(f"   - Use velocity feedforward from trajectory")
    print(f"   - Implement trajectory interpolation")
    print(f"   - Add root position tracking correction")
    
    print(f"\n4. **Joint Testing Improvements**:")
    print(f"   - Start from neutral pose, not mid-trajectory")
    print(f"   - Test smaller movements (0.05 rad instead of 0.2)")
    print(f"   - Test joints in stable order (spine first, then legs)")

if __name__ == "__main__":
    env = analyze_control_parameters()
    compare_locomujoco_control()
    suggest_fixes()
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Try increasing control frequency (dt=0.01)")
    print(f"2. Reduce PD gains for smoother control")
    print(f"3. Test individual joint control from neutral pose")
    print(f"4. Consider switching to torque control for imitation learning")