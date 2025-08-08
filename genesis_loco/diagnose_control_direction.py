"""
Control Direction Diagnostic Script

Systematically test why joints move in opposite directions to targets.
Tests joint axis directions, gravity effects, control methods, and joint coupling.
"""

import torch
import sys
import os
import time
import numpy as np

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
genesis_loco_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, genesis_loco_dir)

def diagnose_control_direction():
    """Comprehensive diagnosis of control direction issues"""
    print("🔍 Control Direction Diagnostic")
    print("=" * 60)
    
    # Setup Genesis
    import genesis as gs
    gs.init(backend=gs.gpu)
    from environments.skeleton_humanoid import SkeletonHumanoidEnv
    
    # Create environment with trajectory control
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=10.0,
        dt=0.01,
        show_viewer=True,  # Visual feedback
        use_trajectory_control=True
    )
    
    print(f"Environment Setup:")
    print(f"  - Total DOFs: {env.num_dofs}")
    print(f"  - Control frequency: {1/env.dt:.0f} Hz")
    print(f"  - Device: {env.device}")
    
    # Test sequence
    test_results = {}
    
    print("\n" + "="*60)
    print("TEST 1: Joint Identity Verification")
    test_results['joint_identity'] = test_joint_identity(env)
    
    print("\n" + "="*60)
    print("TEST 2: Bidirectional Control Test")
    test_results['bidirectional'] = test_bidirectional_control(env)
    
    print("\n" + "="*60) 
    print("TEST 3: Gravity Effect Analysis")
    test_results['gravity'] = test_gravity_effects(env)
    
    print("\n" + "="*60)
    print("TEST 4: Torque vs Position Control")
    test_results['control_methods'] = test_control_methods(env)
    
    print("\n" + "="*60)
    print("TEST 5: Joint Axis Direction Test")
    test_results['axis_direction'] = test_axis_directions(env)
    
    print("\n" + "="*60)
    print("TEST 6: Multiple Joint Test")
    test_results['multiple_joints'] = test_multiple_joints(env)
    
    # Final analysis
    analyze_results(test_results)
    
    return test_results

def test_joint_identity(env):
    """Verify which joint we're actually testing"""
    print("Testing joint identity mapping...")
    
    results = {}
    
    # Test the first few controllable DOFs
    test_dofs = [0, 1, 2, 6, 11]  # Mix of different joint types
    
    for dof_idx in test_dofs:
        if dof_idx < env.num_dofs:
            # Get joint name from DOF index
            try:
                dof_name = env.dof_names[dof_idx] if hasattr(env, 'dof_names') else f"DOF_{dof_idx}"
                print(f"\nTesting DOF {dof_idx}: {dof_name}")
                
                # Find corresponding Genesis joint
                joint_info = None
                for joint in env.robot.joints:
                    if hasattr(joint, 'dofs_idx_local') and dof_idx in joint.dofs_idx_local:
                        joint_info = {
                            'name': joint.name,
                            'type': joint.type,
                            'axis': getattr(joint, 'axis', 'unknown'),
                            'dof_start': getattr(joint, 'dof_start', 'unknown'),
                            'dofs_idx_local': joint.dofs_idx_local
                        }
                        break
                
                print(f"  Genesis joint info: {joint_info}")
                
                results[dof_idx] = {
                    'dof_name': dof_name,
                    'joint_info': joint_info
                }
                
            except Exception as e:
                print(f"  Error getting info for DOF {dof_idx}: {e}")
                results[dof_idx] = {'error': str(e)}
    
    return results

def test_bidirectional_control(env):
    """Test both positive and negative position targets"""
    print("Testing bidirectional position control...")
    
    results = {}
    
    # Reset to neutral position
    obs, _ = env.reset()
    stabilize_robot(env)
    
    test_dof = 0  # Test first controllable DOF
    initial_pos = env.dof_pos[0, test_dof].item()
    
    print(f"Testing DOF {test_dof}")
    print(f"Initial position: {initial_pos:.4f}")
    
    # Test positive target
    print(f"\n--- Testing POSITIVE target (+0.1) ---")
    target_pos = initial_pos + 0.1
    pos_result = apply_position_target(env, test_dof, target_pos)
    results['positive'] = pos_result
    
    # Reset robot
    stabilize_robot(env)
    
    # Test negative target  
    print(f"\n--- Testing NEGATIVE target (-0.1) ---")
    target_pos = initial_pos - 0.1
    neg_result = apply_position_target(env, test_dof, target_pos)
    results['negative'] = neg_result
    
    # Analysis
    if pos_result and neg_result:
        pos_direction = np.sign(pos_result['final_pos'] - pos_result['initial_pos'])
        neg_direction = np.sign(neg_result['final_pos'] - neg_result['initial_pos'])
        
        print(f"\nBidirectional Analysis:")
        print(f"  Positive target → direction: {pos_direction:+.0f}")
        print(f"  Negative target → direction: {neg_direction:+.0f}")
        
        if pos_direction > 0 and neg_direction < 0:
            print("  ✅ Correct: Both directions work as expected")
        elif pos_direction < 0 and neg_direction > 0:
            print("  ❌ INVERTED: Both directions are reversed")
        else:
            print("  ⚠️ INCONSISTENT: Direction behavior is inconsistent")
    
    return results

def test_gravity_effects(env):
    """Test control with and without gravity"""
    print("Testing gravity effects on control...")
    
    results = {}
    
    # Test with gravity (current state)
    print("\n--- Testing WITH gravity ---")
    gravity_result = test_single_position_control(env, "with_gravity")
    results['with_gravity'] = gravity_result
    
    # Test without gravity
    print("\n--- Testing WITHOUT gravity ---")
    try:
        # Temporarily disable gravity
        original_gravity = env.scene._sim_options.gravity
        env.scene._sim_options.gravity = (0.0, 0.0, 0.0)
        env.scene._rigid_sim_config.gravity = torch.tensor([0.0, 0.0, 0.0], device=env.device)
        
        # Reset and test
        obs, _ = env.reset()
        stabilize_robot(env, steps=10)  # Less stabilization needed without gravity
        
        no_gravity_result = test_single_position_control(env, "no_gravity")
        results['no_gravity'] = no_gravity_result
        
        # Restore gravity
        env.scene._sim_options.gravity = original_gravity
        
    except Exception as e:
        print(f"  Error testing without gravity: {e}")
        results['no_gravity'] = {'error': str(e)}
    
    # Compare results
    if 'with_gravity' in results and 'no_gravity' in results:
        compare_gravity_results(results['with_gravity'], results['no_gravity'])
    
    return results

def test_control_methods(env):
    """Compare torque control vs position control"""
    print("Testing different control methods...")
    
    results = {}
    
    # Reset
    obs, _ = env.reset()
    stabilize_robot(env)
    
    test_dof = 0
    initial_pos = env.dof_pos[0, test_dof].item()
    
    print(f"Testing DOF {test_dof}, Initial pos: {initial_pos:.4f}")
    
    # Test 1: Position control
    print(f"\n--- Testing POSITION control ---")
    target_pos = initial_pos + 0.1
    pos_result = test_position_control_method(env, test_dof, target_pos)
    results['position'] = pos_result
    
    # Reset
    stabilize_robot(env)
    
    # Test 2: Torque control
    print(f"\n--- Testing TORQUE control ---")
    torque_result = test_torque_control_method(env, test_dof, 50.0)  # 50 N⋅m
    results['torque'] = torque_result
    
    return results

def test_axis_directions(env):
    """Test if joint axes are correctly oriented"""
    print("Testing joint axis directions...")
    
    results = {}
    
    # Test specific joints with known axes
    test_joints = {
        'hip_flexion_r': {'expected_axis': 'Z', 'test_torque': 100.0},
        'lumbar_bending': {'expected_axis': 'X', 'test_torque': 50.0},
        'knee_angle_r': {'expected_axis': 'Z', 'test_torque': 100.0}
    }
    
    for joint_name, config in test_joints.items():
        if joint_name in [j.name for j in env.robot.joints]:
            print(f"\n--- Testing {joint_name} ---")
            
            try:
                joint_obj = env.robot.get_joint(joint_name)
                dof_idx = joint_obj.dofs_idx_local[0] if joint_obj.dofs_idx_local else None
                
                if dof_idx is not None:
                    print(f"  DOF index: {dof_idx}")
                    print(f"  Expected axis: {config['expected_axis']}")
                    
                    # Reset and test
                    obs, _ = env.reset()
                    stabilize_robot(env)
                    
                    result = test_joint_axis_response(env, joint_name, dof_idx, config['test_torque'])
                    results[joint_name] = result
                
            except Exception as e:
                print(f"  Error testing {joint_name}: {e}")
                results[joint_name] = {'error': str(e)}
    
    return results

def test_multiple_joints(env):
    """Test multiple joints simultaneously to check for coupling"""
    print("Testing multiple joint interactions...")
    
    results = {}
    
    # Reset
    obs, _ = env.reset()
    stabilize_robot(env)
    
    # Record initial positions
    initial_positions = env.dof_pos[0].clone()
    
    # Apply position targets to multiple joints
    target_positions = initial_positions.clone()
    
    # Modify a few key joints
    test_joints = [0, 1, 2]  # First few DOFs
    for i, dof_idx in enumerate(test_joints):
        if dof_idx < env.num_dofs:
            target_positions[dof_idx] += 0.1 * ((-1) ** i)  # Alternating signs
    
    print(f"Testing joints: {test_joints}")
    print(f"Initial positions: {[initial_positions[i].item() for i in test_joints]}")
    print(f"Target positions: {[target_positions[i].item() for i in test_joints]}")
    
    # Apply control
    target_tensor = target_positions.unsqueeze(0)
    
    positions_over_time = []
    for step in range(100):
        env.robot.control_dofs_position(target_tensor)
        env.scene.step()
        env._update_robot_state()
        
        if step % 20 == 0:
            current_positions = [env.dof_pos[0, i].item() for i in test_joints]
            positions_over_time.append(current_positions)
            print(f"  Step {step:2d}: {[f'{pos:+.4f}' for pos in current_positions]}")
    
    final_positions = [env.dof_pos[0, i].item() for i in test_joints]
    
    results = {
        'test_joints': test_joints,
        'initial_positions': [initial_positions[i].item() for i in test_joints],
        'target_positions': [target_positions[i].item() for i in test_joints],
        'final_positions': final_positions,
        'positions_over_time': positions_over_time
    }
    
    return results

# Helper functions

def stabilize_robot(env, steps=30):
    """Stabilize robot in neutral position"""
    for _ in range(steps):
        env.step(torch.zeros(1, env.num_actions, device=env.device))

def apply_position_target(env, dof_idx, target_pos):
    """Apply a position target and track response"""
    initial_pos = env.dof_pos[0, dof_idx].item()
    
    # Create target tensor
    target_positions = torch.zeros(1, env.num_dofs, device=env.device)
    target_positions[0, dof_idx] = target_pos
    
    print(f"  Target: {target_pos:.4f}")
    
    # Track response
    positions = [initial_pos]
    for step in range(50):
        env.robot.control_dofs_position(target_positions)
        env.scene.step()
        env._update_robot_state()
        
        current_pos = env.dof_pos[0, dof_idx].item()
        positions.append(current_pos)
        
        if step % 10 == 0:
            error = abs(current_pos - target_pos)
            direction = "↑" if current_pos > initial_pos else "↓" if current_pos < initial_pos else "→"
            print(f"    Step {step:2d}: {current_pos:+.4f} {direction} (error: {error:.4f})")
    
    final_pos = positions[-1]
    final_error = abs(final_pos - target_pos)
    direction_correct = np.sign(final_pos - initial_pos) == np.sign(target_pos - initial_pos)
    
    print(f"  Result: {final_pos:+.4f} (error: {final_error:.4f})")
    print(f"  Direction correct: {direction_correct}")
    
    return {
        'initial_pos': initial_pos,
        'target_pos': target_pos,
        'final_pos': final_pos,
        'final_error': final_error,
        'direction_correct': direction_correct,
        'positions': positions
    }

def test_single_position_control(env, test_name):
    """Single position control test"""
    obs, _ = env.reset()
    stabilize_robot(env)
    
    test_dof = 0
    initial_pos = env.dof_pos[0, test_dof].item()
    target_pos = initial_pos + 0.1
    
    return apply_position_target(env, test_dof, target_pos)

def test_position_control_method(env, dof_idx, target_pos):
    """Test position control method specifically"""
    initial_pos = env.dof_pos[0, dof_idx].item()
    
    target_positions = torch.zeros(1, env.num_dofs, device=env.device)
    target_positions[0, dof_idx] = target_pos
    
    for step in range(50):
        env.robot.control_dofs_position(target_positions)
        env.scene.step()
        env._update_robot_state()
        
        if step % 10 == 0:
            current_pos = env.dof_pos[0, dof_idx].item()
            print(f"    Step {step:2d}: {current_pos:+.4f}")
    
    final_pos = env.dof_pos[0, dof_idx].item()
    direction_correct = np.sign(final_pos - initial_pos) == np.sign(target_pos - initial_pos)
    
    return {
        'method': 'position',
        'initial_pos': initial_pos,
        'target_pos': target_pos,
        'final_pos': final_pos,
        'direction_correct': direction_correct
    }

def test_torque_control_method(env, dof_idx, torque):
    """Test torque control method"""
    initial_pos = env.dof_pos[0, dof_idx].item()
    
    torques = torch.zeros(1, env.num_dofs, device=env.device)
    torques[0, dof_idx] = torque
    
    for step in range(50):
        env.robot.control_dofs_force(torques)
        env.scene.step()
        env._update_robot_state()
        
        if step % 10 == 0:
            current_pos = env.dof_pos[0, dof_idx].item()
            print(f"    Step {step:2d}: {current_pos:+.4f}")
    
    final_pos = env.dof_pos[0, dof_idx].item()
    movement_direction = np.sign(final_pos - initial_pos)
    torque_direction = np.sign(torque)
    direction_correct = movement_direction == torque_direction
    
    return {
        'method': 'torque',
        'initial_pos': initial_pos,
        'applied_torque': torque,
        'final_pos': final_pos,
        'movement_direction': movement_direction,
        'torque_direction': torque_direction,
        'direction_correct': direction_correct
    }

def test_joint_axis_response(env, joint_name, dof_idx, test_torque):
    """Test joint axis response to torque"""
    initial_pos = env.dof_pos[0, dof_idx].item()
    
    torques = torch.zeros(1, env.num_dofs, device=env.device)
    torques[0, dof_idx] = test_torque
    
    print(f"  Applying torque: {test_torque:+.1f} N⋅m")
    
    for step in range(30):
        env.robot.control_dofs_force(torques)
        env.scene.step()
        env._update_robot_state()
    
    final_pos = env.dof_pos[0, dof_idx].item()
    movement = final_pos - initial_pos
    
    print(f"  Movement: {movement:+.4f} rad ({np.degrees(movement):+.2f}°)")
    
    # Test opposite torque
    torques[0, dof_idx] = -test_torque
    
    for step in range(30):
        env.robot.control_dofs_force(torques)
        env.scene.step()
        env._update_robot_state()
    
    opposite_pos = env.dof_pos[0, dof_idx].item()
    opposite_movement = opposite_pos - final_pos
    
    print(f"  Opposite movement: {opposite_movement:+.4f} rad ({np.degrees(opposite_movement):+.2f}°)")
    
    return {
        'joint_name': joint_name,
        'positive_torque': test_torque,
        'positive_movement': movement,
        'negative_torque': -test_torque,
        'negative_movement': opposite_movement,
        'consistent': np.sign(movement) == -np.sign(opposite_movement)
    }

def compare_gravity_results(with_gravity, no_gravity):
    """Compare results with and without gravity"""
    print(f"\nGravity Effect Analysis:")
    
    if 'error' in with_gravity or 'error' in no_gravity:
        print("  Cannot compare due to errors")
        return
    
    gravity_movement = with_gravity['final_pos'] - with_gravity['initial_pos']
    no_gravity_movement = no_gravity['final_pos'] - no_gravity['initial_pos']
    
    print(f"  With gravity movement: {gravity_movement:+.4f}")
    print(f"  No gravity movement: {no_gravity_movement:+.4f}")
    
    if abs(no_gravity_movement) > abs(gravity_movement):
        print("  ✅ Control works better without gravity")
    elif abs(gravity_movement) > abs(no_gravity_movement):
        print("  ⚠️ Gravity enhances movement (unusual)")
    else:
        print("  ⚠️ Similar response with/without gravity")

def analyze_results(test_results):
    """Analyze all test results and provide diagnosis"""
    print("COMPREHENSIVE DIAGNOSIS")
    print("=" * 60)
    
    issues = []
    
    # Check bidirectional test
    if 'bidirectional' in test_results:
        bid_results = test_results['bidirectional']
        if 'positive' in bid_results and 'negative' in bid_results:
            pos_correct = bid_results['positive'].get('direction_correct', False)
            neg_correct = bid_results['negative'].get('direction_correct', False)
            
            if not pos_correct and not neg_correct:
                issues.append("❌ JOINT AXIS INVERSION: Both directions reversed")
            elif not pos_correct or not neg_correct:
                issues.append("⚠️ INCONSISTENT DIRECTION: One direction works, other doesn't")
    
    # Check control methods
    if 'control_methods' in test_results:
        methods = test_results['control_methods']
        if 'position' in methods and 'torque' in methods:
            pos_correct = methods['position'].get('direction_correct', False)
            torque_correct = methods['torque'].get('direction_correct', False)
            
            if not pos_correct and not torque_correct:
                issues.append("❌ FUNDAMENTAL ISSUE: Both position and torque control wrong")
            elif pos_correct and not torque_correct:
                issues.append("❌ TORQUE CONTROL ISSUE: Position works, torque doesn't")
            elif not pos_correct and torque_correct:
                issues.append("❌ POSITION CONTROL ISSUE: Torque works, position doesn't")
    
    # Check gravity effects
    if 'gravity' in test_results:
        grav_results = test_results['gravity']
        if 'no_gravity' in grav_results and 'error' not in grav_results['no_gravity']:
            no_grav_correct = grav_results['no_gravity'].get('direction_correct', False)
            if no_grav_correct:
                issues.append("✅ GRAVITY INTERFERENCE: Control works without gravity")
    
    # Print diagnosis
    print("IDENTIFIED ISSUES:")
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  No clear issues identified from tests")
    
    print("\nRECOMMENDED FIXES:")
    
    if any("JOINT AXIS INVERSION" in issue for issue in issues):
        print("  1. Fix XML joint axes - multiply by -1 or rotate 180°")
        print("  2. Or invert control commands in software")
    
    if any("GRAVITY INTERFERENCE" in issue for issue in issues):
        print("  1. Increase PD gains for stronger control")
        print("  2. Start from stable poses (not mid-air)")
        print("  3. Add gravity compensation")
    
    if any("TORQUE CONTROL" in issue for issue in issues):
        print("  1. Check torque limits and signs")
        print("  2. Verify joint motor specifications")
    
    if any("POSITION CONTROL" in issue for issue in issues):
        print("  1. Tune PD gains (lower kp, higher kv)")
        print("  2. Check position target ranges vs joint limits")
    
    print("\nNEXT STEPS:")
    print("  1. If joint axis inversion confirmed, fix XML or software")
    print("  2. Test with individual joints in isolation")
    print("  3. Verify against Genesis examples (Franka robot)")

if __name__ == "__main__":
    try:
        print("Starting comprehensive control direction diagnosis...")
        results = diagnose_control_direction()
        
        print(f"\n🎯 Diagnosis completed!")
        print("Review the analysis above to identify the root cause.")
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()