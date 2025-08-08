"""
Fixed Control Direction Diagnostic Script

FIXED: Uses correct Genesis DOF indexing instead of LocoMujoco indexing
- Uses SkeletonHumanoidEnv's motor detection to get correct controllable DOFs
- Tests actual controllable joint DOFs instead of root DOFs
- Matches action-to-DOF mapping used by the environment
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

def diagnose_control_direction_fixed():
    """Comprehensive diagnosis using CORRECT Genesis DOF indexing"""
    print("🔍 FIXED Control Direction Diagnostic")
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
    print(f"  - Controllable actions: {env.num_skeleton_actions}")
    print(f"  - Control frequency: {1/env.dt:.0f} Hz")
    print(f"  - Device: {env.device}")
    
    # Get controllable DOF mapping from environment (this is the KEY fix!)
    controllable_dof_indices = []
    action_names = []
    
    print(f"\nControllable DOF Mapping (from SkeletonHumanoidEnv):")
    for action_name in env.action_spec:
        dof_idx = env.action_to_joint_idx[action_name]
        controllable_dof_indices.append(dof_idx)
        action_names.append(action_name)
        print(f"  Action: {action_name} -> DOF {dof_idx}")
    
    # Test sequence using CORRECT DOFs
    test_results = {}
    
    print("\n" + "="*60)
    print("TEST 1: FIXED Joint Identity Verification")
    test_results['joint_identity'] = test_joint_identity_fixed(env, controllable_dof_indices, action_names)
    
    print("\n" + "="*60)
    print("TEST 2: FIXED Bidirectional Control Test")
    # Test the first controllable joint (not root!)
    first_controllable_dof = controllable_dof_indices[0]
    first_action_name = action_names[0]
    test_results['bidirectional'] = test_bidirectional_control_fixed(env, first_controllable_dof, first_action_name)
    
    print("\n" + "="*60) 
    print("TEST 3: FIXED Control Methods Comparison")
    test_results['control_methods'] = test_control_methods_fixed(env, first_controllable_dof, first_action_name)
    
    print("\n" + "="*60)
    print("TEST 4: FIXED Joint Axis Direction Test")
    test_results['axis_direction'] = test_axis_directions_fixed(env, controllable_dof_indices, action_names)
    
    print("\n" + "="*60)
    print("TEST 5: FIXED Multiple Joint Test")
    # Test first 3 controllable joints (not root DOFs!)
    test_dofs = controllable_dof_indices[:3]
    test_actions = action_names[:3]
    test_results['multiple_joints'] = test_multiple_joints_fixed(env, test_dofs, test_actions)
    
    # Final analysis
    analyze_results_fixed(test_results)
    
    return test_results

def test_joint_identity_fixed(env, controllable_dof_indices, action_names):
    """Verify joint identity using CORRECT Genesis DOF mapping"""
    print("Testing joint identity mapping...")
    
    results = {}
    
    # Test key controllable DOFs (not root DOFs!)
    sample_dofs = controllable_dof_indices[:5]  # First 5 controllable DOFs
    sample_actions = action_names[:5]
    
    for dof_idx, action_name in zip(sample_dofs, sample_actions):
        print(f"\nTesting DOF {dof_idx}: {action_name}")
        
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
        
        if joint_info:
            print(f"  Genesis joint info: {joint_info}")
            results[f"DOF_{dof_idx}"] = {
                'action_name': action_name,
                'joint_info': joint_info,
                'status': 'found'
            }
        else:
            print(f"  ERROR: No joint found for DOF {dof_idx}")
            results[f"DOF_{dof_idx}"] = {
                'action_name': action_name,
                'status': 'not_found'
            }
    
    return results

def test_bidirectional_control_fixed(env, test_dof_idx, action_name):
    """Test bidirectional control on CONTROLLABLE joint"""
    print(f"Testing bidirectional position control...")
    print(f"Testing DOF {test_dof_idx} ({action_name})")
    
    results = {}
    
    # Reset environment
    obs, _ = env.reset()
    stabilize_robot(env)
    
    initial_pos = env.dof_pos[0, test_dof_idx].item()
    print(f"Initial position: {initial_pos:.4f}")
    
    # Test positive target
    print(f"\n--- Testing POSITIVE target (+0.1) ---")
    target_pos = initial_pos + 0.1
    print(f"  Target: {target_pos:.4f}")
    
    pos_result = test_position_control_single_dof(env, test_dof_idx, target_pos)
    direction_correct_pos = (pos_result['final_pos'] - initial_pos) * (target_pos - initial_pos) > 0
    
    print(f"  Result: {pos_result['final_pos']:.4f} (error: {abs(pos_result['final_pos'] - target_pos):.4f})")
    print(f"  Direction correct: {direction_correct_pos}")
    
    # Reset
    stabilize_robot(env)
    
    # Test negative target  
    print(f"\n--- Testing NEGATIVE target (-0.1) ---")
    target_pos = initial_pos - 0.1
    print(f"  Target: {target_pos:.4f}")
    
    neg_result = test_position_control_single_dof(env, test_dof_idx, target_pos)
    direction_correct_neg = (neg_result['final_pos'] - initial_pos) * (target_pos - initial_pos) > 0
    
    print(f"  Result: {neg_result['final_pos']:.4f} (error: {abs(neg_result['final_pos'] - target_pos):.4f})")
    print(f"  Direction correct: {direction_correct_neg}")
    
    # Analysis
    print(f"\nBidirectional Analysis:")
    pos_direction = +1 if pos_result['final_pos'] > initial_pos else -1
    neg_direction = +1 if neg_result['final_pos'] > initial_pos else -1
    
    print(f"  Positive target → direction: {pos_direction:+d}")
    print(f"  Negative target → direction: {neg_direction:+d}")
    
    if direction_correct_pos and direction_correct_neg:
        print(f"  ✅ CONSISTENT: Both directions work correctly")
        status = "consistent"
    elif direction_correct_pos or direction_correct_neg:
        print(f"  ⚠️ INCONSISTENT: One direction works, other doesn't")
        status = "inconsistent"  
    else:
        print(f"  ❌ BROKEN: Neither direction works")
        status = "broken"
    
    return {
        'dof_idx': test_dof_idx,
        'action_name': action_name,
        'initial_pos': initial_pos,
        'positive_test': pos_result,
        'negative_test': neg_result,
        'direction_correct_pos': direction_correct_pos,
        'direction_correct_neg': direction_correct_neg,
        'status': status
    }

def test_position_control_single_dof(env, dof_idx, target_pos, steps=50):
    """Test position control on single DOF"""
    trajectory = []
    
    for step in range(steps):
        # Create target positions tensor (all DOFs) 
        target_positions = env.dof_pos[0].clone()
        target_positions[dof_idx] = target_pos
        
        # Apply position control to ALL DOFs (Genesis expects this)
        env.robot.control_dofs_position(target_positions.unsqueeze(0))
        env.scene.step()
        # env._update_buffers()
        
        current_pos = env.dof_pos[0, dof_idx].item()
        error = abs(current_pos - target_pos)
        
        trajectory.append(current_pos)
        
        # Print progress
        if step % 10 == 0 or step == steps-1:
            direction = "↑" if current_pos > trajectory[0] else "↓"
            print(f"    Step {step:2d}: {current_pos:+.4f} {direction} (error: {error:.4f})")
    
    return {
        'final_pos': trajectory[-1], 
        'trajectory': trajectory,
        'target': target_pos,
        'error': abs(trajectory[-1] - target_pos)
    }

def test_control_methods_fixed(env, test_dof_idx, action_name):
    """Compare position vs torque control on controllable joint"""
    print(f"Testing different control methods...")
    print(f"Testing DOF {test_dof_idx} ({action_name})")
    
    results = {}
    
    # Reset
    obs, _ = env.reset()
    stabilize_robot(env)
    
    initial_pos = env.dof_pos[0, test_dof_idx].item()
    print(f"Initial pos: {initial_pos:.4f}")
    
    # Test 1: Position control
    print(f"\n--- Testing POSITION control ---")
    target_pos = initial_pos + 0.1
    pos_result = test_position_control_single_dof(env, test_dof_idx, target_pos)
    results['position'] = pos_result
    
    # Reset
    stabilize_robot(env)
    
    # Test 2: Torque control  
    print(f"\n--- Testing TORQUE control ---")
    torque_result = test_torque_control_single_dof(env, test_dof_idx, 50.0)  # 50 N⋅m
    results['torque'] = torque_result
    
    return results

def test_torque_control_single_dof(env, dof_idx, torque_value, steps=50):
    """Test torque control on single DOF"""
    trajectory = []
    
    for step in range(steps):
        # Create torque tensor for all controllable DOFs
        torques = torch.zeros((env.num_envs, env.num_skeleton_actions), device=env.device)
        
        # Find which action index corresponds to this DOF
        action_idx = None
        for i, action_name in enumerate(env.action_spec):
            if env.action_to_joint_idx[action_name] == dof_idx:
                action_idx = i
                break
        
        if action_idx is not None:
            torques[0, action_idx] = torque_value
            
            # Apply torques using environment's method (this handles DOF mapping correctly)
            env._apply_actions(torques)
            env.scene.step()
            # env._update_buffers()
        
        current_pos = env.dof_pos[0, dof_idx].item()
        trajectory.append(current_pos)
        
        # Print progress
        if step % 10 == 0 or step == steps-1:
            print(f"    Step {step:2d}: {current_pos:+.4f}")
    
    return {
        'final_pos': trajectory[-1],
        'trajectory': trajectory,
        'torque': torque_value
    }

def test_axis_directions_fixed(env, controllable_dof_indices, action_names):
    """Test joint axis directions using controllable joints"""
    print("Testing joint axis directions...")
    
    results = {}
    
    # Test key joints with known expected axes
    test_joints = {
        'hip_flexion_r': {'expected_axis': 'Z', 'test_torque': 100.0},
        'lumbar_bending': {'expected_axis': 'X', 'test_torque': 50.0},
        'knee_angle_r': {'expected_axis': 'Z', 'test_torque': 100.0}
    }
    
    # Find these joints in our controllable DOFs
    for joint_name, config in test_joints.items():
        action_name = None
        dof_idx = None
        
        # Find matching action
        for i, act_name in enumerate(action_names):
            if joint_name in act_name or act_name.replace('mot_', '') == joint_name:
                action_name = act_name
                dof_idx = controllable_dof_indices[i]
                break
        
        if dof_idx is not None:
            print(f"\n--- Testing {joint_name} ---")
            print(f"  DOF index: {dof_idx}")
            print(f"  Expected axis: {config['expected_axis']}")
            
            # Reset and test
            obs, _ = env.reset()
            stabilize_robot(env)
            
            result = test_joint_axis_response(env, action_name, dof_idx, config['test_torque'])
            results[joint_name] = result
        else:
            print(f"\n--- {joint_name} not found in controllable joints ---")
    
    return results

def test_joint_axis_response(env, action_name, dof_idx, torque_magnitude):
    """Test joint response to positive/negative torques"""
    
    # Test positive torque
    obs, _ = env.reset()
    stabilize_robot(env)
    
    initial_pos = env.dof_pos[0, dof_idx].item()
    
    print(f"  Applying torque: +{torque_magnitude} N⋅m")
    positive_result = test_torque_control_single_dof(env, dof_idx, torque_magnitude, steps=30)
    positive_movement = positive_result['final_pos'] - initial_pos
    
    print(f"  Movement: {positive_movement:+.4f} rad ({np.degrees(positive_movement):+.2f}°)")
    
    # Test negative torque
    stabilize_robot(env)
    negative_result = test_torque_control_single_dof(env, dof_idx, -torque_magnitude, steps=30)
    negative_movement = negative_result['final_pos'] - initial_pos
    
    print(f"  Opposite movement: {negative_movement:+.4f} rad ({np.degrees(negative_movement):+.2f}°)")
    
    return {
        'dof_idx': dof_idx,
        'action_name': action_name,
        'positive_torque': torque_magnitude,
        'positive_movement': positive_movement,
        'negative_movement': negative_movement,
        'consistent': (positive_movement * negative_movement) < 0  # Should be opposite signs
    }

def test_multiple_joints_fixed(env, test_dofs, test_actions):
    """Test multiple controllable joints simultaneously"""
    print("Testing multiple joint interactions...")
    print(f"Testing joints: {test_dofs}")
    
    results = {}
    
    # Reset
    obs, _ = env.reset()
    stabilize_robot(env)
    
    # Record initial positions
    initial_positions = [env.dof_pos[0, dof_idx].item() for dof_idx in test_dofs]
    
    # Set target positions
    target_positions = env.dof_pos[0].clone()
    targets = []
    
    for i, dof_idx in enumerate(test_dofs):
        target_pos = initial_positions[i] + 0.1 * ((-1) ** i)  # Alternating signs
        target_positions[dof_idx] = target_pos
        targets.append(target_pos)
    
    print(f"Initial positions: {[f'{pos:.4f}' for pos in initial_positions]}")
    print(f"Target positions: {[f'{pos:.4f}' for pos in targets]}")
    
    # Apply control for multiple steps
    trajectory = []
    for step in range(0, 100, 20):
        env.robot.control_dofs_position(target_positions.unsqueeze(0))
        
        for _ in range(20):
            env.scene.step()
            # env._update_buffers()
        
        current_positions = [env.dof_pos[0, dof_idx].item() for dof_idx in test_dofs]
        trajectory.append(current_positions)
        
        print(f"  Step {step:2d}: {[f'{pos:+.4f}' for pos in current_positions]}")
    
    return {
        'test_dofs': test_dofs,
        'test_actions': test_actions,
        'initial_positions': initial_positions,
        'target_positions': targets,
        'trajectory': trajectory
    }

def stabilize_robot(env, steps=10):
    """Let robot settle to stable position"""
    for _ in range(steps):
        env.scene.step()
        # env._update_buffers()

def analyze_results_fixed(results):
    """Analyze diagnostic results"""
    print("COMPREHENSIVE DIAGNOSIS")
    print("="*60)
    
    issues = []
    
    # Check bidirectional control
    if 'bidirectional' in results:
        bid_result = results['bidirectional']
        if bid_result['status'] == 'inconsistent':
            issues.append("⚠️ INCONSISTENT DIRECTION: One direction works, other doesn't")
        elif bid_result['status'] == 'broken':
            issues.append("❌ BROKEN CONTROL: Neither direction works")
        else:
            print("✅ BIDIRECTIONAL CONTROL: Working correctly")
    
    # Check joint identity
    if 'joint_identity' in results:
        identity_results = results['joint_identity']
        missing_joints = [key for key, value in identity_results.items() if value['status'] == 'not_found']
        if missing_joints:
            issues.append(f"❌ MISSING JOINTS: {len(missing_joints)} joints not found")
    
    # Report issues
    if issues:
        print("IDENTIFIED ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        
        print(f"\nRECOMMENDED FIXES:")
        print(f"  1. ✅ DOF indexing is now FIXED - using controllable DOFs only")
        print(f"  2. Check for joint axis inversions in XML file")  
        print(f"  3. Verify PD gain settings are appropriate")
    else:
        print("✅ NO MAJOR ISSUES DETECTED")
        print("  Control system appears to be working correctly")
    
    print(f"\n🎯 Fixed diagnosis completed!")
    print(f"Review the analysis above to identify any remaining issues.")

if __name__ == "__main__":
    diagnose_control_direction_fixed()