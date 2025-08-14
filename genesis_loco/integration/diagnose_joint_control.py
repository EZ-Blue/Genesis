#!/usr/bin/env python3
"""
Joint Control Diagnostic Script

Tests each joint individually with position control to verify:
1. Joint responds to commands
2. Joint moves in expected direction
3. Joint reaches target positions
4. Control gains are appropriate
"""

import torch
import numpy as np
import sys
import os
import time

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.skeleton_humanoid import SkeletonHumanoidEnv
import genesis as gs


def safe_init_genesis():
    """Safely initialize Genesis"""
    try:
        gs.init(backend=gs.gpu)
        return True, "Genesis initialized"
    except Exception as e:
        if "already initialized" in str(e):
            return True, "Genesis already initialized"
        else:
            return False, f"Genesis initialization failed: {e}"


def test_individual_joint_control(env: SkeletonHumanoidEnv, joint_idx: int, joint_name: str, 
                                test_amplitude: float = 0.5, test_duration: int = 100):
    """Test control of a single joint"""
    
    print(f"\n--- Testing Joint {joint_idx}: {joint_name} ---")
    
    # Reset environment
    env.reset()
    initial_pos = env.dof_pos[0, joint_idx].item()
    print(f"Initial position: {initial_pos:.4f}")
    
    # Create test trajectory: sine wave
    positions_recorded = []
    commands_sent = []
    
    for step in range(test_duration):
        # Generate sinusoidal command
        t = step / test_duration * 2 * np.pi  # One full cycle
        target_pos = initial_pos + test_amplitude * np.sin(t)
        
        # Create action vector (all zeros except target joint)
        action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        action[0, joint_idx] = target_pos
        
        # Step environment
        obs, rewards, dones, info = env.step(action)
        
        # Record actual position
        actual_pos = env.dof_pos[0, joint_idx].item()
        positions_recorded.append(actual_pos)
        commands_sent.append(target_pos)
    
    # Analyze results
    positions_recorded = np.array(positions_recorded)
    commands_sent = np.array(commands_sent)
    
    # Calculate metrics
    position_range = positions_recorded.max() - positions_recorded.min()
    command_range = commands_sent.max() - commands_sent.min()
    final_error = abs(positions_recorded[-1] - commands_sent[-1])
    mean_tracking_error = np.mean(np.abs(positions_recorded - commands_sent))
    
    # Responsiveness check
    response_threshold = 0.1 * command_range
    is_responsive = position_range > response_threshold
    
    print(f"Command range: [{commands_sent.min():.4f}, {commands_sent.max():.4f}] = {command_range:.4f}")
    print(f"Actual range: [{positions_recorded.min():.4f}, {positions_recorded.max():.4f}] = {position_range:.4f}")
    print(f"Mean tracking error: {mean_tracking_error:.4f}")
    print(f"Final error: {final_error:.4f}")
    print(f"Responsive: {'✅' if is_responsive else '❌'} (moved {position_range:.4f})")
    
    # Determine status
    if not is_responsive:
        status = "UNRESPONSIVE"
    elif mean_tracking_error > 0.2:
        status = "POOR_TRACKING"
    elif final_error > 0.1:
        status = "SLOW_RESPONSE"
    else:
        status = "GOOD"
    
    print(f"Status: {status}")
    
    return {
        'joint_idx': joint_idx,
        'joint_name': joint_name,
        'initial_pos': initial_pos,
        'command_range': command_range,
        'position_range': position_range,
        'mean_tracking_error': mean_tracking_error,
        'final_error': final_error,
        'is_responsive': is_responsive,
        'status': status,
        'positions': positions_recorded,
        'commands': commands_sent
    }


def test_all_joints_control(env: SkeletonHumanoidEnv):
    """Test control of all joints systematically"""
    
    print("🔧 JOINT CONTROL DIAGNOSTIC")
    print("=" * 60)
    
    print(f"Environment info:")
    print(f"  Total DOFs: {env.robot.n_dofs}")
    print(f"  Controlled DOFs: {len(env.motors_dof_idx)}")
    print(f"  Action dimension: {env.num_actions}")
    print(f"  Joint names: {len(env.joint_names)}")
    
    # Test each controlled joint
    joint_results = []
    
    # Convert joint_names set to list for indexing
    joint_names_list = list(env.joint_names)
    
    for i, joint_idx in enumerate(env.motors_dof_idx):
        joint_name = joint_names_list[i] if i < len(joint_names_list) else f"joint_{i}"
        
        result = test_individual_joint_control(env, i, joint_name, 
                                             test_amplitude=0.3, test_duration=50)
        joint_results.append(result)
    
    return joint_results


def analyze_control_results(results):
    """Analyze and summarize joint control test results"""
    
    print(f"\n📊 CONTROL ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Categorize results
    good_joints = [r for r in results if r['status'] == 'GOOD']
    slow_joints = [r for r in results if r['status'] == 'SLOW_RESPONSE']
    poor_tracking = [r for r in results if r['status'] == 'POOR_TRACKING']
    unresponsive = [r for r in results if r['status'] == 'UNRESPONSIVE']
    
    print(f"Joint Status Summary:")
    print(f"  ✅ Good control: {len(good_joints)}/{len(results)} joints")
    print(f"  🐌 Slow response: {len(slow_joints)}/{len(results)} joints")
    print(f"  📉 Poor tracking: {len(poor_tracking)}/{len(results)} joints")
    print(f"  ❌ Unresponsive: {len(unresponsive)}/{len(results)} joints")
    
    if unresponsive:
        print(f"\n⚠️  UNRESPONSIVE JOINTS:")
        for joint in unresponsive:
            print(f"    - {joint['joint_name']} (idx {joint['joint_idx']})")
    
    if poor_tracking:
        print(f"\n⚠️  POOR TRACKING JOINTS:")
        for joint in poor_tracking:
            print(f"    - {joint['joint_name']}: error = {joint['mean_tracking_error']:.4f}")
    
    if slow_joints:
        print(f"\n⚠️  SLOW RESPONSE JOINTS:")
        for joint in slow_joints:
            print(f"    - {joint['joint_name']}: final error = {joint['final_error']:.4f}")
    
    # Overall assessment
    responsive_joints = len(results) - len(unresponsive)
    control_quality = responsive_joints / len(results) if results else 0
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"  Responsive joints: {responsive_joints}/{len(results)} ({control_quality*100:.1f}%)")
    
    if control_quality > 0.9:
        print("  ✅ Excellent control - all joints responding well")
    elif control_quality > 0.7:
        print("  ⚠️  Good control - most joints working, some issues")
    elif control_quality > 0.5:
        print("  ❌ Poor control - many joints not responding properly")
    else:
        print("  🚨 Critical control issues - most joints unresponsive")
    
    return {
        'total_joints': len(results),
        'good_joints': len(good_joints),
        'responsive_joints': responsive_joints,
        'control_quality': control_quality,
        'issues': {
            'unresponsive': [j['joint_name'] for j in unresponsive],
            'poor_tracking': [j['joint_name'] for j in poor_tracking],
            'slow_response': [j['joint_name'] for j in slow_joints]
        }
    }


def test_whole_body_coordination(env: SkeletonHumanoidEnv):
    """Test coordinated movement of multiple joints"""
    
    print(f"\n🤖 WHOLE BODY COORDINATION TEST")
    print("=" * 50)
    
    env.reset()
    
    # Test 1: All joints move together (walking-like motion)
    print("Test 1: Coordinated walking motion...")
    
    for step in range(100):
        t = step / 100.0 * 2 * np.pi
        
        # Create walking-like joint pattern
        action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        
        # Hip joints (assume first few joints are legs)
        if env.num_actions >= 6:
            action[0, 0] = 0.2 * np.sin(t)      # Left hip
            action[0, 1] = -0.2 * np.sin(t)     # Right hip
            action[0, 2] = 0.3 * np.sin(t + np.pi/2)  # Left knee
            action[0, 3] = 0.3 * np.sin(t + np.pi/2)  # Right knee
        
        obs, rewards, dones, info = env.step(action)
        
        if step % 20 == 0:
            root_pos = env.root_pos[0].cpu().numpy()
            print(f"  Step {step}: Root pos = [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")
    
    final_root_pos = env.root_pos[0].cpu().numpy()
    initial_height = 0.975  # Default Genesis height
    height_change = abs(final_root_pos[2] - initial_height)
    
    print(f"Final root position: [{final_root_pos[0]:.3f}, {final_root_pos[1]:.3f}, {final_root_pos[2]:.3f}]")
    print(f"Height change: {height_change:.3f}")
    
    if height_change < 0.1:
        print("✅ Good stability - minimal height change")
    else:
        print("⚠️  Stability issues - significant height change")


def main():
    """Main diagnostic function"""
    
    print("🔧 GENESIS SKELETON JOINT CONTROL DIAGNOSTIC")
    print("=" * 70)
    
    try:
        # Initialize Genesis
        success, message = safe_init_genesis()
        if not success:
            raise RuntimeError(message)
        print(f"✅ {message}")
        
        # Create environment
        print("\n1. Creating Genesis skeleton environment...")
        env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=10.0,
            dt=0.01,
            show_viewer=True,
            use_box_feet=True
        )
        print("   ✅ Environment created")
        
        # Test individual joint control
        print("\n2. Testing individual joint control...")
        joint_results = test_all_joints_control(env)
        
        # Analyze results
        print("\n3. Analyzing control results...")
        summary = analyze_control_results(joint_results)
        
        # Test coordination
        print("\n4. Testing whole body coordination...")
        test_whole_body_coordination(env)
        
        print("\n" + "=" * 70)
        print("✅ JOINT CONTROL DIAGNOSTIC COMPLETE")
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS:")
        if summary['control_quality'] < 0.7:
            print("1. Check PD control gains in environment configuration")
            print("2. Verify joint limits and motor specifications")
            print("3. Test with smaller action commands")
        else:
            print("1. Joint control appears functional")
            print("2. Issues may be in policy training or reward structure")
            print("3. Consider checking trajectory following capabilities")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()