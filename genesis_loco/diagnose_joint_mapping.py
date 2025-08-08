"""
Joint Mapping Diagnostic Script

This script will definitively identify why joint control mapping is incorrect
by testing each joint individually and comparing expected vs actual movement.
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

def diagnose_joint_mapping():
    """Comprehensive joint mapping diagnosis"""
    print("🔍 Joint Mapping Diagnosis - Genesis LocoMujoco Integration")
    print("=" * 70)
    
    # Setup Genesis
    import genesis as gs
    gs.init(backend=gs.gpu)
    from environments.skeleton_humanoid import SkeletonHumanoidEnv
    
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=10.0,
        dt=0.019,
        show_viewer=True,
        use_box_feet=True
    )
    
    print(f"Environment Setup:")
    print(f"  - Total DOFs: {env.num_dofs}")
    print(f"  - Controllable Actions: {env.num_actions}")
    print(f"  - Device: {env.device}")
    
    # Reset and stabilize
    obs, _ = env.reset()
    print("\nStabilizing robot...")
    for _ in range(50):
        env.step(torch.zeros(1, env.num_actions, device=env.device))
    
    # Critical joints to test (the problematic ones)
    test_joints = [
        "lumbar_bending",    # User reports this controls hip rotation instead
        "lumbar_rotation",   # Test to see what this actually controls
        "hip_flexion_r",     # Key locomotion joint
        "hip_rotation_r",    # The joint that lumbar_bending allegedly controls
    ]
    
    print(f"\n📋 Index Analysis for Problematic Joints:")
    print(f"{'Joint Name':<20} {'Action Idx':<10} {'DOF Idx':<10} {'dof_start':<12} {'Global-Local'}")
    print("-" * 75)
    
    joint_info = {}
    for joint_name in test_joints:
        if joint_name in env.dof_names:
            # Get joint object
            joint_obj = env.robot.get_joint(joint_name)
            
            # Get the DOF index used by the environment
            action_name = None
            for act_name, joint_dof_idx in env.action_to_joint_idx.items():
                if env.dof_names[joint_dof_idx] == joint_name:
                    action_name = act_name
                    break
            
            if action_name:
                action_idx = env.action_spec.index(action_name)
                dof_idx = env.action_to_joint_idx[action_name]
                dof_start = joint_obj.dof_start
                entity_dof_start = joint_obj._entity.dof_start
                
                print(f"{joint_name:<20} {action_idx:<10} {dof_idx:<10} {dof_start:<12} {dof_start - entity_dof_start}")
                
                joint_info[joint_name] = {
                    'action_name': action_name,
                    'action_idx': action_idx,
                    'dof_idx': dof_idx,
                    'dof_start': dof_start,
                    'joint_obj': joint_obj
                }
    
    print(f"\n🎯 Individual Joint Movement Test:")
    print("Testing each joint individually to see which body part actually moves...\n")
    
    for joint_name, info in joint_info.items():
        print(f"Testing: {joint_name} (Action: {info['action_name']})")
        
        # Record initial joint positions
        initial_positions = env.dof_pos[0].clone()
        
        # Create action vector with only this joint activated
        test_action = torch.zeros(1, env.num_actions, device=env.device)
        test_action[0, info['action_idx']] = 0.5  # Moderate torque
        
        print(f"  Applying torque {test_action[0, info['action_idx']].item():.1f} to action index {info['action_idx']}")
        
        # Apply for several steps
        for step in range(100):  # 2 seconds at 50Hz
            env.step(test_action)
            
            if step % 25 == 0:  # Every 0.5 seconds
                current_positions = env.dof_pos[0]
                position_changes = torch.abs(current_positions - initial_positions)
                
                # Find joints that moved significantly
                moved_joints = torch.where(position_changes > 0.01)[0]
                
                if len(moved_joints) > 0:
                    print(f"    Step {step:3d}: Joints that moved (>0.01 rad):")
                    for moved_idx in moved_joints:
                        change = position_changes[moved_idx].item()
                        moved_joint_name = env.dof_names[moved_idx]
                        print(f"      {moved_joint_name:<20}: {change:+.4f} rad ({np.degrees(change):+.2f}°)")
                else:
                    print(f"    Step {step:3d}: No significant movement detected")
        
        # Return to neutral
        print(f"  Returning to neutral...")
        for _ in range(50):
            env.step(torch.zeros(1, env.num_actions, device=env.device))
        
        print()
        time.sleep(1.0)  # Brief pause between tests
    
    print(f"\n📊 Analysis Summary:")
    print("If joint control is working correctly, each test should move only the named joint.")
    print("If there's a mapping issue, the test will reveal which joint actually moves.")
    
    return env, joint_info

def verify_dof_name_mapping(env, joint_info):
    """Verify DOF name to index mapping consistency"""
    print(f"\n🔍 DOF Name Mapping Verification:")
    print(f"{'Joint Name':<20} {'Expected DOF Name':<20} {'Actual DOF Name':<20} {'Match?'}")
    print("-" * 80)
    
    for joint_name, info in joint_info.items():
        dof_idx = info['dof_idx']
        expected_dof_name = joint_name
        actual_dof_name = env.dof_names[dof_idx]
        match = "✅" if expected_dof_name == actual_dof_name else "❌"
        
        print(f"{joint_name:<20} {expected_dof_name:<20} {actual_dof_name:<20} {match}")
        
        if expected_dof_name != actual_dof_name:
            print(f"    ⚠️  MISMATCH DETECTED: Action targets '{expected_dof_name}' but controls '{actual_dof_name}'")

def suggest_fixes(env, joint_info):
    """Suggest potential fixes based on findings"""
    print(f"\n💡 Potential Fixes:")
    
    # Check if issue is in DOF name ordering
    print("1. DOF Name Order Verification:")
    print("   First 10 DOF names:", env.dof_names[:10])
    
    print("\n2. Alternative Index Mapping Strategies:")
    print("   A. Use dof_start instead of dofs_idx_local[0]:")
    for joint_name, info in joint_info.items():
        alt_idx = info['dof_start'] - info['joint_obj']._entity.dof_start
        print(f"      {joint_name}: current={info['dof_idx']}, alternative={alt_idx}")
    
    print("\n3. Direct Joint Name Lookup:")
    print("   Try finding DOF index by name instead of joint mapping:")
    for joint_name in joint_info.keys():
        if joint_name in env.dof_names:
            name_based_idx = env.dof_names.index(joint_name)
            current_idx = joint_info[joint_name]['dof_idx']
            print(f"      {joint_name}: current={current_idx}, name_based={name_based_idx}")

if __name__ == "__main__":
    try:
        print("Starting comprehensive joint mapping diagnosis...")
        env, joint_info = diagnose_joint_mapping()
        
        print("\n" + "="*70)
        verify_dof_name_mapping(env, joint_info)
        
        print("\n" + "="*70)
        suggest_fixes(env, joint_info)
        
        print(f"\n🎯 Diagnosis Complete!")
        print("Review the movement test results to identify the root cause of joint mapping issues.")
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()