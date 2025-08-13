#!/usr/bin/env python3
"""
Simple test to verify balanced PD gains can maintain standing stability
"""

import torch
import time
import sys
import os

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.skeleton_humanoid import SkeletonHumanoidEnv
import genesis as gs

def test_standing_stability():
    """Test if the skeleton can stand still with balanced PD gains"""
    print("🧪 Testing Standing Stability with Balanced PD Gains")
    print("=" * 60)
    
    # Initialize environment with balanced control
    gs.init(backend=gs.gpu)
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=10.0,
        dt=0.01,  # 100Hz for stability
        show_viewer=True,
        use_trajectory_control=True,  # This now uses balanced gains
        use_box_feet=True  # Enable box feet for better contact stability
    )
    
    print(f"\n📊 Environment Setup:")
    print(f"    DOFs: {env.num_dofs}")
    print(f"    Actions: {env.num_actions}")
    print(f"    Device: {env.device}")
    
    # Get initial standing pose (neutral)
    env.reset()
    initial_pos = env.root_pos[0].clone()
    initial_height = initial_pos[2].item()
    
    print(f"\n🏃 Initial State:")
    print(f"    Height: {initial_height:.3f}m")
    print(f"    Position: [{initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f}]")
    
    # Test parameters
    test_duration = 5.0  # seconds
    steps = int(test_duration / env.dt)
    
    print(f"\n⏱️  Running {test_duration}s stability test ({steps} steps)...")
    
    # Track stability metrics
    heights = []
    positions = []
    max_tilt = 0.0
    
    start_time = time.time()
    
    for step in range(steps):
        # Zero actions - just let PD control hold the pose
        actions = torch.zeros((1, env.num_actions), device=env.device)
        
        # Step environment
        obs, rewards, dones, info = env.step(actions)
        
        # Track metrics
        current_pos = env.root_pos[0]
        current_height = current_pos[2].item()
        heights.append(current_height)
        positions.append(current_pos.cpu().numpy())
        
        # Check for falling/instability
        if current_height < 0.7:  # Fallen if below 70cm
            print(f"\n❌ Model fell at step {step} (height: {current_height:.3f}m)")
            break
        
        # Track maximum tilt
        root_euler = env.root_quat[0]  # Could convert to euler if needed
        
        # Progress update
        if step % 100 == 0:
            elapsed = step * env.dt
            print(f"    Step {step:4d}/{steps} | t={elapsed:.1f}s | Height: {current_height:.3f}m | Stable: {not torch.any(dones).item()}")
        
        # Check for reset (instability)
        if torch.any(dones):
            print(f"\n⚠️  Environment reset at step {step} - instability detected")
            break
    
    # Analysis
    print(f"\n📈 Stability Analysis:")
    if heights:
        min_height = min(heights)
        max_height = max(heights)
        final_height = heights[-1]
        height_variance = torch.var(torch.tensor(heights)).item()
        
        print(f"    Initial height: {initial_height:.3f}m")
        print(f"    Final height:   {final_height:.3f}m")
        print(f"    Height range:   {min_height:.3f}m - {max_height:.3f}m")
        print(f"    Height drift:   {final_height - initial_height:+.3f}m")
        print(f"    Height variance: {height_variance:.6f}")
        
        # Stability verdict
        height_stable = abs(final_height - initial_height) < 0.05  # 5cm drift tolerance
        no_falling = min_height > 0.8  # Never below 80cm
        
        if height_stable and no_falling:
            print(f"\n✅ STABLE: PD gains successfully maintain standing pose")
        elif no_falling:
            print(f"\n⚠️  MARGINALLY STABLE: No falling but some drift")
        else:
            print(f"\n❌ UNSTABLE: Model cannot maintain standing pose")
            
        # Recommendations
        if not height_stable:
            print(f"\n💡 Recommendations:")
            if height_variance > 0.001:
                print(f"    - High variance suggests oscillations - reduce kv (damping)")
            if final_height < initial_height - 0.02:
                print(f"    - Sinking suggests insufficient kp - increase leg joint gains")
            if min_height < 0.9:
                print(f"    - Low minimum height suggests instability - check ankle gains")
    
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.1f}s")

def main():
    """Main test function"""
    try:
        test_standing_stability()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()