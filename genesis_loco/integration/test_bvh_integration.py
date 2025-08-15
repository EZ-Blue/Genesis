#!/usr/bin/env python3
"""
Test script for BVH integration with Genesis imitation learning

This script tests the complete pipeline:
1. BVH preprocessing (using your script)
2. Loading preprocessed NPZ trajectories 
3. Genesis environment setup
4. Data bridge compatibility

Usage:
python test_bvh_integration.py --bvh_file path/to/your/file.bvh
"""

import argparse
import os
import sys
from pathlib import Path

# Fix Qt/OpenCV display issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

# Add paths
sys.path.append('/home/choonspin/intuitive_autonomy/loco-mujoco/preprocess_scripts')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs


def test_bvh_preprocessing(bvh_file: str):
    """Test BVH preprocessing pipeline"""
    print("🔄 Testing BVH preprocessing...")
    
    try:
        from bvh_general_pipeline import convert_bvh_general
        
        bvh_path = Path(bvh_file)
        if not bvh_path.exists():
            print(f"❌ BVH file not found: {bvh_file}")
            return None
        
        # Convert BVH to NPZ
        trajectory, metadata = convert_bvh_general(str(bvh_path))
        
        if trajectory is None:
            print("❌ BVH conversion failed")
            return None
        
        # Save NPZ file
        output_file = bvh_path.with_suffix('.npz').name
        trajectory.save(output_file)
        
        print(f"✅ BVH preprocessing successful:")
        print(f"   Input: {bvh_file}")
        print(f"   Output: {output_file}")
        print(f"   Frames: {trajectory.data.qpos.shape[0]}")
        print(f"   Frequency: {trajectory.info.frequency} Hz")
        
        return output_file
        
    except Exception as e:
        print(f"❌ BVH preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_genesis_environment():
    """Test Genesis environment setup"""
    print("\n🔄 Testing Genesis environment...")
    
    try:
        # Initialize Genesis
        gs.init(backend=gs.gpu)
        
        # Import and create environment
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        env = SkeletonHumanoidEnv(
            num_envs=4,  # Small test
            episode_length_s=5.0,
            dt=0.01,
            use_box_feet=True,
            show_viewer=False
        )
        
        print(f"✅ Genesis environment created:")
        print(f"   Environments: {env.num_envs}")
        print(f"   Actions: {env.num_actions}")
        print(f"   Observations: {env.num_observations}")
        
        # Test reset
        obs, _ = env.reset()
        print(f"   Reset successful, obs shape: {obs.shape}")
        
        return env
        
    except Exception as e:
        print(f"❌ Genesis environment setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_bridge(env, npz_file: str):
    """Test data bridge with NPZ trajectory"""
    print(f"\n🔄 Testing data bridge with {npz_file}...")
    
    try:
        from integration.data_bridge import LocoMujocoDataBridge
        
        # Create data bridge
        bridge = LocoMujocoDataBridge(env)
        
        # Load NPZ trajectory
        success = bridge.load_trajectory(npz_file)
        
        if not success:
            print("❌ Failed to load NPZ trajectory")
            return False
        
        print(f"✅ Data bridge integration successful:")
        print(f"   Trajectory length: {bridge.trajectory_length} timesteps")
        print(f"   Trajectory frequency: {bridge.trajectory_frequency} Hz")
        print(f"   Segments created: {len(bridge.segments)}")
        
        # Test getting trajectory state
        state = bridge.get_trajectory_state(0)
        if state is not None:
            print(f"   ✅ Can extract trajectory states")
            print(f"      DOF pos shape: {state['dof_pos'].shape}")
            print(f"      Root pos: {state['root_pos']}")
        
        # Test applying trajectory state
        bridge.apply_trajectory_state(state, env_ids=None)
        print(f"   ✅ Can apply trajectory states to environment")
        
        return True
        
    except Exception as e:
        print(f"❌ Data bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_amp_integration(env, bridge):
    """Test AMP integration"""
    print(f"\n🔄 Testing AMP integration...")
    
    try:
        from integration.amp_integration import AMPGenesisIntegration
        
        # AMP config
        discriminator_config = {
            'hidden_layers': [512, 256],
            'activation': 'tanh',
            'learning_rate': 5e-6,
            'use_running_norm': True
        }
        
        # Create AMP integration
        amp = AMPGenesisIntegration(
            genesis_env=env,
            data_bridge=bridge,
            discriminator_config=discriminator_config
        )
        
        # Load expert data
        success = amp.load_expert_data()
        
        if not success:
            print("❌ Failed to load expert data for AMP")
            return False
        
        print(f"✅ AMP integration successful:")
        print(f"   Expert samples: {amp.n_expert_samples}")
        print(f"   Discriminator parameters: {sum(p.numel() for p in amp.discriminator.parameters())}")
        
        # Test AMP reward computation
        obs, _ = env.reset()
        amp_rewards = amp.compute_amp_rewards(obs)
        print(f"   ✅ AMP rewards computed, shape: {amp_rewards.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ AMP integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test BVH integration with Genesis')
    parser.add_argument('--bvh_file', '-b', type=str, 
                       help='Path to BVH file to test')
    parser.add_argument('--npz_file', '-n', type=str,
                       help='Path to existing NPZ file to test (skip BVH processing)')
    
    args = parser.parse_args()
    
    if not args.bvh_file and not args.npz_file:
        print("❌ Please provide either --bvh_file or --npz_file")
        return
    
    print("🧪 BVH Integration Test Suite")
    print("=" * 50)
    
    # Step 1: BVH preprocessing (if needed)
    if args.bvh_file:
        npz_file = test_bvh_preprocessing(args.bvh_file)
        if npz_file is None:
            return
    else:
        npz_file = args.npz_file
        if not os.path.exists(npz_file):
            print(f"❌ NPZ file not found: {npz_file}")
            return
    
    # Step 2: Genesis environment
    env = test_genesis_environment()
    if env is None:
        return
    
    # Step 3: Data bridge
    bridge_success = test_data_bridge(env, npz_file)
    if not bridge_success:
        return
    
    # Step 4: AMP integration
    from integration.data_bridge import LocoMujocoDataBridge
    bridge = LocoMujocoDataBridge(env)
    bridge.load_trajectory(npz_file)
    
    amp_success = test_amp_integration(env, bridge)
    
    # Final results
    print("\n" + "=" * 50)
    if bridge_success and amp_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ BVH integration is working correctly")
        print(f"✅ You can now use '{npz_file}' with the comprehensive trainer")
        print("\nNext steps:")
        print(f"1. Run: python comprehensive_imitation_trainer.py")
        print(f"2. Select option 4 (custom)")
        print(f"3. Enter path: {npz_file}")
    else:
        print("❌ Some tests failed - check output above")
    print("=" * 50)


if __name__ == "__main__":
    main()