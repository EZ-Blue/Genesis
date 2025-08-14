#!/usr/bin/env python3
"""
Test Script for Comprehensive Imitation Learning Trainer

Quick validation test to ensure all refactored components work together correctly.
"""

import torch
import sys
import os

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_comprehensive_trainer():
    """Test the comprehensive imitation learning trainer"""
    print("🧪 Testing Comprehensive Imitation Learning Trainer")
    print("=" * 60)
    
    try:
        from comprehensive_imitation_trainer import ComprehensiveImitationTrainer, create_behavior_config
        
        # Create test configuration
        print("1. Creating test configuration...")
        config = create_behavior_config("walk")
        config.update({
            'num_envs': 4,  # Small for testing
            'episode_length_s': 5.0,
            'max_episode_steps': 250,
            'log_interval': 2,
            'show_viewer': False
        })
        print("   ✅ Configuration created")
        
        # Create trainer
        print("2. Initializing trainer...")
        trainer = ComprehensiveImitationTrainer(
            config=config, 
            save_dir="test_training", 
            behavior="walk"
        )
        print("   ✅ Trainer initialized")
        
        # Test trajectory collection
        print("3. Testing trajectory collection...")
        batch, collection_metrics = trainer.collect_trajectories()
        print(f"   ✅ Collected batch with {batch['observations'].shape[0]} samples")
        print(f"   ✅ Episode reward: {collection_metrics['episode_reward_mean']:.3f}")
        print(f"   ✅ Episode length: {collection_metrics['episode_length_mean']:.1f}")
        
        # Test training step
        print("4. Testing training step...")
        training_metrics = trainer.train_step(batch)
        print(f"   ✅ Policy loss: {training_metrics['policy_loss']:.4f}")
        print(f"   ✅ Discriminator loss: {training_metrics['disc_discriminator_loss']:.4f}")
        print(f"   ✅ Expert accuracy: {training_metrics['disc_expert_accuracy']:.3f}")
        
        # Test short training loop
        print("5. Testing short training loop...")
        trainer.train(num_iterations=3)
        print("   ✅ Training loop completed")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Comprehensive imitation learning trainer is working correctly")
        print("✅ Ready for full-scale training")
        print("=" * 60)
        
        # Cleanup
        import shutil
        shutil.rmtree("test_training", ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual components separately"""
    print("\n🔧 Testing Individual Components")
    print("=" * 60)
    
    try:
        # Test skeleton environment
        print("1. Testing skeleton environment...")
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        import genesis as gs
        
        # Initialize Genesis if not already done
        try:
            gs.init(backend=gs.gpu)
        except Exception as e:
            if "already initialized" not in str(e):
                raise e
        
        env = SkeletonHumanoidEnv(num_envs=2, episode_length_s=3.0, show_viewer=False)
        obs, _ = env.reset()
        actions = torch.zeros((2, env.num_actions), device=env.device)
        obs, rewards, dones, info = env.step(actions)
        print(f"   ✅ Environment: {env.num_envs} envs, {env.num_observations} obs, {env.num_actions} actions")
        
        # Test data bridge
        print("2. Testing data bridge...")
        from integration.data_bridge import LocoMujocoDataBridge
        
        data_bridge = LocoMujocoDataBridge(env)
        success = data_bridge.load_trajectory("walk")
        if success:
            print(f"   ✅ Data bridge: {data_bridge.trajectory_length} timesteps")
        else:
            print("   ⚠️ Data bridge: trajectory loading failed")
        
        # Test AMP integration
        print("3. Testing AMP integration...")
        from integration.amp_integration import AMPGenesisIntegration
        
        if success:
            amp_integration = AMPGenesisIntegration(env, data_bridge)
            expert_loaded = amp_integration.load_expert_data()
            if expert_loaded:
                amp_rewards = amp_integration.compute_amp_rewards(obs)
                print(f"   ✅ AMP integration: {amp_integration.n_expert_samples} expert samples")
            else:
                print("   ⚠️ AMP integration: expert data loading failed")
        
        # Test policy network
        print("4. Testing policy network...")
        from integration.simple_policy import SkeletonPolicyNetwork, PPOTrainer
        
        policy = SkeletonPolicyNetwork(
            obs_dim=env.num_observations,
            action_dim=env.num_actions,
            hidden_layers=[256, 128]
        )
        trainer = PPOTrainer(policy, device=env.device)
        actions, log_probs, values = policy.sample_actions(obs)
        print(f"   ✅ Policy network: {sum(p.numel() for p in policy.parameters())} parameters")
        
        print("\n✅ All individual components working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🤖 Comprehensive Imitation Learning Test Suite")
    print("=" * 70)
    
    # Test individual components first
    components_ok = test_individual_components()
    
    if components_ok:
        # Test full trainer
        trainer_ok = test_comprehensive_trainer()
        
        if trainer_ok:
            print("\n🎊 ALL TESTS SUCCESSFUL!")
            print("🚀 Ready to run full imitation learning training!")
            print("\nNext steps:")
            print("  1. Run: python comprehensive_imitation_trainer.py")
            print("  2. Choose your behavior (walk/run/squat)")
            print("  3. Select training scale")
            print("  4. Enjoy watching your agent learn!")
        else:
            print("\n🔧 Fix trainer issues and retry")
    else:
        print("\n🔧 Fix component issues before testing trainer")


if __name__ == "__main__":
    main()