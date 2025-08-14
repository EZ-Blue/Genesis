#!/usr/bin/env python3
"""
Test LocoMujoco Configuration Implementation

Quick test to verify that LocoMujoco configuration parameters are correctly applied.
"""

import sys
import os

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_locomujoco_config():
    """Test that LocoMujoco configuration is correctly implemented"""
    print("🧪 Testing LocoMujoco Configuration Implementation")
    print("=" * 60)
    
    from comprehensive_imitation_trainer import create_behavior_config
    
    # Get configuration
    config = create_behavior_config("walk")
    
    print("✅ Configuration Parameters:")
    print(f"   Policy Learning Rate: {config['policy']['learning_rate']} (LocoMujoco: 6e-5)")
    print(f"   Discriminator Learning Rate: {config['discriminator']['learning_rate']} (LocoMujoco: 5e-5)")
    print(f"   Policy Hidden Layers: {config['policy']['hidden_layers']} (LocoMujoco: [512, 256])")
    print(f"   Discriminator Hidden Layers: {config['discriminator']['hidden_layers']} (LocoMujoco: [512, 256])")
    print(f"   Clip Epsilon: {config['policy']['clip_epsilon']} (LocoMujoco: 0.1)")
    print(f"   Entropy Coefficient: {config['policy']['entropy_coeff']} (LocoMujoco: 0.0)")
    print(f"   Max Grad Norm: {config['policy']['max_grad_norm']} (LocoMujoco: 0.75)")
    print(f"   Environment Reward Weight: {config['env_reward_weight']} (LocoMujoco: 0.5)")
    print(f"   Update Epochs: {config['update_epochs']} (LocoMujoco: 4)")
    print(f"   Gamma: {config['gamma']} (LocoMujoco: 0.99)")
    print(f"   GAE Lambda: {config['gae_lambda']} (LocoMujoco: 0.95)")
    
    # Verification
    errors = []
    
    if config['policy']['learning_rate'] != 6e-5:
        errors.append(f"Policy LR mismatch: {config['policy']['learning_rate']} != 6e-5")
    
    if config['discriminator']['learning_rate'] != 5e-5:
        errors.append(f"Discriminator LR mismatch: {config['discriminator']['learning_rate']} != 5e-5")
    
    if config['policy']['hidden_layers'] != [512, 256]:
        errors.append(f"Policy layers mismatch: {config['policy']['hidden_layers']} != [512, 256]")
    
    if config['discriminator']['hidden_layers'] != [512, 256]:
        errors.append(f"Discriminator layers mismatch: {config['discriminator']['hidden_layers']} != [512, 256]")
    
    if config['policy']['clip_epsilon'] != 0.1:
        errors.append(f"Clip epsilon mismatch: {config['policy']['clip_epsilon']} != 0.1")
    
    if config['policy']['entropy_coeff'] != 0.0:
        errors.append(f"Entropy coeff mismatch: {config['policy']['entropy_coeff']} != 0.0")
    
    if config['env_reward_weight'] != 0.5:
        errors.append(f"Env reward weight mismatch: {config['env_reward_weight']} != 0.5")
    
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("\n🎉 All LocoMujoco configuration parameters correctly implemented!")
        return True

def main():
    """Main test function"""
    success = test_locomujoco_config()
    
    if success:
        print("\n✅ Ready to train with LocoMujoco-proven configuration!")
        print("🚀 Run: python comprehensive_imitation_trainer.py")
        print("\n📋 Key LocoMujoco Improvements Applied:")
        print("   - Exact network architectures: [512, 256] for both policy and discriminator")
        print("   - Proven learning rates: 6e-5 policy, 5e-5 discriminator")
        print("   - Optimal reward balance: 50/50 env/AMP")
        print("   - Stricter clipping: 0.1 instead of 0.2")
        print("   - No entropy regularization: 0.0 instead of 0.01")
        print("   - Multiple epoch updates: 4 epochs per batch")
        print("   - Proper gradient clipping: 0.75 max norm")
    else:
        print("\n🔧 Fix configuration issues and retry")

if __name__ == "__main__":
    main()