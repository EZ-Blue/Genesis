"""
GAIL Training Script for Walking Imitation

Complete training script for GAIL-based walking imitation using Genesis environment
and LocoMujoco expert trajectories. Follows LocoMujoco's GAIL training approach.
"""

import torch
import numpy as np
import time
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

# Genesis imports
import genesis as gs

# Local imports
import sys
sys.path.append('/home/ez/Documents/Genesis/genesis_loco')
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
from integration.gail_trainer import GAILGenesisTrainer, GAILConfig


def create_training_config(args) -> GAILConfig:
    """Create GAIL training configuration"""
    return GAILConfig(
        # Environment parameters
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        episode_length_s=args.episode_length,
        dt=0.01,
        
        # Training parameters
        total_timesteps=args.total_timesteps,
        update_epochs=args.update_epochs,
        n_disc_epochs=args.n_disc_epochs,
        disc_minibatch_size=args.disc_minibatch_size,
        num_minibatches=args.num_minibatches,
        
        # PPO parameters
        lr=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        weight_decay=0.0,
        
        # Discriminator parameters
        disc_lr=args.disc_lr,
        disc_ent_coef=0.01,
        
        # Network architecture
        hidden_layers=[256, 128],
        activation='tanh',
        init_std=0.125,
        learnable_std=False,
        
        # Reward mixing (0.0 = pure GAIL)
        proportion_env_reward=args.proportion_env_reward,
        
        # Logging
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )


def plot_training_metrics(metrics: dict, save_dir: str):
    """Plot and save training metrics"""
    # Check if we have any data to plot
    if not metrics or len(metrics.get('mean_episode_return', [])) == 0:
        print("   ⚠️  No metrics data to plot")
        return
    
    # Create timesteps if missing or wrong length
    timesteps = metrics.get('timestep', [])
    n_points = len(metrics['mean_episode_return'])
    
    if len(timesteps) != n_points:
        print(f"   ⚠️  Timesteps length mismatch ({len(timesteps)} vs {n_points}), generating sequence")
        timesteps = list(range(1, n_points + 1))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('GAIL Training Metrics', fontsize=16)
    
    # Episode returns
    axes[0, 0].plot(timesteps, metrics['mean_episode_return'])
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].set_xlabel('Update Steps' if len(timesteps) != n_points else 'Timesteps')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True)
    
    # Policy loss
    axes[0, 1].plot(timesteps, metrics['policy_loss'])
    axes[0, 1].set_title('Policy Loss')
    axes[0, 1].set_xlabel('Update Steps' if len(timesteps) != n_points else 'Timesteps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Value loss
    axes[0, 2].plot(timesteps, metrics['value_loss'])
    axes[0, 2].set_title('Value Loss')
    axes[0, 2].set_xlabel('Update Steps' if len(timesteps) != n_points else 'Timesteps')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True)
    
    # Discriminator loss
    axes[1, 0].plot(timesteps, metrics['discriminator_loss'])
    axes[1, 0].set_title('Discriminator Loss')
    axes[1, 0].set_xlabel('Update Steps' if len(timesteps) != n_points else 'Timesteps')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Discriminator outputs
    axes[1, 1].plot(timesteps, metrics['discriminator_output_policy'], label='Policy', alpha=0.7)
    axes[1, 1].plot(timesteps, metrics['discriminator_output_expert'], label='Expert', alpha=0.7)
    axes[1, 1].set_title('Discriminator Outputs')
    axes[1, 1].set_xlabel('Update Steps' if len(timesteps) != n_points else 'Timesteps')
    axes[1, 1].set_ylabel('Output')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Policy accuracy (approximation)
    if len(metrics['discriminator_output_policy']) > 0:
        policy_acc = [1 - p for p in metrics['discriminator_output_policy']]  # Lower is better for policy
        expert_acc = metrics['discriminator_output_expert']  # Higher is better for expert
        axes[1, 2].plot(timesteps, policy_acc, label='Policy Accuracy', alpha=0.7)
        axes[1, 2].plot(timesteps, expert_acc, label='Expert Accuracy', alpha=0.7)
        axes[1, 2].set_title('Discriminator Accuracy')
        axes[1, 2].set_xlabel('Update Steps' if len(timesteps) != n_points else 'Timesteps')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gail_training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training metrics plot saved to {save_dir}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='GAIL Training for Walking Imitation')
    
    # Environment parameters
    parser.add_argument('--num_envs', type=int, default=512, help='Number of parallel environments')
    parser.add_argument('--num_steps', type=int, default=20, help='Steps per environment per update')
    parser.add_argument('--episode_length', type=float, default=8.0, help='Episode length in seconds')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=5_000_000, help='Total training timesteps')
    parser.add_argument('--update_epochs', type=int, default=4, help='PPO update epochs')
    parser.add_argument('--n_disc_epochs', type=int, default=5, help='Discriminator update epochs')
    parser.add_argument('--disc_minibatch_size', type=int, default=1024, help='Discriminator batch size')
    parser.add_argument('--num_minibatches', type=int, default=16, help='Number of PPO minibatches')
    
    # Learning rates - Reduced for better stability
    parser.add_argument('--lr', type=float, default=3e-5, help='PPO learning rate')
    parser.add_argument('--disc_lr', type=float, default=1e-5, help='Discriminator learning rate')
    
    # Reward mixing
    parser.add_argument('--proportion_env_reward', type=float, default=0.0, 
                       help='Proportion of environment reward (0.0=pure GAIL, 1.0=pure env)')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=100, help='Save interval')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory')
    
    # Environment settings
    parser.add_argument('--show_viewer', action='store_true', help='Show Genesis viewer')
    parser.add_argument('--trajectory', type=str, default='walk', help='Expert trajectory to use')
    
    # Expert data loading
    parser.add_argument('--use_physics_data', action='store_true', 
                       help='Use physics-aware expert data (slow but accurate). Default: fast traditional approach')
    
    # Testing
    parser.add_argument('--test_only', action='store_true', help='Run tests only')
    
    args = parser.parse_args()
    
    # Create save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"./gail_walking_{timestamp}"
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("🚀 GAIL Walking Imitation Training")
    print("=" * 50)
    print(f"Save directory: {args.save_dir}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Environments: {args.num_envs}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Expert data approach: {'Physics-aware' if args.use_physics_data else 'Traditional (fast)'}")
    print(f"Proportion env reward: {args.proportion_env_reward}")
    
    try:
        # Initialize Genesis
        print("\n1. Initializing Genesis...")
        gs.init(backend=gs.gpu)
        print("   ✅ Genesis initialized")
        
        # Create environment
        print("2. Creating environment...")
        env = SkeletonHumanoidEnv(
            num_envs=args.num_envs,
            episode_length_s=args.episode_length,
            dt=0.01,
            use_box_feet=True,
            show_viewer=args.show_viewer
        )
        print(f"   ✅ Environment created: {env.num_envs} envs, {env.num_observations} obs, {env.num_actions} actions")
        
        # Create data bridge
        print("3. Loading expert trajectory...")
        data_bridge = LocoMujocoDataBridge(env)
        success = data_bridge.load_trajectory(args.trajectory)
        if not success:
            raise RuntimeError(f"Failed to load trajectory: {args.trajectory}")
        print(f"   ✅ Trajectory loaded: {data_bridge.trajectory_length} timesteps")
        
        # Create configuration
        print("4. Creating training configuration...")
        config = create_training_config(args)
        print(f"   ✅ Config created: {config.num_updates} updates, {config.minibatch_size} minibatch size")
        
        # Create GAIL trainer
        print("5. Creating GAIL trainer...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = GAILGenesisTrainer(env, data_bridge, config, device)
        
        # Run tests if requested
        if args.test_only:
            print("6. Running tests...")
            
            # Test expert data loading
            success = trainer.load_expert_data(use_physics=args.use_physics_data)
            if not success:
                raise RuntimeError("Failed to load expert data")
            print(f"   ✅ Expert data loaded: {trainer.expert_observations.shape[0]} samples")
            
            # Test single training step
            metrics = trainer.train_step(0)
            print(f"   ✅ Training step test completed")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"      {key}: {value:.4f}")
            
            print("\n🎉 All tests passed successfully!")
            return
        
        # Start training
        print("6. Starting training...")
        start_time = time.time()
        
        training_metrics = trainer.train(save_dir=args.save_dir, use_physics_data=args.use_physics_data)
        
        training_time = time.time() - start_time
        
        # Plot metrics
        print("7. Generating training plots...")
        plot_training_metrics(training_metrics, args.save_dir)
        
        # Save final configuration
        config_path = os.path.join(args.save_dir, 'gail_config.txt')
        with open(config_path, 'w') as f:
            f.write("GAIL Training Configuration\n")
            f.write("=" * 30 + "\n")
            for key, value in vars(config).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nTraining time: {training_time:.1f}s\n")
            f.write(f"Final episode return: {training_metrics['mean_episode_return'][-1]:.3f}\n")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Training time: {training_time:.1f}s")
        print(f"   Final episode return: {training_metrics['mean_episode_return'][-1]:.3f}")
        print(f"   Results saved to: {args.save_dir}")
        
        # Evaluate final policy
        print("\n8. Evaluating final policy...")
        env.reset()
        
        total_reward = 0
        steps = 0
        obs = env._get_observations()
        
        for _ in range(500):  # Run for 500 steps
            with torch.no_grad():
                action, _, _, _ = trainer.policy.get_action_and_value(obs)
            
            obs, reward, reset_buf, info = env.step(action)
            done = reset_buf
            
            total_reward += reward.mean().item()
            steps += 1
            
            if done.any():
                break
        
        avg_reward = total_reward / steps
        print(f"   ✅ Evaluation completed: {steps} steps, avg reward: {avg_reward:.3f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())