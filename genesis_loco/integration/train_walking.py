"""
Full-Scale Walking Training with Visualization

Scale up the imitation learning training to get a walking agent.
Includes visualization, monitoring, and model saving.
"""

import torch
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import sys

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_trainer import SimpleImitationTrainer, create_default_config


class WalkingTrainer:
    """
    Enhanced trainer for full-scale walking imitation learning
    
    Features:
    - Visualization with Genesis viewer
    - Training progress monitoring
    - Model checkpointing
    - Performance metrics logging
    - Real-time plotting
    """
    
    def __init__(self, config: dict, save_dir: str = "walking_training"):
        self.config = config
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize trainer
        print("🚀 Initializing Full-Scale Walking Trainer")
        self.trainer = SimpleImitationTrainer(config)
        
        # Training metrics for monitoring
        self.training_history = {
            'iteration': [],
            'episode_reward': [],
            'episode_length': [],
            'amp_reward': [],
            'env_reward': [],
            'policy_loss': [],
            'discriminator_loss': [],
            'expert_accuracy': [],
            'policy_accuracy': [],
            'training_time': []
        }
        
        print(f"✅ Trainer ready! Saving to: {save_dir}")
    
    def train_full_scale(self, total_iterations: int = 2000):
        """
        Run full-scale training with monitoring and checkpointing
        
        Args:
            total_iterations: Total number of training iterations
        """
        print(f"\n🎯 Starting Full-Scale Walking Training")
        print(f"   Total iterations: {total_iterations}")
        print(f"   Environments: {self.config['num_envs']}")
        print(f"   Episode length: {self.config['episode_length_s']}s")
        print(f"   Visualization: {self.config['show_viewer']}")
        print("=" * 60)
        
        start_time = time.time()
        best_reward = float('-inf')
        
        for iteration in range(total_iterations):
            iter_start_time = time.time()
            
            # Collect trajectories and train
            batch, collection_metrics = self.trainer.collect_trajectories()
            training_metrics = self.trainer.train_step(batch)
            
            iter_time = time.time() - iter_start_time
            
            # Store metrics
            self._update_history(iteration, collection_metrics, training_metrics, iter_time)
            
            # Log progress
            if iteration % self.config['log_interval'] == 0:
                self._log_progress(iteration, total_iterations, collection_metrics, training_metrics, iter_time)
            
            # Plot progress
            if iteration % (self.config['log_interval'] * 4) == 0 and iteration > 0:
                self._plot_training_progress()
            
            # Save checkpoints
            if iteration % 100 == 0:
                self._save_checkpoint(iteration, collection_metrics['episode_reward_mean'])
            
            # Save best model
            current_reward = collection_metrics['episode_reward_mean']
            if current_reward > best_reward:
                best_reward = current_reward
                self._save_best_model(iteration, current_reward)
        
        total_time = time.time() - start_time
        
        # Final results
        self._print_final_results(total_iterations, total_time, best_reward)
        self._plot_training_progress(save=True)
        
        return self.trainer
    
    def _update_history(self, iteration: int, collection_metrics: dict, training_metrics: dict, iter_time: float):
        """Update training history with current metrics"""
        self.training_history['iteration'].append(iteration)
        self.training_history['episode_reward'].append(collection_metrics['episode_reward_mean'])
        self.training_history['episode_length'].append(collection_metrics['episode_length_mean'])
        self.training_history['amp_reward'].append(collection_metrics['amp_reward_mean'])
        self.training_history['env_reward'].append(collection_metrics['env_reward_mean'])
        self.training_history['policy_loss'].append(training_metrics['policy_loss'])
        self.training_history['discriminator_loss'].append(training_metrics['disc_discriminator_loss'])
        self.training_history['expert_accuracy'].append(training_metrics['disc_expert_accuracy'])
        self.training_history['policy_accuracy'].append(training_metrics['disc_policy_accuracy'])
        self.training_history['training_time'].append(iter_time)
    
    def _log_progress(self, iteration: int, total_iterations: int, collection_metrics: dict, 
                     training_metrics: dict, iter_time: float):
        """Log training progress"""
        progress = (iteration / total_iterations) * 100
        
        print(f"\n🏃 Iteration {iteration:4d}/{total_iterations} ({progress:5.1f}%)")
        print(f"   Time: {iter_time:.2f}s")
        print(f"   📊 Episode Reward: {collection_metrics['episode_reward_mean']:8.3f}")
        print(f"   📏 Episode Length: {collection_metrics['episode_length_mean']:8.1f} steps")
        print(f"   🎭 AMP Reward:     {collection_metrics['amp_reward_mean']:8.3f}")
        print(f"   🎯 Env Reward:     {collection_metrics['env_reward_mean']:8.3f}")
        print(f"   🧠 Policy Loss:    {training_metrics['policy_loss']:8.4f}")
        print(f"   🤖 Expert Acc:     {training_metrics['disc_expert_accuracy']:8.3f}")
        print(f"   👤 Policy Acc:     {training_metrics['disc_policy_accuracy']:8.3f}")
        print("-" * 50)
    
    def _plot_training_progress(self, save: bool = False):
        """Plot training progress"""
        if len(self.training_history['iteration']) < 10:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Walking Training Progress', fontsize=16)
        
        iterations = self.training_history['iteration']
        
        # Episode metrics
        axes[0, 0].plot(iterations, self.training_history['episode_reward'], 'b-', label='Episode Reward')
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(iterations, self.training_history['episode_length'], 'g-', label='Episode Length')
        axes[0, 1].set_title('Episode Length (steps)')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].grid(True)
        
        # Reward breakdown
        axes[0, 2].plot(iterations, self.training_history['amp_reward'], 'r-', label='AMP Reward')
        axes[0, 2].plot(iterations, self.training_history['env_reward'], 'orange', label='Env Reward')
        axes[0, 2].set_title('Reward Breakdown')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Training losses
        axes[1, 0].plot(iterations, self.training_history['policy_loss'], 'purple', label='Policy Loss')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(iterations, self.training_history['discriminator_loss'], 'brown', label='Discriminator Loss')
        axes[1, 1].set_title('Discriminator Loss')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].grid(True)
        
        # Discriminator accuracies
        axes[1, 2].plot(iterations, self.training_history['expert_accuracy'], 'cyan', label='Expert Acc')
        axes[1, 2].plot(iterations, self.training_history['policy_accuracy'], 'magenta', label='Policy Acc')
        axes[1, 2].set_title('Discriminator Accuracy')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
            print(f"   📊 Training plot saved to {self.save_dir}/training_progress.png")
        else:
            plt.savefig(os.path.join(self.save_dir, f'progress_iter_{iterations[-1]}.png'), dpi=150)
        
        plt.close()
    
    def _save_checkpoint(self, iteration: int, reward: float):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': self.trainer.policy.state_dict(),
            'ppo_optimizer_state_dict': self.trainer.ppo_trainer.optimizer.state_dict(),
            'discriminator_state_dict': self.trainer.amp_integration.discriminator.state_dict(),
            'disc_optimizer_state_dict': self.trainer.amp_integration.trainer.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'reward': reward
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_iter_{iteration}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if iteration % 500 == 0:  # Less frequent logging for checkpoints
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self, iteration: int, reward: float):
        """Save best performing model"""
        best_model = {
            'iteration': iteration,
            'reward': reward,
            'policy_state_dict': self.trainer.policy.state_dict(),
            'discriminator_state_dict': self.trainer.amp_integration.discriminator.state_dict(),
            'config': self.config
        }
        
        best_path = os.path.join(self.save_dir, 'best_model.pt')
        torch.save(best_model, best_path)
        print(f"   🏆 New best model saved! Reward: {reward:.3f}")
    
    def _print_final_results(self, total_iterations: int, total_time: float, best_reward: float):
        """Print final training results"""
        hours = total_time / 3600
        final_reward = self.training_history['episode_reward'][-1]
        final_length = self.training_history['episode_length'][-1]
        
        print("\n" + "=" * 60)
        print("🎉 WALKING TRAINING COMPLETED!")
        print("=" * 60)
        print(f"   Total iterations: {total_iterations}")
        print(f"   Training time: {hours:.2f} hours")
        print(f"   Final episode reward: {final_reward:.3f}")
        print(f"   Final episode length: {final_length:.1f} steps")
        print(f"   Best reward achieved: {best_reward:.3f}")
        print(f"   Models saved to: {self.save_dir}")
        print("=" * 60)


def create_walking_config() -> dict:
    """Create optimized configuration for walking training"""
    config = create_default_config()
    
    # Scale up for serious training
    config.update({
        # Environment scaling
        'num_envs': 64,              # More parallel environments
        'episode_length_s': 15.0,    # Longer episodes for learning
        'dt': 0.02,                  # 50Hz simulation
        'show_viewer': False,         # VISUALIZATION ENABLED!
        
        # Training parameters
        'max_episode_steps': 375,    # 7.5 seconds at 50Hz
        'min_episode_steps': 200,    # Minimum collection steps
        'env_reward_weight': 0.1,    # 90% AMP reward, 10% environment reward
        'log_interval': 10,          # Log every 10 iterations
        
        # Policy network - larger for better performance
        'policy': {
            'hidden_layers': [512, 256, 128],
            'activation': 'tanh',
            'learning_rate': 3e-4,
            'clip_epsilon': 0.2
        },
        
        # Discriminator - robust architecture
        'discriminator': {
            'hidden_layers': [512, 256],
            'activation': 'tanh',
            'learning_rate': 5e-5,     # Lower LR for stable discriminator training
            'use_running_norm': True
        }
    })
    
    return config


def create_fast_config() -> dict:
    """Create configuration for faster training (fewer environments)"""
    config = create_walking_config()
    
    config.update({
        'num_envs': 16,              # Fewer environments for faster iteration
        'episode_length_s': 10.0,   # Shorter episodes
        'max_episode_steps': 250,    # 5 seconds at 50Hz
        'log_interval': 5,           # More frequent logging
    })
    
    return config


def main():
    """Main training function"""
    print("🚶‍♂️ Genesis Skeleton Walking Training")
    print("=" * 50)
    
    # Choose configuration
    print("Select training configuration:")
    print("1. Full Scale (64 envs, 15s episodes) - Best results, slower")
    print("2. Fast Training (16 envs, 10s episodes) - Faster iteration")
    print("3. Quick Test (4 envs, 5s episodes) - Very fast, basic testing")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        config = create_walking_config()
        iterations = 2000
        print("🎯 Full Scale Training Selected")
    elif choice == "2":
        config = create_fast_config()
        iterations = 1000
        print("🚀 Fast Training Selected")
    else:  # Default to quick test
        config = create_default_config()
        config['show_viewer'] = False
        config['num_envs'] = 4
        config['log_interval'] = 5
        iterations = 100
        print("⚡ Quick Test Selected")
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"walking_training_{timestamp}"
    
    print(f"\nTraining Configuration:")
    print(f"   Environments: {config['num_envs']}")
    print(f"   Episode Length: {config['episode_length_s']}s")
    print(f"   Max Iterations: {iterations}")
    print(f"   Visualization: {config['show_viewer']}")
    print(f"   Save Directory: {save_dir}")
    
    input("\nPress Enter to start training...")
    
    # Initialize and run training
    trainer = WalkingTrainer(config, save_dir)
    trainer.train_full_scale(total_iterations=iterations)
    
    print(f"\n🎊 Training Complete! Check {save_dir} for results.")
    print("📊 View training_progress.png for performance plots")
    print("🏆 Load best_model.pt to test your trained walking agent")


if __name__ == "__main__":
    main()