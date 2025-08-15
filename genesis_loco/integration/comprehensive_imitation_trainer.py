#!/usr/bin/env python3
"""
Comprehensive Imitation Learning Trainer

A complete, efficient training system for Genesis skeleton humanoid imitation learning.
Supports multiple behaviors (walking, running, squatting) with AMP discriminator training.

NEW: BVH Integration Support
- Custom NPZ trajectory files from BVH preprocessing
- Interactive file selection for custom motions
- Full compatibility with BVH preprocessing pipeline

BVH Workflow:
1. Preprocess BVH: /home/choonspin/intuitive_autonomy/loco-mujoco/preprocess_scripts/bvh_general_pipeline.py
2. Train with NPZ: Select option 4 (custom) in this trainer
3. Test integration: test_bvh_integration.py

Compatible with refactored components:
- skeleton_humanoid.py (SkeletonHumanoidEnv)
- data_bridge.py (LocoMujocoDataBridge) - Enhanced with NPZ support
- amp_integration.py (AMPGenesisIntegration)
- amp_discriminator.py (AMPDiscriminator, AMPTrainer)
"""

# Fix Qt/OpenCV display issues
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import torch
import numpy as np
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
import json

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import refactored components
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
from integration.amp_integration import AMPGenesisIntegration
from integration.simple_policy import SkeletonPolicyNetwork, PPOTrainer
import genesis as gs


def safe_init_genesis():
    """Safely initialize Genesis, handling already-initialized case"""
    try:
        gs.init(backend=gs.gpu)
        return True, "Genesis initialized"
    except Exception as e:
        if "already initialized" in str(e):
            return True, "Genesis already initialized"
        else:
            return False, f"Genesis initialization failed: {e}"


class TrajectoryBuffer:
    """Efficient trajectory collection buffer with GAE computation"""
    
    def __init__(self, num_envs: int, max_episode_length: int, obs_dim: int, action_dim: int, device: torch.device):
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Pre-allocate buffers
        self.observations = torch.zeros((max_episode_length, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((max_episode_length, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((max_episode_length, num_envs), device=device)
        self.values = torch.zeros((max_episode_length, num_envs), device=device)
        self.log_probs = torch.zeros((max_episode_length, num_envs), device=device)
        self.dones = torch.zeros((max_episode_length, num_envs), dtype=torch.bool, device=device)
        
        self.step = 0
        
    def add(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
            values: torch.Tensor, log_probs: torch.Tensor, dones: torch.Tensor):
        """Add step data to buffer"""
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.dones[self.step] = dones
        self.step += 1
        
    def compute_gae(self, next_values: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        gae = 0
        for t in reversed(range(self.step)):
            if t == self.step - 1:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = next_values
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
        
        return advantages, returns
    
    def get_batch(self, advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get flattened batch for training"""
        batch = {
            'observations': self.observations[:self.step].reshape(-1, self.obs_dim),
            'actions': self.actions[:self.step].reshape(-1, self.action_dim),
            'old_log_probs': self.log_probs[:self.step].reshape(-1),
            'advantages': advantages[:self.step].reshape(-1),
            'returns': returns[:self.step].reshape(-1)
        }
        return batch
    
    def clear(self):
        """Clear buffer for next episode collection"""
        self.step = 0


class MetricsTracker:
    """Track and log training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size)
        self.amp_rewards = deque(maxlen=self.window_size)
        self.env_rewards = deque(maxlen=self.window_size)
        self.policy_losses = deque(maxlen=self.window_size)
        self.discriminator_losses = deque(maxlen=self.window_size)
        self.expert_accuracies = deque(maxlen=self.window_size)
        self.policy_accuracies = deque(maxlen=self.window_size)
        
        self.iteration_times = deque(maxlen=self.window_size)
        
        # Full history for plotting
        self.full_history = {
            'iterations': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'amp_rewards': [],
            'env_rewards': [],
            'policy_losses': [],
            'discriminator_losses': [],
            'expert_accuracies': [],
            'policy_accuracies': []
        }
    
    def update(self, iteration: int, collection_metrics: Dict, training_metrics: Dict, iter_time: float):
        """Update metrics with current iteration data"""
        # Collection metrics
        self.episode_rewards.append(collection_metrics['episode_reward_mean'])
        self.episode_lengths.append(collection_metrics['episode_length_mean'])
        self.amp_rewards.append(collection_metrics['amp_reward_mean'])
        self.env_rewards.append(collection_metrics['env_reward_mean'])
        
        # Training metrics
        self.policy_losses.append(training_metrics['policy_loss'])
        self.discriminator_losses.append(training_metrics['disc_discriminator_loss'])
        self.expert_accuracies.append(training_metrics['disc_expert_accuracy'])
        self.policy_accuracies.append(training_metrics['disc_policy_accuracy'])
        
        # Timing
        self.iteration_times.append(iter_time)
        
        # Full history
        self.full_history['iterations'].append(iteration)
        self.full_history['episode_rewards'].append(collection_metrics['episode_reward_mean'])
        self.full_history['episode_lengths'].append(collection_metrics['episode_length_mean'])
        self.full_history['amp_rewards'].append(collection_metrics['amp_reward_mean'])
        self.full_history['env_rewards'].append(collection_metrics['env_reward_mean'])
        self.full_history['policy_losses'].append(training_metrics['policy_loss'])
        self.full_history['discriminator_losses'].append(training_metrics['disc_discriminator_loss'])
        self.full_history['expert_accuracies'].append(training_metrics['disc_expert_accuracy'])
        self.full_history['policy_accuracies'].append(training_metrics['disc_policy_accuracy'])
    
    def get_recent_stats(self) -> Dict[str, float]:
        """Get recent windowed statistics"""
        return {
            'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'episode_reward_std': np.std(self.episode_rewards) if self.episode_rewards else 0.0,
            'episode_length_mean': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'amp_reward_mean': np.mean(self.amp_rewards) if self.amp_rewards else 0.0,
            'env_reward_mean': np.mean(self.env_rewards) if self.env_rewards else 0.0,
            'policy_loss_mean': np.mean(self.policy_losses) if self.policy_losses else 0.0,
            'discriminator_loss_mean': np.mean(self.discriminator_losses) if self.discriminator_losses else 0.0,
            'expert_accuracy_mean': np.mean(self.expert_accuracies) if self.expert_accuracies else 0.0,
            'policy_accuracy_mean': np.mean(self.policy_accuracies) if self.policy_accuracies else 0.0,
            'iteration_time_mean': np.mean(self.iteration_times) if self.iteration_times else 0.0
        }


class ComprehensiveImitationTrainer:
    """
    Comprehensive imitation learning trainer for multiple behaviors
    
    Features:
    - Multi-behavior support (walk, run, squat, etc.)
    - Efficient training pipeline with PPO + AMP
    - Comprehensive metrics tracking and visualization
    - Model checkpointing and resumption
    - Real-time monitoring and logging
    """
    
    def __init__(self, config: Dict, save_dir: str = None, behavior: str = "walk"):
        self.config = config
        self.behavior = behavior
        
        # Create save directory
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"imitation_training_{behavior}_{timestamp}"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"🚀 Initializing Comprehensive Imitation Learning Trainer")
        print(f"   Behavior: {behavior}")
        print(f"   Save directory: {save_dir}")
        print("=" * 70)
        
        # Initialize Genesis (only if not already initialized)
        print("1. Initializing Genesis...")
        success, message = safe_init_genesis()
        if not success:
            raise RuntimeError(message)
        print(f"   ✅ {message}")
        
        self.device = gs.device
        print(f"   ✅ Device: {self.device}")
        
        # Setup components
        self._setup_environment()
        self._setup_data_bridge()
        self._setup_amp_integration()
        self._setup_policy()
        self._setup_training_buffer()
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(window_size=config.get('metrics_window', 100))
        
        # Training state
        self.iteration = 0
        self.best_reward = float('-inf')
        
        print("✅ Trainer initialization complete!")
        print("=" * 70)
    
    def _setup_environment(self):
        """Setup Genesis skeleton environment"""
        print("2. Setting up environment...")
        
        self.env = SkeletonHumanoidEnv(
            num_envs=self.config['num_envs'],
            episode_length_s=self.config['episode_length_s'],
            dt=self.config['dt'],
            show_viewer=self.config.get('show_viewer', False),
            use_box_feet=self.config.get('use_box_feet', True)
        )
        
        self.obs_dim = self.env.num_observations
        self.action_dim = self.env.num_actions
        
        print(f"   ✅ Environment: {self.config['num_envs']} envs")
        print(f"   ✅ Observations: {self.obs_dim}")
        print(f"   ✅ Actions: {self.action_dim}")
    
    def _setup_data_bridge(self):
        """Setup LocoMujoco data bridge for expert trajectories"""
        print("3. Setting up data bridge...")
        
        self.data_bridge = LocoMujocoDataBridge(self.env)
        
        # Load behavior-specific trajectory
        success = self.data_bridge.load_trajectory(self.behavior)
        if not success:
            raise RuntimeError(f"Failed to load {self.behavior} trajectory")
        
        print(f"   ✅ Expert {self.behavior} trajectory loaded:")
        print(f"      - Length: {self.data_bridge.trajectory_length} timesteps")
        print(f"      - Frequency: {self.data_bridge.trajectory_frequency} Hz")
    
    def _setup_amp_integration(self):
        """Setup AMP discriminator and expert data"""
        print("4. Setting up AMP integration...")
        
        self.amp_integration = AMPGenesisIntegration(
            genesis_env=self.env,
            data_bridge=self.data_bridge,
            discriminator_config=self.config['discriminator']
        )
        
        # Load expert data for discriminator
        success = self.amp_integration.load_expert_data()
        if not success:
            raise RuntimeError("Failed to load expert observations")
        
        print(f"   ✅ AMP discriminator ready:")
        print(f"      - Expert samples: {self.amp_integration.n_expert_samples}")
        print(f"      - Discriminator parameters: {sum(p.numel() for p in self.amp_integration.discriminator.parameters())}")
    
    def _setup_policy(self):
        """Setup PPO policy network"""
        print("5. Setting up policy network...")
        
        self.policy = SkeletonPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=self.config['policy']['hidden_layers'],
            activation=self.config['policy']['activation'],
            use_obs_norm=self.config['policy'].get('use_obs_norm', True)
        )
        
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            learning_rate=self.config['policy']['learning_rate'],
            clip_epsilon=self.config['policy']['clip_epsilon'],
            value_coeff=self.config['policy'].get('value_coeff', 0.5),
            entropy_coeff=self.config['policy'].get('entropy_coeff', 0.0),
            max_grad_norm=self.config['policy'].get('max_grad_norm', 0.75),
            device=self.device
        )
        
        param_count = sum(p.numel() for p in self.policy.parameters())
        print(f"   ✅ Policy network: {param_count} parameters")
    
    def _setup_training_buffer(self):
        """Setup trajectory collection buffer"""
        max_steps = self.config['max_episode_steps']
        
        self.buffer = TrajectoryBuffer(
            num_envs=self.config['num_envs'],
            max_episode_length=max_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        print(f"   ✅ Buffer: {max_steps} max steps per episode")
    
    def collect_trajectories(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Collect trajectories using current policy"""
        self.policy.eval()
        self.buffer.clear()
        
        # Reset environment
        obs, _ = self.env.reset()
        episode_rewards = torch.zeros(self.config['num_envs'], device=self.device)
        episode_lengths = torch.zeros(self.config['num_envs'], device=self.device)
        
        env_rewards_sum = torch.zeros(self.config['num_envs'], device=self.device)
        amp_rewards_sum = torch.zeros(self.config['num_envs'], device=self.device)
        
        completed_episodes = []
        
        # Collect steps
        for step in range(self.config['max_episode_steps']):
            # Sample actions from policy
            with torch.no_grad():
                actions, log_probs, values = self.policy.sample_actions(obs)
            
            # Environment step
            next_obs, env_rewards, dones, _ = self.env.step(actions)
            
            # Compute AMP rewards
            amp_rewards = self.amp_integration.compute_amp_rewards(obs)
            
            # Mix environment and AMP rewards
            mixed_rewards = (self.config['env_reward_weight'] * env_rewards + 
                           (1 - self.config['env_reward_weight']) * amp_rewards)
            
            # Store in buffer
            self.buffer.add(obs, actions, mixed_rewards, values, log_probs, dones)
            
            # Update metrics
            episode_rewards += mixed_rewards
            episode_lengths += 1
            env_rewards_sum += env_rewards
            amp_rewards_sum += amp_rewards
            
            # Handle episode completion
            if dones.any():
                finished_episodes_mask = dones
                for env_idx in range(self.config['num_envs']):
                    if finished_episodes_mask[env_idx]:
                        length = episode_lengths[env_idx].item()
                        completed_episodes.append({
                            'reward': episode_rewards[env_idx].item(),
                            'length': length,
                            'env_reward': env_rewards_sum[env_idx].item(),
                            'amp_reward': amp_rewards_sum[env_idx].item(),
                            'early_termination': length < self.config['max_episode_steps'] * 0.8  # Flag early terminations
                        })
                
                # Reset completed environments
                episode_rewards[dones] = 0
                episode_lengths[dones] = 0
                env_rewards_sum[dones] = 0
                amp_rewards_sum[dones] = 0
            
            obs = next_obs
        
        # Compute final values for GAE
        with torch.no_grad():
            _, next_values = self.policy(obs)
        
        # Compute advantages and returns
        advantages, returns = self.buffer.compute_gae(next_values, 
                                                     gamma=self.config.get('gamma', 0.99),
                                                     gae_lambda=self.config.get('gae_lambda', 0.95))
        
        # Prepare batch
        batch = self.buffer.get_batch(advantages, returns)
        
        # Collection metrics
        if completed_episodes:
            early_terminations = sum(ep.get('early_termination', False) for ep in completed_episodes)
            metrics = {
                'episode_reward_mean': np.mean([ep['reward'] for ep in completed_episodes]),
                'episode_length_mean': np.mean([ep['length'] for ep in completed_episodes]),
                'amp_reward_mean': np.mean([ep['amp_reward'] for ep in completed_episodes]),
                'env_reward_mean': np.mean([ep['env_reward'] for ep in completed_episodes]),
                'completed_episodes': len(completed_episodes),
                'early_terminations': early_terminations
            }
        else:
            # Use current episode progress if no completions
            metrics = {
                'episode_reward_mean': episode_rewards.mean().item(),
                'episode_length_mean': episode_lengths.mean().item(),
                'amp_reward_mean': amp_rewards_sum.mean().item() / max(episode_lengths.mean().item(), 1),
                'env_reward_mean': env_rewards_sum.mean().item() / max(episode_lengths.mean().item(), 1),
                'completed_episodes': 0,
                'early_terminations': 0
            }
        
        return batch, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step: update policy and discriminator with LocoMujoco schedule"""
        
        # LocoMujoco-style multiple epoch updates
        update_epochs = self.config.get('update_epochs', 4)
        
        ppo_metrics_list = []
        for epoch in range(update_epochs):
            ppo_metrics = self.ppo_trainer.update_policy(
                obs=batch['observations'],
                actions=batch['actions'],
                old_log_probs=batch['old_log_probs'],
                advantages=batch['advantages'],
                returns=batch['returns']
            )
            ppo_metrics_list.append(ppo_metrics)
        
        # Average metrics across epochs
        ppo_metrics = {}
        for key in ppo_metrics_list[0].keys():
            ppo_metrics[key] = sum(m[key] for m in ppo_metrics_list) / len(ppo_metrics_list)
        
        # Update discriminator only if not overfitting (key fix!)
        recent_expert_acc = getattr(self, '_last_expert_acc', 0.5)
        if recent_expert_acc < 0.90:  # Prevent discriminator overfitting
            disc_metrics = self.amp_integration.train_discriminator_step(batch['observations'])
            self._last_expert_acc = disc_metrics.get('expert_accuracy', 0.5)
        else:
            # Skip discriminator training, use last metrics
            disc_metrics = getattr(self, '_last_disc_metrics', {
                'discriminator_loss': 0.0, 'expert_accuracy': recent_expert_acc, 
                'policy_accuracy': 0.5, 'expert_score_mean': 1.0, 'policy_score_mean': -1.0
            })
            print(f"   ⚠️ Skipping discriminator training (expert_acc={recent_expert_acc:.3f})")
        
        self._last_disc_metrics = disc_metrics
        
        # Combine metrics
        combined_metrics = {**ppo_metrics}
        combined_metrics.update({f"disc_{k}": v for k, v in disc_metrics.items()})
        
        return combined_metrics
    
    def train(self, num_iterations: int):
        """Main training loop"""
        print(f"\n🎯 Starting {self.behavior.upper()} Training")
        print(f"   Iterations: {num_iterations}")
        print(f"   Environments: {self.config['num_envs']}")
        print(f"   Episode length: {self.config['episode_length_s']}s")
        print("=" * 70)
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            iter_start_time = time.time()
            
            # Collect trajectories and train
            batch, collection_metrics = self.collect_trajectories()
            training_metrics = self.train_step(batch)
            
            iter_time = time.time() - iter_start_time
            
            # Update metrics
            self.metrics_tracker.update(iteration, collection_metrics, training_metrics, iter_time)
            
            # Log progress
            if iteration % self.config['log_interval'] == 0:
                self._log_progress(iteration, num_iterations, collection_metrics, training_metrics, iter_time)
            
            # Save checkpoints
            if iteration % self.config.get('checkpoint_interval', 100) == 0 and iteration > 0:
                self._save_checkpoint(iteration)
            
            # Plot progress
            if iteration % (self.config['log_interval'] * 4) == 0 and iteration > 0:
                self._plot_training_progress(iteration)
            
            # Save best model
            current_reward = collection_metrics['episode_reward_mean']
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self._save_best_model(iteration, current_reward)
            
            self.iteration = iteration
        
        total_time = time.time() - start_time
        self._print_final_results(num_iterations, total_time)
        self._plot_training_progress(iteration, save=True)
        
        return self
    
    def _log_progress(self, iteration: int, total_iterations: int, collection_metrics: Dict, 
                     training_metrics: Dict, iter_time: float):
        """Log training progress"""
        progress = (iteration / total_iterations) * 100
        stats = self.metrics_tracker.get_recent_stats()
        
        print(f"\n🏃 Iteration {iteration:4d}/{total_iterations} ({progress:5.1f}%)")
        print(f"   ⏱️  Time: {iter_time:.2f}s (avg: {stats['iteration_time_mean']:.2f}s)")
        print(f"   🎯 Episode Reward: {collection_metrics['episode_reward_mean']:8.3f} (σ: {stats['episode_reward_std']:.3f})")
        print(f"   📏 Episode Length: {collection_metrics['episode_length_mean']:8.1f}")
        print(f"   🎭 AMP Reward:     {collection_metrics['amp_reward_mean']:8.3f}")
        print(f"   🌍 Env Reward:     {collection_metrics['env_reward_mean']:8.3f}")
        print(f"   🧠 Policy Loss:    {training_metrics['policy_loss']:8.4f}")
        print(f"   🤖 Expert Acc:     {training_metrics['disc_expert_accuracy']:8.3f}")
        print(f"   👤 Policy Acc:     {training_metrics['disc_policy_accuracy']:8.3f}")
        print(f"   📦 Episodes:       {collection_metrics['completed_episodes']}")
        print("-" * 60)
    
    def _plot_training_progress(self, iteration: int, save: bool = False):
        """Plot training progress"""
        if len(self.metrics_tracker.full_history['iterations']) < 10:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.behavior.title()} Imitation Learning Progress - Iteration {iteration}', fontsize=16)
        
        iterations = self.metrics_tracker.full_history['iterations']
        
        # Episode metrics
        axes[0, 0].plot(iterations, self.metrics_tracker.full_history['episode_rewards'], 'b-', linewidth=2)
        axes[0, 0].set_title('Episode Reward', fontsize=14)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(iterations, self.metrics_tracker.full_history['episode_lengths'], 'g-', linewidth=2)
        axes[0, 1].set_title('Episode Length', fontsize=14)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward breakdown
        axes[0, 2].plot(iterations, self.metrics_tracker.full_history['amp_rewards'], 'r-', linewidth=2, label='AMP')
        axes[0, 2].plot(iterations, self.metrics_tracker.full_history['env_rewards'], 'orange', linewidth=2, label='Env')
        axes[0, 2].set_title('Reward Components', fontsize=14)
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Reward')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Training losses
        axes[1, 0].plot(iterations, self.metrics_tracker.full_history['policy_losses'], 'purple', linewidth=2)
        axes[1, 0].set_title('Policy Loss', fontsize=14)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(iterations, self.metrics_tracker.full_history['discriminator_losses'], 'brown', linewidth=2)
        axes[1, 1].set_title('Discriminator Loss', fontsize=14)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Discriminator accuracies
        axes[1, 2].plot(iterations, self.metrics_tracker.full_history['expert_accuracies'], 'cyan', linewidth=2, label='Expert')
        axes[1, 2].plot(iterations, self.metrics_tracker.full_history['policy_accuracies'], 'magenta', linewidth=2, label='Policy')
        axes[1, 2].set_title('Discriminator Accuracy', fontsize=14)
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f'training_progress_final.png'), dpi=300, bbox_inches='tight')
            print(f"   📊 Final training plot saved to {self.save_dir}/training_progress_final.png")
        else:
            plt.savefig(os.path.join(self.save_dir, f'progress_iter_{iteration}.png'), dpi=150)
        
        plt.close()
    
    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'behavior': self.behavior,
            'config': self.config,
            'policy_state_dict': self.policy.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'discriminator_state_dict': self.amp_integration.discriminator.state_dict(),
            'disc_optimizer_state_dict': self.amp_integration.trainer.optimizer.state_dict(),
            'metrics_history': self.metrics_tracker.full_history,
            'best_reward': self.best_reward
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_iter_{iteration}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if iteration % 500 == 0:
            print(f"   💾 Checkpoint saved: iteration {iteration}")
    
    def _save_best_model(self, iteration: int, reward: float):
        """Save best performing model"""
        best_model = {
            'iteration': iteration,
            'behavior': self.behavior,
            'reward': reward,
            'config': self.config,
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.amp_integration.discriminator.state_dict()
        }
        
        best_path = os.path.join(self.save_dir, 'best_model.pt')
        torch.save(best_model, best_path)
        print(f"   🏆 New best model! Reward: {reward:.3f}")
    
    def _print_final_results(self, total_iterations: int, total_time: float):
        """Print final training results"""
        stats = self.metrics_tracker.get_recent_stats()
        hours = total_time / 3600
        
        print("\n" + "=" * 70)
        print(f"🎉 {self.behavior.upper()} IMITATION LEARNING COMPLETED!")
        print("=" * 70)
        print(f"   Behavior: {self.behavior}")
        print(f"   Total iterations: {total_iterations}")
        print(f"   Training time: {hours:.2f} hours")
        print(f"   Final episode reward: {stats['episode_reward_mean']:.3f}")
        print(f"   Final episode length: {stats['episode_length_mean']:.1f} steps")
        print(f"   Best reward achieved: {self.best_reward:.3f}")
        print(f"   Final expert accuracy: {stats['expert_accuracy_mean']:.3f}")
        print(f"   Final policy accuracy: {stats['policy_accuracy_mean']:.3f}")
        print(f"   Models saved to: {self.save_dir}")
        print("=" * 70)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint['ppo_optimizer_state_dict'])
        self.amp_integration.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.amp_integration.trainer.optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        self.iteration = checkpoint['iteration']
        self.best_reward = checkpoint['best_reward']
        self.metrics_tracker.full_history = checkpoint['metrics_history']
        
        print(f"✅ Checkpoint loaded: iteration {self.iteration}, best reward {self.best_reward:.3f}")


def create_behavior_config(behavior: str = "walk") -> Dict:
    """Create optimized configuration for specific behavior"""
    
    # Base configuration with improved settings
    base_config = {
        # Environment
        'num_envs': 128,
        'episode_length_s': 15.0,
        'dt': 0.01,
        'show_viewer': False,
        'use_box_feet': True,
        
        # Training 
        'max_episode_steps': 15,
        'env_reward_weight': 0.5,  # LocoMujoco: proportion_env_reward: 0.5
        'gamma': 0.99,             # LocoMujoco: gamma: 0.99
        'gae_lambda': 0.95,        # LocoMujoco: gae_lambda: 0.95
        'update_epochs': 4,        # LocoMujoco: update_epochs: 4
        'num_minibatches': 8,      # Adapted from LocoMujoco: 32 (scaled for fewer envs)
        
        # Logging and checkpointing
        'log_interval': 10,
        'checkpoint_interval': 100,
        'metrics_window': 100,
        
        # Policy network - EXACT LOCOMUJOCO CONFIG
        'policy': {
            'hidden_layers': [512, 256],     # LocoMujoco: [512, 256]
            'activation': 'tanh',
            'learning_rate': 6e-5,           # LocoMujoco: 6e-5
            'clip_epsilon': 0.1,             # LocoMujoco: 0.1 (was 0.2)
            'value_coeff': 0.5,
            'entropy_coeff': 0.0,            # LocoMujoco: 0.0 (was 0.01)
            'use_obs_norm': True,
            'max_grad_norm': 0.75            # LocoMujoco: 0.75
        },
        
        # Discriminator - EXACT LOCOMUJOCO CONFIG
        'discriminator': {
            'hidden_layers': [512, 256],     # LocoMujoco: [512, 256]
            'activation': 'tanh',
            'learning_rate': 5e-6,           # LocoMujoco: 5e-5
            'use_running_norm': True
        }
    }
    
    # Behavior-specific adjustments
    if behavior == "walk":
        base_config.update({
            'episode_length_s': 15.0,
            'env_reward_weight': 0.1
        })
    elif behavior == "run":
        base_config.update({
            'episode_length_s': 12.0,
            'max_episode_steps': 600,
            'env_reward_weight': 0.15
        })
    elif behavior == "squat":
        base_config.update({
            'episode_length_s': 10.0,
            'max_episode_steps': 500,
            'env_reward_weight': 0.2
        })
    
    return base_config


def main():
    """Main training function with behavior selection"""
    print("🤖 Genesis Skeleton Comprehensive Imitation Learning")
    print("=" * 70)
    
    # Behavior selection
    print("Available options:")
    print("1. walk - Natural human walking (LocoMujoco)")
    print("2. run - Running/jogging motion (LocoMujoco)") 
    print("3. squat - Squatting exercise (LocoMujoco)")
    print("4. custom - Load custom NPZ trajectory file (preprocessed BVH)")
    
    choice = input("Select option (1/2/3/4 or walk/run/squat/custom): ").strip().lower()
    
    behavior_map = {"1": "walk", "2": "run", "3": "squat", "4": "custom"}
    if choice in behavior_map:
        behavior = behavior_map[choice]
    elif choice in ["walk", "run", "squat", "custom"]:
        behavior = choice
    else:
        print("Invalid choice, defaulting to 'walk'")
        behavior = "walk"
    
    # Handle custom NPZ file
    if behavior == "custom":
        npz_path = input("Enter path to NPZ trajectory file: ").strip()
        if not npz_path.endswith('.npz'):
            npz_path += '.npz'
        if not os.path.exists(npz_path):
            print(f"❌ File not found: {npz_path}")
            print("Defaulting to 'walk' behavior")
            behavior = "walk"
        else:
            behavior = npz_path
            print(f"🎯 Selected custom trajectory: {npz_path}")
    else:
        print(f"\n🎯 Selected behavior: {behavior.upper()}")
    
    # Training configuration
    print("\nTraining scale:")
    print("1. Quick test (16 envs, 100 iterations)")
    print("2. Medium training (64 envs, 1000 iterations)")
    print("3. Full scale (128 envs, 3000 iterations)")
    
    scale_choice = input("Select scale (1/2/3): ").strip()
    
    config = create_behavior_config(behavior)
    
    if scale_choice == "1":
        config.update({'num_envs': 16, 'log_interval': 5})
        iterations = 100
        print("⚡ Quick test configuration")
    elif scale_choice == "3":
        config.update({'num_envs': 2048})
        iterations = 3000
        print("🚀 Full scale configuration")
    else:
        iterations = 1000
        print("🎯 Medium training configuration")
    
    # Visualization option
    vis_choice = input("Enable visualization during training? (y/n): ").strip().lower()
    config['show_viewer'] = vis_choice == 'y'
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"imitation_{behavior}_{timestamp}"
    
    print(f"\nFinal Configuration:")
    print(f"   Behavior: {behavior}")
    print(f"   Environments: {config['num_envs']}")
    print(f"   Episode Length: {config['episode_length_s']}s")
    print(f"   Max Iterations: {iterations}")
    print(f"   Visualization: {config['show_viewer']}")
    print(f"   Save Directory: {save_dir}")
    
    input("\nPress Enter to start training...")
    
    try:
        # Initialize and run training
        trainer = ComprehensiveImitationTrainer(config, save_dir, behavior)
        trainer.train(num_iterations=iterations)
        
        print(f"\n🎊 Training Complete!")
        print(f"📁 Results saved to: {save_dir}")
        print(f"📊 View training_progress_final.png for performance plots")
        print(f"🏆 Load best_model.pt to test your trained {behavior} agent")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()