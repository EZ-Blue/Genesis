"""
Unified Imitation Learning Trainer for Genesis-LocoMujoco Integration

This is a comprehensive training script that enables imitation learning for various motion tasks
like walking, running, and squatting using the Genesis physics simulator with LocoMujoco datasets.

Features:
- Supports multiple LocoMujoco datasets (walk, run, squat, etc.)
- AMP (Adversarial Motion Priors) discriminator training
- PPO policy optimization
- Genesis physics integration
- Configurable training parameters
- Progress tracking and visualization
"""

import torch
import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Optional
import argparse
import yaml
import json
from pathlib import Path

# Fix import paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integration.simple_trainer import SimpleImitationTrainer, create_default_config
from integration.amp_integration import AMPGenesisIntegration
from integration.data_bridge import LocoMujocoDataBridge


class UnifiedImitationTrainer:
    """
    Unified trainer supporting multiple LocoMujoco motion tasks
    
    Supports training on various motion patterns:
    - Locomotion: walk, run, jump
    - Exercises: squat, pushup  
    - Dance: various dance motions
    - Custom: user-defined trajectories
    """
    
    # Available LocoMujoco datasets by category
    AVAILABLE_DATASETS = {
        'locomotion': [
            'walk', 'run', 'jog', 'walk_backwards', 'walk_sideways_left', 'walk_sideways_right',
            'run_backwards', 'skip', 'gallop', 'hop_left', 'hop_right', 'hop_both'
        ],
        'exercise': [
            'squat', 'lunge_left', 'lunge_right', 'pushup', 'situp', 'burpee',
            'jumping_jacks', 'mountain_climber', 'plank', 'leg_raise'
        ],
        'dance': [
            'dance1', 'dance2', 'dance3', 'dance4', 'dance5',
            'breakdance', 'salsa', 'ballet', 'hip_hop'
        ],
        'martial_arts': [
            'punch_left', 'punch_right', 'kick_left', 'kick_right',
            'block_high', 'block_low', 'kata1', 'kata2'
        ]
    }
    
    def __init__(self, config: Dict):
        """
        Initialize unified trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_name = config['task']['name']
        self.datasets = config['task']['datasets']
        
        print("🚀 Unified Imitation Learning Trainer")
        print(f"   Task: {self.task_name}")
        print(f"   Datasets: {self.datasets}")
        print(f"   Device: {self.device}")
        
        # Initialize components
        self._setup_environment()
        self._setup_multi_dataset_bridge()
        self._setup_amp_integration()
        self._setup_policy()
        self._setup_training_components()
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'discriminator_losses': [],
            'expert_accuracies': [],
            'policy_accuracies': [],
            'amp_rewards': [],
            'env_rewards': []
        }
        
        print("✅ Unified trainer initialization complete!")
    
    def _setup_environment(self):
        """Setup Genesis skeleton environment"""
        print("   Setting up Genesis environment...")
        
        import genesis as gs
        gs.init(backend=gs.gpu)
        
        # Import skeleton environment
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from environments.skeleton_humanoid import SkeletonHumanoidEnv
        
        self.env = SkeletonHumanoidEnv(
            num_envs=self.config['environment']['num_envs'],
            episode_length_s=self.config['environment']['episode_length_s'],
            dt=self.config['environment']['dt'],
            show_viewer=self.config['environment'].get('show_viewer', False),
            use_trajectory_control=True,  # Enable smooth trajectory following
            use_box_feet=True  # Enable stable ground contact
        )
        
        self.obs_dim = self.env.num_observations
        self.action_dim = self.env.num_actions
        
        print(f"     ✓ {self.config['environment']['num_envs']} environments")
        print(f"     ✓ Observation dim: {self.obs_dim}")
        print(f"     ✓ Action dim: {self.action_dim}")
    
    def _setup_multi_dataset_bridge(self):
        """Setup data bridge supporting multiple datasets"""
        print("   Setting up multi-dataset bridge...")
        
        self.data_bridges = {}
        self.trajectory_data = {}
        
        for dataset_name in self.datasets:
            print(f"     Loading dataset: {dataset_name}")
            
            # Create data bridge for this dataset
            bridge = LocoMujocoDataBridge(self.env)
            
            # Load trajectory
            success, _ = bridge.load_trajectory(dataset_name)
            if not success:
                print(f"     ⚠️ Failed to load {dataset_name}, skipping...")
                continue
            
            # Build joint mapping
            success, mapping_info = bridge.build_joint_mapping()
            if not success:
                print(f"     ⚠️ Failed to build joint mapping for {dataset_name}")
                continue
            
            # Convert to Genesis format
            success, traj_data = bridge.convert_to_genesis_format()
            if not success:
                print(f"     ⚠️ Failed to convert {dataset_name} trajectory")
                continue
            
            self.data_bridges[dataset_name] = bridge
            self.trajectory_data[dataset_name] = traj_data
            
            print(f"       ✓ {dataset_name}: {traj_data['info']['timesteps']} timesteps")
            print(f"       ✓ Joint mapping: {mapping_info['match_percentage']:.1f}%")
        
        if not self.data_bridges:
            raise RuntimeError("No datasets loaded successfully!")
        
        print(f"     ✓ Successfully loaded {len(self.data_bridges)} datasets")
    
    def _setup_amp_integration(self):
        """Setup AMP discriminator with multi-dataset support"""
        print("   Setting up multi-dataset AMP integration...")
        
        # Use the first available dataset for initial setup
        primary_dataset = list(self.data_bridges.keys())[0]
        primary_bridge = self.data_bridges[primary_dataset]
        
        # Create AMP integration
        self.amp_integration = AMPGenesisIntegration(
            genesis_env=self.env,
            data_bridge=primary_bridge,
            discriminator_config=self.config['discriminator']
        )
        
        # Load expert data from all datasets
        self._load_multi_dataset_expert_data()
        
        print("     ✓ Multi-dataset AMP discriminator ready")
    
    def _load_multi_dataset_expert_data(self):
        """Load and combine expert data from all datasets"""
        print("     Loading expert data from all datasets...")
        
        all_expert_observations = []
        
        for dataset_name, bridge in self.data_bridges.items():
            trajectory_data = self.trajectory_data[dataset_name]
            
            # Generate expert observations by applying trajectory to Genesis
            expert_obs_list = []
            n_timesteps = min(500, trajectory_data['info']['timesteps'])  # Limit for memory
            
            print(f"       Generating {n_timesteps} expert observations from {dataset_name}...")
            
            for t in range(0, n_timesteps, 10):  # Sample every 10th timestep
                # Apply trajectory state to Genesis
                dof_pos = trajectory_data['dof_pos'][t:t+1]
                root_pos = trajectory_data['root_pos'][t:t+1]
                root_quat = trajectory_data['root_quat'][t:t+1]
                
                # Set Genesis state using proper DOF control
                env_ids = torch.tensor([0], device=self.device)
                if hasattr(bridge, 'genesis_dof_indices'):
                    self.env.robot.set_dofs_position(
                        dof_pos, 
                        dofs_idx_local=bridge.genesis_dof_indices, 
                        envs_idx=env_ids, 
                        zero_velocity=True
                    )
                else:
                    self.env.robot.set_dofs_position(dof_pos, envs_idx=env_ids, zero_velocity=True)
                
                self.env.robot.set_pos(root_pos, envs_idx=env_ids, zero_velocity=True)
                self.env.robot.set_quat(root_quat, envs_idx=env_ids, zero_velocity=True)
                
                # Update environment state and get observation
                self.env._update_robot_state()
                obs = self.env._get_observations()
                expert_obs_list.append(obs[0])
            
            dataset_expert_obs = torch.stack(expert_obs_list, dim=0)
            all_expert_observations.append(dataset_expert_obs)
            
            print(f"       ✓ {dataset_name}: {dataset_expert_obs.shape[0]} observations")
        
        # Combine all expert observations
        combined_expert_obs = torch.cat(all_expert_observations, dim=0)
        self.amp_integration.expert_observations = combined_expert_obs
        
        print(f"     ✓ Combined expert data: {combined_expert_obs.shape[0]} total observations")
    
    def _setup_policy(self):
        """Setup PPO policy network"""
        print("   Setting up policy network...")
        
        from integration.simple_policy import SkeletonPolicyNetwork, PPOTrainer
        
        self.policy = SkeletonPolicyNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_layers=self.config['policy']['hidden_layers'],
            activation=self.config['policy']['activation'],
            use_obs_norm=True
        )
        
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            learning_rate=self.config['policy']['learning_rate'],
            clip_epsilon=self.config['policy']['clip_epsilon'],
            device=self.device
        )
        
        param_count = sum(p.numel() for p in self.policy.parameters())
        print(f"     ✓ Policy network: {param_count} parameters")
    
    def _setup_training_components(self):
        """Setup training buffer and other components"""
        print("   Setting up training components...")
        
        from integration.simple_trainer import TrajectoryBuffer
        
        max_steps = int(self.config['environment']['episode_length_s'] / self.config['environment']['dt'])
        
        self.buffer = TrajectoryBuffer(
            num_envs=self.config['environment']['num_envs'],
            max_episode_length=max_steps,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        print(f"     ✓ Training buffer: {max_steps} max steps per episode")
    
    def collect_trajectories(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Collect trajectories using current policy with mixed rewards from all datasets
        
        Returns:
            Tuple of (batch_data, metrics)
        """
        self.policy.eval()
        self.buffer.clear()
        
        # Reset environment
        obs, _ = self.env.reset()
        episode_rewards = torch.zeros(self.config['environment']['num_envs'], device=self.device)
        episode_lengths = torch.zeros(self.config['environment']['num_envs'], device=self.device)
        
        amp_rewards_total = 0.0
        env_rewards_total = 0.0
        steps_collected = 0
        
        # Collect steps
        for step in range(self.config['training']['max_episode_steps']):
            # Sample actions from policy
            with torch.no_grad():
                actions, log_probs, values = self.policy.sample_actions(obs)
            
            # Environment step
            next_obs, env_rewards, dones, _ = self.env.step(actions)
            
            # Compute AMP rewards from combined expert data
            amp_rewards = self.amp_integration.compute_amp_rewards(obs)
            
            # Mix environment and AMP rewards
            env_weight = self.config['training']['env_reward_weight']
            mixed_rewards = env_weight * env_rewards + (1 - env_weight) * amp_rewards
            
            # Store in buffer
            self.buffer.add(obs, actions, mixed_rewards, values, log_probs, dones)
            
            # Update metrics
            episode_rewards += mixed_rewards
            episode_lengths += 1
            
            amp_rewards_total += amp_rewards.mean().item()
            env_rewards_total += env_rewards.mean().item()
            steps_collected += 1
            
            # Handle episode termination
            if dones.any():
                finished_episodes = dones.nonzero(as_tuple=False).squeeze(-1)
                for env_idx in finished_episodes:
                    self.training_history['episode_rewards'].append(episode_rewards[env_idx].item())
                    self.training_history['episode_lengths'].append(episode_lengths[env_idx].item())
                
                episode_rewards[dones] = 0
                episode_lengths[dones] = 0
            
            obs = next_obs
        
        # Compute final values for GAE
        with torch.no_grad():
            _, next_values = self.policy(obs)
        
        # Compute advantages and returns
        advantages, returns = self.buffer.compute_gae(next_values)
        
        # Prepare batch
        batch = self.buffer.get_batch(advantages, returns)
        
        # Collection metrics
        metrics = {
            'episode_reward_mean': np.mean(self.training_history['episode_rewards'][-self.config['environment']['num_envs']:]) if self.training_history['episode_rewards'] else 0.0,
            'episode_length_mean': np.mean(self.training_history['episode_lengths'][-self.config['environment']['num_envs']:]) if self.training_history['episode_lengths'] else 0.0,
            'amp_reward_mean': amp_rewards_total / steps_collected if steps_collected > 0 else 0.0,
            'env_reward_mean': env_rewards_total / steps_collected if steps_collected > 0 else 0.0
        }
        
        return batch, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step: update policy and discriminator
        
        Args:
            batch: Training batch data
            
        Returns:
            Training metrics
        """
        # Update policy with PPO
        ppo_metrics = self.ppo_trainer.update_policy(
            obs=batch['observations'],
            actions=batch['actions'],
            old_log_probs=batch['old_log_probs'],
            advantages=batch['advantages'],
            returns=batch['returns']
        )
        
        # Update discriminator with multi-dataset expert data
        disc_metrics = self.amp_integration.train_discriminator_step(batch['observations'])
        
        # Store metrics
        self.training_history['policy_losses'].append(ppo_metrics['policy_loss'])
        self.training_history['discriminator_losses'].append(disc_metrics['discriminator_loss'])
        self.training_history['expert_accuracies'].append(disc_metrics['expert_accuracy'])
        self.training_history['policy_accuracies'].append(disc_metrics['policy_accuracy'])
        
        # Combine metrics
        combined_metrics = {**ppo_metrics}
        combined_metrics.update({f"disc_{k}": v for k, v in disc_metrics.items()})
        
        return combined_metrics
    
    def train(self, num_iterations: int):
        """
        Main training loop
        
        Args:
            num_iterations: Number of training iterations
        """
        print(f"\n🎯 Starting unified training for {num_iterations} iterations")
        print(f"   Task: {self.task_name}")
        print(f"   Datasets: {len(self.datasets)} ({', '.join(self.datasets)})")
        print("=" * 60)
        
        best_reward = float('-inf')
        patience_count = 0
        
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Collect trajectories
            batch, collection_metrics = self.collect_trajectories()
            
            # Training step
            training_metrics = self.train_step(batch)
            
            # Compute iteration time
            iteration_time = time.time() - start_time
            
            # Track best performance
            current_reward = collection_metrics['episode_reward_mean']
            if current_reward > best_reward:
                best_reward = current_reward
                patience_count = 0
                
                # Save best model
                if iteration > 0:  # Skip saving on first iteration
                    self.save_checkpoint(f"best_model_{self.task_name}.pt")
            else:
                patience_count += 1
            
            # Log progress
            if iteration % self.config['training']['log_interval'] == 0:
                self._log_progress(iteration, num_iterations, iteration_time, 
                                 collection_metrics, training_metrics, best_reward)
            
            # Save periodic checkpoint
            if iteration % self.config['training']['save_interval'] == 0 and iteration > 0:
                self.save_checkpoint(f"checkpoint_{self.task_name}_{iteration:04d}.pt")
            
            # Early stopping check
            if (patience_count >= self.config['training']['patience'] and 
                iteration > self.config['training']['min_iterations']):
                print(f"\n🛑 Early stopping triggered after {patience_count} iterations without improvement")
                break
        
        # Final save
        self.save_checkpoint(f"final_model_{self.task_name}.pt")
        
        print("\n✅ Training completed!")
        print(f"   Best reward: {best_reward:.3f}")
        print(f"   Total iterations: {iteration + 1}")
    
    def _log_progress(self, iteration: int, total_iterations: int, iteration_time: float,
                     collection_metrics: Dict[str, float], training_metrics: Dict[str, float],
                     best_reward: float):
        """Log training progress"""
        print(f"\nIteration {iteration:4d}/{total_iterations}")
        print(f"Time: {iteration_time:.2f}s | Best Reward: {best_reward:.3f}")
        print(f"Episode Reward: {collection_metrics['episode_reward_mean']:.3f}")
        print(f"Episode Length: {collection_metrics['episode_length_mean']:.1f}")
        print(f"AMP Reward: {collection_metrics['amp_reward_mean']:.3f}")
        print(f"Env Reward: {collection_metrics['env_reward_mean']:.3f}")
        print(f"Policy Loss: {training_metrics['policy_loss']:.4f}")
        print(f"Disc Loss: {training_metrics['disc_discriminator_loss']:.4f}")
        print(f"Expert Acc: {training_metrics['disc_expert_accuracy']:.3f}")
        print(f"Policy Acc: {training_metrics['disc_policy_accuracy']:.3f}")
        print("-" * 40)
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'config': self.config,
            'training_history': self.training_history,
            'policy_state_dict': self.policy.state_dict(),
            'ppo_optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'discriminator_state_dict': self.amp_integration.discriminator.state_dict(),
            'disc_optimizer_state_dict': self.amp_integration.trainer.optimizer.state_dict(),
            'datasets': self.datasets,
            'task_name': self.task_name
        }
        
        filepath = checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"📄 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint['ppo_optimizer_state_dict'])
        self.amp_integration.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.amp_integration.trainer.optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        print(f"📂 Checkpoint loaded: {filepath}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained policy
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        print(f"\n🧪 Evaluating policy for {num_episodes} episodes...")
        
        self.policy.eval()
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                
                for step in range(self.config['training']['max_episode_steps']):
                    actions, _, _ = self.policy.sample_actions(obs)
                    obs, rewards, dones, _ = self.env.step(actions)
                    
                    episode_reward += rewards.mean().item()
                    episode_length += 1
                    
                    if dones.any():
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Count as success if episode lasted reasonable time and got positive reward
                if episode_length > 50 and episode_reward > 0:
                    success_count += 1
        
        metrics = {
            'eval_reward_mean': np.mean(episode_rewards),
            'eval_reward_std': np.std(episode_rewards),
            'eval_length_mean': np.mean(episode_lengths),
            'eval_length_std': np.std(episode_lengths),
            'success_rate': success_count / num_episodes
        }
        
        print(f"   Mean Reward: {metrics['eval_reward_mean']:.3f} ± {metrics['eval_reward_std']:.3f}")
        print(f"   Mean Length: {metrics['eval_length_mean']:.1f} ± {metrics['eval_length_std']:.1f}")
        print(f"   Success Rate: {metrics['success_rate']:.1%}")
        
        return metrics


def create_unified_config(task_name: str, datasets: List[str], **overrides) -> Dict:
    """
    Create unified training configuration
    
    Args:
        task_name: Name of the training task
        datasets: List of datasets to train on
        **overrides: Configuration overrides
        
    Returns:
        Configuration dictionary
    """
    config = {
        'task': {
            'name': task_name,
            'datasets': datasets
        },
        
        'environment': {
            'num_envs': 32,
            'episode_length_s': 15.0,
            'dt': 0.02,  # 50Hz
            'show_viewer': False
        },
        
        'training': {
            'max_episode_steps': 500,  # 10 seconds at 50Hz
            'env_reward_weight': 0.1,  # 10% env, 90% AMP for strong imitation
            'log_interval': 10,
            'save_interval': 100,
            'checkpoint_dir': 'checkpoints',
            'patience': 50,  # Early stopping patience
            'min_iterations': 100  # Minimum iterations before early stopping
        },
        
        'policy': {
            'hidden_layers': [512, 256, 128],
            'activation': 'tanh',
            'learning_rate': 3e-4,
            'clip_epsilon': 0.2
        },
        
        'discriminator': {
            'hidden_layers': [256, 128],
            'activation': 'tanh',
            'learning_rate': 5e-5,
            'use_running_norm': True
        }
    }
    
    # Apply overrides
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(config, overrides)
    return config


def main():
    """Main training function with command line interface"""
    parser = argparse.ArgumentParser(description='Unified Imitation Learning Trainer')
    
    parser.add_argument('--task', type=str, default='locomotion_basic',
                       help='Training task name')
    parser.add_argument('--datasets', nargs='+', default=['walk'],
                       help='List of datasets to train on')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--num-envs', type=int, default=32,
                       help='Number of parallel environments')
    parser.add_argument('--show-viewer', action='store_true',
                       help='Show Genesis viewer during training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate existing checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_unified_config(
            task_name=args.task,
            datasets=args.datasets,
            environment={'num_envs': args.num_envs, 'show_viewer': args.show_viewer}
        )
    
    print(f"🎯 Unified Imitation Learning Training")
    print(f"   Task: {config['task']['name']}")
    print(f"   Datasets: {config['task']['datasets']}")
    print(f"   Iterations: {args.iterations}")
    print(f"   Environments: {config['environment']['num_envs']}")
    
    try:
        # Create trainer
        trainer = UnifiedImitationTrainer(config)
        
        # Load checkpoint if specified
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        if args.eval_only:
            # Evaluation only
            metrics = trainer.evaluate(num_episodes=20)
            print("\n📊 Evaluation completed!")
        else:
            # Training
            trainer.train(num_iterations=args.iterations)
            
            # Final evaluation
            print("\n🧪 Running final evaluation...")
            metrics = trainer.evaluate(num_episodes=10)
        
        print("\n✅ All tasks completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())