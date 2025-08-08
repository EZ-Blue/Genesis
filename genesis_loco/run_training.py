#!/usr/bin/env python3
"""
Quick Training Script for Genesis-LocoMujoco Imitation Learning

Simplified interface for running common training tasks without command line arguments.
Perfect for testing and interactive development.
"""

import sys
import os
import yaml
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_imitation_trainer import UnifiedImitationTrainer, create_unified_config


def load_task_config(config_name: str) -> dict:
    """Load task configuration from YAML file"""
    config_file = Path(__file__).parent / "task_configs.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Task configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    if config_name not in all_configs:
        available = list(all_configs.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    
    return all_configs[config_name]


def run_walk_training():
    """Train walking behavior only"""
    print("🚶‍♂️ Starting walk-only training...")
    
    config = load_task_config("walk_only")
    trainer = UnifiedImitationTrainer(config)
    trainer.train(num_iterations=100)
    
    print("✅ Walk training completed!")
    return trainer


def run_run_training():
    """Train running behavior only"""
    print("🏃‍♂️ Starting run-only training...")
    
    config = load_task_config("run_only")
    trainer = UnifiedImitationTrainer(config)
    trainer.train(num_iterations=100)
    
    print("✅ Run training completed!")
    return trainer


def run_squat_training():
    """Train squatting behavior only"""
    print("🏋️‍♂️ Starting squat-only training...")
    
    config = load_task_config("squat_only")
    trainer = UnifiedImitationTrainer(config)
    trainer.train(num_iterations=75)
    
    print("✅ Squat training completed!")
    return trainer


def run_basic_locomotion():
    """Train basic locomotion (walking)"""
    print("🚶‍♂️ Starting basic locomotion training...")
    
    config = load_task_config("locomotion_basic")
    trainer = UnifiedImitationTrainer(config)
    trainer.train(num_iterations=200)
    
    print("✅ Basic locomotion training completed!")
    return trainer


def run_advanced_locomotion():
    """Train advanced locomotion (walk + run + variations)"""
    print("🏃‍♂️ Starting advanced locomotion training...")
    
    config = load_task_config("locomotion_advanced")
    trainer = UnifiedImitationTrainer(config)
    trainer.train(num_iterations=500)
    
    print("✅ Advanced locomotion training completed!")
    return trainer


def run_exercise_training():
    """Train exercise movements"""
    print("🏋️‍♂️ Starting exercise training...")
    
    config = load_task_config("exercise_training")
    trainer = UnifiedImitationTrainer(config)
    trainer.train(num_iterations=300)
    
    print("✅ Exercise training completed!")
    return trainer


def run_debug_training():
    """Quick debug training session"""
    print("🐛 Starting debug training...")
    
    config = load_task_config("debug_quick")
    trainer = UnifiedImitationTrainer(config)
    trainer.train(num_iterations=10)
    
    print("✅ Debug training completed!")
    return trainer


def run_custom_training(datasets, num_iterations=200, num_envs=32, show_viewer=False):
    """Run custom training with specified datasets"""
    print(f"🎯 Starting custom training with datasets: {datasets}")
    
    config = create_unified_config(
        task_name="custom_training",
        datasets=datasets,
        environment={'num_envs': num_envs, 'show_viewer': show_viewer},
        training={'log_interval': 5}
    )
    
    trainer = UnifiedImitationTrainer(config)
    trainer.train(num_iterations=num_iterations)
    
    print("✅ Custom training completed!")
    return trainer


def test_pipeline():
    """Test the complete pipeline with minimal settings"""
    print("🧪 Testing complete pipeline...")
    
    try:
        config = load_task_config("debug_quick")
        trainer = UnifiedImitationTrainer(config)
        
        # Test trajectory collection
        print("   Testing trajectory collection...")
        batch, metrics = trainer.collect_trajectories()
        print(f"     ✓ Collected batch: {batch['observations'].shape}")
        print(f"     ✓ Mean reward: {metrics['episode_reward_mean']:.3f}")
        
        # Test training step
        print("   Testing training step...")
        train_metrics = trainer.train_step(batch)
        print(f"     ✓ Policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"     ✓ Discriminator loss: {train_metrics['disc_discriminator_loss']:.4f}")
        
        print("✅ Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_menu():
    """Interactive menu for selecting training tasks"""
    print("\n🎯 Genesis-LocoMujoco Imitation Learning Trainer")
    print("=" * 50)
    print("Available training options:")
    print("1. Test Pipeline (quick verification)")
    print("2. Walk Only (simple walking)")
    print("3. Run Only (simple running)")  
    print("4. Squat Only (exercise movement)")
    print("5. Basic Locomotion (walking focus)")
    print("6. Advanced Locomotion (walk + run + variations)")
    print("7. Exercise Training (squats, lunges, etc.)")
    print("8. Custom Training (specify datasets)")
    print("9. Debug Training (very quick test)")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (0-9): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                test_pipeline()
            elif choice == '2':
                run_walk_training()
            elif choice == '3':
                run_run_training()
            elif choice == '4':
                run_squat_training()
            elif choice == '5':
                run_basic_locomotion()
            elif choice == '6':
                run_advanced_locomotion()
            elif choice == '7':
                run_exercise_training()
            elif choice == '8':
                datasets = input("Enter datasets (comma-separated): ").strip().split(',')
                datasets = [d.strip() for d in datasets if d.strip()]
                iterations = int(input("Number of iterations (default 200): ").strip() or "200")
                show_viewer = input("Show viewer? (y/n, default n): ").strip().lower() == 'y'
                run_custom_training(datasets, iterations, show_viewer=show_viewer)
            elif choice == '9':
                run_debug_training()
            else:
                print("❌ Invalid option. Please select 0-9.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        task = sys.argv[1].lower()
        
        task_functions = {
            'test': test_pipeline,
            'walk': run_walk_training,
            'run': run_run_training,
            'squat': run_squat_training,
            'basic': run_basic_locomotion,
            'advanced': run_advanced_locomotion,
            'exercise': run_exercise_training,
            'debug': run_debug_training
        }
        
        if task in task_functions:
            print(f"🚀 Running {task} training...")
            task_functions[task]()
        else:
            print(f"❌ Unknown task: {task}")
            print(f"Available tasks: {list(task_functions.keys())}")
            print("Or run without arguments for interactive menu")
    else:
        # Interactive mode
        interactive_menu()