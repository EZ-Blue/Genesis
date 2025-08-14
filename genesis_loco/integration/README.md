# Genesis Skeleton Imitation Learning

A comprehensive, efficient imitation learning pipeline for training humanoid behaviors using Genesis physics simulation and LocoMujoco expert trajectories.

## 🚀 Quick Start

1. **Test the pipeline:**
   ```bash
   python test_comprehensive_trainer.py
   ```

2. **Run full training:**
   ```bash
   python comprehensive_imitation_trainer.py
   ```

3. **Choose your behavior:** walking, running, or squatting
4. **Select training scale:** quick test, medium, or full scale
5. **Watch your agent learn!**

## 📁 File Overview

### Core Components (Refactored)
- **`../environments/skeleton_humanoid.py`** - Consolidated Genesis skeleton environment
- **`data_bridge.py`** - Simple LocoMujoco trajectory interface  
- **`amp_integration.py`** - AMP discriminator integration
- **`amp_discriminator.py`** - Efficient discriminator network
- **`simple_policy.py`** - PPO policy network for skeleton control

### Training Scripts
- **`comprehensive_imitation_trainer.py`** - 🆕 **MAIN TRAINING SCRIPT**
  - Multi-behavior support (walk/run/squat)
  - Efficient training pipeline with PPO + AMP
  - Comprehensive metrics and visualization
  - Model checkpointing and resumption

- **`test_comprehensive_trainer.py`** - Quick validation test
- **`simple_trainer.py`** - Basic training loop (updated for refactored components)
- **`train_walking.py`** - ⚠️ DEPRECATED (use comprehensive trainer instead)

### Legacy/Reference Files
- **`test_walking.py`** - Basic walking test
- **`simple_policy.py`** - Policy network implementation
- **`verify_trajectory.py`** - Trajectory compatibility checker
- **`debug_control.py`** - Control debugging utilities

## 🎯 Supported Behaviors

### Walking (`walk`)
- Natural human walking gait
- 15-second episodes
- Optimized for stable bipedal locomotion

### Running (`run`) 
- Running/jogging motion
- 12-second episodes
- Higher-speed locomotion patterns

### Squatting (`squat`)
- Squatting exercise motion
- 10-second episodes
- Focused on vertical movement patterns

## 🔧 Configuration

The trainer uses behavior-specific configurations optimized for each motion type:

```python
config = {
    # Environment
    'num_envs': 64,              # Parallel environments
    'episode_length_s': 15.0,    # Episode duration
    'dt': 0.01,                  # Simulation timestep (100Hz)
    
    # Training
    'max_episode_steps': 750,    # Max steps per episode
    'env_reward_weight': 0.1,    # 10% env reward, 90% AMP reward
    
    # Policy network
    'policy': {
        'hidden_layers': [512, 256, 128],
        'learning_rate': 3e-4,
        'clip_epsilon': 0.2
    },
    
    # Discriminator
    'discriminator': {
        'hidden_layers': [512, 256],
        'learning_rate': 5e-5,
        'use_running_norm': True
    }
}
```

## 📊 Training Process

1. **Environment Setup:** Genesis skeleton humanoid with proper joint mapping
2. **Expert Data Loading:** LocoMujoco trajectories converted to Genesis format
3. **AMP Discriminator Training:** Distinguishes expert vs policy behavior
4. **PPO Policy Training:** Learns actions to maximize AMP + environment rewards
5. **Metrics Tracking:** Real-time performance monitoring and visualization

## 📈 Monitoring

The trainer provides comprehensive monitoring:

- **Real-time logging:** Episode rewards, lengths, discriminator accuracy
- **Progress plots:** Training curves saved every few iterations
- **Checkpointing:** Regular model saves for resumption
- **Best model tracking:** Automatically saves best-performing models

## 🏆 Results

After training, you'll find:

- **`best_model.pt`** - Best performing model
- **`checkpoint_iter_X.pt`** - Regular training checkpoints
- **`training_progress_final.png`** - Final training curves
- **`config.json`** - Training configuration used

## 🔍 Troubleshooting

### Common Issues

1. **"Failed to load trajectory"**
   - Ensure LocoMujoco is properly installed
   - Check that `/home/ez/Documents/loco-mujoco` path is correct

2. **Genesis initialization errors**
   - Make sure CUDA is available if using GPU backend
   - Try reducing `num_envs` if running out of memory

3. **Import errors**
   - Verify all refactored components are in correct locations
   - Run `test_comprehensive_trainer.py` to validate setup

### Performance Tips

- **More environments:** Increase `num_envs` for more stable training
- **Longer episodes:** Increase `episode_length_s` for complex behaviors  
- **Adjust reward mixing:** Tune `env_reward_weight` (0.0 = pure AMP, 1.0 = pure environment)

## 🧪 Testing

Before full training, validate your setup:

```bash
# Test all components
python test_comprehensive_trainer.py

# Quick training test (100 iterations)
python comprehensive_imitation_trainer.py
# Select: behavior=walk, scale=1 (quick test)
```

## 🚀 Advanced Usage

### Custom Behaviors

To add new behaviors:

1. Add trajectory data to LocoMujoco
2. Update `create_behavior_config()` in `comprehensive_imitation_trainer.py`
3. Adjust episode length and reward weights as needed

### Multi-Behavior Training

Train multiple behaviors sequentially:

```bash
# Train walking
python comprehensive_imitation_trainer.py  # Choose walk

# Train running  
python comprehensive_imitation_trainer.py  # Choose run

# Train squatting
python comprehensive_imitation_trainer.py  # Choose squat
```

### Resume Training

Load a checkpoint to resume training:

```python
trainer = ComprehensiveImitationTrainer(config, save_dir, behavior)
trainer.load_checkpoint("path/to/checkpoint_iter_1000.pt")
trainer.train(num_iterations=2000)  # Continue from iteration 1000
```

## 📋 Architecture Summary

```
Genesis Skeleton Environment
├── skeleton_humanoid.py (consolidated environment)
├── LocoMujoco Integration
│   ├── data_bridge.py (trajectory interface)
│   └── Expert trajectory data
├── AMP Training
│   ├── amp_integration.py (main interface)
│   └── amp_discriminator.py (neural network)
├── Policy Learning
│   ├── simple_policy.py (PPO implementation)
│   └── comprehensive_imitation_trainer.py (training loop)
└── Results
    ├── Model checkpoints
    ├── Training metrics
    └── Visualization plots
```

This refactored pipeline provides a clean, efficient, and maintainable foundation for imitation learning with Genesis and LocoMujoco! 🎉