# Genesis-LocoMujoco Imitation Learning

This repository contains a complete imitation learning pipeline that integrates Genesis physics simulation with LocoMujoco's motion capture datasets. Train humanoid skeletons to perform various locomotion and exercise tasks using AMP (Adversarial Motion Priors).

## 🚀 Quick Start

### Option 1: Interactive Training (Recommended)
```bash
python run_training.py
```
This launches an interactive menu where you can select from pre-configured training tasks.

### Option 2: Command Line Training
```bash
# Train walking only
python run_training.py walk

# Train running only  
python run_training.py run

# Train squatting only
python run_training.py squat

# Test the complete pipeline
python run_training.py test
```

### Option 3: Advanced Training
```bash
# Use the unified trainer directly
python unified_imitation_trainer.py --task locomotion_advanced --datasets walk run jog --iterations 500 --num-envs 64 --show-viewer
```

## 🎯 Available Tasks

### Locomotion Tasks
- **walk_only**: Basic walking behavior
- **run_only**: Running behavior  
- **squat_only**: Squatting exercise
- **locomotion_basic**: Walking with stability focus
- **locomotion_advanced**: Walk + run + variations
- **exercise_training**: Squats, lunges, jumping jacks

### Specialized Tasks
- **dance_training**: Artistic movement patterns
- **martial_arts**: Combat movements
- **high_performance**: Multi-dataset training for powerful hardware

## 📁 Project Structure

```
genesis_loco/
├── unified_imitation_trainer.py    # Main training system
├── run_training.py                 # Simple training interface  
├── task_configs.yaml               # Pre-configured training tasks
├── environments/
│   └── skeleton_humanoid.py        # Genesis humanoid environment
├── integration/
│   ├── data_bridge.py              # LocoMujoco data integration
│   ├── amp_integration.py          # AMP discriminator integration
│   ├── amp_discriminator.py        # AMP discriminator network
│   ├── simple_policy.py            # PPO policy network
│   ├── simple_trainer.py           # Basic training components
│   └── verify_trajectory.py        # Trajectory verification tools
└── checkpoints/                    # Saved model checkpoints
```

## 🛠 System Architecture

### Core Components

1. **Genesis Physics Environment** (`skeleton_humanoid.py`)
   - 27-DOF humanoid skeleton with torque control
   - Box feet for stable ground contact
   - Trajectory-optimized PD control
   - LocoMujoco-compatible action/observation spaces

2. **Data Bridge** (`data_bridge.py`) 
   - Loads LocoMujoco motion capture datasets
   - Converts trajectories to Genesis tensor format
   - Handles joint mapping between LocoMujoco and Genesis
   - Supports trajectory replay and verification

3. **AMP Integration** (`amp_integration.py`)
   - Adversarial Motion Priors discriminator
   - Multi-dataset expert data sampling
   - Mixed reward computation (environment + AMP)
   - PyTorch-based discriminator training

4. **Policy Network** (`simple_policy.py`)
   - PPO-based policy optimization
   - Observation normalization
   - Action distribution sampling
   - Value function estimation

5. **Unified Trainer** (`unified_imitation_trainer.py`)
   - Multi-dataset training support
   - Configurable task definitions
   - Progress tracking and checkpointing
   - Evaluation and testing utilities

### Training Pipeline

```
LocoMujoco Datasets → Data Bridge → Genesis Environment
                                         ↓
    AMP Discriminator ← Expert Data ← Trajectory Replay
           ↓
    Mixed Rewards → PPO Policy → Action Selection → Physics Simulation
```

## 🎮 Usage Examples

### Training a Walking Policy
```python
from unified_imitation_trainer import UnifiedImitationTrainer, create_unified_config

# Create configuration
config = create_unified_config(
    task_name="walk_training",
    datasets=["walk"],
    environment={'num_envs': 32, 'show_viewer': True}
)

# Train
trainer = UnifiedImitationTrainer(config)
trainer.train(num_iterations=200)

# Evaluate
metrics = trainer.evaluate(num_episodes=10)
print(f"Success rate: {metrics['success_rate']:.1%}")
```

### Multi-Dataset Training
```python
# Train on multiple locomotion patterns
config = create_unified_config(
    task_name="advanced_locomotion",
    datasets=["walk", "run", "jog", "walk_backwards"],
    environment={'num_envs': 64},
    training={'env_reward_weight': 0.1}  # Strong imitation focus
)

trainer = UnifiedImitationTrainer(config)
trainer.train(num_iterations=500)
```

### Custom Task Configuration
```python
# Define custom training task
custom_config = {
    'task': {
        'name': 'my_custom_task',
        'datasets': ['squat', 'lunge_left', 'lunge_right']
    },
    'environment': {
        'num_envs': 48,
        'episode_length_s': 8.0,
        'show_viewer': True
    },
    'training': {
        'max_episode_steps': 200,
        'env_reward_weight': 0.25,
        'log_interval': 5
    }
}

trainer = UnifiedImitationTrainer(custom_config)
trainer.train(num_iterations=300)
```

## ⚙️ Configuration

### Environment Settings
- `num_envs`: Number of parallel environments (16-128)
- `episode_length_s`: Episode duration in seconds
- `dt`: Physics timestep (0.01-0.02 recommended)
- `show_viewer`: Enable/disable visualization

### Training Parameters
- `max_episode_steps`: Maximum steps per episode
- `env_reward_weight`: Balance between task and imitation rewards (0.0-1.0)
- `log_interval`: Steps between progress logs
- `patience`: Early stopping patience

### Network Architecture
- `policy.hidden_layers`: Policy network layer sizes
- `discriminator.hidden_layers`: AMP discriminator layer sizes
- `learning_rate`: Optimizer learning rates

## 📊 Monitoring Training

The trainer automatically tracks:
- Episode rewards and lengths
- AMP discriminator accuracy
- Policy loss and value function loss
- Expert vs. policy classification accuracy
- Success rates and completion metrics

Checkpoints are automatically saved to `checkpoints/` directory.

## 🎯 Available LocoMujoco Datasets

### Locomotion
- `walk`, `run`, `jog`
- `walk_backwards`, `run_backwards`
- `walk_sideways_left`, `walk_sideways_right`
- `skip`, `gallop`, `hop_left`, `hop_right`

### Exercise & Fitness
- `squat`, `lunge_left`, `lunge_right`
- `pushup`, `situp`, `burpee`
- `jumping_jacks`, `mountain_climber`

### Dance & Artistic
- `dance1`, `dance2`, `dance3`
- `breakdance`, `salsa`, `ballet`

### Martial Arts
- `punch_left`, `punch_right`
- `kick_left`, `kick_right`
- `block_high`, `block_low`

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `num_envs` (try 16 or 32)
   - Use smaller network architectures
   - Reduce `episode_length_s`

2. **LocoMujoco Dataset Not Found**
   - Ensure LocoMujoco is installed: `pip install loco-mujoco`
   - Check dataset name spelling
   - Some datasets may need separate download

3. **Training Instability**
   - Lower learning rates
   - Increase `env_reward_weight` for more stable task rewards
   - Use trajectory-optimized control in environment

4. **Slow Training**
   - Ensure CUDA is available and being used
   - Use smaller networks for faster iteration
   - Reduce physics timestep (`dt = 0.02` instead of `0.01`)

### Performance Optimization

- **GPU Memory**: Use 16-64 environments depending on GPU memory
- **CPU Usage**: More environments = more CPU cores needed
- **Storage**: Checkpoints are saved regularly, ensure sufficient disk space

## 📚 Technical Details

### AMP (Adversarial Motion Priors)
- Discriminator distinguishes between expert and policy motion
- Uses least-squares loss with expert=+1, policy=-1 targets
- Provides dense imitation rewards: `max(0, 1 - 0.25*(score-1)²)`
- Running normalization for stable training

### Genesis Integration
- Uses PD control with trajectory-optimized gains
- Proper DOF mapping to avoid root joint control issues
- Box feet for stable ground contact
- 55-dimensional observation space matching LocoMujoco

### Multi-Dataset Training
- Expert data sampled from all loaded datasets
- Mixed reward combining environment and AMP components
- Joint mapping verified across all datasets
- Balanced sampling to prevent dataset bias

## 🤝 Contributing

This is research code integrating Genesis physics with LocoMujoco datasets. Contributions welcome for:
- New task configurations
- Additional LocoMujoco dataset integrations  
- Training algorithm improvements
- Visualization and analysis tools

## 📄 License

This project integrates multiple components:
- Genesis: Check Genesis license terms
- LocoMujoco: Apache 2.0 License
- Original integration code: MIT License

## 🔗 References

- [Genesis Physics Engine](https://github.com/Genesis-Embodied-AI/Genesis)
- [LocoMujoco](https://github.com/robfiras/loco-mujoco)
- [AMP Paper](https://arxiv.org/abs/2104.02180)
- [PPO Paper](https://arxiv.org/abs/1707.06347)