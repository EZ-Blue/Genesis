# 🚶‍♂️ Genesis Skeleton Walking Training

Complete imitation learning pipeline for training a walking humanoid skeleton using Genesis physics and LocoMujoco expert data.

## 🚀 Quick Start

### 1. Train a Walking Agent
```bash
cd /home/ez/Documents/Genesis/genesis_loco/integration
python train_walking.py
```

**Training Options:**
- **Full Scale** (64 envs, 15s episodes) - Best results, ~2-4 hours
- **Fast Training** (16 envs, 10s episodes) - Good results, ~1-2 hours  
- **Quick Test** (4 envs, 5s episodes) - Basic testing, ~15 minutes

### 2. Test Your Trained Agent
```bash
python test_walking.py
```

**Test Modes:**
- **Automatic**: Runs 5 test episodes automatically
- **Interactive**: Manual control with step-by-step execution

## 📊 What You'll See

### During Training
- **Genesis 3D Viewer**: Watch skeleton learn to walk in real-time
- **Console Metrics**: Episode rewards, lengths, discriminator accuracy
- **Progress Plots**: Automatically saved training curves
- **Model Checkpoints**: Saved every 100 iterations

### Key Metrics to Watch
- **Episode Length**: Should increase as agent learns balance
- **AMP Reward**: Measures similarity to expert walking motion  
- **Expert/Policy Accuracy**: Discriminator's ability to distinguish expert vs policy

### Expected Learning Progression
1. **Initial** (0-100 iters): Random falling, short episodes
2. **Balance** (100-500 iters): Agent learns to stand/balance
3. **Walking** (500-1500 iters): Develops walking gait
4. **Refinement** (1500+ iters): Improves walking quality and efficiency

## 🎯 Training Results

### Success Indicators
- Episode length > 200 steps (4+ seconds of walking)
- AMP reward > 0.7 (good motion similarity)
- Stable, forward walking motion in viewer

### File Outputs
```
walking_training_TIMESTAMP/
├── best_model.pt              # Best performing model
├── checkpoint_iter_*.pt       # Regular checkpoints  
├── training_progress.png      # Training curves plot
└── progress_iter_*.png        # Intermediate progress plots
```

## 🔧 Configuration Options

### Environment Settings
```python
'num_envs': 64,              # Parallel environments (more = faster learning)
'episode_length_s': 15.0,    # Episode duration (longer = more learning per episode)
'show_viewer': True,         # Enable/disable 3D visualization
```

### Reward Mixing
```python
'env_reward_weight': 0.1,    # 10% environment reward, 90% AMP reward
```
- **Higher env_reward_weight**: Focuses on task completion
- **Lower env_reward_weight**: Focuses on natural motion style

### Network Architecture
```python
'policy': {
    'hidden_layers': [512, 256, 128],  # Policy network size
    'learning_rate': 3e-4              # Learning rate
}
```

## 🎮 Advanced Usage

### Resume Training
```python
# Load checkpoint and continue training
checkpoint = torch.load('walking_training_TIMESTAMP/checkpoint_iter_1000.pt')
# ... resume from checkpoint
```

### Different Datasets
```python
# In data_bridge.py, change dataset:
bridge.load_trajectory("run")    # Running motion
bridge.load_trajectory("jump")   # Jumping motion
```

### Hyperparameter Tuning
Key parameters to experiment with:
- `env_reward_weight`: Balance between task and imitation
- `learning_rate`: Policy and discriminator learning rates
- `episode_length_s`: Longer episodes for more complex behaviors
- `hidden_layers`: Network capacity

## 🐛 Troubleshooting

### Common Issues

**Skeleton Falls Immediately**
- Normal in early training (first 100-200 iterations)
- Try increasing `episode_length_s` for more learning time
- Check that PD gains are properly set for torque control

**Training Stalls**
- Discriminator may be too strong - lower discriminator learning rate
- Try adjusting reward mixing ratio
- Increase policy learning rate slightly

**Poor Walking Quality**
- Increase `env_reward_weight` for more task focus
- Train longer (2000+ iterations)
- Check expert trajectory quality

**GPU Memory Issues**
- Reduce `num_envs` (try 32 or 16)
- Reduce `episode_length_s`
- Use smaller network `hidden_layers`

### Performance Tips
- **More environments** = faster learning but more GPU memory
- **Longer episodes** = better learning but slower iterations
- **Visualization off** during training = faster iteration speed

## 📈 Understanding the Metrics

### Episode Reward
- Combination of environment reward and AMP reward
- Should generally increase over training
- Target: > 0.5 for good walking

### Episode Length  
- Number of simulation steps before termination
- Early training: ~10-50 steps (falling)
- Good walking: 200+ steps (sustained locomotion)

### AMP Reward
- Measures motion similarity to expert data
- Range: 0.0 (worst) to 1.0 (perfect match)
- Target: > 0.7 for natural-looking motion

### Discriminator Accuracy
- **Expert Accuracy**: Should be high (>90%) - discriminator recognizes expert data
- **Policy Accuracy**: Should increase gradually - policy becoming more expert-like

## 🎊 Success Criteria

Your agent has successfully learned to walk when:
1. ✅ Episode length consistently > 200 steps
2. ✅ Forward walking motion visible in viewer
3. ✅ AMP reward > 0.6 
4. ✅ Stable balance and gait pattern
5. ✅ Natural-looking human-like motion

**Congratulations! You've successfully trained a walking humanoid using imitation learning! 🎉**