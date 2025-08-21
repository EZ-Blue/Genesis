# BVH Integration with Genesis Imitation Learning

This integration allows you to use custom BVH motion capture data for imitation learning with the Genesis SkeletonTorque model.

## 🔄 Complete Workflow

### 1. **Preprocess BVH Files**
**Location**: `/home/choonspin/intuitive_autonomy/loco-mujoco/preprocess_scripts/`

```bash
cd /home/choonspin/intuitive_autonomy/loco-mujoco/preprocess_scripts
python bvh_general_pipeline.py --input your_motion.bvh --output your_motion.npz --frequency 40
```

**Dependencies for BVH preprocessing**:
- `loco-mujoco` package
- `bvh` Python package
- `scipy` for rotations
- `numpy`, `jax`

### 2. **Test Integration** (Optional but Recommended)
```bash
cd /home/choonspin/intuitive_autonomy/integration/Genesis/genesis_loco/integration
python test_bvh_integration.py --npz_file /path/to/your_motion.npz
```

### 3. **Train with Custom BVH Data**
```bash
python comprehensive_imitation_trainer.py
# Select option 4: "custom - Load custom NPZ trajectory file"
# Enter path to your NPZ file
```

## 📁 File Structure

```
Genesis/genesis_loco/integration/
├── comprehensive_imitation_trainer.py  # Enhanced with NPZ support
├── data_bridge.py                     # Enhanced to load NPZ files
├── test_bvh_integration.py           # Integration test suite
└── README_BVH_INTEGRATION.md         # This file

loco-mujoco/preprocess_scripts/
├── bvh_general_pipeline.py           # BVH → NPZ conversion
├── simple_motion_viewer.py           # View processed motions
└── your_motion.npz                   # Example processed file
```

## 🔧 Key Integration Features

### **Enhanced Data Bridge** (`data_bridge.py`)
- **NPZ File Detection**: Automatically detects if input is a file path vs behavior name
- **Backward Compatibility**: Still supports LocoMujoco behaviors (walk, run, squat)
- **Trajectory Segmentation**: Creates overlapping segments for AMP training

### **Trainer Interface** (`comprehensive_imitation_trainer.py`)
- **Option 4**: Custom NPZ trajectory file loading
- **Interactive Input**: File path validation and selection
- **Qt/Display Fixes**: Headless training support

### **Test Suite** (`test_bvh_integration.py`)
- **Complete Pipeline Testing**: BVH → NPZ → Genesis → AMP
- **Compatibility Validation**: Joint mapping and trajectory format
- **Integration Verification**: End-to-end functionality

## 🎯 Supported BVH Features

The BVH preprocessing pipeline (`bvh_general_pipeline.py`) supports:

- **Joint Mapping**: Comprehensive skeleton joint coverage
- **Coordinate Alignment**: +Y forward alignment for consistency
- **Height Normalization**: Adaptive ground contact prevention
- **Frequency Conversion**: Configurable output frequency (default 40Hz)
- **SkeletonTorque Compatibility**: Direct mapping to Genesis joints

## 📊 Expected Results

After preprocessing, you should see:
- **Joint Compatibility**: ~27/28 joints matched (excellent)
- **Trajectory Segments**: Multiple overlapping segments for training
- **AMP Integration**: Expert observations for discriminator training

## 🧪 Testing Your Integration

Run the test suite to verify everything works:

```bash
python test_bvh_integration.py --npz_file your_motion.npz
```

**Expected output**:
```
✅ Genesis environment created
✅ Data bridge integration successful
✅ AMP integration successful
🎉 ALL TESTS PASSED!
```

## 🔗 Dependencies

### **Genesis Training Environment**
- Genesis physics engine
- SkeletonTorque model with box feet
- PyTorch for neural networks

### **BVH Preprocessing** (External)
- Located at: `/home/choonspin/intuitive_autonomy/loco-mujoco/preprocess_scripts/`
- LocoMujoco trajectory format compatibility
- BVH parsing and coordinate transformation

## 🚀 Usage Examples

### **Quick Test with Existing Data**
```bash
# Test with preprocessed motion
python test_bvh_integration.py --npz_file /home/choonspin/intuitive_autonomy/loco-mujoco/your_motion.npz
```

### **Train on Custom Motion**
```bash
# 1. Preprocess your BVH
cd /home/choonspin/intuitive_autonomy/loco-mujoco/preprocess_scripts
python bvh_general_pipeline.py --input my_dance.bvh --output my_dance.npz

# 2. Train with Genesis
cd /home/choonspin/intuitive_autonomy/integration/Genesis/genesis_loco/integration
python comprehensive_imitation_trainer.py
# Select: 4 (custom)
# Enter: my_dance.npz
```

### **Development Workflow**
```bash
# Test preprocessing
python bvh_general_pipeline.py --input motion.bvh --output motion.npz

# Validate integration
python test_bvh_integration.py --npz_file motion.npz

# Train if tests pass
python comprehensive_imitation_trainer.py
```

## 🐛 Troubleshooting

### **BVH Preprocessing Issues**
- Ensure BVH file has standard joint names (Character1_*)
- Check for missing rotation channels in BVH
- Verify BVH frequency and duration

### **Genesis Integration Issues**
- Run test suite first: `python test_bvh_integration.py`
- Check joint compatibility output
- Verify NPZ file format with LocoMujoco trajectory structure

### **Training Issues**
- Ensure sufficient trajectory length (>300 timesteps recommended)
- Check expert observation generation
- Verify AMP discriminator initialization

## 📝 Notes

- **Performance**: Use 40Hz frequency for optimal training performance
- **Memory**: Large BVH files may require segmentation
- **Compatibility**: Designed for humanoid SkeletonTorque model
- **Quality**: Better BVH data → better imitation learning results