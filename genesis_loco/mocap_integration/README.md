# Mocap Integration for Genesis

This directory contains the essential files for motion capture integration with Genesis physics simulation.

## Files

- `mocap_player_genesis.py` - Main player for Genesis mocap animation with full joint mapping and coordinate fixes
- `mocap_converter.py` - Converter from CSV mocap data to JSON format with proper joint processing
- `mocap_data/` - Sample LOCAL mocap data:
  - `Take 2025-08-19 03.01.18 PM_LOCAL.csv` - Original raw mocap CSV export
  - `Take 2025-08-19 03.01.18 PM_LOCAL_processed.json` - Processed JSON ready for playback

## Usage

### Convert mocap CSV to JSON:
```bash
python3 mocap_converter.py
```

### Play mocap animation:
```bash
python3 mocap_player_genesis.py
```

The player will automatically use the LOCAL CSV data as default. For interactive file selection:
```bash
python3 mocap_player_genesis.py --interactive
```

## Features

- Complete 37-DOF humanoid joint mapping
- Coordinate system fixes for hip, knee, shoulder, and arm joints
- Support for LOCAL coordinate mocap data
- Kinematic and physics simulation modes
- Interactive file selection and auto-play modes
- Proper parent-child joint relationship handling

## Testing

We provide multiple test suites to ensure robust functionality:

### Run Tests
```bash
# Run simple basic tests (fast)
python3 test_mocap_simple.py

# Run comprehensive tests with actual LOCAL data (recommended)
python3 test_mocap_real.py
```

### Recommended Test Suite
```bash
# This is the best test to run - uses actual mocap data
python3 test_mocap_real.py
```

**Test Coverage:**

**test_mocap_real.py** (Primary - Recommended):
- ✅ Real LOCAL CSV and JSON data loading
- ✅ Complete MocapConverter pipeline with 1400+ frames
- ✅ Joint mapping completeness (37-DOF skeleton)
- ✅ Coordinate system transformations (hip/knee/shoulder inversions)
- ✅ Joint index mapping validation
- ✅ Euler to quaternion conversion
- ✅ Full integration pipeline testing
- ✅ Error handling and edge cases

**test_mocap_simple.py** (Basic - Fast):
- ✅ Basic MocapGenesisPlayer functionality
- ✅ Joint mapping structure validation
- ✅ Coordinate transformation functions
- ✅ DOF index lookup
- ✅ Mathematical conversions

### Manual Testing
```bash
# Test basic playback
python3 mocap_player_genesis.py

# Test converter
python3 mocap_converter.py

# Test interactive mode
python3 mocap_player_genesis.py --interactive
```

## Requirements

- Genesis physics engine
- PyTorch
- NumPy
- Standard Python libraries (json, math, pathlib, time)
- For testing: unittest (built-in) or pytest (optional)