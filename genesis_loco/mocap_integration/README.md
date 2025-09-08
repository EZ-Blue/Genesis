# Mocap Integration for Genesis

This directory contains the essential files for motion capture integration with Genesis physics simulation.

## Files

- `mocap_player_genesis.py` - Main player for Genesis mocap animation with full joint mapping and coordinate fixes
- `mocap_converter_fixed.py` - Converter from CSV mocap data to JSON format with proper joint processing
- `skeleton/` - Genesis skeleton XML files for humanoid model
- `csv/` - Sample LOCAL mocap data processed and ready for playback

## Usage

### Convert mocap CSV to JSON:
```bash
python3 mocap_converter_fixed.py
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

## Requirements

- Genesis physics engine
- PyTorch
- NumPy
- Standard Python libraries (json, math, pathlib, time)