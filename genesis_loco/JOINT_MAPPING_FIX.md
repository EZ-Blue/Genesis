# Genesis Joint Control Mapping Fix

## Problem Summary

The Genesis-LocoMujoco integration had **incorrect joint control mapping** where:
- `mot_lumbar_bend` controlled `knee_angle_l` instead of `lumbar_bending`
- `mot_hip_flexion_r` controlled `lumbar_bending` instead of `hip_flexion_r`
- Other joints were similarly misaligned

## Root Cause

The issue was in **DOF index mapping**. The code used `joint.dofs_idx_local[0]` which gave incorrect DOF indices that didn't match the actual controllable joints in Genesis.

## Solution: Genesis Motor Detection Approach

Applied Genesis' proven motor detection approach from `_main.py` `view()` function:

1. **Filter joint types**: Skip `FREE` and `FIXED` joints (non-controllable)
2. **Use correct DOF indexing**: Use Genesis' own motor detection logic
3. **Consistent mapping**: Apply same approach in both environment and data bridge

## Files Modified

### 1. `environments/skeleton_humanoid.py`

**Added new methods:**
```python
def _get_motors_info(self):
    """Get controllable motor DOF indices using Genesis' approach"""
    # Same logic as Genesis _main.py view() function

def _setup_action_spec(self):
    """Setup action specification using Genesis motor detection"""
    # Uses _get_motors_info() instead of joint.dofs_idx_local[0]
```

**Key change:**
```python
# OLD (BROKEN):
self.action_to_joint_idx[action_name] = self.robot.get_joint(joint_name).dofs_idx_local[0]

# NEW (FIXED):
motors_dof_idx, motors_dof_name = self._get_motors_info()
if joint_name in motors_dof_name:
    motor_idx = motors_dof_name.index(joint_name)
    dof_idx = motors_dof_idx[motor_idx]
    self.action_to_joint_idx[action_name] = dof_idx
```

### 2. `integration/data_bridge.py`

**Added new methods:**
```python
def _get_genesis_motors_info(self):
    """Get controllable motor DOF indices using Genesis' proven approach"""
    # Same logic as Genesis _main.py view() function

def build_joint_mapping(self):
    """Build mapping using Genesis motor detection instead of env.dof_names"""

def _convert_joint_data(self):
    """Convert using Genesis motor detection for correct DOF indices"""
```

**Key changes:**
- Replaced reliance on `env.dof_names` (custom attribute) with Genesis motor detection
- Used same DOF indexing approach as skeleton environment
- Consistent joint filtering and mapping

## Testing

### Test Scripts Created:

1. **`test_genesis_mapping.py`**: Tests the new Genesis motor detection in skeleton environment
2. **`test_trajectory_with_fix.py`**: Tests trajectory following with fixed joint mapping
3. **`simple_mapping_check.py`**: Simple diagnostic to verify joint control mapping

### How to Test:

```bash
cd /home/ez/Documents/Genesis/genesis_loco

# Test the environment fix
python test_genesis_mapping.py

# Test trajectory following with fix
python test_trajectory_with_fix.py

# Quick mapping verification
python simple_mapping_check.py
```

## Expected Results After Fix

1. **Correct Joint Control**: Each action should control its intended joint
   - `mot_lumbar_bend` → `lumbar_bending` ✅
   - `mot_hip_flexion_r` → `hip_flexion_r` ✅
   - All other joints correctly mapped ✅

2. **Successful Trajectory Following**: 
   - Skeleton should follow LocoMujoco walking trajectories accurately
   - No more cross-joint control issues
   - Proper imitation learning performance

3. **Diagnostic Output**: Shows Genesis-detected motors and correct mapping
   ```
   Genesis detected 27 controllable DOFs:
     0: DOF  6 = hip_flexion_r
     1: DOF  7 = hip_flexion_l  
     2: DOF  8 = lumbar_extension
     ...
   ```

## Technical Notes

- **Motor Detection Logic**: Identical to Genesis `_main.py` view function (lines 81-97)
- **Joint Type Filtering**: Skips `gs.JOINT_TYPE.FREE` and `gs.JOINT_TYPE.FIXED`
- **DOF Index Consistency**: Both environment and data bridge use same indexing
- **Multi-DOF Support**: Handles joints with multiple DOFs correctly

## Verification

The fix can be verified by:
1. Running joint control tests - each action should move only its intended joint
2. Comparing DOF indices between old and new approaches  
3. Testing trajectory following accuracy
4. Checking Genesis viewer joint control (should match our implementation)

## Impact

This fix resolves the core joint control mapping issue, enabling:
- ✅ Accurate trajectory following
- ✅ Proper imitation learning
- ✅ Correct motor control for locomotion
- ✅ Consistent Genesis-LocoMujoco integration