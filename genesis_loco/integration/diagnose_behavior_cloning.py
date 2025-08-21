#!/usr/bin/env python3
"""
Behavior Cloning Diagnostic Tool

Analyzes potential issues causing low training loss but poor performance:
1. Action range and distribution analysis
2. Observation-action correlation analysis
3. Expert action smoothness vs model predictions
4. PD controller parameter compatibility
5. Model capacity and overfitting analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge
from integration.behavior_cloning_trainer import BehaviorCloningMLP

def safe_init_genesis():
    """Safely initialize Genesis"""
    try:
        gs.init(backend=gs.gpu)
        return True
    except Exception as e:
        if "already initialized" in str(e):
            return True
        else:
            print(f"Genesis initialization failed: {e}")
            return False

class BehaviorCloningDiagnostic:
    """Comprehensive behavior cloning diagnostic tool"""
    
    def __init__(self, model_path: str, behavior: str = "walk"):
        self.model_path = model_path
        self.behavior = behavior
        
        print("🔬 BEHAVIOR CLONING DIAGNOSTIC")
        print("=" * 60)
        
        # Initialize Genesis and environment
        if not safe_init_genesis():
            raise RuntimeError("Failed to initialize Genesis")
        
        self.env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=10.0,
            dt=0.01,
            show_viewer=False,
            use_box_feet=True
        )
        
        # Initialize data bridge
        self.data_bridge = LocoMujocoDataBridge(self.env)
        success = self.data_bridge.load_trajectory(behavior)
        if not success:
            raise RuntimeError(f"Failed to load {behavior} trajectory")
        
        # Load model
        self.model = self._load_model()
        
        print(f"✅ Diagnostic setup complete")
        print(f"   Environment: {self.env.num_observations} obs, {self.env.num_actions} actions")
        print(f"   Trajectory: {self.data_bridge.trajectory_length} timesteps")
    
    def _load_model(self):
        """Load the trained behavior cloning model"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
        
        model = BehaviorCloningMLP(obs_dim=obs_dim, action_dim=action_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded: {obs_dim} obs → {action_dim} actions")
        return model
    
    def diagnose_all(self):
        """Run comprehensive diagnostics"""
        print("\n🔍 RUNNING COMPREHENSIVE DIAGNOSTICS")
        print("=" * 60)
        
        # 1. Action range and distribution analysis
        print("\n1. ACTION RANGE & DISTRIBUTION ANALYSIS")
        self._analyze_action_distributions()
        
        # 2. Expert vs model prediction comparison
        print("\n2. EXPERT vs MODEL PREDICTION COMPARISON")
        self._compare_expert_vs_predictions()
        
        # 3. Action smoothness analysis
        print("\n3. ACTION SMOOTHNESS ANALYSIS")
        self._analyze_action_smoothness()
        
        # 4. PD controller compatibility
        print("\n4. PD CONTROLLER COMPATIBILITY")
        self._analyze_pd_compatibility()
        
        # 5. Observation quality analysis
        print("\n5. OBSERVATION QUALITY ANALYSIS")
        self._analyze_observation_quality()
        
        # 6. Model capacity analysis
        print("\n6. MODEL CAPACITY ANALYSIS")
        self._analyze_model_capacity()
    
    def _analyze_action_distributions(self):
        """Analyze expert action ranges and distributions"""
        print("   Collecting expert actions...")
        
        expert_actions = []
        timesteps = np.linspace(0, self.data_bridge.trajectory_length-1, 1000, dtype=int)
        
        for timestep in timesteps:
            state = self.data_bridge.get_trajectory_state(timestep)
            if state and 'dof_pos' in state:
                dof_pos = state['dof_pos']
                if hasattr(dof_pos, 'cpu'):
                    dof_pos = dof_pos.cpu().numpy()
                expert_actions.append(dof_pos)
        
        expert_actions = np.array(expert_actions)
        
        print(f"   📊 Expert action statistics:")
        print(f"      Shape: {expert_actions.shape}")
        print(f"      Range: [{expert_actions.min():.3f}, {expert_actions.max():.3f}]")
        print(f"      Mean: {expert_actions.mean():.6f}")
        print(f"      Std: {expert_actions.std():.6f}")
        
        # Check for joint-specific ranges
        print(f"   📊 Per-joint analysis:")
        for i in range(min(10, expert_actions.shape[1])):  # First 10 joints
            joint_min = expert_actions[:, i].min()
            joint_max = expert_actions[:, i].max()
            joint_std = expert_actions[:, i].std()
            print(f"      Joint {i:2d}: range [{joint_min:+6.3f}, {joint_max:+6.3f}], std {joint_std:.3f}")
        
        # Check for potentially problematic joint ranges
        large_range_joints = []
        small_range_joints = []
        
        for i in range(expert_actions.shape[1]):
            joint_range = expert_actions[:, i].max() - expert_actions[:, i].min()
            if joint_range > 2.0:  # Large range (>2 radians)
                large_range_joints.append((i, joint_range))
            elif joint_range < 0.1:  # Very small range
                small_range_joints.append((i, joint_range))
        
        if large_range_joints:
            print(f"   ⚠️  Large range joints (>{2.0} rad): {large_range_joints[:5]}...")
        if small_range_joints:
            print(f"   ⚠️  Small range joints (<{0.1} rad): {small_range_joints[:5]}...")
        
        return expert_actions
    
    def _compare_expert_vs_predictions(self):
        """Compare expert actions vs model predictions"""
        print("   Generating model predictions for expert observations...")
        
        # Get expert observations
        expert_observations = self.data_bridge.get_expert_observations_cached(
            dataset_name=self.behavior,
            num_timesteps=1000,  # Sample subset for analysis
            start_timestep=0,
            step_interval=max(1, self.data_bridge.trajectory_length // 1000)
        )
        
        if expert_observations is None:
            print("   ❌ Failed to get expert observations")
            return
        
        # Get corresponding expert actions
        expert_actions = []
        timesteps = np.linspace(0, self.data_bridge.trajectory_length-1, len(expert_observations), dtype=int)
        
        for timestep in timesteps:
            state = self.data_bridge.get_trajectory_state(timestep)
            if state and 'dof_pos' in state:
                dof_pos = state['dof_pos']
                if hasattr(dof_pos, 'cpu'):
                    dof_pos = dof_pos.cpu().numpy()
                expert_actions.append(dof_pos)
        
        expert_actions = np.array(expert_actions)
        
        # Generate model predictions
        with torch.no_grad():
            predicted_actions = self.model(expert_observations).cpu().numpy()
        
        # Compare statistics
        print(f"   📊 Prediction vs Expert comparison:")
        print(f"      Expert actions shape: {expert_actions.shape}")
        print(f"      Predicted actions shape: {predicted_actions.shape}")
        
        # MSE per joint
        mse_per_joint = np.mean((predicted_actions - expert_actions)**2, axis=0)
        mae_per_joint = np.mean(np.abs(predicted_actions - expert_actions), axis=0)
        
        print(f"      Overall MSE: {np.mean(mse_per_joint):.6f}")
        print(f"      Overall MAE: {np.mean(mae_per_joint):.6f}")
        
        # Identify problematic joints
        high_error_joints = []
        for i in range(len(mse_per_joint)):
            if mse_per_joint[i] > 0.05:  # High MSE threshold
                high_error_joints.append((i, mse_per_joint[i], mae_per_joint[i]))
        
        if high_error_joints:
            print(f"   ⚠️  High error joints (MSE > 0.05):")
            for joint, mse, mae in high_error_joints[:10]:
                print(f"      Joint {joint:2d}: MSE {mse:.6f}, MAE {mae:.6f}")
        
        # Check prediction ranges vs expert ranges
        pred_ranges = predicted_actions.max(axis=0) - predicted_actions.min(axis=0)
        expert_ranges = expert_actions.max(axis=0) - expert_actions.min(axis=0)
        
        range_issues = []
        for i in range(len(pred_ranges)):
            range_ratio = pred_ranges[i] / (expert_ranges[i] + 1e-8)
            if range_ratio < 0.5 or range_ratio > 2.0:
                range_issues.append((i, range_ratio, pred_ranges[i], expert_ranges[i]))
        
        if range_issues:
            print(f"   ⚠️  Range mismatch joints (ratio < 0.5 or > 2.0):")
            for joint, ratio, pred_range, expert_range in range_issues[:10]:
                print(f"      Joint {joint:2d}: ratio {ratio:.3f}, pred {pred_range:.3f}, expert {expert_range:.3f}")
        
        return expert_actions, predicted_actions
    
    def _analyze_action_smoothness(self):
        """Analyze smoothness of expert vs predicted actions"""
        print("   Analyzing action smoothness...")
        
        # Get sequential expert actions
        sequence_length = 200
        expert_sequence = []
        predicted_sequence = []
        
        # Get expert observations for sequence
        expert_obs = self.data_bridge.get_expert_observations_cached(
            dataset_name=self.behavior,
            num_timesteps=sequence_length,
            start_timestep=1000,  # Start from middle of trajectory
            step_interval=1
        )
        
        if expert_obs is None:
            print("   ❌ Failed to get expert observation sequence")
            return
        
        # Get expert actions for same sequence
        for i in range(sequence_length):
            timestep = 1000 + i
            state = self.data_bridge.get_trajectory_state(timestep)
            if state and 'dof_pos' in state:
                dof_pos = state['dof_pos']
                if hasattr(dof_pos, 'cpu'):
                    dof_pos = dof_pos.cpu().numpy()
                expert_sequence.append(dof_pos)
        
        expert_sequence = np.array(expert_sequence)
        
        # Generate predicted sequence
        with torch.no_grad():
            predicted_sequence = self.model(expert_obs).cpu().numpy()
        
        # Calculate smoothness metrics (total variation)
        expert_smoothness = np.mean(np.sum(np.abs(np.diff(expert_sequence, axis=0)), axis=0))
        predicted_smoothness = np.mean(np.sum(np.abs(np.diff(predicted_sequence, axis=0)), axis=0))
        
        print(f"   📊 Smoothness analysis:")
        print(f"      Expert smoothness (total variation): {expert_smoothness:.6f}")
        print(f"      Predicted smoothness: {predicted_smoothness:.6f}")
        print(f"      Smoothness ratio: {predicted_smoothness / (expert_smoothness + 1e-8):.3f}")
        
        if predicted_smoothness > expert_smoothness * 2:
            print("   ⚠️  Model predictions are much less smooth than expert!")
        elif predicted_smoothness < expert_smoothness * 0.5:
            print("   ⚠️  Model predictions may be over-smoothed!")
        else:
            print("   ✅ Model smoothness is reasonable")
    
    def _analyze_pd_compatibility(self):
        """Analyze if predicted actions are compatible with PD controller"""
        print("   Analyzing PD controller compatibility...")
        
        # Check if environment uses PD control
        if hasattr(self.env, 'p_gains') and hasattr(self.env, 'd_gains'):
            print(f"      PD gains - P: {self.env.p_gains[:5].cpu().numpy()}... D: {self.env.d_gains[:5].cpu().numpy()}...")
            
            # Check gain magnitudes
            p_gain_range = self.env.p_gains.max().item() - self.env.p_gains.min().item()
            d_gain_range = self.env.d_gains.max().item() - self.env.d_gains.min().item()
            
            print(f"      P gain range: {p_gain_range:.3f}, D gain range: {d_gain_range:.3f}")
            
            # High gains + large action errors = instability
            avg_p_gain = self.env.p_gains.mean().item()
            if avg_p_gain > 1000:
                print("   ⚠️  High P gains detected - small action errors can cause large forces!")
        else:
            print("      ⚠️  PD gains not accessible - check controller configuration")
        
        # Check for action clipping/saturation
        sample_actions = []
        for i in range(100):
            timestep = i * 10
            state = self.data_bridge.get_trajectory_state(timestep)
            if state and 'dof_pos' in state:
                dof_pos = state['dof_pos']
                if hasattr(dof_pos, 'cpu'):
                    dof_pos = dof_pos.cpu().numpy()
                sample_actions.append(dof_pos)
        
        sample_actions = np.array(sample_actions)
        
        # Check if actions are within reasonable joint limits
        extreme_actions = []
        for i in range(sample_actions.shape[1]):
            joint_values = sample_actions[:, i]
            if np.any(np.abs(joint_values) > 3.0):  # > 3 radians is extreme
                extreme_actions.append((i, joint_values.min(), joint_values.max()))
        
        if extreme_actions:
            print(f"   ⚠️  Joints with extreme values (>3 rad):")
            for joint, min_val, max_val in extreme_actions[:5]:
                print(f"      Joint {joint}: [{min_val:.3f}, {max_val:.3f}]")
    
    def _analyze_observation_quality(self):
        """Analyze observation quality and potential issues"""
        print("   Analyzing observation quality...")
        
        # Get sample observations
        expert_obs = self.data_bridge.get_expert_observations_cached(
            dataset_name=self.behavior,
            num_timesteps=1000,
            start_timestep=0,
            step_interval=max(1, self.data_bridge.trajectory_length // 1000)
        )
        
        if expert_obs is None:
            print("   ❌ Failed to get expert observations")
            return
        
        obs_np = expert_obs.cpu().numpy()
        
        print(f"   📊 Observation statistics:")
        print(f"      Shape: {obs_np.shape}")
        print(f"      Range: [{obs_np.min():.6f}, {obs_np.max():.6f}]")
        print(f"      Mean: {obs_np.mean():.6f}")
        print(f"      Std: {obs_np.std():.6f}")
        
        # Check for constant or near-constant features
        feature_stds = np.std(obs_np, axis=0)
        low_variance_features = np.where(feature_stds < 1e-6)[0]
        
        if len(low_variance_features) > 0:
            print(f"   ⚠️  Low variance features ({len(low_variance_features)}): {low_variance_features[:10]}...")
        
        # Check for extreme values
        extreme_features = []
        for i in range(obs_np.shape[1]):
            feature_values = obs_np[:, i]
            if np.any(np.abs(feature_values) > 100):
                extreme_features.append((i, feature_values.min(), feature_values.max()))
        
        if extreme_features:
            print(f"   ⚠️  Features with extreme values (>100):")
            for feat, min_val, max_val in extreme_features[:5]:
                print(f"      Feature {feat}: [{min_val:.3f}, {max_val:.3f}]")
        
        # Check observation normalization
        obs_means = np.mean(obs_np, axis=0)
        obs_stds = np.std(obs_np, axis=0)
        
        unnormalized_features = []
        for i in range(len(obs_means)):
            if np.abs(obs_means[i]) > 1.0 or obs_stds[i] > 10.0:
                unnormalized_features.append((i, obs_means[i], obs_stds[i]))
        
        if unnormalized_features:
            print(f"   ⚠️  Potentially unnormalized features:")
            for feat, mean, std in unnormalized_features[:5]:
                print(f"      Feature {feat}: mean {mean:.3f}, std {std:.3f}")
    
    def _analyze_model_capacity(self):
        """Analyze if model has sufficient capacity"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"   📊 Model capacity analysis:")
        print(f"      Total parameters: {total_params:,}")
        print(f"      Input dimension: {self.model.network[0].in_features}")
        print(f"      Output dimension: {self.model.network[-1].out_features}")
        
        # Simple capacity heuristic
        input_dim = self.model.network[0].in_features
        output_dim = self.model.network[-1].out_features
        
        # Rule of thumb: parameters should be roughly 10-100x the output dimension
        param_ratio = total_params / output_dim
        
        print(f"      Parameter to output ratio: {param_ratio:.1f}")
        
        if param_ratio < 10:
            print("   ⚠️  Model may be undercapacity (ratio < 10)")
        elif param_ratio > 1000:
            print("   ⚠️  Model may be overcapacity and overfitting (ratio > 1000)")
        else:
            print("   ✅ Model capacity seems reasonable")
        
        # Check for potential overfitting signs
        # (This would require training/validation metrics analysis)
        print("   💡 For overfitting analysis, check if training loss << validation loss")

def main():
    """Main diagnostic function"""
    import glob
    
    print("🔬 BEHAVIOR CLONING DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Find recent model files
    model_files = glob.glob("behavior_cloning_walk_best_model_*.pth")
    if not model_files:
        print("❌ No behavior cloning model files found in current directory")
        print("   Expected pattern: behavior_cloning_walk_best_model_*.pth")
        return 1
    
    # Use most recent model
    latest_model = max(model_files, key=os.path.getctime)
    print(f"🔍 Analyzing model: {latest_model}")
    
    try:
        diagnostic = BehaviorCloningDiagnostic(latest_model, behavior="walk")
        diagnostic.diagnose_all()
        
        print(f"\n🎯 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print("Key areas to investigate:")
        print("1. Check if action ranges match expert data")
        print("2. Verify observation normalization")
        print("3. Consider increasing model capacity")
        print("4. Check PD controller gains")
        print("5. Verify physics integration in training data")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())