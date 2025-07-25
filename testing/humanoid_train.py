import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from humanoid_env import HumanoidEnv


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.008,  # Reduced from 0.01 for stability
            "entropy_coef": 0.005,  # Reduced from 0.01 
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0005,  # Reduced from 0.001
            "max_grad_norm": 0.5,  # Reduced from 1.0
            "num_learning_epochs": 4,  # Reduced from 5
            "num_mini_batches": 8,  # Increased from 4
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 0.5,  # Reduced from 1.0 for stability
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        # Humanoid has 23 controllable joints based on the XML
        "num_actions": 23,  # Based on actual Mujoco humanoid skeleton
        # joint/link names - updated for humanoid skeleton from XML
        "default_joint_angles": {  # [rad] - humanoid default pose
            # Legs
            "hip_flexion_r": 0.0,
            "hip_adduction_r": 0.0,
            "hip_rotation_r": 0.0,
            "knee_angle_r": 0.0,
            "ankle_angle_r": 0.0,
            "hip_flexion_l": 0.0,
            "hip_adduction_l": 0.0,
            "hip_rotation_l": 0.0,
            "knee_angle_l": 0.0,
            "ankle_angle_l": 0.0,
            # Torso
            "lumbar_extension": 0.0,
            "lumbar_bending": 0.0,
            "lumbar_rotation": 0.0,
            # Right arm
            "arm_flex_r": 0.0,
            "arm_add_r": 0.0,
            "arm_rot_r": 0.0,
            "elbow_flex_r": 0.0,
            "pro_sup_r": 0.0,
            # Left arm  
            "arm_flex_l": 0.0,
            "arm_add_l": 0.0,
            "arm_rot_l": 0.0,
            "elbow_flex_l": 0.0,
            "pro_sup_l": 0.0,
        },
        "joint_names": [
            # Legs (10 joints)
            "hip_flexion_r",
            "hip_adduction_r", 
            "hip_rotation_r",
            "knee_angle_r",
            "ankle_angle_r",
            "hip_flexion_l",
            "hip_adduction_l",
            "hip_rotation_l",
            "knee_angle_l",
            "ankle_angle_l",
            # Torso (3 joints)
            "lumbar_extension",
            "lumbar_bending",
            "lumbar_rotation",
            # Right arm (5 joints)
            "arm_flex_r",
            "arm_add_r",
            "arm_rot_r",
            "elbow_flex_r",
            "pro_sup_r",
            # Left arm (5 joints)
            "arm_flex_l",
            "arm_add_l", 
            "arm_rot_l",
            "elbow_flex_l",
            "pro_sup_l",
        ],
        # PD control parameters - adjusted for humanoid stability
        "kp": 50.0,  # Reduced from 100.0 for smoother control
        "kd": 5.0,   # Reduced from 10.0 for smoother control
        # termination conditions - more restrictive for humanoid
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 20,
        # base pose - humanoid starts standing (from XML: pelvis at 0.975m)
        "base_init_pos": [0.0, 0.0, 0.975],  # Standing height from XML
        "base_init_quat": [0.7071067811865475, 0.7071067811865475, 0.0, 0.0],  # 90° rotation around X-axis to stand upright (from XML)
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # Reduced from 0.4 for safer exploration
        "simulate_action_latency": False,  # No latency for humanoid
        "clip_actions": 10.0,  # Reduced from 100.0 for better control
    }
    
    obs_cfg = {
        "num_obs": 39,  # Updated for humanoid walking: 3 + 3 + 3 + 10 + 10 + 10 = 39 (legs only)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "tracking_sigma": 0.5,  # Increased tolerance
        "base_height_target": 1.0,  # Target standing height (adjusted from XML pelvis height)
        "reward_scales": {
            # Primary reward for forward speed
            "forward_speed": 1.0,      # Reduced for initial training stability
            "upright": 1.0,            # Stay upright is critical
            "stable": -0.1,            # Reduced penalty
            "smooth_actions": -0.001,  # Very small penalty
            
            # Legacy rewards (much lower weights, focusing on legs)
            "tracking_lin_vel": 0.5,   # Main tracking reward
            "tracking_ang_vel": 0.1,
            "lin_vel_z": -0.5,
            "base_height": -0.5,       # Reduced penalty
            "action_rate": -0.001,     # Very small penalty
            "similar_to_default": -0.01, # Very small penalty
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        # Fixed speed command for 1.5 m/s forward
        "lin_vel_x_range": [1.5, 1.5],  # 1.5 m/s forward
        "lin_vel_y_range": [0.0, 0.0],  # No lateral movement
        "ang_vel_range": [0.0, 0.0],    # No turning
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="humanoid-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)  # Fewer envs for humanoid
    parser.add_argument("--max_iterations", type=int, default=500)   # More iterations for humanoid
    parser.add_argument("--show_viewer", action="store_true", help="Show viewer during training")
    args = parser.parse_args()

    gs.init(logging_level="warning", precision="64", backend=gs.cuda)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = HumanoidEnv(
        num_envs=args.num_envs, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg,
        show_viewer=args.show_viewer
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# Training - Focus on walking with legs only
python humanoid_train.py --exp_name humanoid-1.5ms --max_iterations 1000

# Training with viewer (slower)
python humanoid_train.py --exp_name humanoid-1.5ms --show_viewer --max_iterations 100

# Notes:
# - Uses 23 total joints but observation space focuses on 10 leg joints for walking
# - Arms and detailed foot joints are controlled but not in observation space
# - Reward system optimized for forward walking at 1.5 m/s
"""