import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
import glob
from typing import Dict, List

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv
from integration.data_bridge import LocoMujocoDataBridge

class SingleTrajectoryMLP(nn.Module):
    """
    Recreate the model architecture (must match training script)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 128],
                 dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        
        # Hidden layers with dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
    

def main():
    gs.init(backend=gs.cuda)

    env = SkeletonHumanoidEnv(
        num_envs=1,
        use_box_feet=True,
        show_viewer=True,
        sim_options=gs.options.SimOptions(
                dt=0.1, 
                substeps=2,
                gravity=(0.0,0.0,0.0)
            ),
    )

    data_bridge = LocoMujocoDataBridge(env)
    trajectory_name = 'walk'
    tj = data_bridge.load_trajectory(trajectory_name)

    state = data_bridge.get_trajectory_state(50)

    checkpoint = torch.load(
        "/home/ez/Documents/Genesis/genesis_loco/final_single_trajectory_seg0-500_20250823_132004.pth", 
        map_location=gs.device
        )
    
    hidden_dims = [512, 256]

    model = SingleTrajectoryMLP(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dims=hidden_dims,
            dropout_rate=0.0  # No dropout during inference
        ).to(gs.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")
    # print(f"   Input: {checkpoint["obs_dim"]} observations")
    # print(f"   Output: {model.action_dim} joint positions")


    data_bridge.apply_trajectory_state(state)

    for key in state:
        print(f"{key}: {state[key]}, length: {len(state[key])}")
        continue

    obs = env._get_observations()[0]

    print(f"obs: {obs}")

    actions = model(obs)

    actions = actions.to(env.device)

    print(f"Actions Shape: {actions.shape}, Actions: {actions}")

    obs, rewards, dones, info = env.step(actions.unsqueeze(0))

    new_obs = env.robot.get_dofs_position(dofs_idx_local=env.motors_dof_idx)

    print(f"Env observations: {obs}")

    print(f"Robot joint obs: {new_obs}")

    expert_next_state = data_bridge.get_trajectory_state(51)

    for key in expert_next_state:
        print(f"{key}: {expert_next_state[key]}, length: {len(expert_next_state[key])}")
        continue

    data_bridge.apply_trajectory_state(expert_next_state)

    new_obs = env.robot.get_dofs_position(dofs_idx_local=env.motors_dof_idx)

    print(f"Next state trajectory Robot joint obs: {new_obs}")

    while True:
        pass


if __name__ == "__main__":
    main()


