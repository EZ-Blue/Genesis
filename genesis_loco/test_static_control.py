import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv

import sys

sys.path.append('/home/ez/Documents/Genesis/genesis_loco')

from integration.data_bridge import LocoMujocoDataBridge


def main():

    gs.init(backend=gs.cuda)

    env = SkeletonHumanoidEnv(
        num_envs=1,
        use_box_feet=False,
        show_viewer=True,
        sim_options=gs.options.SimOptions(
                # dt=self.dt, 
                substeps=2,
                gravity=(0.0,0.0,0.0)
            ),
    )

    

    data_bridge = LocoMujocoDataBridge(env)
    trajectory_name = 'walk'
    tj = data_bridge.load_trajectory(trajectory_name)

    state = data_bridge.get_trajectory_state(50)
    
    data_bridge.apply_trajectory_state(state)

    for key in state:
        print(f"{key}: {state[key]}, length: {len(state[key])}")

    obs = env._get_observations()[0]

    print(f"get_obs: {env._get_observations()}")

    static_positions = env.robot.get_dofs_position([idx - env.robot.dof_start for idx in env.motors_dof_idx])

    for i in range(1000):

        env.robot.control_dofs_position(static_positions, dofs_idx_local=env.motors_dof_idx)

        if i % 500 == 0:
            print(f"static_positions: {env.robot.get_dofs_position([idx - env.robot.dof_start for idx in env.motors_dof_idx])}")
            print(f"Observations: {env._get_observations()}")

        env.scene.step()



if __name__ == "__main__":
    main()
