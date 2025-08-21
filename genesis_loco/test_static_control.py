import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv


def main():

    gs.init(backend=gs.cuda)

    env = SkeletonHumanoidEnv(
        num_envs=1,
        use_box_feet=False,
        show_viewer=True,
    )

    static_positions = env.robot.get_dofs_position([idx - env.robot.dof_start for idx in env.motors_dof_idx])

    for i in range(1000):

        env.robot.control_dofs_position(static_positions, dofs_idx_local=env.motors_dof_idx)

        env.scene.step()



if __name__ == "__main__":
    main()
