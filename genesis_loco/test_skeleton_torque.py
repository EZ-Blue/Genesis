import genesis as gs



def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True
    )


    plane = scene.add_entity(
        gs.morphs.Plane()
    )

    skeleton = scene.add_entity(
        gs.morphs.MJCF(file='/home/ez/Documents/Genesis/genesis_loco/skeleton/skeleton_torque.xml')
    )

    scene.build()

    for i in range(1000):
        scene.step()

if __name__ == "__main__":
    main()