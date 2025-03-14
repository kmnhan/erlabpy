import matplotlib.pyplot as plt

import erlab.lattice
from erlab.plotting.atoms import CrystalProperty


def test_crystalproperty():
    prop = CrystalProperty.from_fractional(
        {
            "C": [
                (0.0, 0.0, 0.0),
                (2 / 3, 1 / 3, 1 / 3),
                (1 / 3, 2 / 3, 2 / 3),
            ],
            "Gd": [
                (0.0, 0.0, 0.25833),
                (0.0, 0.0, 0.74167),
                (2 / 3, 1 / 3, 0.59166),
                (2 / 3, 1 / 3, 0.07500),
                (1 / 3, 2 / 3, 0.92500),
                (1 / 3, 2 / 3, 0.40834),
            ],
            "e": [
                (0.0, 0.0, 1 / 2),
                (2 / 3, 1 / 3, 5 / 6),
                (1 / 3, 2 / 3, 1 / 6),
            ],
        },
        avec=erlab.lattice.abc2avec(3.64028, 3.64028, 18.48817, 90, 90, 120),
        radii=(0.7, 1.8, 1.0),
        colors=("#888888", "#55cc55", "#cccc00"),
        repeat=(5, 5, 1),
        bounds={"x": (-10, 10), "y": (-10, 10), "z": (-16, 16)},
        mask=None,
        r_factor=0.4,
    )

    prop.add_bonds("C", "Gd", 0, 2.6)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    prop.plot(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_proj_type("persp", focal_length=1e10)
    ax.set_aspect("equal")
    ax.view_init(30, 30, 45)

    plt.close()
