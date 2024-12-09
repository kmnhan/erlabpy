import matplotlib.pyplot as plt
import numpy as np
import pytest

from erlab.plotting.annotations import copy_mathtext, mark_points, property_labels


def test_mark_points() -> None:
    _, ax = plt.subplots()
    x = np.linspace(0, 8, 100)
    ax.plot(x, 0.1 * x)
    # Mark some points on the plot
    points = [1, 3, 5, 7]
    labels = ["G", "D", "K", "L"]

    mark_points(points, labels, ax=ax)

    # Check if the labels are above the horizontal axis
    for i in range(len(points)):
        assert ax.texts[i].get_position()[1] == 0.0

    # Check if the labels have the correct font properties
    for i in range(len(points)):
        assert ax.texts[i].get_fontweight() == "normal"

    # Check if the labels have the correct offset
    for i in range(len(points)):
        assert ax.texts[i].get_position()[0] == points[i]
        assert ax.texts[i].get_position()[1] == 0.0

    # Check if the labels have the correct alignment
    for i in range(len(points)):
        assert ax.texts[i].get_horizontalalignment() == "center"
        assert ax.texts[i].get_verticalalignment() == "baseline"

    plt.close()


@pytest.mark.parametrize("outline", [True, False])
def test_copy_mathtext(outline) -> None:
    assert copy_mathtext("$c_1$", svg=True, outline=outline).startswith(
        '<?xml version="1.0" encoding="utf-8"'
    )


def test_property_labels() -> None:
    values = {
        "T": [1.234, 2.345, 3.456],
        "eV": [0.1, 0.2, 0.3],
        "t": [0.01, 0.02, 0.03],
    }
    expected_labels = [
        "$T = 1230$ mK\n$E-E_F = 100$ meV\n$t = 10$ ms",
        "$T = 2340$ mK\n$E-E_F = 200$ meV\n$t = 20$ ms",
        "$T = 3460$ mK\n$E-E_F = 300$ meV\n$t = 30$ ms",
    ]
    labels = property_labels(values, decimals=-1, si=-3)
    assert labels == expected_labels
