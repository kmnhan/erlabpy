import matplotlib.pyplot as plt
import numpy as np
import pytest
from erlab.plotting.annotations import copy_mathtext, mark_points


def test_mark_points():
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
def test_copy_mathtext(outline):
    assert copy_mathtext("$c_1$", svg=True, outline=outline).startswith(
        '<?xml version="1.0" encoding="utf-8"'
    )
