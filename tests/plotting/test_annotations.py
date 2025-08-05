import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from erlab.plotting.annotations import (
    _SIFormatter,
    copy_mathtext,
    mark_points,
    mark_points_outside,
    property_labels,
    scale_units,
    set_titles,
    set_xlabels,
    set_ylabels,
    sizebar,
)


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


def test_scale_units_x_axis_prefix():
    fig, ax = plt.subplots()
    ax.plot([0, 1e-3, 2e-3], [0, 1, 2])
    ax.set_xlabel("Voltage (V)")
    # Apply scale_units to rescale x-axis to mV
    scale_units(ax, "x", si=-3, prefix=True)
    # Check that the label now contains 'mV'
    assert "mV" in ax.get_xlabel()
    # Check that the formatter is _SIFormatter
    assert isinstance(
        ax.xaxis.get_major_formatter(), type(ax.xaxis.get_major_formatter())
    )
    plt.close(fig)


def test_scale_units_y_axis_power():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1e-6, 2e-6])
    ax.set_ylabel("Current (A)")
    # Apply scale_units to rescale y-axis to μA using power notation
    scale_units(ax, "y", si=-6, prefix=True, power=True)
    # Check that the label now contains scientific notation
    assert "10" in ax.get_ylabel() or "μA" in ax.get_ylabel()
    plt.close(fig)


def test_scale_units_no_unit_in_label():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 2])
    ax.set_xlabel("Position")
    # Should not raise or change label if no unit is present
    scale_units(ax, "x", si=3, prefix=True)
    assert ax.get_xlabel() == "Position"
    plt.close(fig)


def test_scale_units_iterable_axes():
    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        ax.plot([0, 1, 2], [0, 1, 2])
        ax.set_ylabel("Energy (eV)")
    # Apply to both axes
    scale_units(axs, "y", si=-3, prefix=True)
    for ax in axs:
        assert "meV" in ax.get_ylabel()
    plt.close(fig)


def test_scale_units_axis_formatter_type():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 2])
    ax.set_xlabel("Distance (m)")
    scale_units(ax, "x", si=6, prefix=True)
    assert isinstance(ax.xaxis.get_major_formatter(), _SIFormatter)
    plt.close(fig)


def test_set_titles():
    fig, axs = plt.subplots(1, 3)
    titles = ["First", "Second", "Third"]
    set_titles(axs, titles)
    for ax, expected in zip(axs, titles, strict=True):
        assert ax.get_title() == expected
    plt.close(fig)

    fig, axs = plt.subplots(2, 2)
    set_titles(axs, "Common Title")
    for ax in axs.flat:
        assert ax.get_title() == "Common Title"
    plt.close(fig)


def test_set_xlabels():
    fig, axs = plt.subplots(1, 3)
    xlabels = ["X1", "X2", "X3"]
    set_xlabels(axs, xlabels)
    for ax, expected in zip(axs, xlabels, strict=True):
        assert ax.get_xlabel() == expected
    plt.close(fig)

    fig, axs = plt.subplots(2, 2)
    set_xlabels(axs, "Common X Label")
    for ax in axs.flat:
        assert ax.get_xlabel() == "Common X Label"
    plt.close(fig)


def test_set_ylabels():
    fig, axs = plt.subplots(1, 3)
    ylabels = ["Y1", "Y2", "Y3"]
    set_ylabels(axs, ylabels)
    for ax, expected in zip(axs, ylabels, strict=True):
        assert ax.get_ylabel() == expected
    plt.close(fig)

    fig, axs = plt.subplots(2, 2)
    set_ylabels(axs, "Common Y Label")
    for ax in axs.flat:
        assert ax.get_ylabel() == "Common Y Label"
    plt.close(fig)


def test_sizebar_basic():
    fig, ax = plt.subplots()
    # Add a sizebar of 2 μm, axes in mm
    asb = sizebar(ax, value=2e-6, unit="m", si=-6, resolution=1e-3)
    assert isinstance(asb, AnchoredSizeBar)

    # Check that the sizebar artist is added to the axes
    found = any(isinstance(artist, type(asb)) for artist in ax.artists)
    assert found
    # Check that the label is correct
    assert asb.txt_label.get_text() == "2 μm"
    plt.close(fig)


def test_sizebar_with_label_override():
    fig, ax = plt.subplots()
    asb = sizebar(
        ax, value=5e-3, unit="m", si=-3, resolution=1e-3, label="custom label"
    )
    assert asb.txt_label.get_text() == "custom label"
    plt.close(fig)


def test_sizebar_with_decimals():
    fig, ax = plt.subplots()
    asb = sizebar(ax, value=1.2345e-3, unit="m", si=-3, resolution=1e-3, decimals=2)
    # Should round to 1.23 mm
    assert asb.txt_label.get_text().startswith("1.23")
    assert "mm" in asb.txt_label.get_text()
    plt.close(fig)


def test_mark_points_outside_x_axis_labels():
    fig, ax = plt.subplots()
    x = np.linspace(0, 8, 100)
    ax.plot(x, 0.1 * x)
    points = [1, 3, 5, 7]
    labels = ["A", "B", "C", "D"]

    mark_points_outside(points, labels, axis="x", ax=ax)

    # There should be a twiny axes
    assert len(fig.axes) == 2
    label_ax = fig.axes[1]
    # Check that the twiny axes has the correct tick locations and labels
    assert np.allclose(label_ax.get_xticks(), points)
    assert [tick.get_text() for tick in label_ax.get_xticklabels()] == [
        "$\\mathdefault{A}$",
        "$\\mathdefault{B}$",
        "$\\mathdefault{C}$",
        "$\\mathdefault{\\Delta}$",
    ]
    plt.close(fig)


def test_mark_points_outside_y_axis_labels():
    fig, ax = plt.subplots()
    y = np.linspace(0, 8, 100)
    ax.plot(0.1 * y, y)
    points = [2, 4, 6, 8]
    labels = ["X", "Y", "Z", "W"]

    mark_points_outside(points, labels, axis="y", ax=ax)

    # There should be a twiny axes for y
    assert len(fig.axes) == 2
    label_ax = fig.axes[1]
    # Check that the twiny axes has the correct tick locations and labels
    assert np.allclose(label_ax.get_yticks(), points)
    assert [tick.get_text() for tick in label_ax.get_yticklabels()] == [
        "$\\mathdefault{X}$",
        "$\\mathdefault{Y}$",
        "$\\mathdefault{Z}$",
        "$\\mathdefault{W}$",
    ]
    plt.close(fig)


def test_mark_points_outside_literal_labels():
    fig, ax = plt.subplots()
    points = [0, 1]
    labels = ["foo", "bar"]

    mark_points_outside(points, labels, axis="x", ax=ax, literal=True)

    label_ax = fig.axes[1]
    assert [tick.get_text() for tick in label_ax.get_xticklabels()] == ["foo", "bar"]
    plt.close(fig)


def test_mark_points_outside_iterable_axes():
    fig, axs = plt.subplots(1, 2)
    points = [1, 2]
    labels = ["A", "B"]

    mark_points_outside(points, labels, axis="x", ax=axs)

    # Should add a twiny axes to each subplot
    assert len(fig.axes) == 4
    for i in range(1):
        label_ax = fig.axes[2 + i]
        assert np.allclose(label_ax.get_xticks(), points)
        assert [tick.get_text() for tick in label_ax.get_xticklabels()] == [
            "$\\mathdefault{A}$",
            "$\\mathdefault{B}$",
        ]
    plt.close(fig)
