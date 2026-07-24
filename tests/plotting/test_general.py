import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import erlab.accessors.general as accessor_general
import erlab.plotting.general as plotting_general
from erlab.plotting.general import (
    clean_labels,
    fermiline,
    place_inset,
    plot_array,
    plot_slices,
)


@pytest.mark.parametrize(
    "x",
    [
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 3.0]),
    ],
    ids=("uniform", "nonuniform"),
)
def test_plot_array_validates_norm_before_adding_image(x) -> None:
    data = xr.DataArray(
        np.arange(1.0, 7.0).reshape(2, 3),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0], "x": x},
    )
    figure, axis = plt.subplots()
    try:
        with pytest.raises(ValueError, match="minvalue"):
            plot_array(data, ax=axis, vmax=0.0, colorbar=True)
        assert not axis.images
        assert figure.axes == [axis]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("invalid", [np.nan, np.inf], ids=("nan", "infinity"))
def test_plot_array_norm_ignores_nonfinite_data(invalid: float) -> None:
    data = xr.DataArray(
        np.array([[1.0, 2.0], [invalid, 4.0]]),
        dims=("y", "x"),
    )
    figure, axis = plt.subplots()
    try:
        image = plot_array(data, ax=axis)
        figure.canvas.draw()
        assert image.norm.vmin == 1.0
        assert image.norm.vmax == 4.0
    finally:
        plt.close(figure)


def test_validate_image_norm_allows_unresolved_limits(monkeypatch) -> None:
    norm = matplotlib.colors.Normalize()
    monkeypatch.setattr(norm, "autoscale_None", lambda _values: None)

    plotting_general._validate_image_norm(norm, np.arange(4.0))

    assert norm.vmin is None
    assert norm.vmax is None


def test_plot_slices_selects_slice_stack_once_per_map(monkeypatch) -> None:
    eV = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    data0 = xr.DataArray(
        np.arange(eV.size * y.size * x.size, dtype=float).reshape(
            eV.size, y.size, x.size
        ),
        dims=("eV", "y", "x"),
        coords={"eV": eV, "y": y, "x": x},
    )
    data1 = data0 + 100.0
    expected0 = data0.qsel(
        eV=[0.0, 2.0],
        eV_width=[0.1, 0.1],
        y=slice(1.0, 3.0),
        x=slice(1.0, 3.0),
    )
    expected1 = data1.qsel(
        eV=[0.0, 2.0],
        eV_width=[0.1, 0.1],
        y=slice(1.0, 3.0),
        x=slice(1.0, 3.0),
    )
    calls: list[dict[str, object]] = []
    original_qsel = accessor_general.SelectionAccessor.__call__

    def counted_qsel(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original_qsel(self, *args, **kwargs)

    monkeypatch.setattr(
        accessor_general.SelectionAccessor,
        "__call__",
        counted_qsel,
    )

    fig, axes = plot_slices(
        [data0, data1],
        eV=[0.0, 2.0],
        eV_width=[0.1, 0.1],
        xlim=(1.0, 3.0),
        ylim=(1.0, 3.0),
    )

    assert len(calls) == 2
    assert calls[0]["eV"] == [0.0, 2.0]
    assert calls[0]["eV_width"] == [0.1, 0.1]
    assert axes.shape == (2, 2)
    assert all(len(axis.images) == 1 for axis in axes.ravel())
    np.testing.assert_allclose(axes[0, 0].images[0].get_array(), expected0[0].values)
    np.testing.assert_allclose(axes[0, 1].images[0].get_array(), expected0[1].values)
    np.testing.assert_allclose(axes[1, 0].images[0].get_array(), expected1[0].values)
    np.testing.assert_allclose(axes[1, 1].images[0].get_array(), expected1[1].values)
    plt.close(fig)


def test_plot_slices_general(monkeypatch) -> None:
    # Create some sample data
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 11)

    data0 = xr.DataArray(
        np.random.default_rng(0).random((11, 11)), coords=[x, y], dims=["x", "y"]
    )
    data1 = xr.DataArray(
        np.random.default_rng(1).random((11, 11)), coords=[x, y], dims=["x", "y"]
    )
    maps = [data0, data1]

    # Test basic functionality
    fig, axes = plot_slices(maps)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2, 1)
    plt.close()

    # Test single input slicing
    fig, axes = plot_slices(data0, y=[0.2, 0.4], y_width=0.1)
    assert axes.shape == (1, 2)
    plt.close()

    # Test slice along another dimension
    fig, axes = plot_slices(maps, y=[0.2, 0.4], y_width=0.1, x=slice(0.1, 0.3))
    line = axes[0, 0].get_children()[0]
    assert axes.shape == (2, 2)
    assert np.allclose(line.get_xdata(), np.array([0.1, 0.2]))
    assert np.allclose(line.get_ydata(), np.array([0.03358558, 0.61538511]))
    plt.close()

    # Test axes input
    fig, axes = plt.subplots(1, 2)
    plot_slices(data0, axes=axes, y=[0.2, 0.4], y_width=0.1)
    line = axes[0].get_children()[0]
    assert np.allclose(line.get_xdata(), x)
    assert np.allclose(
        line.get_ydata(),
        np.array(
            [
                0.04097352,
                0.03358558,
                0.61538511,
                0.31024188,
                0.22715759,
                0.79632427,
                0.36511017,
                0.46004514,
                0.92742393,
                0.23237292,
                0.71921977,
            ]
        ),
    )
    plt.close()

    # Test axes manual input
    fig, axes = plt.subplots(1, 2)
    plot_slices(data0, axes=[axes[0], axes[1]], y=[0.2, 0.4], y_width=0.1)
    plt.close()

    # Test wrong dtype
    fig, ax = plt.subplots()
    with pytest.raises(
        TypeError, match=r"axes must be an iterable of matplotlib.axes.Axes"
    ):
        plot_slices(data0, axes=ax, y=[0.2, 0.4], y_width=0.1)
    plt.close()

    # Test input dimensions
    with pytest.raises(
        ValueError, match="All input arrays must have the same dimensions"
    ):
        plot_slices([data0, data0[0]], x=0.1, y=[0.2, 0.4])
    with pytest.raises(ValueError, match="The data to plot must be 1D or 2D"):
        plot_slices(data0, x=0.1, y=[0.2, 0.4])
    with pytest.raises(ValueError, match="Only one slice dimension is allowed"):
        plot_slices(data0, x=[0.1, 0.2], y=[0.2, 0.4])

    # Test line plots with gradient
    fig, axes = plot_slices(maps, y=[0.1, 0.2, 0.3], transpose=True, gradient=True)
    assert axes[0, 0].lines[0].get_ydata().tolist() == x.tolist()
    assert axes.shape == (2, 3)
    plt.close()

    # Test xlim and ylim options
    fig, axes = plot_slices(maps, xlim=(0.2, 0.8), ylim=(0.3, 0.7))
    assert axes[0, 0].get_xlim() == (0.2, 0.8)
    assert axes[0, 0].get_ylim() == (0.3, 0.7)
    plt.close()

    fig, axes = plot_slices(maps, xlim=(0.2, None), ylim=(None, 0.7))
    assert axes[0, 0].get_xlim()[0] == 0.2
    assert axes[0, 0].get_ylim()[1] == 0.7
    plt.close()

    # Test 3D slicing
    test_darr = xr.DataArray(
        np.random.default_rng(0).random((11, 11, 11)),
        coords=[
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 11),
        ],
        dims=["x", "y", "z"],
    )
    fig, axes = plot_slices(
        [test_darr],
        x=0.1,
        y=[0.2, 0.4],
        x_width=0.2,
        y_width=0.2,
    )
    assert axes.shape == (1, 2)
    test_darr_sel = test_darr.qsel(x=0.1, y=[0.2, 0.4], x_width=0.2, y_width=0.2)
    np.testing.assert_allclose(
        axes[0, 0].lines[0].get_ydata(), test_darr_sel.isel(y=0).values
    )
    np.testing.assert_allclose(
        axes[0, 1].lines[0].get_ydata(), test_darr_sel.isel(y=1).values
    )
    plt.close()

    # Test cmap option
    fig, axes = plot_slices(maps, cmap="viridis")
    assert axes[0, 0].get_images()[0].get_cmap().name == "viridis"
    plt.close()

    # Test norm option
    norm = plt.Normalize(vmin=0, vmax=1)
    fig, axes = plot_slices(maps, norm=norm)
    assert axes[0, 0].get_images()[0].norm.vmin == 0
    assert axes[0, 0].get_images()[0].norm.vmax == 1
    plt.close()

    # Test norm array
    norm = [plt.Normalize(vmin=0, vmax=0.5), plt.Normalize(vmin=0, vmax=1)]
    fig, axes = plot_slices(maps, norm=norm)
    assert axes[0, 0].get_images()[0].norm.vmax == 0.5
    assert axes[1, 0].get_images()[0].norm.vmax == 1.0
    plt.close()

    # Test colorbars and order
    fig, axes = plot_slices(maps, colorbar="right", order="F")
    plt.close()
    fig, axes = plot_slices(maps, colorbar="rightspan", order="F")
    plt.close()
    fig, axes = plot_slices(maps, colorbar="all", order="F")
    assert axes.shape == (1, 2)
    assert isinstance(axes[0, 1], plt.Axes)
    assert axes[0, 1].get_title() == ""
    plt.close()

    # Test same_limits
    fig, axes = plot_slices(maps, same_limits="all", order="F")
    assert axes[0, 0].get_images()[0].norm.vmin == axes[0, 1].get_images()[0].norm.vmin
    assert axes[0, 0].get_images()[0].norm.vmax == axes[0, 1].get_images()[0].norm.vmax
    plt.close()

    fig, axes = plot_slices(maps, same_limits="row", order="F")
    assert axes[0, 0].get_images()[0].norm.vmin == axes[0, 1].get_images()[0].norm.vmin
    assert axes[0, 0].get_images()[0].norm.vmax == axes[0, 1].get_images()[0].norm.vmax
    plt.close()

    fig, axes = plot_slices(maps, same_limits="col", order="F")
    assert axes[0, 0].get_images()[0].norm.vmin != axes[0, 1].get_images()[0].norm.vmin
    assert axes[0, 0].get_images()[0].norm.vmax != axes[0, 1].get_images()[0].norm.vmax
    plt.close()

    # Test other options
    fig, axes = plot_slices(maps, figsize=(8, 6), same_limits=True, order="F")
    assert tuple(fig.get_size_inches()) == (8, 6)
    assert axes[0, 0].get_images()[0].norm.vmin == axes[0, 1].get_images()[0].norm.vmin
    assert axes[0, 0].get_images()[0].norm.vmax == axes[0, 1].get_images()[0].norm.vmax
    plt.close()


@pytest.mark.parametrize("order", ["C", "F"], ids=["C order", "F order"])
@pytest.mark.parametrize("transpose", [False, True], ids=["no transpose", "transpose"])
@pytest.mark.parametrize("colorbar", ["none", "right", "rightspan", "all"])
@pytest.mark.parametrize(
    "same_limits",
    [False, True, "row", "col", "all"],
    ids=[
        "no limits",
        "same limits",
        "same row limits",
        "same column limits",
        "same all limits",
    ],
)
def test_plot_slices_2d_options(order, transpose, colorbar, same_limits):
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 8)
    arr1 = xr.DataArray(
        np.random.default_rng(0).random((8, 8)), coords=[x, y], dims=["x", "y"]
    )
    arr2 = xr.DataArray(
        np.random.default_rng(1).random((8, 8)), coords=[x, y], dims=["x", "y"]
    )
    maps = [arr1, arr2]
    fig, axes = plot_slices(
        maps,
        transpose=transpose,
        colorbar=colorbar,
        same_limits=same_limits,
        order=order,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    if order == "F":
        assert axes.shape == (1, 2)
    else:
        assert axes.shape == (2, 1)
    plt.close(fig)


def test_plot_slices_gradient_and_gradient_kw():
    x = np.linspace(0, 1, 10)
    arr = xr.DataArray(np.sin(2 * np.pi * x), coords=[x], dims=["x"], name="sin")
    fig, axes = plot_slices(
        [arr], gradient=True, gradient_kw={"alpha": 0.5, "color": "green"}
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)


def test_plot_slices_1d_and_2d_mix_error():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    arr1 = xr.DataArray(
        np.random.default_rng(0).random((5, 5)), coords=[x, y], dims=["x", "y"]
    )
    arr2 = xr.DataArray(np.random.default_rng(1).random(5), coords=[x], dims=["x"])
    with pytest.raises(
        ValueError, match="All input arrays must have the same dimensions"
    ):
        plot_slices([arr1, arr2])


def test_plot_slices_invalid_axes_type():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    arr = xr.DataArray(
        np.random.default_rng(0).random((5, 5)), coords=[x, y], dims=["x", "y"]
    )
    fig, ax = plt.subplots()
    with pytest.raises(
        TypeError, match=r"axes must be an iterable of matplotlib.axes.Axes"
    ):
        plot_slices(arr, axes=ax)
    plt.close(fig)


def test_plot_slices_with_custom_axes_and_order():
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    arr1 = xr.DataArray(
        np.random.default_rng(0).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )
    arr2 = xr.DataArray(
        np.random.default_rng(1).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )
    fig, axes = plt.subplots(2, 1)
    plot_slices([arr1, arr2], axes=axes, order="F")
    plt.close(fig)


def test_plot_slices_with_cmap_and_norm_iterables():
    x = np.linspace(0, 1, 7)
    y = np.linspace(0, 1, 7)
    arr1 = xr.DataArray(
        np.random.default_rng(0).random((7, 7)), coords=[x, y], dims=["x", "y"]
    )
    arr2 = xr.DataArray(
        np.random.default_rng(1).random((7, 7)), coords=[x, y], dims=["x", "y"]
    )
    cmaps = ["viridis", "plasma"]
    norms = [plt.Normalize(0, 0.5), plt.Normalize(0, 1)]
    fig, axes = plot_slices([arr1, arr2], cmap=cmaps, norm=norms)
    assert axes.shape == (2, 1)
    plt.close(fig)


def test_plot_slices_with_annotate_kw_and_subplot_kw():
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 4)
    arr = xr.DataArray(
        np.random.default_rng(0).random((4, 4)), coords=[x, y], dims=["x", "y"]
    )
    fig, _axes = plot_slices(
        arr, annotate_kw={"fontsize": 8}, subplot_kw={"sharex": "all"}
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_slices_with_axis_and_show_all_labels():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    arr = xr.DataArray(
        np.random.default_rng(0).random((5, 5)), coords=[x, y], dims=["x", "y"]
    )
    fig, axes = plot_slices(arr, axis="equal", show_all_labels=True)
    for ax in axes.flat:
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
    plt.close(fig)


def test_plot_slices_width_length_mismatch() -> None:
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 11)
    data = xr.DataArray(
        np.random.default_rng(0).random((11, 11)), coords=[x, y], dims=["x", "y"]
    )

    with pytest.raises(ValueError, match="Number of widths must match"):
        plot_slices(data, y=[0.2, 0.4], y_width=[0.1])


def test_plot_slices_with_invalid_slice_dim():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    arr = xr.DataArray(
        np.random.default_rng(0).random((5, 5)), coords=[x, y], dims=["x", "y"]
    )
    with pytest.raises(ValueError, match="Only one slice dimension is allowed"):
        plot_slices(arr, x=[0.1, 0.2], y=[0.3, 0.4])


def test_plot_slices_with_invalid_ndim():
    arr = xr.DataArray(np.random.default_rng(0).random((2, 2, 2)), dims=["a", "b", "c"])
    with pytest.raises(ValueError, match="The data to plot must be 1D or 2D"):
        plot_slices(arr, a=0.1, b=[0.2, 0.3], c=0.4)


def test_plot_slices_with_colorbar_kw_and_hide_colorbar_ticks_false():
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    arr = xr.DataArray(
        np.random.default_rng(0).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )
    fig, _axes = plot_slices(
        arr, colorbar="all", colorbar_kw={"shrink": 0.5}, hide_colorbar_ticks=False
    )
    plt.close(fig)


def test_plot_slices_with_crop_false_and_xlim_ylim():
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    arr = xr.DataArray(
        np.random.default_rng(0).random((10, 10)), coords=[x, y], dims=["x", "y"]
    )
    fig, axes = plot_slices(arr, crop=False, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    for ax in axes.flat:
        assert ax.get_xlim() == (-0.5, 0.5)
        assert ax.get_ylim() == (-0.5, 0.5)
    plt.close(fig)


def test_plot_slices_with_1d_line_ignores_cmap_color():
    x = np.linspace(0, 1, 12)
    arr = xr.DataArray(np.cos(2 * np.pi * x), coords=[x], dims=["x"], name="cos")
    fig, axes = plot_slices(arr, cmap="viridis")
    assert axes[0, 0].lines[0].get_color() != "viridis"
    plt.close(fig)


def test_plot_slices_with_1d_gradient_auto_color():
    x = np.linspace(0, 1, 12)
    arr = xr.DataArray(np.cos(2 * np.pi * x), coords=[x], dims=["x"], name="cos")
    fig, axes = plot_slices(arr, gradient=True, line_kw={"color": "red"})
    assert axes[0, 0].lines[0].get_color() == "red"
    np.testing.assert_allclose(
        axes[0, 0].images[0].cmap(1.0),
        matplotlib.colors.to_rgba("red"),
        rtol=1e-7,
    )
    plt.close(fig)


def test_plot_slices_with_1d_line_style_kwargs_and_aliases() -> None:
    x = np.linspace(0, 1, 12)
    arr = xr.DataArray(np.cos(2 * np.pi * x), coords=[x], dims=["x"], name="cos")
    fig, axes = plot_slices(
        arr,
        color="red",
        line_kw={
            "c": "blue",
            "ls": ":",
            "lw": 1.5,
            "marker": "o",
            "ms": 6.0,
            "mfc": "yellow",
            "mec": "black",
        },
    )
    line = axes[0, 0].lines[0]
    assert line.get_color() == "blue"
    assert line.get_linestyle() == ":"
    assert line.get_linewidth() == 1.5
    assert line.get_marker() == "o"
    assert line.get_markersize() == 6.0
    assert line.get_markerfacecolor() == "yellow"
    assert line.get_markeredgecolor() == "black"
    plt.close(fig)


def test_plot_slices_with_1d_per_panel_line_styles() -> None:
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    arr1 = xr.DataArray(
        np.random.default_rng(0).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )
    arr2 = xr.DataArray(
        np.random.default_rng(1).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )

    fig, axes = plot_slices(
        [arr1, arr2],
        y=[0.2, 0.4],
        line_kw=(
            (
                {"color": "red", "linestyle": "-", "marker": "o"},
                {"color": "blue", "linestyle": "--", "marker": "s"},
            ),
            (
                {"color": "green", "linestyle": ":", "marker": "^"},
                {"color": "black", "linestyle": "-.", "marker": "x"},
            ),
        ),
        line_order="C",
    )
    assert axes[0, 0].lines[0].get_color() == "red"
    assert axes[0, 1].lines[0].get_color() == "blue"
    assert axes[1, 0].lines[0].get_color() == "green"
    assert axes[1, 1].lines[0].get_color() == "black"
    assert axes[0, 1].lines[0].get_linestyle() == "--"
    assert axes[1, 0].lines[0].get_marker() == "^"
    plt.close(fig)

    fig, axes = plot_slices(
        [arr1, arr2],
        y=[0.2, 0.4],
        order="F",
        line_kw=(
            {"color": "red"},
            {"color": "blue"},
            {"color": "green"},
            {"color": "black"},
        ),
        line_order="F",
    )
    assert axes[0, 0].lines[0].get_color() == "red"
    assert axes[0, 1].lines[0].get_color() == "blue"
    assert axes[1, 0].lines[0].get_color() == "green"
    assert axes[1, 1].lines[0].get_color() == "black"
    plt.close(fig)


def test_plot_slices_with_1d_flat_line_style_order_and_empty_values() -> None:
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    arr1 = xr.DataArray(
        np.random.default_rng(0).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )
    arr2 = xr.DataArray(
        np.random.default_rng(1).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )

    fig, axes = plot_slices([arr1, arr2], y=[0.2, 0.4], line_kw=[])
    assert axes[0, 0].lines
    plt.close(fig)

    fig, axes = plot_slices(
        [arr1, arr2],
        y=[0.2, 0.4],
        line_kw=(
            {"color": "red"},
            {"color": "blue"},
            {"color": "green"},
            {"color": "black"},
        ),
        line_order="C",
    )
    assert axes[0, 0].lines[0].get_color() == "red"
    assert axes[0, 1].lines[0].get_color() == "blue"
    assert axes[1, 0].lines[0].get_color() == "green"
    assert axes[1, 1].lines[0].get_color() == "black"
    plt.close(fig)


def test_plot_slices_with_1d_line_kwargs_aliases_from_plot_kwargs() -> None:
    x = np.linspace(0, 1, 12)
    arr = xr.DataArray(np.cos(2 * np.pi * x), coords=[x], dims=["x"], name="cos")
    fig, axes = plot_slices(arr, c="red", lw=2.5)
    line = axes[0, 0].lines[0]
    assert line.get_color() == "red"
    assert line.get_linewidth() == 2.5
    plt.close(fig)


def test_plot_slices_rejects_line_options_for_2d_output() -> None:
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    arr = xr.DataArray(
        np.random.default_rng(0).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )
    with pytest.raises(ValueError, match="Line styling options only apply to 1D"):
        plot_slices(arr, line_kw={"lw": 2.0})


def test_plot_slices_accepts_empty_line_kwargs_for_2d_output() -> None:
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    arr = xr.DataArray(
        np.random.default_rng(0).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )
    fig, axes = plot_slices(arr, line_kw={})
    assert axes[0, 0].images
    plt.close(fig)


def test_plot_slices_rejects_invalid_1d_line_kw_shapes() -> None:
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    arr = xr.DataArray(
        np.random.default_rng(0).random((6, 6)), coords=[x, y], dims=["x", "y"]
    )

    with pytest.raises(TypeError, match="line_kw values must be mappings"):
        plot_slices(arr, y=[0.2], line_kw=("red",))

    with pytest.raises(ValueError, match="line_kw must be a mapping"):
        plot_slices(
            arr,
            y=[0.2, 0.4],
            line_kw=({"color": "red"}, {"color": "blue"}, {"color": "green"}),
        )


def test_plot_slices_with_2d_and_crop_and_limits():
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    arr = xr.DataArray(
        np.random.default_rng(0).random((20, 20)), coords=[x, y], dims=["x", "y"]
    )
    fig, axes = plot_slices(arr, crop=True, xlim=(-1, 1), ylim=(-1, 1))
    for ax in axes.flat:
        assert ax.get_xlim() == (-1, 1)
        assert ax.get_ylim() == (-1, 1)
    plt.close(fig)


def test_plot_slices_with_annotate_false():
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 8)
    arr = xr.DataArray(
        np.random.default_rng(0).random((8, 8)), coords=[x, y], dims=["x", "y"]
    )
    fig, _axes = plot_slices(arr, annotate=False)
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs", [{}, {"vmin": 0.1}, {"vmax": 0.1}, {"vmin": 0.1, "vmax": 0.9}]
)
@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("ylim", [None, 0.5, (-0.3, 0.7)])
@pytest.mark.parametrize("xlim", [None, 0.5, (-0.3, 0.7)])
@pytest.mark.parametrize("colorbar", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        xr.DataArray(
            np.random.default_rng(0).random((11, 11)),
            coords=[np.linspace(-1, 1, 11), np.linspace(-1, 1, 11)],
            dims=["x", "y"],
        ),
        xr.DataArray(
            np.random.default_rng(0).random((11, 11)),
            coords=[np.linspace(-1, 1, 11) ** 3, np.linspace(-1, 1, 11)],
            dims=["x", "y"],
        ),
    ],
)
def test_plot_array(data, colorbar, xlim, ylim, crop, kwargs) -> None:
    _, ax = plt.subplots()
    plot_array(data, colorbar=colorbar, xlim=xlim, ylim=ylim, crop=crop, **kwargs)
    assert ax.get_xlabel() == "y"
    assert ax.get_ylabel() == "x"

    if xlim is not None:
        if xlim == 0.5:
            assert ax.get_xlim() == (-0.5, 0.5)
        else:
            assert ax.get_xlim() == xlim
    if ylim is not None:
        if ylim == 0.5:
            assert ax.get_ylim() == (-0.5, 0.5)
        else:
            assert ax.get_ylim() == ylim
    plt.close()


def test_place_inset_basic():
    fig, ax = plt.subplots()
    inset_ax = place_inset(ax, width=1.0, height=1.0)
    assert isinstance(inset_ax, plt.Axes)
    assert inset_ax is not ax
    plt.close(fig)


def test_place_inset_relative_size():
    fig, ax = plt.subplots()
    inset_ax = place_inset(ax, width="50%", height="50%")
    assert isinstance(inset_ax, plt.Axes)
    plt.close(fig)


@pytest.mark.parametrize(
    "loc",
    [
        "upper left",
        "upper center",
        "upper right",
        "center left",
        "center",
        "center right",
        "lower left",
        "lower center",
        "lower right",
    ],
)
def test_place_inset_locations(loc):
    fig, ax = plt.subplots()
    inset_ax = place_inset(ax, width=0.5, height=0.5, loc=loc)
    assert isinstance(inset_ax, plt.Axes)
    plt.close(fig)


def test_place_inset_with_pad_tuple():
    fig, ax = plt.subplots()
    inset_ax = place_inset(ax, width=0.5, height=0.5, pad=(0.2, 0.3))
    assert isinstance(inset_ax, plt.Axes)
    plt.close(fig)


def test_place_inset_passes_kwargs():
    fig, ax = plt.subplots()
    inset_ax = place_inset(ax, width=0.5, height=0.5, facecolor="red")
    assert isinstance(inset_ax, plt.Axes)
    # Check that the facecolor is set (axes patch color)
    fc = inset_ax.patch.get_facecolor()
    assert np.allclose(fc[:3], matplotlib.colors.to_rgb("red"))
    plt.close(fig)


def test_fermiline_horizontal_and_vertical():
    fig, ax = plt.subplots()
    # Horizontal line at y=0.0
    line_h = fermiline(ax, value=0.0, orientation="h")
    assert isinstance(line_h, matplotlib.lines.Line2D)
    assert np.allclose(line_h.get_ydata(), [0.0, 0.0])
    # Vertical line at x=1.0
    line_v = fermiline(ax, value=1.0, orientation="v")
    assert isinstance(line_v, matplotlib.lines.Line2D)
    assert np.allclose(line_v.get_xdata(), [1.0, 1.0])
    plt.close(fig)


def test_fermiline_with_custom_kwargs():
    fig, ax = plt.subplots()
    line = fermiline(ax, value=2.0, orientation="h", color="red", lw=2, ls="--")
    assert line.get_color() == "red"
    assert line.get_linewidth() == 2
    assert line.get_linestyle() == "--"
    plt.close(fig)


def test_fermiline_iterable_axes():
    fig, axs = plt.subplots(2)
    lines = fermiline(axs, value=0.5, orientation="v", color="blue")
    assert isinstance(lines, list)
    assert all(isinstance(ln, matplotlib.lines.Line2D) for ln in lines)
    for ln in lines:
        assert ln.get_color() == "blue"
        assert np.allclose(ln.get_xdata(), [0.5, 0.5])
    plt.close(fig)


def test_fermiline_invalid_orientation():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="`orientation` must be either 'v' or 'h'"):
        fermiline(ax, value=0.0, orientation="invalid")
    plt.close(fig)


def test_clean_labels():
    fig, axes = plt.subplots(2, 2)
    # Set custom labels to check if they are removed
    for ax in axes.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    clean_labels(axes)
    # Only bottom row should have xlabel, only left column should have ylabel
    for i, ax in enumerate(axes.flat):
        row, col = divmod(i, 2)
        if row == 1:
            assert ax.get_xlabel() != ""
        else:
            assert ax.get_xlabel() == ""
        if col == 0:
            assert ax.get_ylabel() != ""
        else:
            assert ax.get_ylabel() == ""
    plt.close(fig)
