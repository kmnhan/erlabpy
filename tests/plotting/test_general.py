import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from erlab.plotting.general import place_inset, plot_array, plot_slices


def test_plot_slices_general() -> None:
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
        TypeError, match="axes must be an iterable of matplotlib.axes.Axes"
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
        TypeError, match="axes must be an iterable of matplotlib.axes.Axes"
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
    fig, axes = plot_slices(
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
    fig, axes = plot_slices(
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


def test_plot_slices_with_1d_line_and_cmap_color():
    x = np.linspace(0, 1, 12)
    arr = xr.DataArray(np.cos(2 * np.pi * x), coords=[x], dims=["x"], name="cos")
    fig, axes = plot_slices(arr, cmap="red")
    plt.close(fig)


def test_plot_slices_with_1d_gradient_auto_color():
    x = np.linspace(0, 1, 12)
    arr = xr.DataArray(np.cos(2 * np.pi * x), coords=[x], dims=["x"], name="cos")
    fig, axes = plot_slices(arr, gradient=True)
    plt.close(fig)


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
    fig, axes = plot_slices(arr, annotate=False)
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
