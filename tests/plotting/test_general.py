import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from erlab.plotting.general import plot_array, plot_slices


def test_plot_slices():
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
    fig, axes = plot_slices(
        [
            xr.DataArray(
                np.random.default_rng(0).random((11, 11, 11)),
                coords=[
                    np.linspace(0, 1, 11),
                    np.linspace(0, 1, 11),
                    np.linspace(0, 1, 11),
                ],
                dims=["x", "y", "z"],
            )
        ],
        x=0.1,
        y=[0.2, 0.4],
        x_width=0.2,
        y_width=0.2,
    )
    assert axes.shape == (1, 2)
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
    fig, axes = plot_slices(maps, same_limits=True, order="F")
    assert axes[0, 0].get_images()[0].norm.vmin == axes[0, 1].get_images()[0].norm.vmin
    assert axes[0, 0].get_images()[0].norm.vmax == axes[0, 1].get_images()[0].norm.vmax
    plt.close()

    # Test other options
    fig, axes = plot_slices(maps, figsize=(8, 6), same_limits=True, order="F")
    assert tuple(fig.get_size_inches()) == (8, 6)
    assert axes[0, 0].get_images()[0].norm.vmin == axes[0, 1].get_images()[0].norm.vmin
    assert axes[0, 0].get_images()[0].norm.vmax == axes[0, 1].get_images()[0].norm.vmax
    plt.close()


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
def test_plot_array(data, colorbar, xlim, ylim, crop, kwargs):
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
