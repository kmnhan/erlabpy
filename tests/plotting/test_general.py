import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from erlab.plotting.general import plot_slices


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

    # Test line plots with gradient
    fig, axes = plot_slices(maps, y=[0.1, 0.2, 0.3], transpose=True, gradient=True)
    assert axes[0, 0].lines[0].get_ydata().tolist() == x.tolist()
    assert axes.shape == (2, 3)

    # Test xlim and ylim options
    fig, axes = plot_slices(maps, xlim=(0.2, 0.8), ylim=(0.3, 0.7))
    assert axes[0, 0].get_xlim() == (0.2, 0.8)
    assert axes[0, 0].get_ylim() == (0.3, 0.7)

    # Test cmap option
    fig, axes = plot_slices(maps, cmap="viridis")
    assert axes[0, 0].get_images()[0].get_cmap().name == "viridis"

    # Test norm option
    norm = plt.Normalize(vmin=0, vmax=1)
    fig, axes = plot_slices(maps, norm=norm)
    assert axes[0, 0].get_images()[0].norm.vmin == 0
    assert axes[0, 0].get_images()[0].norm.vmax == 1

    # Test norm array
    norm = [plt.Normalize(vmin=0, vmax=0.5), plt.Normalize(vmin=0, vmax=1)]
    fig, axes = plot_slices(maps, norm=norm)
    assert axes[0, 0].get_images()[0].norm.vmax == 0.5
    assert axes[1, 0].get_images()[0].norm.vmax == 1.0

    # Test colorbars and order
    fig, axes = plot_slices(maps, colorbar="right", order="F")
    fig, axes = plot_slices(maps, colorbar="rightspan", order="F")
    fig, axes = plot_slices(maps, colorbar="all", order="F")
    assert axes.shape == (1, 2)
    assert isinstance(axes[0, 1], plt.Axes)
    assert axes[0, 1].get_title() == ""

    # Test same_limits
    fig, axes = plot_slices(maps, same_limits=True, order="F")
    assert axes[0, 0].get_images()[0].norm.vmin == axes[0, 1].get_images()[0].norm.vmin
    assert axes[0, 0].get_images()[0].norm.vmax == axes[0, 1].get_images()[0].norm.vmax

    # Test other options
    fig, axes = plot_slices(maps, figsize=(8, 6), same_limits=True, order="F")
    assert tuple(fig.get_size_inches()) == (8, 6)
    assert axes[0, 0].get_images()[0].norm.vmin == axes[0, 1].get_images()[0].norm.vmin
    assert axes[0, 0].get_images()[0].norm.vmax == axes[0, 1].get_images()[0].norm.vmax

    plt.close()
