import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

from erlab.plotting.bz import plot_bz, plot_hex_bz


def test_plot_hex_bz():
    fig, ax = plt.subplots()
    # Offset and rotate
    offset = (1.0, 2.0)
    rotate = 30.0
    patch = plot_hex_bz(a=2.0, rotate=rotate, offset=offset, ax=ax)
    np.testing.assert_allclose(patch.xy, offset)
    assert np.isclose(patch.orientation, np.deg2rad(rotate))

    # Reciprocal vs real space
    patch_r = plot_hex_bz(a=2.0, reciprocal=True, ax=ax)
    patch_real = plot_hex_bz(a=2.0, reciprocal=False, ax=ax)
    assert not np.isclose(patch_r.radius, patch_real.radius)

    # Kwargs / abbreviations
    patch_kw = plot_hex_bz(a=2.0, ax=ax, ls="-.", lw=2.5, ec="red")
    assert patch_kw.get_linestyle() == "-."
    assert patch_kw.get_linewidth() == 2.5
    assert patch_kw.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)

    # Iterable axes
    fig2, axs = plt.subplots(1, 2)
    patches = plot_hex_bz(a=2.0, ax=axs)
    assert isinstance(patches, list)
    assert all(hasattr(p, "get_path") for p in patches)
    assert all(p in ax_i.patches for p, ax_i in zip(patches, axs, strict=True))

    # Clip path
    clip = matplotlib.patches.Circle((0, 0), radius=1)
    patch_clip = plot_hex_bz(a=2.0, ax=ax, clip_path=clip)
    assert patch_clip.get_clip_path() is not None

    plt.close(fig)
    plt.close(fig2)


def test_plot_bz_basic_shapes():
    fig, ax = plt.subplots()
    a = 1.0
    basis_sq = np.array([[a, 0], [0, a]])
    patch_sq = plot_bz(basis_sq, ax=ax)
    assert isinstance(patch_sq, matplotlib.patches.Polygon)
    assert patch_sq.get_xy().shape[0] == 5
    expected_sq = np.array(
        [
            [np.pi / a, np.pi / a],
            [-np.pi / a, np.pi / a],
            [-np.pi / a, -np.pi / a],
            [np.pi / a, -np.pi / a],
        ]
    )
    for v in expected_sq:
        assert np.any(np.all(np.isclose(patch_sq.get_xy()[:-1], v, atol=1e-8), axis=1))

    basis_hex = np.array([[a, 0], [a / 2, a * np.sqrt(3) / 2]])
    patch_hex = plot_bz(basis_hex, ax=ax)
    assert isinstance(patch_hex, matplotlib.patches.Polygon)
    assert patch_hex.get_xy().shape[0] == 7
    plt.close(fig)


def test_plot_bz_options_and_iterable():
    # Rotate and offset
    fig1, ax1 = plt.subplots()
    a = 1.0
    basis = np.eye(2) * a
    rotate = 45.0
    offset = (2.0, -1.0)
    patch_rot = plot_bz(basis, rotate=rotate, offset=offset, ax=ax1)
    verts = patch_rot.get_xy()[:-1]
    center = np.mean(verts, axis=0)
    np.testing.assert_allclose(center, offset, atol=1e-1)

    # Reciprocal basis
    fig2, ax2 = plt.subplots()
    a2 = 2.0
    basis_rec = np.array([[2 * np.pi / a2, 0], [0, 2 * np.pi / a2]])
    patch_rec = plot_bz(basis_rec, reciprocal=True, ax=ax2)
    expected = np.array(
        [
            [np.pi / a2, np.pi / a2],
            [-np.pi / a2, np.pi / a2],
            [-np.pi / a2, -np.pi / a2],
            [np.pi / a2, -np.pi / a2],
        ]
    )
    for v in expected:
        assert np.any(np.all(np.isclose(patch_rec.get_xy()[:-1], v, atol=1e-8), axis=1))

    # Kwargs / abbreviations + fill/closed
    fig3, ax3 = plt.subplots()
    patch_kw = plot_bz(basis, ax=ax3, ls=":", lw=3.0, ec="blue", fill=True, closed=True)
    assert patch_kw.get_linestyle() == ":"
    assert patch_kw.get_linewidth() == 3.0
    assert patch_kw.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)
    assert patch_kw.get_fill()
    assert patch_kw.get_closed()

    # Iterable axes (manual list, as original test)
    fig4, axs = plt.subplots(1, 2)
    patches = [plot_bz(basis, ax=ax) for ax in axs]
    assert isinstance(patches, list)
    assert all(isinstance(p, matplotlib.patches.Polygon) for p in patches)
    assert all(p in ax.patches for p, ax in zip(patches, axs, strict=True))

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
