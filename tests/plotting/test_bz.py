import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

import erlab
import erlab.plotting as eplt
from erlab.plotting.bz import (
    plot_bz,
    plot_hex_bz,
    plot_in_plane_bz,
    plot_out_of_plane_bz,
)
from erlab.plotting.colors import axes_textcolor


def _assert_line_artists_match_segments(lines, segments):
    assert len(lines) == len(segments)
    for line, segment in zip(lines, segments, strict=True):
        np.testing.assert_allclose(line.get_xdata(), segment[:, 0])
        np.testing.assert_allclose(line.get_ydata(), segment[:, 1])


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


def test_plot_in_plane_bz_matches_lattice_segments():
    bvec = erlab.lattice.to_reciprocal(np.eye(3))
    bounds = (-3.5, 3.5, -3.5, 3.5)
    segments, _, _ = erlab.lattice.get_in_plane_bz(
        bvec, kz=0.0, angle=45.0, bounds=bounds, return_midpoints=True
    )

    fig, ax = plt.subplots()
    lines, vertex_artist, midpoint_artist = plot_in_plane_bz(
        bvec,
        kz=0.0,
        angle=45.0,
        bounds=bounds,
        ax=ax,
        c="tab:purple",
        lw=1.5,
        ls="-.",
    )

    _assert_line_artists_match_segments(lines, segments)
    assert vertex_artist is None
    assert midpoint_artist is None
    assert all(line.get_color() == "tab:purple" for line in lines)
    assert all(line.get_linewidth() == 1.5 for line in lines)
    assert all(line.get_linestyle() == "-." for line in lines)

    plt.close(fig)


def test_plot_out_of_plane_bz_infers_bounds_from_axes():
    bvec = erlab.lattice.to_reciprocal(np.eye(3))
    bounds = (-4.0, 4.0, -4.0, 4.0)
    segments, vertices, midpoints = erlab.lattice.get_out_of_plane_bz(
        bvec,
        k_parallel=0.0,
        angle=0.0,
        bounds=bounds,
        return_midpoints=True,
    )

    fig, ax = plt.subplots()
    ax.set_xlim(bounds[1], bounds[0])
    ax.set_ylim(bounds[3], bounds[2])
    lines, vertex_artist, midpoint_artist = plot_out_of_plane_bz(
        bvec,
        k_parallel=0.0,
        angle=0.0,
        ax=ax,
        vertices=True,
        midpoints=True,
        vertex_kwargs={"s": 12, "c": "tab:red"},
        midpoint_kwargs={"s": 18, "color": "tab:blue"},
    )

    _assert_line_artists_match_segments(lines, segments)
    assert vertex_artist is not None
    assert midpoint_artist is not None
    np.testing.assert_allclose(vertex_artist.get_offsets(), vertices)
    np.testing.assert_allclose(midpoint_artist.get_offsets(), midpoints)
    assert vertex_artist.get_sizes()[0] == 12
    assert midpoint_artist.get_sizes()[0] == 18
    np.testing.assert_allclose(
        vertex_artist.get_facecolors()[0], matplotlib.colors.to_rgba("tab:red")
    )
    np.testing.assert_allclose(
        midpoint_artist.get_facecolors()[0], matplotlib.colors.to_rgba("tab:blue")
    )
    assert all(line.get_color() == axes_textcolor(ax) for line in lines)
    assert all(line.get_linewidth() == 0.5 for line in lines)
    assert all(line.get_linestyle() == "--" for line in lines)

    plt.close(fig)


def test_bz_slice_helpers_export_from_plotting_namespace():
    assert eplt.plot_in_plane_bz is plot_in_plane_bz
    assert eplt.plot_out_of_plane_bz is plot_out_of_plane_bz


def test_bz_slice_helpers_accept_sequence_bvec():
    bvec = erlab.lattice.to_reciprocal(np.eye(3)).tolist()
    bounds = (-4.0, 4.0, -4.0, 4.0)

    fig, ax = plt.subplots()
    in_plane_lines, _, _ = plot_in_plane_bz(bvec, bounds=bounds, ax=ax)
    out_of_plane_lines, _, _ = plot_out_of_plane_bz(bvec, bounds=bounds, ax=ax)

    assert in_plane_lines
    assert out_of_plane_lines
    plt.close(fig)
