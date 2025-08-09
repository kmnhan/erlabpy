import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pytest

from erlab.plotting.bz import get_bz_edge, plot_bz, plot_hex_bz


def test_get_bz_edge_square_lattice():
    # 2D square lattice, real space basis
    a = 1.0
    basis = np.array([[a, 0], [0, a]])
    lines, vertices = get_bz_edge(basis, reciprocal=False)
    # Should return lines and vertices for a square BZ
    assert lines.shape[1:] == (2, 2)
    assert vertices.shape[1] == 2
    # Vertices should be at +/- pi/a
    expected = np.array(
        [
            [np.pi / a, np.pi / a],
            [-np.pi / a, np.pi / a],
            [-np.pi / a, -np.pi / a],
            [np.pi / a, -np.pi / a],
        ]
    )
    # Each expected vertex should be present (within tolerance)
    for v in expected:
        assert np.any(np.all(np.isclose(vertices, v, atol=1e-8), axis=1))


def test_get_bz_edge_hexagonal_lattice():
    # 2D hexagonal lattice, real space basis
    a = 1.0
    basis = np.array([[a, 0], [a / 2, a * np.sqrt(3) / 2]])
    lines, vertices = get_bz_edge(basis, reciprocal=False)
    # Should return lines and vertices for a hexagonal BZ
    assert lines.shape[1:] == (2, 2)
    assert vertices.shape[1] == 2
    # There should be 6 unique vertices for the first BZ
    # (allowing for floating point tolerance)
    unique = []
    for v in vertices:
        if not any(np.allclose(v, u, atol=1e-8) for u in unique):
            unique.append(v)
    assert len(unique) == 6


def test_get_bz_edge_reciprocal_input():
    # Provide reciprocal basis, should not invert
    a = 2.0
    basis = np.array([[2 * np.pi / a, 0], [0, 2 * np.pi / a]])
    _, vertices = get_bz_edge(basis, reciprocal=True)
    # Vertices should be at +/- pi/a
    expected = np.array(
        [
            [np.pi / a, np.pi / a],
            [-np.pi / a, np.pi / a],
            [-np.pi / a, -np.pi / a],
            [np.pi / a, -np.pi / a],
        ]
    )
    for v in expected:
        assert np.any(np.all(np.isclose(vertices, v, atol=1e-8), axis=1))


def test_get_bz_edge_extend():
    # Test extension of BZ
    a = 1.0
    basis = np.eye(2) * a
    lines, vertices = get_bz_edge(basis, reciprocal=False, extend=(2, 2))
    # There should be more vertices than for the first BZ
    lines1, vertices1 = get_bz_edge(basis, reciprocal=False)
    assert vertices.shape[0] > vertices1.shape[0]
    assert lines.shape[0] > lines1.shape[0]


def test_get_bz_edge_invalid_shape():
    # Should raise ValueError for invalid basis shape
    with pytest.raises(
        ValueError, match=r"Shape of `basis` must be \(N, N\) where N = 2 or 3."
    ):
        get_bz_edge(np.eye(4))


def test_get_bz_edge_3d():
    # 3D cubic lattice
    a = 1.0
    basis = np.eye(3) * a
    lines, vertices = get_bz_edge(basis, reciprocal=False)
    assert lines.shape[1:] == (2, 3)
    assert vertices.shape[1] == 3
    # There should be 8 unique vertices for a cube
    unique = []
    for v in vertices:
        if not any(np.allclose(v, u, atol=1e-8) for u in unique):
            unique.append(v)
    assert len(unique) == 8


def test_plot_hex_bz_offset_and_rotate():
    fig, ax = plt.subplots()
    offset = (1.0, 2.0)
    rotate = 30.0
    patch = plot_hex_bz(a=2.0, rotate=rotate, offset=offset, ax=ax)
    # Check offset and orientation
    np.testing.assert_allclose(patch.xy, offset)
    assert np.isclose(patch.orientation, np.deg2rad(rotate))
    plt.close(fig)


def test_plot_hex_bz_reciprocal_and_real_space():
    fig, ax = plt.subplots()
    # Reciprocal
    patch1 = plot_hex_bz(a=2.0, reciprocal=True, ax=ax)
    # Real space
    patch2 = plot_hex_bz(a=2.0, reciprocal=False, ax=ax)
    # Radii should be different
    assert not np.isclose(patch1.radius, patch2.radius)
    plt.close(fig)


def test_plot_hex_bz_kwargs_and_abbrv():
    fig, ax = plt.subplots()
    # Test that linestyle and linewidth are passed via abbreviations
    patch = plot_hex_bz(a=2.0, ax=ax, ls="-.", lw=2.5, ec="red")
    assert patch.get_linestyle() == "-."
    assert patch.get_linewidth() == 2.5
    assert patch.get_edgecolor() == (1.0, 0.0, 0.0, 1.0)  # red in RGBA
    plt.close(fig)


def test_plot_hex_bz_iterable_axes():
    fig, axs = plt.subplots(1, 2)
    patches = plot_hex_bz(a=2.0, ax=axs)
    assert isinstance(patches, list)
    assert all(hasattr(p, "get_path") for p in patches)
    assert all(p in ax.patches for p, ax in zip(patches, axs, strict=True))
    plt.close(fig)


def test_plot_hex_bz_clip_path():
    fig, ax = plt.subplots()
    clip = matplotlib.patches.Circle((0, 0), radius=1)
    patch = plot_hex_bz(a=2.0, ax=ax, clip_path=clip)
    assert patch.get_clip_path() is not None
    plt.close(fig)


def test_plot_bz_square_lattice():
    fig, ax = plt.subplots()
    a = 1.0
    basis = np.array([[a, 0], [0, a]])
    patch = plot_bz(basis, ax=ax)
    # Should be a Polygon with 4 vertices (square)
    assert isinstance(patch, matplotlib.patches.Polygon)
    assert patch.get_xy().shape[0] == 5  # 4 vertices + closing point
    # Vertices should be at +/- pi/a
    expected = np.array(
        [
            [np.pi / a, np.pi / a],
            [-np.pi / a, np.pi / a],
            [-np.pi / a, -np.pi / a],
            [np.pi / a, -np.pi / a],
        ]
    )
    for v in expected:
        assert np.any(np.all(np.isclose(patch.get_xy()[:-1], v, atol=1e-8), axis=1))
    plt.close(fig)


def test_plot_bz_hexagonal_lattice():
    fig, ax = plt.subplots()
    a = 1.0
    basis = np.array([[a, 0], [a / 2, a * np.sqrt(3) / 2]])
    patch = plot_bz(basis, ax=ax)
    # Should be a Polygon with 6 vertices (hexagon)
    assert isinstance(patch, matplotlib.patches.Polygon)
    assert patch.get_xy().shape[0] == 7  # 6 vertices + closing point
    plt.close(fig)


def test_plot_bz_rotate_and_offset():
    fig, ax = plt.subplots()
    a = 1.0
    basis = np.eye(2) * a
    rotate = 45.0
    offset = (2.0, -1.0)
    patch = plot_bz(basis, rotate=rotate, offset=offset, ax=ax)
    # All vertices should be offset by (2, -1)
    verts = patch.get_xy()[:-1]
    center = np.mean(verts, axis=0)
    np.testing.assert_allclose(center, offset, atol=1e-1)
    plt.close(fig)


def test_plot_bz_reciprocal_basis():
    fig, ax = plt.subplots()
    a = 2.0
    basis = np.array([[2 * np.pi / a, 0], [0, 2 * np.pi / a]])
    patch = plot_bz(basis, reciprocal=True, ax=ax)
    # Vertices should be at +/- pi/a
    expected = np.array(
        [
            [np.pi / a, np.pi / a],
            [-np.pi / a, np.pi / a],
            [-np.pi / a, -np.pi / a],
            [np.pi / a, -np.pi / a],
        ]
    )
    for v in expected:
        assert np.any(np.all(np.isclose(patch.get_xy()[:-1], v, atol=1e-8), axis=1))
    plt.close(fig)


def test_plot_bz_kwargs_and_abbrv():
    fig, ax = plt.subplots()
    a = 1.0
    basis = np.eye(2) * a
    patch = plot_bz(basis, ax=ax, ls=":", lw=3.0, ec="blue")
    assert patch.get_linestyle() == ":"
    assert patch.get_linewidth() == 3.0
    assert patch.get_edgecolor() == (0.0, 0.0, 1.0, 1.0)  # blue in RGBA
    plt.close(fig)


def test_plot_bz_fill_and_closed():
    fig, ax = plt.subplots()
    a = 1.0
    basis = np.eye(2) * a
    patch = plot_bz(basis, ax=ax, fill=True, closed=True)
    assert patch.get_fill()
    assert patch.get_closed()
    plt.close(fig)


def test_plot_bz_iterable_axes():
    fig, axs = plt.subplots(1, 2)
    a = 1.0
    basis = np.eye(2) * a
    patches = [plot_bz(basis, ax=ax) for ax in axs]
    assert isinstance(patches, list)
    assert all(isinstance(p, matplotlib.patches.Polygon) for p in patches)
    assert all(p in ax.patches for p, ax in zip(patches, axs, strict=True))
    plt.close(fig)
