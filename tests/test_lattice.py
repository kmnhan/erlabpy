import numpy as np
import pytest

from erlab.lattice import (
    get_bz_edge,
    get_bz_slice,
    get_in_plane_bz,
    get_out_of_plane_bz,
    to_primitive,
    to_reciprocal,
)


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


@pytest.mark.parametrize(
    ("centering", "expected_abs_det"),
    [
        ("P", 1.0),
        ("A", 0.5),
        ("B", 0.5),
        ("C", 0.5),
        ("I", 0.5),
        ("F", 0.25),
        ("R", 1.0 / 3.0),
    ],
)
def test_to_primitive_volume_ratio(centering, expected_abs_det):
    avec = np.eye(3)
    avec_prim = to_primitive(avec, centering)
    det_ratio = np.linalg.det(avec_prim) / np.linalg.det(avec)
    assert np.isclose(abs(det_ratio), expected_abs_det)


def test_get_bz_slice_cubic_plane():
    bvec = to_reciprocal(np.eye(3))
    bounds = (-3.5, 3.5, -3.5, 3.5)
    segments, vertices, midpoints = get_bz_slice(
        bvec,
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
        plane_bounds=bounds,
        return_midpoints=True,
    )
    assert segments.shape[1:] == (2, 2)
    assert vertices.shape[1] == 2
    assert midpoints.shape[1] == 2
    assert np.all(segments[..., 0] >= bounds[0] - 1e-6)
    assert np.all(segments[..., 0] <= bounds[1] + 1e-6)
    assert np.all(segments[..., 1] >= bounds[2] - 1e-6)
    assert np.all(segments[..., 1] <= bounds[3] + 1e-6)
    expected = np.array(
        [
            [np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, -np.pi],
            [np.pi, -np.pi],
        ]
    )
    for v in expected:
        assert np.any(np.all(np.isclose(vertices, v, atol=1e-6), axis=1))


def test_get_bz_slice_return_3d():
    bvec = to_reciprocal(np.eye(3))
    plane_point = np.array([0.0, 0.0, 1.0])
    segments, vertices = get_bz_slice(
        bvec,
        plane_point=plane_point,
        plane_normal=np.array([0.0, 0.0, 1.0]),
        plane_bounds=(-3.5, 3.5, -3.5, 3.5),
        return_3d=True,
    )
    assert segments.shape[1:] == (2, 3)
    assert vertices.shape[1] == 3
    assert np.allclose(segments[..., 2], plane_point[2])
    assert np.allclose(vertices[:, 2], plane_point[2])


def test_get_in_plane_bz_rotation():
    bvec = to_reciprocal(np.eye(3))
    segments, vertices, midpoints = get_in_plane_bz(
        bvec,
        kz=0.0,
        angle=45.0,
        bounds=(-3.5, 3.5, -3.5, 3.5),
        return_midpoints=True,
    )
    assert segments.shape[1:] == (2, 2)
    assert vertices.shape[1] == 2
    assert midpoints.shape[1] == 2
    assert segments.shape[0] > 0
    bounds = (-3.5, 3.5, -3.5, 3.5)
    assert np.all(segments[..., 0] >= bounds[0] - 1e-6)
    assert np.all(segments[..., 0] <= bounds[1] + 1e-6)
    assert np.all(segments[..., 1] >= bounds[2] - 1e-6)
    assert np.all(segments[..., 1] <= bounds[3] + 1e-6)
    if midpoints.size:
        assert np.all(midpoints[:, 0] >= bounds[0] - 1e-6)
        assert np.all(midpoints[:, 0] <= bounds[1] + 1e-6)
        assert np.all(midpoints[:, 1] >= bounds[2] - 1e-6)
        assert np.all(midpoints[:, 1] <= bounds[3] + 1e-6)


def test_get_out_of_plane_bz_basic():
    bvec = to_reciprocal(np.eye(3))
    segments, vertices, midpoints = get_out_of_plane_bz(
        bvec,
        k_parallel=0.0,
        angle=0.0,
        bounds=(-3.5, 3.5, -3.5, 3.5),
        return_midpoints=True,
    )
    assert segments.shape[1:] == (2, 2)
    assert vertices.shape[1] == 2
    assert midpoints.shape[1] == 2
    unique = np.unique(np.round(vertices, 6), axis=0)
    assert unique.shape[0] == 4
    radii = np.sqrt(np.sum(unique**2, axis=1))
    assert np.allclose(radii, np.sqrt(2) * np.pi, atol=1e-6)
