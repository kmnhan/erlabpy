import numpy as np
import pytest

from erlab.lattice import (
    _dedup_segments_2d,
    _plane_frame,
    _snap_polyline_endpoints,
    _surface_bz_active_pair_bounds,
    _surface_bz_translation_grid,
    get_bz_edge,
    get_bz_slice,
    get_in_plane_bz,
    get_out_of_plane_bz,
    get_surface_bz,
    to_primitive,
    to_reciprocal,
)


def _surface_from_plane(plane_point, plane_normal, xvals, yvals):
    _, u, v = _plane_frame(plane_normal)
    return (
        plane_point[None, None, :]
        + xvals[None, :, None] * u[None, None, :]
        + yvals[:, None, None] * v[None, None, :]
    )


def _segmentize_polylines(lines):
    if not lines:
        return np.empty((0, 2, 2), dtype=float)
    return np.concatenate(
        [
            np.stack([line[:-1], line[1:]], axis=1)
            for line in lines
            if line.shape[0] > 1
        ],
        axis=0,
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


def test_get_surface_bz_matches_planar_out_of_plane_slice():
    bvec = to_reciprocal(np.eye(3))
    k_parallel = 0.5
    angle = 30.0
    bounds = (-3.5, 3.5, -3.5, 3.5)
    theta = np.deg2rad(angle)
    plane_point = np.array(
        [k_parallel * np.cos(theta), k_parallel * np.sin(theta), 0.0], dtype=float
    )
    plane_normal = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    plot_x = np.linspace(bounds[0], bounds[1], 501)
    plot_y = np.linspace(bounds[2], bounds[3], 501)
    surface = _surface_from_plane(plane_point, plane_normal, plot_x, plot_y)

    lines, vertices, midpoints = get_surface_bz(
        bvec, plot_x, plot_y, surface, return_midpoints=True
    )
    legacy_lines, legacy_vertices, legacy_midpoints = get_out_of_plane_bz(
        bvec,
        k_parallel=k_parallel,
        angle=angle,
        bounds=bounds,
        return_midpoints=True,
    )

    assert len(lines) == legacy_lines.shape[0]
    for expected in legacy_vertices:
        assert any(
            np.min(np.linalg.norm(line - expected, axis=1)) < 2e-2 for line in lines
        )
    for expected in legacy_midpoints:
        assert any(
            np.min(np.linalg.norm(line - expected, axis=1)) < 2e-2 for line in lines
        )
    for vertex in vertices:
        assert np.min(np.linalg.norm(legacy_vertices - vertex, axis=1)) < 2e-2
    assert legacy_midpoints.shape[0] > 0
    assert midpoints.shape == (0, 2)

    endpoints = np.asarray([point for line in lines for point in (line[0], line[-1])])
    pair_dist = np.sqrt(
        np.sum((endpoints[:, None, :] - endpoints[None, :, :]) ** 2, axis=-1)
    )
    close_but_distinct = (pair_dist > 1e-8) & (pair_dist < 2e-2)
    assert not np.any(close_but_distinct)


def test_get_surface_bz_omits_midpoints_for_clipped_planar_segments():
    bvec = to_reciprocal(np.eye(3))
    k_parallel = 0.5
    angle = 30.0
    bounds = (-3.5, 3.5, -1.0, 1.0)
    theta = np.deg2rad(angle)
    plane_point = np.array(
        [k_parallel * np.cos(theta), k_parallel * np.sin(theta), 0.0], dtype=float
    )
    plane_normal = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    plot_x = np.linspace(bounds[0], bounds[1], 401)
    plot_y = np.linspace(bounds[2], bounds[3], 201)
    surface = _surface_from_plane(plane_point, plane_normal, plot_x, plot_y)

    lines, vertices, midpoints = get_surface_bz(
        bvec, plot_x, plot_y, surface, return_midpoints=True
    )
    legacy_lines, legacy_vertices, legacy_midpoints = get_out_of_plane_bz(
        bvec,
        k_parallel=k_parallel,
        angle=angle,
        bounds=bounds,
        return_midpoints=True,
    )

    assert len(lines) == legacy_lines.shape[0] == 1
    assert vertices.shape == legacy_vertices.shape
    assert legacy_midpoints.shape == (1, 2)
    assert midpoints.shape == (0, 2)


def test_get_surface_bz_spans_multiple_repeated_zones_without_duplicates():
    bvec = to_reciprocal(np.eye(3))
    bounds = (-7.5, 7.5, -7.5, 7.5)
    plot_x = np.linspace(bounds[0], bounds[1], 601)
    plot_y = np.linspace(bounds[2], bounds[3], 601)
    surface = _surface_from_plane(
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        plot_x,
        plot_y,
    )

    lines, vertices, midpoints = get_surface_bz(
        bvec, plot_x, plot_y, surface, return_midpoints=True
    )
    segments = _segmentize_polylines(lines)

    assert len(lines) > 4
    assert np.array_equal(_dedup_segments_2d(segments, tol=1e-8), segments)
    assert vertices.shape[0] >= 4
    assert midpoints.shape == (0, 2)


def test_get_surface_bz_only_contours_active_owner_pairs(monkeypatch):
    import contourpy

    bvec = to_reciprocal(np.eye(3))
    bounds = (-7.5, 7.5, -7.5, 7.5)
    plot_x = np.linspace(bounds[0], bounds[1], 201)
    plot_y = np.linspace(bounds[2], bounds[3], 201)
    surface = _surface_from_plane(
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        plot_x,
        plot_y,
    )

    shifts = _surface_bz_translation_grid(bvec, surface, pad_cells=1)
    dist_sq = np.sum((surface[None, ...] - shifts[:, None, None, :]) ** 2, axis=-1)
    owner = np.argmin(dist_sq, axis=0)
    expected_pairs = sorted(_surface_bz_active_pair_bounds(owner))

    call_count = 0
    original = contourpy.contour_generator

    def _counting_generator(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(contourpy, "contour_generator", _counting_generator)

    get_surface_bz(bvec, plot_x, plot_y, surface, return_midpoints=True)

    assert call_count == len(expected_pairs)
    assert call_count < len(shifts)


def test_snap_polyline_endpoints_does_not_merge_unrelated_pairs():
    lines = [
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        np.array([[1.02, 0.02], [2.02, 0.02]], dtype=float),
    ]

    snapped = _snap_polyline_endpoints(
        lines,
        [(0, 1), (2, 3)],
        snap_tol=0.05,
        sig_tol=1e-10,
    )

    assert np.allclose(snapped[0], lines[0])
    assert np.allclose(snapped[1], lines[1])


def test_snap_polyline_endpoints_does_not_merge_parallel_same_pair_segments():
    lines = [
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        np.array([[0.05, 0.08], [1.05, 0.08]], dtype=float),
    ]

    snapped = _snap_polyline_endpoints(
        lines,
        [(0, 1), (0, 1)],
        snap_tol=0.2,
        sig_tol=1e-10,
    )

    assert np.allclose(snapped[0], lines[0])
    assert np.allclose(snapped[1], lines[1])


def test_snap_polyline_endpoints_separates_nearby_distinct_vertices():
    lines = [
        np.array([[-1.0, 0.0], [0.00, 0.00]], dtype=float),
        np.array([[0.0, -1.0], [0.01, 0.00]], dtype=float),
        np.array([[1.0, -1.0], [0.00, 0.01]], dtype=float),
        np.array([[0.04, -1.0], [0.04, 0.00]], dtype=float),
        np.array([[1.0, 0.0], [0.05, 0.00]], dtype=float),
        np.array([[1.0, 1.0], [0.04, 0.01]], dtype=float),
    ]

    snapped = _snap_polyline_endpoints(
        lines,
        [(0, 1), (0, 2), (1, 2), (0, 1), (0, 3), (1, 3)],
        snap_tol=0.05,
        sig_tol=1e-10,
    )

    endpoints = np.asarray([line[-1] for line in snapped], dtype=float)
    first = endpoints[:3]
    second = endpoints[3:]

    assert np.allclose(first, first[0])
    assert np.allclose(second, second[0])
    assert not np.allclose(first[0], second[0])


def test_snap_polyline_endpoints_rejects_nonlocal_cycle_cluster():
    lines = [
        np.array([[-1.0, 0.0], [0.00, 0.00]], dtype=float),
        np.array([[0.04, -1.0], [0.04, 0.00]], dtype=float),
        np.array([[1.04, 0.04], [0.04, 0.04]], dtype=float),
        np.array([[0.08, 1.04], [0.08, 0.04]], dtype=float),
    ]

    snapped = _snap_polyline_endpoints(
        lines,
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        snap_tol=0.1,
        sig_tol=1e-10,
    )

    for line, original in zip(snapped, lines, strict=True):
        assert np.allclose(line, original)


def test_snap_polyline_endpoints_does_not_collapse_single_short_segment():
    lines = [
        np.array([[0.0, 0.0], [0.0, 0.2]], dtype=float),
        np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float),
    ]

    snapped = _snap_polyline_endpoints(
        lines,
        [(0, 1), (2, 3)],
        snap_tol=0.5,
        sig_tol=1e-10,
    )

    assert np.allclose(snapped[0], lines[0])
    assert np.allclose(snapped[1], lines[1])


def test_get_surface_bz_sparse_grid_preserves_short_segments():
    bvec = to_reciprocal(np.eye(3))
    k_parallel = 0.5
    angle = 30.0
    bounds = (-3.5, 3.5, -3.5, 3.5)
    theta = np.deg2rad(angle)
    plane_point = np.array(
        [k_parallel * np.cos(theta), k_parallel * np.sin(theta), 0.0], dtype=float
    )
    plane_normal = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    plot_x = np.linspace(bounds[0], bounds[1], 61)
    plot_y = np.linspace(bounds[2], bounds[3], 61)
    surface = _surface_from_plane(plane_point, plane_normal, plot_x, plot_y)

    lines, _, _ = get_surface_bz(bvec, plot_x, plot_y, surface, return_midpoints=True)
    line_lengths = np.array([np.linalg.norm(line[-1] - line[0]) for line in lines])

    assert len(lines) == 7
    assert np.all(line_lengths > 1e-8)


def test_get_surface_bz_covers_interior_extrema_with_default_padding():
    bvec = to_reciprocal(np.eye(3))
    plot = np.linspace(-1.0, 1.0, 101)
    xvals, yvals = np.meshgrid(plot, plot)
    surface = np.zeros((plot.size, plot.size, 3), dtype=float)
    surface[..., 0] = 20.0 * np.exp(-(xvals**2 + yvals**2) / 0.1)
    surface[..., 1] = 0.1 * xvals
    surface[..., 2] = 0.1 * yvals

    lines_default, vertices_default = get_surface_bz(
        bvec, plot, plot, surface, pad_cells=1
    )
    lines_padded, vertices_padded = get_surface_bz(
        bvec, plot, plot, surface, pad_cells=3
    )

    assert len(lines_default) == len(lines_padded)
    assert sum(max(len(line) - 1, 0) for line in lines_default) == sum(
        max(len(line) - 1, 0) for line in lines_padded
    )
    assert vertices_default.shape == vertices_padded.shape
