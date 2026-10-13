import numpy as np
import pytest

from erlab.lattice import (
    _dedup_segments_2d,
    _plane_frame,
    _snap_polyline_endpoints,
    _surface_bz_active_pair_bounds,
    _surface_bz_cells,
    _surface_bz_translation_bounds,
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


@pytest.mark.parametrize("reciprocal", [False, True])
@pytest.mark.parametrize(
    "basis",
    [
        np.eye(2) * 3.5,
        np.diag([1.0, 4.0]),
        np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]]),
        np.array([[1.2, 0.3], [-0.2, 1.7]]),
    ],
)
@pytest.mark.parametrize(
    "transform",
    [
        np.array([[1, 0], [2, 1]]),
        np.array([[13, 21], [8, 13]]),
        np.array([[0, -1], [1, 7]]),
        np.array([[1, 0], [2, -1]]),
    ],
)
def test_get_bz_edge_equivalent_2d_bases(basis, transform, reciprocal):
    transformed = transform @ basis
    original = transformed.copy()
    expected_lines, expected_vertices = get_bz_edge(basis, reciprocal=reciprocal)
    lines, vertices = get_bz_edge(transformed, reciprocal=reciprocal)
    np.testing.assert_array_equal(transformed, original)
    assert lines.shape == expected_lines.shape
    assert vertices.shape == expected_vertices.shape
    distances = np.linalg.norm(vertices[:, None] - expected_vertices[None, :], axis=-1)
    np.testing.assert_allclose(distances.min(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(distances.min(axis=1), 0, atol=1e-10)
    for line in lines:
        assert any(
            np.allclose(line, expected) or np.allclose(line, expected[::-1])
            for expected in expected_lines
        )


def test_get_bz_edge_nearly_coincident_corners():
    basis = np.array([[13, 21], [8, 13]]) @ (np.eye(2) * 3.5)
    lines, vertices = get_bz_edge(to_reciprocal(basis))
    assert lines.shape == (4, 2, 2)
    assert vertices.shape == (4, 2)
    np.testing.assert_allclose(np.abs(vertices), np.pi / 3.5)
    np.testing.assert_allclose(
        np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1), 2 * np.pi / 3.5
    )


def test_get_bz_edge_nonreduced_extension():
    basis = np.array([[1.0, 0.0], [2.0, 1.0]])
    base_lines, base_vertices = get_bz_edge(np.eye(2))
    lines, vertices = get_bz_edge(basis, extend=(1, 2))
    # Repetitions must use the original reciprocal vectors, even after reduction.
    np.testing.assert_allclose(vertices.mean(axis=0), [0.0, 0.0], atol=1e-14)
    for shift in (-basis[1], np.zeros(2), basis[1]):
        for vertex in base_vertices + shift:
            assert np.any(np.all(np.isclose(vertices, vertex), axis=1))
        for line in base_lines + shift:
            assert any(
                np.allclose(line, actual) or np.allclose(line, actual[::-1])
                for actual in lines
            )
    assert vertices.shape[0] == base_vertices.shape[0] * 3
    assert lines.shape[0] == base_lines.shape[0] * 3


@pytest.mark.parametrize(
    "basis",
    [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 2.0], [2.0, 4.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
        np.array([[1.0, np.nan], [0.0, 1.0]]),
    ],
)
def test_get_bz_edge_invalid_2d_basis(basis):
    with pytest.raises(ValueError, match="finite and linearly independent"):
        get_bz_edge(basis)


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


@pytest.mark.parametrize("centering", ["P", "F"])
def test_surface_bz_cells_selects_reciprocal_cells(centering):
    basis = to_reciprocal(to_primitive(2.0 * np.pi * np.eye(3), centering))
    surface = np.array(
        [
            [[0.1, 0.1, 0.1], [0.7, 0.1, 0.1]],
            [[1.1, 0.9, 0.9], [2.1, 0.1, 0.1]],
        ]
    )
    expected = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0 if centering == "P" else 0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]],
        ]
    )

    shifts, owner = _surface_bz_cells(basis, surface)

    assert owner.shape == surface.shape[:-1]
    np.testing.assert_allclose(shifts[owner], expected, atol=1e-14)


def test_surface_bz_cells_uses_first_translation_at_ties():
    coords = np.array([-0.5, 0.5, 1.5])
    x, y = np.meshgrid(coords, coords)
    surface = np.stack((x, y, np.full_like(x, 0.5)), axis=-1)

    shifts, owner = _surface_bz_cells(np.eye(3), surface)

    distance_squared = np.sum((surface[..., None, :] - shifts) ** 2, axis=-1)
    np.testing.assert_array_equal(owner, np.argmin(distance_squared, axis=-1))
    np.testing.assert_array_equal(shifts[owner], np.floor(surface))


@pytest.mark.parametrize("pad_cells", [0, 1, 3])
@pytest.mark.parametrize(("length", "angle", "nearest_x"), [(5, 30, 3), (30, 5, 15)])
def test_surface_bz_cells_finds_nearest_cell_for_skew_basis(
    pad_cells, length, angle, nearest_x
):
    angle = np.deg2rad(angle)
    basis = np.array(
        [[1, 0, 0], [length * np.cos(angle), length * np.sin(angle), 0], [0, 0, 1]]
    )
    surface = (0.49 * (basis[0] + basis[1]))[None, None, :]

    shifts, owner = _surface_bz_cells(basis, surface, pad_cells=pad_cells)

    np.testing.assert_allclose(shifts[owner[0, 0]], [nearest_x, 0, 0], atol=1e-14)
    lo, hi = _surface_bz_translation_bounds(basis, surface, pad_cells=pad_cells)
    assert len(shifts) == np.prod(hi - lo + 1)


def test_surface_bz_cells_skew_basis_is_batch_and_translation_invariant():
    basis = np.array([[1, 0, 0], [5 * np.cos(np.pi / 6), 2.5, 0], [0.4, -0.3, 1.2]])
    points = np.array(
        [0.49 * (basis[0] + basis[1]), [4, 0, 0.1], [0.8, 2.1, -0.6], [-1.3, 0.2, 1.1]]
    )
    integers = np.stack(np.meshgrid(*([np.arange(-12, 13)] * 3)), axis=-1)
    candidates = integers.reshape(-1, 3) @ basis
    distances = np.sum((points[:, None, :] - candidates[None, :, :]) ** 2, axis=-1)
    expected = candidates[np.argmin(distances, axis=-1)]
    shifts, owner = _surface_bz_cells(basis, points[None, :, :])
    np.testing.assert_allclose(shifts[owner[0]], expected, atol=1e-13)

    for point, nearest in zip(points, expected, strict=True):
        for translation in (np.zeros(3), np.array([21, -13, 4]) @ basis):
            shifts, owner = _surface_bz_cells(
                basis, (point + translation)[None, None, :]
            )
            np.testing.assert_allclose(
                shifts[owner[0, 0]], nearest + translation, atol=1e-13
            )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_surface_bz_cells_masks_invalid_points(value):
    surface = np.array([[[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, value, 0.1]]])

    shifts, owner = _surface_bz_cells(np.eye(3), surface)

    assert owner[0, 2] == -1
    np.testing.assert_array_equal(shifts[owner[0, :2]], [[0, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_surface_bz_cells_empty_invalid_surface(value):
    shifts, owner = _surface_bz_cells(np.eye(3), np.full((2, 4, 3), value))

    assert shifts.shape == (0, 3)
    np.testing.assert_array_equal(owner, np.full((2, 4), -1))


@pytest.mark.parametrize("surface_kind", ["curved", "bisector", "masked", "skew"])
def test_get_surface_bz_matches_dense_distance_reference(surface_kind):
    import contourpy

    plot = np.linspace(-1.75, 1.75, 29)
    x, y = np.meshgrid(plot, plot)
    if surface_kind == "bisector":
        basis = np.eye(3)
        z = np.full_like(x, 0.5)
    elif surface_kind == "skew":
        basis = np.array([[1, 0, 0], [5 * np.cos(np.pi / 6), 2.5, 0], [0, 0, 1]])
        x = 2.61 + 0.03 * x
        y = 1.225 + 0.03 * y
        z = np.full_like(x, 0.17)
    else:
        basis = np.array([[1.0, 0.1, 0.0], [-0.3, 1.1, 0.2], [0.1, -0.1, 0.9]])
        # Avoid accidental floating-point ties in this curved-surface comparison.
        # The separate bisector case checks a surface made entirely of exact ties.
        x = x + 0.003
        z = 0.17 + 0.2 * x**2 + 0.11 * y
    surface = np.stack((x, y, z), axis=-1)
    if surface_kind == "masked":
        surface[11:18, 11:18] = np.nan
    finite = np.isfinite(surface).all(axis=-1)
    if surface_kind == "skew":
        # Cover the reference independently of the production search bounds.
        integers = np.stack(
            np.meshgrid(*([np.arange(-8, 9)] * 3), indexing="ij"), axis=-1
        )
        shifts = integers.reshape(-1, 3) @ basis
    else:
        shifts = _surface_bz_translation_grid(basis, surface, pad_cells=1)
    # Independent dense reference: retain the former global squared distances.
    distance_squared = (
        np.sum(shifts**2, axis=1)[:, None, None]
        - 2.0 * np.tensordot(shifts, surface, axes=([1], [2]))
        + np.sum(surface**2, axis=-1)[None]
    )
    owners = np.argmin(distance_squared, axis=0)
    owners[~finite] = -1
    reference_lines = []
    reference_pairs = []
    for (i, j), (r0, r1, c0, c1) in _surface_bz_active_pair_bounds(owners).items():
        if i < 0:
            continue
        rows = slice(max(r0 - 1, 0), min(r1 + 2, len(plot)))
        cols = slice(max(c0 - 1, 0), min(c1 + 2, len(plot)))
        local_owners = owners[rows, cols]
        local_mask = (local_owners == i) | (local_owners == j)
        delta = distance_squared[i, rows, cols] - distance_squared[j, rows, cols]
        lines = contourpy.contour_generator(
            x=plot[cols], y=plot[rows], z=np.ma.array(delta, mask=~local_mask)
        ).lines(0.0)
        for line in lines:
            if line.shape[0] >= 2 and np.linalg.norm(np.ptp(line, axis=0)) > 1e-9:
                reference_lines.append(line)
                reference_pairs.append((i, j))
    reference_lines = _snap_polyline_endpoints(
        reference_lines, reference_pairs, np.sqrt(2) * (plot[1] - plot[0]), 1e-9
    )
    actual_lines, _ = get_surface_bz(basis, plot, plot, surface)

    def canonical_segments(lines):
        segments = np.round(_segmentize_polylines(lines), decimals=8)
        segments = np.sort(segments.view([("x", float), ("y", float)]), axis=1)
        return np.unique(segments.view(float).reshape(-1, 4), axis=0)

    np.testing.assert_allclose(
        canonical_segments(actual_lines), canonical_segments(reference_lines), atol=1e-8
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("return_midpoints", [False, True])
def test_get_surface_bz_empty_invalid_surface(value, return_midpoints):
    result = get_surface_bz(
        np.eye(3),
        [-1.0, 1.0],
        [-1.0, 1.0],
        np.full((2, 2, 3), value),
        return_midpoints=return_midpoints,
    )
    assert result[0] == []
    assert result[1].shape == (0, 2)
    if return_midpoints:
        assert result[2].shape == (0, 2)


def test_get_surface_bz_masks_nonfinite_gap():
    plot = np.linspace(-0.8, 0.8, 33)
    x, y = np.meshgrid(plot, plot)
    surface = np.stack((x, y, np.full_like(x, 0.1)), axis=-1)
    surface[10:23, :, :] = np.inf
    lines, _ = get_surface_bz(np.eye(3), plot, plot, surface)
    assert lines
    for line in lines:
        assert np.isfinite(line).all()
        assert np.all(line[:, 1] < plot[10]) or np.all(line[:, 1] > plot[22])
