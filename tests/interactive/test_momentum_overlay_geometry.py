import tracemalloc

import numpy as np
import pytest
import scipy.interpolate

from erlab.interactive._momentum_overlays._geometry import (
    ellipse_points,
    expand_pocket,
    symmetry_matrices,
    warp_image,
)


def test_rotated_offcenter_ellipse():
    points = ellipse_points([2.0, -1.0], [3.0, 1.0], 90.0, npoints=5)
    np.testing.assert_allclose(
        points, [[2, 2], [1, -1], [2, -4], [3, -1], [2, 2]], atol=1e-14
    )


@pytest.mark.parametrize(
    ("group", "count"),
    [
        ("1", 1),
        ("2", 2),
        ("3", 3),
        ("4", 4),
        ("6", 6),
        ("m", 2),
        ("2mm", 4),
        ("3m", 6),
        ("4mm", 8),
        ("6mm", 12),
    ],
)
def test_point_groups(group, count):
    matrices = symmetry_matrices(group, rotation=17.0)
    assert len(matrices) == count
    for matrix in matrices:
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(2), atol=1e-14)
    determinants = np.linalg.det(matrices)
    assert np.count_nonzero(determinants < 0) == (count // 2 if "m" in group else 0)


def test_mirror_orientation_and_lattice_expansion():
    reflection = symmetry_matrices("m", rotation=45.0)[1]
    np.testing.assert_allclose(reflection @ [2, 1], [1, 2], atol=1e-14)
    contours = expand_pocket([0.0, 0.0], [0.2, 0.2], 17.0, group="6mm")
    assert len(contours) == 1
    assert len(expand_pocket([0, 0], [0.3, 0.2], 0, group="4mm")) == 2
    basis = np.array([[2, 0], [1, 3]])
    translated = expand_pocket(
        [0, 0], [0.2, 0.2], 0, group="2", bvec=basis, repeats=(1, 1)
    )
    assert len(translated) == 9
    # Opposite points average to the transformed ellipse center.
    centers = np.array([(curve[0] + curve[90]) / 2 for curve in translated])
    np.testing.assert_allclose(
        centers,
        [np.array([i, j]) @ basis for i in (-1, 0, 1) for j in (-1, 0, 1)],
        atol=1e-14,
    )


def test_ellipse_and_symmetry_validation():
    for center, radii in [([0], [1, 1]), ([0, 0], [1, -1]), ([np.nan, 0], [1, 1])]:
        with pytest.raises(ValueError, match="Ellipse"):
            ellipse_points(center, radii)
    with pytest.raises(ValueError, match="rotation"):
        ellipse_points([0, 0], [1, 1], rotation=np.nan)
    with pytest.raises(ValueError, match="npoints"):
        ellipse_points([0, 0], [1, 1], npoints=3)
    with pytest.raises(ValueError, match="point group"):
        symmetry_matrices("5")
    with pytest.raises(ValueError, match="Repeat"):
        expand_pocket([0, 0], [1, 1], 0, repeats=(-1, 0))
    with pytest.raises(ValueError, match="bvec"):
        expand_pocket([0, 0], [1, 1], 0, bvec=np.eye(3))


def test_image_orientation_aspect_and_transparency():
    image = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]],
            [
                [0, 255, 255, 255],
                [255, 0, 255, 255],
                [80, 80, 80, 255],
                [255, 255, 255, 255],
            ],
        ],
        dtype=np.uint8,
    )
    # A width of four gives one momentum unit per pixel in both directions.
    x, y = np.meshgrid([-1.5, -0.5, 0.5, 1.5], [0.5, -0.5])
    np.testing.assert_array_equal(warp_image(image, x, y, [0, 0], 4.0), image)
    outside = warp_image(image, [0, 2.1, np.nan, 0], [1.1, 0, 0, -1.1], [0, 0], 4)
    np.testing.assert_array_equal(outside, np.zeros((4, 4), dtype=np.uint8))
    np.testing.assert_array_equal(warp_image(image, x + 3, y - 4, [3, -4], 4), image)


def test_image_premultiplied_interpolation():
    image = np.array([[[255, 0, 0, 255], [0, 0, 255, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(warp_image(image, 0, 0, [0, 0], 2), [255, 0, 0, 128])
    np.testing.assert_array_equal(
        warp_image(image / 255.0, 0, 0, [0, 0], 2), [255, 0, 0, 128]
    )
    np.testing.assert_array_equal(warp_image(image, 0.5, 0, [0, 0], 2), [0, 0, 0, 0])
    np.testing.assert_array_equal(
        warp_image(np.array([[[30, 40, 50, 255]]], dtype=np.uint8), 0, 0, [0, 0], 2),
        [30, 40, 50, 255],
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (5, 1), (5, 7)])
@pytest.mark.parametrize("rotation", [0.0, 32.0])
@pytest.mark.parametrize("dtype", [np.uint8, np.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_image_warp_matches_regular_grid_interpolation(shape, rotation, dtype, strided):
    rng = np.random.default_rng(1)
    source_shape = tuple(2 * size for size in shape) if strided else shape
    image = rng.integers(0, 256, size=(*source_shape, 4), dtype=np.uint8)
    if dtype == np.float64:
        image = image.astype(dtype) / 255.0
    if strided:
        image = image[::2, ::2]
    height, width = shape
    image[0, 0, 3] = 0
    image.setflags(write=False)
    # Avoid exact pixel-edge and half-byte ties when rotating the query coordinates.
    rows = rng.uniform(-0.75, height - 0.25, (19, 23))
    cols = rng.uniform(-0.75, width - 0.25, (19, 23))
    valid = (
        (rows >= -0.5) & (rows <= height - 0.5) & (cols >= -0.5) & (cols <= width - 0.5)
    )
    source = image.astype(float) / (255.0 if dtype == np.uint8 else 1.0)
    source[..., :3] *= source[..., 3, None]
    # SciPy supports singleton grid axes. Pixel-edge strips repeat the nearest pixel.
    expected = scipy.interpolate.interpn(
        (np.arange(height), np.arange(width)),
        source,
        np.stack((rows.clip(0, height - 1), cols.clip(0, width - 1)), axis=-1),
        bounds_error=False,
        fill_value=None,
    )
    expected[..., :3] = np.divide(
        expected[..., :3],
        expected[..., 3, None],
        out=np.zeros_like(expected[..., :3]),
        where=expected[..., 3, None] > 0,
    )
    expected[~valid] = 0
    expected = np.round(expected.clip(0, 1) * 255).astype(np.uint8)
    # Use one momentum unit per pixel, then rotate and translate the image placement.
    x, y = cols - (width - 1) / 2, (height - 1) / 2 - rows
    c, s = np.cos(np.deg2rad(rotation)), np.sin(np.deg2rad(rotation))
    actual = warp_image(
        image, 2 + c * x - s * y, -3 + s * x + c * y, [2, -3], width, rotation=rotation
    )
    np.testing.assert_array_equal(actual, expected)


def test_image_warp_large_source_uses_bounded_memory():
    pixel = np.array([30, 90, 160, 200], dtype=np.uint8)
    image = np.broadcast_to(pixel, (2160, 3840, 4))
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 180), np.linspace(-0.05, 0.05, 180))
    tracemalloc.start()
    try:
        result = warp_image(image, x, y, (0, 0), 0.3)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(result, np.broadcast_to(pixel, result.shape))
    # Allow ample space for output-sized temporaries. One full-source float64
    # buffer alone would require 253 MiB, regardless of the preview resolution.
    assert peak_bytes < 32 * 1024**2


def test_image_validation():
    with pytest.raises(ValueError, match="shape"):
        warp_image(np.zeros((2, 2, 3)), 0, 0, [0, 0], 1)
    with pytest.raises(ValueError, match="positive"):
        warp_image(np.zeros((2, 2, 4)), 0, 0, [0, 0], 0)


@pytest.mark.parametrize("radii", [(0.2, 0.2), (0.3, 0.2)])
@pytest.mark.parametrize("center", [(0.5, 0.0), (1.0, 0.0)])
def test_expansion_deduplicates_symmetry_after_translation(radii, center):
    contours = expand_pocket(
        center, radii, 0.0, group="4mm", bvec=np.eye(2), repeats=(1, 1)
    )
    # Form geometric descriptors independently, without sampling-point order.
    expected = []
    for matrix in symmetry_matrices("4mm"):
        shape = matrix @ np.diag(np.square(radii)) @ matrix.T
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                position = matrix @ center + [i, j]
                if not any(
                    np.allclose(position, old_position, atol=1e-10, rtol=1e-10)
                    and np.allclose(shape, old_shape, atol=1e-10, rtol=1e-10)
                    for old_position, old_shape in expected
                ):
                    expected.append((position, shape))
    assert len(contours) == len(expected)
    for contour, (position, shape) in zip(contours, expected, strict=True):
        points = contour[:-1]
        np.testing.assert_allclose(points.mean(axis=0), position, atol=1e-12)
        centered = points - position
        np.testing.assert_allclose(
            2 * centered.T @ centered / len(points), shape, atol=1e-12
        )


def test_expansion_spatial_buckets_preserve_boundary_coincidences():
    # Distinct centers within the tolerance can lie on opposite sides of zero.
    contours = expand_pocket(
        [0.0, 0.0],
        [0.2, 0.2],
        0.0,
        bvec=[[5e-11, 0.0], [0.0, 1.0]],
        repeats=(1, 0),
    )
    assert len(contours) == 1


@pytest.mark.parametrize("rotation", [90.0, 31.0, -48.0])
def test_image_rotation_about_fixed_center(rotation):
    image = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 128], [0, 0, 255, 255]],
            [[255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    # Rotate source pixel centers counterclockwise around a nonzero placement center.
    center = [2.0, -3.0]
    x, y = np.meshgrid([-1.0, 0.0, 1.0], [0.5, -0.5])
    angle = np.deg2rad(rotation)
    kx = center[0] + np.cos(angle) * x - np.sin(angle) * y
    ky = center[1] + np.sin(angle) * x + np.cos(angle) * y
    np.testing.assert_array_equal(
        warp_image(image, kx, ky, center, 3.0, rotation=rotation), image
    )
    np.testing.assert_array_equal(
        warp_image(image, np.inf, 0.0, center, 3.0, rotation=rotation), [0, 0, 0, 0]
    )
    # Sampling outside the rotated height stays transparent.
    outside_x = center[0] - 1.1 * np.sin(angle)
    outside_y = center[1] + 1.1 * np.cos(angle)
    np.testing.assert_array_equal(
        warp_image(image, outside_x, outside_y, center, 3.0, rotation=rotation),
        [0, 0, 0, 0],
    )


def test_image_rotated_premultiplied_interpolation():
    image = np.array([[[255, 0, 0, 255], [0, 0, 255, 0]]], dtype=np.uint8)
    # Positive 90 degrees puts the opaque left pixel below the center.
    np.testing.assert_array_equal(
        warp_image(image, 2.0, -3.5, [2.0, -3.0], 2.0, rotation=90.0),
        [255, 0, 0, 255],
    )
    np.testing.assert_array_equal(
        warp_image(image, 2.0, -3.0, [2.0, -3.0], 2.0, rotation=27.0),
        [255, 0, 0, 128],
    )
    with pytest.raises(ValueError, match="rotation"):
        warp_image(image, 0, 0, [0, 0], 2.0, rotation=np.nan)


def test_explicit_pocket_translations():
    shifts = np.array([[0, 0], [2, 1], [2, 1], [-3, 2]])
    contours = expand_pocket(
        [0, 0],
        [0.3, 0.2],
        0,
        group="4mm",
        translations=shifts,
        bvec=np.eye(3),
        repeats=(-1, 0),
    )
    assert len(contours) == 6
    for index, contour in enumerate(contours):
        center = (contour[0] + contour[90]) / 2
        np.testing.assert_allclose(center, shifts[[0, 1, 3][index % 3]], atol=1e-14)
    # A circle is invariant under all point-group operations.
    assert len(expand_pocket([0, 0], [1, 1], 0, "6mm", translations=shifts)) == 3
    assert expand_pocket([0, 0], [1, 1], 0, translations=np.empty((0, 2))) == []
    base = ellipse_points([0.2, -0.1], [0.3, 0.2], 37)
    translated = expand_pocket([0.2, -0.1], [0.3, 0.2], 37, translations=shifts)
    np.testing.assert_allclose(
        translated, [base + shift for shift in shifts[[0, 1, 3]]]
    )


@pytest.mark.parametrize(
    "shifts", [[], [1, 2], [[0]], [[0, 1, 2]], [[np.nan, 0]], [[0, np.inf]]]
)
def test_explicit_pocket_translation_validation(shifts):
    with pytest.raises(ValueError, match="translations"):
        expand_pocket([0, 0], [1, 1], 0, translations=shifts)
