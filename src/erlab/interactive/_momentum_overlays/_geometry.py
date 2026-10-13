"""Ellipse symmetry and RGBA sampling in a momentum plane."""

from __future__ import annotations

import itertools
import typing

import numpy as np

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def ellipse_points(center, radii, rotation: float = 0.0, npoints: int = 181):
    """Return a closed ellipse as an N by 2 array; radii are semiaxes."""
    center, radii = np.asarray(center, dtype=float), np.asarray(radii, dtype=float)
    if (
        center.shape != (2,)
        or radii.shape != (2,)
        or not np.all(np.isfinite([center, radii]))
    ):
        raise ValueError(
            "Ellipse center and radii must each contain two finite values."
        )
    if not np.isfinite(rotation):
        raise ValueError("Ellipse rotation must be finite.")
    if np.any(radii <= 0.0) or npoints < 4:
        raise ValueError(
            "Ellipse radii must be positive and npoints must be at least 4."
        )
    angle = np.linspace(0.0, 2.0 * np.pi, npoints)
    return (np.column_stack((np.cos(angle), np.sin(angle))) * radii) @ _rotation(
        rotation
    ).T + center


def _rotation(angle: float):
    cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return np.array([[cosine, -sine], [sine, cosine]])


def symmetry_matrices(group: str, rotation: float = 0.0):
    """Return planar point-group matrices; rotation sets the first mirror axis."""
    orders = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "6": 6,
        "m": 1,
        "2mm": 2,
        "3m": 3,
        "4mm": 4,
        "6mm": 6,
    }
    if group not in orders:
        raise ValueError(f"Unsupported planar point group: {group!r}")
    matrices = [
        _rotation(360.0 * index / orders[group]) for index in range(orders[group])
    ]
    if "m" in group:
        mirror = _rotation(rotation) @ np.diag([1.0, -1.0]) @ _rotation(-rotation)
        matrices += [matrix @ mirror for matrix in matrices]
    return matrices


def expand_pocket(
    center,
    radii,
    rotation: float,
    group: str = "1",
    symmetry_angle: float = 0.0,
    bvec=None,
    repeats: tuple[int, ...] = (0, 0),
    npoints: int = 181,
    *,
    translations: npt.ArrayLike | None = None,
):
    """Expand an ellipse by point symmetry, then translations along row-vector bvec.

    Each repeat count includes cells from its negative through its positive value.
    Explicit translations replace the basis and repeat counts when supplied.
    They must be a finite N by 2 array; an empty array produces no contours.
    Coincident ellipses are returned only once, including circles and special positions.
    """
    points = ellipse_points(center, radii, rotation, npoints)
    if translations is None:
        if len(repeats) != 2 or any(
            int(count) != count or count < 0 for count in repeats
        ):
            raise ValueError("Repeat counts must be two nonnegative integers.")
        basis = np.eye(2) if bvec is None else np.asarray(bvec, dtype=float)
        if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
            raise ValueError("bvec must be a finite 2 by 2 array of row vectors.")
        shifts = np.array(
            [
                np.array(indices) @ basis
                for indices in itertools.product(
                    *(range(-int(count), int(count) + 1) for count in repeats)
                )
            ]
        )
    else:
        shifts = np.asarray(translations, dtype=float)
        if shifts.ndim != 2 or shifts.shape[1] != 2 or not np.isfinite(shifts).all():
            raise ValueError("translations must be a finite N by 2 array.")
    matrices = symmetry_matrices(group, symmetry_angle)
    if len(shifts) == 0:
        return []
    covariance = (
        _rotation(rotation)
        @ np.diag(np.asarray(radii, dtype=float) ** 2)
        @ _rotation(-rotation)
    )
    contours = []
    seen: dict[
        tuple[int, int],
        list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    ] = {}
    # Compare only nearby centers. The bucket width bounds the allclose tolerance
    # across the expanded lattice; adjacent buckets retain boundary coincidences.
    center = np.asarray(center, dtype=float)
    tolerance = 1e-10 * (1.0 + np.linalg.norm(center) + np.max(np.abs(shifts)))
    neighbors = tuple(itertools.product((-1, 0, 1), repeat=2))
    for matrix in matrices:
        shape = matrix @ covariance @ matrix.T
        transformed_center = matrix @ center
        transformed_points = points @ matrix.T
        for shift in shifts:
            position = transformed_center + shift
            bucket = tuple(int(value) for value in np.floor(position / tolerance))
            candidates = (
                entry
                for dx, dy in neighbors
                for entry in seen.get((bucket[0] + dx, bucket[1] + dy), ())
            )
            if any(
                np.allclose(position, old_center, rtol=1e-10, atol=1e-10)
                and np.allclose(shape, old_shape, rtol=1e-10, atol=1e-10)
                for old_center, old_shape in candidates
            ):
                continue
            seen.setdefault((bucket[0], bucket[1]), []).append((position, shape))
            contours.append(transformed_points + shift)
    return contours


def warp_image(
    image_rgba: npt.ArrayLike, kx, ky, center, width: float, *, rotation: float = 0.0
):
    """Sample RGBA at momenta with premultiplied-alpha bilinear interpolation.

    Return uint8 RGBA. Width spans the pixel edges; height follows the pixel aspect.
    Columns increase kx and rows decrease ky. Invalid points are transparent.
    Points outside the image are transparent.
    Floating input channels must be between zero and one. No Jacobian is applied.
    Rotation is counterclockwise in degrees about the fixed momentum-space center.
    """
    image = np.asarray(image_rgba)
    if image.ndim != 3 or image.shape[2] != 4 or min(image.shape[:2]) < 1:
        raise ValueError("The source image must have shape (height, width, 4).")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("Image width must be positive and finite.")
    if not np.isfinite(rotation):
        raise ValueError("Image rotation must be finite.")
    kx, ky = np.broadcast_arrays(
        np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)
    )
    height_px, width_px = image.shape[:2]
    dx, dy = kx - center[0], ky - center[1]
    cosine, sine = np.cos(np.deg2rad(rotation)), np.sin(np.deg2rad(rotation))
    # Apply the inverse placement rotation before sampling source pixel coordinates.
    with np.errstate(invalid="ignore"):
        local_x, local_y = cosine * dx + sine * dy, -sine * dx + cosine * dy
    col = local_x * width_px / width + (width_px - 1) / 2
    row = -local_y * width_px / width + (height_px - 1) / 2
    valid = np.isfinite(col) & np.isfinite(row)
    valid &= (
        (col >= -0.5)
        & (col <= width_px - 0.5)
        & (row >= -0.5)
        & (row <= height_px - 0.5)
    )
    col = np.clip(np.where(valid, col, 0.0), 0, width_px - 1)
    row = np.clip(np.where(valid, row, 0.0), 0, height_px - 1)
    row0, col0 = row.astype(np.intp), col.astype(np.intp)
    row1, col1 = np.minimum(row0 + 1, height_px - 1), np.minimum(col0 + 1, width_px - 1)
    row_weight, col_weight = row - row0, col - col0
    result = np.zeros((*row.shape, 4), dtype=float)
    channel_scale = 255.0 if np.issubdtype(image.dtype, np.integer) else 1.0
    # Convert only sampled pixels. Temporary memory then scales with the output
    # grid, even for large source images or noncontiguous views.
    for row_index, row_fraction in ((row0, 1 - row_weight), (row1, row_weight)):
        for col_index, col_fraction in ((col0, 1 - col_weight), (col1, col_weight)):
            pixels = image[row_index, col_index].astype(float)
            pixels /= channel_scale
            np.clip(pixels, 0.0, 1.0, out=pixels)
            pixels[..., :3] *= pixels[..., 3, None]
            pixels *= (row_fraction * col_fraction)[..., None]
            result += pixels
    result[..., :3] = np.divide(
        result[..., :3],
        result[..., 3, None],
        out=np.zeros_like(result[..., :3]),
        where=result[..., 3, None] > 0.0,
    )
    result = np.where(valid[..., None], result, 0.0)
    return np.round(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
