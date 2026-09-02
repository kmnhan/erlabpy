"""Extensions to mplot3d."""

import matplotlib.collections
import matplotlib.patches
import mpl_toolkits.mplot3d
import mpl_toolkits.mplot3d.art3d
import mpl_toolkits.mplot3d.proj3d
import numpy as np


class FancyArrow3D(matplotlib.patches.FancyArrow):
    """A Matplotlib arrow defined by a point and a vector in three dimensions.

    Parameters
    ----------
    x, y, z
        Coordinates of the arrow tail in data coordinates.
    dx, dy, dz
        Components of the arrow vector in data-coordinate units.
    **kwargs
        Additional keyword arguments passed to
        :class:`matplotlib.patches.FancyArrow`.

    Notes
    -----
    Add the arrow to a three-dimensional Matplotlib axes. Its two-dimensional path is
    updated from the current projection when Matplotlib draws it.
    """

    def __init__(self, x, y, z, dx, dy, dz, **kwargs) -> None:
        super().__init__(0, 0, 0, 0, **kwargs)
        posA, posB = (x, y, z), (x + dx, y + dy, z + dz)
        self._verts3d = tuple((posA[i], posB[i]) for i in range(3))

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mpl_toolkits.mplot3d.proj3d.proj_transform(
            xs3d, ys3d, zs3d, self.axes.M
        )
        self.set_data(x=xs[0], y=ys[0], dx=xs[1] - xs[0], dy=ys[1] - ys[0])
        return np.min(zs)


class FancyArrowPatch3D(matplotlib.patches.FancyArrowPatch):
    """A Matplotlib arrow patch between two points in three dimensions.

    Parameters
    ----------
    posA, posB
        Three-coordinate sequences for the arrow endpoints in data coordinates.
    *args, **kwargs
        Additional arguments passed to :class:`matplotlib.patches.FancyArrowPatch`.

    Notes
    -----
    Add the patch to a three-dimensional Matplotlib axes. Its two-dimensional path is
    updated from the current projection when Matplotlib draws it.
    """

    def __init__(self, posA, posB, *args, **kwargs) -> None:
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = tuple((posA[i], posB[i]) for i in range(3))

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mpl_toolkits.mplot3d.proj3d.proj_transform(
            xs3d, ys3d, zs3d, self.axes.M
        )
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


# arbitrary patch placing
# improved from https://stackoverflow.com/q/18228966


def _transform_zdir(zdir):
    zdir = mpl_toolkits.mplot3d.art3d.get_dir_vector(zdir)
    zn = zdir / np.linalg.norm(zdir)

    cos_angle = zn[2]
    sin_angle = np.linalg.norm(zn[:2])
    if sin_angle == 0:
        return np.sign(cos_angle) * np.eye(3)

    d = np.array((zn[1], -zn[0], 0))
    d /= sin_angle
    ddt = np.outer(d, d)
    skew = np.array([[0, 0, -d[1]], [0, 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64)
    return ddt + cos_angle * (np.eye(3) - ddt) + sin_angle * skew


def set_3d_properties(self, verts, zs=0, zdir="z") -> None:
    """Store two-dimensional vertices in a three-dimensional direction.

    Parameters
    ----------
    self
        Patch-like object on which to store the transformed vertices.
    verts
        Sequence of two-dimensional vertices.
    zs
        Scalar or one value per vertex along `zdir`.
    zdir
        Direction normal to the original vertices, as accepted by Matplotlib's 3D
        conversion utilities.

    Notes
    -----
    This low-level helper modifies ``self._segment3d`` in place.
    """
    zs = np.broadcast_to(zs, len(verts))
    self._segment3d = np.asarray(
        [
            (*np.dot(_transform_zdir(zdir), (x, y, 0)), 0, 0, z)
            for ((x, y), z) in zip(verts, zs, strict=True)
        ]
    )


def pathpatch_translate(pathpatch, delta) -> None:
    """Translate a converted 3D path patch in place.

    Parameters
    ----------
    pathpatch
        Three-dimensional path patch produced by Matplotlib's path-patch conversion.
    delta
        Three-component translation in data-coordinate units.
    """
    pathpatch._segment3d += np.asarray(delta)


def to_3d(pathpatch, z=0.0, zdir="z", delta=(0, 0, 0)):
    """Convert a two-dimensional path patch to a translated 3D patch.

    Parameters
    ----------
    pathpatch
        Matplotlib path patch attached to a three-dimensional axes. The patch is
        converted in place.
    z
        Position along `zdir`, in data-coordinate units.
    zdir
        Direction normal to the original two-dimensional patch. It accepts the values
        supported by :func:`mpl_toolkits.mplot3d.art3d.pathpatch_2d_to_3d`.
    delta
        Three-component translation applied after conversion, in data-coordinate
        units.

    Returns
    -------
    mpl_toolkits.mplot3d.art3d.Patch3D
        The same patch object after in-place conversion and translation.

    Raises
    ------
    ValueError
        If `pathpatch` is not attached to a three-dimensional axes.
    """
    if not hasattr(pathpatch.axes, "get_zlim"):
        raise ValueError("Axes projection must be 3D")
    mpl_toolkits.mplot3d.art3d.pathpatch_2d_to_3d(pathpatch, z=z, zdir=zdir)
    pathpatch.translate(delta)
    return pathpatch


# mpl_toolkits.mplot3d.art3d.Patch3D.set_3d_properties = set_3d_properties
mpl_toolkits.mplot3d.art3d.Patch3D.translate = pathpatch_translate
# matplotlib.patches.Patch.to_3d = to_3d
