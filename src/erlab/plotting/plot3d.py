"""Extensions to mplot3d."""

import contextlib

import matplotlib.collections
import matplotlib.patches
import mpl_toolkits.mplot3d
import mpl_toolkits.mplot3d.art3d
import mpl_toolkits.mplot3d.proj3d
import numpy as np


class FancyArrow3D(matplotlib.patches.FancyArrow):
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


def _transform_zdir(zn):
    cos_angle = zn[2]
    sin_angle = np.linalg.norm(zn[:2])
    if sin_angle == 0:
        return np.sign(cos_angle) * np.eye(3)

    d = np.array((zn[1], -zn[0], 0))
    d /= sin_angle
    ddt = np.outer(d, d)
    skew = np.array([[0, 0, -d[1]], [0, 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64)
    return ddt + cos_angle * (np.eye(3) - ddt) + sin_angle * skew


def _set_arbitrary_3d_properties(
    self, verts, zs=0, zdir="z", axlim_clip: bool = False
) -> None:
    """
    Set the *z* position and direction of the patch.

    Parameters
    ----------
    verts :
        The vertices of the patch.
    zs : float
        The location along the *zdir* axis in 3D space to position the
        patch.
    zdir : {'x', 'y', 'z'} or 3-element array-like, optional
        Plane to plot patch orthogonal to. Default: 'z'.
    axlim_clip : bool, default: False
        Whether to hide patches with a vertex outside the axes view limits.
    """
    zs = np.broadcast_to(zs, len(verts))
    zn = zdir = mpl_toolkits.mplot3d.art3d.get_dir_vector(zdir)
    zn = zdir / np.linalg.norm(zdir)
    self._segment3d = np.asarray(
        [
            np.dot(_transform_zdir(zn), (x, y, 0.0)) + zn * z
            for ((x, y), z) in zip(verts, zs, strict=True)
        ]
    )
    self._axlim_clip = axlim_clip


def _translate_patch(pathpatch, delta) -> None:
    pathpatch._segment3d += np.asarray(delta)


def to_3d(
    patch: matplotlib.patches.Patch, z=0.0, zdir="z", delta=(0, 0, 0)
) -> matplotlib.patches.Patch:
    if not hasattr(patch.axes, "get_zlim"):
        raise ValueError("Axes projection must be 3D")
    mpl_toolkits.mplot3d.art3d.pathpatch_2d_to_3d(patch, z=z, zdir=zdir)
    _translate_patch(patch, delta)
    return patch


@contextlib.contextmanager
def patch3d_context():
    _set_3d_properties = mpl_toolkits.mplot3d.art3d.Patch3D.set_3d_properties
    mpl_toolkits.mplot3d.art3d.Patch3D.set_3d_properties = _set_arbitrary_3d_properties
    matplotlib.patches.Patch.to_3d = to_3d

    try:
        yield
    finally:
        mpl_toolkits.mplot3d.art3d.Patch3D.set_3d_properties = _set_3d_properties
        del matplotlib.patches.Patch.to_3d
