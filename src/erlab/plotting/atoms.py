"""Plot atoms.

Classes and functions for plotting atoms and bonds in a crystal structure using
matplotlib's 3D plotting capabilities.

Some of the projection code was adapted from kwant.

"""

import contextlib
import functools
import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, cast

import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import mpl_toolkits.mplot3d.art3d
import mpl_toolkits.mplot3d.proj3d
import numpy as np
import numpy.typing as npt
from matplotlib.typing import ColorType

__all__ = ["Atom3DCollection", "Bond3DCollection", "CrystalProperty"]


def sunflower_sphere(n: int = 100) -> npt.NDArray[np.float64]:
    """Return n points on a sphere using the sunflower algorithm."""
    indices = np.arange(0, n, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.c_[
        np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    ].T


unit_sphere: npt.NDArray[np.float64] = sunflower_sphere(100)


def projected_length(
    ax: mpl_toolkits.mplot3d.Axes3D, length: np.float64 | Sequence[np.float64]
):
    if np.iterable(length):
        return np.asarray([projected_length(ax, d) for d in np.asarray(length).flat])

    # rc = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).sum(1) / 2

    rc = np.array(
        mpl_toolkits.mplot3d.proj3d.inv_transform(
            0.5,
            0.5,
            0.5,
            mpl_toolkits.mplot3d.proj3d.world_transformation(
                *ax.get_xlim3d(),
                *ax.get_ylim3d(),
                *ax.get_zlim3d(),
                pb_aspect=ax._roll_to_vertical(ax._box_aspect),
            ),
        )
    )
    rs = unit_sphere * length + rc.reshape(-1, 1)

    transform = mpl_toolkits.mplot3d.proj3d.proj_transform
    proj = ax.get_proj().round(15)
    rp = np.asarray(transform(*rs, proj)[:2])
    rc[:2] = transform(*rc, proj)[:2]

    coords = rp - np.repeat(rc[:2].reshape(-1, 1), len(rs[0]), axis=1)
    return np.linalg.norm(coords, axis=0).mean()


def projected_length_pos(ax: mpl_toolkits.mplot3d.Axes3D, length, position):
    if position.ndim > 1:
        if not np.iterable(length):
            length = np.ones(len(position)) * length
        return np.asarray(
            [
                projected_length_pos(ax, d, p)
                for d, p in zip(
                    np.asarray(length).flat, np.asarray(position), strict=True
                )
            ]
        )
    rc = np.asarray(position).reshape(-1, 1)
    rs = unit_sphere * length + rc

    transform = mpl_toolkits.mplot3d.proj3d.proj_transform
    proj = ax.get_proj()
    rp = np.asarray(transform(*rs, proj)[:2])
    rc[:2] = transform(*rc, proj)[:2]

    coords = rp - rc[:2]
    return np.linalg.norm(coords, axis=0).mean()


def _zalpha(colors, zs):
    """Set alpha based on zdepth.

    Modified from `mpl_toolkits.mplot3d.art3d` to modifies the brightness of the
    color based on depth, rather than setting the transparency.
    """
    if len(colors) == 0 or len(zs) == 0:
        return np.zeros((0, 4))
    norm = plt.Normalize(min(zs), max(zs))
    sats = 1 - norm(zs) * 0.7
    rgba = np.broadcast_to(matplotlib.colors.to_rgba_array(colors), (len(zs), 4))
    rgba = rgba.T * sats
    rgba += 1 - sats
    rgba = rgba.T
    rgba[:, 3] = 1
    return rgba


class Atom3DCollection(mpl_toolkits.mplot3d.art3d.Path3DCollection):
    """A collection of 3D scatter points that represents atoms in a crystal structure.

    This class is not for direct instantiation, but is used to patch the collection
    returned by the `scatter` method. See the implementation of `CrystalProperty.plot`
    for usage.

    """

    def __init__(self, *args, scale_size: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._scale_size = scale_size

    def _maybe_depth_shade_and_sort_colors(self, color_array):
        if self._alpha == 1.0:
            return super()._maybe_depth_shade_and_sort_colors(color_array)
        color_array = (
            _zalpha(color_array, self._vzs)
            if self._vzs is not None and self._depthshade
            else color_array
        )
        if len(color_array) > 1:
            color_array = color_array[self._z_markers_idx]
        return matplotlib.colors.to_rgba_array(color_array, self._alpha)

    def set_sizes(self, sizes: np.ndarray, dpi: float = 72.0):
        super().set_sizes(sizes, dpi)
        self.sizes_orig = np.asarray(sizes) if np.iterable(sizes) else sizes

    @contextlib.contextmanager
    def _use_zordered_offset(self):
        if self._offset_zordered is None:
            yield
        else:
            old_offset = self._offsets
            super().set_offsets(self._offset_zordered)
            old_sizes = self._sizes
            super().set_sizes(
                self._sizes[
                    old_offset[:, 0].argsort()[
                        self._offset_zordered[:, 0].argsort().argsort()
                    ]
                ],
                self.figure.dpi,
            )
            try:
                yield
            finally:
                self._sizes = old_sizes
                self._offsets = old_offset

    def draw(self, renderer):
        if self._scale_size:  # and np.isfinite(self.axes._focal_length):
            # proj_sizes = projected_length(self.axes, np.sqrt(self.sizes_orig))
            proj_sizes = projected_length_pos(
                self.axes, np.sqrt(self.sizes_orig), np.c_[self._offsets3d]
            )
            args = self.axes.transData.frozen().to_values()
            proj_sizes = proj_sizes * ((args[0] + args[3]) / 2) * 72.0 / self.figure.dpi
        else:
            proj_sizes = np.sqrt(self.sizes_orig)

        super().set_sizes(proj_sizes**2, self.figure.dpi)
        with self._use_zordered_offset():
            with matplotlib.cbook._setattr_cm(self, _in_draw=True):
                matplotlib.collections.Collection.draw(self, renderer)

    # def draw(self, renderer):
    #     # Note: unlike in the 2D case, where we can enforce equal
    #     #       aspect ratio, this (currently) does not work with
    #     #       3D plots in matplotlib. As an approximation, we
    #     #       thus scale with the average of the x- and y-axis
    #     #       transformation.
    #     self._factor = (
    #         self._sz_real * ((args[0] + args[3]) / 2) * 72.0 / self.figure.dpi
    #     )
    #     super().draw(renderer)


class Bond3DCollection(mpl_toolkits.mplot3d.art3d.Line3DCollection):
    """A collection of 3D lines that represents bonds in a crystal structure.

    Parameters
    ----------
    segments
        List of segments representing the bonds.
    scale_linewidths
        Boolean indicating whether to scale the linewidths based on the plot's
        perspective.
    **kwargs
        Additional keyword arguments to be passed to
        `mpl_toolkits.mplot3d.art3d.Line3DCollection`.

    """

    def __init__(self, segments, *, scale_linewidths: bool = True, **kwargs):
        super().__init__(segments, **kwargs)
        self._scale_linewidths: bool = scale_linewidths

    def set_linewidth(self, lw):
        super().set_linewidth(lw)
        self.linewidths_orig = np.asarray(lw) if np.iterable(lw) else lw

    def draw(self, renderer):
        if self._scale_linewidths:  # and np.isfinite(self.axes._focal_length):
            proj_len = projected_length(self.axes, self.linewidths_orig)
            args = self.axes.transData.frozen().to_values()
            proj_len = proj_len * ((args[0] + args[3]) / 2) * 72.0 / self.figure.dpi
        else:
            proj_len = self.linewidths_orig

        super().set_linewidth(proj_len)
        super().draw(renderer)

        # if sizes is None:
        #     self._sizes = np.array([])
        #     self._transforms = np.empty((0, 3, 3))
        # else:
        #     self._sizes = np.asarray(sizes)
        #     self._transforms = np.zeros((len(self._sizes), 3, 3))
        #     scale = np.sqrt(self._sizes) * dpi / 72.0 * self._factor
        #     self._transforms[:, 0, 0] = scale
        #     self._transforms[:, 1, 1] = scale
        #     self._transforms[:, 2, 2] = 1.0
        # self.stale = True


class CrystalProperty:
    def __init__(
        self,
        atom_pos: Mapping[
            str, Iterable[float | np.floating | npt.NDArray[np.floating]]
        ],
        avec: npt.NDArray[np.float64],
        offset: Iterable[float] = (0.0, 0.0, 0.0),
        radii: Iterable[float] | None = None,
        colors: Iterable[ColorType] | None = None,
        repeat: tuple[int, int, int] = (1, 1, 1),
        bounds: Mapping[Literal["x", "y", "z"], tuple[float, float]] | None = None,
        mask: Callable | None = None,
        r_factor: float = 0.4,
    ):
        """Properties of a crystal structure for plotting.

        Stores the information required to plot a three dimensional crystal structure.
        The crystal can be repeated along the lattice vectors, and can be masked. Bonds
        can be added between atoms.

        Parameters
        ----------
        atom_pos
            A dictionary of atom positions in the unit cell, where the keys are the atom
            names and the values are the positions of the atoms. The positions can be
            given as a (N, 3) numpy array or a list of N numpy arrays of shape (3,).
        avec
            Lattice vectors of the crystal.
        offset
            Relative offset of the crystal in the plot, by default (0.0, 0.0, 0.0)
        radii
            Sequence of radii for each atom, must be the same length as the number of
            keys given in `atom_pos`. If not provided, all atoms will be given the same
            radius.
        colors
            Sequence of valid matplotlib colors for each atom, must be the same length
            as the number of keys given in `atom_pos`. If not provided, atoms will be
            colored with the current matplotlib prop_cycle in the order they are given
            in `atom_pos`.
        repeat
            Number of times to repeat the crystal in each direction, by default (1, 1,
            1)
        bounds
            A dictionary of bounds for the crystal in each direction. Atoms that exceed
            the bounds will not be plotted
        mask
            A function that takes three cartesian coordinates as the input and returns a
            boolean. If the function returns False, the atom will not be plotted, by
            default `None`
        r_factor
            Additional scaling factor for the bond lengths, by default 0.4

        """
        self.atom_pos_given = atom_pos
        self.avec = avec
        self.offset = np.asarray(offset)

        if radii is None:
            radii = [1.0] * len(self.atoms)
        self.atom_radii: dict[str, float] = dict(zip(self.atoms, radii, strict=True))

        if colors is None:
            colors = [f"C{i}" for i in range(len(self.atoms))]

        self.atom_color: dict[str, str] = {
            k: matplotlib.colors.to_hex(v)
            for k, v in zip(self.atoms, colors, strict=True)
        }

        self.repeat: tuple[int, int, int] = repeat
        self._bounds: Mapping[Literal["x", "y", "z"], tuple[float, float]] = (
            {} if bounds is None else bounds
        )
        self.mask: Callable | None = mask

        self.segments: list[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = []
        self._bond_lw: list[float] = []
        self._bond_c: list[str | tuple[float, ...]] = []

        self.r_factor: float = r_factor

    @classmethod
    def from_fractional(
        cls,
        frac_pos: dict[str, list[tuple[float, float, float]]],
        avec: npt.NDArray[np.float64],
        *args,
        **kwargs,
    ):
        atom_pos: dict[str, list[npt.NDArray[np.float64]]] = {}
        for k, v in frac_pos.items():
            atom_pos[k] = [
                np.asarray(
                    x[0] * avec[0] + x[1] * avec[1] + x[2] * avec[2], dtype=np.float64
                )
                for x in v
            ]
        return cls(atom_pos, avec, *args, **kwargs)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        bound_list = []
        for dim in ("x", "y", "z"):
            try:
                bound_list.append(self._bounds[cast(Literal["x", "y", "z"], dim)])
            except KeyError:
                bound_list.append((-np.inf, np.inf))
        return bound_list

    @property
    def atoms(self) -> list[str]:
        return list(self.atom_pos_given.keys())

    @functools.cached_property
    def atom_pos(self) -> dict[str, npt.NDArray[np.float64]]:
        atom_pos = {k: np.zeros((1, 3)) for k in self.atom_pos_given}
        for k, v in self.atom_pos_given.items():
            for x, y, z in itertools.product(*[range(-n + 1, n) for n in self.repeat]):
                atom_pos[k] = np.r_[
                    atom_pos[k],
                    v
                    + x * self.avec[0]
                    + y * self.avec[1]
                    + z * self.avec[2]
                    + self.offset,
                ]
            atom_pos[k] = atom_pos[k][1:]

        # Remove duplicates
        for k, v in atom_pos.items():
            atom_pos[k] = v[np.unique(v.round(5), axis=0, return_index=True)[1]]

        xlim, ylim, zlim = self.bounds
        valid = {}
        for k, v in atom_pos.items():
            mask_arr = (
                (xlim[0] < v[:, 0])
                * (v[:, 0] < xlim[1])
                * (ylim[0] < v[:, 1])
                * (v[:, 1] < ylim[1])
                * (zlim[0] < v[:, 2])
                * (v[:, 2] < zlim[1])
            )
            if self.mask is not None:
                mask_arr = mask_arr * self.mask(v[:, 0], v[:, 1], v[:, 2])
            valid[k] = np.where(mask_arr)
        masked_atom_pos = {}
        for k, v in atom_pos.items():
            masked_atom_pos[k] = np.asarray(
                [p for i, p in enumerate(v) if i in valid[k][0]]
            )
        return masked_atom_pos

    @property
    def _color_array(self) -> npt.NDArray[np.str_]:
        return np.asarray(
            [
                self.atom_color[k]
                for k, v in self.atom_pos.items()
                for _ in range(len(v))
            ]
        )

    @property
    def _size_array(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                self.atom_radii[k]
                for k, v in self.atom_pos.items()
                for _ in range(len(v))
            ]
        )

    @property
    def _atom_pos_array(self) -> npt.NDArray[np.float64]:
        return np.r_[*list(self.atom_pos.values())]

    def clear_bonds(self) -> None:
        self.segments.clear()
        self._bond_lw.clear()
        self._bond_c.clear()

    def add_bonds(
        self,
        atom1: str,
        atom2: str,
        min_length: float = 0.0,
        max_length: float = 2.6,
        linewidth: float = 0.25,
        color: str | tuple[float, ...] | None = None,
    ) -> None:
        new_seg = self._generate_segments(atom1, atom2, min_length, max_length)
        self.segments += new_seg
        self._bond_lw += [linewidth] * len(new_seg)
        self._bond_c += ["#b2b2b2" if color is None else color] * len(new_seg)

    def plot(
        self,
        ax: mpl_toolkits.mplot3d.Axes3D | None = None,
        scale_bonds: bool = True,
        scale_atoms: bool = True,
        clean_axes: bool = True,
        bond_kw: dict | None = None,
        atom_kw: dict | None = None,
    ):
        """
        Plot the crystal structure.

        Parameters
        ----------
        ax
            A 3D axes object to plot the crystal on. If not provided, add_subplot will
            be called on the current figure.
        scale_bonds
            Whether to scale the bond linewidths based on the distance from the camera,
            by default True
        scale_atoms
            Whether to scale the atom sizes based on the distance from the camera, by
            default True
        clean_axes
            Whether to clean the axes by removing the background and grid, setting pane
            color, and removing the margins, by default True
        bond_kw
            Keyword arguments passed onto `Bond3DCollection`
        atom_kw
            Keyword arguments passed onto `mpl_toolkits.mplot3d.Axes3D.scatter` used to
            plot the atoms.

        """
        if ax is None:
            ax = plt.gcf().add_subplot(projection="3d")
        ax = cast(mpl_toolkits.mplot3d.Axes3D, ax)

        if clean_axes:
            ax.set_facecolor("none")
            ax.xaxis.set_pane_color((1, 1, 1, 0))
            ax.yaxis.set_pane_color((1, 1, 1, 0))
            ax.zaxis.set_pane_color((1, 1, 1, 0))
            ax.grid(False)
            ax.set_xmargin(0)
            ax.set_ymargin(0)

        if bond_kw is None:
            bond_kw = {}
        if atom_kw is None:
            atom_kw = {}

        bond_kw.setdefault("linewidths", self._bond_lw)
        bond_kw.setdefault("colors", self._bond_c)
        bond_kw.setdefault("clip_on", False)
        bond_collection = Bond3DCollection(
            self.segments, scale_linewidths=scale_bonds, **bond_kw
        )
        ax.add_collection(bond_collection)

        atom_kw.setdefault("c", self._color_array)
        atom_kw.setdefault("edgecolors", "none")
        atom_kw.setdefault("clip_on", False)
        atom_kw.setdefault("lw", 0)

        ax.figure.canvas.draw()
        sc = ax.scatter(*self._atom_pos_array.T, **atom_kw)
        sc.__class__ = Atom3DCollection
        sc._scale_size = scale_atoms
        sc.set_sizes(self._size_array**2)

    def _generate_segments(
        self, atom1: str, atom2: str, min_length: float, max_length: float
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        atom1_pos, atom2_pos = self.atom_pos[atom1], self.atom_pos[atom2]
        r1, r2 = self.atom_radii[atom1], self.atom_radii[atom2]
        segments = []
        for p1 in atom1_pos:
            for p2 in atom2_pos:
                vec = p2 - p1
                d = np.linalg.norm(vec)
                vec_u = vec / d
                if d >= min_length and d <= max_length:
                    segments.append(
                        (
                            p1 + vec_u * r1 * self.r_factor,
                            p2 - vec_u * r2 * self.r_factor,
                        )
                    )
        return segments
