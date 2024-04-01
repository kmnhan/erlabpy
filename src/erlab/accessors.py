"""
Some `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ for convenient
data analysis and visualization.

.. currentmodule:: erlab.accessors

"""

__all__ = [
    "PlotAccessor",
    "ImageToolAccessor",
    "SelectionAccessor",
    "MomentumAccessor",
    "OffsetView",
]

import functools
import time
import warnings
from collections.abc import Callable, ItemsView, Iterable, Iterator
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

import erlab.plotting.erplot as eplt
from erlab.analysis.interpolate import interpn
from erlab.analysis.kspace import AxesConfiguration, get_kconv_func, kz_func
from erlab.constants import rel_kconv, rel_kzconv
from erlab.interactive.imagetool import ImageTool, itool
from erlab.interactive.kspace import ktool


class ERLabAccessor:
    """Base class for accessors."""

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj


@xr.register_dataarray_accessor("qplot")
class PlotAccessor(ERLabAccessor):
    """`xarray.DataArray.qplot` accessor for plotting data."""

    def __call__(self, *args, **kwargs):
        """
        Plot the data. If a 2D data array is provided, it is plotted using
        :func:`plot_array <erlab.plotting.general.plot_array>`. Otherwise, it is
        equivalent to calling :meth:`xarray.DataArray.plot`.

        Parameters
        ----------
        *args
            Positional arguments to be passed to the plotting function.
        **kwargs
            Keyword arguments to be passed to the plotting function.

        """
        if len(self._obj.dims) == 2:
            return eplt.plot_array(self._obj, *args, **kwargs)
        else:
            return self._obj.plot(*args, **kwargs)


@xr.register_dataarray_accessor("qshow")
class ImageToolAccessor(ERLabAccessor):
    """`xarray.DataArray.qshow` accessor for interactive visualization."""

    def __call__(self, *args, **kwargs) -> ImageTool:
        if len(self._obj.dims) >= 2:
            return itool(self._obj, *args, **kwargs)
        else:
            raise ValueError("Data must have at leasst two dimensions.")


@xr.register_dataarray_accessor("qsel")
class SelectionAccessor(ERLabAccessor):
    """
    `xarray.DataArray.qsel` accessor for conveniently selecting and averaging
    data.
    """

    def __call__(
        self,
        indexers: dict[str, float | slice] | None = None,
        *,
        verbose: bool = False,
        **indexers_kwargs: dict[str, float | slice],
    ):
        """Select and average data along specified dimensions.

        Parameters
        ----------
        indexers
            Dictionary specifying the dimensions and their values or slices.
            Position along a dimension can be specified in three ways:

            - As a scalar value: `alpha=-1.2`

              If no width is specified, the data is selected along the nearest value. It
              is equivalent to `xarray.DataArray.sel` with `method='nearest'`.

            - As a value and width: `alpha=5, alpha_width=0.5`

              The data is *averaged* over a slice of width `alpha_width`, centered at
              `alpha`.

            - As a slice: `alpha=slice(-10, 10)`

              The data is selected over the specified slice. No averaging is performed.

            One of `indexers` or `indexers_kwargs` must be provided.
        verbose
            If `True`, print information about the selected data and averaging process.
            Default is `False`.
        **indexers_kwargs
            The keyword arguments form of `indexers`. One of `indexers` or
            `indexers_kwargs` must be provided.

        Returns
        -------
        xarray.DataArray
            The selected and averaged data.

        Raises
        ------
        ValueError
            If a specified dimension is not present in the data.
        """

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "qsel")

        # Bin widths for each dimension, zero if width not specified
        bin_widths: dict[str, float] = {}

        for dim in indexers:
            if not dim.endswith("_width"):
                bin_widths[dim] = indexers.get(f"{dim}_width", 0.0)
                if dim not in self._obj.dims:
                    raise ValueError(f"Dimension `{dim}` not found in data.")

        scalars: dict[str, float] = {}
        slices: dict[str, slice] = {}
        avg_dims: list[str] = []

        for dim, width in bin_widths.items():
            if width == 0.0:
                if isinstance(indexers[dim], slice):
                    slices[dim] = indexers[dim]
                else:
                    scalars[dim] = float(indexers[dim])
            else:
                slices[dim] = slice(
                    indexers[dim] - width / 2, indexers[dim] + width / 2
                )
                avg_dims.append(dim)

        if len(scalars) >= 1:
            out = self._obj.sel(**scalars, method="nearest")
        else:
            out = self._obj

        if len(slices) >= 1:
            out = out.sel(**slices)

            lost_coords = {k: out[k].mean() for k in avg_dims}
            out = out.mean(dim=avg_dims, keep_attrs=True)
            out = out.assign_coords(lost_coords)

        if verbose:
            print(
                f"Selected data with {scalars} and {slices}, averaging over {avg_dims}"
            )

        return out


def only_angles(method: Callable | None = None):
    """
    A decorator that ensures the data is in angle space before executing the decorated
    method.

    If the data is not in angle space (i.e., if "kx" or "ky" dimensions are present), a
    `ValueError` is raised.
    """

    def wrapper(method: Callable):
        @functools.wraps(method)
        def _impl(self, *args, **kwargs):
            if "kx" in self._obj.dims or "ky" in self._obj.dims:
                raise ValueError(
                    f"`{method.__name__}` cannot be called for data in momentum space."
                )
            return method(self, *args, **kwargs)

        return _impl

    if method is not None:
        return wrapper(method)
    return wrapper


def only_momentum(method: Callable | None = None):
    """
    A decorator that ensures the data is in momentum space before executing the
    decorated method.

    If the data is not in momentum space (i.e., if "kx" nor "ky" dimensions are
    present), a `ValueError` is raised.
    """

    def wrapper(method: Callable):
        @functools.wraps(method)
        def _impl(self, *args, **kwargs):
            if not ("kx" in self._obj.dims or "ky" in self._obj.dims):
                raise ValueError(
                    f"`{method.__name__}` cannot be called for data in angle space."
                )
            return method(self, *args, **kwargs)

        return _impl

    if method is not None:
        return wrapper(method)
    return wrapper


class OffsetView:
    r"""A class representing an offset view for an `xarray.DataArray`.

    This class provides a convenient way to access and manipulate angle offsets
    associated with the given data.

    Parameters
    ----------
    xarray_obj
        The `xarray.DataArray` for which the offset view is created.

    Methods
    -------
    __len__() -> int:
        Returns the number of valid offset keys.

    __iter__() -> Iterator[str, float]:
        Returns an iterator over the valid offset keys and their corresponding values.

    __getitem__(key: str) -> float:
        Returns the offset value associated with the given key.

    __setitem__(key: str, value: float) -> None:
        Sets the offset value for the given key.

    __eq__(other: object) -> bool:
        Compares the offset view with another object for equality. `True` if the
        dictionary representation is equal, `False` otherwise.

    __repr__() -> str:
        Returns a string representation of the offset view.

    _repr_html_() -> str:
        Returns an HTML representation of the offset view.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        for k in self._obj.kspace.valid_offset_keys:
            if k + "_offset" not in self._obj.attrs:
                self[k] = 0.0

    def __len__(self) -> int:
        return len(self._obj.kspace.valid_offset_keys)

    def __iter__(self) -> Iterator[str, float]:
        for key in self._obj.kspace.valid_offset_keys:
            yield key, self.__getitem__(key)

    def __getitem__(self, key: str) -> float:
        if key in self._obj.kspace.valid_offset_keys:
            return float(self._obj.attrs[key + "_offset"])
        else:
            raise KeyError(
                f"Invalid offset key `{key}` for experimental configuration "
                f"{self._obj.kspace.configuration}"
            )

    def __setitem__(self, key: str, value: float) -> None:
        if key in self._obj.kspace.valid_offset_keys:
            self._obj.attrs[key + "_offset"] = float(value)

    def __eq__(self, other: object) -> bool:
        return dict(self) == dict(other)

    def __repr__(self) -> str:
        return dict(self).__repr__()

    def _repr_html_(self) -> str:
        out = ""
        out += "<table><tbody>"
        for k, v in self.items():
            out += (
                "<tr>"
                f"<td style='text-align:left;'><b>{k}</b></td>"
                f"<td style='text-align:left;'>{v}</td>"
                "</tr>"
            )
        out += "</tbody></table>"
        return out

    def update(
        self,
        other: dict | Iterable[tuple[str, float]] | None = None,
        **kwargs: dict[str, float],
    ) -> "OffsetView":
        """Updates the offset view with the provided key-value pairs."""
        if other is not None:
            for k, v in other.items() if isinstance(other, dict) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v
        return self

    def items(self) -> ItemsView[str, float]:
        """Returns a view of the offset view as a list of (key, value) pairs."""
        return dict(self).items()

    def reset(self) -> "OffsetView":
        """Reset all angle offsets to zero."""
        for k in self._obj.kspace.valid_offset_keys:
            self[k] = 0.0
        return self


@xr.register_dataarray_accessor("kspace")
class MomentumAccessor:
    """`xarray.DataArray.kspace` accessor for momentum conversion related utilities.

    This class provides convenient access to various momentum-related properties of a
    data object. It allows getting and setting properties such as configuration, inner
    potential, work function, angle resolution, slit axis, momentum axes, angle
    parameters, and offsets.

    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    @property
    def configuration(self) -> AxesConfiguration:
        """Returns the experimental configuration.

        For a properly implemented data loader, the configuration attribute must be set
        on data import. See :class:`erlab.analysis.kspace.AxesConfiguration` for
        details.
        """
        if "configuration" not in self._obj.attrs:
            raise ValueError(
                "Configuration not found in data attributes! "
                "Data attributes may have been discarded since initial import."
            )
        return AxesConfiguration(int(self._obj.attrs.get("configuration")))

    @configuration.setter
    def configuration(self, value: AxesConfiguration | int):
        self._obj.attrs["configuration"] = int(value)

    @property
    def inner_potential(self) -> float:
        """Inner potential of the sample in eV.

        The inner potential is stored in the ``inner_potential`` attribute of the data.
        If the inner potential is not set, a warning is issued and a default value of
        10.0 eV is assumed.

        Note
        ----
        This property provides a setter method that takes a float value and sets the
        data attribute accordingly.

        Example
        -------
        >>> data.kspace.inner_potential = 13.0
        >>> data.kspace.inner_potential
        13.0
        """

        if "inner_potential" in self._obj.attrs:
            return float(self._obj.attrs["inner_potential"])
        else:
            warnings.warn(
                "Inner potential not found in data attributes, assuming 10 eV",
                stacklevel=1,
            )
            return 10.0

    @inner_potential.setter
    def inner_potential(self, value: float):
        self._obj.attrs["inner_potential"] = float(value)

    @property
    def work_function(self) -> float:
        """Work function of the sample in eV.

        The work function is stored in the ``sample_workfunction`` attribute of the
        data. If the work function is not set, a warning is issued and a default value
        of 4.5 eV is assumed.

        Note
        ----
        This property provides a setter method that takes a float value and sets the
        data attribute accordingly.

        Example
        -------
        >>> data.kspace.work_function = 4.5
        >>> data.kspace.work_function
        4.5
        """

        if "sample_workfunction" in self._obj.attrs:
            return float(self._obj.attrs["sample_workfunction"])
        else:
            warnings.warn(
                "Work function not found in data attributes, assuming 4.5 eV",
                stacklevel=1,
            )
            return 4.5

    @work_function.setter
    def work_function(self, value: float):
        self._obj.attrs["sample_workfunction"] = float(value)

    @property
    def angle_resolution(self) -> float:
        """Angular resolution of the data in degrees.

        The angular resolution is stored in the ``angle_resolution`` attribute of the
        data. If it is not set, a default value of 0.1° is silently used. It is used in
        `best_kp_resolution` when automatically estimating momentum steps through
        `estimate_resolution`.
        """

        try:
            return float(self._obj.attrs["angle_resolution"])
        except KeyError:
            # warnings.warn(
            #     "Angle resolution not found in data attributes, assuming 0.1 degrees"
            # )
            return 0.1

    @angle_resolution.setter
    def angle_resolution(self, value: float):
        self._obj.attrs["angle_resolution"] = float(value)

    @property
    def slit_axis(self) -> str:
        """Returns the momentum axis parallel to the slit.

        Returns
        -------
        str
            Returns ``'kx'`` for type 1 configurations, ``'ky'`` otherwise.
        """
        match self.configuration:
            case AxesConfiguration.Type1 | AxesConfiguration.Type1DA:
                return "kx"
            case _:
                return "ky"

    @property
    @only_angles
    def momentum_axes(self) -> tuple[str, ...]:
        """Returns the momentum axes of the data after conversion.

        Returns
        -------
        tuple
            For photon energy dependent scans, it returns the slit axis and ``'kz'``.
            For maps, it returns ``'kx'`` and ``'ky'``. Otherwise, it returns only the
            slit axis.

        """
        if self.has_hv:
            return (self.slit_axis, "kz")
        elif self.has_beta:
            return ("kx", "ky")
        else:
            return (self.slit_axis,)

    @property
    def angle_params(self) -> dict[str, float]:
        """Parameters passed to :func:`erlab.analysis.kspace.get_kconv_func`."""
        params = {
            "delta": self.offsets["delta"],
            "xi": float(self._obj["xi"].values),
            "xi0": self.offsets["xi"],
        }
        match self.configuration:
            case AxesConfiguration.Type1 | AxesConfiguration.Type2:
                params["beta0"] = self.offsets["beta"]
            case _:
                params["chi"] = float(self._obj["chi"].values)
                params["chi0"] = self.offsets["chi"]
        return params

    @property
    def _photon_energy(self) -> npt.NDArray[np.floating] | float:
        # Make photon energy axis always comes last
        if self.has_hv:
            return self._obj.hv.values[
                tuple(
                    slice(None) if (i == self._obj.ndim - 1) else np.newaxis
                    for i in range(self._obj.ndim)
                )
            ]
        else:
            # Scalar
            return float(self._obj.hv)

    @property
    def _beta(self) -> npt.NDArray[np.floating] | float:
        if self.has_beta:
            if self.has_hv:
                # Assume hv-dependent beta, make same shape as hv (last axis)
                return self._obj["beta"].values[
                    tuple(
                        slice(None) if (i == self._obj.ndim - 1) else np.newaxis
                        for i in range(self._obj.ndim)
                    )
                ]
            else:
                # First axis
                return self._obj["beta"].values[
                    tuple(
                        slice(None) if (i == 0) else np.newaxis
                        for i in range(self._obj.ndim)
                    )
                ]
        else:
            # Scalar
            return float(self._obj["beta"])

    @property
    @only_angles
    def _alpha(self) -> npt.NDArray[np.floating] | float:
        if not self.has_beta and not self.has_hv:
            return self._obj["alpha"].values
        else:
            return self._obj["alpha"].values[
                tuple(
                    slice(None) if (i == 0) else np.newaxis
                    for i in range(self._obj.ndim)
                )
            ]

    @property
    def _binding_energy(self) -> npt.NDArray[np.floating] | float:
        if self.has_eV:
            return self._obj.eV.values[:, np.newaxis]
        else:
            return float(self._obj.eV.values)

    @property
    def _kinetic_energy(self) -> npt.NDArray[np.floating] | float:
        return self._photon_energy - self.work_function + self._binding_energy

    @property
    def has_eV(self) -> bool:
        """Returns `True` if object has an energy axis."""
        return "eV" in self._obj.dims

    @property
    @only_angles
    def has_hv(self) -> bool:
        """Returns `True` for photon energy dependent data."""
        return self._obj["hv"].size > 1

    @property
    @only_angles
    def has_beta(self) -> bool:
        """Check if the coordinate array for :math:`β` has more than one element.

        Returns
        -------
        bool
            Returns `True` if the size of the coordinate array for :math:`β` is greater
            than 1, `False` otherwise.

        Note
        ----
        This may be `True` for data that are not maps. For instance,
        :math:`hν`-dependent cuts with an in-plane momentum offset may have a
        :math:`hν`-dependent :math:`β` offset.

        """
        return self._obj.beta.size > 1

    @property
    def valid_offset_keys(self) -> tuple[str, ...]:
        """
        Get valid offset angles based on the experimental configuration.

        Returns
        -------
        tuple
            A tuple containing the valid offset keys. For configurations with a
            deflector, returns ``("delta", "chi", "xi")``. Otherwise, returns
            ``("delta", "xi", "beta")``.
        """
        match self.configuration:
            case AxesConfiguration.Type1 | AxesConfiguration.Type2:
                return ("delta", "xi", "beta")
            case _:
                return ("delta", "chi", "xi")

    @property
    def offsets(self) -> OffsetView:
        """Angle offsets used in momentum conversion.

        Returns
        -------
        OffsetView
            A mapping between valid offset keys and their corresponding offsets.

        Examples
        --------

        - View all offsets

          >>> data.kspace.offsets
          {'delta': 0.0, 'xi': 0.0, 'beta': 0.0}

        - View single offset

          >>> data.kspace.offsets["beta"]
          0.0

        - Offsets to dictionary

          >>> dict(data.kspace.offsets)
          {'delta': 0.0, 'xi': 0.0, 'beta': 0.0}

        - Set single offset

          >>> data.kspace.offsets["beta"] = 3.0
          >>> data.kspace.offsets
          {'delta': 0.0, 'xi': 0.0, 'beta': 3.0}

        - Overwrite offsets with dictionary

          >>> data.kspace.offsets = dict(delta=1.5, xi=2.7)
          >>> data.kspace.offsets
          {'delta': 1.5, 'xi': 2.7, 'beta': 0.0}

        - Update offsets

          >>> data.kspace.offsets.update(beta=0.1, xi=0.0)
          {'delta': 1.5, 'xi': 0.0, 'beta': 0.1}

        - Reset all offsets

          >>> data.kspace.offsets.reset()
          {'delta': 0.0, 'xi': 0.0, 'beta': 0.0}
        """
        if not hasattr(self, "_offsetview"):
            self._offsetview = OffsetView(self._obj)

        return self._offsetview

    @offsets.setter
    def offsets(self, offset_dict: dict[str, float]):
        if not hasattr(self, "_offsetview"):
            self._offsetview = OffsetView(self._obj)

        self._offsetview.reset()
        self._offsetview.update(offset_dict)

    def estimate_bounds(self) -> dict[str, tuple[float, float]]:
        """
        Estimates the bounds of the data in momentum space based on the available
        parameters.

        Returns
        -------
        bounds : dict[str, tuple[float, float]]
            A dictionary containing the estimated bounds for each parameter. The keys of
            the dictionary are 'kx', 'ky', and 'kz' (for :math:`hν`-dependent data). The
            values are tuples representing the minimum and maximum values.

        """
        if self.has_hv:
            # hv dependent cut
            kx, ky = self._forward_func_raw(self._alpha, self._beta)
            kz = kz_func(self._kinetic_energy, self.inner_potential, kx, ky)

            return {
                "kx": (kx.min(), kx.max()),
                "ky": (ky.min(), ky.max()),
                "kz": (kz.min(), kz.max()),
            }

        elif self.has_beta:
            # kxy map
            # construct boundary array from meshgrid
            alpha, beta = (
                np.r_[arr[0, :-1], arr[:-1, -1], arr[-1, ::-1], arr[-2:0:-1, 0]]
                for arr in np.meshgrid(self._obj["alpha"], self._obj["beta"])
            )

            kx, ky = self._forward_func_raw(alpha[np.newaxis, :], beta[np.newaxis, :])

            return {"kx": (kx.min(), kx.max()), "ky": (ky.min(), ky.max())}

        else:
            # just cut
            kx, ky = self._forward_func_raw(self._alpha, self._beta)

            return {"kx": (kx.min(), kx.max()), "ky": (ky.min(), ky.max())}

    @only_angles
    def estimate_resolution(
        self,
        axis: Literal["kx", "ky", "kz"],
        lims: tuple[float, float] | None = None,
        from_numpoints: bool = False,
    ) -> float:
        """Estimate the resolution for a given momentum axis.

        Parameters
        ----------
        axis
            Axis to estimate the resolution for.
        lims
            The limits of the axis used when `from_numpoints` is `True`. If not
            provided, reasonable limits will be calculated by :meth:`estimate_bounds`,
            by default `None`
        from_numpoints
            If `True`, estimate the resolution from the number of points in the relevant
            axis. If `False`, estimate the resolution based on the data, by default
            `False`

        Returns
        -------
        float
            The estimated resolution.

        Raises
        ------
        ValueError
            If no photon energy axis is found in data for axis ``'kz'``.

        """
        if from_numpoints and (lims is None):
            lims = self.estimate_bounds()[axis]

        if axis == self.slit_axis:
            dim = "alpha"

        elif axis == "kz":
            dim = "hv"

            if not self.has_hv:
                raise ValueError("No photon energy axis found.")

        else:
            dim = "beta"

        if from_numpoints:
            return float((lims[1] - lims[0]) / len(self._obj[dim]))
        elif axis == "kz":
            return self.best_kz_resolution
        else:
            return self.best_kp_resolution

    @property
    @only_angles
    def best_kp_resolution(self) -> float:
        r"""
        Estimates the minimum in-plane momentum resolution based on the kinetic energy
        and angular resolution:

        .. math:: \Delta k_{\parallel} \sim \sqrt{2 m_e E_k/\hbar^2} \cos(\alpha) \Delta\alpha

        """
        min_Ek = np.amin(self._kinetic_energy)
        max_angle = max(np.abs(self._obj["alpha"].values))
        return (
            rel_kconv
            * np.sqrt(min_Ek)
            * np.cos(np.deg2rad(max_angle))
            * np.deg2rad(self.angle_resolution)
        )

    @property
    @only_angles
    def best_kz_resolution(self) -> float:
        r"""
        Estimates the minimum out-of-plane momentum resolution based on the mean free
        path :cite:p:`Seah1979` and the kinetic energy.

        .. math:: \Delta k_z \sim 1/\lambda

        """
        kin = self._kinetic_energy.flatten()
        c1, c2 = 641.0, 0.096
        imfp = (c1 / (kin**2) + c2 * np.sqrt(kin)) * 10
        return np.amin(1 / imfp)

    def _forward_func_raw(self, alpha, beta):
        return get_kconv_func(
            self._kinetic_energy, self.configuration, self.angle_params
        )[0](alpha, beta)

    def _inverse_func_raw(self, kx, ky, kz=None):
        return get_kconv_func(
            self._kinetic_energy, self.configuration, self.angle_params
        )[1](kx, ky, kz)

    # def forward(self, phi, theta):
    #     # make alpha and beta (1,N) where N = len(phi) * len(theta)
    #     alpha, beta = np.array(np.meshgrid(phi, theta)).reshape(2, -1)[:, None]

    #     if self.has_hv:
    #         alpha = alpha[:, :, np.newaxis]
    #         beta = beta[:, :, np.newaxis]
    #     # binding energy 없이 photon energy만 주어지는 경우는 없겠지?

    #     return self._forward_func(alpha, beta)

    def _inverse(
        self,
        kx: npt.NDArray[np.floating],
        ky: npt.NDArray[np.floating],
        kz: npt.NDArray[np.floating] | None = None,
    ) -> dict[str, npt.NDArray[np.floating]]:
        if self.has_eV:
            ndim = self._kinetic_energy.ndim
        else:
            ndim = 1
        slicer = tuple(
            slice(None) if (i == ndim - 1) else np.newaxis for i in range(ndim)
        )

        if kz is None:
            # make kxval and kyval (...,N) where N = len(kx) * len(ky)
            kxval, kyval = np.array(np.meshgrid(kx, ky)).reshape(2, -1)
            kxval, kyval = kxval[slicer], kyval[slicer]
            kzval = None
        else:
            kxval, kyval, kzval = np.array(np.meshgrid(kx, ky, kz)).reshape(3, -1)
            kxval, kyval, kzval = kxval[slicer], kyval[slicer], kzval[slicer]

        alpha, beta = self._inverse_func_raw(kxval, kyval, kzval)

        out_dict = {"alpha": alpha, "beta": beta}

        if self.has_eV:
            # Ensure all arrays have the same shape
            binding = self._binding_energy * np.ones_like(alpha)
            for k in out_dict:
                out_dict[k] = out_dict[k] * np.ones_like(binding)
            out_dict["eV"] = binding

        if self.has_hv:
            # Photon energy
            out_dict["hv"] = (
                rel_kzconv * (kxval**2 + kyval**2 + kzval**2)
                - self.inner_potential
                + self.work_function
            )
            if self.has_eV:
                out_dict["hv"] = out_dict["hv"] - out_dict["eV"]
            else:
                out_dict["hv"] = out_dict["hv"] - self._binding_energy

        return out_dict

    @only_angles
    def convert(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        resolution: dict[str, float] | None = None,
        *,
        silent: bool = False,
    ) -> xr.DataArray:
        """Convert to momentum space.

        Parameters
        ----------
        bounds
            A dictionary specifying the bounds for each coordinate axis. The keys are
            the names of the axes, and the values are tuples representing the lower and
            upper bounds of the axis. If not provided, the bounds will be estimated
            based on the data.
        resolution
            A dictionary specifying the resolution for each momentum axis. The keys are
            the names of the axes, and the values are floats representing the desired
            resolution of the axis. If not provided, the resolution will be estimated
            based on the data. For in-plane momentum, the resolution is estimated from
            the angle resolution and kinetic energy. For out-of-plane momentum, two
            values are calculated. One is based on the number of photon energy points,
            and the other is estimated as the inverse of the photoelectron inelastic
            mean free path given by the universal curve. The resolution is estimated as
            the smaller of the two values.
        silent
            If `True`, suppresses printing, by default `False`.

        Returns
        -------
        xarray.DataArray
            The converted data.

        Note
        ----
        This method converts the data to a new coordinate system specified by the
        provided bounds and resolution. It uses interpolation to map the data from the
        original coordinate system to the new one.

        The converted data is returned as a DataArray object with updated coordinates
        and dimensions.

        Examples
        --------
        Set parameters and convert with automatic bounds and resolution:

        .. code-block:: python

            data.kspace.offsets = {"delta": 0.1, "xi": 0.0, "beta": 0.3}
            data.kspace.work_function = 4.3
            data.kspace.inner_potential = 12.0
            converted_data = data.kspace.convert()


        Convert with specified bounds and resolution:

        .. code-block:: python

            bounds = {"kx": (0.0, 1.0), "ky": (-1.0, 1.0)}
            resolution = {"kx": 0.01, "ky": 0.01}
            converted_data = data.kspace.convert(bounds, resolution)

        """

        if bounds is None:
            bounds = {}

        if resolution is None:
            resolution = {}

        if not silent:
            print("Estimating bounds and resolution")

        calculated_bounds: dict[str, tuple[float, float]] = self.estimate_bounds()

        new_size: dict[str, int] = {}
        interp_coords: dict[str, np.ndarray] = {}
        other_coords: dict[str, np.ndarray] = {}

        for k, lims in calculated_bounds.items():
            if k in self.momentum_axes:
                lims = bounds.get(k, lims)

                res = self.estimate_resolution(k, lims, from_numpoints=False)
                if k == "kz":
                    res_n = self.estimate_resolution(k, lims, from_numpoints=True)
                    res = min(res, res_n)
                res = resolution.get(k, res)

                new_size[k] = round((lims[1] - lims[0]) / res + 1)

                interp_coords[k] = np.linspace(*lims, new_size[k])
            else:
                # Take the mean for axes that will not appear in converted data
                if not silent and lims[1] - lims[0] > 0.001:
                    print(f"Data spans about {lims[1] - lims[0]:.3f} Å⁻¹ of {k}")
                other_coords[k] = np.array([(lims[0] + lims[1]) / 2])

        if not silent:
            print("Calculating destination coordinates")

        target_dict = self._inverse(
            (interp_coords | other_coords).get("kx"),
            (interp_coords | other_coords).get("ky"),
            (interp_coords | other_coords).get("kz", None),
        )

        if self.has_hv:
            old_dim_order = ("alpha", "hv")
            new_dim_order = (self.slit_axis, "kz")

        elif self.has_beta:
            old_dim_order = ("alpha", "beta")
            new_dim_order = ("ky", "kx")
        else:
            old_dim_order = ("alpha",)
            new_dim_order = (self.slit_axis,)

        if self.has_eV:
            old_dim_order = ("eV", *old_dim_order)
            new_dim_order = ("eV", *new_dim_order)
            new_size["eV"] = len(self._binding_energy)
            interp_coords["eV"] = np.squeeze(self._binding_energy)

        target = tuple(target_dict[dim] for dim in old_dim_order)

        for d in self._obj.dims:
            if d not in old_dim_order:
                old_dim_order += (d,)
                new_dim_order += (d,)
                new_size[d] = len(self._obj[d])
                interp_coords[d] = self._obj[d].values

        # Get non-dimension coordinates
        for d, c in self._obj.coords.items():
            if d not in interp_coords and d not in other_coords:
                if c.ndim == 0:
                    other_coords[d] = c.values

        # Only keep scalar coordinates
        for k, v in other_coords.items():
            if v.size != 1:
                del other_coords[k]
            else:
                other_coords[k] = v.item()

        if not silent:
            print(f"Converting {old_dim_order}  ->  {new_dim_order}")
            t_start = time.perf_counter()

        out = xr.DataArray(
            interpn(
                tuple(self._obj[dim].values for dim in old_dim_order),
                self._obj.copy(deep=True).transpose(*old_dim_order).values,
                target,
                bounds_error=False,
            ).reshape(tuple(new_size[d] for d in new_dim_order)),
            coords=interp_coords,
            dims=new_dim_order,
        )
        out = out.assign_attrs(self._obj.attrs)
        out = out.assign_coords(other_coords)
        if not silent:
            print(f"Interpolated in {time.perf_counter() - t_start:.3f} s")

        return out

    def interactive(self, **kwargs) -> ktool:
        """Open the interactive momentum space conversion tool."""
        if self._obj.ndim < 3:
            raise ValueError("Interactive tool requires three-dimensional data.")
        return ktool(self._obj, **kwargs)

    @only_momentum
    def hv_to_kz(self, hv: float) -> xr.DataArray:
        """Returns :math:`k_z` for a given photon energy.

        Useful when creating overlays on :math:`hν`-dependent data.

        Parameters
        ----------
        hv
            Photon energy in eV.

        Note
        ----
        This will be inexact for hv-dependent cuts that do not pass through the BZ
        center since we lost the exact angle values, i.e. the exact momentum
        perpendicular to the slit, during momentum conversion.
        """
        # Get kinetic energies for the given photon energy
        kinetic = hv - self.work_function + self._obj.eV

        # Get momentum conversion functions
        ang2k, k2ang = get_kconv_func(kinetic, self.configuration, self.angle_params)

        # Transformation yields in-plane momentum at given photon energy
        kx, ky = ang2k(*k2ang(self._obj.kx, self._obj.ky))

        # Calculate kz
        kz = kz_func(kinetic, self.inner_potential, kx, ky)

        return kz
