"""
Some `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ for convenient
data analysis and visualization.

.. currentmodule:: erlab.accessors

"""

__all__ = ["MomentumAccessor"]

import functools
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

import erlab.interactive
from erlab.analysis.interpolate import interpn
from erlab.analysis.kspace import AxesConfiguration, get_kconv_func, kz_func
from erlab.constants import rel_kconv, rel_kzconv
from erlab.interactive.imagetool import itool


class ERLabAccessor:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj


@xr.register_dataarray_accessor("er")
class ERLabDataArrayAccessor(ERLabAccessor):
    def show(self, *args, **kwargs):
        return itool(self._obj, *args, **kwargs)


def only_angles(method: Callable | None = None):
    """
    A decorator that ensures the data is in angle space before executing the decorated method.

    If the data is not in angle space (i.e., if "kx" or "ky" dimensions are present), a `ValueError` is raised.

    """

    def wrapper(method: Callable):
        @functools.wraps(method)
        def _impl(self, *args, **kwargs):
            if "kx" in self._obj.dims or "ky" in self._obj.dims:
                raise ValueError("Data is in momentum space.")
            return method(self, *args, **kwargs)

        return _impl

    if method is not None:
        return wrapper(method)
    return wrapper


@xr.register_dataarray_accessor("kspace")
class MomentumAccessor:

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        for k in self.valid_offset_keys:
            if k + "_offset" not in self._obj.attrs:
                self._obj.attrs[k + "_offset"] = 0.0

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

    @property
    def inner_potential(self) -> float:
        """Inner potential of the sample in eV.

        The inner potential is stored in the ``inner_potential`` attribute of the data. If
        the inner potential is not set, a warning is issued and a default value of 10.0
        eV is assumed.

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
                "Inner potential not found in data attributes, assuming 10 eV"
            )
            return 10.0

    @inner_potential.setter
    def inner_potential(self, value: float):
        self._obj.attrs["inner_potential"] = float(value)

    @property
    def work_function(self) -> float:
        """Work function of the sample in eV.

        The work function is stored in the ``sample_workfunction`` attribute of the data.
        If the work function is not set, a warning is issued and a default value of 4.5
        eV is assumed.

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
            warnings.warn("Work function not found in data attributes, assuming 4.5 eV")
            return 4.5

    @work_function.setter
    def work_function(self, value: float):
        self._obj.attrs["sample_workfunction"] = float(value)

    @property
    def angle_resolution(self) -> float:
        """Angular resolution of the data in degrees.

        The angular resolution is stored in the ``angle_resolution`` attribute of the
        data. If it is not set, a default value of 0.1° is silently used.
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
        """Returns the momentum axes of the data.

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
        params = dict(
            delta=self.get_offset("delta"),
            xi=float(self._obj["xi"].values),
            xi0=self.get_offset("xi"),
        )
        match self.configuration:
            case AxesConfiguration.Type1 | AxesConfiguration.Type2:
                params["beta0"] = self.get_offset("beta")
            case _:
                params["chi"] = float(self._obj["chi"].values)
                params["chi0"] = self.get_offset("chi")
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
    def offsets(self) -> dict[str, float]:
        """Angle offsets used in momentum conversion.

        Returns
        -------
        dict
            A mapping between valid offset keys and their corresponding offsets.

        Note
        ----
        This property provides a setter method that takes a dictionary of valid offset
        keys and their corresponding offsets. Assigning new offsets will reset all
        previous offsets including the ones not specified in the new dictionary. To
        update only a subset of the offsets, use the :meth:`set_offsets` method.

        Example
        -------
        >>> data.kspace.offsets
        {'delta': 0.0, 'xi': 0.0, 'beta': 0.0}
        >>> data.kspace.offsets = dict(delta=1.5, xi=2.7)
        >>> data.kspace.offsets
        {'delta': 1.5, 'xi': 2.7, 'beta': 0.0}
        """
        return {k: self.get_offset(k) for k in self.valid_offset_keys}

    @offsets.setter
    def offsets(self, offset_dict: dict[str, float]):
        self.reset_offsets()
        self.set_offsets(**offset_dict)

    def reset_offsets(self) -> None:
        """Reset all angle offsets to zero."""
        for k in self.valid_offset_keys:
            self._obj.attrs[k + "_offset"] = 0.0

    def get_offset(self, axis: str) -> float:
        """
        Retrieve the offset for the specified angle coordinate. Valid keys differ based
        on the experimental configuration.

        See :attr:`valid_offset_keys` for details.
        """
        return float(self._obj.attrs[axis + "_offset"])

    def set_offsets(self, **kwargs) -> None:
        """Set the offsets for specified angle coordinates.

        Valid keys differ based on the experimental configuration. See
        :attr:`valid_offset_keys` for details.

        """
        for k, v in kwargs.items():
            if k in self.valid_offset_keys:
                self._obj.attrs[k + "_offset"] = float(v)

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

            return dict(
                kx=(kx.min(), kx.max()),
                ky=(ky.min(), ky.max()),
                kz=(kz.min(), kz.max()),
            )

        elif self.has_beta:

            # kxy map
            # construct boundary array from meshgrid
            alpha, beta = [
                np.r_[arr[0, :-1], arr[:-1, -1], arr[-1, ::-1], arr[-2:0:-1, 0]]
                for arr in np.meshgrid(self._obj["alpha"], self._obj["beta"])
            ]

            kx, ky = self._forward_func_raw(alpha[np.newaxis, :], beta[np.newaxis, :])

            return dict(kx=(kx.min(), kx.max()), ky=(ky.min(), ky.max()))

        else:
            # just cut
            kx, ky = self._forward_func_raw(self._alpha, self._beta)

            return dict(kx=(kx.min(), kx.max()), ky=(ky.min(), ky.max()))

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
        else:
            if axis == "kz":
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
        path :cite:p:`SD79` and the kinetic energy.

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

        out_dict = dict(alpha=alpha, beta=beta)

        if self.has_eV:
            # Ensure all arrays have the same shape
            binding = self._binding_energy * np.ones_like(alpha)
            for k in out_dict.keys():
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
        silent: bool = False,
    ) -> xr.DataArray:
        """
        Convert to momentum space.

        Parameters
        ----------
        bounds
            A dictionary specifying the bounds for each coordinate axis.
            The keys are the names of the axes, and the values are tuples
            representing the lower and upper bounds of the axis. If not
            provided, the bounds will be estimated based on the data.
        resolution
            A dictionary specifying the resolution for each momentum axis. The keys are
            the names of the axes, and the values are floats representing the desired
            resolution of the axis. If not provided, the resolution will be estimated
            based on the data. For in-plane momentum, the resolution is estimated from
            the angle resolution and kinetic energy. For out-of-plane momentum, two
            values are calculated. One is based on the number of photon energy points,
            and the other is estimated as the inverse of the photoelectron inelastic mean
            free path given by the universal curve. The resolution is estimated as the
            smaller of the two values.
        silent
            If `True`, suppresses printing, by default `False`.

        Returns
        -------
        xarray.DataArray
            The converted data.

        Note
        ----
        This method converts the data to a new coordinate system specified
        by the provided bounds and resolution. It uses interpolation to
        map the data from the original coordinate system to the new one.

        The converted data is returned as a DataArray object with updated
        coordinates and dimensions.

        Examples
        --------
        Set parameters and convert with automatic bounds and resolution:

        .. code-block:: python

            data.kspace.offsets = {'delta': 0.1, 'xi': 0.0, 'beta': 0.3}
            data.kspace.work_function = 4.3
            data.kspace.inner_potential = 12.0
            converted_data = data.kspace.convert()


        Convert with specified bounds and resolution:

        .. code-block:: python

            bounds = {'kx': (0.0, 1.0), 'ky': (-1.0, 1.0)}
            resolution = {'kx': 0.01, 'ky': 0.01}
            converted_data = data.kspace.convert(bounds, resolution)

        """

        if bounds is None:
            bounds = dict()

        if resolution is None:
            resolution = dict()

        calculated_bounds: dict[str, tuple[float, float]] = self.estimate_bounds()

        new_size: dict[str, int] = dict()
        interp_coords: dict[str, np.ndarray] = dict()
        other_coords: dict[str, np.ndarray] = dict()

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
                    print(f"Data spans about {lims[1] - lims[0]:.3f} Å⁻¹ of {k}.")
                other_coords[k] = np.array([(lims[0] + lims[1]) / 2])

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
            old_dim_order = ("eV",) + old_dim_order
            new_dim_order = ("eV",) + new_dim_order
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
            print(f"Converted in {time.perf_counter() - t_start:.3f} seconds")

        return out
