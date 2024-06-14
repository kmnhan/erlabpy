__all__ = [
    "MomentumAccessor",
    "OffsetView",
]

import functools
import time
import warnings
from collections.abc import Hashable, ItemsView, Iterable, Iterator, Mapping
from typing import Literal, cast

import numpy as np
import xarray as xr

from erlab.accessors.utils import ERLabDataArrayAccessor
from erlab.analysis.interpolate import interpn
from erlab.analysis.kspace import AxesConfiguration, get_kconv_func, kz_func
from erlab.constants import rel_kconv, rel_kzconv


def only_angles(method=None):
    """Decorate methods that require data to be in angle space.

    Ensures the data is in angle space before executing the decorated method.

    If the data is not in angle space (i.e., if "kx" or "ky" dimensions are present), a
    `ValueError` is raised.
    """

    def wrapper(method):
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


def only_momentum(method=None):
    """Decorate methods that require data to be in momentum space.

    Ensure the data is in momentum space before executing the decorated method.

    If the data is not in momentum space (i.e., if "kx" nor "ky" dimensions are
    present), a `ValueError` is raised.
    """

    def wrapper(method):
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

    def __iter__(self) -> Iterator[tuple[str, float]]:
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
        if isinstance(other, Mapping):
            return dict(self) == dict(other)
        else:
            return False

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
        other: dict[str, float] | Iterable[tuple[str, float]] | None = None,
        **kwargs,
    ) -> "OffsetView":
        """Update the offset view with the provided key-value pairs."""
        if other is not None:
            for k, v in other.items() if isinstance(other, dict) else other:
                self[str(k)] = v
        for k, v in kwargs.items():
            self[k] = v
        return self

    def items(self) -> ItemsView[str, float]:
        """Return a view of the offset view as a list of (key, value) pairs."""
        return dict(self).items()

    def reset(self) -> "OffsetView":
        """Reset all angle offsets to zero."""
        for k in self._obj.kspace.valid_offset_keys:
            self[k] = 0.0
        return self


@xr.register_dataarray_accessor("kspace")
class MomentumAccessor(ERLabDataArrayAccessor):
    """`xarray.DataArray.kspace` accessor for momentum conversion related utilities.

    This class provides convenient access to various momentum-related properties of a
    data object. It allows getting and setting properties such as configuration, inner
    potential, work function, angle resolution, slit axis, momentum axes, angle
    parameters, and offsets.

    """

    @property
    def configuration(self) -> AxesConfiguration:
        """Return the experimental configuration.

        For a properly implemented data loader, the configuration attribute must be set
        on data import. See :class:`erlab.analysis.kspace.AxesConfiguration` for
        details.
        """
        if "configuration" not in self._obj.attrs:
            raise ValueError(
                "Configuration not found in data attributes! "
                "Data attributes may have been discarded since initial import."
            )
        return AxesConfiguration(int(self._obj.attrs.get("configuration", 0)))

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
        data. If not found, a warning is issued and a default value of 4.5 eV is
        assumed.

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
        """Retrieve the angular resolution of the data in degrees.

        Checks for the ``angle_resolution`` attribute of the data. If not found, a
        default value of 0.1° is silently assumed.

        This property is used in `best_kp_resolution` upon estimating momentum step
        sizes through `estimate_resolution`.
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
    def slit_axis(self) -> Literal["kx", "ky"]:
        """Return the momentum axis parallel to the slit.

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
    def other_axis(self) -> Literal["kx", "ky"]:
        """Return the momentum axis perpendicular to the slit.

        Returns
        -------
        str
            Returns ``'ky'`` for type 1 configurations, ``'kx'`` otherwise.
        """
        match self.configuration:
            case AxesConfiguration.Type1 | AxesConfiguration.Type1DA:
                return "ky"
            case _:
                return "kx"

    @property
    @only_angles
    def momentum_axes(self) -> tuple[Literal["kx", "ky", "kz"], ...]:
        """Return the momentum axes of the data after conversion.

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
    @only_angles
    def alpha(self) -> xr.DataArray:
        return self._obj.alpha

    @property
    @only_angles
    def beta(self) -> xr.DataArray:
        return self._obj.beta

    @property
    def hv(self) -> xr.DataArray:
        return self._obj.hv

    @property
    def binding_energy(self) -> xr.DataArray:
        if self._obj.eV.values.min() > 0:
            # eV values are kinetic, transform to binding energy
            binding = self._obj.eV - self.hv + self.work_function
            return binding.assign_coords(eV=binding.values)
        else:
            return self._obj.eV

    @property
    def kinetic_energy(self) -> xr.DataArray:
        return self.hv - self.work_function + self.binding_energy

    @property
    def has_eV(self) -> bool:
        """Return `True` if object has an energy axis."""
        return "eV" in self._obj.dims

    @property
    @only_angles
    def has_hv(self) -> bool:
        """Return `True` for photon energy dependent data."""
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
        return self.beta.size > 1

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

    @property
    @only_angles
    def best_kp_resolution(self) -> float:
        r"""Estimate the minimum in-plane momentum resolution.

        The resolution is estimated with the kinetic energy and angular resolution:

        .. math:: \Delta k_{\parallel} \sim \sqrt{2 m_e E_k/\hbar^2} \cos(\alpha) \Delta\alpha

        """
        min_Ek = np.amin(self.kinetic_energy.values)
        max_angle = max(np.abs(self.alpha.values))
        return (
            rel_kconv
            * np.sqrt(min_Ek)
            * np.cos(np.deg2rad(max_angle))
            * np.deg2rad(self.angle_resolution)
        )

    @property
    @only_angles
    def best_kz_resolution(self) -> float:
        r"""Estimate the minimum out-of-plane momentum resolution.

        The resolution is estimated based on the mean free path :cite:p:`seah1979imfp`
        and the kinetic energy.

        .. math:: \Delta k_z \sim 1/\lambda

        """
        kin = self.kinetic_energy.values
        c1, c2 = 641.0, 0.096
        imfp = (c1 / (kin**2) + c2 * np.sqrt(kin)) * 10
        return float(np.amin(1 / imfp))

    def _get_transformed_coords(self) -> dict[Literal["kx", "ky", "kz"], xr.DataArray]:
        kx, ky = self._forward_func(self.alpha, self.beta)
        if "hv" in kx.dims:
            kz = kz_func(self.kinetic_energy, self.inner_potential, kx, ky)
            return {"kx": kx, "ky": ky, "kz": kz}
        else:
            return {"kx": kx, "ky": ky}

    def estimate_bounds(self) -> dict[Literal["kx", "ky", "kz"], tuple[float, float]]:
        """
        Estimate the bounds of the data in momentum space.

        Returns
        -------
        bounds : dict[str, tuple[float, float]]
            A dictionary containing the estimated bounds for each parameter. The keys of
            the dictionary are 'kx', 'ky', and 'kz' (for :math:`hν`-dependent data). The
            values are tuples representing the minimum and maximum values.

        """
        return {
            k: (v.values.min(), v.values.max())
            for k, v in self._get_transformed_coords().items()
        }

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
        elif axis == self.other_axis:
            dim = "beta"
        elif axis == "kz":
            dim = "hv"
            if not self.has_hv:
                raise ValueError("No photon energy axis found.")
        else:
            raise ValueError(f"`{axis}` is not a valid momentum axis.")

        if from_numpoints and (lims is not None):
            return float((lims[1] - lims[0]) / len(self._obj[dim]))
        elif axis == "kz":
            return self.best_kz_resolution
        else:
            return self.best_kp_resolution

    def _forward_func(self, alpha, beta):
        return get_kconv_func(
            self.kinetic_energy, self.configuration, self.angle_params
        )[0](alpha, beta)

    def _inverse_func(self, kx, ky, kz=None):
        return get_kconv_func(
            self.kinetic_energy, self.configuration, self.angle_params
        )[1](kx, ky, kz)

    def _inverse_broadcast(self, kx, ky, kz=None) -> dict[str, xr.DataArray]:
        kxval = xr.DataArray(kx, dims="kx", coords={"kx": kx})
        kyval = xr.DataArray(ky, dims="ky", coords={"ky": ky})
        if kz is not None:
            kzval = xr.DataArray(kz, dims="kz", coords={"kz": kz})
        else:
            kzval = None

        alpha, beta = self._inverse_func(kxval, kyval, kzval)

        out_dict = {"alpha": alpha, "beta": beta}

        if self.has_eV:
            out_dict["eV"] = self.binding_energy

        if kzval is not None:
            out_dict["hv"] = (
                rel_kzconv * (kxval**2 + kyval**2 + kzval**2)
                - self.inner_potential
                + self.work_function
                - self.binding_energy
            )

        return cast(
            dict[str, xr.DataArray],
            dict(
                zip(
                    cast(list[str], out_dict.keys()),
                    xr.broadcast(*out_dict.values()),
                    strict=True,
                )
            ),
        )

    @only_angles
    def convert_coords(self) -> xr.DataArray:
        """Convert the coordinates to momentum space.

        Assigns new exact momentum coordinates to the data. This is useful when you want
        to work with momentum coordinates but don't want to interpolate the data.

        Returns
        -------
        xarray.DataArray
            The DataArray with transformed coordinates.
        """
        return self._obj.assign_coords(self._get_transformed_coords())

    @only_angles
    def _get_coord_for_conversion(self, name: Hashable) -> xr.DataArray:
        """Get the coordinte array for given dimension name.

        This just ensures that the energy coordinates are given as binding energy.
        """
        if name == "eV":
            return self.binding_energy
        else:
            return self._obj[name]

    @only_angles
    def _data_ensure_binding(self) -> xr.DataArray:
        """Return the data while ensuring that the energy axis is in binding energy."""
        return self._obj.assign_coords(eV=self.binding_energy)

    @only_angles
    def convert(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        resolution: dict[str, float] | None = None,
        *,
        silent: bool = False,
        **coords,
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
        **coords
            Array-like keyword arguments that specifies the coordinate array for each
            momentum axis. If provided, the bounds and resolution will be ignored.

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

        momentum_coords: dict[str, np.ndarray] = {}

        for k, lims in self.estimate_bounds().items():
            if k in self.momentum_axes:
                lims = bounds.get(k, lims)

                res = self.estimate_resolution(k, lims, from_numpoints=False)
                if k == "kz":
                    res_n = self.estimate_resolution(k, lims, from_numpoints=True)
                    res = min(res, res_n)
                res = resolution.get(k, res)

                momentum_coords[k] = np.linspace(
                    *lims, round((lims[1] - lims[0]) / res + 1)
                )
            else:
                # Take the mean for axes that will not appear in converted data
                if not silent and lims[1] - lims[0] > 0.001:
                    print(f"Data spans {lims[1] - lims[0]:.3f} Å⁻¹ of {k}")
                momentum_coords[k] = np.array([(lims[0] + lims[1]) / 2])

        for k, v in coords.items():
            if k in self.momentum_axes:
                momentum_coords[k] = v
            else:
                raise ValueError(f"Dimension `{k}` is not a momentum axis")

        if not silent:
            print("Calculating destination coordinates")

        target_dict: dict[str, xr.DataArray] = self._inverse_broadcast(
            momentum_coords.get("kx"),
            momentum_coords.get("ky"),
            momentum_coords.get("kz", None),
        )

        # Coords of first value in target_dict. Output of inverse_broadcast are all
        # broadcasted to each other, so all values will have same coords
        coords_for_transform = next(iter(target_dict.values())).coords

        # Remove non-dimension coordinates
        for k in list(coords_for_transform.keys()):
            if k not in coords_for_transform.dims:
                del coords_for_transform[k]

        # Now, coords_for_transform contains a subset of kx, ky, kz, eV, and hv coords,
        # depending on the dimensions of the input data.

        dim_mapping: dict[str, str] = {}
        for d in coords_for_transform.dims:
            if d == self.slit_axis:
                dim_mapping["alpha"] = str(d)
            elif d == self.other_axis:
                dim_mapping["beta"] = str(d)
            elif d == "kz":
                dim_mapping["hv"] = str(d)
            else:
                dim_mapping[str(d)] = str(d)

        # Delete keys not in the input data, e.g. "beta" for cuts
        for k in list(dim_mapping.keys()):
            if k not in self._obj.dims:
                del dim_mapping[k]

        input_dims: tuple[str, ...] = tuple(dim_mapping.keys())
        output_dims: tuple[str, ...] = tuple(dim_mapping.values())

        if not silent:
            print(f"Converting {input_dims}  ->  {output_dims}")
            t_start = time.perf_counter()

        def _wrap_interpn(arr, *args):
            points, xi = args[: arr.ndim], args[arr.ndim :]
            return interpn(points, arr, xi, bounds_error=False).squeeze()

        input_core_dims = [input_dims]
        input_core_dims.extend([(d,) for d in input_dims])
        input_core_dims.extend(
            [cast(tuple[str, ...], target_dict[d].dims) for d in input_dims]
        )

        out = xr.apply_ufunc(
            _wrap_interpn,
            self._data_ensure_binding(),
            *tuple(self._get_coord_for_conversion(dim) for dim in input_dims),
            *tuple(target_dict[dim] for dim in input_dims),
            vectorize=True,
            dask="parallelized",
            input_core_dims=input_core_dims,
            output_core_dims=[output_dims],
            dask_gufunc_kwargs={
                "allow_rechunk": True,
                "output_sizes": {d: coords_for_transform[d].size for d in output_dims},
            },
            keep_attrs=True,
            output_dtypes=[np.float64],
        ).assign_coords({k: v.squeeze() for k, v in coords_for_transform.items()})

        if not silent:
            print(f"Interpolated in {time.perf_counter() - t_start:.3f} s")

        return out

    def interactive(self, **kwargs):
        """Open the interactive momentum space conversion tool."""
        from erlab.interactive.kspace import ktool

        if self._obj.ndim < 3:
            raise ValueError("Interactive tool requires three-dimensional data.")
        return ktool(self._obj, **kwargs)

    @only_momentum
    def hv_to_kz(self, hv: float | Iterable[float]) -> xr.DataArray:
        """Return :math:`k_z` for a given photon energy.

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
        if isinstance(hv, Iterable) and not isinstance(hv, xr.DataArray):
            hv = xr.DataArray(np.asarray(hv), coords={"hv": hv})

        # Get kinetic energies for the given photon energy
        kinetic = hv - self.work_function + self._obj.eV

        # Get momentum conversion functions
        ang2k, k2ang = get_kconv_func(kinetic, self.configuration, self.angle_params)

        # Transformation yields in-plane momentum at given photon energy
        kx, ky = ang2k(*k2ang(self._obj.kx, self._obj.ky))

        # Calculate kz
        kz = kz_func(kinetic, self.inner_potential, kx, ky)

        return kz
