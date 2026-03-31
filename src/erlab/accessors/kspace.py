"""Defines an accessor for momentum conversion related utilities."""

__all__ = ["IncompleteDataError", "MomentumAccessor", "OffsetView"]

import functools
import time
import typing
from collections.abc import Hashable, ItemsView, Iterable, Iterator, Mapping

import numpy as np
import xarray as xr

import erlab
from erlab.accessors.utils import ERLabDataArrayAccessor
from erlab.constants import AxesConfiguration


class IncompleteDataError(ValueError):
    """Raised when the data is not in the expected format for momentum conversion.

    See :ref:`conventions <data-conventions>` for required data attributes and
    coordinates.
    """

    def __init__(self, kind: typing.Literal["attr", "coord"], name: str) -> None:
        super().__init__(self._make_message(kind, name))

    @staticmethod
    def _make_message(kind: typing.Literal["attr", "coord"], name: str) -> str:
        kind_str = "Attribute" if kind == "attr" else "Coordinate"
        return f"{kind_str} '{name}' is required for momentum conversion."


def _kxy_components(
    slit_axis: typing.Literal["kx", "ky"],
    slit_momentum,
    other_momentum,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    return (
        (slit_momentum, other_momentum)
        if slit_axis == "kx"
        else (other_momentum, slit_momentum)
    )


def _hv_to_kz_root_candidates_1d(
    kz_grid: np.ndarray,
    other_momentum: np.ndarray,
    kinetic_energy,
    slit_momentum,
    *,
    inner_potential: float,
    slit_axis: typing.Literal["kx", "ky"],
) -> np.ndarray:
    kz_grid = np.asarray(kz_grid, dtype=float)
    other_momentum = np.asarray(other_momentum, dtype=float)
    kinetic = float(kinetic_energy)
    slit = float(slit_momentum)

    if not np.isfinite(kinetic) or not np.isfinite(slit):
        return np.array([], dtype=float)

    mask = np.isfinite(kz_grid) & np.isfinite(other_momentum)
    if not np.any(mask):
        return np.array([], dtype=float)

    kz_valid = kz_grid[mask]
    other_valid = other_momentum[mask]
    order = np.argsort(kz_valid)
    kz_valid = kz_valid[order]
    other_valid = other_valid[order]

    kx, ky = _kxy_components(slit_axis, slit, other_valid)
    residual = (
        erlab.analysis.kspace.kz_func(kinetic, inner_potential, kx, ky) - kz_valid
    )

    exact = np.isclose(residual, 0.0, atol=1e-12, rtol=0.0)
    candidates: list[float] = []

    exact_idx = np.flatnonzero(exact)
    if exact_idx.size:
        split_idx = np.flatnonzero(np.diff(exact_idx) != 1) + 1
        candidates.extend(
            float(np.mean(kz_valid[group])) for group in np.split(exact_idx, split_idx)
        )

    brackets = np.flatnonzero(
        ((residual[:-1] < 0.0) & (residual[1:] > 0.0))
        | ((residual[:-1] > 0.0) & (residual[1:] < 0.0))
    )
    for i0 in brackets:
        kz0, kz1 = kz_valid[int(i0)], kz_valid[int(i0) + 1]
        f0, f1 = residual[int(i0)], residual[int(i0) + 1]
        candidates.append(float(kz0 - f0 * (kz1 - kz0) / (f1 - f0)))
    return np.asarray(candidates, dtype=float)


def _solve_hv_to_kz_roots_2d(
    kz_grid: np.ndarray,
    other_momentum: np.ndarray,
    kinetic_energy,
    slit_momentum: np.ndarray,
    *,
    inner_potential: float,
    slit_axis: typing.Literal["kx", "ky"],
) -> np.ndarray:
    other_momentum = np.asarray(other_momentum, dtype=float)
    slit_momentum = np.asarray(slit_momentum, dtype=float)

    roots = np.full(slit_momentum.shape, np.nan, dtype=float)
    candidates = [
        _hv_to_kz_root_candidates_1d(
            kz_grid,
            other_momentum[i],
            kinetic_energy,
            slit_momentum[i],
            inner_potential=inner_potential,
            slit_axis=slit_axis,
        )
        for i in range(slit_momentum.size)
    ]

    unique_indices = [i for i, values in enumerate(candidates) if values.size == 1]
    if not unique_indices:
        return roots

    center = (slit_momentum.size - 1) / 2.0
    seed = min(unique_indices, key=lambda i: abs(i - center))
    roots[seed] = candidates[seed][0]

    def _propagate(indices: range) -> None:
        ref = roots[seed]
        for i in indices:
            values = candidates[i]
            if values.size == 0:
                continue
            if values.size == 1 or not np.isfinite(ref):
                roots[i] = values[0]
            else:
                roots[i] = values[np.argmin(np.abs(values - ref))]
            ref = roots[i]

    _propagate(range(seed + 1, slit_momentum.size))
    _propagate(range(seed - 1, -1, -1))

    return roots


def _only_angles(method):
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

    return wrapper(method)


def _only_momentum(method):
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

    return wrapper(method)


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

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def __iter__(self) -> Iterator[tuple[str, float]]:
        for key in self._obj.kspace._valid_offset_keys:
            yield key, self.__getitem__(key)

    def __getitem__(self, key: str) -> float:
        if key in self._obj.kspace._valid_offset_keys:
            offset_key: str = key + "_offset"
            if offset_key in self._obj.attrs:
                return float(self._obj.attrs[offset_key])

            # If the offset key is not found, return the default value (0.0) This is to
            # ensure that the offset view always has a value for the key even if it
            # hasn't been set yet.
            return 0.0

        raise KeyError(
            f"Invalid offset key `{key}` for experimental configuration "
            f"{self._obj.kspace.configuration}"
        )

    def __setitem__(self, key: str, value: float) -> None:
        if key in self._obj.kspace._valid_offset_keys:
            self._obj.attrs[key + "_offset"] = float(value)
        else:
            raise KeyError(
                f"Invalid offset key '{key}' for experimental configuration "
                f"{self._obj.kspace.configuration}. Valid keys are: "
                f"{self._obj.kspace._valid_offset_keys}."
            )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self) == dict(other)
        return False

    def _normal_emission_angles(self) -> tuple[float, float] | None:
        try:
            alpha, beta = self._obj.kspace._normal_emission_angles()
        except (IncompleteDataError, TypeError, ValueError):
            return None
        if np.isclose(alpha, 0):
            alpha = 0.0
        if np.isclose(beta, 0):
            beta = 0.0
        return alpha, beta

    def __repr__(self) -> str:
        offsets_repr = dict(self).__repr__()
        normal = self._normal_emission_angles()
        if normal is None:
            return offsets_repr
        alpha, beta = normal
        return f"{offsets_repr}\nnormal emission: alpha={alpha}, beta={beta}"

    def _repr_html_(self) -> str:
        rows: list[tuple[str, str]] = [(k, str(v)) for k, v in self.items()]
        normal = self._normal_emission_angles()
        if normal is not None:
            rows.extend(
                [
                    ("normal alpha", str(normal[0])),
                    ("normal beta", str(normal[1])),
                ]
            )
        return erlab.utils.formatting.format_html_table(rows, header_cols=1)

    def update(
        self,
        other: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
        **kwargs,
    ) -> typing.Self:
        """Update the offset view with the provided key-value pairs."""
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[str(k)] = v
        for k, v in kwargs.items():
            self[k] = v
        return self

    def items(self) -> ItemsView[str, float]:
        """Return a view of the offset view as a list of (key, value) pairs."""
        return dict(self).items()

    def reset(self) -> typing.Self:
        """Reset all angle offsets."""
        for k in self._obj.kspace._valid_offset_keys:
            offset_key: str = k + "_offset"
            if offset_key in self._obj.attrs:
                del self._obj.attrs[offset_key]
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
        """Experimental configuration.

        For data loaded with a properly implemented data loader plugin, the
        configuration attribute is automatically set upon loading. If the configuration
        is missing, the attributes of the data may have been lost since loading due to
        averaging or other operations. In such cases, try to reload the data after
        setting ``xr.set_options(keep_attrs='True')``.

        See :class:`erlab.constants.AxesConfiguration` for possible configurations.

        Data from some ARPES setups may have a dynamic configuration that changes per
        data. In such cases, the configuration should be converted with
        :meth:`xarray.DataArray.kspace.as_configuration`.
        """
        if "configuration" not in self._obj.attrs:
            raise IncompleteDataError("attr", "configuration")

        return AxesConfiguration(int(self._obj.attrs["configuration"]))

    @configuration.setter
    def configuration(self, value: AxesConfiguration | int) -> None:
        if "configuration" in self._obj.attrs:
            raise AttributeError(
                "Configuration is already set. To modify the experimental "
                "configuration, use `DataArray.kspace.as_configuration`."
            )
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
        erlab.utils.misc.emit_user_level_warning(
            "Inner potential not found in data attributes, assuming 10 eV"
        )
        return 10.0

    @inner_potential.setter
    def inner_potential(self, value: float) -> None:
        self._obj.attrs["inner_potential"] = float(value)

    @property
    def work_function(self) -> float:
        """Work function of the system in eV.

        Here, the work function here refers to the work function of the entire system in
        electrical contact with the sample, which determines the Fermi level.

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
        erlab.utils.misc.emit_user_level_warning(
            "Work function not found in data attributes, assuming 4.5 eV"
        )
        return 4.5

    @work_function.setter
    def work_function(self, value: float) -> None:
        self._obj.attrs["sample_workfunction"] = float(value)

    @property
    def angle_resolution(self) -> float:
        """Retrieve the angular resolution of the data in degrees.

        Checks for the ``angle_resolution`` attribute of the data. If not found, a
        default value of 0.1° is silently assumed.

        This property is used in `best_kp_resolution` upon estimating momentum step
        sizes through `estimate_resolution`.

        Note
        ----
        This property provides a setter method that takes a float value and sets the
        data attribute accordingly.

        Example
        -------
        >>> data.kspace.angle_resolution = 0.05
        >>> data.kspace.angle_resolution
        0.05

        """
        try:
            return float(self._obj.attrs["angle_resolution"])
        except KeyError:
            # erlab.utils.misc.emit_user_level_warning(
            #     "Angle resolution not found in data attributes, assuming 0.1 degrees"
            # )
            return 0.1

    @angle_resolution.setter
    def angle_resolution(self, value: float) -> None:
        self._obj.attrs["angle_resolution"] = float(value)

    @property
    def slit_axis(self) -> typing.Literal["kx", "ky"]:
        """Momentum axis parallel to the analyzer slit.

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
    def other_axis(self) -> typing.Literal["kx", "ky"]:
        """Momentum axis perpendicular to the analyzer slit.

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
    @_only_angles
    def momentum_axes(self) -> tuple[typing.Literal["kx", "ky", "kz"], ...]:
        """Momentum axes of the data after conversion.

        Returns
        -------
        tuple
            For photon energy dependent scans, it returns the slit axis and ``'kz'``.
            For maps, it returns ``'kx'`` and ``'ky'``. Otherwise, it returns only the
            slit axis.

        """
        if self._has_hv:
            return (self.slit_axis, "kz")
        if self._has_beta:
            return ("kx", "ky")
        return (self.slit_axis,)

    @property
    def angle_params(self) -> dict[str, float]:
        """Parameters passed to :func:`erlab.analysis.kspace.get_kconv_func`."""
        if "xi" not in self._obj.coords:
            raise IncompleteDataError("coord", "xi")

        params = {
            "delta": self.offsets["delta"],
            "xi": float(self._obj["xi"].values),
            "xi0": self.offsets["xi"],
        }
        match self.configuration:
            case AxesConfiguration.Type1 | AxesConfiguration.Type2:
                params["beta0"] = self.offsets["beta"]
            case _:
                if "chi" not in self._obj.coords:
                    raise IncompleteDataError("coord", "chi")
                params["chi"] = float(self._obj["chi"].values)
                params["chi0"] = self.offsets["chi"]

        return params

    @property
    @_only_angles
    def _alpha(self) -> xr.DataArray:
        if "alpha" not in self._obj.coords:
            raise IncompleteDataError("coord", "alpha")
        return self._obj.alpha

    @property
    @_only_angles
    def _beta(self) -> xr.DataArray:
        if "beta" not in self._obj.coords:
            raise IncompleteDataError("coord", "beta")
        return self._obj.beta

    @property
    def _hv(self) -> xr.DataArray:
        if "hv" not in self._obj.coords:
            raise IncompleteDataError("coord", "hv")
        return self._obj.hv

    @property
    def _is_energy_kinetic(self) -> bool:
        """Check if the energy axis is in binding energy."""
        # If scalar, may be a constant energy contour above EF
        return self._obj.eV.values.size > 1 and (self._obj.eV.values.min() > 0)

    @property
    def _binding_energy(self) -> xr.DataArray:
        if "eV" not in self._obj.coords:
            raise IncompleteDataError("coord", "eV")
        if self._is_energy_kinetic:
            if self._has_hv:
                raise ValueError(
                    "Energy axis of photon energy dependent data must be in "
                    "binding energy."
                )
            # eV values are kinetic, transform to binding energy
            binding = self._obj.eV - self._hv + self.work_function
            erlab.utils.misc.emit_user_level_warning(
                "The energy axis seems to be in terms of kinetic energy, "
                "attempting conversion to binding energy."
            )
            return binding.assign_coords(eV=binding.values)
        return self._obj.eV

    @property
    def _kinetic_energy(self) -> xr.DataArray:
        return self._hv - self.work_function + self._binding_energy

    @staticmethod
    def _finite_minmax(values: np.ndarray) -> tuple[float, float]:
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return float("nan"), float("nan")
        return (
            float(np.min(values, where=finite_mask, initial=np.inf)),
            float(np.max(values, where=finite_mask, initial=-np.inf)),
        )

    @staticmethod
    def _require_finite_scalar(value: float, name: str) -> float:
        scalar = np.asarray(value, dtype=float)
        if scalar.ndim != 0 or not np.isfinite(scalar):
            raise ValueError(f"`{name}` must be a finite scalar.")
        return float(scalar)

    def _check_kinetic_energy(
        self,
        *,
        context: str,
        kinetic_energy: xr.DataArray | np.ndarray | float | None = None,
        raise_on_violation: bool = True,
    ) -> xr.DataArray | np.ndarray | float:
        if kinetic_energy is None:
            kinetic_energy = self._kinetic_energy

        kinetic = np.asarray(kinetic_energy, dtype=float)
        finite_mask = np.isfinite(kinetic)

        if not np.any(finite_mask):
            msg = (
                "Cannot proceed while "
                f"{context}: kinetic energy contains no finite values."
            )
            if raise_on_violation:
                raise ValueError(msg)
            erlab.utils.misc.emit_user_level_warning(msg)
            return kinetic_energy

        min_kinetic = float(np.min(kinetic, where=finite_mask, initial=np.inf))
        if min_kinetic > 0:
            return kinetic_energy

        hv_min, hv_max = self._finite_minmax(np.asarray(self._hv.values, dtype=float))
        e_min, e_max = self._finite_minmax(
            np.asarray(self._binding_energy.values, dtype=float)
        )

        msg = (
            f"Nonphysical kinetic energy detected while {context}: "
            f"min(E_k)={min_kinetic:.3f} eV <= 0. "
            f"E_k = hv - sample_workfunction + eV (binding). "
            f"Current ranges: hv=[{hv_min:.3f}, {hv_max:.3f}] eV, "
            f"eV=[{e_min:.3f}, {e_max:.3f}] eV, "
            f"sample_workfunction={self.work_function:.3f} eV."
        )
        if raise_on_violation:
            raise ValueError(msg)
        erlab.utils.misc.emit_user_level_warning(msg)
        return kinetic_energy

    @property
    def _has_eV(self) -> bool:
        """Return `True` if object has an energy axis."""
        return "eV" in self._obj.dims

    @property
    @_only_angles
    def _has_hv(self) -> bool:
        """Return `True` for photon energy dependent data."""
        return self._hv.size > 1

    @property
    @_only_angles
    def _has_beta(self) -> bool:
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
        return self._beta.size > 1

    @property
    def _valid_offset_keys(self) -> tuple[str, str, str]:
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

        - Offsets to dictionary

          >>> dict(data.kspace.offsets)
          {'delta': 0.0, 'xi': 0.0, 'beta': 0.0}

        - View single offset

          >>> data.kspace.offsets["beta"]
          0.0

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

        See Also
        --------
        :meth:`set_normal <xarray.DataArray.kspace.set_normal>`
            Method to set angle offsets from normal emission angles.
        """
        if not hasattr(self, "_offsetview"):
            self._offsetview = OffsetView(self._obj)

        return self._offsetview

    @offsets.setter
    def offsets(self, offset_dict: Mapping[str, float]) -> None:
        if not hasattr(self, "_offsetview"):
            self._offsetview = OffsetView(self._obj)

        self._offsetview.reset()
        self._offsetview.update(offset_dict)

    @_only_angles
    def set_normal(
        self, alpha: float, beta: float, *, delta: float | None = None
    ) -> None:
        r"""Set offsets from normal emission angles.

        This method sets the angle offsets so that the provided normal emission angles
        :math:`(\alpha, \beta)` in the data map to :math:`(k_x, k_y) = (0, 0)` in
        momentum space.

        Parameters
        ----------
        alpha
            Angle :math:`\alpha` in degrees corresponding to sample normal emission.
        beta
            Angle :math:`\beta` in degrees corresponding to sample normal emission.
        delta
            Optional azimuthal offset :math:`\delta` in degrees. If omitted, the
            existing ``delta`` offset is preserved.

        Examples
        --------
        >>> data.kspace.set_normal(alpha=1.2, beta=-0.4)
        >>> dict(data.kspace.offsets)
        {'delta': 0.0, 'xi': -1.2, 'beta': -0.4}

        See Also
        --------
        :attr:`offsets <xarray.DataArray.kspace.offsets>`
            Attribute used to manipulate angle offsets directly.
        """
        alpha_normal = self._require_finite_scalar(alpha, "alpha")
        beta_normal = self._require_finite_scalar(beta, "beta")

        if delta is not None:
            self.offsets["delta"] = self._require_finite_scalar(delta, "delta")

        if "xi" not in self._obj.coords:
            raise IncompleteDataError("coord", "xi")
        xi = float(self._obj["xi"].values)

        chi: float | None = None
        if self.configuration in (
            AxesConfiguration.Type1DA,
            AxesConfiguration.Type2DA,
        ):
            if "chi" not in self._obj.coords:
                raise IncompleteDataError("coord", "chi")
            chi = float(self._obj["chi"].values)

        self.offsets.update(
            erlab.analysis.kspace._offsets_from_normal_emission(
                self.configuration,
                alpha_normal,
                beta_normal,
                xi=xi,
                chi=chi,
            )
        )

    @_only_angles
    def set_normal_like(self, other: xr.DataArray) -> None:
        r"""Set offsets like another DataArray.

        This method reads the normal emission angles implied by another DataArray's
        current offsets and applies the same normal emission angles to the current data.

        The azimuthal offset :math:`\delta` is copied as well.

        Parameters
        ----------
        other
            Another DataArray in angle space whose current offsets define the reference
            normal emission position.

        See Also
        --------
        :meth:`set_normal <xarray.DataArray.kspace.set_normal>`
            Method used to set angle offsets from explicitly provided normal emission
            angles.
        """
        if not isinstance(other, xr.DataArray):
            raise TypeError("`other` must be an xarray.DataArray.")

        self.set_normal(
            *other.kspace._normal_emission_angles(), delta=other.kspace.offsets["delta"]
        )

    @_only_angles
    def _normal_emission_angles(self) -> tuple[float, float]:
        """Calculate the normal emission angles based on the current offsets."""
        return erlab.analysis.kspace._normal_emission_from_angle_params(
            self.configuration, self.angle_params
        )

    @property
    @_only_angles
    def best_kp_resolution(self) -> float:
        r"""Estimated minimum in-plane momentum resolution.

        The resolution is estimated with the kinetic energy and angular resolution:

        .. math::

            \Delta k_{\parallel} \sim \sqrt{2 m_e E_k/\hbar^2} \cos(\alpha) \Delta\alpha

        """
        self._check_kinetic_energy(context="estimating in-plane momentum resolution")
        min_Ek = np.amin(self._kinetic_energy.values)
        max_angle = max(np.abs(self._alpha.values))
        return float(
            erlab.constants.rel_kconv
            * np.sqrt(min_Ek)
            * np.cos(np.deg2rad(max_angle))
            * np.deg2rad(self.angle_resolution)
        )

    @property
    @_only_angles
    def best_kz_resolution(self) -> float:
        r"""Estimated minimum out-of-plane momentum resolution.

        The resolution is estimated based on the mean free path :cite:p:`seah1979imfp`
        and the kinetic energy. Note that this is a rough estimate based on the
        universal curve of mean free path.

        .. math:: \Delta k_z \sim 1/\lambda

        """
        self._check_kinetic_energy(
            context="estimating out-of-plane momentum resolution"
        )
        kin = self._kinetic_energy.values

        c1, c2 = 641.0, 0.096
        imfp = (c1 / (kin**2) + c2 * np.sqrt(kin)) * 10
        return float(np.amin(1 / imfp))

    @property
    def _interactive_compatible(self) -> bool:
        """Check if the data is compatible with the interactive tool."""
        if self._obj.ndim == 2:
            # alpha-beta 2D scan
            return set(self._obj.dims) == {"alpha", "beta"}

        if self._obj.ndim == 3:
            # any 3D scan that has alpha & eV
            return "alpha" in self._obj.dims and "eV" in self._obj.dims

        return False

    def _get_transformed_coords(
        self,
    ) -> dict[typing.Literal["kx", "ky", "kz"], xr.DataArray]:
        kx, ky = self._forward_func(self._alpha, self._beta)
        if "hv" in kx.dims:
            kz = erlab.analysis.kspace.kz_func(
                self._kinetic_energy, self.inner_potential, kx, ky
            )
            return {"kx": kx, "ky": ky, "kz": kz}
        return {"kx": kx, "ky": ky}

    def estimate_bounds(
        self,
    ) -> dict[typing.Literal["kx", "ky", "kz"], tuple[float, float]]:
        """Estimate the bounds of the data in momentum space.

        Returns
        -------
        bounds : dict of str to tuple of float
            A dictionary containing the estimated bounds for each parameter. The keys of
            the dictionary are 'kx', 'ky', and 'kz' (for :math:`hν`-dependent data). The
            values are tuples representing the minimum and maximum values.

        """
        self._check_kinetic_energy(context="estimating momentum bounds")
        return {
            k: (v.values.min(), v.values.max())
            for k, v in self._get_transformed_coords().items()
        }

    @_only_angles
    def estimate_resolution(
        self,
        axis: typing.Literal["kx", "ky", "kz"],
        lims: tuple[float, float] | None = None,
        from_numpoints: bool = False,
    ) -> float:
        """Estimate resolution for a given momentum axis.

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

            .. versionchanged:: 3.20.1
                When ``from_numpoints=True``, the estimated step now uses adjacent-point
                spacing over inclusive bounds: ``(max - min) / (N - 1)``. Datasets with
                fewer than 2 points on the relevant axis return ``np.inf``.

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
            if not self._has_hv:
                raise ValueError("No photon energy axis found.")
        else:
            raise ValueError(f"`{axis}` is not a valid momentum axis.")

        if from_numpoints and (lims is not None):
            n_points = len(self._obj[dim])
            if n_points < 2:
                return float("inf")
            return float((lims[1] - lims[0]) / (n_points - 1))

        if axis == "kz":
            return self.best_kz_resolution

        return self.best_kp_resolution

    def _forward_func(self, alpha, beta):
        return erlab.analysis.kspace.get_kconv_forward(self.configuration)(
            alpha, beta, self._kinetic_energy, **self.angle_params
        )

    def _inverse_func(self, kx, ky, kperp=None):
        return erlab.analysis.kspace.get_kconv_inverse(self.configuration)(
            kx, ky, kperp, self._kinetic_energy, **self.angle_params
        )

    def _broadcast_exact_targets(
        self, out_dict: dict[str, xr.DataArray], other_coord: xr.DataArray
    ) -> dict[str, xr.DataArray]:
        keys = tuple(out_dict)
        broadcasted = xr.broadcast(*(out_dict[key] for key in keys))
        if not self._has_beta:
            broadcasted = tuple(
                value.squeeze("beta", drop=True) if "beta" in value.dims else value
                for value in broadcasted
            )
            if "beta" in other_coord.dims:
                other_coord = other_coord.squeeze("beta", drop=True)
        finite_other = np.asarray(other_coord.values, dtype=float)
        finite_other = finite_other[np.isfinite(finite_other)]
        if finite_other.size and np.allclose(finite_other, 0.0, atol=1e-10, rtol=0.0):
            other_coord = xr.DataArray(0.0)
        else:
            other_coord = other_coord.broadcast_like(broadcasted[0]).reset_coords(
                drop=True
            )
        return {
            key: value.assign_coords({self.other_axis: other_coord})
            for key, value in zip(keys, broadcasted, strict=True)
        }

    def _inverse_exact_cut(self, slit_momentum) -> dict[str, xr.DataArray]:
        slit_axis = self.slit_axis
        slit_value = xr.DataArray(
            slit_momentum, dims=slit_axis, coords={slit_axis: slit_momentum}
        )
        alpha = erlab.analysis.kspace.exact_cut_alpha(
            slit_value,
            self._beta,
            self._kinetic_energy,
            self._alpha,
            self.configuration,
            **self.angle_params,
        )
        other_momentum = erlab.analysis.kspace._exact_other_axis_momentum(
            alpha,
            self._beta,
            self._kinetic_energy,
            self.configuration,
            **self.angle_params,
        )

        out_dict: dict[str, xr.DataArray] = {"alpha": alpha}
        if self._has_eV:
            out_dict["eV"] = self._binding_energy

        return self._broadcast_exact_targets(out_dict, other_momentum)

    def _inverse_exact_hv_cut(self, slit_momentum, kz) -> dict[str, xr.DataArray]:
        slit_axis = self.slit_axis
        slit_value = xr.DataArray(
            slit_momentum, dims=slit_axis, coords={slit_axis: slit_momentum}
        )
        kz_value = xr.DataArray(kz, dims="kz", coords={"kz": kz})

        alpha, hv, other_momentum = erlab.analysis.kspace.exact_hv_cut_coords(
            slit_value,
            kz_value,
            self._beta,
            self._hv,
            self._kinetic_energy,
            self._alpha,
            self.configuration,
            self.inner_potential,
            **self.angle_params,
        )

        out_dict: dict[str, xr.DataArray] = {"alpha": alpha, "hv": hv}
        if self._has_eV:
            out_dict["eV"] = self._binding_energy

        return self._broadcast_exact_targets(out_dict, other_momentum)

    def _inverse_broadcast(self, kx, ky, kz=None) -> dict[str, xr.DataArray]:
        kxval = xr.DataArray(kx, dims="kx", coords={"kx": kx})
        kyval = xr.DataArray(ky, dims="ky", coords={"ky": ky})
        if kz is not None:
            kzval = xr.DataArray(kz, dims="kz", coords={"kz": kz})
            kperp = erlab.analysis.kspace.kperp_from_kz(kzval, self.inner_potential)
        else:
            kzval = None
            kperp = None

        alpha, beta = self._inverse_func(kxval, kyval, kperp)

        out_dict = {"alpha": alpha, "beta": beta}

        if self._has_eV:
            out_dict["eV"] = self._binding_energy

        if kzval is not None:
            out_dict["hv"] = erlab.analysis.kspace.hv_func(
                kxval,
                kyval,
                kzval,
                self.inner_potential,
                self.work_function,
                self._binding_energy,
            )

        return typing.cast(
            "dict[str, xr.DataArray]",
            dict(
                zip(
                    typing.cast("list[str]", out_dict.keys()),
                    xr.broadcast(*out_dict.values()),
                    strict=True,
                )
            ),
        )

    @_only_angles
    def convert_coords(self) -> xr.DataArray:
        """Convert coordinates to momentum space.

        Assigns new exact momentum coordinates to the data. This is useful when you want
        to work with momentum coordinates but don't want to interpolate the data.

        Returns
        -------
        xarray.DataArray
            The DataArray with transformed coordinates.
        """
        self._check_kinetic_energy(context="converting coordinates to momentum space")
        return self._obj.assign_coords(self._get_transformed_coords())

    @_only_angles
    def _coord_for_conversion(self, name: Hashable) -> xr.DataArray:
        """Get the coordinate array for given dimension name.

        This just ensures that the energy coordinates are given as binding energy.
        """
        if name == "eV":
            return self._binding_energy
        return self._obj[name]

    @_only_angles
    def _data_ensure_binding(self) -> xr.DataArray:
        """Return the data while ensuring that the energy axis is in binding energy."""
        return self._obj.assign_coords(eV=self._binding_energy)

    @_only_angles
    def convert(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        resolution: dict[str, float] | None = None,
        *,
        method: str = "linear",
        silent: bool = True,
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
        method
            The interpolation method to use, passed to
            :func:`erlab.analysis.interpolate.interpn`. Using methods other than
            ``'linear'`` will result in slower performance.
        silent
            If ``False``, print progress messages during the conversion.
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

        For non-`hv` scans, if the `eV` axis is all-positive, it is interpreted as
        kinetic energy and converted to binding energy. For `hv`-dependent scans, the
        `eV` axis must already be in binding energy.

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
        self._check_kinetic_energy(context="converting to momentum space")

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
                erlab.utils.misc.emit_user_level_warning(
                    f"Skipping unknown momentum axis '{k}', valid "
                    f"axes are {self.momentum_axes}"
                )

        if not silent:
            print("Calculating destination coordinates")

        target_dict: dict[str, xr.DataArray]
        if not (self._has_beta and not self._has_hv):
            if self._has_hv:
                target_dict = self._inverse_exact_hv_cut(
                    momentum_coords[self.slit_axis], momentum_coords["kz"]
                )
            else:
                target_dict = self._inverse_exact_cut(momentum_coords[self.slit_axis])
        else:
            target_dict = self._inverse_broadcast(
                momentum_coords.get("kx"),
                momentum_coords.get("ky"),
                momentum_coords.get("kz"),
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
            return erlab.analysis.interpolate.interpn(
                points, arr, xi, method=method, bounds_error=False
            ).squeeze()

        input_core_dims = [input_dims]
        input_core_dims.extend([(d,) for d in input_dims])
        input_core_dims.extend(
            [typing.cast("tuple[str, ...]", target_dict[d].dims) for d in input_dims]
        )

        out = xr.apply_ufunc(
            _wrap_interpn,
            self._data_ensure_binding(),
            *tuple(self._coord_for_conversion(dim) for dim in input_dims),
            *tuple(target_dict[dim] for dim in input_dims),
            vectorize=True,
            dask="parallelized",
            input_core_dims=input_core_dims,
            output_core_dims=[output_dims],
            dask_gufunc_kwargs={
                "output_sizes": {d: coords_for_transform[d].size for d in output_dims},
            },
            keep_attrs=True,
            output_dtypes=[np.float64],
        ).assign_coords({k: v.squeeze() for k, v in coords_for_transform.items()})
        if not self._has_beta and "beta" in out.dims:
            out = out.squeeze("beta", drop=True)
        if (
            not self._has_beta
            and "beta" in self._obj.coords
            and "beta" not in out.coords
        ):
            out = out.assign_coords(beta=self._beta.squeeze(drop=True))

        if not silent:
            print(f"Interpolated in {time.perf_counter() - t_start:.3f} s")

        return out

    def interactive(self, **kwargs):
        """Open the interactive momentum space conversion tool.

        The interactive tool currently supports the following kinds of data:

        - 2D data with `alpha` and `beta` dimensions (constant energy surfaces)

        - 3D data with dimensions including `alpha` and `eV` (including maps and
          hv-dependent cuts)

        """
        if not self._interactive_compatible:
            raise ValueError("Data is not compatible with the interactive tool.")
        return erlab.interactive.ktool(self._obj, **kwargs)

    @_only_angles
    def as_configuration(self, configuration: AxesConfiguration | int) -> xr.DataArray:
        """Return a new DataArray with modified experimental configuration.

        The coordinates of the new DataArray are renamed to match the given
        configuration. The original data is not modified.

        Parameters
        ----------
        configuration
            The new configuration to apply.

        Note
        ----
        This method assumes a conversion between 4 typical setups listed in the table in
        :ref:`Nomenclature <nomenclature>`. Any non-standard setups should be handled by
        the user.

        """
        return erlab.analysis.kspace.change_configuration(self._obj, configuration)

    def _hv_to_kz_legacy(self, kinetic: xr.DataArray) -> xr.DataArray:
        ang2k, k2ang = erlab.analysis.kspace.get_kconv_func(
            kinetic, self.configuration, self.angle_params
        )
        kx, ky = ang2k(*k2ang(self._obj.kx, self._obj.ky))
        return erlab.analysis.kspace.kz_func(kinetic, self.inner_potential, kx, ky)

    def _hv_to_kz_from_stored_coords(
        self, kinetic: xr.DataArray
    ) -> xr.DataArray | None:
        target_order = tuple(
            dim
            for dim in ("hv", "eV", self.slit_axis)
            if dim in kinetic.dims or dim == self.slit_axis
        )
        if self.other_axis not in self._obj.coords:
            return None

        slit_coord = self._obj[self.slit_axis]
        other_coord = self._obj[self.other_axis]

        if "kz" not in other_coord.dims:
            kx, ky = _kxy_components(self.slit_axis, slit_coord, other_coord)
            out = erlab.analysis.kspace.kz_func(kinetic, self.inner_potential, kx, ky)
            return out.transpose(*target_order)

        kz_coord = self._obj["kz"]
        kz_root = xr.apply_ufunc(
            functools.partial(
                _solve_hv_to_kz_roots_2d,
                inner_potential=self.inner_potential,
                slit_axis=self.slit_axis,
            ),
            kz_coord,
            other_coord,
            kinetic,
            slit_coord,
            input_core_dims=[("kz",), (self.slit_axis, "kz"), (), (self.slit_axis,)],
            output_core_dims=[(self.slit_axis,)],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {self.slit_axis: slit_coord.size}},
            output_dtypes=[np.float64],
        )
        return kz_root.transpose(*target_order)

    @_only_momentum
    def hv_to_kz(self, hv: float | Iterable[float]) -> xr.DataArray:
        r"""Return :math:`k_z` for a given photon energy.

        Useful when creating overlays on :math:`hν`-dependent data.

        Parameters
        ----------
        hv
            Photon energy in eV.

        Note
        ----
        This method returns an overlay curve :math:`k_z(hν)` for converted momentum
        data. The returned values depend on the requested photon energies, binding
        energy, and in-plane momentum coordinates, but never on the converted data's
        current ``kz`` grid.

        If the carried in-plane momentum perpendicular to the slit is independent of the
        converted ``kz`` axis, :math:`k_z` is evaluated directly from

        .. math::

            k_z = \sqrt{\frac{2 m_e}{\hbar^2}(E_k + V_0) - k_x^2 - k_y^2}.

        If that carried orthogonal momentum depends on the converted ``kz`` axis, the
        method solves the sampled fixed-point relation

        .. math::

            k_z = \sqrt{\frac{2 m_e}{\hbar^2}(E_k + V_0) - k_{\mathrm{slit}}^2 -
            k_{\mathrm{other}}(k_z)^2}

        on the stored ``kz`` grid. If several sampled roots are present, a continuous
        branch is selected across the momentum axis along the slit. Slices with no
        sampled solution are returned as ``NaN``. The legacy angle-roundtrip path is
        only used when the orthogonal in-plane momentum coordinate is unavailable.

        """
        if isinstance(hv, Iterable) and not isinstance(hv, xr.DataArray):
            hv = xr.DataArray(np.asarray(hv), coords={"hv": hv})

        # Get kinetic energies for the given photon energy
        kinetic = hv - self.work_function + self._obj.eV
        self._check_kinetic_energy(
            context="calculating kz from photon energy",
            kinetic_energy=kinetic,
        )
        direct = self._hv_to_kz_from_stored_coords(kinetic)
        if direct is not None:
            return direct
        try:
            return self._hv_to_kz_legacy(kinetic)
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                "`hv_to_kz` requires the orthogonal in-plane momentum coordinate or "
                "enough metadata to reconstruct it."
            ) from exc
