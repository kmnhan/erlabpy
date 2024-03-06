"""
`xarray accessors <https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_

.. currentmodule:: erlab.accessors

"""

import warnings

import numpy as np
import xarray as xr

import erlab.interactive
from erlab.analysis.interpolate import interpn
from erlab.analysis.kspace import AxesConfiguration, get_kconv_func
from erlab.constants import rel_kconv
from erlab.interactive.imagetool import itool


class ERLabAccessor:
    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj


@xr.register_dataset_accessor("er")
class ERLabDataArrayAccessor(ERLabAccessor):
    def show(self, *args, **kwargs):
        return itool(self._obj, *args, **kwargs)


@xr.register_dataarray_accessor("kspace")
class KspaceAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self.reset_offsets()

    @property
    def has_eV(self):
        return "eV" in self._obj.dims

    @property
    def has_hv(self):
        return self._obj.hv.size > 1

    @property
    def photon_energy(self):
        if self.has_hv:
            return self._obj.hv.values[
                tuple(
                    slice(np.newaxis) if (i == self._obj.ndim - 1) else np.newaxis
                    for i in range(self._obj.ndim)
                )
            ]
        else:
            return float(self._obj.hv)

    @property
    def binding_energy(self):
        if self.has_eV:
            return self._obj.eV.values[:, np.newaxis]
        else:
            return float(self._obj.eV.values)

    @property
    def kinetic_energy(self):
        return self.photon_energy - self.work_function + self.binding_energy

    @property
    def inner_potential(self) -> float:
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
        if "sample_workfunction" in self._obj.attrs:
            return float(self._obj.attrs["sample_workfunction"])
        else:
            warnings.warn("Work function not found in data attributes, assuming 4.5 eV")
            return 4.5

    @work_function.setter
    def work_function(self, value: float):
        self._obj.attrs["sample_workfunction"] = float(value)

    @property
    def _valid_offset_keys(self) -> tuple[str, ...]:
        if (
            self.configuration == AxesConfiguration.Type1
            or self.configuration == AxesConfiguration.Type2
        ):
            return ("delta", "xi", "beta")
        else:
            return ("delta", "chi", "xi")

    @property
    def offsets(self) -> dict[str, float]:
        return {k: self.get_offset(k) for k in self._valid_offset_keys}

    @offsets.setter
    def offsets(self, offset_dict: dict[str, float]):
        self.reset_offsets()
        self.set_offsets(**offset_dict)

    @property
    def angle_params(self) -> dict[str, float]:
        """Parameters passed to `get_kconv_func`."""
        params = dict(
            delta=self.get_offset("delta"),
            xi=float(self._obj["xi"].values),
            xi0=self.get_offset("xi"),
        )
        if (
            self.configuration == AxesConfiguration.Type1
            or self.configuration == AxesConfiguration.Type2
        ):
            params["beta0"] = self.get_offset("beta")
        else:
            params["chi"] = float(self._obj["chi"].values)
            params["chi0"] = self.get_offset("chi")
        return params

    @property
    def configuration(self) -> AxesConfiguration:
        """
        Returns the experimental configuration from the data attributes. For a properly
        implemented data loader, The configuration attribute must be set on data import.
        See `erlab.analysis.kspace.AxesConfiguration` for details.
        """
        if "configuration" not in self._obj.attrs:
            raise ValueError(
                "Configuration not found in data attributes! "
                "Data attributes may have been discarded since initial import."
            )
        return AxesConfiguration(int(self._obj.attrs.get("configuration")))

    def reset_offsets(self):
        for k in self._valid_offset_keys:
            self._obj.attrs[k + "_offset"] = 0.0

    def get_offset(self, axis: str) -> float:
        return float(self._obj.attrs[axis + "_offset"])

    def set_offsets(self, **kwargs):
        # delta, chi, xi, beta depending on the geometry
        for k, v in kwargs.items():
            if k in self._valid_offset_keys:
                self._obj.attrs[k + "_offset"] = float(v)

    def get_bounds(self):
        # construct boundary array from meshgrid
        alpha, beta = [
            np.r_[arr[0, :-1], arr[:-1, -1], arr[-1, ::-1], arr[-2:0:-1, 0]]
            for arr in np.meshgrid(self._obj["alpha"], self._obj["beta"])
        ]
        kx, ky = self._forward_func(alpha[np.newaxis, :], beta[np.newaxis, :])
        return (kx.min(), kx.max()), (ky.min(), ky.max())

    @property
    def angle_resolution(self) -> float:
        try:
            return float(self._obj.attrs["angle_resolution"])
        except KeyError:
            warnings.warn(
                "Angle resolution not found in data attributes, assuming 0.1 degrees"
            )
            return 0.1

    @angle_resolution.setter
    def angle_resolution(self, value: float):
        self._obj.attrs["angle_resolution"] = float(value)

    @property
    def minimum_k_resolution(self):
        min_Ek = np.min(self.kinetic_energy)
        max_angle = max(np.abs(self._obj["alpha"].values))
        return (
            rel_kconv
            * np.sqrt(min_Ek)
            * np.cos(np.deg2rad(max_angle))
            * np.deg2rad(self.angle_resolution)
        )

    def _forward_func(self, alpha, beta):
        return get_kconv_func(
            self.kinetic_energy, self.configuration, self.angle_params
        )[0](alpha, beta)

    def _inverse_func(self, kx, ky):
        alpha, beta = get_kconv_func(
            self.kinetic_energy, self.configuration, self.angle_params
        )[1](kx, ky)
        if not self.has_eV:
            return alpha, beta
        else:
            eV = self.binding_energy * np.ones_like(alpha)
            return eV, alpha, beta

    # def forward(self, phi, theta):
    #     # make alpha and beta (1,N) where N = len(phi) * len(theta)
    #     alpha, beta = np.array(np.meshgrid(phi, theta)).reshape(2, -1)[:, None]

    #     if self.has_hv:
    #         alpha = alpha[:, :, np.newaxis]
    #         beta = beta[:, :, np.newaxis]
    #     # binding energy 없이 photon energy만 주어지는 경우는 없겠지?

    #     return self._forward_func(alpha, beta)

    def inverse(self, kx, ky):
        # make kxval and kyval (1,N) where N = len(kx) * len(ky)
        kxval, kyval = np.array(np.meshgrid(kx, ky)).reshape(2, -1)[:, None]
        return self._inverse_func(kxval, kyval)

    def convert(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        resolution: dict[str, float] | None = None,
    ) -> xr.DataArray:
        # if self._obj.ndim == 2:
        # from arpes.utilities.conversion import convert_to_kspace
        # return convert_to_kspace(self._obj, bounds=bounds, resolution=resolution)
        if bounds is None:
            bounds = dict()
        if resolution is None:
            resolution = dict()

        kx_lims = bounds.get("kx", self.get_bounds()[0])
        ky_lims = bounds.get("ky", self.get_bounds()[1])

        kx_res = resolution.get("kx", self.minimum_k_resolution)
        ky_res = resolution.get("ky", self.minimum_k_resolution)

        new_size = dict(
            kx=round((kx_lims[1] - kx_lims[0]) / kx_res + 1),
            ky=round((ky_lims[1] - ky_lims[0]) / ky_res + 1),
        )
        kx = np.linspace(*kx_lims, new_size["kx"])
        ky = np.linspace(*ky_lims, new_size["ky"])

        old_dim_order = ("alpha", "beta")
        new_dim_order = ("ky", "kx")
        interp_coords = dict(kx=kx, ky=ky)

        if self.has_eV:
            old_dim_order = ("eV",) + old_dim_order
            new_dim_order = ("eV",) + new_dim_order
            new_size["eV"] = len(self.binding_energy)
            interp_coords["eV"] = np.squeeze(self.binding_energy)
        if self.has_hv:
            old_dim_order += ("hv",)
            new_dim_order += ("kz",)
            new_size["kz"] = len(self.photon_energy)

        for d in self._obj.dims:
            if d not in old_dim_order:
                old_dim_order += (d,)
                new_dim_order += (d,)

        out = xr.DataArray(
            interpn(
                tuple(self._obj[dim].values for dim in old_dim_order),
                self._obj.copy(deep=True).transpose(*old_dim_order).values,
                self.inverse(kx, ky),
            ).reshape(tuple(new_size[d] for d in new_dim_order)),
            coords=interp_coords,
            dims=new_dim_order,
        )
        out.assign_attrs(self._obj.attrs)
        return out


if __name__ == "__main__":
    import numpy as np

    xr.DataArray(np.array([0, 1, 2]))
