"""
`xarray accessors <https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_

.. currentmodule:: erlab.accessors

"""
import numpy as np
import xarray as xr

import erlab.interactive
from erlab.analysis.interpolate import interpn
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
        return self.photon_energy - self._obj.S.work_function + self.binding_energy

    @property
    def xi(self):
        return self._obj.S.lookup_offset_coord("beta")

    @property
    def beta(self):
        return self._obj.S.lookup_offset_coord("theta")

    @property
    def delta(self):
        return self._obj.S.lookup_offset_coord("chi")

    @property
    def alpha(self):
        return self._obj.S.lookup_offset_coord("phi")

    @property
    def ktot(self):
        return rel_kconv * np.sqrt(self.kinetic_energy)

    def get_bounds(self):
        # construct boundary array from meshgrid
        alpha, beta = [
            np.r_[arr[0, :-1], arr[:-1, -1], arr[-1, ::-1], arr[-2:0:-1, 0]]
            for arr in np.meshgrid(self.alpha, self.beta)
        ]
        kx, ky = self._forward_func(alpha[np.newaxis, :], beta[np.newaxis, :])
        return (kx.min(), kx.max()), (ky.min(), ky.max())

    @property
    def angle_resolution(self):
        try:
            return self._obj.attrs["angle_resolution"]
        except KeyError:
            return 0.1

    @angle_resolution.setter
    def angle_resolution(self, value: float):
        self._obj.attrs["angle_resolution"] = value

    @property
    def minimum_k_resolution(self):
        min_ek = np.min(self.kinetic_energy)
        max_angle = max(np.abs(self._obj.phi.values))
        return (
            rel_kconv
            * np.sqrt(min_ek)
            * np.cos(max_angle)
            * np.deg2rad(self.angle_resolution)
        )

    def _forward_func(self, alpha, beta):
        ca = np.cos(alpha)
        cb = np.cos(beta)
        cd = np.cos(self.delta)
        cx = np.cos(self.xi)
        sa = np.sin(alpha)
        sb = np.sin(beta)
        sd = np.sin(self.delta)
        sx = np.sin(self.xi)
        kx = self.ktot * ((sd * sb + cd * sx * cb) * ca - cd * cx * sa)
        ky = self.ktot * ((-cd * sb + sd * sx * cb) * ca - sd * cx * sa)
        return kx, ky

    def kz(self, kx, ky):
        return np.sqrt(self.ktot**2 - kx**2 - ky**2)

    def _inverse_func(self, kx, ky):
        cd = np.cos(self.delta)
        cx = np.cos(self.xi)
        sd = np.sin(self.delta)
        sx = np.sin(self.xi)

        kperp = self.kz(kx, ky)
        alpha = np.arcsin((sx * kperp - cx * (cd * kx + sd * ky)) / self.ktot)
        beta = np.arctan((sd * kx - cd * ky) / (sx * (cd * kx + sd * ky) + cx * kperp))
        if not self.has_eV:
            return alpha, beta + self._obj.S.theta_offset
        else:
            eV = self.binding_energy * np.ones_like(alpha)
            return eV, alpha, beta + self._obj.S.theta_offset

    def forward(self, phi, theta):
        # make alpha and beta (1,N) where N = len(phi) * len(theta)
        alpha, beta = np.array(np.meshgrid(phi, theta)).reshape(2, -1)[:, None]

        if self.has_hv:
            alpha = alpha[:, :, np.newaxis]
            beta = beta[:, :, np.newaxis]
        # binding energy 없이 photon energy만 주어지는 경우는 없겠지?

        return self._forward_func(alpha, beta)

    def inverse(self, kx, ky):
        # make kxval and kyval (1,N) where N = len(kx) * len(ky)
        kxval, kyval = np.array(np.meshgrid(kx, ky)).reshape(2, -1)[:, None]
        return self._inverse_func(kxval, kyval)

    def convert(
        self,
        bounds: dict[str, tuple[float, float]] | None = None,
        resolution: dict[str, float] | None = None,
    ) -> xr.DataArray:
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

        old_dim_order = ("phi", "theta")
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
