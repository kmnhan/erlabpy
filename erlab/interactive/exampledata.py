"""Generates simple simulated ARPES data from Graphene bands."""

import numpy as np
import xarray as xr
import scipy.ndimage

__all__ = ["generate_data"]


def func(kvec, a):
    n1 = np.array([a / np.sqrt(3), 0])
    n2 = np.array([-a / 2 / np.sqrt(3), a / 2])
    n3 = np.array([-a / 2 / np.sqrt(3), -a / 2])
    return (
        np.exp(1j * np.dot(n1, kvec))
        + np.exp(1j * np.dot(n2, kvec))
        + np.exp(1j * np.dot(n3, kvec))
    )


def fermi_dirac(E, T):
    return 1 / (np.exp(E / (8.61733e-5 * T)) + 1)


def band(kvec, t, a):
    return -t * np.abs(func(kvec, a))


def spectral_function(w, bareband, Sreal, Simag):
    return Simag / (np.pi * ((w - bareband - Sreal) ** 2 + Simag**2))


def add_fd_norm(image, eV, temp=30, efermi=0, count=1e3):
    if temp != 0:
        image *= fermi_dirac(eV - efermi, temp)[None, None, :]
    image += 1e-5
    image /= image.max()
    image *= count / np.mean(image)


def generate_data(
    shape: tuple[int, int, int] = (200, 200, 250),
    krange: float = 1.4,
    Erange: tuple[float, float] = (-0.45, 0.09),
    temp: float = 30.0,
    a: float = 6.97,
    t: float = 0.43,
    bandshift: float = -0.2,
    Sreal: float = 0.0,
    Simag: float = 0.05,
    kres: float = 0.01,
    Eres: float = 2.0e-3,
    noise: bool = True,
    count: int = 1000,
    ccd_sigma: float = 0.6,
):
    kx = np.linspace(-krange, krange, shape[0])
    ky = np.linspace(-krange, krange, shape[1])
    eV = np.linspace(*Erange, shape[2])

    dE = eV[1] - eV[0]
    dk = ((kx[1] - kx[0]) + (ky[1] - ky[0])) / 2

    point_iter = np.array(np.meshgrid(kx, ky)).T.reshape(-1, 2).T

    Eij = band(point_iter, t, a).reshape(*shape[:-1], 1)

    Akw_p = spectral_function(eV[None, None, :], Eij + bandshift, Sreal, Simag)
    Akw_m = spectral_function(eV[None, None, :], -Eij + bandshift, Sreal, Simag)

    phase = np.angle(func(point_iter, a)).reshape(shape[0], shape[1])
    Akw_p *= (1 + np.cos(phase))[:, :, None]
    Akw_m *= (1 - np.cos(phase))[:, :, None]
    out = Akw_p + Akw_m

    add_fd_norm(out, eV, efermi=0, temp=temp, count=count)

    if noise:
        rng = np.random.default_rng()
        out = rng.poisson(out).astype(float)

    broadened = scipy.ndimage.gaussian_filter(
        out,
        sigma=[kres / dk, kres / dk, Eres / dE],
        truncate=10.0,
    )
    if noise:
        broadened = scipy.ndimage.gaussian_filter(
            rng.poisson(broadened).astype(float), ccd_sigma, truncate=10.0
        )

    return xr.DataArray(broadened, coords=dict(kx=kx, ky=ky, eV=eV))


if __name__ == "__main__":
    out = generate_data(
        shape=(200, 200, 200),
        krange=1.4,
        Erange=(-0.45, 0.09),
        temp=30,
        bandshift=-0.2,
        count=1000,
        noise=True,
    )
    import arpes.xarray_extensions

    out.S.show()
