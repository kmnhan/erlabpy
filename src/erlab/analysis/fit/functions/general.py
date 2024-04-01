"""Some general functions and utilities used in fitting.

Many functions are `numba`-compiled for speed.
"""

__all__ = [
    "TINY",
    "do_convolve",
    "do_convolve_y",
    "gaussian_wh",
    "lorentzian_wh",
    "fermi_dirac",
    "fermi_dirac_linbkg",
    "fermi_dirac_linbkg_broad",
    "step_linbkg_broad",
    "step_broad",
]

from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt
import scipy.special

from erlab.constants import kb_eV

#: From :mod:`lmfit.lineshapes`, equal to `numpy.finfo(numpy.float64).resolution`
TINY: float = 1.0e-15


@numba.njit(cache=True)
def _gen_kernel(
    x: npt.NDArray[np.float64], resolution: float, pad: int = 5
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Generate a Gaussian kernel for convolution.

    Parameters
    ----------
    x
        The input array of x values.
    resolution
        The resolution of the kernel given as FWHM.
    pad
        Multiples of the standard deviation :math:`\sigma` to truncate the kernel at.

    Returns
    -------
    extended
        The domain of the kernel.
    gauss
        The gaussian kernel defined on `extended`.

    """
    delta_x = x[1] - x[0]
    sigma = abs(resolution) / np.sqrt(8 * np.log(2))  # resolution given in FWHM
    n_pad = int(sigma * pad / delta_x + 0.5)
    x_pad = n_pad * delta_x

    extended = np.linspace(x[0] - x_pad, x[-1] + x_pad, 2 * n_pad + len(x))
    gauss = (
        delta_x
        * np.exp(
            -(np.linspace(-x_pad, x_pad, 2 * n_pad + 1) ** 2) / max(TINY, 2 * sigma**2)
        )
        / max(TINY, np.sqrt(2 * np.pi) * sigma)
    )
    return extended, gauss


def do_convolve(
    x: npt.NDArray[np.float64],
    func: Callable,
    resolution: float,
    pad: int = 5,
    **kwargs: dict,
) -> npt.NDArray[np.float64]:
    r"""Convolves `func` with gaussian of FWHM `resolution` in `x`.

    Parameters
    ----------
    x
        A evenly spaced array specifing where to evaluate the convolution.
    func
        Function to convolve.
    resolution
        FWHM of the gaussian kernel.
    pad
        Multiples of the standard deviation :math:`\sigma` to pad with.
    **kwargs
        Additional keyword arguments to `func`.

    """
    xn, g = _gen_kernel(
        np.asarray(x, dtype=np.float64), float(resolution), pad=int(pad)
    )
    return np.convolve(func(xn, **kwargs), g, mode="valid")


def do_convolve_y(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    func: Callable,
    resolution: float,
    pad: int = 5,
    **kwargs: dict,
) -> npt.NDArray[np.float64]:
    xn, g = _gen_kernel(
        np.asarray(np.squeeze(x), dtype=np.float64), resolution, pad=pad
    )
    if not np.iterable(y):
        y = [y]
    return np.vstack(
        [
            np.convolve(func(xn, yi, **kwargs), g, mode="valid")
            for yi in np.asarray(y).flat
        ]
    ).T


@numba.njit(cache=True)
def gaussian_wh(
    x: npt.NDArray[np.float64], center: float, width: float, height: float
) -> npt.NDArray[np.float64]:
    r"""Gaussian parametrized with FWHM and peak height.

    Note
    ----
    :math:`\sigma=\frac{w}{2\sqrt{2\log{2}}}`

    """
    return height * np.exp(
        -16 * np.log(2) * (1.0 * x - center) ** 2 / max(TINY, width**2)
    )


@numba.njit(cache=True)
def lorentzian_wh(
    x: npt.NDArray[np.float64], center: float, width: float, height: float
) -> npt.NDArray[np.float64]:
    r"""Lorentzian parametrized with FWHM and peak height.

    Note
    ----
    :math:`\sigma=w/2`

    """
    return height / (1 + 4 * ((1.0 * x - center) / max(TINY, width)) ** 2)


@numba.njit(cache=True)
def fermi_dirac(
    x: npt.NDArray[np.float64], center: float, temp: float
) -> npt.NDArray[np.float64]:
    """Fermi-dirac edge in terms of temperature."""
    return 1 / (1 + np.exp((1.0 * x - center) / max(TINY, temp * kb_eV)))


# adapted and improved from KWAN Igor procedures
@numba.njit(cache=True)
def fermi_dirac_linbkg(
    x: npt.NDArray[np.float64],
    center: float,
    temp: float,
    back0: float,
    back1: float,
    dos0: float,
    dos1: float,
) -> npt.NDArray[np.float64]:
    """Fermi-dirac edge with linear backgrounds above and below the fermi level.

    Note
    ----
    `back0` and `back1` corresponds to the linear background above and below EF (due to
    non-homogeneous detector efficiency or residual intensity on the phosphor screen
    during sweep mode), while `dos0` and `dos1` corresponds to the linear density of
    states below EF including the linear background.
    """
    return (back0 + back1 * x) + (dos0 - back0 + (dos1 - back1) * x) / (
        1 + np.exp((1.0 * x - center) / max(TINY, temp * kb_eV))
    )


def fermi_dirac_linbkg_broad(
    x: npt.NDArray[np.float64],
    center: float,
    temp: float,
    resolution: float,
    back0: float,
    back1: float,
    dos0: float,
    dos1: float,
) -> npt.NDArray[np.float64]:
    """
    Resolution-broadened Fermi-dirac edge with linear backgrounds above and below the
    fermi level.
    """
    return do_convolve(
        x,
        fermi_dirac_linbkg,
        resolution=resolution,
        center=center,
        temp=temp,
        back0=back0,
        back1=back1,
        dos0=dos0,
        dos1=dos1,
    )


def step_broad(
    x: npt.NDArray[np.float64],
    center: float = 0.0,
    sigma: float = 1.0,
    amplitude: float = 1.0,
):
    """Step function convolved with a Gaussian."""
    return (
        amplitude
        * 0.5
        * scipy.special.erfc((1.0 * x - center) / max(TINY, np.sqrt(2) * sigma))
    )


def step_linbkg_broad(
    x: npt.NDArray[np.float64],
    center: float,
    sigma: float,
    back0: float,
    back1: float,
    dos0: float,
    dos1: float,
):
    """
    A linear density of states multiplied with a resolution broadened step function with
    a linear background.
    """
    return (back0 + back1 * x) + (dos0 - back0 + (dos1 - back1) * x) * (
        step_broad(x, center, sigma, 1.0)
    )
