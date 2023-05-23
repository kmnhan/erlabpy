from typing import Callable

import numba
import numpy as np
import numpy.typing as npt

from erlab.constants import kb_eV

TINY: float = 1.0e-15  #: From `lmfit.lineshapes`, equal to `numpy.finfo(numpy.float64).resolution`


@numba.njit(cache=True)
def _gen_kernel(
    x: npt.NDArray[np.float64], resolution: float, pad: int = 12
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    delta_x = x[1] - x[0]
    sigma = abs(resolution) / np.sqrt(8 * np.log(2))  # resolution given in FWHM
    n_pad = int(resolution * pad / delta_x + 0.5)
    x_pad = n_pad * delta_x

    extended = np.linspace(x[0] - x_pad, x[-1] + x_pad, 2 * n_pad + len(x))
    gauss = (
        delta_x
        * np.exp(
            -(np.linspace(-x_pad, x_pad, 2 * n_pad + 1) ** 2)
            / max(TINY, 2 * sigma**2)
        )
        / max(TINY, np.sqrt(2 * np.pi) * sigma)
    )
    return extended, gauss


def do_convolve(
    x: npt.NDArray[np.float64],
    func: Callable,
    resolution: float,
    pad: int = 5,
    **kwargs: dict
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


@numba.njit(cache=True)
def gaussian_wh(
    x: npt.NDArray[np.float64], center: float, width: float, height: float
) -> npt.NDArray[np.float64]:
    r"""Gaussian parametrized with FWHM and peak height.

    Notes
    -----
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

    Notes
    -----
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

    Notes
    -----
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
