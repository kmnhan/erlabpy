"""Some general functions and utilities used in fitting.

Many functions are `numba`-compiled for speed.
"""

__all__ = [
    "TINY",
    "bcs_gap",
    "do_convolve",
    "do_convolve_2d",
    "dynes",
    "fermi_dirac",
    "fermi_dirac_linbkg",
    "fermi_dirac_linbkg_broad",
    "gaussian",
    "gaussian_wh",
    "lorentzian",
    "lorentzian_wh",
    "step_broad",
    "step_linbkg_broad",
]

from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt
import scipy.special

from erlab.constants import kb_eV

#: From :mod:`lmfit.lineshapes`, equal to `numpy.finfo(numpy.float64).resolution`
TINY: float = 1.0e-15
S2PI = np.sqrt(2 * np.pi)


@numba.njit(cache=True)
def _infer_meshgrid_shape(arr: np.ndarray) -> tuple[tuple[int, int], int, np.ndarray]:
    if arr.ndim != 1:
        raise ValueError("Array must be 1-dimensional")

    axis = 1

    # Find the index at which the array value decreases
    change_index = np.where(np.diff(arr) <= 0)[0]
    if change_index.size == 0:
        raise ValueError("Array does not appear to be a flattened meshgrid")

    if change_index[0] == 0:
        change_index = np.where(np.diff(arr) > 0)[0]
        axis = 0

    # The shape of the original meshgrid
    shape = len(arr) // (change_index[0] + 1), change_index[0] + 1

    if axis == 0:
        coord = arr.reshape(shape)[:, 0]
    else:
        coord = arr.reshape(shape)[0, :]

    return shape, axis, coord


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
        / max(TINY, S2PI * sigma)
    )
    return extended, gauss


def do_convolve(
    x: npt.NDArray[np.float64],
    func: Callable,
    resolution: float,
    pad: int = 5,
    **kwargs,
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


def do_convolve_2d(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64] | float,
    func: Callable,
    resolution: float,
    pad: int = 5,
    **kwargs,
) -> npt.NDArray[np.float64]:
    idx_x = None

    if not np.iterable(y):
        y = np.asarray([y])

    try:
        # check if x is a meshgrid
        shape_x, idx_x, x = _infer_meshgrid_shape(np.ascontiguousarray(x))
        shape_y, _, y = _infer_meshgrid_shape(np.ascontiguousarray(y))
    except ValueError:
        pass
    else:
        if shape_x != shape_y:
            raise ValueError("x and y do not have matching shape")

    xn, g = _gen_kernel(
        np.asarray(np.squeeze(x), dtype=np.float64), resolution, pad=pad
    )

    convolved = np.vstack(
        [
            np.convolve(func(xn, yi, **kwargs), g, mode="valid")
            for yi in np.asarray(y).flat
        ]
    )

    if idx_x is None:
        return convolved.T
    elif idx_x == 0:
        return convolved.T.ravel()
    else:
        return convolved.ravel()


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
def gaussian(
    x: npt.NDArray[np.float64], center: float, sigma: float, amplitude: float
) -> npt.NDArray[np.float64]:
    return (amplitude / (max(TINY, S2PI * sigma))) * np.exp(
        -((1.0 * x - center) ** 2) / max(TINY, (2 * sigma**2))
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
def lorentzian(
    x: npt.NDArray[np.float64], center: float, sigma: float, amplitude: float
) -> npt.NDArray[np.float64]:
    return (amplitude / (1 + ((1.0 * x - center) / max(TINY, sigma)) ** 2)) / max(
        TINY, (np.pi * sigma)
    )


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
    """Resolution-broadened Fermi edge with linear backgrounds above and below EF."""
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
    """Resolution broadened step function with linear backgrounds."""
    return (back0 + back1 * x) + (dos0 - back0 + (dos1 - back1) * x) * (
        step_broad(x, center, sigma, 1.0)
    )


@numba.njit()
def bcs_gap(
    x, a: float = 1.76, b: float = 1.74, tc: float = 100.0
) -> npt.NDArray[np.float64]:
    r"""Interpolation formula for a temperature dependent BCS-like gap.

    .. math::

        \Delta(T) \simeq a \cdot k_B T_c \cdot \tanh\left(b \sqrt{\frac{T_c}{T} -
        1}\right)

    Parameters
    ----------
    x : array-like
        The temperature values in kelvins at which to calculate the BCS gap.
    a
        Proportionality constant. Default is 1.76.
    b
        Proportionality constant. Default is 1.74.
    tc
        The critical temperature in Kelvins. Default is 100.0.

    """
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        if x[i] < tc:
            out[i] = a * kb_eV * tc * np.tanh(b * np.sqrt(tc / x[i] - 1))
        else:
            out[i] = 0.0
    return out


def dynes(x, n0=1.0, gamma=0.003, delta=0.01):
    r"""Dynes formula for superconducting density of states.

    The formula is given by :cite:p:`dynes1978dynes`:

    .. math::

        f(x) = N_0  \text{Re}\left[\frac{|x| + i \Gamma}{\sqrt{(|x| + i \Gamma)^2 -
        \Delta^2}}\right]

    where :math:`x` is the binding energy, :math:`N_0` is the normal-state density of
    states at the Fermi level, :math:`\Gamma` is the broadening term, and :math:`\Delta`
    is the superconducting energy gap.

    Parameters
    ----------
    x : array-like
        The input array of energy in eV.
    n0
        :math:`N_0`, by default 1.0.
    gamma
        :math:`\Gamma`, by default 0.003.
    delta
        The superconducting energy gap :math:`\Delta`, by default 0.01.

    """
    return n0 * np.real(
        (np.abs(x) + 1j * gamma) / (np.sqrt((np.abs(x) + 1j * gamma) ** 2 - delta**2))
    )
