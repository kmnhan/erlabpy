"""Some general functions and utilities used in fitting.

Many functions are ``numba``-compiled for speed.

.. note::

    ``numba``-compiled functions do not accept xarray objects. To broadcast over xarray
    objects, use :func:`xarray.apply_ufunc`.

"""

__all__ = [
    "TINY",
    "bcs_gap",
    "do_convolve",
    "do_convolve_2d",
    "dynes",
    "fermi_dirac",
    "fermi_dirac_broad",
    "fermi_dirac_linbkg",
    "fermi_dirac_linbkg_broad",
    "gaussian",
    "gaussian_wh",
    "lorentzian",
    "lorentzian_wh",
    "sc_self_energy",
    "sc_spectral_function",
    "step_broad",
    "step_linbkg_broad",
    "tll",
    "voigt",
]

from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.special

from erlab.constants import TINY, kb_eV
from erlab.utils._array_jit import _clip_tiny
from erlab.utils.array import broadcast_args

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

    coord = arr.reshape(shape)[:, 0] if axis == 0 else arr.reshape(shape)[0, :]

    return shape, axis, coord


@numba.njit(cache=True)
def _gen_kernel(
    x: npt.NDArray[np.float64], resolution: float, pad: int = 6
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
    delta_x = np.abs(x[1] - x[0])

    sigma = np.abs(resolution) / np.sqrt(8 * np.log(2))  # resolution given in FWHM
    n_pad = min(int(sigma * pad / delta_x + 0.5), len(x) - 1)

    x_pad = n_pad * delta_x

    extended = np.linspace(x[0] - x_pad, x[-1] + x_pad, 2 * n_pad + len(x))
    gauss = np.exp(
        -(np.linspace(-x_pad, x_pad, 2 * n_pad + 1) ** 2) / _clip_tiny(2 * sigma**2)
    )
    gauss /= gauss.sum()
    return extended, gauss


def _choose_conv_method_na(in1, in2, mode):
    if np.isfinite(in1).all() and np.isfinite(in2).all():
        return scipy.signal.choose_conv_method(in1, in2, mode)
    return "direct"


def _convolve(in1, in2, mode: str = "full"):
    return scipy.signal.convolve(
        in1, in2, mode=mode, method=_choose_conv_method_na(in1, in2, mode)
    )


def do_convolve(
    x: npt.NDArray[np.float64],
    func: Callable,
    resolution: float,
    pad: int = 7,
    oversample: int = 3,
    **kwargs,
) -> npt.NDArray[np.float64]:
    r"""Convolves `func` with gaussian of FWHM `resolution` in `x`.

    Parameters
    ----------
    x
        An evenly spaced 1D array specifying where to evaluate the convolution.
    func
        Function to convolve.
    resolution
        FWHM of the gaussian kernel.
    pad
        Multiples of the standard deviation :math:`\sigma` to pad with.
    oversample
        Factor by which to oversample `x` for convolution to reduce numerical artifacts.
    **kwargs
        Additional keyword arguments to `func`.

    """
    if np.isclose(resolution, 0.0):
        return func(x, **kwargs)
    if oversample == 1:
        xn, g = _gen_kernel(
            np.asarray(x, dtype=np.float64), float(resolution), pad=int(pad)
        )
        return _convolve(func(xn, **kwargs), g, mode="valid")

    fine_dx = (x[1] - x[0]) / oversample
    n_fine = (x.size - 1) * oversample + 1
    x_fine = x[0] + np.arange(n_fine, dtype=np.float64) * fine_dx

    xn_fine, g = _gen_kernel(x_fine, float(resolution), pad=int(pad))
    return _convolve(func(xn_fine, **kwargs), g, mode="valid")[::oversample]


def do_convolve_segments(
    x: npt.NDArray[np.float64],
    func: Callable,
    resolution: float,
    pad: int = 5,
    **kwargs,
) -> npt.NDArray[np.float64]:
    r"""
    Convolves `func` with gaussian of FWHM `resolution` in `x` with uniform segments.

    This function is useful when `x` is piecewise evenly spaced and the convolution
    needs to be performed independently on uniform segments of `x`.

    Parameters
    ----------
    x
        A piecewise evenly spaced 1D array.
    func
        Function to convolve.
    resolution
        FWHM of the gaussian kernel.
    pad
        Multiples of the standard deviation :math:`\sigma` to pad with.
    **kwargs
        Additional keyword arguments to `func`.
    """
    from erlab.utils._array_jit import _split_uniform_segments

    x_segments = _split_uniform_segments(np.asarray(x, dtype=np.float64))
    return np.concatenate(
        [do_convolve(x, func, resolution, pad=pad, **kwargs) for x in x_segments]
    )


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
    if idx_x == 0:
        return convolved.T.ravel()
    return convolved.ravel()


@broadcast_args
@numba.njit(cache=True)
def gaussian_wh(
    x: npt.NDArray[np.float64],
    center: float = 0.0,
    width: float = 1.0,
    height: float = 1.0,
) -> npt.NDArray[np.float64]:
    r"""Gaussian parametrized with FWHM and peak height.

    .. math::

        G(x) = h \exp\left[-\frac{16 \log{2} (x-x_0)^2}{w^2}\right]

    Note
    ----
    :math:`\sigma=\frac{w}{2\sqrt{2\log{2}}}`

    """
    return height * np.exp(
        -16 * np.log(2) * (1.0 * x - center) ** 2 / _clip_tiny(width**2)
    )


@broadcast_args
@numba.njit(cache=True)
def gaussian(
    x: npt.NDArray[np.float64], center: float, sigma: float, amplitude: float
) -> npt.NDArray[np.float64]:
    r"""
    Gaussian parametrized with standard deviation and amplitude.

    .. math::

        G(x) = \frac{A}{\sqrt{2\pi\sigma^2}}
        \exp\left[-\frac{(x-x_0)^2}{2\sigma^2}\right]

    """
    return (amplitude / _clip_tiny(S2PI * sigma)) * np.exp(
        -((1.0 * x - center) ** 2) / _clip_tiny(2 * sigma**2)
    )


@broadcast_args
@numba.njit(cache=True)
def lorentzian_wh(
    x: npt.NDArray[np.float64],
    center: float = 0.0,
    width: float = 1.0,
    height: float = 1.0,
) -> npt.NDArray[np.float64]:
    r"""Lorentzian parametrized with FWHM and peak height.

    .. math::

        L(x) = \frac{h}{1 + 4\left(\frac{x-x_0}{w}\right)^2}

    Note
    ----
    :math:`\sigma=w/2`

    """
    return height / (1 + 4 * ((1.0 * x - center) / _clip_tiny(width)) ** 2)


@broadcast_args
@numba.njit(cache=True)
def lorentzian(
    x: npt.NDArray[np.float64], center: float, gamma: float, amplitude: float
) -> npt.NDArray[np.float64]:
    r"""
    Lorentzian parametrized with HWHM and amplitude.

    .. math::

        L(x) = \frac{A}{\pi\gamma\left[1 + \left(\frac{x-x_0}{\gamma}\right)^2\right]}

    """
    return (
        amplitude / (1 + ((1.0 * x - center) / _clip_tiny(gamma)) ** 2)
    ) / _clip_tiny(np.pi * gamma)


@broadcast_args
def voigt(
    x: npt.NDArray[np.float64],
    center: float = 0.0,
    sigma: float = 0.0,
    gamma: float = 0.5,
    amplitude: float = 1.0,
) -> npt.NDArray[np.float64]:
    r"""Voigt profile.

    The Voigt profile can be expressed as:

    .. math::

        V(x) = A \frac{\text{Re}[w(z)]}{\sigma \sqrt{2\pi}}, \quad z = \frac{x - x_0 + i
        \gamma}{\sigma \sqrt{2}}

    where :math:`w(z)` is the Faddeeva function. This implementation uses
    :func:`scipy.special.voigt_profile` to compute the Voigt profile.

    """
    return amplitude * scipy.special.voigt_profile(x - center, sigma, gamma)


@broadcast_args
@numba.njit(cache=True)
def fermi_dirac(
    x: npt.NDArray[np.float64], center: float, temp: float
) -> npt.NDArray[np.float64]:
    r"""Fermi-dirac distribution.

    .. math::

        f(x) = \frac{1}{1 + e^{(x-x_0)/k_B T}}

    Parameters
    ----------
    x
        Energy values at which to calculate the Fermi edge.
    center
        The Fermi level.
    temp
        The temperature in K.

    """
    return 1 / (1 + np.exp((1.0 * x - center) / _clip_tiny(temp * kb_eV)))


@broadcast_args
def fermi_dirac_broad(
    x: npt.NDArray[np.float64], center: float, temp: float, resolution: float
) -> npt.NDArray[np.float64]:
    r"""Resolution-broadened Fermi edge.

    The Fermi edge is calculated as:

    .. math::

        \frac{1}{1 + e^{(x-x_0)/k_B T}} \otimes \text{g}(\sigma)

    where :math:`\text{g}(\sigma)` is a Gaussian kernel with standard deviation
    :math:`\sigma`. Note that the resolution is given in FWHM rather than the standard
    deviation.

    Parameters
    ----------
    x
        The energy values at which to calculate the Fermi edge.
    center
        The Fermi level.
    temp
        The temperature in K.
    resolution
        The resolution of the Gaussian kernel in eV. Note that this is the FWHM of the
        Gaussian kernel, not the standard deviation.
    """
    return do_convolve(
        x,
        fermi_dirac,
        resolution=resolution,
        center=center,
        temp=temp,
    )


@broadcast_args
# adapted and improved from KWAN Igor procedures
@broadcast_args
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
    r"""Fermi-dirac edge with linear backgrounds above and below the Fermi level.

    .. math::

        I(x) = b_0 + b_1 x + \frac{d_0 - b_0 + (d_1 - b_1) x} {1 + e^{(x-x_0)/k_B T}}

    Parameters
    ----------
    x
        The energy values at which to calculate the Fermi edge.
    center
        The Fermi level.
    temp
        The temperature in K.
    back0
        The constant background above the Fermi level.
    back1
        The slope of the background above the Fermi level.
    dos0
        The constant background below the Fermi level.
    dos1
        The slope of the background below the Fermi level.

    Note
    ----
    `back0` and `back1` corresponds to the linear background above and below EF (due to
    non-homogeneous detector efficiency or residual intensity on the phosphor screen
    during swept measurements), while `dos0` and `dos1` corresponds to the linear
    density of states below EF including the linear background.
    """
    return (back0 + back1 * x) + (dos0 - back0 + (dos1 - back1) * x) / (
        1 + np.exp((1.0 * x - center) / _clip_tiny(temp * kb_eV))
    )


@broadcast_args
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
    r"""Resolution-broadened Fermi edge with linear backgrounds.

    .. math::

        I(x) = \left[ b_0 + b_1 x + \frac{d_0 - b_0 + (d_1 - b_1) x} {1 +
        e^{(x-x_0)/k_B T}} \right] \otimes \text{g}(\sigma)

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


def _tll_bare(x, amp=1.0, center=0.0, alpha=0.1, temp=10.0):
    r"""Tomonaga-Luttinger liquid spectral function.

    The TLL spectral function is calculated as :cite:p:`ohtsubo2015tll`:

    .. math::

        I(x,T) = A T^\alpha \cosh\left(\frac{\epsilon}{2}\right)
        \left|\Gamma\left(\frac{1 + \alpha}{2} + i \frac{\epsilon}{2\pi}\right)\right|^2
        f(\epsilon,T)

    where :math:`\epsilon=(x - x_0)/k_B T` is the temperature-normalized energy,
    :math:`\Gamma` is the gamma function, and :math:`f(\epsilon,T) = 1/(e^{\epsilon}+1)`
    is the Fermi-Dirac distribution.

    Parameters
    ----------
    x
        The energy values at which to calculate the TLL spectral function.
    amp
        The amplitude.
    center
        The center.
    alpha
        The power law exponent.
    temp
        The temperature in K.

    """
    x_n = (x - center) / kb_eV / temp
    gamma_arg = (1.0 + alpha) / 2.0 + (1j * x_n / (2.0 * np.pi))
    return (
        amp
        * temp**alpha
        * np.cosh(x_n / 2.0)
        * np.abs(scipy.special.gamma(gamma_arg)) ** 2.0
        * scipy.special.expit(-x_n)
    )


@broadcast_args
def tll(
    x: npt.NDArray[np.float64],
    amp: float = 1.0,
    center: float = 0.0,
    alpha: float = 0.1,
    temp: float = 10.0,
    resolution: float = 0.01,
    const_bkg: float = 0.0,
) -> npt.NDArray[np.float64]:
    r"""Resolution-broadened Tomonaga-Luttinger liquid (TLL) spectral function.

    The TLL spectral function is calculated as :cite:p:`ohtsubo2015tll`:

    .. math::

        I(x,T) = \left[A T^\alpha \cosh\left(\frac{\epsilon}{2}\right)
        \left|\Gamma\left(\frac{1 + \alpha}{2} + i \frac{\epsilon}{2\pi}\right)\right|^2
        f(\epsilon,T)\right] \otimes \text{g}(\sigma) + B

    where :math:`\epsilon=(x - x_0)/k_B T` is the temperature-normalized energy,
    :math:`\Gamma` is the gamma function, :math:`f(\epsilon,T) = 1/(e^{\epsilon}+1)` is
    the Fermi-Dirac distribution, :math:`\text{g}(\sigma)` is a Gaussian kernel with
    standard deviation :math:`\sigma`, and :math:`B` is a constant background.

    Note that the resolution parameter is the FWHM of the Gaussian kernel, not the
    standard deviation.

    Parameters
    ----------
    x
        The energy values at which to calculate the TLL spectral function.
    amp
        The amplitude.
    center
        The center.
    alpha
        The power law exponent.
    temp
        The temperature in K.
    resolution
        The resolution of the Gaussian kernel in eV. Note that this is the FWHM of the
        Gaussian kernel, not the standard deviation.
    const_bkg
        A constant background to add to the broadened TLL function.

    """
    return (
        do_convolve(
            x,
            _tll_bare,
            resolution=resolution,
            amp=amp,
            center=center,
            alpha=alpha,
            temp=temp,
            oversample=5,
        )
        + const_bkg
    )


@broadcast_args
def step_broad(
    x: npt.NDArray[np.float64],
    center: float = 0.0,
    sigma: float = 1.0,
    amplitude: float = 1.0,
):
    r"""Step function convolved with a Gaussian.

    The broadened step function is calculated as:

    .. math::

        \frac{A}{2}\cdot\text{erfc}\left(\frac{x - x_0}{\sqrt{2\sigma^2}}\right)

    where :math:`\text{erfc}` is the complementary error function.

    Parameters
    ----------
    x
        The input array of x values.
    center
        The center of the step function.
    sigma
        The standard deviation of the Gaussian.
    amplitude
        The amplitude of the step.

    """
    return (
        amplitude
        * 0.5
        * scipy.special.erfc((1.0 * x - center) / _clip_tiny(np.sqrt(2) * sigma))
    )


@broadcast_args
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


@broadcast_args
@numba.njit()
def bcs_gap(
    x, a: float = 1.76, b: float = 1.74, tc: float = 100.0
) -> npt.NDArray[np.float64]:
    r"""Interpolation formula for temperature dependent BCS-like gap magnitude.

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

        N_0  \text{Re}\left[\frac{|x| + i \Gamma}{\sqrt{(|x| + i \Gamma)^2 -
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


def sc_self_energy(x, gamma1=1e-5, gamma0=0.0, delta=0.0):
    r"""General phenomenological self-energy for superconductors.

    The function is given by :cite:p:`norman1998sc`:

    .. math::

        \Sigma(x) = -i \Gamma_1 + \frac{\Delta^2}{x + i \Gamma_0}

    where :math:`\Gamma_1` is the single-particle scattering rate, :math:`\Gamma_0` is
    the pair-breaking rate, and :math:`\Delta` is the energy gap.

    Parameters
    ----------
    x : array-like
        The input array of energy in eV.
    gamma1
        :math:`\Gamma_1`, the single-particle scattering rate.
    gamma0
        :math:`\Gamma_0`, the pair-breaking scattering rate.
    delta
        The energy gap :math:`\Delta`.

    """
    return -1j * gamma1 + (delta**2) / (x + 1j * gamma0)


def spectral_function(x, im=0, re=0):
    r"""General spectral function given imaginary and real parts of self-energy.

    The spectral function is calculated as:

    .. math::

        A(x) = \frac{-\text{Im}\Sigma(x)}{(x - \text{Re}\Sigma(x))^2 +
        (\text{Im}\Sigma(x))^2} \cdot \frac{1}{\pi}

    where :math:`\Sigma(x)` is the complex self-energy.

    Parameters
    ----------
    x : array-like
        The input array of energy in eV.
    im
        The imaginary part of the self-energy :math:`\text{Im}\Sigma(x)`.
    re
        The real part of the self-energy :math:`\text{Re}\Sigma(x)`.
    """
    return -im / ((x - re) ** 2 + im**2) / np.pi


def _sc_spectral_function_bare(x, amp, gamma1, gamma0, delta, lin_bkg, const_bkg):
    r"""Superconducting spectral function with linear background.

    The superconducting spectral function with a linear background is calculated as:

    .. math::

        I(x) = A \cdot A_{sc}(x, \Sigma) + \left(\text{sgn}(x) \cdot m \cdot x + b
        \right)

    where :math:`A_{sc}(x, \Sigma)` is the superconducting spectral function calculated
    using the self-energy :math:`\Sigma(x)` given by :func:`sc_self_energy
    <erlab.analysis.fit.functions.general.sc_self_energy>`.

    Parameters
    ----------
    x : array-like
        The input array of energy in eV.
    amp
        The amplitude :math:`A`.
    gamma1
        :math:`\Gamma_1`, the single-particle scattering rate.
    gamma0
        :math:`\Gamma_0`, the pair-breaking scattering rate.
    delta
        The energy gap :math:`\Delta`.
    lin_bkg
        The slope of the linear background :math:`m`.
    const_bkg
        The constant background :math:`b`.
    """
    se = sc_self_energy(x, gamma1, gamma0, delta)
    akw = spectral_function(x, np.imag(se), np.real(se))
    return amp * akw + (lin_bkg * np.abs(x) + const_bkg)


@broadcast_args
def sc_spectral_function(
    x: npt.NDArray[np.float64],
    amp: float = 1.0,
    gamma1: float = 1e-5,
    gamma0: float = 0.0,
    delta: float = 0.0,
    lin_bkg: float = 0.0,
    const_bkg: float = 0.0,
    resolution: float = 0.01,
) -> npt.NDArray[np.float64]:
    r"""Resolution-broadened superconducting spectral function.

    The superconducting spectral function with a linear background is calculated as:

    .. math::

        I(x) = \left[ A \cdot A_{sc}(x, \Sigma) + \left(m \cdot |x| + b \right) \right]
        \otimes \text{g}(\sigma)

    where :math:`A_{sc}(x, \Sigma)` is the superconducting spectral function calculated
    using the self-energy :math:`\Sigma(x)` given by :func:`sc_self_energy`.
    :math:`\text{g}(\sigma)` is a Gaussian kernel with standard deviation
    :math:`\sigma`. Note that the resolution parameter is the FWHM of the Gaussian
    kernel, not the standard deviation.

    Parameters
    ----------
    x : array-like
        The input array of energy in eV.
    amp
        The overall scale factor :math:`A`.
    gamma1
        :math:`\Gamma_1`, the single-particle scattering rate.
    gamma0
        :math:`\Gamma_0`, the pair-breaking scattering rate.
    delta
        The energy gap :math:`\Delta`.
    lin_bkg
        The slope of the linear background :math:`m`.
    const_bkg
        The constant background :math:`b`.
    resolution
        The broadening in eV. Note that this is the FWHM of the Gaussian kernel, not the
        standard deviation.
    """
    return do_convolve(
        x,
        _sc_spectral_function_bare,
        resolution=resolution,
        amp=amp,
        gamma1=gamma1,
        gamma0=gamma0,
        delta=delta,
        lin_bkg=lin_bkg,
        const_bkg=const_bkg,
    )
