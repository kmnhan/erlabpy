"""Models for fitting data."""

__all__ = [
    "BCSGapModel",
    "DynesModel",
    "FermiDiracModel",
    "FermiEdge2dModel",
    "FermiEdgeModel",
    "MultiPeakModel",
    "PolynomialModel",
    "StepEdgeModel",
    "SymmetrizedGapModel",
    "TLLModel",
]

import typing

import lmfit
import numba
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import xarray as xr

from erlab.analysis.fit.functions import (
    FermiEdge2dFunction,
    MultiPeakFunction,
    PolynomialFunction,
    bcs_gap,
    dynes,
    fermi_dirac_broad,
    fermi_dirac_linbkg_broad,
    sc_spectral_function,
    step_linbkg_broad,
    tll,
)
from erlab.analysis.fit.functions.general import _infer_meshgrid_shape

COMMON_GUESS_DOC = lmfit.models.COMMON_GUESS_DOC.replace(
    """.. versionchanged:: 1.0.3
       Argument ``x`` is now explicitly required to estimate starting values.""",
    "",
)


@numba.njit("f8[:,:](f8[:], i8)", cache=True)
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0], deg + 1), dtype=np.float64)
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_


@numba.njit("f8[:](f8[:,:], f8[:])", cache=True)
def _fit_x(a, b):
    # linalg solves ax = b
    return np.linalg.lstsq(a, b)[0]


@numba.njit("f8[:](f8[:], f8[:], i8)", cache=True)
def fit_poly_jit(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], deg: np.int64):
    """`numba`-accelerated polynomial fitting.

    Parameters
    ----------
    x
        The x-coordinates of the data points.
    y
        The y-coordinates of the data points.
    deg
        The degree of the polynomial to fit.

    Returns
    -------
    p : ndarray
        1D array of polynomial coefficients (including coefficients equal to zero) from
        the constant term to the highest degree.

    """
    a = _coeff_mat(x, deg)
    return _fit_x(a, y)


def fit_edges_linear(x, data, len_fit) -> tuple[float, float, float, float]:
    """Fit :math:`y = mx + n` to each edge of the data.

    Parameters
    ----------
    x
        The x-coordinates of the data points.
    data
        The y-coordinates of the data points.
    len_fit
        The number of data points to use for fitting the edges.

    Returns
    -------
    n0, m0
        The coefficients of the linear fit for the left edge.
    n1, m1
        The coefficients of the linear fit for the right edge.
    """
    na_idx = np.isnan(np.asarray(data))
    xv = np.array(x[~na_idx], dtype=np.float64)
    yv = np.array(data[~na_idx], dtype=np.float64)
    n0, m0 = fit_poly_jit(xv[:len_fit], yv[:len_fit], deg=1)
    n1, m1 = fit_poly_jit(xv[-len_fit:], yv[-len_fit:], deg=1)
    return n0, m0, n1, m1


def _get_edges(x, y, fraction=0.2):
    """
    Get the edges of the input arrays based on a given fraction.

    Primarily used to make initial guesses for backgrounds.

    Parameters
    ----------
    x
        The input x-values.
    y
        The input y-values.
    fraction
        The fraction of the array length to consider as edges. Defaults to 0.2.

    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if fraction < 0 or fraction > 0.5:
        raise ValueError("Fraction must be between 0 and 0.5")
    if len(x) < 10:
        raise ValueError("Array must have at least 10 elements")

    num = max(5, round(len(x) * fraction))

    return np.r_[x[:num], x[-num:]], np.r_[y[:num], y[-num:]]


class FermiEdgeModel(lmfit.Model):
    """Model for fitting a Fermi edge with a linear background.

    The model function is a Fermi-dirac function with linear background above and below
    the fermi level, convolved with a gaussian kernel.

    See Also
    --------
    FermiDiracModel
        A model that does not include a linear background.
    """

    @staticmethod
    def _lin_broad_fd(
        x,
        center=0.0,
        temp=30.0,
        resolution=0.02,
        back0=0.0,
        back1=0.0,
        dos0=1.0,
        dos1=0.0,
    ):
        return fermi_dirac_linbkg_broad(
            x, center, temp, resolution, back0, back1, dos0, dos1
        )

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "FermiEdgeModel")
        super().__init__(self._lin_broad_fd, **kwargs)
        self.set_param_hint("temp", min=0.0)
        self.set_param_hint("resolution", min=0.0)

    def guess(self, data, x, **kwargs):
        pars = self.make_params()

        len_fit = max(round(len(x) * 0.05), 10)
        dos0, dos1, back0, back1 = fit_edges_linear(x, data, len_fit)

        temp = None
        if isinstance(data, xr.DataArray):
            temp = data.qinfo.get_value("sample_temp")
        if temp is None:
            temp = 30.0

        smoothed_deriv = scipy.ndimage.gaussian_filter1d(
            data, 0.2 * len(x), mode="nearest", order=1
        )
        efermi = float(x[np.argmin(smoothed_deriv)])

        pars[f"{self.prefix}center"].set(
            value=efermi, min=np.asarray(x).min(), max=np.asarray(x).max()
        )
        pars[f"{self.prefix}back0"].set(value=back0)
        pars[f"{self.prefix}back1"].set(value=back1)
        pars[f"{self.prefix}dos0"].set(value=dos0)
        pars[f"{self.prefix}dos1"].set(value=dos1)
        pars[f"{self.prefix}temp"].set(value=float(temp))
        pars[f"{self.prefix}resolution"].set(value=0.02)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class FermiDiracModel(lmfit.Model):
    r"""Model that represents a Fermi-Dirac distribution convolved with a Gaussian.

    The model function is given by

    .. math::

        I(\omega) = \left\{\frac{1}{1 + e^{(\omega-\omega_0)/k_B T}}\right\} \otimes
        g(\sigma)

    where :math:`\omega` is the binding energy, :math:`\omega_0` is the center,
    :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature, and
    :math:`g(\sigma)` is a Gaussian kernel with standard deviation :math:`\sigma`. Note
    that the resolution parameter is not the standard deviation of the Gaussian, but
    rather the full width at half maximum (FWHM) of the Gaussian. The relationship is
    given by :math:`\text{FWHM} = 2\sqrt{2\ln(2)}\sigma`.

    See Also
    --------
    FermiEdgeModel
        A model that includes a linear background.
    """

    @staticmethod
    def _broadFermiDirac(x, center=0.0, temp=30.0, resolution=0.02):
        return fermi_dirac_broad(x, center, temp, resolution)

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "FermiDiracModel")
        super().__init__(self._broadFermiDirac, **kwargs)
        self.set_param_hint("temp", min=0.0)
        self.set_param_hint("resolution", min=0.0)

    def guess(self, data, x, **kwargs):
        pars = self.make_params()

        temp = None
        if isinstance(data, xr.DataArray):
            temp = data.qinfo.get_value("sample_temp")
        if temp is None:
            temp = 30.0

        smoothed_deriv = scipy.ndimage.gaussian_filter1d(
            data, 0.2 * len(x), mode="nearest", order=1
        )
        efermi = float(x[np.argmin(smoothed_deriv)])

        pars[f"{self.prefix}center"].set(
            value=efermi, min=np.asarray(x).min(), max=np.asarray(x).max()
        )
        pars[f"{self.prefix}temp"].set(value=float(temp))
        pars[f"{self.prefix}resolution"].set(value=0.02)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class StepEdgeModel(lmfit.Model):
    @staticmethod
    def _step_linbkg_broad(
        x, center=0.0, sigma=0.02, back0=0.0, back1=0.0, dos0=1.0, dos1=0.0
    ):
        return step_linbkg_broad(x, center, sigma, back0, back1, dos0, dos1)

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "StepEdgeModel")
        super().__init__(self._step_linbkg_broad, **kwargs)
        self.set_param_hint("sigma", min=0.0)

    def guess(self, data, x, **kwargs):
        pars = self.make_params()

        len_fit = max(round(len(x) * 0.05), 10)
        dos0, dos1, back0, back1 = fit_edges_linear(x, data, len_fit)
        efermi = x[
            np.argmin(np.gradient(scipy.ndimage.gaussian_filter1d(data, 0.2 * len(x))))
        ]

        pars[f"{self.prefix}center"].set(value=float(efermi))
        pars[f"{self.prefix}back0"].set(value=back0)
        pars[f"{self.prefix}back1"].set(value=back1)
        pars[f"{self.prefix}dos0"].set(value=dos0)
        pars[f"{self.prefix}dos1"].set(value=dos1)
        pars[f"{self.prefix}sigma"].set(value=0.02)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __doc__ = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class PolynomialModel(lmfit.Model):
    def __init__(self, degree=9, **kwargs) -> None:
        kwargs.setdefault("name", f"Poly{degree}")
        super().__init__(PolynomialFunction(degree), **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        if x is None:
            pars["c0"].set(value=float(data.mean()))
            for i in range(1, self.func.degree + 1):
                pars[f"{self.prefix}c{i}"].set(value=0.0)
        else:
            out = np.polyfit(x, data, self.func.degree)
            for i, coef in enumerate(out[::-1]):
                pars[f"{self.prefix}c{i}"].set(value=coef)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __doc__ = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class MultiPeakModel(lmfit.Model):
    def __init__(
        self,
        npeaks: int = 1,
        peak_shapes: list[str] | str | None = None,
        *,
        fd: bool = True,
        background: typing.Literal[
            "none", "constant", "linear", "polynomial"
        ] = "linear",
        degree: int = 2,
        convolve: bool = True,
        oversample: int = 3,
        segmented: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("name", f"{npeaks}Peak")
        super().__init__(
            MultiPeakFunction(
                npeaks,
                peak_shapes=peak_shapes,
                fd=fd,
                background=background,
                degree=degree,
                convolve=convolve,
                oversample=oversample,
                segmented=segmented,
            ),
            **kwargs,
        )

        for i in range(self.func.npeaks):
            for name, kwargs in self.func.peak_param_hints(i, self.prefix).items():
                self.set_param_hint(name, **kwargs)

        if self.func.fd:
            self.set_param_hint("temp", min=0.0)

        if self.func.convolve:
            self.set_param_hint("resolution", min=0.0)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        if x is None:
            x = np.arange(len(data), dtype=float)
        data = np.asarray(data)
        x = np.asarray(x, dtype=float)

        # Drop non-finite values so guesses don't propagate NaNs/Infs.
        mask = np.isfinite(data) & np.isfinite(x)
        if not np.any(mask):
            return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)
        data = data[mask]
        x = x[mask]

        # Ensure monotonic x for edge sampling and peak-finding.
        order = np.argsort(x)
        x, data = x[order], data[order]

        if self.func.fd:
            # Use average intensity above EF as a simple initial offset guess.
            pars[f"{self.prefix}offset"].set(value=float(data[x >= 0].mean()))

        # Sample both ends to estimate a background robustly.
        xc, yc = _get_edges(x, data, fraction=0.2)
        if self.func.background == "constant":
            poly = PolynomialModel(0).guess(yc, xc)
            pars[f"{self.prefix}const_bkg"].set(poly["c0"].value)

        elif self.func.background == "linear":
            poly = PolynomialModel(1).guess(yc, xc)
            pars[f"{self.prefix}const_bkg"].set(poly["c0"].value)
            pars[f"{self.prefix}lin_bkg"].set(poly["c1"].value)

        elif self.func.background == "polynomial":
            poly = PolynomialModel(self.func.bkg_degree).guess(yc, xc)
            for i in range(self.func.bkg_degree + 1):
                pars[f"{self.prefix}c{i}"].set(poly[f"c{i}"].value)

        xrange = float(x.max() - x.min())

        # Subtract estimated background to isolate peaks for initialization.
        baseline = np.zeros_like(data, dtype=float)
        if self.func.background == "constant":
            baseline = float(pars[f"{self.prefix}const_bkg"].value)
        elif self.func.background == "linear":
            baseline = (
                float(pars[f"{self.prefix}const_bkg"].value)
                + float(pars[f"{self.prefix}lin_bkg"].value) * x
            )
        elif self.func.background == "polynomial":
            coeffs = [
                float(pars[f"{self.prefix}c{i}"].value)
                for i in range(self.func.bkg_degree + 1)
            ]
            baseline = np.zeros_like(data, dtype=float)
            for i, coef in enumerate(coeffs):
                baseline = baseline + coef * x**i

        y = data - baseline

        # Light smoothing to reduce noise-driven local maxima.
        if y.size >= 5:
            window = max(3, int(0.02 * y.size))
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window, dtype=float) / window
            y_smooth = np.convolve(y, kernel, mode="same")
        else:
            y_smooth = y

        # Candidate peaks: simple local-max test, then rank by height.
        peak_idx = (
            np.where(
                (y_smooth[1:-1] > y_smooth[:-2]) & (y_smooth[1:-1] > y_smooth[2:])
            )[0]
            + 1
        )
        peak_idx = peak_idx[np.argsort(y_smooth[peak_idx])[::-1]]

        # Enforce a minimum separation so multiple peaks don't collapse to one feature.
        min_sep = max(1, int(len(x) / max(self.func.npeaks * 2, 1)))
        chosen = []
        for idx in peak_idx:
            if all(abs(idx - c) >= min_sep for c in chosen):
                chosen.append(int(idx))
            if len(chosen) >= self.func.npeaks:
                break

        # If not enough peaks found, seed centers approximately evenly across range.
        if len(chosen) < self.func.npeaks:
            fallback = np.linspace(x.min(), x.max(), self.func.npeaks + 2)[1:-1]
            for val in fallback:
                if len(chosen) >= self.func.npeaks:
                    break
                chosen.append(int(np.argmin(np.abs(x - val))))

        height_default = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0
        width_default = 0.1 * xrange if np.isfinite(xrange) else 1.0

        for i in range(self.func.npeaks):
            idx = chosen[i] if i < len(chosen) else int(np.argmax(y_smooth))
            center = float(x[idx])
            height = float(y[idx]) if np.isfinite(y[idx]) else height_default
            width = width_default

            # Estimate width via an approximate FWHM on the smoothed trace.
            peak_height = float(y_smooth[idx])
            if np.isfinite(peak_height) and peak_height > 0:
                half_max = peak_height * 0.5
                left = idx
                while left > 0 and y_smooth[left] > half_max:
                    left -= 1
                right = idx
                while right < y_smooth.size - 1 and y_smooth[right] > half_max:
                    right += 1
                if left != right:
                    x_left = np.interp(
                        half_max,
                        [y_smooth[left], y_smooth[left + 1]],
                        [x[left], x[left + 1]],
                    )
                    x_right = np.interp(
                        half_max,
                        [y_smooth[right - 1], y_smooth[right]],
                        [x[right - 1], x[right]],
                    )
                    width = float(abs(x_right - x_left))

            pars[f"{self.prefix}p{i}_center"].set(value=center)
            if self.func._peak_shapes[i] == "voigt":
                pars[f"{self.prefix}p{i}_gamma"].set(value=width / 2)
                pars[f"{self.prefix}p{i}_amplitude"].set(value=height * width)
            else:
                pars[f"{self.prefix}p{i}_height"].set(value=height)
                pars[f"{self.prefix}p{i}_width"].set(value=width)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def eval_components(self, params=None, **kwargs) -> dict[str, np.ndarray]:
        key = self._prefix
        if len(key) < 1:
            key = self._name

        if params is not None:
            kwargs = kwargs | params.valuesdict()

        # Coerce into numpy arrays
        for k in list(kwargs.keys()):
            if np.iterable(kwargs[k]):
                kwargs[k] = np.asarray(kwargs[k])

        fargs = self.make_funcargs(params, kwargs)

        out = {}
        for i in range(self.func.npeaks):
            out[f"{key}_p{i}"] = self.func.eval_peak(i, **fargs)

        if self.func.background != "none":
            out[f"{key}_bkg"] = self.func.eval_bkg(**fargs)

        if self.func.fd:
            out[f"{key}_fd"] = self.func.eval_fd(**fargs)

        return out

    __doc__ = (
        str(MultiPeakFunction.__doc__)
        + "**kwargs\n        Additional keyword arguments to be passed to the "
        ":class:`lmfit.model.Model` constructor."
    )
    guess.__doc__ = COMMON_GUESS_DOC


class FermiEdge2dModel(lmfit.Model):
    r"""A 2D model for a polynomial Fermi edge with a linear density of states.

    The model function can be written as

    .. math::

        I = \left\{(a\omega + b)\left[1 + \exp\left(\frac{\omega - \sum_{i = 0}^{n} c_i
        \alpha^i}{k_B T}\right)\right]^{-1} + c\right\}\otimes g(\sigma)

    for a :math:`n` th degree polynomial edge with coefficients :math:`c_i` with a
    linear density of states described by :math:`a\omega+b` with a constant background
    :math:`c` convolved with a gaussian, where :math:`\omega` is the binding energy and
    :math:`\alpha` is the detector angle.

    """

    def __init__(self, degree: int = 2, **kwargs) -> None:
        kwargs.setdefault("independent_vars", ["eV", "alpha"])
        super().__init__(FermiEdge2dFunction(degree), **kwargs)
        self.name = f"FermiEdge2dModel (deg {degree})"

        self.set_param_hint("temp", min=0.0)
        self.set_param_hint("resolution", min=0.0)

    def guess(self, data, eV, alpha, **kwargs):
        pars = self.make_params()
        for i in range(self.func.poly.degree + 1):
            pars[f"{self.prefix}c{i}"].set(value=0.0)

        if isinstance(data, xr.DataArray):
            avg_edc = data.mean("alpha").values
        elif data.shape == eV.shape:
            shape, _, eV = _infer_meshgrid_shape(np.ascontiguousarray(eV.ravel()))
            shape_, ax_alpha, alpha = _infer_meshgrid_shape(
                np.ascontiguousarray(alpha.ravel())
            )
            if shape != shape_:
                raise ValueError(
                    "eV and alpha seems to be meshgrids of different shapes."
                )
            avg_edc = data.reshape(shape).mean(axis=ax_alpha)
        else:
            avg_edc = data.reshape(len(eV), len(alpha)).mean(axis=-1)

        len_fit = max(round(len(eV) * 0.05), 10)
        dos0, dos1 = fit_poly_jit(
            np.asarray(eV[:len_fit], dtype=np.float64),
            np.asarray(avg_edc[:len_fit], dtype=np.float64),
            deg=1,
        )

        pars[f"{self.prefix}const_bkg"].set(value=dos0)
        pars[f"{self.prefix}lin_bkg"].set(value=dos1)

        if isinstance(data, xr.DataArray):
            temp = data.qinfo.get_value("sample_temp")
            if temp is not None:
                pars[f"{self.prefix}temp"].set(value=float(temp))

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def fit(self, data, *args, **kwargs):
        if isinstance(data, xr.DataArray):
            data = data.transpose("eV", "alpha").values
        # Ensure flat fit
        return super().fit(data.ravel(), *args, **kwargs)

    guess.__doc__ = COMMON_GUESS_DOC.replace("x : ", "eV, alpha : ")
    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC.replace("['x']", "['eV', 'alpha']")


class BCSGapModel(lmfit.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(bcs_gap, **kwargs)
        self.set_param_hint("a", min=0.0)
        self.set_param_hint("tc", min=0.0)

    __doc__ = bcs_gap.__doc__
    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class DynesModel(lmfit.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(dynes, **kwargs)
        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("delta", min=0.0)

    __doc__ = dynes.__doc__
    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class TLLModel(lmfit.Model):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "TLLModel")
        super().__init__(tll, **kwargs)
        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("temp", min=0.0)
        self.set_param_hint("resolution", min=0.0)

    __doc__ = tll.__doc__
    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC


class SymmetrizedGapModel(lmfit.Model):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("name", "SymmetrizedGapModel")
        super().__init__(sc_spectral_function, **kwargs)
        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("gamma1", min=0.0)
        self.set_param_hint("gamma0", min=0.0, value=0.0, vary=False)
        self.set_param_hint("delta", min=0.0)
        self.set_param_hint("resolution", min=0.0)

    __doc__ = sc_spectral_function.__doc__
    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC
