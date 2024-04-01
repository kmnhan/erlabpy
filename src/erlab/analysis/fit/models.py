"""Models for fitting data."""

__all__ = [
    "ExtendedAffineBroadenedFD",
    "StepEdgeModel",
    "PolynomialModel",
    "MultiPeakModel",
    "FermiEdge2dModel",
]


import lmfit
import numba
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import xarray as xr
from arpes.fits import XModelMixin

from erlab.analysis.fit.functions import (
    fermi_dirac_linbkg_broad,
    step_linbkg_broad,
    PolyFunc,
    MultiPeakFunction,
    FermiEdge2dFunc,
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
    det_ = np.linalg.lstsq(a, b)[0]
    return det_


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
    p = _fit_x(a, y)
    return p


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
    n0, m0 = fit_poly_jit(
        np.array(x[:len_fit], dtype=np.float64),
        np.array(data[:len_fit], dtype=np.float64),
        deg=1,
    )
    n1, m1 = fit_poly_jit(
        np.array(x[-len_fit:], dtype=np.float64),
        np.array(data[-len_fit:], dtype=np.float64),
        deg=1,
    )
    return n0, m0, n1, m1


class ExtendedAffineBroadenedFD(XModelMixin):
    """
    Fermi-dirac function with linear background above and below the fermi level,
    convolved with a gaussian kernel.
    """

    guess_dataarray = True

    @staticmethod
    def LinearBroadFermiDirac(
        x,
        center=0,
        temp=30,
        resolution=0.02,
        back0=1,
        back1=0,
        dos0=1,
        dos1=0,
    ):
        return fermi_dirac_linbkg_broad(
            x, center, temp, resolution, back0, back1, dos0, dos1
        )

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.LinearBroadFermiDirac, **kwargs)
        self.set_param_hint("temp", min=0.0)
        self.set_param_hint("resolution", min=0.0)

    def guess(self, data, x, **kwargs):
        """Make some heuristic guesses."""
        pars = self.make_params()

        len_fit = max(round(len(x) * 0.05), 10)
        dos0, dos1, back0, back1 = fit_edges_linear(x, data, len_fit)
        efermi = x[
            np.argmin(np.gradient(scipy.ndimage.gaussian_filter1d(data, 0.2 * len(x))))
        ]

        temp = 30
        if isinstance(data, xr.DataArray):
            try:
                temp = data.attrs["temp_sample"]
            except KeyError:
                pass

        pars[f"{self.prefix}center"].set(value=efermi, min=x.min(), max=x.max())
        pars[f"{self.prefix}back0"].set(value=back0)
        pars[f"{self.prefix}back1"].set(value=back1)
        pars[f"{self.prefix}dos0"].set(value=dos0)
        pars[f"{self.prefix}dos1"].set(value=dos1)
        pars[f"{self.prefix}temp"].set(value=temp)
        pars[f"{self.prefix}resolution"].set(value=0.02)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC


class StepEdgeModel(XModelMixin):
    guess_dataarray = True

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(step_linbkg_broad, **kwargs)
        self.set_param_hint("sigma", min=0.0)

    def guess(self, data, x, **kwargs):
        """Make some heuristic guesses."""
        pars = self.make_params()

        len_fit = max(round(len(x) * 0.05), 10)
        dos0, dos1, back0, back1 = fit_edges_linear(x, data, len_fit)
        efermi = x[
            np.argmin(np.gradient(scipy.ndimage.gaussian_filter1d(data, 0.2 * len(x))))
        ]

        pars[f"{self.prefix}center"].set(value=efermi)
        pars[f"{self.prefix}back0"].set(value=back0)
        pars[f"{self.prefix}back1"].set(value=back1)
        pars[f"{self.prefix}dos0"].set(value=dos0)
        pars[f"{self.prefix}dos1"].set(value=dos1)
        pars[f"{self.prefix}sigma"].set(value=0.02)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC


class PolynomialModel(XModelMixin):
    def __init__(self, degree=9, **kwargs):
        kwargs.setdefault("name", f"Poly{degree}")
        super().__init__(PolyFunc(degree), **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        if x is None:
            pars["c0"].set(value=data.mean())
            for i in range(1, self.func.degree + 1):
                pars[f"{self.prefix}c{i}"].set(value=0.0)
        else:
            out = np.polyfit(x, data, self.func.degree)
            for i, coef in enumerate(out[::-1]):
                pars[f"{self.prefix}c{i}"].set(value=coef)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC


class MultiPeakModel(XModelMixin):
    """Model for fitting multiple Gaussian or Lorentzian peaks.

    Most input parameters are passed to the
    `erlab.analysis.fit.functions.MultiPeakFunction` constructor.
    """

    def __init__(
        self,
        npeaks: int = 1,
        peak_shapes: list[str] | str | None = None,
        fd: bool = True,
        convolve: bool = True,
        independent_vars=None,
        param_names=None,
        nan_policy="raise",
        prefix="",
        name=None,
        **kws,
    ):
        super().__init__(
            MultiPeakFunction(
                npeaks, peak_shapes=peak_shapes, fd=fd, convolve=convolve
            ),
            independent_vars,
            param_names,
            nan_policy,
            prefix,
            name,
            **kws,
        )

        if self.func.convolve:
            self.set_param_hint("resolution", min=0.0)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        # !TODO: better guesses
        if self.func.fd:
            pars[f"{self.prefix}offset"].set(value=data[x >= 0].mean())

        poly1 = PolynomialModel(1).guess(data, x)
        pars[f"{self.prefix}lin_bkg"].set(poly1["c1"].value)
        pars[f"{self.prefix}const_bkg"].set(poly1["c0"].value)

        # for i, func in enumerate(self.func.peak_funcs):
        # self.func.peak_argnames

        xrange = x.max() - x.min()

        for i in range(self.func.npeaks):  # Number of peaks
            pars[f"{self.prefix}p{i}_center"].set(value=0.0)
            pars[f"{self.prefix}p{i}_height"].set(value=data.mean())
            pars[f"{self.prefix}p{i}_width"].set(value=0.1 * xrange)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC


class FermiEdge2dModel(XModelMixin):
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

    n_dims = 2
    dimension_order = ["eV", "alpha"]
    guess_dataarray = True
    fit_flat = True

    def __init__(
        self,
        degree=2,
        independent_vars=("eV", "alpha"),
        prefix="",
        nan_policy="raise",
        **kwargs,
    ):
        kwargs.update(
            {
                "prefix": prefix,
                "nan_policy": nan_policy,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(FermiEdge2dFunc(degree), **kwargs)
        self.name = f"FermiEdge2dModel (deg {degree})"

    def guess(self, data, eV, alpha, negative=False, **kwargs):
        pars = self.make_params()
        for i in range(self.func.poly.degree + 1):
            pars[f"{self.prefix}c{i}"].set(value=0.0)

        avg_edc = data.mean("alpha").values
        len_fit = max(round(len(eV) * 0.05), 10)
        dos0, dos1 = fit_poly_jit(
            np.asarray(eV[:len_fit], dtype=np.float64),
            np.asarray(avg_edc[:len_fit], dtype=np.float64),
            deg=1,
        )
        pars[f"{self.prefix}const_bkg"].set(value=dos0)
        pars[f"{self.prefix}lin_bkg"].set(value=dos1)
        pars[f"{self.prefix}temp"].set(value=data.attrs["temp_sample"])

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def guess_fit(self, *args, **kwargs):
        return super().guess_fit(*args, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC.replace("['x']", "['eV', 'alpha']")
    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC
