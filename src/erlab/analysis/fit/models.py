__all__ = [
    "ExtendedAffineBroadenedFD",
    "LinearBroadStep",
    "PolynomialModel",
    "MultiPeakModel",
    "FermiEdge2dModel",
]

import functools
import inspect
from typing import Callable

import lmfit
import numba
import numpy as np
import numpy.typing as npt
import scipy.ndimage
import xarray as xr
from arpes.fits import XModelMixin

from erlab.analysis.fit.functions import (
    TINY,
    do_convolve,
    do_convolve_y,
    fermi_dirac,
    fermi_dirac_linbkg_broad,
    gaussian_wh,
    lorentzian_wh,
    step_linbkg_broad,
)
from erlab.constants import kb_eV


def get_args_kwargs(func) -> tuple[list[str], dict[str, int | float]]:
    pos_args = []
    kw_args = {}
    sig = inspect.signature(func)
    for fnam, fpar in sig.parameters.items():
        if fpar.kind == fpar.POSITIONAL_OR_KEYWORD:
            if fpar.default == fpar.empty:
                pos_args.append(fnam)
            else:
                kw_args[fnam] = fpar.default
        elif fpar.kind == fpar.VAR_POSITIONAL:
            raise ValueError(f"varargs '*{fnam}' is not supported")
    return pos_args, kw_args


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
def fit_poly_jit(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    return p


def fit_edges_linear(x, data, len_fit):
    """Fit y = mx + n to each edge of the data."""
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
                temp = data.S.temp
            except AttributeError:
                pass

        pars[f"{self.prefix}center"].set(value=efermi)
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


class DynamicFunction:
    def __init__(self):
        self.__setattr__("__name__", self.__class__.__name__)

    @property
    def argnames(self) -> list[str]:
        return ["x"]

    @property
    def kwargs(self) -> dict[str, int | float]:
        return dict()

    def __call__(self, x: npt.NDArray[np.float64], **params) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Must be overloaded in child classes")


class PolyFunc(DynamicFunction):
    def __init__(self, degree=1) -> None:
        super().__init__()
        self.degree = degree

    @property
    def argnames(self) -> list[str]:
        return ["x"] + [f"c{i}" for i in range(self.degree + 1)]

    def __call__(self, x, *coeffs, **params):
        if len(coeffs) != self.degree + 1:
            coeffs = [params[f"c{d}"] for d in range(self.degree + 1)]
        if isinstance(x, np.ndarray):
            return np.polynomial.polynomial.polyval(x, coeffs)
        else:
            coeffs = xr.DataArray(coeffs, coords={"degree": np.arange(self.degree + 1)})
            return xr.polyval(x, coeffs)


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


class MultiPeakFunction(DynamicFunction):
    PEAK_SHAPES: dict[Callable, list[str]] = {
        lorentzian_wh: ["lorentzian", "lor", "l"],
        gaussian_wh: ["gaussian", "gauss", "g"],
    }

    DEFAULT_PEAK: str = "lorentzian"

    def __init__(
        self,
        npeaks: int,
        peak_shapes: list[str] | str | None = None,
        fd: bool = True,
        convolve: bool = True,
    ):
        """

        space-separated string 'l l g l'

        """
        super().__init__()
        self.npeaks = npeaks
        self.fd = fd
        self.convolve = convolve

        if peak_shapes is None:
            peak_shapes = [self.DEFAULT_PEAK] * self.npeaks
        if isinstance(peak_shapes, str):
            peak_shapes = peak_shapes.split(" ")

        if len(peak_shapes) == 1:
            peak_shapes = peak_shapes * self.npeaks
        elif len(peak_shapes) != self.npeaks:
            raise ValueError("Number of peaks does not match given peak shapes")

        self._peak_shapes = peak_shapes

        self._peak_funcs = [None] * self.npeaks
        for i, name in enumerate(self._peak_shapes):
            for fcn, aliases in self.PEAK_SHAPES.items():
                if name in aliases:
                    self._peak_funcs[i] = fcn

        if None in self._peak_funcs:
            raise ValueError("Invalid peak name")

    @functools.cached_property
    def peak_all_args(self) -> dict[Callable, dict[str, list | dict]]:
        res = {}
        for func in self.PEAK_SHAPES.keys():
            res[func] = {
                k: v for k, v in zip(("args", "kwargs"), get_args_kwargs(func))
            }
        return res

    @functools.cached_property
    def peak_argnames(self) -> dict[Callable, list[str]]:
        res = {}
        for func in self.PEAK_SHAPES.keys():
            res[func] = self.peak_all_args[func]["args"][1:] + list(
                self.peak_all_args[func]["kwargs"].keys()
            )
        return res

    @property
    def peak_funcs(self) -> list[Callable]:
        return self._peak_funcs

    @property
    def argnames(self) -> list[str]:
        args = ["x"]
        for i, func in enumerate(self.peak_funcs):
            args += [f"p{i}_{arg}" for arg in self.peak_all_args[func]["args"][1:]]
        return args

    @property
    def kwargs(self):
        kws = [
            ("lin_bkg", 0),
            ("const_bkg", 0),
        ]
        if self.fd:
            kws += [
                ("efermi", 0.0),  # fermi level
                ("temp", 30),  # temperature
                ("offset", 0),
            ]
        if self.convolve:
            kws += [("resolution", 0.02)]

        for i, func in enumerate(self.peak_funcs):
            for arg, val in self.peak_all_args[func]["kwargs"].items():
                kws.append((f"p{i}_{arg}", val))
        return kws

    def eval_peak(self, index: int, x: npt.NDArray[np.float64], **params: dict):
        return self.peak_funcs[index](
            x, **{k[3:]: v for k, v in params.items() if k.startswith(f"p{index}")}
        )

    def pre_call(
        self, x: npt.NDArray[np.float64], **params: dict
    ) -> npt.NDArray[np.float64]:
        x = np.asarray(x).copy()
        y = np.zeros_like(x)

        for i, func in enumerate(self.peak_funcs):
            y += func(
                x, **{arg: params[f"p{i}_{arg}"] for arg in self.peak_argnames[func]}
            )

        y += params["lin_bkg"] * x + params["const_bkg"]

        if self.fd:
            y *= fermi_dirac(x, center=params["efermi"], temp=params["temp"])
            y += params["offset"]

        return y

    def __call__(
        self, x: npt.NDArray[np.float64], **params: dict
    ) -> npt.NDArray[np.float64]:
        if self.convolve:
            return do_convolve(x, self.pre_call, **params)
        else:
            return self.pre_call(x, **params)


class MultiPeakModel(XModelMixin):
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
        #!TODO: better guesses
        if self.func.fd:
            pars[f"{self.prefix}offset"].set(value=data[x >= 0].mean())

        poly1 = PolynomialModel(1).guess(data, x)
        pars[f"{self.prefix}lin_bkg"].set(poly1["c1"])
        pars[f"{self.prefix}const_bkg"].set(poly1["c0"])

        # for i, func in enumerate(self.func.peak_funcs):
        # self.func.peak_argnames
        # for i in range(self.func.npeaks):  # Number of peaks
        #     pars[f"{self.prefix}c_real_{i}"].set(value=-0.15)
        #     pars[f"{self.prefix}c_imag_{i}"].set(value=-0.01)
        #     for j in range(self.norder):  # Number of order
        #         pars[f"{self.prefix}A_real_{i}_{j}"].set(
        #             value=-data.max() if self.norder == 1 else 0
        #         )
        #         pars[f"{self.prefix}A_imag_{i}_{j}"].set(value=0)
        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lmfit.models.COMMON_INIT_DOC
    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC


class FermiEdge2dFunc(DynamicFunction):
    def __init__(self, degree=1) -> None:
        super().__init__()
        self.poly = PolyFunc(degree)

    @property
    def argnames(self) -> list[str]:
        return ["eV", "alpha"] + self.poly.argnames[1:]

    @property
    def kwargs(self):
        return [
            ("temp", 30.0),
            ("lin_bkg", 0.0),
            ("const_bkg", 1.0),
            ("offset", 0.0),
            ("resolution", 0.02),
        ]

    def pre_call(self, eV, alpha, **params: dict):
        center = self.poly(
            np.asarray(alpha),
            *[params.pop(f"c{i}") for i in range(self.poly.degree + 1)],
        )
        return (params["const_bkg"] - params["offset"] + params["lin_bkg"] * eV) / (
            1 + np.exp((1.0 * eV - center) / max(TINY, params["temp"] * kb_eV))
        ) + params["offset"]

    def __call__(self, eV, alpha, **params: dict):
        return do_convolve_y(eV, alpha, self.pre_call, **params).flatten()


class FermiEdge2dModel(XModelMixin):
    r"""A 2D model for a polynomial Fermi edge with a linear density of states.

    The model function can be written as

    .. math::

        I = \left\{(a\omega + b)\left[1 +
        \exp\left(\frac{\omega - \sum_{i = 0}^{n} c_i \alpha^i}{k_B T}\right)\right]^{-1}
        + c\right\}\otimes g(\sigma)

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
        independent_vars=["eV", "alpha"],
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
        pars[f"{self.prefix}temp"].set(value=data.S.temp)

        return lmfit.models.update_param_vals(pars, self.prefix, **kwargs)

    def guess_fit(self, *args, **kwargs):
        return super().guess_fit(*args, **kwargs)

    __init__.__doc__ = lmfit.models.COMMON_INIT_DOC.replace("['x']", "['eV', 'alpha']")
    guess.__doc__ = lmfit.models.COMMON_GUESS_DOC
