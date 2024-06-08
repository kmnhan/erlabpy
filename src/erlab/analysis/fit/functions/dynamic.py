"""Class-based dynamic functions for fitting.

These functions are not limited to a single function form, and can be used to create
complex models.
"""

__all__ = [
    "DynamicFunction",
    "FermiEdge2dFunction",
    "MultiPeakFunction",
    "PolynomialFunction",
    "get_args_kwargs",
]
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal, TypedDict, no_type_check

import numpy as np
import numpy.typing as npt
import xarray as xr

from erlab.analysis.fit.functions.general import (
    TINY,
    do_convolve,
    do_convolve_2d,
    fermi_dirac,
    gaussian_wh,
    lorentzian_wh,
)
from erlab.constants import kb_eV


class PeakArgs(TypedDict):
    args: list[str]
    kwargs: dict[str, Any]


def get_args_kwargs(func: Callable) -> tuple[list[str], dict[str, Any]]:
    """Get all argument names and default values from a function signature.

    Parameters
    ----------
    func
        The function to inspect.

    Returns
    -------
    args : list of str
        A list of argument names with no default value.
    args_default : dict
        A dictionary of keyword arguments with their default values.

    Note
    ----
    This function does not support function signatures containing varargs.

    Example
    -------

    >>> def my_func(a, b=10):
    ...     pass
    >>> get_args_kwargs(my_func)
    (['a'], {'b': 10})

    """
    args = []
    args_default = {}
    sig = inspect.signature(func)
    for fnam, fpar in sig.parameters.items():
        if fpar.kind == fpar.VAR_POSITIONAL or fpar.kind == fpar.VAR_KEYWORD:
            raise ValueError(f"varargs '*{fnam}' is not supported")
        elif fpar.default == fpar.empty:
            args.append(fnam)
        else:
            args_default[fnam] = fpar.default

    return args, args_default


def get_args_kwargs_dict(func: Callable) -> PeakArgs:
    args, kwargs = get_args_kwargs(func)
    return {"args": args, "kwargs": kwargs}


class DynamicFunction:
    """Base class for dynamic functions.

    Dynamic functions exploits the way `lmfit` handles asteval functions in
    `lmfit.Model._parse_params`.
    """

    @property
    def __name__(self) -> str:
        return str(self.__class__.__name__)

    @property
    def argnames(self) -> list[str]:
        return ["x"]

    @property
    def kwargs(self) -> dict[str, int | float]:
        return {}

    @no_type_check
    def __call__(self, **kwargs):
        raise NotImplementedError("Must be overloaded in child classes")


class PolynomialFunction(DynamicFunction):
    """A callable class for a arbitrary degree polynomial.

    Parameters
    ----------
    degree
        The degree of the polynomial.

    """

    def __init__(self, degree: int = 1) -> None:
        super().__init__()
        self.degree = degree

    @property
    def argnames(self) -> list[str]:
        return ["x"] + [f"c{i}" for i in range(self.degree + 1)]

    def __call__(self, x, *coeffs: float, **params):
        if len(coeffs) != self.degree + 1:
            coeffs = tuple(params[f"c{d}"] for d in range(self.degree + 1))
        if isinstance(x, np.ndarray):
            return np.polynomial.polynomial.polyval(x, coeffs)
        else:
            coeffs_xr = xr.DataArray(
                np.asarray(coeffs), coords={"degree": np.arange(self.degree + 1)}
            )
            return xr.polyval(x, coeffs_xr)


class MultiPeakFunction(DynamicFunction):
    """A callable class for a multi-peak model.

    Parameters
    ----------
    npeaks
        The number of peaks to fit.
    peak_shapes
        The shape(s) of the peaks in the model. If a list of strings is provided, each
        string represents the shape of a peak. If a single string is provided, it will
        be split by spaces to create a list of peak shapes. If not provided, the default
        peak shape will be used for all peaks.
    fd
        Flag indicating whether the model should be multiplied by the Fermi-Dirac
        distribution. This adds three parameters to the model: `efermi`, `temp`, and
        `offset`, each corresponding to the Fermi level, temperature in K, and constant
        background.
    background
        The type of background to include in the model. The options are: ``'constant'``,
        ``'linear'``, ``'polynomial'``, or ``'none'``. If ``'constant'``, adds a
        ``const_bkg`` parameter. If ``'linear'``, adds a ``lin_bkg`` parameter and a
        ``const_bkg`` parameter. If ``'polynomial'``, adds  ``c0``, ``c1``, ...
        corresponding to the polynomial coefficients. The polynomial degree can be
        specified with `degree`. If ``'none'``, no background is added.
    degree
        The degree of the polynomial background. Only used if `background` is
        ``'polynomial'``. Default is 2.
    convolve
        Flag indicating whether the model should be convolved with a gaussian kernel. If
        `True`, adds a `resolution` parameter to the model, corresponding to the FWHM of
        the gaussian kernel.

    """

    PEAK_SHAPES: ClassVar[dict[Callable, list[str]]] = {
        lorentzian_wh: ["lorentzian", "lor", "l"],
        gaussian_wh: ["gaussian", "gauss", "g"],
    }

    DEFAULT_PEAK: str = "lorentzian"

    def __init__(
        self,
        npeaks: int,
        peak_shapes: list[str] | str | None = None,
        fd: bool = True,
        background: Literal["constant", "linear", "polynomial", "none"] = "linear",
        degree: int = 2,
        convolve: bool = True,
    ):
        super().__init__()
        self.npeaks = npeaks
        self.fd = fd
        self.convolve = convolve
        self.background = background

        if self.background == "polynomial":
            self.bkg_degree = degree

        if peak_shapes is None:
            peak_shapes = [self.DEFAULT_PEAK] * self.npeaks

        if isinstance(peak_shapes, str):
            peak_shapes = peak_shapes.split(" ")

        if len(peak_shapes) == 1:
            peak_shapes = peak_shapes * self.npeaks

        elif len(peak_shapes) != self.npeaks:
            raise ValueError("Number of peaks does not match given peak shapes")

        self._peak_shapes = peak_shapes

        self._peak_funcs: list[Callable] = []
        for name in self._peak_shapes:
            for fcn, aliases in self.PEAK_SHAPES.items():
                if name in aliases:
                    self._peak_funcs.append(fcn)

        if len(self._peak_funcs) != self.npeaks:
            raise ValueError("Invalid peak name")

    @functools.cached_property
    def peak_all_args(self) -> dict[Callable, PeakArgs]:
        res: dict[Callable, PeakArgs] = {}
        for func in self.PEAK_SHAPES:
            res[func] = get_args_kwargs_dict(func)
        return res

    @functools.cached_property
    def peak_argnames(self) -> dict[Callable, list[str]]:
        res = {}
        for func in self.PEAK_SHAPES:
            res[func] = self.peak_all_args[func]["args"][1:] + list(
                dict(self.peak_all_args[func]["kwargs"]).keys()
            )
        return res

    @property
    def peak_funcs(self) -> Sequence[Callable]:
        return self._peak_funcs

    @property
    def argnames(self) -> list[str]:
        args = ["x"]
        for i, func in enumerate(self.peak_funcs):
            args += [f"p{i}_{arg}" for arg in self.peak_all_args[func]["args"][1:]]
        return args

    @property
    def kwargs(self):
        kws: list[tuple[str, float]] = []

        if self.background == "constant" or self.background == "linear":
            kws.append(("const_bkg", 0.0))
        if self.background == "linear":
            kws.append(("lin_bkg", 0.0))
        elif self.background == "polynomial":
            kws += [(f"c{i}", 0.0) for i in range(self.bkg_degree + 1)]

        if self.fd:
            kws += [
                ("efermi", 0.0),  # fermi level
                ("temp", 30.0),  # temperature
                ("offset", 0.0),
            ]

        if self.convolve:
            kws += [("resolution", 0.02)]

        for i, func in enumerate(self.peak_funcs):
            for arg, val in dict(self.peak_all_args[func]["kwargs"]).items():
                kws.append((f"p{i}_{arg}", val))
        return kws

    def sigma_expr(self, index: int, prefix: str) -> str | None:
        if self._peak_funcs[index] == gaussian_wh:
            return f"{prefix}p{index}_width / (2 * sqrt(2 * log(2)))"
        elif self._peak_funcs[index] == lorentzian_wh:
            return f"{prefix}p{index}_width / 2"
        else:
            return None

    def amplitude_expr(self, index: int, prefix: str) -> str | None:
        if self._peak_funcs[index] == gaussian_wh:
            return f"{prefix}p{index}_height * {prefix}p{index}_sigma / sqrt(2 * pi)"
        elif self._peak_funcs[index] == lorentzian_wh:
            return f"{prefix}p{index}_height * {prefix}p{index}_sigma * pi"
        else:
            return None

    def eval_peak(self, index: int, x, **params):
        return self.peak_funcs[index](
            x,
            **{
                k[3:]: v
                for k, v in params.items()
                if k.startswith(f"p{index}") and not k.endswith(("sigma", "amplitude"))
            },
        )

    def eval_bkg(self, x, **params):
        match self.background:
            case "constant":
                return 0.0 * x + params["const_bkg"]
            case "linear":
                return params["lin_bkg"] * x + params["const_bkg"]
            case "polynomial":
                return PolynomialFunction(self.bkg_degree)(
                    x, **{f"c{i}": params[f"c{i}"] for i in range(self.bkg_degree + 1)}
                )
            case "none":
                return 0.0 * x

    def pre_call(self, x, **params):
        x = np.asarray(x).copy()
        y = np.zeros_like(x)

        for i, func in enumerate(self.peak_funcs):
            y += func(
                x, **{arg: params[f"p{i}_{arg}"] for arg in self.peak_argnames[func]}
            )

        y += self.eval_bkg(x, **params)

        if self.fd:
            y *= fermi_dirac(x, center=params["efermi"], temp=params["temp"])
            y += params["offset"]

        return y

    def __call__(self, x, **params):
        if isinstance(x, xr.DataArray):
            return x * 0.0 + self.__call__(x.values, **params)

        if self.convolve:
            if "resolution" not in params:
                raise TypeError(
                    "Missing parameter `resolution` required for convolution"
                )
            return do_convolve(x, self.pre_call, **params)
        else:
            return self.pre_call(x, **params)


class FermiEdge2dFunction(DynamicFunction):
    def __init__(self, degree: int = 1):
        super().__init__()
        self.poly = PolynomialFunction(degree)

    @property
    def argnames(self) -> list[str]:
        return ["eV", "alpha"] + self.poly.argnames[1:]

    @property
    def kwargs(self) -> list[tuple[str, float]]:
        return [
            ("temp", 30.0),
            ("lin_bkg", 0.0),
            ("const_bkg", 1.0),
            ("offset", 0.0),
            ("resolution", 0.02),
        ]

    def pre_call(self, eV, alpha, **params):
        center = self.poly(
            np.asarray(alpha),
            *[params.pop(f"c{i}") for i in range(self.poly.degree + 1)],
        )
        return (params["const_bkg"] - params["offset"] + params["lin_bkg"] * eV) / (
            1 + np.exp((1.0 * eV - center) / max(TINY, params["temp"] * kb_eV))
        ) + params["offset"]

    def __call__(
        self,
        eV: npt.NDArray[np.float64] | xr.DataArray,
        alpha: npt.NDArray[np.float64] | xr.DataArray,
        **params,
    ):
        if isinstance(eV, xr.DataArray) and isinstance(alpha, xr.DataArray):
            out = eV * alpha * 0.0
            return out + self.__call__(eV.values, alpha.values, **params).reshape(
                out.shape
            )
        if isinstance(eV, xr.DataArray):
            eV = eV.values
        if isinstance(alpha, xr.DataArray):
            alpha = alpha.values
        if "resolution" not in params:
            raise TypeError("Missing parameter `resolution` required for convolution")
        return do_convolve_2d(eV, alpha, self.pre_call, **params).ravel()
