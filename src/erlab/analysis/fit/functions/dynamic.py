"""Class-based dynamic functions for fitting.

These functions are not limited to a single function form, and can be used to create
complex models.
"""

__all__ = [
    "get_args_kwargs",
    "DynamicFunction",
    "PolyFunc",
    "MultiPeakFunction",
    "FermiEdge2dFunc",
]
import functools
import inspect
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import xarray as xr

from erlab.analysis.fit.functions.general import (
    TINY,
    do_convolve,
    do_convolve_y,
    fermi_dirac,
    gaussian_wh,
    lorentzian_wh,
)
from erlab.constants import kb_eV


def get_args_kwargs(func) -> tuple[list[str], dict[str, object]]:
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


class DynamicFunction:
    """Base class for dynamic functions.

    Dynamic functions exploits the way `lmfit` handles asteval functions in
    `lmfit.Model._parse_params`.
    """

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    @property
    def argnames(self) -> list[str]:
        return ["x"]

    @property
    def kwargs(self) -> dict[str, int | float]:
        return {}

    def __call__(self, x: npt.NDArray[np.float64], **params) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Must be overloaded in child classes")


class PolyFunc(DynamicFunction):
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

    def __call__(self, x, *coeffs, **params):
        if len(coeffs) != self.degree + 1:
            coeffs = [params[f"c{d}"] for d in range(self.degree + 1)]
        if isinstance(x, np.ndarray):
            return np.polynomial.polynomial.polyval(x, coeffs)
        else:
            coeffs = xr.DataArray(coeffs, coords={"degree": np.arange(self.degree + 1)})
            return xr.polyval(x, coeffs)


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
    convolve
        Flag indicating whether the model should be convolved with a gaussian kernel. If
        `True`, adds a `resolution` parameter to the model, corresponding to the FWHM of
        the gaussian kernel.

    """

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
        for func in self.PEAK_SHAPES:
            res[func] = dict(zip(("args", "kwargs"), get_args_kwargs(func)))
        return res

    @functools.cached_property
    def peak_argnames(self) -> dict[Callable, list[str]]:
        res = {}
        for func in self.PEAK_SHAPES:
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
