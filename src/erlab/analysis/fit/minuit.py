from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if not importlib.util.find_spec("iminuit"):
    raise ImportError("`erlab.analysis.fit.minuit` requires `iminuit` to be installed.")

import iminuit.cost
import iminuit.util
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray
from iminuit.util import _detect_log_spacing, _smart_sampling

import erlab.plotting.general

if TYPE_CHECKING:
    import lmfit


class LeastSq(iminuit.cost.LeastSquares):
    """A thin wrapper around `iminuit.cost.LeastSquares` that produces better plots."""

    def visualize(
        self, args: npt.ArrayLike, model_points: int | Sequence[float] = 0
    ) -> tuple[
        tuple[npt.NDArray, npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]
    ]:
        if self._ndim > 1:
            raise ValueError("visualize is not implemented for multi-dimensional data")

        plt.grid(visible=True, axis="both")
        x, y, ye = self._masked.T
        plt.errorbar(
            x, y, ye, fmt="o", lw=0.75, ms=3, mfc="w", zorder=2, c="0.4", capsize=0
        )
        if isinstance(model_points, Iterable):
            xm = np.array(model_points)
            ym = self.model(xm, *args)
        elif model_points > 0:
            if _detect_log_spacing(x):
                xm = np.geomspace(x[0], x[-1], model_points)
            else:
                xm = np.linspace(x[0], x[-1], model_points)
            ym = self.model(xm, *args)
        else:
            xm, ym = _smart_sampling(
                lambda x: self.model(x, *args),
                x[0],
                x[-1],
                start=len(x),
            )
        plt.plot(xm, ym, "r-", lw=1, zorder=3)
        return (x, y, ye), (xm, ym)

    visualize.__doc__ = iminuit.cost.LeastSquares.visualize.__doc__


class _TempFig:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h), layout="constrained")

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        plt.close(self.fig)


class Minuit(iminuit.Minuit):
    """`iminuit.Minuit` with additional functionality.

    This class extends the functionality of the `iminuit.Minuit` class by providing a
    convenient method `from_lmfit` to initialize the `Minuit` object from an
    `lmfit.Model` object.

    For more information on the `iminuit` library, see its documentation.

    Examples
    --------
    >>> import lmfit.models
    >>> import numpy as np
    >>> from erlab.analysis.fit.minuit import Minuit

    >>> # Create an lmfit.Model object
    >>> model = lmfit.models.LinearModel()

    >>> # Generate some data
    >>> x = np.linspace(0, 10, 100)
    >>> y = model.eval(x=x, a=2, b=1)
    >>> rng = np.random.default_rng(1)
    >>> y = rng.normal(y, 0.5)

    >>> # Initialize a Minuit object from the lmfit.Model object
    >>> m = Minuit.from_lmfit(model, y, x)

    >>> # Perform the fit
    >>> m.migrad()

    """

    @classmethod
    def from_lmfit(
        cls,
        model: lmfit.Model,
        data: npt.NDArray | xarray.DataArray,
        ivars: npt.NDArray
        | xarray.DataArray
        | Sequence[npt.NDArray | xarray.DataArray],
        yerr: float | npt.NDArray | None = None,
        return_cost: bool = False,
        **kwargs,
    ) -> Minuit | tuple[LeastSq, Minuit]:
        if len(model.independent_vars) == 1:
            if isinstance(ivars, np.ndarray | xarray.DataArray):
                ivars = [ivars]

        x: npt.NDArray | list[npt.NDArray] = [np.asarray(a) for a in ivars]

        if len(x) != len(model.independent_vars):
            raise ValueError("Number of independent variables does not match model.")

        if len(x) == 1:
            x = x[0]

        if yerr is None:
            yerr = 1.0

        try:
            if len(model.independent_vars) == 1:
                params = model.guess(data, x)
            else:
                params = model.guess(data, *x)

            for key, val in kwargs.items():
                pname = f"{model.prefix}{key}"
                if pname not in params:
                    pname = key
                if pname not in params:
                    continue
                if isinstance(val, dict):
                    params[pname].set(**val)
                else:
                    params[pname].value = val

        except NotImplementedError:
            params = model.make_params(**kwargs)

        # Convert data to numpy array (must be after guessing parameters)
        data = np.asarray(data)

        param_names: list[str] = []
        fixed_params: list[str] = []
        values: dict[str, float] = {}
        limits: dict[str, tuple[float, float]] = {}

        for k, par in params.items():
            if par.expr is not None:
                if par.vary:
                    raise ValueError(
                        "Parameters constrained with expressions are not supported by Minuit."
                    )
                else:
                    continue

            param_names.append(k)
            if not par.vary:
                fixed_params.append(k)

            val = float(par.value)
            if not np.isfinite(val):
                val = 0.0

            values[k] = val
            limits[k] = (float(par.min), float(par.max))

        # Convert to kwargs
        if len(model.independent_vars) == 1:

            def _temp_func(x, *fargs):
                return model.func(
                    x, **dict(zip(model._param_root_names, fargs, strict=True))
                )

        else:

            def _temp_func(x, *fargs):
                return model.func(
                    *x, **dict(zip(model._param_root_names, fargs, strict=True))
                )

        c = LeastSq(x, data, yerr, _temp_func)
        m = cls(c, name=param_names, **values)

        for n in param_names:
            m.fixed[n] = n in fixed_params
            m.limits[n] = limits[n]

        if return_cost:
            return c, m
        return m

    def _repr_html_(self):
        s = ""
        if self.fmin is not None:
            s += self.fmin._repr_html_()
        s += self.params._repr_html_()
        if self.merrors:
            s += self.merrors._repr_html_()
        if self.covariance is not None:
            s += self.covariance._repr_html_()
        if self.fmin is not None:
            try:
                self.visualize()

                with _TempFig(*erlab.plotting.general.figwh()):
                    self.visualize()
                    # import io
                    # with io.StringIO() as io:
                    # plt.savefig(io, format="svg", dpi=10)
                    # io.seek(0)
                    # s += io.read()
            except (ModuleNotFoundError, AttributeError, ValueError):
                pass
        return s
