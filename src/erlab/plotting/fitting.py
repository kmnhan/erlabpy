"""Plotting helpers for fitted curves and their uncertainty."""

from __future__ import annotations

__all__ = ["FitArtists", "plot_fit"]

import copy
import operator
import typing

import matplotlib.axes
import matplotlib.cbook
import matplotlib.collections
import matplotlib.container
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr


class FitArtists(typing.NamedTuple):
    """Artists returned by :func:`plot_fit`.

    Hidden data, fit, and band layers are ``None``. Hidden component lines and bands
    are empty dictionaries.
    """

    data: matplotlib.lines.Line2D | matplotlib.container.ErrorbarContainer | None
    fit: matplotlib.lines.Line2D | None
    band: matplotlib.collections.FillBetweenPolyCollection | None
    components: dict[str, matplotlib.lines.Line2D]
    component_bands: dict[str, matplotlib.collections.FillBetweenPolyCollection]


def plot_fit(
    result: xr.Dataset,
    ax: matplotlib.axes.Axes | None = None,
    *,
    datafmt: str | None = None,
    fitfmt: str | None = None,
    data_var: str | None = None,
    coord: str | None = None,
    numpoints: int | None = None,
    sigma: float | None = None,
    show_data: bool = True,
    show_fit: bool = True,
    show_components: bool = False,
    show_yerr: bool = True,
    data_kws: Mapping[str, typing.Any] | None = None,
    fit_kws: Mapping[str, typing.Any] | None = None,
    band_kws: Mapping[str, typing.Any] | None = None,
    component_kws: Mapping[str, Mapping[str, typing.Any]] | None = None,
    component_band_kws: Mapping[str, Mapping[str, typing.Any]] | None = None,
) -> FitArtists:
    """Plot a single fitted curve from an xarray-lmfit result.

    Parameters
    ----------
    result
        Result dataset for one curve. Select a curve before plotting stacked results.
    ax
        Target axes. Defaults to the current axes.
    datafmt
        Matplotlib format string for the data. Defaults to circles (``"o"``).
    fitfmt
        Matplotlib format string for the best fit. Defaults to a solid line (``"-"``).
    data_var
        Fitted variable to plot. Required only for results with multiple fitted
        variables.
    coord
        Horizontal coordinate. Defaults to the fitted data dimension.
    numpoints
        Number of evenly spaced points within the fitted domain. ``None`` uses the
        stored best fit. Resampling requires ``coord`` to match the fitted input
        and cannot be used with sampled auxiliary inputs.
    sigma
        Level passed to :meth:`~lmfit.model.ModelResult.eval_uncertainty`: a number of
        standard deviations for ``sigma >= 1``, or a probability for ``0 < sigma < 1``.
        ``None`` hides uncertainty bands.
    show_data
        Show the measured data.
    show_fit
        Show the best-fit line.
    show_components
        Show model components, with their uncertainty bands when ``sigma`` is set.
    show_yerr
        Show error bars as ``1 / weights`` when fit weights are available.
        Nonpositive or nonfinite weights have no error bar.
    data_kws
        Plotting keywords for the data markers or error bars.
    fit_kws
        Plotting keywords for the best-fit line.
    band_kws
        Plotting keywords for all uncertainty bands. Each band defaults to its
        curve's color. ``label`` applies only to the total band.
    component_kws
        Per-component plotting keywords, keyed by the names returned by
        :meth:`~lmfit.model.ModelResult.eval_components`.
    component_band_kws
        Per-component band keywords. Override ``band_kws``, including legend labels.

    Returns
    -------
    FitArtists
        Data, best-fit, and band artists, plus component lines and bands keyed by name.

    Notes
    -----
    Resampling, components, and uncertainty bands require a retained
    :class:`~lmfit.model.ModelResult`. Bands also require covariance and parameter
    errors. They describe approximate pointwise uncertainty in the fitted mean.
    """
    suffix = "modelfit_data"
    prefixes = [
        str(name)[: -len(suffix)]
        for name in result.data_vars
        if str(name).endswith(suffix)
    ]
    if data_var is None:
        if len(prefixes) != 1:
            raise ValueError(
                "Select data_var for a Dataset with multiple fitted variables."
            )
        prefix = prefixes[0]
    else:
        prefix = f"{data_var}_"
        if prefix not in prefixes:
            raise ValueError(f"No fitted data variable {data_var!r}.")
    data = result[prefix + suffix].squeeze(drop=True)
    best = result[prefix + "modelfit_best_fit"].squeeze(drop=True)
    if data.ndim != 1 or best.dims != data.dims:
        raise ValueError("Select a single one-dimensional fit before plotting.")
    axis_coord = data.dims[0] if coord is None else coord
    if (axis_coord != data.dims[0] and axis_coord not in data.coords) or data[
        axis_coord
    ].dims != data.dims:
        raise ValueError("coord must name a one-dimensional coordinate of the fit.")
    x = np.asarray(data[axis_coord].values, dtype=float)
    measured = np.asarray(data.values, dtype=float)
    fitted = np.asarray(best.values, dtype=float)
    plot_x = x
    fit_valid = np.isfinite(measured) & np.isfinite(fitted)
    valid = np.isfinite(x) & fit_valid
    if numpoints is not None:
        if isinstance(numpoints, bool):
            raise ValueError("numpoints must be an integer of at least two.")
        try:
            numpoints = operator.index(numpoints)
        except TypeError as exc:
            raise ValueError("numpoints must be an integer of at least two.") from exc
        if numpoints < 2:
            raise ValueError("numpoints must be an integer of at least two.")
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0.0):
        raise ValueError("sigma must be positive and finite.")
    errors = None
    weights_key = prefix + "modelfit_weights"
    if show_data and show_yerr and weights_key in result:
        weights_data = result[weights_key].squeeze(drop=True)
        if weights_data.ndim != 0 and weights_data.dims != data.dims:
            raise ValueError("Fit weights must be scalar or match the data dimension.")
        weights = np.broadcast_to(
            np.asarray(weights_data.values, dtype=float), measured.shape
        )
        errors = np.full(measured.shape, np.nan)
        np.divide(
            1.0, weights, out=errors, where=np.isfinite(weights) & (weights > 0.0)
        )
    uncertainty = None
    component_uncertainties: dict[str, np.ndarray] = {}
    component_values: dict[str, np.ndarray] = {}
    component_options: dict[str, dict[str, typing.Any]] = {}
    if numpoints is not None or sigma is not None or show_components:
        import lmfit.model

        key = prefix + "modelfit_results"
        model_result = result[key].squeeze(drop=True).item() if key in result else None
        if not isinstance(model_result, lmfit.model.ModelResult):
            raise ValueError("Model evaluation requires a retained lmfit ModelResult.")
        if not valid.any():
            raise ValueError("The fit has no finite fitted domain.")
        independent = model_result.model.independent_vars
        if not independent:
            raise ValueError("The model has no independent variable.")
        kwargs = dict(model_result.userkws)
        if numpoints is not None:
            if any(
                np.asarray(value).size > 1
                for key, value in kwargs.items()
                if key != independent[0]
            ):
                raise ValueError(
                    "Sampled auxiliary model inputs cannot use a new grid."
                )
            input_x = np.asarray(kwargs.get(independent[0]), dtype=float)
            if input_x.shape == x.shape:
                input_x = input_x[fit_valid]
            if input_x.shape != (np.count_nonzero(fit_valid),) or not np.array_equal(
                input_x, x[fit_valid]
            ):
                raise ValueError(
                    "A new grid requires an axis coordinate that matches the fitted "
                    "independent input. Set coord to the coordinate used for fitting."
                )
            input_x = input_x[np.isfinite(input_x)]
            plot_x = np.linspace(input_x.min(), input_x.max(), numpoints)
            kwargs[independent[0]] = plot_x
            fitted = np.asarray(model_result.eval(**kwargs), dtype=float)
            if fitted.shape != plot_x.shape or not np.all(np.isfinite(fitted)):
                raise ValueError(
                    "Model evaluation did not return a finite one-dimensional curve."
                )
        if sigma is not None:
            if (
                not model_result.errorbars
                or model_result.covar is None
                or not np.all(np.isfinite(model_result.covar))
                or any(
                    p.stderr is None or not np.isfinite(p.stderr)
                    for p in model_result.params.values()
                )
            ):
                raise ValueError(
                    "A confidence band requires covariance and parameter errors."
                )
            # lmfit stores uncertainty caches on the result.
            uncertainty_result = copy.copy(model_result)
            deltas: dict[str | None, typing.Any] = {
                None: uncertainty_result.eval_uncertainty(
                    params=model_result.params.copy(), sigma=sigma, **kwargs
                )
            }
            if show_components:
                deltas.update(uncertainty_result.dely_comps)
            expected_shape = (
                plot_x.shape
                if numpoints is not None
                else (np.count_nonzero(fit_valid),)
            )
            for name, values in deltas.items():
                delta = np.asarray(values, dtype=float)
                if numpoints is None and delta.shape == x.shape:
                    delta = delta[fit_valid]
                if delta.shape != expected_shape or not np.all(np.isfinite(delta)):
                    raise ValueError(
                        "Model uncertainty did not return finite curve errors."
                    )
                if numpoints is None:
                    native = np.full(x.shape, np.nan)
                    native[fit_valid] = delta
                    delta = native
                if name is None:
                    uncertainty = delta
                else:
                    component_uncertainties[name] = delta
        if show_components:
            evaluated = model_result.eval_components(
                params=model_result.params.copy(), **kwargs
            )
            unknown = (
                set(component_kws or {}) | set(component_band_kws or {})
            ) - evaluated.keys()
            if unknown:
                raise ValueError(
                    "Unknown model component names: "
                    + ", ".join(repr(name) for name in sorted(unknown))
                    + "."
                )
            expected_shape = (
                plot_x.shape
                if numpoints is not None
                else (np.count_nonzero(fit_valid),)
            )
            for name, values in evaluated.items():
                values = np.asarray(values, dtype=float)
                if values.ndim == 0:
                    values = np.full(expected_shape, values.item())
                elif numpoints is None and values.shape == x.shape:
                    values = values[fit_valid]
                if values.shape != expected_shape or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Model component {name!r} did not return a finite "
                        "one-dimensional curve."
                    )
                if numpoints is None:
                    native = np.full(x.shape, np.nan)
                    native[fit_valid] = values
                    values = native
                component_values[name] = values
                component_options[name] = {
                    "label": name,
                    "linestyle": "--",
                } | matplotlib.cbook.normalize_kwargs(
                    dict((component_kws or {}).get(name, {})),
                    alias_mapping=matplotlib.lines.Line2D,
                )
            if sigma is not None and component_uncertainties.keys() != evaluated.keys():
                raise ValueError(
                    "Model uncertainty did not match evaluated components."
                )
    # Finish numerical preparation before adding artists to the axes.
    if ax is None:
        ax = plt.gca()
    fit_options = {"label": "Best fit"} | matplotlib.cbook.normalize_kwargs(
        dict(fit_kws or {}), alias_mapping=matplotlib.lines.Line2D
    )
    data_options = {"label": "Data"} | matplotlib.cbook.normalize_kwargs(
        dict(data_kws or {}), alias_mapping=matplotlib.lines.Line2D
    )
    if errors is None:
        for option in (
            "ecolor",
            "elinewidth",
            "capsize",
            "capthick",
            "barsabove",
            "lolims",
            "uplims",
            "xlolims",
            "xuplims",
            "errorevery",
        ):
            data_options.pop(option, None)
    if datafmt is None:
        data_options = {"marker": "o", "linestyle": "none"} | data_options
    if fitfmt is None:
        fit_options = {"linestyle": "-"} | fit_options
    fit_args = () if fitfmt is None else (fitfmt,)
    data_args = () if datafmt is None else (datafmt,)
    line = ax.plot(plot_x, fitted, *fit_args, **fit_options)[0] if show_fit else None

    shared_band_options = matplotlib.cbook.normalize_kwargs(
        dict(band_kws or {}), alias_mapping=matplotlib.collections.Collection
    )

    def draw_band(
        values: np.ndarray,
        delta: np.ndarray,
        curve: matplotlib.lines.Line2D | None,
        options: Mapping[str, typing.Any],
    ) -> matplotlib.collections.FillBetweenPolyCollection:
        band_options = {"alpha": 0.2} | dict(options)
        if curve is not None:
            if "color" not in band_options:
                band_options.setdefault("facecolor", curve.get_color())
            band_options.setdefault("zorder", curve.get_zorder() - 1)
        else:
            band_options.setdefault("zorder", 1)
        return ax.fill_between(plot_x, values - delta, values + delta, **band_options)

    band = (
        draw_band(fitted, uncertainty, line, shared_band_options)
        if uncertainty is not None
        else None
    )
    points = None
    if show_data:
        if errors is None:
            points = ax.plot(x, measured, *data_args, **data_options)[0]
        else:
            points = ax.errorbar(
                x, measured, yerr=errors, fmt=datafmt or "", **data_options
            )
    components = {
        name: ax.plot(plot_x, values, **component_options[name])[0]
        for name, values in component_values.items()
    }
    component_bands = {
        name: draw_band(
            component_values[name],
            delta,
            components[name],
            {key: value for key, value in shared_band_options.items() if key != "label"}
            | matplotlib.cbook.normalize_kwargs(
                dict((component_band_kws or {}).get(name, {})),
                alias_mapping=matplotlib.collections.Collection,
            ),
        )
        for name, delta in component_uncertainties.items()
    }
    return FitArtists(points, line, band, components, component_bands)
