"""Fermi edge fitting."""

__all__ = [
    "correct_with_edge",
    "edge",
    "guess_edge_fit_range",
    "poly",
    "poly_from_edge",
    "quick_fit",
    "quick_resolution",
    "resolution",
    "resolution_roi",
    "spline_from_edge",
]
import typing
import warnings
from collections.abc import Callable

import lmfit
import lmfit.model
import matplotlib
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import numpy.typing as npt
import scipy
import xarray as xr

import erlab

if typing.TYPE_CHECKING:
    import joblib
else:
    import lazy_loader as _lazy

    joblib = _lazy.load("joblib")


_FWHM_TO_SIGMA = 1 / np.sqrt(8 * np.log(2))


def _eval_edge(modelresult: lmfit.model.ModelResult, *, evalute_at: npt.NDArray):
    model = modelresult.model
    if isinstance(model, erlab.analysis.fit.models.FermiEdge2dModel):
        return np.polynomial.polynomial.polyval(
            evalute_at,
            tuple(
                modelresult.best_values[f"c{i}"]
                for i in range(modelresult.model.func.poly.degree + 1)
            ),
        )
    return modelresult.eval(x=evalute_at)


def _normalize_range_bounds(
    value_range: tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
    lower, upper = value_range
    lower = None if lower is None else float(lower)
    upper = None if upper is None else float(upper)
    if lower is not None and upper is not None and lower > upper:
        lower, upper = upper, lower
    return lower, upper


def _range_slice_for_coord(
    coord: xr.DataArray, value_range: tuple[float | None, float | None]
) -> slice:
    lower, upper = _normalize_range_bounds(value_range)
    values = np.asarray(coord.values)

    if values.size < 2:
        return slice(lower, upper)

    if not erlab.utils.array.is_monotonic(values):
        coord_name = str(coord.name) if coord.name is not None else "<unknown>"
        raise ValueError(
            f"Coordinate `{coord_name}` is not monotonic. "
            "Sort the data before applying a range selection."
        )

    if values[0] <= values[-1]:
        start, stop = lower, upper
    else:
        start, stop = upper, lower
    return slice(start, stop)


def _range_limits_for_coord(
    coord: xr.DataArray, value_range: tuple[float | None, float | None]
) -> tuple[float, float]:
    coord_sel = coord.sel({coord.dims[0]: _range_slice_for_coord(coord, value_range)})
    values = np.asarray(coord_sel.values, dtype=float)
    return float(np.min(values)), float(np.max(values))


def _parse_deprecated_fast(use_step_edge: bool, kwargs: dict[str, typing.Any]) -> bool:
    if "fast" not in kwargs:
        return use_step_edge

    fast = bool(kwargs.pop("fast"))
    warnings.warn(
        "The `fast` argument is deprecated and will be removed in a future version. "
        "Use `use_step_edge` instead.",
        FutureWarning,
        stacklevel=3,
    )
    if use_step_edge and not fast:
        raise TypeError("`use_step_edge` and deprecated `fast` have conflicting values")
    return use_step_edge or fast


def _raise_unexpected_kwargs(function_name: str, kwargs: dict[str, typing.Any]) -> None:
    if kwargs:
        argument = next(iter(kwargs))
        raise TypeError(
            f"{function_name}() got an unexpected keyword argument {argument!r}"
        )


def _edge_sigma(*, temp: float, resolution: float, use_step_edge: bool) -> float:
    if use_step_edge:
        return float(
            (resolution + 3.5255 * erlab.constants.kb_eV * temp) * _FWHM_TO_SIGMA
        )
    return float(
        np.hypot(
            resolution * _FWHM_TO_SIGMA,
            np.pi * erlab.constants.kb_eV * temp / np.sqrt(3),
        )
    )


def _edge_model_and_params(
    *,
    temp: float,
    resolution: float,
    vary_temp: bool,
    bkg_slope: bool,
    use_step_edge: bool,
) -> tuple[lmfit.Model, lmfit.Parameters]:
    if use_step_edge:
        model = erlab.analysis.fit.models.StepEdgeModel()
        params = lmfit.create_params(
            sigma={
                "value": _edge_sigma(
                    temp=temp, resolution=resolution, use_step_edge=True
                ),
                "min": 0,
            }
        )
    else:
        model = erlab.analysis.fit.models.FermiEdgeModel()
        params = lmfit.create_params(
            temp={"value": temp, "vary": vary_temp, "min": 0},
            resolution={"value": resolution, "min": 0},
        )
    if not bkg_slope:
        params["back1"] = lmfit.Parameter("back1", value=0, vary=False)
    return model, params


def _estimate_local_noise(
    y: npt.NDArray, *, sigma: float, dx: float
) -> tuple[npt.NDArray, float]:
    """Estimate local point noise with a rolling second-difference MAD."""
    noise_residual = 2 * y[2:-2] - y[:-4] - y[4:]
    noise_window_size = min(
        max(2 * int(np.ceil(8 * sigma / dx)) + 1, 25),
        noise_residual.size,
    )
    noise_floor = np.finfo(float).eps * max(float(np.max(np.abs(y))), 1.0)

    noise_centers = np.clip(np.arange(y.size) - 2, 0, noise_residual.size - 1)
    noise_starts, noise_window_indices = np.unique(
        np.clip(
            noise_centers - noise_window_size // 2,
            0,
            noise_residual.size - noise_window_size,
        ),
        return_inverse=True,
    )
    window_noise = np.empty(noise_starts.size)
    for window_index, noise_start in enumerate(noise_starts):
        local_residual = noise_residual[noise_start : noise_start + noise_window_size]
        residual_median = float(np.median(local_residual))
        window_noise[window_index] = (
            1.4826 * np.median(np.abs(local_residual - residual_median)) / np.sqrt(6)
        )
    return window_noise[noise_window_indices], noise_floor


def _find_falling_edge_candidates(
    x: npt.NDArray,
    y: npt.NDArray,
    *,
    sigma: float,
    dx: float,
    local_noise: npt.NDArray,
    noise_floor: float,
) -> list[tuple[float, float]]:
    """Find falling-edge candidates with multiscale Gaussian derivatives."""
    local_median = scipy.ndimage.median_filter(y, size=5, mode="nearest")
    outliers = np.abs(y - local_median) > 6 * np.maximum(local_noise, noise_floor)
    detection_y = y.copy()
    detection_y[outliers] = local_median[outliers]

    signal_span = float(np.percentile(detection_y, 98) - np.percentile(detection_y, 2))
    candidates: list[tuple[float, float]] = []
    for scale in (0.5, 1.0, 2.0, 4.0):
        filter_sigma = max(scale * sigma / dx, 1.0)
        margin = max(3, int(np.ceil(3 * filter_sigma)))

        response = -scipy.ndimage.gaussian_filter1d(
            detection_y, filter_sigma, order=1, mode="nearest"
        )

        impulse = np.zeros(2 * margin + 1)
        impulse[margin] = 1.0
        kernel = scipy.ndimage.gaussian_filter1d(
            impulse, filter_sigma, order=1, mode="constant"
        )
        response_noise_factor = float(np.linalg.norm(kernel))

        interior_response = response[3:-3]
        left_neighbor = response[2:-4]
        right_neighbor = response[4:-2]
        # Select the right sample when adjacent response values are equal.
        is_local_maximum = (interior_response >= left_neighbor) & (
            interior_response > right_neighbor
        )
        peak_indices = np.flatnonzero(is_local_maximum) + 3

        for candidate_index in peak_indices:
            side_points = min(
                max(round(2 * filter_sigma), 3),
                candidate_index,
                x.size - candidate_index - 1,
            )
            occupied = detection_y[candidate_index - side_points : candidate_index]
            unoccupied = detection_y[
                candidate_index + 1 : candidate_index + side_points + 1
            ]
            contrast = float(np.median(occupied) - np.median(unoccupied))
            prominence = float(response[candidate_index])
            point_noise = float(local_noise[candidate_index])
            contrast_noise = point_noise * np.sqrt(np.pi / side_points)
            # Keep a practical contrast floor so a noiseless background slope cannot
            # qualify as an edge.
            if prominence > max(
                point_noise * response_noise_factor, noise_floor
            ) and contrast > max(
                3 * contrast_noise,
                0.01 * signal_span,
                noise_floor,
            ):
                denominator = (
                    response[candidate_index - 1]
                    - 2 * response[candidate_index]
                    + response[candidate_index + 1]
                )
                offset = float(
                    np.clip(
                        0.5
                        * (
                            response[candidate_index - 1]
                            - response[candidate_index + 1]
                        )
                        / denominator,
                        -1.0,
                        1.0,
                    )
                )
                candidates.append(
                    (
                        float(x[candidate_index] + offset * dx),
                        point_noise,
                    )
                )
    return candidates


def _guess_edge_fit_range(
    x: npt.NDArray,
    y: npt.NDArray,
    *,
    temp: float,
    resolution: float,
    use_step_edge: bool,
    bkg_slope: bool = True,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 12:
        raise ValueError("At least 12 finite points are required")

    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Energy coordinates must be unique")

    uniform_x = np.linspace(x[0], x[-1], x.size)
    uniform_y = np.interp(uniform_x, x, y)
    uniform_dx = float(uniform_x[1] - uniform_x[0])
    sigma = max(
        _edge_sigma(temp=temp, resolution=resolution, use_step_edge=use_step_edge),
        uniform_dx,
    )
    local_noise, noise_floor = _estimate_local_noise(
        uniform_y, sigma=sigma, dx=uniform_dx
    )
    candidates = _find_falling_edge_candidates(
        uniform_x,
        uniform_y,
        sigma=sigma,
        dx=uniform_dx,
        local_noise=local_noise,
        noise_floor=noise_floor,
    )
    if not candidates:
        raise ValueError("No falling edge was detected")

    # The Fermi edge is the terminal valid falling edge. Do not require agreement
    # across scales because a broad edge can be visible only at a coarse scale.
    center, point_noise = max(candidates, key=lambda item: item[0])

    trial_x_mean = float(np.mean(x))
    trial_x_scale = float(np.std(x))

    def fit_trial(lower_index: int) -> dict[str, float] | None:
        fit_x = x[lower_index:]
        fit_y = y[lower_index:]
        normalized_x = (fit_x - trial_x_mean) / trial_x_scale

        model = erlab.analysis.fit.models.StepEdgeModel()
        center_value = (center - trial_x_mean) / trial_x_scale
        sigma_value = sigma / trial_x_scale
        step = erlab.analysis.fit.functions.step_broad(
            normalized_x,
            center=center_value,
            sigma=sigma_value,
            amplitude=1.0,
        )
        if bkg_slope:
            linear_design = np.column_stack(
                (
                    1 - step,
                    normalized_x * (1 - step),
                    step,
                    normalized_x * step,
                )
            )
            parameter_names = ("back0", "back1", "dos0", "dos1")
        else:
            linear_design = np.column_stack((1 - step, step, normalized_x * step))
            parameter_names = ("back0", "dos0", "dos1")
        coefficients, _, rank, _ = np.linalg.lstsq(
            linear_design,
            fit_y,
            rcond=None,
        )
        if rank == linear_design.shape[1]:
            params = model.make_params(
                center=center_value,
                sigma=sigma_value,
                **dict(zip(parameter_names, coefficients, strict=True)),
            )
        else:
            params = model.guess(
                fit_y,
                x=normalized_x,
                center=center_value,
                sigma=sigma_value,
            )
        if not bkg_slope:
            params["back1"].set(value=0.0, vary=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fit_result = model.fit(
                    fit_y,
                    params,
                    x=normalized_x,
                    method="least_squares",
                )
        except (
            FloatingPointError,
            np.linalg.LinAlgError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return None

        fit_center = float(
            fit_result.best_values["center"] * trial_x_scale + trial_x_mean
        )
        fit_sigma = abs(float(fit_result.best_values["sigma"] * trial_x_scale))
        if not fit_result.success or not np.isfinite(fit_center + fit_sigma):
            return None

        lower = float(x[lower_index])
        if not (
            lower <= fit_center <= x[-1]
            and 0.05 * sigma <= fit_sigma <= 3 * sigma
            and fit_center - lower >= 3 * fit_sigma
            and x[-1] - fit_center >= 0.5 * fit_sigma
        ):
            return None
        return {
            "lower": lower,
            "center": fit_center,
            "sigma": fit_sigma,
        }

    def trials_are_stable(trials: list[dict[str, float]]) -> bool:
        centers = [trial["center"] for trial in trials]
        widths = [trial["sigma"] for trial in trials]
        return bool(
            max(centers) - min(centers) <= 0.5 * sigma
            and max(widths) / min(widths) <= 3
        )

    trial_sequence: list[dict[str, float] | None] = []
    previous_lower_index: int | None = None
    for width in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0):
        lower = max(float(x[0]), center - width * sigma)
        lower_index = min(
            int(np.searchsorted(x, lower, side="left")),
            x.size - 12,
        )
        if lower_index == previous_lower_index:
            continue
        previous_lower_index = lower_index
        trial_sequence.append(fit_trial(lower_index))

        recent_trials = trial_sequence[-3:]
        accepted_trials = [trial for trial in recent_trials if trial is not None]
        if len(accepted_trials) == 3 and trials_are_stable(accepted_trials):
            return accepted_trials[1]["lower"], float(x[-1])

    # If three consecutive trials do not stabilize, use the longest shorter
    # plateau. This keeps locally cropped EDCs usable. A complete rejection falls
    # through to the previous single-fit range heuristic below.
    best_score: tuple[int, int] | None = None
    best_trial: dict[str, float] | None = None
    for start in range(len(trial_sequence)):
        accepted_trials = []
        for trial in trial_sequence[start:]:
            if trial is None:
                break
            accepted_trials.append(trial)
            if not trials_are_stable(accepted_trials):
                break
            score = (len(accepted_trials), -start)
            if best_score is None or score > best_score:
                best_score = score
                best_trial = accepted_trials[(len(accepted_trials) - 1) // 2]
    if best_trial is not None:
        return best_trial["lower"], float(x[-1])

    # A balanced local range prevents occupied-side peaks from dominating the
    # provisional step fit, including when little unoccupied data is available.
    seed_lower = max(float(x[0]), 2 * center - float(x[-1]))
    seed_start = int(np.searchsorted(x, seed_lower, side="left"))
    if x.size - seed_start < 12:
        return float(x[0]), float(x[-1])

    occupied = y[seed_start : np.searchsorted(x, center, side="left")]
    unoccupied = y[np.searchsorted(x, center, side="right") :]
    contrast_points = min(occupied.size, unoccupied.size)
    if contrast_points < 3:
        return float(x[0]), float(x[-1])
    terminal_contrast = float(
        np.median(occupied[-contrast_points:]) - np.median(unoccupied[:contrast_points])
    )
    if terminal_contrast <= max(
        3 * point_noise * np.sqrt(np.pi / contrast_points), noise_floor
    ):
        raise ValueError("No falling edge was detected")

    fit_x = x[seed_start:]
    fit_y = y[seed_start:]
    x_mean = float(np.mean(fit_x))
    x_scale = float(np.std(fit_x))
    y_offset = float(np.percentile(fit_y, 2))
    y_scale = max(float(np.percentile(fit_y, 98) - y_offset), np.finfo(float).eps)
    normalized_x = (fit_x - x_mean) / x_scale
    normalized_y = (fit_y - y_offset) / y_scale

    model = erlab.analysis.fit.models.StepEdgeModel()
    params = model.guess(normalized_y, x=normalized_x)
    params["center"].set(
        value=(center - x_mean) / x_scale,
        min=float(normalized_x[0]),
        max=float(normalized_x[-1]),
    )
    params["sigma"].set(
        value=sigma / x_scale,
        min=0.0,
        max=0.5 * float(np.ptp(normalized_x)),
    )
    if not bkg_slope:
        params["back1"].set(value=0.0, vary=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit_result = model.fit(
                normalized_y,
                params,
                x=normalized_x,
                method="least_squares",
            )
        fit_center = float(fit_result.best_values["center"] * x_scale + x_mean)
        fit_sigma = float(fit_result.best_values["sigma"] * x_scale)
    except (FloatingPointError, np.linalg.LinAlgError, TypeError, ValueError):
        return float(x[0]), float(x[-1])

    if not fit_result.success or not np.isfinite(fit_center + fit_sigma):
        return float(x[0]), float(x[-1])

    # Four fitted widths contain the transition. The nominal-width floor also keeps
    # two nominal widths of occupied baseline when the provisional fit is too narrow.
    lower = max(float(x[0]), min(center, fit_center) - max(4 * fit_sigma, 6 * sigma))
    lower_index = min(int(np.searchsorted(x, lower, side="left")), x.size - 12)
    return float(x[lower_index]), float(x[-1])


def _guess_edge_fit_range_or_default(
    x: npt.NDArray,
    y: npt.NDArray,
    *,
    temp: float,
    resolution: float,
    use_step_edge: bool,
    bkg_slope: bool = True,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    finite_x = x[np.isfinite(x)]
    if finite_x.size == 0:
        raise ValueError("Energy coordinates must contain finite values")
    fallback = float(np.min(finite_x)), float(np.max(finite_x))
    try:
        return _guess_edge_fit_range(
            x,
            y,
            temp=temp,
            resolution=resolution,
            use_step_edge=use_step_edge,
            bkg_slope=bkg_slope,
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return fallback


def guess_edge_fit_range(
    edc: xr.DataArray,
    *,
    energy_dim: str = "eV",
    temp: float | None = None,
    resolution: float = 0.02,
    bkg_slope: bool = True,
    use_step_edge: bool = False,
    **kwargs,
) -> tuple[float, float]:
    r"""Estimate the energy range for fitting one Fermi edge.

    The returned bounds are ordered from low to high energy and can be used as
    ``edc.sel(eV=slice(*bounds))`` when the energy coordinate is increasing.

    Parameters
    ----------
    edc
        One energy distribution curve.
    energy_dim
        Name of the energy dimension.
    temp
        Sample temperature in kelvins. If `None`, infer it from `edc`. The broadened
        step model uses 10 K when the temperature is not available.
    resolution
        Initial energy-resolution FWHM in electronvolts.
    bkg_slope
        If `True`, include a linear background above the Fermi level.
    use_step_edge
        If `True`, use the nominal width of a broadened step edge. If `False`, use the
        combined thermal and resolution width of a Fermi edge. Defaults to `False`.
    **kwargs
        Deprecated arguments. Use `use_step_edge` instead of ``fast``.

    Returns
    -------
    lower, upper
        Estimated energy bounds in the units of the energy coordinate.

    Notes
    -----
    The estimator uses the following steps:

    1. It calculates a nominal edge standard deviation. For a Fermi edge model, this is

       .. math::

           \sigma_0 = \max\left[\Delta x, \sqrt{\left(\frac{R}{\sqrt{8\ln 2}}\right)^2 +
           \left(\frac{\pi k_\mathrm{B}T}{\sqrt{3}}\right)^2}\right].

       For the step function, it uses

       .. math::

           \sigma_0 = \max\left[\Delta x, \frac{R + 3.5255 k_\mathrm{B}T}{\sqrt{8\ln
           2}}\right].

       Here, :math:`R` is the resolution FWHM and :math:`\Delta x` is the energy
       spacing.

    2. It estimates the local point noise from the second-difference residual

       .. math::

           r_i = 2y_i-y_{i-2}-y_{i+2}, \qquad \hat\sigma_{n,i} =
           \frac{1.4826}{\sqrt{6}}\operatorname{MAD}(r).

       The MAD uses a rolling window of approximately :math:`16\sigma_0`, with a
       25-sample minimum when that many residual samples are available. The factors
       1.4826 and :math:`\sqrt{6}` correct the MAD for Gaussian noise and the variance
       of the second difference, respectively.

    3. It replaces samples that differ from a five-sample median by more than six local
       noise levels in the detection copy. This replacement does not modify the data
       used for fits.

    4. It performs a multiscale search for local maxima of negative Gaussian derivatives
       with widths of :math:`\{0.5,1,2,4\}\sigma_0`. A candidate must have derivative
       prominence above the propagated local noise. It must also have a falling-side
       contrast above three contrast-noise levels and 1% of the robust signal span. A
       quadratic interpolation refines its position. The highest-energy valid candidate
       becomes the edge anchor.

    5. It fits broadened step models with lower bounds at offsets of
       :math:`\{4,6,8,10,12,16,20,24,32\}\sigma_0` below the anchor. The anchor and
       nominal width initialize each fit. The linear coefficients are initialized
       conditionally on the broadened step :math:`s(x)`:

       .. math::

           y(x) = [1-s(x)](b_0+b_1x) + s(x)(d_0+d_1x).

       When ``bkg_slope=False``, the fit fixes :math:`b_1=0`.

    6. A trial is accepted when the fitted center remains inside the fit domain and
       the fitted width is between :math:`0.05\sigma_0` and :math:`3\sigma_0`. The
       domain must contain at least three fitted widths below the center and half a
       fitted width above it.

    7. The estimator returns the middle lower bound of the first three consecutive
       accepted fits whose centers span at most :math:`0.5\sigma_0` and whose widths
       differ by at most a factor of three. If this plateau is unavailable, it uses the
       longest shorter plateau, then a local single-step fallback. The fallback can
       return the full input range when the available support is insufficient. The upper
       bound is the highest input energy.

    Raises
    ------
    ValueError
        If the input is not one-dimensional, lacks required temperature metadata, has
        insufficient data, or has no qualifying falling edge.

    """
    use_step_edge = _parse_deprecated_fast(use_step_edge, kwargs)
    _raise_unexpected_kwargs("guess_edge_fit_range", kwargs)

    if edc.dims != (energy_dim,):
        raise ValueError(f"Expected a 1D DataArray along {energy_dim!r}")

    if temp is None:
        temp = edc.qinfo.get_value("sample_temp")
        if temp is None:
            if use_step_edge:
                temp = 10.0
            else:
                raise ValueError(
                    "Temperature not found in data attributes, please provide manually"
                )

    return _guess_edge_fit_range(
        np.asarray(edc[energy_dim]),
        np.asarray(edc),
        temp=float(temp),
        resolution=float(resolution),
        use_step_edge=use_step_edge,
        bkg_slope=bkg_slope,
    )


def _evaluate_edge_model(
    darr: xr.DataArray,
    modelresult: lmfit.model.ModelResult
    | xr.Dataset
    | npt.NDArray[np.floating]
    | Callable
    | tuple[float, ...],
    *,
    along: str = "alpha",
) -> xr.DataArray:
    if isinstance(modelresult, xr.Dataset):
        if "modelfit_results" in modelresult:
            results = modelresult.modelfit_results
            modelresult = xr.apply_ufunc(
                _eval_edge,
                results,
                output_core_dims=[[along]],
                output_dtypes=[float],
                dask="parallelized",
                dask_gufunc_kwargs={"output_sizes": {along: darr.sizes[along]}},
                vectorize=True,
                kwargs={"evalute_at": darr[along].values},
            ).assign_coords({along: darr[along]})

        elif "modelfit_coefficients" in modelresult:
            # Only coefficients are provided
            coeffs = modelresult.modelfit_coefficients
            if all(p.startswith("c") for p in coeffs.param.values):
                coeffs = coeffs.assign_coords(
                    param=[int(d.removeprefix("c")) for d in coeffs.param.values]
                ).rename(param="degree")
                modelresult = xr.polyval(darr[along], coeffs)
            else:
                raise ValueError(
                    "Fit result dataset does not seem to contain valid polynomial "
                    "coefficients."
                )
        else:
            raise ValueError(
                "Fit result dataset does not seem to contain valid fit results."
            )
    if isinstance(modelresult, lmfit.model.ModelResult):
        modelresult = _eval_edge(modelresult, evalute_at=darr[along].values)

    if callable(modelresult):
        edge_quad = modelresult(darr[along].values)

    elif isinstance(modelresult, tuple):
        edge_quad = np.polynomial.polynomial.polyval(darr[along], modelresult)

    elif isinstance(modelresult, np.ndarray):
        if len(darr[along]) != len(modelresult):
            raise ValueError(
                "Length of modelresult array does not match length of data along "
                f"dimension '{along}'."
            )
        edge_quad = modelresult

    elif isinstance(modelresult, xr.DataArray):
        edge_quad = modelresult

    else:
        raise TypeError(
            "modelresult must be one of lmfit.model.ModelResult, xarray.Dataset, "
            "numpy.ndarray, callable, or tuple of float."
        )

    if np.isscalar(edge_quad) or (
        isinstance(edge_quad, np.ndarray) and edge_quad.ndim == 0
    ):
        edge_quad = xr.DataArray(edge_quad).broadcast_like(darr[along])
    elif isinstance(edge_quad, np.ndarray):
        edge_quad = xr.DataArray(edge_quad, coords={along: darr[along]}, dims=[along])

    dimension_order = [
        dim for dim in darr.dims if dim in edge_quad.dims and dim != "eV"
    ]
    dimension_order.extend(dim for dim in edge_quad.dims if dim not in dimension_order)
    return edge_quad.transpose(*dimension_order)


def correct_with_edge(
    darr: xr.DataArray,
    modelresult: lmfit.model.ModelResult
    | xr.Dataset
    | npt.NDArray[np.floating]
    | Callable
    | tuple[float, ...],
    *,
    along: str = "alpha",
    shift_coords: bool = True,
    plot: bool = False,
    plot_kw: dict | None = None,
    **shift_kwargs,
):
    """Corrects the given data array `darr` with the given values or fit result.

    Parameters
    ----------
    darr
        The input data array to be corrected.
    modelresult
        The model result that contains the Fermi edge information. It can be an instance
        of `lmfit.model.ModelResult`, a numpy array containing the edge position at each
        angle, a fit result dataset that contains polynomial coefficients, a callable
        function that takes an array of angles and returns the corresponding energy
        value, or a tuple of coefficients for a polynomial (lowest order first).
    along
        The angular dimension name in the data. If `None`, it is assumed to be
        ``"alpha"``.
    shift_coords
        If `True`, the coordinates of the output data will be changed so that the output
        contains all the values of the original data. If `False`, the coordinates and
        shape of the original data will be retained, and only the data will be shifted.
        Defaults to `False`.
    plot
        Whether to plot the original and corrected data arrays. Defaults to `False`.
    plot_kw
        Additional keyword arguments for the plot. Defaults to `None`.
    **shift_kwargs
        Additional keyword arguments to :func:`erlab.analysis.transform.shift`.

    Returns
    -------
    corrected : xarray.DataArray
        The edge corrected data.
    """
    if plot_kw is None:  # pragma: no branch
        plot_kw = {}

    edge_quad = _evaluate_edge_model(darr, modelresult, along=along)

    corrected = erlab.analysis.transform.shift(
        darr, -edge_quad, "eV", shift_coords=shift_coords, **shift_kwargs
    )

    if plot is True:
        if edge_quad.ndim > 1:
            raise ValueError("Plotting is only supported for 1D edge corrections.")
        axes = typing.cast(
            "npt.NDArray", plt.subplots(1, 2, layout="constrained", figsize=(10, 5))[1]
        )

        plot_kw.setdefault("cmap", "copper")
        plot_kw.setdefault("gamma", 0.5)

        if darr.ndim > 2:
            avg_dims = list(darr.dims)[:]
            avg_dims.remove(along)
            avg_dims.remove("eV")
            erlab.plotting.plot_array(darr.mean(avg_dims), ax=axes[0], **plot_kw)
            erlab.plotting.plot_array(corrected.mean(avg_dims), ax=axes[1], **plot_kw)
        else:
            erlab.plotting.plot_array(darr, ax=axes[0], **plot_kw)
            erlab.plotting.plot_array(corrected, ax=axes[1], **plot_kw)
        edge_plot = typing.cast("typing.Any", edge_quad.plot)
        edge_plot(ax=axes[0], ls="--", color="0.35")

        erlab.plotting.proportional_colorbar(ax=axes[0])
        erlab.plotting.proportional_colorbar(ax=axes[1])
        axes[0].set_title("Data")
        axes[1].set_title("Edge Corrected")

    return corrected


def edge(
    gold: xr.DataArray,
    *,
    along: str = "alpha",
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    adaptive: bool = False,
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    bkg_slope: bool = True,
    resolution: float = 0.02,
    use_step_edge: bool = False,
    method: str = "least_squares",
    scale_covar: bool = True,
    normalize: bool = True,
    fixed_center: float | None = None,
    progress: bool = True,
    parallel_kw: dict | None = None,
    parallel_obj: joblib.Parallel | None = None,
    return_full: bool = False,
    drop_nans: bool = False,
    **kwargs,
) -> tuple[xr.DataArray, xr.DataArray] | xr.Dataset:
    """
    Fit a Fermi edge to the given gold data.

    Only successful fits with valid error estimates are returned.

    Parameters
    ----------
    gold
        The gold data to fit the edge model to.
    along
        The dimension along which to parallelize the fitting. By default ``"alpha"``. It
        is better to choose the dimension with the largest number of points.

        If `gold` is chunked, this parameter is only used to specify the dimension along
        which to apply `angle_range`.
    angle_range
        The range of values along the ``along`` dimension to consider.
    eV_range
        The range of eV values to consider.
    adaptive
        If `True`, estimate and use a separate energy range for each EDC within
        `eV_range`. If no valid falling edge is detected in one EDC, that EDC uses the
        complete `eV_range`. Defaults to `False`.
    bin_size
        The bin size for coarsening the gold data, by default (1, 1).
    temp
        The temperature in Kelvins. If `None`, the temperature is inferred from the
        attributes, by default `None`
    vary_temp
        Whether to fit the temperature value during fitting, by default `False`.
    bkg_slope
        Whether to include a linear background above the Fermi level. If `False`, the
        background above the Fermi level is fit with a constant. Defaults to `True`.
    resolution
        The initial resolution value to use for fitting, by default `0.02`.
    use_step_edge
        Whether to use the Gaussian-broadened step function to fit the edge, by default
        `False`.
    method
        The fitting method to use, by default ``"least_squares"``.
    scale_covar
        Whether to scale the covariance matrix, by default `True`.
    fixed_center
        The fixed center value. If provided, the Fermi level will be fixed at the given
        value, by default `None`.
    normalize
        Whether to normalize the energy coordinates, by default `True`.
    progress
        Whether to display the fitting progress, by default `True`.
    parallel_kw
        Additional keyword arguments for parallel fitting, by default `None`.
    parallel_obj
        The `joblib.Parallel` object to use for fitting, by default `None`. If provided,
        `parallel_kw` will be ignored.
    return_full
        Whether to return the full fit results, by default `False`.
    drop_nans
        Whether to drop fits that resulted in NaN values, by default `False`. If `True`,
        the function will always return the computed data even for chunked inputs,
        because dropping NaNs requires computing all fit results. If ``return_full`` is
        `True`, this option is ignored.
    **kwargs
        Additional keyword arguments to fitting. The old ``fast`` argument is accepted
        with a deprecation warning. Use `use_step_edge` instead.

    Returns
    -------
    center_arr, center_stderr
        The fitted center values and their standard errors, returned when `return_full`
        is `False`.
    fit_result
        A dataset containing the full fit results, returned when `return_full` is
        `True`.

    .. versionchanged:: 3.27.0

        Added `use_step_edge`. The old ``fast`` argument is deprecated.

    """
    use_step_edge = _parse_deprecated_fast(use_step_edge, kwargs)

    if any(b != 1 for b in bin_size):
        gold_binned = gold.coarsen(
            {along: bin_size[0], "eV": bin_size[1]}, boundary="trim"
        )
        gold = gold_binned.mean()  # type: ignore[attr-defined]

    gold_sel = gold.sel(
        {
            along: _range_slice_for_coord(gold[along], angle_range),
            "eV": _range_slice_for_coord(gold["eV"], eV_range),
        }
    )

    if temp is None:
        temp = gold.qinfo.get_value("sample_temp")
        if temp is None:
            if use_step_edge:
                temp = 10.0
            else:
                raise ValueError(
                    "Temperature not found in data attributes, please provide manually"
                )

    if adaptive and kwargs.get("skipna") is False:
        raise ValueError("adaptive fitting requires skipna=True")

    if normalize:
        # Normalize energy coordinates
        avgx, stdx = gold_sel.eV.values.mean(), gold_sel.eV.values.std()
        gold_sel = gold_sel.assign_coords(eV=(gold_sel.eV - avgx) / stdx)

    if normalize:
        fit_temp = float(temp / stdx)
        fit_resolution = float(resolution / stdx)
    else:
        fit_temp = float(temp)
        fit_resolution = float(resolution)

    model, params = _edge_model_and_params(
        temp=fit_temp,
        resolution=fit_resolution,
        vary_temp=vary_temp,
        bkg_slope=bkg_slope,
        use_step_edge=use_step_edge,
    )

    if parallel_kw is None:
        parallel_kw = {}

    if fixed_center is not None:
        fixed_center_fit = (
            (float(fixed_center) - avgx) / stdx if normalize else float(fixed_center)
        )
        params["center"] = lmfit.Parameter("center", value=fixed_center_fit, vary=False)

    # Assuming Poisson noise, the weights are the square root of the counts.
    weights = (1 / gold_sel.sum("eV").clip(min=1e-15)) ** 0.5

    n_fits = len(gold_sel[along])

    if parallel_obj is None:
        parallel_kw.setdefault("n_jobs", -1 if n_fits > 40 else 1)
        parallel_kw.setdefault("max_nbytes", None)
        parallel_kw.setdefault("return_as", "generator")
        parallel_kw.setdefault("pre_dispatch", "n_jobs")
        parallel_kw.setdefault("return_as", "generator")
        if erlab.utils.misc._IS_PACKAGED:
            # https://github.com/joblib/joblib/issues/1002
            parallel_kw["backend"] = "threading"

        parallel_obj = joblib.Parallel(**parallel_kw)

    def _fit(data, w):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=(
                    "Using UFloat objects with std_dev==0 may give unexpected results."
                ),
            )
            if adaptive:
                range_kwargs = {
                    "temp": fit_temp,
                    "resolution": fit_resolution,
                    "bkg_slope": bkg_slope,
                    "use_step_edge": use_step_edge,
                }
                if data.dims == ("eV",):
                    lower, upper = _guess_edge_fit_range_or_default(
                        np.asarray(data["eV"]),
                        np.asarray(data),
                        **range_kwargs,
                    )
                else:
                    lower, upper = xr.apply_ufunc(
                        _guess_edge_fit_range_or_default,
                        data["eV"],
                        data,
                        input_core_dims=[["eV"], ["eV"]],
                        output_core_dims=[[], []],
                        vectorize=True,
                        dask="parallelized",
                        output_dtypes=[float, float],
                        kwargs=range_kwargs,
                    )
                energy = data["eV"]
                data = data.where((energy >= lower) & (energy <= upper))
            return data.xlm.modelfit(
                "eV",
                model=model,
                params=params,
                method=method,
                scale_covar=scale_covar,
                weights=w,
                guess=True,
                **kwargs,
            )

    if gold_sel.chunks is None:
        tqdm_kw = {"desc": "Fitting", "total": n_fits, "disable": not progress}

        if parallel_obj.return_generator:
            tqdm = erlab.utils.misc.get_tqdm()

            fit_result = tqdm(
                parallel_obj(
                    joblib.delayed(_fit)(
                        gold_sel.isel({along: i}), weights.isel({along: i})
                    )
                    for i in range(n_fits)
                ),
                **tqdm_kw,
            )
        elif progress:
            with erlab.utils.parallel.joblib_progress(**tqdm_kw) as _:
                fit_result = parallel_obj(
                    joblib.delayed(_fit)(
                        gold_sel.isel({along: i}), weights.isel({along: i})
                    )
                    for i in range(n_fits)
                )
        else:
            fit_result = parallel_obj(
                joblib.delayed(_fit)(
                    gold_sel.isel({along: i}), weights.isel({along: i})
                )
                for i in range(n_fits)
            )
        fit_result = xr.concat(fit_result, along)
    else:
        fit_result = _fit(gold_sel, weights)

    if return_full:
        return fit_result

    vals = fit_result.modelfit_coefficients.sel(param="center").drop_vars("param")
    errs = fit_result.modelfit_stderr.sel(param="center").drop_vars("param")

    if drop_nans:
        if vals.chunks is not None:
            import dask

            vals, errs = dask.compute(vals, errs)

        mask = errs.isnull()
        vals, errs = (vals.where(~mask, drop=True), errs.where(~mask, drop=True))

    if normalize:
        vals = vals * stdx + avgx
        errs = errs * stdx

    # Clear attrs to match previous behavior
    vals.attrs = {}
    errs.attrs = {}

    return vals, errs


def poly_from_edge(
    center: xr.DataArray,
    weights=None,
    degree: int = 4,
    method="least_squares",
    scale_covar=True,
    along: str = "alpha",
) -> xr.Dataset:
    model = erlab.analysis.fit.models.PolynomialModel(degree=degree)
    return center.xlm.modelfit(
        along,
        model=model,
        guess=True,
        weights=weights,
        method=method,
        scale_covar=scale_covar,
        output_result=True,
    )


def spline_from_edge(
    center,
    weights: npt.ArrayLike | None = None,
    lam: float | None = None,
    along: str = "alpha",
) -> scipy.interpolate.BSpline:
    return scipy.interpolate.make_smoothing_spline(
        center[along].values, center.values, w=np.asarray(weights), lam=lam
    )


def _plot_gold_fit(
    fig, gold, along, angle_range, eV_range, center_arr, center_stderr, res
) -> None:
    if isinstance(res, xr.Dataset) and "modelfit_results" in res:
        is_callable = False
        res = res.modelfit_results.values.item()
    elif isinstance(res, lmfit.model.ModelResult):
        is_callable = False
    elif callable(res):
        is_callable = True
    else:
        raise TypeError(
            "res must be one of callable, lmfit.model.ModelResult, "
            "and fit result dataset"
        )

    if not isinstance(fig, plt.Figure):
        fig = plt.figure(figsize=erlab.plotting.figwh(0.75, wscale=1.75))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    plt.tick_params("x", labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)

    if gold.dims[0] == "eV":
        gold.qplot(ax=ax0, cmap="copper", gamma=0.5)
    else:
        gold.T.qplot(ax=ax0, cmap="copper", gamma=0.5)

    angle_lims = _range_limits_for_coord(gold[along], angle_range)
    eV_lims = _range_limits_for_coord(gold["eV"], eV_range)
    rect = matplotlib.patches.Rectangle(
        (angle_lims[0], eV_lims[0]),
        angle_lims[1] - angle_lims[0],
        eV_lims[1] - eV_lims[0],
        ec="w",
        alpha=0.5,
        lw=0.75,
        fc="none",
    )
    ax0.add_patch(rect)
    ax0.errorbar(
        center_arr[along],
        center_arr,
        center_stderr,
        fmt="o",
        lw=0.5,
        mfc="w",
        zorder=0,
        ms=2,
    )

    if is_callable:
        ax0.plot(gold[along], res(gold[along]), "r-", lw=0.75)
    else:
        ax0.plot(gold[along], res.eval(res.params, x=gold[along]), "r-", lw=0.75)
    ax0.set_ylim(gold.eV[[0, -1]])

    data_kws: dict[str, typing.Any] = {
        "lw": 0.5,
        "ms": 2,
        "mfc": "w",
        "zorder": 0,
        "c": "0.4",
        "capsize": 0,
    }
    fit_kws: dict[str, typing.Any] = {"c": "r", "lw": 0.75}

    if is_callable:
        residuals = res(center_arr[along].values) - center_arr.values
        x_eval = np.linspace(
            min(center_arr[along].values),
            max(center_arr[along].values),
            3 * len(center_arr[along]),
        )
        ax1.axhline(0, **fit_kws)
        ax1.errorbar(
            center_arr[along],
            residuals,
            yerr=lmfit.model.propagate_err(
                center_arr.values, center_stderr.values, "abs"
            ),
            fmt="o",
            **data_kws,
        )
        ax1.set_ylabel("residuals")

        ax2.errorbar(
            center_arr[along],
            center_arr.values,
            yerr=lmfit.model.propagate_err(
                center_arr.values, center_stderr.values, "abs"
            ),
            fmt="o",
            label="data",
            **data_kws,
        )
        ax2.plot(x_eval, res(x_eval), "-", label="best fit", **fit_kws)
        ax2.legend()
        ax1.relim()
        ax2.relim()
    else:
        res.plot_residuals(ax=ax1, data_kws=data_kws, fit_kws=fit_kws)
        res.plot_fit(
            ax=ax2,
            data_kws=data_kws,
            fit_kws=fit_kws,
            numpoints=3 * len(center_arr[along]),
        )
        ax1.relim()
        ax2.relim()
    ax1.set_title("")
    ax2.set_title("")


def poly(
    gold: xr.DataArray,
    *,
    along: str = "alpha",
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    adaptive: bool = False,
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    bkg_slope: bool = True,
    resolution: float = 0.02,
    use_step_edge: bool = False,
    method: str = "least_squares",
    normalize: bool = True,
    degree: int = 4,
    correct: bool = False,
    return_edge: bool = False,
    crop_correct: bool = False,
    parallel_kw: dict | None = None,
    plot: bool = True,
    fig: matplotlib.figure.Figure | None = None,
    scale_covar: bool = True,
    scale_covar_edge: bool = True,
    **kwargs,
) -> xr.Dataset | xr.DataArray | tuple[xr.Dataset, xr.DataArray]:
    use_step_edge = _parse_deprecated_fast(use_step_edge, kwargs)
    _raise_unexpected_kwargs("poly", kwargs)
    if correct and return_edge:
        raise ValueError("`correct` and `return_edge` cannot both be True.")

    center_arr, center_stderr = typing.cast(
        "tuple[xr.DataArray, xr.DataArray]",
        edge(
            gold,
            along=along,
            angle_range=angle_range,
            eV_range=eV_range,
            adaptive=adaptive,
            bin_size=bin_size,
            temp=temp,
            vary_temp=vary_temp,
            bkg_slope=bkg_slope,
            resolution=resolution,
            use_step_edge=use_step_edge,
            method=method,
            normalize=normalize,
            parallel_kw=parallel_kw,
            scale_covar=scale_covar_edge,
            drop_nans=True,
        ),
    )

    results = poly_from_edge(
        center_arr,
        weights=1.0 / center_stderr,
        degree=degree,
        method=method,
        scale_covar=scale_covar,
        along=along,
    )
    if plot:
        _plot_gold_fit(
            fig, gold, along, angle_range, eV_range, center_arr, center_stderr, results
        )
    if return_edge:
        return _evaluate_edge_model(gold, results, along=along)
    if correct:
        if crop_correct:
            gold = gold.sel(
                {
                    along: _range_slice_for_coord(gold[along], angle_range),
                    "eV": _range_slice_for_coord(gold["eV"], eV_range),
                }
            )
        corr = correct_with_edge(gold, results, along=along, plot=False)
        return results, corr
    return results


def spline(
    gold: xr.DataArray,
    *,
    along: str = "alpha",
    angle_range: tuple[float, float],
    eV_range: tuple[float, float],
    adaptive: bool = False,
    bin_size: tuple[int, int] = (1, 1),
    temp: float | None = None,
    vary_temp: bool = False,
    bkg_slope: bool = True,
    resolution: float = 0.02,
    use_step_edge: bool = False,
    method: str = "least_squares",
    lam: float | None = None,
    correct: bool = False,
    return_edge: bool = False,
    crop_correct: bool = False,
    parallel_kw: dict | None = None,
    plot: bool = True,
    fig: matplotlib.figure.Figure | None = None,
    scale_covar_edge: bool = True,
    **kwargs,
) -> (
    scipy.interpolate.BSpline
    | xr.DataArray
    | tuple[scipy.interpolate.BSpline, xr.DataArray]
):
    use_step_edge = _parse_deprecated_fast(use_step_edge, kwargs)
    _raise_unexpected_kwargs("spline", kwargs)
    if correct and return_edge:
        raise ValueError("`correct` and `return_edge` cannot both be True.")

    center_arr, center_stderr = typing.cast(
        "tuple[xr.DataArray, xr.DataArray]",
        edge(
            gold,
            along=along,
            angle_range=angle_range,
            eV_range=eV_range,
            adaptive=adaptive,
            bin_size=bin_size,
            temp=temp,
            vary_temp=vary_temp,
            bkg_slope=bkg_slope,
            resolution=resolution,
            use_step_edge=use_step_edge,
            method=method,
            parallel_kw=parallel_kw,
            scale_covar=scale_covar_edge,
            drop_nans=True,
        ),
    )

    spl = spline_from_edge(center_arr, weights=1 / center_stderr, lam=lam, along=along)
    if plot:
        _plot_gold_fit(
            fig, gold, along, angle_range, eV_range, center_arr, center_stderr, spl
        )
    if return_edge:
        return _evaluate_edge_model(gold, spl, along=along)
    if correct:
        if crop_correct:
            gold = gold.sel(
                {
                    along: _range_slice_for_coord(gold[along], angle_range),
                    "eV": _range_slice_for_coord(gold["eV"], eV_range),
                }
            )
        corr = correct_with_edge(gold, spl, along=along, plot=False)
        return spl, corr
    return spl


def quick_fit(
    darr: xr.DataArray,
    *,
    eV_range: tuple[float, float] | None = None,
    method: str = "leastsq",
    temp: float | None = None,
    resolution: float | None = None,
    center: float | None = None,
    fix_temp: bool = True,
    fix_center: bool = False,
    fix_resolution: bool = False,
    bkg_slope: bool = True,
    plot: bool = False,
    ax: matplotlib.axes.Axes | None = None,
    plot_fit_kwargs: dict[str, typing.Any] | None = None,
    plot_data_kwargs: dict[str, typing.Any] | None = None,
    plot_line_kwargs: dict[str, typing.Any] | None = None,
    plot_span_kwargs: dict[str, typing.Any] | None = None,
    **kwargs,
) -> xr.Dataset:
    """Perform a Fermi edge fit on an EDC.

    This function is a convenient wrapper around :meth:`DataArray.xlm.modelfit` that
    fits a Fermi edge to the given data.

    If data with 2 or more dimensions is provided, the data is averaged over all
    dimensions except the energy prior to fitting.

    Parameters
    ----------
    darr
        The input data to be fitted.
    eV_range
        The energy range to consider for fitting. If `None`, the entire energy range is
        used. Defaults to `None`.
    method
        The fitting method to use that is compatible with `lmfit`. Defaults to
        "leastsq".
    temp
        The temperature value to use for fitting. If `None`, the temperature is inferred
        from the data attributes.
    resolution
        The initial resolution value to use for fitting. If `None`, the resolution is
        set to 0.02, or to the ``'TotalResolution'`` attribute if present.
    center
        The initial center value to use for fitting. If `None`, the center is
        automatically guessed if `fix_center` is `False`. Otherwise, the center is fixed
        to 0.
    fix_temp
        Whether to fix the temperature value during fitting. Defaults to `True`.
    fix_center
        Whether to fix the Fermi level during fitting. If `True`, the Fermi level is
        fixed to 0. Defaults to `False`.
    fix_resolution
        Whether to fix the resolution value during fitting. Defaults to `False`.
    bkg_slope
        Whether to include a linear background above the Fermi level. If `False`, the
        background above the Fermi level is fit with a constant. Defaults to `True`.
    plot
        Whether to plot the result of the fit. Defaults to `False`.
    ax
        The axes to plot the result on if ``plot`` is `True`. If `None`, the current
        axes are used.
    plot_fit_kwargs
        Additional keyword arguments for the fit plot, passed to
        :meth:`matplotlib.axes.Axes.plot`. Defaults to `None`.
    plot_data_kwargs
        Additional keyword arguments for the data plot, passed to
        :meth:`matplotlib.axes.Axes.plot`. Defaults to `None`.
    plot_line_kwargs
        Additional keyword arguments for the plot line that indicates the fitted center,
        passed to :meth:`matplotlib.axes.Axes.axvline`. Defaults to `None`.
    plot_span_kwargs
        Additional keyword arguments for the plot span that indicates the fitted FWHM,
        passed to :meth:`matplotlib.axes.Axes.axvspan`. Defaults to `None`.
    **kwargs
        Additional keyword arguments to :meth:`DataArray.xlm.modelfit`.

    Returns
    -------
    result : xarray.Dataset
        The result of the fit.

    """
    with xr.set_options(keep_attrs=True):
        data = darr.mean([d for d in darr.dims if d != "eV"])
        data_fit = (
            data.sel(eV=_range_slice_for_coord(data["eV"], eV_range))
            if eV_range is not None
            else data
        )

    if temp is None:
        temp = data.qinfo.get_value("sample_temp")
        if temp is None:
            raise ValueError(
                "Temperature not found in data attributes, please provide manually"
            )

    if resolution is None:
        if "TotalResolution" in data.attrs:
            resolution = float(data.attrs["TotalResolution"]) * 1e-3
        else:
            resolution = 0.02

    params = {
        "temp": {"value": temp, "vary": not fix_temp, "min": 0},
        "resolution": {"value": resolution, "vary": not fix_resolution, "min": 0},
    }

    if not bkg_slope:
        params["back1"] = {"value": 0, "vary": False}

    if center is not None:
        params["center"] = {"value": center}

    if fix_center:
        if center is None:
            params["center"] = {"value": 0.0}
        params["center"]["vary"] = False

    kwargs.setdefault("guess", True)
    fit_result = data_fit.xlm.modelfit(
        "eV",
        model=erlab.analysis.fit.models.FermiEdgeModel(),
        method=method,
        params=params,
        **kwargs,
    )
    if plot:
        if ax is None:
            ax = plt.gca()

        _plot_resolution_fit(
            ax=ax,
            data=data_fit,
            result=fit_result,
            fix_center=fix_center,
            plot_fit_kwargs=plot_fit_kwargs,
            plot_data_kwargs=plot_data_kwargs,
            plot_line_kwargs=plot_line_kwargs,
            plot_span_kwargs=plot_span_kwargs,
        )

    return fit_result


def _plot_resolution_fit(
    ax: matplotlib.axes.Axes,
    data: xr.DataArray,
    result: xr.Dataset,
    fix_center: bool,
    plot_fit_kwargs: dict[str, typing.Any] | None = None,
    plot_data_kwargs: dict[str, typing.Any] | None = None,
    plot_line_kwargs: dict[str, typing.Any] | None = None,
    plot_span_kwargs: dict[str, typing.Any] | None = None,
) -> None:
    """Plot the results of a single Fermi edge fit."""
    plot_data_kwargs = {} if plot_data_kwargs is None else plot_data_kwargs
    plot_fit_kwargs = {} if plot_fit_kwargs is None else plot_fit_kwargs
    plot_line_kwargs = {} if plot_line_kwargs is None else plot_line_kwargs
    plot_span_kwargs = {} if plot_span_kwargs is None else plot_span_kwargs

    plot_data_kwargs["ls"] = plot_data_kwargs.pop(
        "ls", plot_data_kwargs.pop("linestyle", "none")
    )
    plot_data_kwargs["ms"] = plot_data_kwargs.pop(
        "ms", plot_data_kwargs.pop("markersize", 5)
    )
    plot_data_kwargs["mew"] = plot_data_kwargs.pop(
        "mew", plot_data_kwargs.pop("markeredgewidth", 0.4)
    )
    plot_data_kwargs["mec"] = plot_data_kwargs.pop(
        "mec", plot_data_kwargs.pop("markeredgecolor", "0.5")
    )
    plot_data_kwargs["mfc"] = plot_data_kwargs.pop(
        "mfc", plot_data_kwargs.pop("markerfacecolor", "none")
    )
    plot_data_kwargs.setdefault("marker", ".")
    plot_data_kwargs.setdefault("label", "Data")
    ax.plot(data.eV, data, **plot_data_kwargs)

    plot_fit_kwargs["c"] = plot_fit_kwargs.pop(
        "c", plot_fit_kwargs.pop("color", "tab:red")
    )
    plot_fit_kwargs.setdefault("label", "Fit")
    ax.plot(result.modelfit_best_fit.eV, result.modelfit_best_fit, **plot_fit_kwargs)

    ax.set_ylabel("Intensity (arb. units)")
    if (data.eV[0] * data.eV[-1]) < 0:
        ax.set_xlabel("$E - E_F$ (eV)")
    else:
        ax.set_xlabel(r"$E_{kin}$ (eV)")

    coeffs = result.modelfit_coefficients
    modelresult: lmfit.model.ModelResult = result.modelfit_results.item()

    if hasattr(modelresult, "uvars"):
        center = modelresult.uvars["center"]
        resolution = modelresult.uvars["resolution"]
        center_bounds = (
            (center - resolution / 2).n,
            (center + resolution / 2).n,
        )

        center_repr = (
            f"$E_F = {center * 1e3:L}$ meV"
            if center.n < 0.1
            else f"$E_F = {center:L}$ eV"
        )
        resolution_repr = f"$\\Delta E = {resolution * 1e3:L}$ meV"

    else:
        center = coeffs.sel(param="center")
        resolution = coeffs.sel(param="resolution")
        center_bounds = (center - resolution / 2, center + resolution / 2)

        center_repr = (
            f"$E_F = {center * 1e3:.3f}$ meV"
            if center < 0.1
            else f"$E_F = {center:.6f}$ eV"
        )
        resolution_repr = f"$\\Delta E = {resolution * 1e3:.3f}$ meV"

    info_list: list[str] = [
        f"$T ={coeffs.sel(param='temp'):.3f}$ K",
        center_repr,
        resolution_repr,
    ]

    if fix_center:
        info_list.pop(1)

    fig = ax.figure
    if fig is not None:
        ax.text(
            0,
            0,
            "\n".join(info_list),
            ha="left",
            va="baseline",
            transform=ax.transAxes
            + matplotlib.transforms.ScaledTranslation(
                6 / 72, 6 / 72, fig.dpi_scale_trans
            ),
        )
    ax.set_xlim(data.eV[[0, -1]])
    ax.set_title("")

    plot_line_kwargs["c"] = plot_line_kwargs.pop(
        "c", plot_line_kwargs.pop("color", "k")
    )
    plot_line_kwargs["ls"] = plot_line_kwargs.pop(
        "ls", plot_line_kwargs.pop("linestyle", "--")
    )
    plot_line_kwargs["lw"] = plot_line_kwargs.pop(
        "lw", plot_line_kwargs.pop("linewidth", 0.4)
    )
    plot_line_kwargs.setdefault("alpha", 0.5)
    ax.axvline(coeffs.sel(param="center"), **plot_line_kwargs)

    plot_span_kwargs["fc"] = plot_span_kwargs.pop(
        "fc", plot_span_kwargs.pop("facecolor", "tab:red")
    )
    plot_span_kwargs["ec"] = plot_span_kwargs.pop(
        "ec", plot_span_kwargs.pop("edgecolor", "none")
    )
    plot_span_kwargs.setdefault("label", "FWHM")
    plot_span_kwargs.setdefault("alpha", 0.2)
    plot_span_kwargs.setdefault("ymin", -0.01)
    plot_span_kwargs.setdefault("ymax", 1.01)
    ax.axvspan(*center_bounds, **plot_span_kwargs)


def quick_resolution(
    darr: xr.DataArray, ax: matplotlib.axes.Axes | None = None, **kwargs
) -> xr.Dataset:
    """Fit a Fermi edge to the given data and plot the results.

    .. deprecated:: 3.5.1

        Use :func:`quick_fit` with ``plot=True`` instead.

    """
    warnings.warn(
        "erlab.analysis.gold.quick_resolution is deprecated, "
        "use erlab.analysis.gold.quick_fit with plot=True instead",
        FutureWarning,
        stacklevel=1,
    )

    kwargs["plot"] = True
    kwargs["ax"] = ax
    return quick_fit(darr, **kwargs)


def resolution(
    gold: xr.DataArray,
    angle_range: tuple[float, float],
    eV_range_edge: tuple[float, float],
    eV_range_fit: tuple[float, float] | None = None,
    bin_size: tuple[int, int] = (1, 1),
    degree: int = 4,
    use_step_edge: bool = False,
    method: str = "leastsq",
    plot: bool = True,
    parallel_kw: dict | None = None,
    scale_covar: bool = True,
    **kwargs,
) -> lmfit.model.ModelResult:  # pragma: no cover
    """Fit a Fermi edge and obtain the resolution from the corrected data.

    .. deprecated:: 3.5.1

        Use :func:`poly` and :func:`quick_fit` instead.

    """
    warnings.warn(
        "erlab.analysis.gold.resolution is deprecated, "
        "use erlab.analysis.gold.quick_fit instead",
        FutureWarning,
        stacklevel=1,
    )
    use_step_edge = _parse_deprecated_fast(use_step_edge, kwargs)
    _raise_unexpected_kwargs("resolution", kwargs)

    pol, gold_corr = typing.cast(
        "tuple[xr.Dataset, xr.DataArray]",
        poly(
            gold,
            angle_range=angle_range,
            eV_range=eV_range_edge,
            bin_size=bin_size,
            degree=degree,
            correct=True,
            use_step_edge=use_step_edge,
            method=method,
            plot=plot,
            parallel_kw=parallel_kw,
        ),
    )

    if eV_range_fit is None:
        eV_range_fit = tuple(r - np.mean(pol.best_fit) for r in eV_range_edge)
    del pol
    gold_roi = gold_corr.sel(
        alpha=_range_slice_for_coord(gold_corr["alpha"], angle_range)
    )
    edc_avg = gold_roi.mean("alpha").sel(
        eV=_range_slice_for_coord(gold_roi["eV"], eV_range_fit)
    )

    params = lmfit.create_params(
        temp={"value": gold_roi.attrs["sample_temp"], "vary": False},
        resolution={"value": 0.1, "vary": True, "min": 0},
    )
    model = erlab.analysis.fit.models.FermiEdgeModel()
    params = model.guess(edc_avg, x=edc_avg["eV"]).update(params)
    fit = erlab.analysis.fit.models.FermiEdgeModel().fit(
        edc_avg, x=edc_avg["eV"], params=params, method=method, scale_covar=scale_covar
    )
    if plot:
        plt.show()
        ax = plt.gca()
        gold_corr.qplot(ax=ax, cmap="copper", gamma=0.5)
        angle_lims = _range_limits_for_coord(gold_corr["alpha"], angle_range)
        eV_lims = _range_limits_for_coord(gold_roi["eV"], eV_range_fit)
        rect = matplotlib.patches.Rectangle(
            (angle_lims[0], eV_lims[0]),
            angle_lims[1] - angle_lims[0],
            eV_lims[1] - eV_lims[0],
            ec="w",
            alpha=0.5,
            lw=0.75,
            fc="none",
        )
        ax.add_patch(rect)
        ax.set_ylim(gold_corr.eV[[0, -1]])

        fit.plot(
            data_kws={"lw": 0.75, "ms": 4, "mfc": "w", "zorder": 0, "c": "0.4"},
            fit_kws={"c": "r", "lw": 1.5},
        )

    if hasattr(fit, "uvars"):
        center_uf = fit.uvars["center"]
        res_uf = fit.uvars["resolution"]
        print(f"center = {center_uf:S} eV\nresolution = {res_uf:.4S} eV")
    else:
        print(
            f"center = {fit.params['center'].value} eV\n"
            f"resolution = {fit.params['resolution'].value} eV"
        )

    return fit


def resolution_roi(
    gold_roi: xr.DataArray,
    eV_range: tuple[float, float],
    fix_temperature: bool = True,
    method: str = "leastsq",
    plot: bool = True,
    scale_covar: bool = True,
) -> lmfit.model.ModelResult:  # pragma: no cover
    """Fit a Fermi edge to the data and obtain the resolution.

    .. deprecated:: 3.5.1

        Use :func:`quick_fit` instead.

    """
    warnings.warn(
        "erlab.analysis.gold.resolution is deprecated, "
        "use erlab.analysis.gold.quick_fit instead",
        FutureWarning,
        stacklevel=1,
    )

    edc_avg = gold_roi.mean("alpha").sel(
        eV=_range_slice_for_coord(gold_roi["eV"], eV_range)
    )

    params = lmfit.create_params(
        temp={"value": gold_roi.attrs["sample_temp"], "vary": not fix_temperature},
        resolution={"value": 0.1, "vary": True, "min": 0},
    )
    model = erlab.analysis.fit.models.FermiEdgeModel()
    params = model.guess(edc_avg, x=edc_avg["eV"]).update(params)
    fit = model.fit(
        edc_avg,
        x=edc_avg["eV"],
        params=params,
        method=method,
        scale_covar=scale_covar,  # weights=1 / edc_stderr
    )
    if plot:
        ax = plt.gca()
        gold_roi.qplot(ax=ax, cmap="copper", gamma=0.5)
        eV_lims = _range_limits_for_coord(gold_roi["eV"], eV_range)
        ax.fill_between(
            gold_roi.alpha,
            eV_lims[0],
            eV_lims[1],
            ec="w",
            fc="none",
            alpha=0.4,
            lw=0.75,
        )
        ax.set_ylim(gold_roi.eV[[0, -1]])

        fit.plot(
            data_kws={"lw": 0.75, "ms": 4, "mfc": "w", "zorder": 0, "c": "0.4"},
            fit_kws={"c": "r", "lw": 1.5},
        )

    if hasattr(fit, "uvars"):
        center_uf = fit.uvars["center"]
        res_uf = fit.uvars["resolution"]
        print(f"center = {center_uf:S} eV\nresolution = {res_uf:.4S} eV")
    else:
        print(
            f"center = {fit.params['center'].value} eV\n"
            f"resolution = {fit.params['resolution'].value} eV"
        )

    return fit
