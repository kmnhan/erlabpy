"""Shared image preparation for plot-array and plot-slices operations."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._text import _RawCode
from erlab.interactive.imagetool._colorlevels import (
    display_safe_values,
    image_display_dtype,
)

if typing.TYPE_CHECKING:
    import xarray as xr

    from erlab.interactive._figurecomposer._model._state import FigureOperationState


def _image_plot_kwargs(
    operation: FigureOperationState, *, code: bool = False
) -> dict[str, typing.Any]:
    kwargs: dict[str, typing.Any] = {}
    if operation.robust:
        kwargs["robust"] = True
    limit = operation.image_value_abs_limit
    if limit is not None:
        if code:
            kwargs["func"] = _RawCode(
                "lambda image: image.astype("
                "'float32' if image.dtype == 'float32' else 'float64', copy=False)"
                ".where(lambda values: "
                f"np.isfinite(values) & (abs(values) <= {limit!r}))"
            )
        else:

            def mask_values(image: xr.DataArray) -> xr.DataArray:
                prepared = image.astype(image_display_dtype(image.dtype), copy=False)
                values = prepared.values
                masked = display_safe_values(values, limit)
                return prepared if masked is values else prepared.copy(data=masked)

            kwargs["func"] = mask_values
    return kwargs
