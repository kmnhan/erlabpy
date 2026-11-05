"""Bounded scalar-color snapshots for the current rendered editor selection."""

from __future__ import annotations

import typing
import weakref

import matplotlib.colors
import numpy as np

from erlab.interactive._figurecomposer._rendering import _OPERATION_ID_ATTR

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure

_FIT_STYLE_ATTR = "_erlab_figure_composer_fit_style"
_COLOR_PROPERTIES = (
    "color",
    "markerfacecolor",
    "markeredgecolor",
    "facecolor",
    "edgecolor",
)
_BASE_FIT_FIELDS = frozenset({"fit_data_kw", "fit_line_kw", "fit_band_kw"})
_COMPONENT_FIT_FIELDS = frozenset({"fit_component_kw", "fit_component_band_kw"})
_RGBA = tuple[float, float, float, float]


def _scalar_color(value: object) -> _RGBA | None:
    if value is None:
        return None
    if not isinstance(value, str):
        colors = np.asarray(value)
        if not colors.size:
            return (0.0, 0.0, 0.0, 0.0)
        if colors.ndim > 1:
            value = colors[0]
    red, green, blue, alpha = matplotlib.colors.to_rgba(
        typing.cast("typing.Any", value)
    )
    return (float(red), float(green), float(blue), float(alpha))


class _RenderedColorCache:
    def __init__(self) -> None:
        self._figure_ref: weakref.ReferenceType[Figure] | None = None
        self._scope: tuple[str, bool] | None = None
        self._component: str | None = None
        self._values: dict[tuple[str | None, str], _RGBA | None] = {}

    def invalidate(self) -> None:
        self._figure_ref = None
        self._scope = None
        self._component = None
        self._values.clear()

    def mark_rendered(self, figure: Figure) -> None:
        self.invalidate()
        self._figure_ref = weakref.ref(figure)

    def color(
        self,
        figure: Figure,
        operation_id: str,
        key: str,
        *,
        field: str | None = None,
        component: str | None = None,
    ) -> _RGBA | None:
        if (
            self._figure_ref is None
            or self._figure_ref() is not figure
            or key not in _COLOR_PROPERTIES
            or (
                field is not None
                and field not in _BASE_FIT_FIELDS | _COMPONENT_FIT_FIELDS
            )
        ):
            return None
        scope = (operation_id, field is not None)
        if self._scope != scope or (
            field in _COMPONENT_FIT_FIELDS and component != self._component
        ):
            self._scope = scope
            self._component = component if field in _COMPONENT_FIT_FIELDS else None
            self._values = {}
            self._values = self._collect(figure, operation_id, fit=field is not None)
        return self._values.get((field, key))

    def _collect(
        self, figure: Figure, operation_id: str, *, fit: bool
    ) -> dict[tuple[str | None, str], _RGBA | None]:
        # Five properties for one method, or five fit roles including one component:
        # at most 25 scalar values, with no artist or array references retained.
        values: dict[tuple[str | None, str], _RGBA | None] = {}
        for artist in figure.findobj():
            if getattr(artist, _OPERATION_ID_ATTR, None) != operation_id:
                continue
            field: str | None = None
            if fit:
                style = getattr(artist, _FIT_STYLE_ATTR, None)
                if not isinstance(style, tuple) or len(style) != 2:
                    continue
                field, component = style
                if not isinstance(field, str):
                    continue
                if not (
                    (field in _BASE_FIT_FIELDS and component is None)
                    or (
                        field in _COMPONENT_FIT_FIELDS
                        and component is not None
                        and component == self._component
                    )
                ):
                    continue
            for key in _COLOR_PROPERTIES:
                cache_key = (field, key)
                if cache_key in values:
                    continue
                getter = getattr(artist, f"get_{key}", None)
                if callable(getter):
                    try:
                        values[cache_key] = _scalar_color(getter())
                    except Exception:
                        # Unsupported custom artists must not hide other colors.
                        values[cache_key] = None
        return values
