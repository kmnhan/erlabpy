"""Narrow operation-facing context for Figure Composer rendering."""

from __future__ import annotations

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from erlab.interactive._figurecomposer._model._document import FigureRecipeContext

_T = typing.TypeVar("_T")


class _RenderDataGetter(typing.Protocol):
    def __call__(
        self,
        stage: str,
        plan: object,
        factory: Callable[[], _T],
    ) -> _T: ...


@dataclasses.dataclass(frozen=True)
class FigureRenderContext:
    """State and services available to operation renderers."""

    document: FigureRecipeContext
    source_display_name: Callable[[str], str]
    _render_data_getter: _RenderDataGetter
    _subplot_adjust_defaults_getter: Callable[[], dict[str, float]]
    _default_image_cmap_getter: Callable[[], str]

    def cached_data(
        self,
        stage: str,
        plan: object,
        factory: Callable[[], _T],
    ) -> _T:
        """Return prepared data scoped to the active render cache."""
        return self._render_data_getter(stage, plan, factory)

    def subplot_adjust_defaults(self) -> dict[str, float]:
        """Return subplot-adjust defaults for the current figure setup."""
        return self._subplot_adjust_defaults_getter()

    def default_image_cmap(self) -> str:
        """Return the configured image colormap for this render."""
        return self._default_image_cmap_getter()
