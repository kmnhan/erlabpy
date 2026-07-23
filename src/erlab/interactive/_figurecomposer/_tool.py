"""Manager-facing Figure Composer tool window."""

from __future__ import annotations

import base64
import binascii
import collections
import contextlib
import functools
import json
import keyword
import logging
import math
import re
import traceback
import typing
import unicodedata
import uuid
import weakref

from erlab.interactive.imagetool._provenance._model import (
    ScriptInput,
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    public_data,
    rebase_script_input_node_uids,
    script,
    to_replay_provenance_spec,
)
from erlab.interactive.imagetool._provenance._operations import (
    IselOperation,
    QSelAggregationOperation,
    QSelOperation,
)

# Matplotlib's Qt backend should see the qtpy-selected binding first.
# isort: off
from qtpy import QtCore, QtGui, QtWidgets

from matplotlib import colormaps
from matplotlib.figure import Figure
# isort: on

import numpy as np
import pydantic

import erlab
import erlab.interactive._figurecomposer._codegen
import erlab.interactive._figurecomposer._provenance
import erlab.interactive._figurecomposer._ui._toolbar_dialogs
import erlab.interactive._qt_state as _qt_state
from erlab.interactive._figurecomposer._defaults import (
    _figure_draw_context,
    _figure_style_context,
    _styled_rcparams_value,
    figure_options_context,
)
from erlab.interactive._figurecomposer._exceptions import FigureComposerInputError
from erlab.interactive._figurecomposer._model._axes import (
    _all_axes,
    _axes_expression_value,
)
from erlab.interactive._figurecomposer._model._custom_code import (
    _custom_code_bound_names,
)
from erlab.interactive._figurecomposer._model._document import (
    FigureDocument,
    FigureSourceAddResult,
)
from erlab.interactive._figurecomposer._model._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_axis_code_names,
    _gridspec_axis_display_name,
    _gridspec_axis_display_names,
    _gridspec_invalid_axes_ids,
    _gridspec_reserved_axis_code_names,
    _gridspec_valid_axes_ids,
)
from erlab.interactive._figurecomposer._model._operation_metadata import (
    declared_operation_source_names,
)
from erlab.interactive._figurecomposer._model._sources import (
    _FIGURE_CODE_RESERVED_NAMES,
    _default_plot_operation,
    _default_setup_for_data,
    _public_source_data,
    _selected_data,
    _source_display_label,
    _source_has_selection,
    _source_name,
    _source_selection,
    _source_unique_name,
    _source_with_selection,
    selection_dim_mode,
    selection_dim_value_text,
    selection_dim_width_text,
    selection_has_effect,
    selection_value_from_text,
    selection_width_from_text,
    shared_selection,
)
from erlab.interactive._figurecomposer._model._state import (
    FigureAxesSelectionState,
    FigureDataSelectionState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._norms import _matplotlib_cmap_name
from erlab.interactive._figurecomposer._operations import _registry
from erlab.interactive._figurecomposer._operations._plot_slices._cache import (
    _PlotSlicesSelectionCache,
)
from erlab.interactive._figurecomposer._operations._plot_slices._model import (
    _effective_extra_kwargs,
    _effective_slice_kwargs,
    _is_slice_kwarg_key,
    _operation_dim_names,
    _plot_slices_panel_keys,
    _plot_slices_shape,
    _selection_updates_from_kwargs,
    _selection_values,
    _selection_width,
)
from erlab.interactive._figurecomposer._operations._plot_slices._render import (
    _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    _PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
)
from erlab.interactive._figurecomposer._rendering import (
    _axes_from_selection,
    _iter_axes,
    _live_layout_axes,
    _render_into_figure,
    _render_preview,
    _rendered_output_figure,
)
from erlab.interactive._figurecomposer._text import _format_axes_tuple
from erlab.interactive._figurecomposer._ui._axes_widgets import (
    _AxesSelectorWidget,
    _gridspec_target_preview_descriptor,
    _GridSpecViewWidget,
    _subplot_target_preview_descriptor,
)
from erlab.interactive._figurecomposer._ui._figure_window import (
    _FigureComposerDisplayWindow,
)
from erlab.interactive._figurecomposer._ui._layout_panel import FigureLayoutPanel
from erlab.interactive._figurecomposer._ui._operation_editor import (
    COMMON_AXES_SECTION_TOOLTIP,
    COMMON_SOURCE_SECTION_TOOLTIP,
    FigureOperationEditor,
    OperationEditorBinding,
    OperationEditRequest,
    OperationRecipeEditRequest,
    StepSection,
)
from erlab.interactive._figurecomposer._ui._operation_panel import (
    FigureOperationAction,
    FigureOperationPanel,
    FigureOperationRow,
)
from erlab.interactive._figurecomposer._ui._source_inspector import (
    source_metadata_tooltip,
)
from erlab.interactive._figurecomposer._ui._source_panel import (
    FigureSourceDetail,
    FigureSourcePanel,
    FigureSourceRow,
    FigureSourceSelectionRow,
)

if typing.TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterable,
        Iterator,
        Mapping,
        MutableMapping,
        Sequence,
    )

    import xarray as xr
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

    from erlab.interactive._options.schema import AppOptions


_OPERATION_EDITOR_UPDATE_DELAY_MS = 25
_PREVIEW_RENDER_UPDATE_DELAY_MS = 50
_EDITOR_CONTROL_RENDER_UPDATE_DELAY_MS = 300
_FIGURE_RESIZE_RENDER_DELAY_MS = 120
_FIGURE_RESIZE_HISTORY_DELAY_MS = 250
_PREVIEW_PIXMAP_UPDATE_DELAY_MS = 250
_PERSISTED_PREVIEW_CACHE_ATTR = "figure_composer_preview_cache_png"
_PERSISTED_PREVIEW_CACHE_STALE_ATTR = "figure_composer_preview_cache_stale"
_PERSISTED_SELECTED_SOURCE_DATA_ATTR = "figure_composer_selected_source_data"
_PERSISTED_PREVIEW_CACHE_SIZE = (512, 384)
_PERSISTED_PREVIEW_CACHE_MAX_BYTES = 384_000
_RESTORE_OPERATION_EDITOR_KEY = "figure_composer_operation_editor"
_RESTORE_REDRAW_KEY = "figure_composer_restored_redraw"
_STEPS_CLIPBOARD_MIME = "application/x-erlab-figure-composer-steps+json"
_STEPS_CLIPBOARD_PAYLOAD_TYPE = "erlab.figure_composer.steps"
_STEPS_CLIPBOARD_PAYLOAD_VERSION = 1
logger = logging.getLogger(__name__)

_OPERATION_STATUS_LABELS = {
    "invalid_target": "Invalid target",
    "missing_source": "Missing source",
    "invalid_input": "Invalid input",
    "render_error": "Render error",
}


class _FigureComposerStepMimeData(QtCore.QMimeData):
    """Clipboard payload that can also carry live source data in one process."""

    def __init__(
        self,
        payload_text: str,
        step_code_text: str,
        source_data: Mapping[str, xr.DataArray],
        selection_base_data: Mapping[str, xr.DataArray],
        *,
        cut_source_tool_id: str | None = None,
    ) -> None:
        super().__init__()
        self.figure_composer_source_data: dict[str, xr.DataArray] = dict(source_data)
        self.figure_composer_selection_base_data: dict[str, xr.DataArray] = dict(
            selection_base_data
        )
        self.figure_composer_cut_source_tool_id = cut_source_tool_id
        self.setData(_STEPS_CLIPBOARD_MIME, payload_text.encode("utf-8"))
        self.setText(step_code_text)


def _step_clipboard_payload_text(
    operations: Sequence[FigureOperationState], sources: Sequence[FigureSourceState]
) -> str:
    return json.dumps(
        {
            "type": _STEPS_CLIPBOARD_PAYLOAD_TYPE,
            "version": _STEPS_CLIPBOARD_PAYLOAD_VERSION,
            "operations": [
                operation.model_dump(mode="json") for operation in operations
            ],
            "sources": [source.model_dump(mode="json") for source in sources],
        },
        indent=2,
    )


def _step_clipboard_code_text(
    tool: FigureComposerTool,
    operations: Sequence[FigureOperationState],
) -> str:
    lines: list[str] = []
    for operation in operations:
        if not operation.enabled:
            continue
        try:
            lines.extend(_registry.spec_for(operation.kind).code_lines(tool, operation))
        except Exception as exc:
            return (
                "# Could not generate Python code for the copied "
                f"Figure Composer steps: {exc}"
            )
    return "\n".join(lines)


def _step_clipboard_payload(
    mime: QtCore.QMimeData | None,
) -> (
    tuple[
        tuple[FigureOperationState, ...],
        tuple[FigureSourceState, ...],
        dict[str, xr.DataArray],
        dict[str, xr.DataArray],
    ]
    | None
):
    if mime is None:
        return None
    try:
        if mime.hasFormat(_STEPS_CLIPBOARD_MIME):
            payload_text = bytes(mime.data(_STEPS_CLIPBOARD_MIME).data()).decode(
                "utf-8"
            )
        elif mime.hasText():
            payload_text = mime.text()
        else:
            return None
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            return None
        if payload.get("type") != _STEPS_CLIPBOARD_PAYLOAD_TYPE:
            return None
        if payload.get("version") != _STEPS_CLIPBOARD_PAYLOAD_VERSION:
            return None
        raw_operations = payload.get("operations")
        raw_sources = payload.get("sources", [])
        if not isinstance(raw_operations, list) or not raw_operations:
            return None
        if not isinstance(raw_sources, list):
            return None
        operations = tuple(
            FigureOperationState.model_validate(operation)
            for operation in raw_operations
        )
        sources = tuple(
            FigureSourceState.model_validate(source) for source in raw_sources
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None

    source_data = getattr(mime, "figure_composer_source_data", {})
    if not isinstance(source_data, dict):
        source_data = {}
    selection_base_data = getattr(mime, "figure_composer_selection_base_data", {})
    if not isinstance(selection_base_data, dict):
        selection_base_data = {}
    return operations, sources, dict(source_data), dict(selection_base_data)


def _target_axes_count_text(count: int) -> str:
    suffix = "axis" if count == 1 else "axes"
    return f"{count} target {suffix}"


def _removed_axes_summary_text(count: int) -> str:
    return f"{_target_axes_count_text(count)} removed"


def _removed_axes_status_text(count: int, layout: str) -> str:
    verb = "was" if count == 1 else "were"
    return f"{_target_axes_count_text(count)} {verb} removed by the {layout}."


class FigureComposerTool(erlab.interactive.utils.ToolWindow[FigureRecipeState]):
    """Editable Matplotlib figure recipe window."""

    tool_name = "fig"
    manager_collection = "figures"

    StateModel = FigureRecipeState

    def __init__(
        self,
        data: xr.DataArray,
        *,
        recipe: FigureRecipeState | None = None,
        source_data: Mapping[str, xr.DataArray] | None = None,
    ) -> None:
        super().__init__()
        self._options_getter: Callable[[], AppOptions] | None = None
        self._updating_controls = False
        self._rendering = False
        self._auto_redraw_dirty = False
        self._operation_editor_update_pending = False
        self._preview_render_update_pending = False
        self._preview_render_update_generation = 0
        self._operation_render_errors: dict[str, str] = {}
        self._plot_slices_cache = _PlotSlicesSelectionCache()
        self._plot_slices_cache_source_revision = -1
        self._figure_resize_render_generation = 0
        self._figure_resize_history_pending = False
        self._figure_resize_history_timer = QtCore.QTimer(self)
        self._figure_resize_history_timer.setSingleShot(True)
        self._figure_resize_history_timer.setInterval(_FIGURE_RESIZE_HISTORY_DELAY_MS)
        self._figure_resize_history_timer.timeout.connect(
            self._flush_pending_figure_resize_history_write
        )
        self._figure_resize_history_state: FigureRecipeState | None = None
        self._figure_resize_history_source_data: (
            tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]] | None
        ) = None
        self._preview_pixmap_cache: QtGui.QPixmap | None = None
        self._preview_pixmap_generation = 0
        self._preview_thumbnail_cache: dict[
            tuple[int, int], tuple[int, QtGui.QPixmap]
        ] = {}
        self._preview_pixmap_stale = True
        self._preview_pixmap_update_pending = False
        self._preview_pixmap_update_generation = 0
        self._show_figure_window_generation = 0
        self._show_figure_window_pending = False
        self._closing = False
        self._source_refresh_available_callback: Callable[[str], bool] | None = None
        self._source_refresh_callback: Callable[[str], bool] | None = None
        self._source_refresh_label_callback: Callable[[str], str | None] | None = None
        self._source_reveal_available_callback: Callable[[str], bool] | None = None
        self._source_reveal_callback: Callable[[Sequence[str]], bool] | None = None
        self._source_add_available_callback: Callable[[], bool] | None = None
        self._source_add_callback: Callable[[], bool] | None = None
        self._source_drop_available_callback: (
            Callable[[QtCore.QMimeData], bool] | None
        ) = None
        self._source_drop_callback: Callable[[QtCore.QMimeData], bool] | None = None
        self._source_inspector_target: str | None = None
        self._projected_axis_name_sources: frozenset[str] | None = None
        initial_recipe = recipe or self._default_recipe(data)
        initial_source_data = dict(source_data or {})
        if source_data is None:
            if initial_recipe.primary_source in {
                source.name for source in initial_recipe.sources
            }:
                source_name = initial_recipe.primary_source
            else:
                source_name = _source_name(data)
            initial_source_data[source_name] = data
        if initial_recipe.primary_source not in initial_source_data:
            initial_source_data[initial_recipe.primary_source] = data
        self._document = FigureDocument(initial_recipe, source_data=initial_source_data)
        self._figure_window: _FigureComposerDisplayWindow | None = None
        self._subplot_adjust_dialog: QtWidgets.QDialog | None = None
        self._axes_customize_dialog: QtWidgets.QDialog | None = None
        self._connected_step_clipboard: QtGui.QClipboard | None = None
        self._step_clipboard_tool_id = uuid.uuid4().hex
        self._prev_source_data_states: collections.deque[
            tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
        ] = collections.deque(maxlen=self._prev_states.maxlen)
        self._next_source_data_states: collections.deque[
            tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
        ] = collections.deque(maxlen=self._next_states.maxlen)

        self._normalize_operation_source_selections()
        self._build_ui()
        self.setAcceptDrops(True)
        self._apply_recipe_to_controls()
        self._write_state()

    def set_options_getter(self, getter: Callable[[], AppOptions] | None) -> None:
        self._options_getter = getter

    def _plot_slices_cache_for_render(
        self,
    ) -> MutableMapping[Hashable, tuple[xr.DataArray, ...]] | None:
        source_revision = self._document.source_revision
        if source_revision != self._plot_slices_cache_source_revision:
            self._plot_slices_cache.clear()
            self._plot_slices_cache_source_revision = source_revision
        if any(
            operation.enabled
            and operation.kind == FigureOperationKind.CUSTOM
            and operation.trusted
            and bool(operation.code.strip())
            for operation in self._document.recipe.operations
        ):
            self._plot_slices_cache.clear()
            return None
        return self._plot_slices_cache

    @contextlib.contextmanager
    def _figure_options_context(self) -> Iterator[None]:
        options_model = None
        if self._options_getter is not None:
            with contextlib.suppress(Exception):
                options_model = self._options_getter()
        with figure_options_context(options_model):
            yield

    @staticmethod
    def _default_recipe(data: xr.DataArray) -> FigureRecipeState:
        source_name = _source_name(data)
        setup = _default_setup_for_data(data)
        source = FigureSourceState(name=source_name)
        return FigureRecipeState(
            setup=setup,
            sources=(source,),
            operations=(_default_plot_operation(source_name, data, setup=setup),),
            primary_source=source_name,
        )

    @classmethod
    def from_sources(
        cls,
        source_data: Mapping[str, xr.DataArray],
        *,
        sources: tuple[FigureSourceState, ...],
        operations: tuple[FigureOperationState, ...] | None = None,
        setup: FigureSubplotsState | None = None,
        primary_source: str | None = None,
    ) -> FigureComposerTool:
        if not source_data:
            raise ValueError("At least one source is required")
        primary = primary_source or next(iter(source_data))
        resolved_setup = setup or FigureSubplotsState()
        recipe = FigureRecipeState(
            setup=resolved_setup,
            sources=sources,
            operations=operations
            if operations is not None
            else tuple(
                _default_plot_operation(name, data, setup=resolved_setup)
                for name, data in source_data.items()
            ),
            primary_source=primary,
        )
        return cls(source_data[primary], recipe=recipe, source_data=source_data)

    @property
    def figure_window(self) -> _FigureComposerDisplayWindow:
        return self._ensure_figure_window()

    @property
    def figure(self) -> Figure:
        return self.figure_window.figure

    @property
    def canvas(self) -> FigureCanvas:
        return self.figure_window.canvas

    def _ensure_figure_window(self) -> _FigureComposerDisplayWindow:
        if self._figure_window is None or not erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            self._figure_window = _FigureComposerDisplayWindow(
                self._document.recipe.setup,
                export_callback=lambda: self.export_figure(),
                subplot_adjust_callback=lambda: self._show_subplot_adjust_dialog(),
                axes_customize_callback=lambda: self._show_axes_customize_dialog(),
                show_composer_callback=lambda: self._show_composer_from_figure_window(),
                navigation_callback=self._figure_window_navigation_changed,
                colorbar_callback=self._figure_window_colorbar_changed,
                undo_callback=self.undo,
                redo_callback=self.redo,
                undoable_callback=lambda: self.undoable,
                redoable_callback=lambda: self.redoable,
                source_drop_available_callback=self._source_drop_available,
                source_drop_callback=self._add_sources_from_mime,
            )
            window_ref = weakref.ref(self._figure_window)
            tool_ref = weakref.ref(self)

            def figure_window_destroyed(
                _obj: QtCore.QObject | None = None,
                *,
                owner_ref: weakref.ReferenceType[FigureComposerTool] = tool_ref,
                ref: weakref.ReferenceType[_FigureComposerDisplayWindow] = window_ref,
            ) -> None:
                owner = owner_ref()
                if owner is not None and erlab.interactive.utils.qt_is_valid(owner):
                    owner._figure_window_destroyed(ref)

            self._figure_window.sigCanvasSizeChanged.connect(
                self._figure_window_canvas_size_changed
            )
            self._figure_window.destroyed.connect(figure_window_destroyed)
        self._figure_window.setWindowTitle(self._figure_window_title())
        self._configure_managed_secondary_window(self._figure_window)
        return self._figure_window

    def _figure_window_destroyed(
        self, window_ref: weakref.ReferenceType[_FigureComposerDisplayWindow]
    ) -> None:
        window = window_ref()
        if window is not None and self._figure_window is window:
            self._figure_window = None

    def _update_history_actions(self) -> None:
        super()._update_history_actions()
        window = getattr(self, "_figure_window", None)
        if window is not None and erlab.interactive.utils.qt_is_valid(window):
            window.toolbar.set_history_buttons()

    def _disconnect_figure_window(self, window: _FigureComposerDisplayWindow) -> None:
        with contextlib.suppress(TypeError, RuntimeError):
            window.sigCanvasSizeChanged.disconnect(
                self._figure_window_canvas_size_changed
            )

    def _figure_window_title(self) -> str:
        display_name = self._tool_display_name
        if display_name:
            return f"{self.tool_name}: {display_name}"
        title = self.windowTitle()
        return title if title and title != self.tool_name else "Figure"

    def _managed_secondary_windows(
        self,
    ) -> tuple[tuple[QtWidgets.QWidget, str], ...]:
        if self._figure_window is None or not erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            return ()
        return ((self._figure_window, self._figure_window_title()),)

    @QtCore.Slot()
    def _show_figure_window_requested(self) -> None:
        self._request_show_figure_window(activate=True)

    def _show_composer_from_figure_window(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def show_figure_window(self, *, activate: bool = True) -> None:
        figure_window = self.figure_window
        figure_window.show_for_setup(
            self._document.recipe.setup, self._figure_window_title(), activate=activate
        )
        self._configure_managed_secondary_window(figure_window)
        self._cancel_preview_render_update()
        self._redraw_plot(show_window=True)

    def _auto_redraw_enabled(self) -> bool:
        check = getattr(self, "auto_redraw_check", None)
        return not isinstance(check, QtWidgets.QCheckBox) or check.isChecked()

    def _redraw_plot(
        self, *, show_window: bool | None = None, emit_info: bool = False
    ) -> None:
        self._cancel_preview_render_update()
        if show_window is None:
            _render_preview(self)
        else:
            _render_preview(self, show_window=show_window)
        self._auto_redraw_dirty = False
        if emit_info:
            self.sigInfoChanged.emit()

    def _maybe_redraw_plot(self, *, show_window: bool | None = None) -> bool:
        if not self._auto_redraw_enabled():
            self._auto_redraw_dirty = True
            self._cancel_preview_render_update()
            self._mark_preview_pixmap_stale()
            return False
        self._redraw_plot(show_window=show_window)
        return True

    @QtCore.Slot(bool)
    def _auto_redraw_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._cancel_preview_render_update()
            return
        if self._auto_redraw_dirty:
            self._redraw_plot(emit_info=True)

    @QtCore.Slot()
    def _redraw_plot_requested(self) -> None:
        self._redraw_plot(emit_info=True)

    def _request_show_figure_window(self, *, activate: bool) -> None:
        if self._closing:
            return
        self._show_figure_window_generation += 1
        self._show_figure_window_pending = True
        generation = self._show_figure_window_generation
        erlab.interactive.utils.single_shot(
            self,
            0,
            lambda: self._run_requested_show_figure_window(generation, activate),
        )

    def _run_requested_show_figure_window(
        self, generation: int, activate: bool
    ) -> None:
        if generation != self._show_figure_window_generation:
            return
        self._show_figure_window_pending = False
        if self._closing or not self.isVisible():
            return
        self.show_figure_window(activate=activate)

    def _cancel_queued_show_figure_window(self) -> None:
        self._show_figure_window_generation += 1
        self._show_figure_window_pending = False

    @QtCore.Slot(float, float)
    def _figure_window_canvas_size_changed(
        self, width_inches: float, height_inches: float
    ) -> None:
        if self._updating_controls:
            return
        if self._set_recipe_figsize_from_canvas(
            width_inches,
            height_inches,
            draw=False,
            emit_info=False,
            history="deferred",
        ):
            self._queue_figure_resize_render()

    def _figure_window_navigation_changed(
        self, changes: Mapping[object, tuple[bool, bool]]
    ) -> None:
        if self._updating_controls or self._rendering:
            return
        layout_axes = _live_layout_axes(self)
        if layout_axes is None:
            return

        operations = list(self._document.recipe.operations)
        changed_operation_ids: set[str] = set()
        changed = False
        for axis, (x_changed, y_changed) in changes.items():
            selection = self._navigation_axis_selection(layout_axes, axis)
            if selection is None:
                continue
            for method_name, limits in self._navigation_axis_limit_updates(
                axis, x_changed=x_changed, y_changed=y_changed
            ):
                index = self._matching_navigation_limit_operation_index(
                    operations, method_name, selection
                )
                updated = FigureOperationState.method(
                    family=FigureMethodFamily.AXES,
                    name=method_name,
                    axes=selection,
                    args=limits,
                )
                if index is None:
                    operations.append(updated)
                    changed_operation_ids.add(updated.operation_id)
                    changed = True
                    continue
                previous = operations[index]
                updated = previous.model_copy(update={"method_args": limits})
                if previous.model_dump() == updated.model_dump():
                    continue
                operations[index] = updated
                changed_operation_ids.add(updated.operation_id)
                changed = True
        if not changed:
            return

        self._apply_live_figure_operation_updates(operations, changed_operation_ids)

    def _figure_window_colorbar_changed(
        self, changes: Mapping[object, tuple[float, float]]
    ) -> None:
        if self._updating_controls or self._rendering:
            return
        operations = list(self._document.recipe.operations)
        changed_operation_ids: set[str] = set()
        changed = False
        for mappable, clim in changes.items():
            target = self._image_mappable_target(mappable, operations)
            if target is None:
                continue
            index, operation, panel_key = target
            updated = self._operation_with_colorbar_clim(operation, panel_key, clim)
            if operation.model_dump() == updated.model_dump():
                continue
            operations[index] = updated
            changed_operation_ids.add(updated.operation_id)
            changed = True
        if not changed:
            return

        self._apply_live_figure_operation_updates(operations, changed_operation_ids)

    def _apply_live_figure_operation_updates(
        self,
        operations: Sequence[FigureOperationState],
        changed_operation_ids: set[str],
    ) -> None:
        current = self._current_operation()
        self._document.replace_operations(operations)
        self._refresh_operation_list()
        if current is not None and current[0] < len(operations):
            self._set_current_operation_row_silent(current[0])
        current_operation = self._current_operation()
        self._sync_axes_selector()
        self._update_step_action_buttons()
        self._refresh_step_section_button_texts()
        self._update_source_status(current_operation[1] if current_operation else None)
        self._refresh_source_detail_panel()
        if current_operation is not None and (
            current_operation[1].operation_id in changed_operation_ids
        ):
            self._update_operation_editor_safely()
        self._mark_preview_pixmap_stale()
        self.sigInfoChanged.emit()
        self._write_state()

    def _image_mappable_target(
        self,
        mappable: object,
        operations: Sequence[FigureOperationState],
    ) -> tuple[int, FigureOperationState, tuple[int, int]] | None:
        operation_id = getattr(mappable, _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR, None)
        panel_key = getattr(mappable, _PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR, None)
        if not isinstance(operation_id, str):
            return None
        if (
            not isinstance(panel_key, tuple)
            or len(panel_key) != 2
            or not all(isinstance(value, int) for value in panel_key)
        ):
            return None
        for index, operation in enumerate(operations):
            if operation.operation_id == operation_id and operation.kind in (
                FigureOperationKind.PLOT_ARRAY,
                FigureOperationKind.PLOT_SLICES,
            ):
                return index, operation, typing.cast("tuple[int, int]", panel_key)
        return None

    def _try_update_live_colormap(
        self,
        index: int,
        previous: FigureOperationState,
        updated: FigureOperationState,
    ) -> bool:
        """Update verified live image artists for one cmap-only recipe edit."""
        if (
            self._closing
            or self._rendering
            or self._auto_redraw_dirty
            or self._preview_render_update_pending
            or not self._auto_redraw_enabled()
        ):
            return False
        if (
            updated.kind
            not in (FigureOperationKind.PLOT_ARRAY, FigureOperationKind.PLOT_SLICES)
            or updated.model_copy(update={"cmap": previous.cmap}) != previous
            or "cmap" in updated.extra_kwargs
            or updated.operation_id in self._operation_render_errors
            or self.operation_editor.has_input_error(updated)
            or any(
                operation.enabled
                and operation.kind
                in (FigureOperationKind.METHOD, FigureOperationKind.CUSTOM)
                for operation in self._document.recipe.operations[index + 1 :]
            )
        ):
            return False

        try:
            if updated.kind == FigureOperationKind.PLOT_ARRAY:
                expected_panel_keys = {(0, 0)}
            else:
                shape = _plot_slices_shape(self._document, updated)
                if (
                    not shape.valid
                    or shape.plot_ndim != 2
                    or updated.panel_styles_enabled
                    or bool(updated.panel_styles)
                ):
                    return False
                expected_panel_keys = {
                    (key.map_index, key.slice_index)
                    for key in _plot_slices_panel_keys(
                        self._document, self._source_display_name, updated
                    )
                }
        except Exception:
            return False

        window = self._figure_window
        if (
            window is None
            or not erlab.interactive.utils.qt_is_valid(window)
            or not window.isVisible()
        ):
            return False
        canvas = window.canvas
        if not erlab.interactive.utils.qt_is_valid(canvas):
            return False

        tagged_mappables: list[tuple[typing.Any, tuple[int, int]]] = []
        for axis in window.figure.axes:
            for mappable in (*axis.images, *axis.collections):
                target = self._image_mappable_target(
                    mappable, self._document.recipe.operations
                )
                if (
                    target is not None
                    and target[1].operation_id == updated.operation_id
                ):
                    tagged_mappables.append((mappable, target[2]))
        if (
            len(tagged_mappables) != len(expected_panel_keys)
            or {panel_key for _mappable, panel_key in tagged_mappables}
            != expected_panel_keys
        ):
            return False

        try:
            cmap_name = updated.cmap
            if cmap_name is None:
                cmap_name = str(self._editor_styled_rcparams_value("image.cmap"))
            cmap = colormaps.get_cmap(_matplotlib_cmap_name(cmap_name))
            for mappable, _panel_key in tagged_mappables:
                mappable.set_cmap(cmap)
            self._mark_preview_pixmap_stale()
            canvas.draw_idle()
        except Exception:
            return False
        return True

    def _operation_with_colorbar_clim(
        self,
        operation: FigureOperationState,
        panel_key: tuple[int, int],
        clim: tuple[float, float],
    ) -> FigureOperationState:
        if operation.kind == FigureOperationKind.PLOT_ARRAY:
            if panel_key != (0, 0):
                return operation
            vmin, vmax = clim
            return operation.model_copy(update={"vmin": vmin, "vmax": vmax})
        panel_keys = _plot_slices_panel_keys(
            self._document, self._source_display_name, operation
        )
        valid_keys = {(key.map_index, key.slice_index) for key in panel_keys}
        if panel_key not in valid_keys:
            return operation
        vmin, vmax = clim
        if len(panel_keys) == 1 or (
            operation.same_limits is not False and not operation.panel_styles_enabled
        ):
            return operation.model_copy(update={"vmin": vmin, "vmax": vmax})

        styles = {
            (style.map_index, style.slice_index): style
            for style in operation.panel_styles
            if (style.map_index, style.slice_index) in valid_keys
        }
        current_style = styles.get(
            panel_key,
            FigurePlotSlicesPanelStyleState(
                map_index=panel_key[0],
                slice_index=panel_key[1],
            ),
        )
        styles[panel_key] = current_style.model_copy(
            update={"vmin": vmin, "vmax": vmax}
        )
        panel_styles = tuple(
            sorted(
                styles.values(), key=lambda style: (style.map_index, style.slice_index)
            )
        )
        return operation.model_copy(
            update={"panel_styles_enabled": True, "panel_styles": panel_styles}
        )

    def _navigation_axis_selection(
        self,
        layout_axes: object,
        axis: object,
    ) -> FigureAxesSelectionState | None:
        if isinstance(layout_axes, dict):
            for axes_id, candidate in layout_axes.items():
                if candidate is axis:
                    return FigureAxesSelectionState(axes_ids=(str(axes_id),))
            return None
        shape = getattr(layout_axes, "shape", ())
        if not isinstance(shape, tuple) or len(shape) != 2:
            return None
        row_count, column_count = typing.cast("tuple[int, int]", shape)
        layout_array = typing.cast("typing.Any", layout_axes)
        for row in range(row_count):
            for col in range(column_count):
                if layout_array[row, col] is axis:
                    return FigureAxesSelectionState(axes=((row, col),))
        return None

    @staticmethod
    def _navigation_axis_limit_updates(
        axis: object, *, x_changed: bool, y_changed: bool
    ) -> tuple[tuple[str, tuple[float, float]], ...]:
        updates: list[tuple[str, tuple[float, float]]] = []
        if x_changed and hasattr(axis, "get_xlim"):
            xlim = typing.cast("typing.Any", axis).get_xlim()
            updates.append(("set_xlim", (float(xlim[0]), float(xlim[1]))))
        if y_changed and hasattr(axis, "get_ylim"):
            ylim = typing.cast("typing.Any", axis).get_ylim()
            updates.append(("set_ylim", (float(ylim[0]), float(ylim[1]))))
        return tuple(updates)

    @staticmethod
    def _matching_navigation_limit_operation_index(
        operations: Sequence[FigureOperationState],
        method_name: str,
        selection: FigureAxesSelectionState,
    ) -> int | None:
        selection_payload = selection.model_dump()
        for index in range(len(operations) - 1, -1, -1):
            operation = operations[index]
            if (
                operation.kind == FigureOperationKind.METHOD
                and operation.method_family == FigureMethodFamily.AXES
                and operation.method_name == method_name
                and operation.axes.model_dump() == selection_payload
            ):
                return index
        return None

    def _queue_figure_resize_render(self) -> None:
        self._figure_resize_render_generation += 1
        generation = self._figure_resize_render_generation
        erlab.interactive.utils.single_shot(
            self,
            _FIGURE_RESIZE_RENDER_DELAY_MS,
            lambda: self._run_queued_figure_resize_render(generation),
        )

    def _run_queued_figure_resize_render(self, generation: int) -> None:
        if (
            generation != self._figure_resize_render_generation
            or self._closing
            or not erlab.interactive.utils.qt_is_valid(self)
        ):
            return
        window = self._figure_window
        if window is not None and erlab.interactive.utils.qt_is_valid(window):
            try:
                canvas = window.canvas
            except RuntimeError:
                canvas = None
            if (
                canvas is not None
                and erlab.interactive.utils.qt_is_valid(canvas)
                and window.isVisible()
            ):
                if self._rendering:
                    self._queue_figure_resize_render()
                    return
                canvas.draw_idle()
        self.sigInfoChanged.emit()

    def _sync_recipe_figsize_to_canvas(self, *, draw: bool, emit_info: bool) -> bool:
        canvas_size = self.canvas.size()
        if canvas_size.isEmpty():
            return False
        dpi = float(typing.cast("typing.Any", self.figure)._original_dpi)
        if dpi <= 0.0:
            return False
        return self._set_recipe_figsize_from_canvas(
            canvas_size.width() / dpi,
            canvas_size.height() / dpi,
            draw=draw,
            emit_info=emit_info,
        )

    def _sync_figure_window_to_recipe_setup(self) -> None:
        window = self._figure_window
        if window is None or not erlab.interactive.utils.qt_is_valid(window):
            return
        window.resize_to_setup(self._document.recipe.setup)
        with contextlib.suppress(RuntimeError):
            window.canvas.flush_events()

    def _set_recipe_figsize_from_canvas(
        self,
        width_inches: float,
        height_inches: float,
        *,
        draw: bool,
        emit_info: bool,
        history: typing.Literal["immediate", "deferred"] = "immediate",
    ) -> bool:
        setup = self._document.recipe.setup
        if math.isclose(width_inches, setup.figsize[0], abs_tol=0.005) and math.isclose(
            height_inches, setup.figsize[1], abs_tol=0.005
        ):
            return False
        figsize = (round(width_inches, 4), round(height_inches, 4))
        if not self._document.replace_setup(
            setup.model_copy(update={"figsize": figsize})
        ):
            return False
        self._mark_preview_pixmap_stale()
        self.figure.set_size_inches(figsize, forward=False)
        self.layout_panel.set_setup(
            self._document.recipe.setup,
            reserved_names=self._document.source_names(),
        )
        if draw:
            self.canvas.draw()
        if emit_info:
            self.sigInfoChanged.emit()
        if history == "deferred":
            self._queue_figure_resize_history_write()
        else:
            self._write_state()
        return True

    def _show_subplot_adjust_dialog(self) -> None:
        erlab.interactive._figurecomposer._ui._toolbar_dialogs.show_subplot_adjust_dialog(
            self
        )

    def _show_axes_customize_dialog(self) -> None:
        erlab.interactive._figurecomposer._ui._toolbar_dialogs.show_axes_customize_dialog(
            self
        )

    def _hide_figure_window(self) -> None:
        if self._figure_window is not None and erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            self._figure_window.hide()

    def _close_figure_window(self) -> None:
        erlab.interactive._figurecomposer._ui._toolbar_dialogs.close_toolbar_dialogs(
            self
        )
        if self._figure_window is None or not erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            self._figure_window = None
            return
        window = self._figure_window
        self._figure_window = None
        self._disconnect_figure_window(window)
        window.close_from_owner()

    def showEvent(self, event: QtGui.QShowEvent | None) -> None:
        if event is not None:
            super().showEvent(event)
        self.operation_editor.refresh_current_background()
        self._request_show_figure_window(activate=False)

    def hideEvent(self, event: QtGui.QHideEvent | None) -> None:
        self._cancel_queued_show_figure_window()
        self._hide_figure_window()
        if event is not None:
            super().hideEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self.operation_editor.flush_pending_commits()
        self._flush_pending_figure_resize_history_write()
        self._closing = True
        self._cancel_queued_show_figure_window()
        self._figure_resize_render_generation += 1
        self._preview_render_update_generation += 1
        self._preview_render_update_pending = False
        self._preview_pixmap_update_generation += 1
        self._preview_pixmap_update_pending = False
        self._clear_preview_pixmap_cache(stale=False)
        self.operation_panel.release()
        self.layout_panel.release()
        self._disconnect_step_clipboard()
        self._close_figure_window()
        if event is not None:
            super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(4)

        self.editor_tabs = QtWidgets.QTabWidget(root)
        self.editor_tabs.setObjectName("figureComposerEditorTabs")

        action_layout = QtWidgets.QVBoxLayout()
        action_layout.setSpacing(2)

        self.operation_panel = FigureOperationPanel(
            self.editor_tabs,
            tuple(
                FigureOperationAction(
                    action_spec.action_id, action_spec.text, action_spec.tooltip
                )
                for action_spec in _registry.add_step_actions()
            ),
        )
        self.operation_editor: FigureOperationEditor = self.operation_panel.editor
        self.operation_panel.add_requested.connect(self._add_operation)
        self.operation_panel.copy_requested.connect(self._copy_selected_operations)
        self.operation_panel.cut_requested.connect(self._cut_selected_operations)
        self.operation_panel.paste_requested.connect(
            self._paste_operations_from_clipboard
        )
        self.operation_panel.delete_requested.connect(self._remove_current_operation)
        self.operation_panel.duplicate_requested.connect(
            self._duplicate_current_operation
        )
        self.operation_panel.move_requested.connect(self._move_current_operation)
        self.operation_panel.reorder_requested.connect(self._operation_list_reordered)
        self.operation_panel.enabled_requested.connect(
            self._operation_enabled_requested
        )
        self.operation_panel.selection_changed.connect(
            self._operation_selection_changed
        )
        self.show_figure_button = QtWidgets.QPushButton("Show Plot", root)
        self.show_figure_button.setObjectName("figureComposerShowFigureButton")
        self.show_figure_button.setToolTip(
            "Open or raise the separate Matplotlib plot window."
        )
        self.show_figure_button.clicked.connect(self._show_figure_window_requested)
        copy_button = QtWidgets.QPushButton("Copy Python", root)
        copy_button.setToolTip("Copy standalone Python code for this figure recipe.")
        copy_button.clicked.connect(self.copy_code)
        export_button = QtWidgets.QPushButton("Export", root)
        export_button.setToolTip("Save the rendered Matplotlib figure to a file.")
        export_button.clicked.connect(self.export_figure)

        output_action_layout = QtWidgets.QHBoxLayout()
        output_action_layout.setSpacing(4)
        output_action_layout.addWidget(self.show_figure_button)
        output_action_layout.addWidget(copy_button)
        output_action_layout.addWidget(export_button)
        output_action_layout.addStretch(1)
        self.auto_redraw_check = QtWidgets.QCheckBox("Auto redraw", root)
        self.auto_redraw_check.setObjectName("figureComposerAutoRedrawCheck")
        self.auto_redraw_check.setToolTip(
            "Automatically redraw the plot after recipe changes."
        )
        self.auto_redraw_check.setChecked(True)
        self.auto_redraw_check.toggled.connect(self._auto_redraw_toggled)
        output_action_layout.addWidget(self.auto_redraw_check)
        self.redraw_plot_button = QtWidgets.QToolButton(root)
        self.redraw_plot_button.setObjectName("figureComposerRedrawPlotButton")
        self.redraw_plot_button.setAccessibleName("Redraw Plot")
        self.redraw_plot_button.setToolTip("Redraw and update the plot now.")
        self.redraw_plot_button.setIcon(
            erlab.interactive.utils.qtawesome.icon("ph.arrow-clockwise")
        )
        self.redraw_plot_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.redraw_plot_button.setAutoRaise(True)
        self.redraw_plot_button.clicked.connect(self._redraw_plot_requested)
        output_action_layout.addWidget(self.redraw_plot_button)
        action_layout.addLayout(output_action_layout)
        root_layout.addLayout(action_layout)
        root_layout.addWidget(self.editor_tabs, 1)

        self.source_panel = FigureSourcePanel(self.editor_tabs)
        self.source_panel.add_requested.connect(self._request_add_sources_from_button)
        self.source_panel.remove_requested.connect(self._remove_source_names)
        self.source_panel.refresh_requested.connect(self._refresh_source_names)
        self.source_panel.refresh_all_requested.connect(
            self._refresh_all_sources_from_button
        )
        self.source_panel.reveal_requested.connect(self._reveal_source_names)
        self.source_panel.rename_requested.connect(self._rename_source_requested)
        self.source_panel.duplicate_requested.connect(self._duplicate_source_names)
        self.source_panel.move_requested.connect(self._move_source_names)
        self.source_panel.reorder_requested.connect(self._source_list_reordered)
        self.source_panel.selection_changed.connect(
            self._source_panel_selection_changed
        )
        self.source_panel.selection_dimension_requested.connect(
            self._update_selected_source_dimension
        )
        self.source_panel.set_drop_handlers(
            self._source_drop_available, self._add_sources_from_mime
        )
        self.source_panel.install_drop_target(self)

        self.target_axes_page = self.operation_editor.create_page(
            "figureComposerTargetAxesPage"
        )
        target_axes_layout = QtWidgets.QVBoxLayout(self.target_axes_page)
        target_axes_layout.setContentsMargins(6, 6, 6, 6)
        target_axes_layout.setSpacing(4)
        self.axes_selector = _AxesSelectorWidget(self.target_axes_page)
        self.axes_selector.sigSelectionChanged.connect(self._axes_selection_changed)
        target_axes_layout.addWidget(self.axes_selector)
        self.operation_panel.install_target_delegate(self.axes_selector)
        self.gridspec_axes_selector = _GridSpecViewWidget(
            self.target_axes_page, mode="select"
        )
        self.gridspec_axes_selector.sigSelectionChanged.connect(
            self._gridspec_axes_selection_changed
        )
        target_axes_layout.addWidget(self.gridspec_axes_selector)
        self.target_axes_status_label = QtWidgets.QLabel(self.target_axes_page)
        self.target_axes_status_label.setObjectName("figureComposerAxesStatus")
        self.target_axes_status_label.setWordWrap(True)
        target_axes_layout.addWidget(self.target_axes_status_label)
        target_axes_action_layout = QtWidgets.QHBoxLayout()
        self.use_all_axes_button = QtWidgets.QPushButton(
            "Use All Axes", self.target_axes_page
        )
        self.use_all_axes_button.setToolTip(
            "Retarget the selected step to every axis in the current figure grid."
        )
        self.use_all_axes_button.clicked.connect(
            self._target_current_operation_all_axes
        )
        self.keep_valid_axes_button = QtWidgets.QPushButton(
            "Drop Removed Axes", self.target_axes_page
        )
        self.keep_valid_axes_button.setToolTip(
            "Keep only selected axes that still exist in the current figure grid."
        )
        self.keep_valid_axes_button.clicked.connect(
            self._target_current_operation_valid_axes
        )
        target_axes_action_layout.addWidget(self.use_all_axes_button)
        target_axes_action_layout.addWidget(self.keep_valid_axes_button)
        target_axes_action_layout.addStretch(1)
        target_axes_layout.addLayout(target_axes_action_layout)
        self.axes_expression_edit = QtWidgets.QLineEdit(self.target_axes_page)
        self.axes_expression_edit.setPlaceholderText("Advanced: axs[0, :] or axs[:, 0]")
        self.axes_expression_edit.setToolTip(
            "Optional advanced axes expression used verbatim in copied Python code."
        )
        self.axes_expression_edit.editingFinished.connect(self._axes_expression_changed)
        target_axes_layout.addWidget(self.axes_expression_edit)
        target_axes_layout.addStretch(1)
        self.layout_panel = FigureLayoutPanel(self.editor_tabs)
        self.layout_panel.setup_requested.connect(self._layout_setup_requested)
        self.layout_panel.layout_mode_requested.connect(self._layout_mode_requested)
        self.axes_selector.sigAddRowRequested.connect(
            functools.partial(self._grow_subplot_grid, "row")
        )
        self.axes_selector.sigAddColumnRequested.connect(
            functools.partial(self._grow_subplot_grid, "column")
        )

        sources_index = self.editor_tabs.addTab(self.source_panel, "Sources")
        self.editor_tabs.setTabToolTip(
            sources_index, "Named data variables captured for this figure."
        )
        layout_index = self.editor_tabs.addTab(self.layout_panel, "Layout")
        self.editor_tabs.setTabToolTip(
            layout_index, "Subplot grid, figure size, and shared axes."
        )
        recipe_index = self.editor_tabs.addTab(self.operation_panel, "Recipe")
        self.editor_tabs.setTabToolTip(
            recipe_index,
            "Ordered plotting steps and controls for the selected step.",
        )
        self.editor_tabs.setCurrentWidget(self.operation_panel)

        self.operation_editor.track_controls(self.layout_panel)
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            clipboard = application.clipboard()
            if clipboard is not None:
                clipboard.dataChanged.connect(self._update_step_action_buttons)
                self._connected_step_clipboard = clipboard

        self.operation_editor.bind(
            OperationEditorBinding(
                context=self._document,
                current_operation_id=self._current_operation_id,
                editable_operation_ids=self._editable_operation_ids,
                updates_allowed=lambda: (
                    not self._updating_controls and not self._closing
                ),
                selected_axes_state=self._selected_axes_state,
                source_display_name=self._source_display_name,
                source_tooltip=self._source_tooltip,
                first_live_axis=self._editor_first_live_axis,
                subplot_parameter_default=self._editor_subplot_parameter_default,
                rendered_value=self._editor_rendered_value,
                styled_rcparams_value=self._editor_styled_rcparams_value,
            )
        )
        self.operation_editor.edit_requested.connect(self._apply_operation_edit_request)
        self.operation_editor.validation_changed.connect(
            self._operation_editor_validation_changed
        )

        self.setCentralWidget(root)
        self.setWindowTitle("Figure Composer")

    def _apply_recipe_to_controls(self) -> None:
        self._updating_controls = True
        try:
            setup = self._document.recipe.setup
            self.layout_panel.set_setup(
                setup, reserved_names=self._document.source_names()
            )
            self._refresh_source_list()
            self._rebuild_axes_grid()
            self._refresh_operation_list()
            if self.operation_panel.current_index() < 0:
                self.operation_panel.select_row(0)
            self._refresh_operation_editor()
        finally:
            self._updating_controls = False

    @staticmethod
    def _set_combo_value(combo: QtWidgets.QComboBox, value: str) -> None:
        combo.setCurrentIndex(max(combo.findText(value), 0))

    def _set_source_refresh_callbacks(
        self,
        *,
        can_refresh_source: Callable[[str], bool] | None = None,
        refresh_source: Callable[[str], bool] | None = None,
        source_label: Callable[[str], str | None] | None = None,
    ) -> None:
        self._source_refresh_available_callback = can_refresh_source
        self._source_refresh_callback = refresh_source
        self._source_refresh_label_callback = source_label
        self.refresh_source_controls()

    def _set_source_reveal_callbacks(
        self,
        *,
        can_reveal_source: Callable[[str], bool] | None = None,
        reveal_sources: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        self._source_reveal_available_callback = can_reveal_source
        self._source_reveal_callback = reveal_sources
        self.refresh_source_controls()

    def _set_source_add_callbacks(
        self,
        *,
        can_add_sources: Callable[[], bool] | None = None,
        add_sources: Callable[[], bool] | None = None,
        can_drop_sources: Callable[[QtCore.QMimeData], bool] | None = None,
        drop_sources: Callable[[QtCore.QMimeData], bool] | None = None,
    ) -> None:
        self._source_add_available_callback = can_add_sources
        self._source_add_callback = add_sources
        self._source_drop_available_callback = can_drop_sources
        self._source_drop_callback = drop_sources
        if self._figure_window is not None and erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            self._figure_window.set_source_drop_callbacks(
                can_drop=self._source_drop_available,
                drop=self._add_sources_from_mime,
            )
        self._refresh_source_controls()

    def _source_add_available(self) -> bool:
        if self._source_add_callback is None:
            return False
        if self._source_add_available_callback is None:
            return True
        with contextlib.suppress(LookupError, RuntimeError, ValueError):
            return bool(self._source_add_available_callback())
        return False

    @QtCore.Slot()
    def _request_add_sources_from_button(self) -> None:
        callback = self._source_add_callback
        if callback is None or not self._source_add_available():
            self._refresh_source_controls()
            return
        callback()
        self._refresh_source_controls()

    def _source_drop_available(self, mime: QtCore.QMimeData | None) -> bool:
        if mime is None or self._source_drop_available_callback is None:
            return False
        with contextlib.suppress(LookupError, RuntimeError, ValueError):
            return bool(self._source_drop_available_callback(mime))
        return False

    def _add_sources_from_mime(self, mime: QtCore.QMimeData | None) -> bool:
        if mime is None or self._source_drop_callback is None:
            return False
        with contextlib.suppress(LookupError, RuntimeError, ValueError):
            return bool(self._source_drop_callback(mime))
        return False

    def _refresh_source_list(self) -> None:
        """Synchronize the complete source panel from document state."""
        selected_names = set(self.source_panel.selected_names())
        current_name = self.source_panel.current_name()
        source_by_name = {
            source.name: source for source in self._document.recipe.sources
        }
        used_sources = self._document.sources_used_by_recipe()
        rows: list[FigureSourceRow] = []
        shown: set[str] = set()
        for source in self._document.recipe.sources:
            name = source.name
            shown.add(name)
            rows.append(
                FigureSourceRow(
                    name=name,
                    display=_source_display_label(source, name),
                    tooltip=self._source_tooltip(name),
                    data=self._document.source_data.get(name),
                    missing=name not in self._document.source_data,
                    used=name in used_sources,
                )
            )
        for name, data in self._document.source_data.items():
            if name in shown:
                continue
            rows.append(
                FigureSourceRow(
                    name=name,
                    display=_source_display_label(source_by_name.get(name), name),
                    tooltip=self._source_tooltip(name),
                    data=data,
                    missing=name not in source_by_name,
                    used=name in used_sources,
                )
            )
        available_names = set(self._document.source_names())
        selected_names.intersection_update(available_names)
        if current_name not in available_names:
            current_name = (
                self._source_inspector_target
                if self._source_inspector_target in available_names
                else self._default_source_inspector_target()
            )
        if not selected_names and current_name is not None:
            selected_names.add(current_name)
        self.source_panel.set_sources(
            rows, selected_names=selected_names, current_name=current_name
        )
        self._source_inspector_target = current_name
        self._refresh_source_controls()
        self._refresh_source_detail_panel()
        self._refresh_source_selection_editor()
        self._refresh_axis_name_projections()

    def _refresh_axis_name_projections(self) -> None:
        """Synchronize every view whose axes names depend on source names."""
        source_names = self._document.source_names()
        source_name_set = frozenset(source_names)
        if source_name_set == self._projected_axis_name_sources:
            return
        self.layout_panel.set_reserved_names(source_names)
        self._sync_axes_selector()
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._projected_axis_name_sources = source_name_set

    @QtCore.Slot(object, object)
    def _source_panel_selection_changed(
        self, _selected_names: object, current_name: object
    ) -> None:
        self._source_inspector_target = (
            current_name if isinstance(current_name, str) else None
        )
        self._set_source_validation_text(None)
        self._refresh_source_controls()
        self._refresh_source_detail_panel()
        self._refresh_source_selection_editor()

    def _source_refresh_available(self, name: str) -> bool:
        if (
            self._source_refresh_available_callback is None
            or self._source_refresh_callback is None
        ):
            return False
        with contextlib.suppress(LookupError, RuntimeError, ValueError):
            return bool(self._source_refresh_available_callback(name))
        return False

    def _source_reveal_available(self, name: str) -> bool:
        if (
            self._source_reveal_available_callback is None
            or self._source_reveal_callback is None
        ):
            return False
        with contextlib.suppress(LookupError, RuntimeError, ValueError):
            return bool(self._source_reveal_available_callback(name))
        return False

    def _source_refresh_label(self, name: str) -> str | None:
        if self._source_refresh_label_callback is None:
            return None
        with contextlib.suppress(LookupError, RuntimeError, ValueError):
            label = self._source_refresh_label_callback(name)
            return label or None
        return None

    def _refreshable_source_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._document.source_names()
            if self._source_refresh_available(name)
        )

    def _refresh_source_controls(self) -> None:
        selected_names = self.source_panel.selected_names()
        add_enabled = self._source_add_available()
        selected_refreshable = tuple(
            name for name in selected_names if self._source_refresh_available(name)
        )
        selected_revealable = tuple(
            name for name in selected_names if self._source_reveal_available(name)
        )
        self.source_panel.set_action_availability(
            add=add_enabled,
            refresh=bool(selected_refreshable),
            reveal=bool(selected_revealable),
            remove=any(
                self._document.source_is_removable(name) for name in selected_names
            ),
            refresh_all=bool(self._refreshable_source_names()),
            duplicate=any(
                name in self._document.source_by_name() for name in selected_names
            ),
            move_up=self._document.can_move_sources(selected_names, -1),
            move_down=self._document.can_move_sources(selected_names, 1),
        )

    def refresh_source_controls(self) -> None:
        self.source_panel.update_tooltips(
            {name: self._source_tooltip(name) for name in self._document.source_names()}
        )
        self._refresh_source_controls()
        self._refresh_source_detail_panel()

    def _set_source_panel_status(self, text: str | None) -> None:
        self.source_panel.set_status(text)

    def _set_source_validation_text(self, text: str | None) -> None:
        self.source_panel.set_validation(text)

    @QtCore.Slot(object)
    def _reveal_source_names(self, requested_names: object) -> None:
        if not isinstance(requested_names, tuple) or not all(
            isinstance(name, str) for name in requested_names
        ):
            return
        callback = self._source_reveal_callback
        source_names = tuple(
            name for name in requested_names if self._source_reveal_available(name)
        )
        if callback is None or not source_names:
            self._refresh_source_controls()
            return
        callback(source_names)
        self._refresh_source_controls()

    @QtCore.Slot()
    def _refresh_all_sources_from_button(self) -> None:
        self._refresh_source_names(self._document.source_names())

    def _refresh_source_names(self, source_names: Sequence[str]) -> None:
        callback = self._source_refresh_callback
        requested = tuple(dict.fromkeys(source_names))
        refreshable = tuple(
            name for name in requested if self._source_refresh_available(name)
        )
        unavailable = tuple(name for name in requested if name not in refreshable)
        if callback is None or not refreshable:
            if unavailable:
                self._set_source_panel_status(
                    "Unavailable: " + ", ".join(unavailable) + "."
                )
            self._refresh_source_controls()
            return

        refreshed: list[str] = []
        failed: list[str] = []
        failure_messages: list[str] = []
        self._set_source_panel_status(None)
        for name in refreshable:
            self._set_source_panel_status(None)
            try:
                refreshed_source = callback(name)
            except Exception as exc:
                logger.exception(
                    "Failed to refresh Figure Composer source %r",
                    name,
                    extra={"suppress_ui_alert": True},
                )
                failed.append(name)
                message = str(exc) or exc.__class__.__name__
                failure_messages.append(f"Could not refresh {name}: {message}.")
                continue
            if refreshed_source:
                refreshed.append(name)
                continue
            failed.append(name)
            if message := self.source_panel.source_status_label.text().strip():
                failure_messages.append(message)

        if len(failed) == 1 and not refreshed and not unavailable and failure_messages:
            status = failure_messages[0]
        else:
            parts: list[str] = []
            if refreshed:
                suffix = "source" if len(refreshed) == 1 else "sources"
                parts.append(f"Refreshed {len(refreshed)} {suffix}.")
            if failed:
                if failure_messages:
                    parts.extend(dict.fromkeys(failure_messages))
                else:
                    parts.append("Could not refresh: " + ", ".join(failed) + ".")
            if unavailable:
                parts.append("Unavailable: " + ", ".join(unavailable) + ".")
            status = " ".join(parts)
        self._set_source_panel_status(status or None)
        self._refresh_source_controls()
        self._refresh_source_detail_panel()

    @QtCore.Slot(object)
    def _remove_source_names(self, requested_names: object) -> None:
        if not isinstance(requested_names, tuple) or not all(
            isinstance(name, str) for name in requested_names
        ):
            return
        current_name = self.source_panel.current_name()
        removed = self._document.remove_sources(requested_names)
        if not removed:
            self._refresh_source_controls()
            return
        remaining_order = self._document.source_names()
        remaining_names = set(remaining_order)
        selected_names = {name for name in requested_names if name in remaining_names}
        if current_name not in remaining_names:
            current_name = next(
                (name for name in remaining_order if name in selected_names), None
            )
        if current_name is None:
            current_name = next(iter(remaining_order), None)
        if not selected_names and current_name is not None:
            selected_names.add(current_name)
        self._finish_source_structure_change(selected_names, current_name)

    @QtCore.Slot(str, str)
    def _rename_source_requested(self, original: str, alias: str) -> None:
        self.operation_editor.flush_pending_commits()
        if error := self._document.source_alias_error(alias, current=original):
            self._set_source_validation_text(error)
            return
        try:
            changed = self._document.rename_source(original, alias)
        except ValueError as exc:
            self._set_source_validation_text(str(exc))
            self.source_panel.reset_alias(original)
            return
        self._set_source_validation_text(None)
        if not changed:
            self.source_panel.reset_alias(original)
            self._refresh_source_controls()
            return
        self._finish_source_structure_change({alias}, alias)

    def _finish_source_structure_change(
        self, selected_names: set[str], current_name: str | None
    ) -> None:
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._refresh_source_list()
        self.source_panel.set_selected_names(selected_names, current_name=current_name)
        self._refresh_source_controls()
        self._source_inspector_target = current_name
        self._refresh_source_detail_panel()
        self._refresh_source_selection_editor()
        self._update_source_section()
        self._maybe_redraw_plot()
        self._set_source_panel_status(None)
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()

    @QtCore.Slot(object)
    def _duplicate_source_names(self, requested_names: object) -> None:
        if not isinstance(requested_names, tuple) or not all(
            isinstance(name, str) for name in requested_names
        ):
            return
        duplicated_names = self._document.duplicate_sources(requested_names)
        if not duplicated_names:
            return
        self._finish_source_structure_change(set(duplicated_names), duplicated_names[0])

    @QtCore.Slot(object, int)
    def _move_source_names(self, requested_names: object, offset: int) -> None:
        if not isinstance(requested_names, tuple) or not all(
            isinstance(name, str) for name in requested_names
        ):
            return
        selected_names = set(requested_names)
        if not self._document.move_sources(requested_names, offset):
            self._refresh_source_controls()
            return
        current = self.source_panel.current_name()
        current_name = (
            current
            if current in selected_names
            else next(
                source.name
                for source in self._document.recipe.sources
                if source.name in selected_names
            )
        )
        self._finish_source_structure_change(selected_names, current_name)

    @QtCore.Slot(object, object, object)
    def _source_list_reordered(
        self,
        source_names: object,
        selected_names: object,
        current_name: object,
    ) -> None:
        if not isinstance(source_names, (tuple, list)):
            self._refresh_source_list()
            return
        ordered_names = tuple(name for name in source_names if isinstance(name, str))
        source_by_name = {
            source.name: source for source in self._document.recipe.sources
        }
        if (
            len(ordered_names) != len(source_names)
            or len(ordered_names) != len(source_by_name)
            or set(ordered_names) != set(source_by_name)
        ):
            self._refresh_source_list()
            return
        selected_name_set: set[str] = set()
        if isinstance(selected_names, (set, frozenset, tuple, list)):
            selected_name_set = {
                name
                for name in selected_names
                if isinstance(name, str) and name in source_by_name
            }
        current_source_name = (
            current_name
            if isinstance(current_name, str) and current_name in source_by_name
            else None
        )
        if not selected_name_set and current_source_name is not None:
            selected_name_set = {current_source_name}
        if current_source_name is None:
            current_source_name = next(
                (name for name in ordered_names if name in selected_name_set),
                next(iter(ordered_names), None),
            )

        if not self._document.reorder_sources(ordered_names):
            return
        self._finish_source_structure_change(selected_name_set, current_source_name)

    def _sync_source_list_used_state(self) -> None:
        self.source_panel.set_used_sources(self._document.sources_used_by_recipe())

    def _default_source_inspector_target(self) -> str | None:
        source_name = self.source_panel.current_name()
        if source_name in self._document.source_names():
            return source_name
        if self._document.recipe.primary_source in self._document.source_data:
            return self._document.recipe.primary_source
        if self._document.recipe.sources:
            return self._document.recipe.sources[0].name
        return next(iter(self._document.source_data), None)

    def _refresh_source_detail_panel(self) -> None:
        selected_names = self.source_panel.selected_names()
        if len(selected_names) != 1:
            self.source_panel.set_detail(None, selection_count=len(selected_names))
            return
        target = selected_names[0]
        self._source_inspector_target = target
        self.source_panel.set_detail(
            FigureSourceDetail(
                name=target,
                data=self._document.source_data.get(target),
                context_lines=self._source_detail_context_lines(target),
                usage_count=self._document.source_usage_count(target),
                origin=self._source_refresh_label(target) or "",
                alias_enabled=target in self._document.source_by_name(),
            ),
            selection_count=1,
        )

    def _refresh_source_selection_editor(self) -> None:
        selected_names = self.source_panel.selected_names()
        if not selected_names:
            self.source_panel.set_selection_editor(())
            return

        source_by_name = self._document.source_by_name()
        available_names = tuple(
            name
            for name in selected_names
            if name in source_by_name and name in self._document.source_data
        )
        if not available_names:
            self.source_panel.set_selection_editor(
                (),
                message="No selected source data is available.",
                message_tooltip=(
                    "Source selections can be edited after source data is available."
                ),
            )
            return

        dimensions = self._common_source_selection_dims(available_names)
        if not dimensions:
            self.source_panel.set_selection_editor(
                (),
                message="No common dimensions.",
                message_tooltip=(
                    "The selected source data has no common editable dimensions."
                ),
            )
            return

        rows: list[FigureSourceSelectionRow] = []
        for dim_name in dimensions:
            dimension_context = self._source_selection_dimension_tooltip(
                dim_name, available_names
            )
            selections = tuple(
                _source_selection(source_by_name[name]) for name in available_names
            )
            modes = tuple(
                selection_dim_mode(selection, dim_name) for selection in selections
            )
            mode_mixed = len(set(modes)) > 1
            value_texts = tuple(
                selection_dim_value_text(selection, dim_name)
                for selection in selections
            )
            value_mixed = len(set(value_texts)) > 1
            width_texts = tuple(
                selection_dim_width_text(selection, dim_name)
                for selection in selections
            )
            width_mixed = len(set(width_texts)) > 1
            current_mode = None if mode_mixed else modes[0]
            rows.append(
                FigureSourceSelectionRow(
                    dimension=dim_name,
                    tooltip=dimension_context,
                    mode=current_mode,
                    mode_mixed=mode_mixed,
                    value_text=value_texts[0],
                    value_mixed=value_mixed,
                    width_text=width_texts[0],
                    width_mixed=width_mixed,
                )
            )
        self.source_panel.set_selection_editor(rows)

    def _source_selection_dimension_tooltip(
        self, dim: str, source_names: Sequence[str]
    ) -> str:
        sizes: set[int] = set()
        dtypes: set[str] = set()
        endpoints: set[tuple[str, str]] = set()
        for source_name in source_names:
            data = self._document.source_selection_input_data(source_name)
            if data is None:
                continue
            public = _public_source_data(data)
            if dim not in public.dims:
                continue
            sizes.add(public.sizes[dim])
            coord = public.coords.get(dim)
            if coord is None:
                continue
            dtypes.add(str(coord.dtype))
            if coord.ndim != 1 or coord.size == 0 or coord.dims != (dim,):
                continue
            with contextlib.suppress(Exception):
                first = erlab.utils.formatting.format_value(coord.isel({dim: 0}).item())
                last = erlab.utils.formatting.format_value(coord.isel({dim: -1}).item())
                endpoints.add((first, last))

        lines = [f"Dimension {dim!r}."]
        if sizes:
            sizes_text = ", ".join(str(size) for size in sorted(sizes))
            lines.append(f"Size: {sizes_text}.")
        if dtypes:
            lines.append(f"Coordinate dtype: {', '.join(sorted(dtypes))}.")
        if len(endpoints) == 1:
            first, last = next(iter(endpoints))
            lines.append(f"Coordinate range: {first} to {last}.")
        return " ".join(lines)

    def _common_source_selection_dims(
        self, source_names: Sequence[str]
    ) -> tuple[str, ...]:
        common: list[str] | None = None
        for source_name in source_names:
            data = self._document.source_selection_input_data(source_name)
            if data is None:
                continue
            dims = [str(dim) for dim in _public_source_data(data).dims]
            if common is None:
                common = dims
            else:
                dim_set = set(dims)
                common = [dim for dim in common if dim in dim_set]
        return tuple(common or ())

    def _update_selected_source_dimension(
        self,
        requested_names: object,
        dim: str,
        mode: str,
        value_text: str,
        width_text: str,
    ) -> None:
        if not isinstance(requested_names, (tuple, list, set, frozenset)):
            return
        source_names = tuple(name for name in requested_names if isinstance(name, str))
        try:
            value = (
                selection_value_from_text(value_text)
                if mode in {"isel", "qsel"}
                else None
            )
            width = selection_width_from_text(width_text) if mode == "qsel" else None
        except FigureComposerInputError as exc:
            self._set_source_validation_text(str(exc))
            return

        result = self._document.update_source_selection_dimension(
            source_names,
            dim,
            mode,
            value,
            width,
        )
        status_text = (
            "Selection was not applied to: "
            + ", ".join(f"{name} ({detail})" for name, detail in result.skipped)
            if result.skipped
            else None
        )
        if not result:
            self._set_source_validation_text(status_text)
            self._refresh_source_selection_editor()
            return
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
        self._set_source_validation_text(status_text)
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()

    def _normalize_operation_source_selections(self) -> bool:
        """Normalize legacy selections as one recipe-and-data transaction."""
        operations: list[FigureOperationState] = []
        source_list = list(self._document.recipe.sources)
        source_by_name = {source.name: source for source in source_list}
        source_data = dict(self._document.source_data)
        selection_base_data = dict(self._document.source_selection_base_data)
        original_source_data = dict(source_data)
        original_selection_base_data = dict(selection_base_data)
        reserved = set(source_by_name)
        reserved.update(source_data)
        changed = False
        for operation in self._document.recipe.operations:
            if not operation.map_selections:
                operations.append(operation)
                continue
            updated = self._operation_with_legacy_source_selections(
                operation,
                source_list=source_list,
                source_by_name=source_by_name,
                reserved=reserved,
                source_data=source_data,
                selection_base_data=selection_base_data,
            )
            operations.append(updated)
            changed = changed or updated != operation
        data_changed = (
            source_data.keys() != original_source_data.keys()
            or any(
                source_data[name] is not data
                for name, data in original_source_data.items()
            )
            or selection_base_data.keys() != original_selection_base_data.keys()
            or any(
                selection_base_data[name] is not data
                for name, data in original_selection_base_data.items()
            )
        )
        if not changed and not data_changed:
            return False
        self._document.replace_recipe(
            self._document.recipe.model_copy(
                update={"sources": tuple(source_list), "operations": tuple(operations)}
            )
        )
        self._document.replace_source_payloads(source_data, selection_base_data)
        return data_changed

    def _operation_with_legacy_source_selections(
        self,
        operation: FigureOperationState,
        *,
        source_list: list[FigureSourceState],
        source_by_name: dict[str, FigureSourceState],
        reserved: set[str],
        source_data: dict[str, xr.DataArray],
        selection_base_data: dict[str, xr.DataArray],
    ) -> FigureOperationState:
        if operation.kind == FigureOperationKind.PLOT_SLICES:
            updated = self._plot_slices_operation_with_shared_legacy_selection(
                operation
            )
            if updated is not None:
                return updated
            return self._plot_slices_operation_with_legacy_source_aliases(
                operation,
                source_list=source_list,
                source_by_name=source_by_name,
                reserved=reserved,
                source_data=source_data,
                selection_base_data=selection_base_data,
            )

        if len(operation.map_selections) != 1:
            # A legacy Line/Profile step can contain one picked profile per cursor.
            # Keep that compatibility representation intact so rendering, styling,
            # offsets, and placement still see every profile in cursor order.
            if operation.kind == FigureOperationKind.LINE:
                return operation
            return self._operation_without_map_selections(operation, None)

        selection = operation.map_selections[0]
        source_name = selection.source
        if not selection_has_effect(selection):
            return self._operation_without_map_selections(
                operation,
                self._legacy_selection_fallback_source(operation, source_name),
            )
        alias = self._source_alias_for_legacy_selection(
            selection,
            source_list=source_list,
            source_by_name=source_by_name,
            reserved=reserved,
            source_data=source_data,
            selection_base_data=selection_base_data,
        )
        return self._operation_without_map_selections(operation, alias)

    def _plot_slices_operation_with_shared_legacy_selection(
        self, operation: FigureOperationState
    ) -> FigureOperationState | None:
        selection_sources = tuple(
            selection.source for selection in operation.map_selections
        )
        if operation.sources and selection_sources != operation.sources:
            return None
        legacy_sources = operation.sources or selection_sources

        def preserve_legacy_sources(
            updated: FigureOperationState,
        ) -> FigureOperationState:
            if not legacy_sources:
                return updated
            return updated.model_copy(update={"sources": legacy_sources})

        selection = shared_selection(operation.map_selections)
        if selection is None:
            return None
        if not selection_has_effect(selection):
            fallback = (
                self._legacy_selection_fallback_source(
                    operation, operation.map_selections[0].source
                )
                if operation.map_selections
                else None
            )
            return preserve_legacy_sources(
                self._operation_without_map_selections(operation, fallback)
            )
        if selection.isel or selection.mean_dims:
            return None

        dims = _operation_dim_names(self._document, operation)
        if not dims:
            return preserve_legacy_sources(
                self._plot_slices_operation_with_legacy_qsel(operation, selection.qsel)
            )
        if any(not _is_slice_kwarg_key(key, dims) for key in selection.qsel):
            return None
        updates = _selection_updates_from_kwargs(
            self._document,
            operation,
            {
                **_effective_slice_kwargs(self._document, operation),
                **selection.qsel,
            },
            _effective_extra_kwargs(self._document, operation),
        )
        updates["map_selections"] = ()
        return preserve_legacy_sources(operation.model_copy(update=updates))

    @staticmethod
    def _plot_slices_operation_with_legacy_qsel(
        operation: FigureOperationState, qsel: Mapping[str, typing.Any]
    ) -> FigureOperationState:
        slice_kwargs = {**operation.slice_kwargs, **qsel}
        updates: dict[str, typing.Any] = {"map_selections": ()}
        slice_dim = operation.slice_dim
        if slice_dim is not None:
            values = _selection_values(slice_kwargs.get(slice_dim))
            if values:
                updates["slice_values"] = values
                slice_kwargs.pop(slice_dim, None)
            width = _selection_width(slice_kwargs.get(f"{slice_dim}_width"))
            if width is not None:
                updates["slice_width"] = width
                slice_kwargs.pop(f"{slice_dim}_width", None)
        else:
            candidates = [
                (key, values)
                for key, value in slice_kwargs.items()
                if (not key.endswith("_width") and (values := _selection_values(value)))
            ]
            if len(candidates) == 1:
                slice_dim, values = candidates[0]
                updates["slice_dim"] = slice_dim
                updates["slice_values"] = values
                slice_kwargs.pop(slice_dim, None)
                width = _selection_width(slice_kwargs.get(f"{slice_dim}_width"))
                if width is not None:
                    updates["slice_width"] = width
                    slice_kwargs.pop(f"{slice_dim}_width", None)
        updates["slice_kwargs"] = slice_kwargs
        return operation.model_copy(update=updates)

    def _plot_slices_operation_with_legacy_source_aliases(
        self,
        operation: FigureOperationState,
        *,
        source_list: list[FigureSourceState],
        source_by_name: dict[str, FigureSourceState],
        reserved: set[str],
        source_data: dict[str, xr.DataArray],
        selection_base_data: dict[str, xr.DataArray],
    ) -> FigureOperationState:
        indexed_by_source: dict[
            str, collections.deque[tuple[int, FigureDataSelectionState]]
        ] = {}
        for index, selection in enumerate(operation.map_selections):
            indexed_by_source.setdefault(selection.source, collections.deque()).append(
                (index, selection)
            )
        source_counts = collections.Counter(operation.sources)

        def source_for_selection(selection: FigureDataSelectionState) -> str:
            if not selection_has_effect(selection):
                return selection.source
            return self._source_alias_for_legacy_selection(
                selection,
                source_list=source_list,
                source_by_name=source_by_name,
                reserved=reserved,
                source_data=source_data,
                selection_base_data=selection_base_data,
            )

        updated_sources: list[str] = []
        for source_name in operation.sources:
            selections = indexed_by_source.get(source_name)
            if not selections:
                updated_sources.append(source_name)
                continue
            if source_counts[source_name] == 1:
                while selections:
                    _index, selection = selections.popleft()
                    updated_sources.append(source_for_selection(selection))
            else:
                _index, selection = selections.popleft()
                updated_sources.append(source_for_selection(selection))

        remaining = sorted(
            (entry for entries in indexed_by_source.values() for entry in entries),
            key=lambda entry: entry[0],
        )
        updated_sources.extend(
            source_for_selection(selection) for _index, selection in remaining
        )
        return operation.model_copy(
            update={"map_selections": (), "sources": tuple(updated_sources)}
        )

    @staticmethod
    def _legacy_selection_fallback_source(
        operation: FigureOperationState,
        source_name: str,
    ) -> str | None:
        if operation.kind == FigureOperationKind.LINE:
            return source_name if operation.line_source is None else None
        if operation.kind in {
            FigureOperationKind.PLOT_ARRAY,
            FigureOperationKind.PLOT_SLICES,
        }:
            return source_name if not operation.sources else None
        return None

    def _source_alias_for_legacy_selection(
        self,
        selection: FigureDataSelectionState,
        *,
        source_list: list[FigureSourceState],
        source_by_name: dict[str, FigureSourceState],
        reserved: set[str],
        source_data: dict[str, xr.DataArray],
        selection_base_data: dict[str, xr.DataArray],
    ) -> str:
        source_name = selection.source
        for source in source_list:
            if source.selection_source != source_name:
                continue
            if (
                source.isel == selection.isel
                and source.qsel == selection.qsel
                and source.mean_dims == selection.mean_dims
            ):
                if source.name not in source_data:
                    base_data = source_data.get(source_name)
                    if base_data is not None:
                        selection_base_data[source.name] = base_data
                        try:
                            selected = FigureDocument.source_data_from_selection(
                                base_data, source
                            )
                        except (IndexError, KeyError, TypeError, ValueError):
                            pass
                        else:
                            source_data[source.name] = selected
                return source.name

        alias = self._selected_source_alias(source_name, reserved)
        reserved.add(alias)
        base_source = source_by_name.get(
            source_name, FigureSourceState(name=source_name)
        )
        selected_source = _source_with_selection(
            FigureDocument.source_with_name(base_source, alias).model_copy(
                update={"selection_source": source_name}
            ),
            selection.model_copy(update={"source": alias}),
        )
        source_by_name[alias] = selected_source
        source_list.append(selected_source)
        base_data = source_data.get(source_name)
        if base_data is None:
            return alias
        selection_base_data[alias] = base_data
        try:
            selected = _selected_data(source_data, selection)
        except (IndexError, KeyError, TypeError, ValueError):
            return alias
        if selected is not None:
            source_data[alias] = selected.copy(deep=False)
        return alias

    @staticmethod
    def _operation_without_map_selections(
        operation: FigureOperationState,
        source_name: str | None,
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {"map_selections": ()}
        if operation.kind == FigureOperationKind.LINE:
            if operation.map_selections:
                # The legacy map-selection rendering branch bypassed these input
                # extraction controls.  Activating stale values while converting a
                # single cursor to a source alias changes the saved figure.
                updates.update(
                    {
                        "line_selection": {},
                        "line_y": None,
                        "line_iter_dim": None,
                        "line_reduce": "disabled",
                        "line_reduce_coarsen": 2,
                        "line_reduce_thin": 2,
                    }
                )
            if source_name is not None:
                updates["line_source"] = source_name
            return operation.model_copy(update=updates)
        if operation.kind == FigureOperationKind.PLOT_ARRAY:
            if source_name is not None:
                updates["sources"] = (source_name,)
            elif operation.sources:
                updates["sources"] = operation.sources[:1]
            return operation.model_copy(update=updates)
        if operation.kind == FigureOperationKind.PLOT_SLICES:
            if source_name is not None:
                updates["sources"] = (source_name,)
            return operation.model_copy(update=updates)
        return operation.model_copy(update=updates)

    @staticmethod
    def _selected_source_alias(source_name: str, reserved: set[str]) -> str:
        stem = f"{source_name}_selected"
        alias = stem
        suffix = 2
        while alias in reserved:
            alias = f"{stem}_{suffix}"
            suffix += 1
        return alias

    def _rebuild_axes_grid(self) -> None:
        self.axes_selector.set_grid(
            self._document.recipe.setup.nrows,
            self._document.recipe.setup.ncols,
        )
        self._sync_axes_selector()

    def _refresh_gridspec_axes_selector(self) -> None:
        selected_ids: set[str] = set()
        current = self._current_operation()
        if current is not None and _registry.spec_for(current[1].kind).uses_axes(
            current[1]
        ):
            selected_ids = set(current[1].axes.axes_ids)
        blocker = QtCore.QSignalBlocker(self.gridspec_axes_selector)
        axes_ids = _gridspec_all_axes_ids(self._document.recipe.setup)
        self.gridspec_axes_selector.set_layout(
            self._document.recipe.setup.gridspec.root,
            {
                axes_id: _gridspec_axis_display_name(
                    self._document.recipe.setup,
                    axes_id,
                    reserved_names=self._document.source_names(),
                )
                for axes_id in axes_ids
            },
        )
        self.gridspec_axes_selector.set_selected_axes_ids(tuple(selected_ids))
        del blocker

    def _sync_axes_selector(self) -> None:
        current = self._current_operation()
        selected_axes = set()
        selected_axes_ids: tuple[str, ...] = ()
        expression = ""
        invalid_axes: tuple[tuple[int, int], ...] = ()
        invalid_axes_ids: tuple[str, ...] = ()
        if current is not None:
            _, operation = current
            spec = _registry.spec_for(operation.kind)
            if spec.uses_axes(operation):
                if self._document.recipe.setup.layout_mode == "gridspec":
                    selected_axes_ids = _gridspec_valid_axes_ids(
                        self._document.recipe.setup, operation.axes.axes_ids
                    )
                    invalid_axes_ids = _gridspec_invalid_axes_ids(
                        self._document.recipe.setup, operation.axes.axes_ids
                    )
                else:
                    selected_axes = set(
                        operation.axes.valid_axes(self._document.recipe.setup)
                    )
                    invalid_axes = operation.axes.invalid_axes(
                        self._document.recipe.setup
                    )
                expression = operation.axes.expression

        was_updating_controls = self._updating_controls
        self._updating_controls = True
        try:
            grid_mode = self._document.recipe.setup.layout_mode == "gridspec"
            self.axes_selector.setVisible(not grid_mode)
            self.axes_expression_edit.setVisible(not grid_mode)
            self.gridspec_axes_selector.setVisible(grid_mode)
            if grid_mode:
                self._refresh_gridspec_axes_selector()
            else:
                self.axes_selector.set_selected_axes(tuple(sorted(selected_axes)))
            if self.axes_expression_edit.text() != expression:
                blocker = QtCore.QSignalBlocker(self.axes_expression_edit)
                self.axes_expression_edit.setText(expression)
                del blocker
            self.keep_valid_axes_button.setEnabled(
                bool(invalid_axes_ids) if grid_mode else bool(invalid_axes)
            )
            if current is None:
                self.target_axes_status_label.setText("Select a step to choose axes.")
            elif not _registry.spec_for(current[1].kind).uses_axes(current[1]):
                self.target_axes_status_label.setText(
                    _registry.spec_for(current[1].kind).target_text(self, current[1])
                )
            elif expression:
                self.target_axes_status_label.setText(
                    "Using the advanced axes expression instead of grid selection."
                )
            elif invalid_axes:
                self.target_axes_status_label.setText(
                    "Target axes removed by the current layout: "
                    f"{_format_axes_tuple(invalid_axes)}"
                )
            elif invalid_axes_ids:
                self.target_axes_status_label.setText(
                    _removed_axes_status_text(
                        len(invalid_axes_ids), "current GridSpec layout"
                    )
                )
            else:
                if grid_mode:
                    axes_text = ", ".join(
                        _gridspec_axis_display_names(
                            self._document.recipe.setup,
                            selected_axes_ids,
                            reserved_names=self._document.source_names(),
                        )
                    )
                else:
                    axes_text = _format_axes_tuple(
                        tuple(sorted(selected_axes)),
                        nrows=self._document.recipe.setup.nrows,
                        ncols=self._document.recipe.setup.ncols,
                    )
                self.target_axes_status_label.setText(f"Targets: {axes_text}")
        finally:
            self._updating_controls = was_updating_controls

    def _refresh_operation_list(self) -> None:
        current_id = self.operation_panel.current_id()
        selected_ids = self._selected_operation_ids()
        if not selected_ids and current_id is not None:
            selected_ids = {current_id}
        self.operation_panel.set_rows(
            tuple(
                self._operation_row(operation)
                for operation in self._document.recipe.operations
            ),
            selected_ids=selected_ids,
            current_id=current_id,
        )
        self._sync_source_list_used_state()

    def _operation_row(self, operation: FigureOperationState) -> FigureOperationRow:
        issues = self._operation_issues(operation)
        tooltip = self._operation_tooltip(operation)
        target_text = _registry.spec_for(operation.kind).target_text(self, operation)
        target_description = (
            "No target" if target_text.casefold() == "none" else target_text
        )
        status_tooltip = "\n".join(
            f"{_OPERATION_STATUS_LABELS[code]}: {detail}" for code, detail in issues
        )
        return FigureOperationRow(
            operation.operation_id,
            self._operation_display_text(operation),
            operation.enabled,
            tooltip,
            self._operation_target_preview_descriptor(operation),
            target_description,
            self._operation_status_text(issues),
            tuple(code for code, _detail in issues),
            status_tooltip,
        )

    def _set_current_operation_row_silent(
        self, index: int, *, preserve_selection: bool = True
    ) -> None:
        self.operation_panel.set_current_row(
            index, preserve_selection=preserve_selection
        )

    def _selected_operation_ids(self) -> set[str]:
        return set(self.operation_panel.selected_ids())

    def _set_selected_operation_ids_silent(self, operation_ids: set[str]) -> None:
        self.operation_panel.set_selected_ids(operation_ids)

    def _operation_display_text(self, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).display_text(self, operation)

    def _operation_tooltip(self, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).tooltip(self, operation)

    def _operation_target_preview_descriptor(
        self, operation: FigureOperationState
    ) -> tuple[object, ...]:
        spec = _registry.spec_for(operation.kind)
        if not spec.uses_axes(operation):
            target_text = spec.target_text(self, operation)
            return (
                "text",
                "—" if target_text.casefold() == "none" else target_text,
            )
        setup = self._document.recipe.setup
        if setup.layout_mode == "gridspec":
            return _gridspec_target_preview_descriptor(
                setup.gridspec.root,
                _gridspec_valid_axes_ids(setup, operation.axes.axes_ids),
            )
        unresolved = False
        selected_axes = operation.axes.valid_axes(setup)
        if operation.axes.expression:
            try:
                tokens = np.arange(setup.nrows * setup.ncols).reshape(
                    setup.nrows, setup.ncols
                )
                resolved = np.asarray(
                    _axes_expression_value(operation.axes.expression, tokens)
                )
                flat_indices = tuple(
                    dict.fromkeys(int(value) for value in resolved.flat)
                )
            except (TypeError, ValueError):
                selected_axes = ()
                unresolved = True
            else:
                if flat_indices and all(
                    0 <= value < tokens.size for value in flat_indices
                ):
                    selected_axes = tuple(
                        divmod(value, setup.ncols) for value in flat_indices
                    )
                else:
                    selected_axes = ()
                    unresolved = True
        return _subplot_target_preview_descriptor(
            setup.nrows,
            setup.ncols,
            selected_axes,
            unresolved=unresolved,
        )

    def _operation_issues(
        self, operation: FigureOperationState
    ) -> tuple[tuple[str, str], ...]:
        issues: list[tuple[str, str]] = []
        spec = _registry.spec_for(operation.kind)
        if spec.has_invalid_target(self._document, operation):
            issues.append(("invalid_target", spec.target_text(self, operation)))
        missing_sources = tuple(
            source
            for source in declared_operation_source_names(operation)
            if source not in self._document.source_data
        )
        if missing_sources:
            issues.append(
                (
                    "missing_source",
                    ", ".join(self._source_display_names(missing_sources)),
                )
            )
        input_error = self.operation_editor.input_error_text(operation)
        if input_error is not None:
            issues.append(("invalid_input", input_error))
        render_error = self._operation_render_errors.get(operation.operation_id)
        if render_error is not None:
            issues.append(("render_error", render_error))
        return tuple(issues)

    @staticmethod
    def _operation_status_text(issues: Sequence[tuple[str, str]]) -> str:
        if not issues:
            return ""
        if len(issues) == 1:
            return _OPERATION_STATUS_LABELS[issues[0][0]]
        return f"{len(issues)} issues"

    def _set_operation_render_errors(self, errors: Mapping[str, str]) -> None:
        render_errors = dict(errors)
        if render_errors == self._operation_render_errors:
            return
        self._operation_render_errors = render_errors
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        current = self._current_operation()
        self._update_source_status(current[1] if current is not None else None)

    @QtCore.Slot()
    def _operation_editor_validation_changed(self) -> None:
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        current = self._current_operation()
        self._update_source_status(current[1] if current is not None else None)

    def _axes_target_text(self, selection: FigureAxesSelectionState) -> str:
        if selection.expression:
            return selection.expression
        if self._document.recipe.setup.layout_mode == "gridspec":
            invalid_ids = _gridspec_invalid_axes_ids(
                self._document.recipe.setup, selection.axes_ids
            )
            if invalid_ids:
                return _removed_axes_summary_text(len(invalid_ids))
            valid_ids = _gridspec_valid_axes_ids(
                self._document.recipe.setup, selection.axes_ids
            )
            if not valid_ids:
                return "none"
            return ", ".join(
                _gridspec_axis_display_names(
                    self._document.recipe.setup,
                    valid_ids,
                    reserved_names=self._document.source_names(),
                )
            )
        invalid_axes = selection.invalid_axes(self._document.recipe.setup)
        if invalid_axes:
            return f"removed axes {_format_axes_tuple(invalid_axes)}"
        valid_axes = selection.valid_axes(self._document.recipe.setup)
        if not valid_axes:
            return "none"
        return _format_axes_tuple(
            valid_axes,
            nrows=self._document.recipe.setup.nrows,
            ncols=self._document.recipe.setup.ncols,
        )

    def _operation_has_invalid_axes(self, operation: FigureOperationState) -> bool:
        return _registry.spec_for(operation.kind).has_invalid_target(
            self._document, operation
        )

    def _invalid_operation_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, operation in enumerate(self._document.recipe.operations)
            if operation.enabled
            and (
                self._operation_has_invalid_axes(operation)
                or self.operation_editor.has_input_error(operation)
            )
        )

    def _invalid_operation_target_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, operation in enumerate(self._document.recipe.operations)
            if operation.enabled and self._operation_has_invalid_axes(operation)
        )

    def _warn_invalid_operation_targets(self) -> bool:
        indices = self._invalid_operation_target_indices()
        if not indices:
            return False
        self.editor_tabs.setCurrentWidget(self.operation_panel)
        self.operation_panel.select_row(indices[0])
        self.operation_editor.select_section("axes")
        QtWidgets.QMessageBox.warning(
            self,
            "Retarget Figure Steps",
            "Some enabled steps target axes that no longer exist in the current "
            "figure layout. Use All Axes or Drop Removed Axes before copying, "
            "exporting, or rendering the full recipe.",
        )
        return True

    def _current_operation(self) -> tuple[int, FigureOperationState] | None:
        row = self._current_operation_index()
        if row < 0 or row >= len(self._document.recipe.operations):
            return None
        return row, self._document.recipe.operations[row]

    def _current_operation_id(self) -> str | None:
        current = self._current_operation()
        return None if current is None else current[1].operation_id

    def _current_operation_index(self) -> int:
        return self.operation_panel.current_index()

    def _selected_operation_indices(self) -> tuple[int, ...]:
        selected_ids = self._selected_operation_ids()
        if not selected_ids:
            current = self._current_operation()
            return () if current is None else (current[0],)
        return tuple(
            index
            for index, operation in enumerate(self._document.recipe.operations)
            if operation.operation_id in selected_ids
        )

    def _operation_duplicate_possible(self) -> bool:
        return bool(self._selected_operation_indices())

    def _operation_move_possible(self, offset: int) -> bool:
        indices = self._selected_operation_indices()
        return self._document.can_move_operations(
            (self._document.recipe.operations[index].operation_id for index in indices),
            offset,
        )

    @staticmethod
    def _operation_editor_schema_key(
        operation: FigureOperationState,
    ) -> tuple[object, ...]:
        if operation.kind.value == "method":
            return (operation.kind, operation.method_family, operation.method_name)
        return (operation.kind,)

    def _selected_operations_are_compatible(self) -> bool:
        indices = self._selected_operation_indices()
        if len(indices) <= 1:
            return True
        keys = {
            self._operation_editor_schema_key(self._document.recipe.operations[index])
            for index in indices
        }
        return len(keys) == 1

    def _editable_operation_indices(self) -> tuple[int, ...]:
        indices = self._selected_operation_indices()
        if len(indices) <= 1:
            return indices
        return indices if self._selected_operations_are_compatible() else ()

    def _editable_operations(self) -> tuple[tuple[int, FigureOperationState], ...]:
        return tuple(
            (index, self._document.recipe.operations[index])
            for index in self._editable_operation_indices()
        )

    def _editable_operation_ids(self) -> tuple[str, ...]:
        return tuple(
            operation.operation_id for _index, operation in self._editable_operations()
        )

    @QtCore.Slot(object)
    def _apply_operation_edit_request(self, request_object: object) -> None:
        if self._closing:
            return
        if isinstance(request_object, OperationEditRequest):
            self._update_operations_by_ids(
                request_object.operation_ids,
                request_object.updater,
                render=request_object.render,
                defer_render=request_object.defer_render,
                rebuild_editor=request_object.rebuild_editor,
                defer_editor_rebuild=request_object.defer_editor_rebuild,
                sync_axes=request_object.sync_axes,
            )
        elif isinstance(request_object, OperationRecipeEditRequest):
            self._apply_operation_recipe_edit_request(request_object)

    def _apply_operation_recipe_edit_request(
        self, request: OperationRecipeEditRequest
    ) -> bool:
        previous_operations = self._document.recipe.operations
        updated_operations = tuple(
            request.updater(previous_operations, request.operation_ids)
        )
        if updated_operations == previous_operations:
            return False

        previous_by_id = {
            operation.operation_id: operation for operation in previous_operations
        }
        updated_by_id = {
            operation.operation_id: operation for operation in updated_operations
        }
        operation_ids = previous_by_id.keys() | updated_by_id.keys()

        def operation_affects_preview(operation_id: str) -> bool:
            previous = previous_by_id.get(operation_id)
            updated = updated_by_id.get(operation_id)
            if previous is None:
                return updated is not None and updated.enabled
            if updated is None:
                return previous.enabled
            return previous != updated and self._operation_change_affects_preview(
                previous, updated
            )

        preview_affected = any(map(operation_affects_preview, operation_ids))

        self._document.replace_operations(updated_operations)
        self._refresh_operation_list()
        if request.sync_axes:
            self._sync_axes_selector()
        self._update_step_action_buttons()
        self._refresh_step_section_button_texts()
        current = self._current_operation()
        self._update_source_status(current[1] if current is not None else None)
        if request.rebuild_editor:
            if request.defer_editor_rebuild:
                self._queue_operation_editor_update()
            else:
                self._update_operation_editor_safely()
        self._notify_operation_changed(
            preview_affected=request.render and preview_affected,
            defer_render=request.defer_render,
        )
        self._write_state()
        return True

    def _update_operations_by_ids(
        self,
        operation_ids: Iterable[str],
        updater: Callable[[int, FigureOperationState], FigureOperationState],
        *,
        render: bool = True,
        defer_render: bool = False,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> bool:
        operation_id_set = set(operation_ids)
        if not operation_id_set:
            return False
        current = self._current_operation()
        preview_affected = False
        operation_changes: list[
            tuple[int, FigureOperationState, FigureOperationState]
        ] = []

        def tracked_update(
            index: int, operation: FigureOperationState
        ) -> FigureOperationState:
            nonlocal preview_affected
            updated = updater(index, operation)
            if updated != operation:
                operation_changes.append((index, operation, updated))
                if self._operation_change_affects_preview(operation, updated):
                    preview_affected = True
            return updated

        if not self._document.update_operations_by_ids(
            operation_id_set, tracked_update
        ):
            return False
        self._refresh_operation_list()
        if current is not None:
            self._set_current_operation_row_silent(current[0])
        current_operation = self._current_operation()
        if sync_axes:
            self._sync_axes_selector()
        self._update_step_action_buttons()
        self._refresh_step_section_button_texts()
        self._update_source_status(current_operation[1] if current_operation else None)
        if rebuild_editor:
            if defer_editor_rebuild:
                self._queue_operation_editor_update()
            else:
                self._update_operation_editor_safely()
        if (
            render
            and preview_affected
            and not defer_render
            and len(operation_changes) == 1
            and self._try_update_live_colormap(*operation_changes[0])
        ):
            self.sigInfoChanged.emit()
        else:
            self._notify_operation_changed(
                preview_affected=render and preview_affected,
                defer_render=defer_render,
            )
        self._write_state()
        return True

    def _replace_operation(
        self,
        index: int,
        operation: FigureOperationState,
        *,
        render: bool = True,
        defer_render: bool = False,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> None:
        previous_operation = self._document.recipe.operations[index]
        self._document.replace_operation(index, operation)
        self.operation_editor.clear_input_errors(
            (previous_operation.operation_id, operation.operation_id)
        )
        self._refresh_operation_list()
        self._set_current_operation_row_silent(index)
        if sync_axes:
            self._sync_axes_selector()
        self._update_step_action_buttons()
        self._refresh_step_section_button_texts()
        self._update_source_status(operation)
        if rebuild_editor:
            if defer_editor_rebuild:
                self._queue_operation_editor_update()
            else:
                self._update_operation_editor_safely()
        self._notify_operation_changed(
            preview_affected=render
            and self._operation_change_affects_preview(previous_operation, operation),
            defer_render=defer_render,
        )
        self._write_state()

    @staticmethod
    def _operation_change_affects_preview(
        previous: FigureOperationState,
        updated: FigureOperationState,
    ) -> bool:
        return previous.enabled or updated.enabled

    def _notify_operation_changed(
        self, *, preview_affected: bool, defer_render: bool = False
    ) -> None:
        if preview_affected and self._notify_operation_preview_changed(
            defer=defer_render
        ):
            return
        self.sigInfoChanged.emit()

    def _notify_operation_preview_changed(self, *, defer: bool = False) -> bool:
        if not self._auto_redraw_enabled():
            self._auto_redraw_dirty = True
            self._cancel_preview_render_update()
            self._mark_preview_pixmap_stale()
            self.sigInfoChanged.emit()
            return True
        active_editor_widget = self.operation_editor.active_signal_widget
        if defer or active_editor_widget is not None:
            delay_ms = (
                _EDITOR_CONTROL_RENDER_UPDATE_DELAY_MS
                if active_editor_widget is not None
                else _PREVIEW_RENDER_UPDATE_DELAY_MS
            )
            self._queue_preview_render_update(delay_ms=delay_ms)
            return False
        self._cancel_preview_render_update()
        self._redraw_plot()
        self.sigInfoChanged.emit()
        return True

    def _queue_preview_render_update(
        self, *, delay_ms: int = _PREVIEW_RENDER_UPDATE_DELAY_MS
    ) -> None:
        if self._closing:
            return
        if not self._auto_redraw_enabled():
            self._auto_redraw_dirty = True
            self._cancel_preview_render_update()
            self._mark_preview_pixmap_stale()
            return
        self._preview_render_update_generation += 1
        generation = self._preview_render_update_generation
        self._preview_render_update_pending = True
        erlab.interactive.utils.single_shot(
            self,
            delay_ms,
            functools.partial(self._run_queued_preview_render_update, generation),
        )

    def _cancel_preview_render_update(self) -> None:
        if not self._preview_render_update_pending:
            return
        self._preview_render_update_generation += 1
        self._preview_render_update_pending = False

    def _run_queued_preview_render_update(self, generation: int) -> None:
        if (
            generation != self._preview_render_update_generation
            or self._closing
            or not erlab.interactive.utils.qt_is_valid(self)
        ):
            return
        self._preview_render_update_pending = False
        if not self._auto_redraw_enabled():
            self._auto_redraw_dirty = True
            self._mark_preview_pixmap_stale()
            return
        if self._rendering:
            self._queue_preview_render_update()
            return
        self._redraw_plot()
        self.sigInfoChanged.emit()

    def _update_step_action_buttons(self) -> None:
        indices = self._selected_operation_indices()
        self.operation_panel.set_action_availability(
            selection=bool(indices),
            paste=self._clipboard_step_payload() is not None,
            duplicate=bool(indices),
            move_up=self._operation_move_possible(-1),
            move_down=self._operation_move_possible(1),
        )

    @QtCore.Slot(object)
    def _layout_setup_requested(self, setup_object: object) -> None:
        setup = typing.cast("FigureSubplotsState", setup_object)
        self._commit_layout_setup(setup)

    def _commit_layout_setup(self, setup: FigureSubplotsState) -> bool:
        if self._updating_controls or self._closing:
            return False
        previous_setup = self._document.recipe.setup
        try:
            changed = self._document.replace_setup(setup)
        except ValueError as exc:
            self.layout_panel.set_validation(str(exc))
            self.layout_panel.set_setup(
                previous_setup, reserved_names=self._document.source_names()
            )
            return False
        if not changed:
            self.layout_panel.set_setup(
                previous_setup, reserved_names=self._document.source_names()
            )
            return False
        self._finish_layout_change(previous_setup)
        return True

    @QtCore.Slot(str)
    def _layout_mode_requested(self, mode: str) -> None:
        if self._updating_controls or self._closing:
            return
        previous_setup = self._document.recipe.setup
        try:
            changed = self._document.convert_layout_mode(
                typing.cast('typing.Literal["subplots", "gridspec"]', mode)
            )
        except ValueError as exc:
            self.layout_panel.set_validation(str(exc))
            self.layout_panel.set_setup(
                previous_setup, reserved_names=self._document.source_names()
            )
            return
        if not changed:
            self.layout_panel.set_setup(
                previous_setup, reserved_names=self._document.source_names()
            )
            return
        self._finish_layout_change(previous_setup)

    def _finish_layout_change(self, previous_setup: FigureSubplotsState) -> None:
        setup = self._document.recipe.setup
        self.layout_panel.set_setup(setup, reserved_names=self._document.source_names())
        size_changed = (setup.figsize, setup.dpi) != (
            previous_setup.figsize,
            previous_setup.dpi,
        )
        if (
            size_changed
            and self._figure_window is not None
            and erlab.interactive.utils.qt_is_valid(self._figure_window)
            and self._figure_window.isVisible()
        ):
            self._figure_window.resize_to_setup(setup)
            self.canvas.flush_events()
            self._sync_recipe_figsize_to_canvas(draw=False, emit_info=False)
        self._rebuild_axes_grid()
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._update_operation_editor()
        self._maybe_redraw_plot()
        self.sigInfoChanged.emit()
        self._write_state()

    def _grow_subplot_grid(self, direction: typing.Literal["row", "column"]) -> bool:
        setup = self._document.recipe.setup
        if setup.layout_mode != "subplots":
            return False
        updates: dict[str, int]
        if direction == "row":
            if setup.nrows >= self.layout_panel.nrows_spin.maximum():
                return False
            updates = {"nrows": setup.nrows + 1}
        else:
            if setup.ncols >= self.layout_panel.ncols_spin.maximum():
                return False
            updates = {"ncols": setup.ncols + 1}
        return self._commit_layout_setup(setup.model_copy(update=updates))

    def _set_layout_engine(self, text: str) -> bool:
        layout = None if text == "default" else text
        setup = self._document.recipe.setup.model_copy(update={"layout": layout})
        return self._commit_layout_setup(setup)

    @QtCore.Slot(object)
    def _axes_selection_changed(self, axes_obj: object) -> None:
        if self._updating_controls:
            return
        current = self._current_operation()
        if current is None:
            return
        if not _registry.spec_for(current[1].kind).uses_axes(current[1]):
            return
        axes = typing.cast("tuple[tuple[int, int], ...]", axes_obj)
        if not axes:
            axes = ((0, 0),)
        index, operation = current
        selection = operation.axes.model_copy(update={"axes": axes, "expression": ""})
        if selection == operation.axes:
            erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)
            return
        self._replace_operation(
            index,
            operation.model_copy(update={"axes": selection}),
            defer_render=True,
            sync_axes=False,
        )
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot()
    def _gridspec_axes_selection_changed(self) -> None:
        if self._updating_controls:
            return
        current = self._current_operation()
        if current is None:
            return
        if not _registry.spec_for(current[1].kind).uses_axes(current[1]):
            return
        axes_ids = self.gridspec_axes_selector.selected_axes_ids()
        index, operation = current
        selection = operation.axes.model_copy(
            update={"axes_ids": axes_ids, "expression": ""}
        )
        if selection == operation.axes:
            erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)
            return
        self._replace_operation(
            index,
            operation.model_copy(update={"axes": selection}),
            defer_render=True,
            sync_axes=False,
        )
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot()
    def _axes_expression_changed(self) -> None:
        if self._updating_controls:
            return
        current = self._current_operation()
        if current is None:
            return
        if not _registry.spec_for(current[1].kind).uses_axes(current[1]):
            return
        index, operation = current
        selection = operation.axes.model_copy(
            update={"expression": self.axes_expression_edit.text().strip()}
        )
        if selection == operation.axes:
            erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)
            return
        self._replace_operation(
            index,
            operation.model_copy(update={"axes": selection}),
            defer_render=True,
            sync_axes=False,
        )
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot(object, object)
    def _operation_selection_changed(
        self,
        _current_id: object,
        _selected_ids: object,
    ) -> None:
        self._update_step_action_buttons()
        self._sync_axes_selector()
        self._refresh_operation_editor()

    def _refresh_operation_editor(self) -> None:
        if self._defer_restore_work(
            self._update_operation_editor,
            key=_RESTORE_OPERATION_EDITOR_KEY,
            run_on_show=True,
        ):
            return
        self._update_operation_editor()

    def _disconnect_step_clipboard(self) -> None:
        clipboard = self._connected_step_clipboard
        self._connected_step_clipboard = None
        if clipboard is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                clipboard.dataChanged.disconnect(self._update_step_action_buttons)

    def _source_data_history_state(
        self,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
        return dict(self._document.source_data), dict(
            self._document.source_selection_base_data
        )

    def _restore_source_data_history_state(
        self,
        state: tuple[Mapping[str, xr.DataArray], Mapping[str, xr.DataArray]],
    ) -> None:
        source_data, selection_base_data = state
        self._document.replace_source_payloads(source_data, selection_base_data)
        self._mark_preview_pixmap_stale()

    def _clear_pending_figure_resize_history_write(self) -> None:
        self._figure_resize_history_timer.stop()
        self._figure_resize_history_pending = False
        self._figure_resize_history_state = None
        self._figure_resize_history_source_data = None

    def _reset_history_stack(self) -> None:
        self._clear_pending_figure_resize_history_write()
        self._prev_states.clear()
        self._next_states.clear()
        self._prev_source_data_states.clear()
        self._next_source_data_states.clear()
        self._prev_states.append(self.tool_status)
        self._prev_source_data_states.append(self._source_data_history_state())
        self._update_history_actions()

    @contextlib.contextmanager
    def _history_suppressed(self):
        self._flush_pending_figure_resize_history_write()
        with super()._history_suppressed():
            yield

    def _append_history_state(
        self,
        state: FigureRecipeState | None = None,
        source_data: (
            tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]] | None
        ) = None,
    ) -> bool:
        curr_state = self.tool_status if state is None else state
        curr_source_data = (
            self._source_data_history_state() if source_data is None else source_data
        )
        last_state = self._prev_states[-1] if self._prev_states else None
        if self._history_state_equal(last_state, curr_state):
            return False
        self._prev_states.append(curr_state)
        self._prev_source_data_states.append(curr_source_data)
        self._next_states.clear()
        self._next_source_data_states.clear()
        self._update_history_actions()
        return True

    def _queue_figure_resize_history_write(self) -> None:
        if not self._write_history:
            return
        self._figure_resize_history_pending = True
        self._figure_resize_history_state = self.tool_status
        self._figure_resize_history_source_data = self._source_data_history_state()
        self._figure_resize_history_timer.start()

    @QtCore.Slot()
    def _flush_pending_figure_resize_history_write(self) -> bool:
        if not self._figure_resize_history_pending:
            return False
        self._figure_resize_history_timer.stop()
        state = self._figure_resize_history_state
        source_data = self._figure_resize_history_source_data
        self._figure_resize_history_pending = False
        self._figure_resize_history_state = None
        self._figure_resize_history_source_data = None
        if not self._write_history or state is None or source_data is None:
            return False
        return self._append_history_state(state, source_data)

    @QtCore.Slot()
    def _write_state(self, *_args: typing.Any) -> None:
        if not self._write_history:
            return
        self._flush_pending_figure_resize_history_write()
        self._append_history_state()

    @QtCore.Slot()
    def _replace_last_state(self, *_args: typing.Any) -> None:
        if not self._write_history:
            return
        self._flush_pending_figure_resize_history_write()
        curr_state = self.tool_status
        source_data = self._source_data_history_state()
        if self._prev_states:
            self._prev_states[-1] = curr_state
            self._prev_source_data_states[-1] = source_data
        else:
            self._prev_states.append(curr_state)
            self._prev_source_data_states.append(source_data)
        self._update_history_actions()

    @QtCore.Slot()
    def undo(self) -> None:
        """Undo the most recent recorded Figure Composer recipe change."""
        self._flush_pending_figure_resize_history_write()
        if not self.undoable:
            return
        with self._history_suppressed():
            self._next_states.append(self._prev_states.pop())
            self._next_source_data_states.append(self._prev_source_data_states.pop())
            self._restore_source_data_history_state(self._prev_source_data_states[-1])
            self.tool_status = self._prev_states[-1]
        self._update_history_actions()

    @QtCore.Slot()
    def redo(self) -> None:
        """Redo the most recently undone Figure Composer recipe change."""
        self._flush_pending_figure_resize_history_write()
        if not self.redoable:
            return
        with self._history_suppressed():
            next_state = self._next_states.pop()
            next_source_data = self._next_source_data_states.pop()
            self._prev_states.append(next_state)
            self._prev_source_data_states.append(next_source_data)
            self._restore_source_data_history_state(next_source_data)
            self.tool_status = next_state
        self._update_history_actions()

    @QtCore.Slot(str, bool)
    def _operation_enabled_requested(self, operation_id: str, enabled: bool) -> None:
        if self._updating_controls:
            return
        index = self._document.operation_index(operation_id)
        if index is None:
            return
        operation = self._document.recipe.operations[index]
        updated = operation.model_copy(update={"enabled": enabled})
        if updated == operation:
            return
        self._document.replace_operation(index, updated)
        if index == self._current_operation_index():
            self._sync_axes_selector()
            self._update_source_status(updated)
        self._refresh_step_section_button_texts()
        if self._operation_change_affects_preview(operation, updated):
            self._notify_operation_preview_changed()
        self._write_state()

    @QtCore.Slot()
    def _target_current_operation_all_axes(self) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        if not _registry.spec_for(operation.kind).uses_axes(operation):
            return
        if self._document.recipe.setup.layout_mode == "gridspec":
            selection = operation.axes.model_copy(
                update={
                    "axes_ids": _gridspec_all_axes_ids(self._document.recipe.setup),
                    "expression": "",
                }
            )
            self._replace_operation(
                index, operation.model_copy(update={"axes": selection})
            )
            return
        selection = operation.axes.model_copy(
            update={"axes": _all_axes(self._document.recipe.setup), "expression": ""}
        )
        self._replace_operation(index, operation.model_copy(update={"axes": selection}))

    @QtCore.Slot()
    def _target_current_operation_valid_axes(self) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        if not _registry.spec_for(operation.kind).uses_axes(operation):
            return
        if self._document.recipe.setup.layout_mode == "gridspec":
            axes_ids = _gridspec_valid_axes_ids(
                self._document.recipe.setup, operation.axes.axes_ids
            )
            if not axes_ids:
                axes_ids = _gridspec_all_axes_ids(self._document.recipe.setup)[:1]
            selection = operation.axes.model_copy(
                update={"axes_ids": axes_ids, "expression": ""}
            )
            self._replace_operation(
                index, operation.model_copy(update={"axes": selection})
            )
            return
        axes = operation.axes.valid_axes(self._document.recipe.setup)
        if not axes:
            axes = ((0, 0),)
        selection = operation.axes.model_copy(update={"axes": axes, "expression": ""})
        self._replace_operation(index, operation.model_copy(update={"axes": selection}))

    def _source_display_name(self, name: str) -> str:
        sources = self._document.source_by_name()
        source = sources.get(name)
        return _source_display_label(source, name)

    def _source_display_names(self, names: Sequence[str]) -> tuple[str, ...]:
        return tuple(self._source_display_name(name) for name in names)

    def _source_detail_context_lines(self, name: str) -> tuple[str, ...]:
        source = self._document.source_by_name().get(name)
        lines: list[str] = []
        if source is None:
            lines.append("This source is missing from the recipe")
        else:
            if source.selection_source is not None and source.selection_source != name:
                lines.append(f"Selected from {source.selection_source}")
            if origin := self._source_refresh_label(name):
                lines.append(f"From ImageTool {origin}")
            elif source.node_uid is not None:
                lines.append("Original ImageTool is no longer available")
        usage_count = self._document.source_usage_count(name)
        if usage_count:
            suffix = "step" if usage_count == 1 else "steps"
            lines.append(f"Used by {usage_count} recipe {suffix}")
        else:
            lines.append("Not used by any recipe steps")
        return tuple(lines)

    def _source_tooltip(self, name: str) -> str:
        lines = [
            source_metadata_tooltip(
                self._document.source_by_name().get(name),
                name,
                self._document.source_data.get(name),
            ),
            *self._source_detail_context_lines(name),
        ]
        if self._source_refresh_available(name):
            lines.append("Can refresh from the linked ImageTool")
        return "\n".join(lines)

    def _selected_axes_state(self) -> FigureAxesSelectionState:
        if self._document.recipe.setup.layout_mode == "gridspec":
            axes_ids = self.gridspec_axes_selector.selected_axes_ids()
            if not axes_ids:
                axes_ids = _gridspec_all_axes_ids(self._document.recipe.setup)[:1]
            return FigureAxesSelectionState(axes_ids=axes_ids)
        axes = self.axes_selector.selected_axes()
        if not axes:
            axes = ((0, 0),)
        return FigureAxesSelectionState(
            axes=axes,
            expression=self.axes_expression_edit.text().strip(),
        )

    def _editor_first_live_axis(
        self, selection: FigureAxesSelectionState
    ) -> typing.Any:
        layout_axes = _live_layout_axes(self, render_if_missing=True)
        if layout_axes is None:
            return None
        if isinstance(layout_axes, dict) and not selection.axes_ids:
            selection = selection.model_copy(
                update={
                    "axes_ids": _gridspec_valid_axes_ids(
                        self._document.recipe.setup,
                        _gridspec_all_axes_ids(self._document.recipe.setup),
                    )[:1]
                }
            )
        try:
            axes = _iter_axes(
                _axes_from_selection(
                    self,
                    selection,
                    layout_axes,
                    for_plot_slices=False,
                )
            )
        except (IndexError, TypeError, ValueError):
            return None
        return axes[0] if axes else None

    def _editor_subplot_parameter_default(self, key: str) -> float:
        figure_window = self._figure_window
        if figure_window is not None and erlab.interactive.utils.qt_is_valid(
            figure_window
        ):
            return float(getattr(figure_window.figure.subplotpars, key))
        figure = Figure(
            figsize=self._document.recipe.setup.figsize,
            dpi=self._document.recipe.setup.dpi,
            layout=typing.cast("typing.Any", self._document.recipe.setup.layout),
        )
        return float(getattr(figure.subplotpars, key))

    def _editor_rendered_value(
        self,
        operation: FigureOperationState,
        reader: Callable[[Sequence[typing.Any]], typing.Any],
    ) -> typing.Any:
        if self._preview_render_update_pending or self._operation_has_invalid_axes(
            operation
        ):
            return None
        layout_axes = _live_layout_axes(self)
        if layout_axes is None:
            return None
        try:
            axes = _iter_axes(
                _axes_from_selection(
                    self,
                    operation.axes,
                    layout_axes,
                    for_plot_slices=False,
                )
            )
            return None if not axes else reader(axes)
        except Exception:
            return None

    def _editor_styled_rcparams_value(self, key: str) -> typing.Any:
        with self._figure_options_context(), _figure_style_context():
            return _styled_rcparams_value(key)

    def _add_operation(self, action_id: str) -> None:
        operation = _registry.spec_for_action(action_id).create_operation(self)
        index = self._document.append_operation(operation)
        self._refresh_operation_list()
        self.operation_panel.select_row(index)
        self._maybe_redraw_plot()
        self.sigInfoChanged.emit()
        self._write_state()

    def _clipboard(self) -> QtGui.QClipboard | None:
        application = QtWidgets.QApplication.instance()
        if not isinstance(application, QtWidgets.QApplication):
            return None
        return application.clipboard()

    def _clipboard_step_payload(
        self,
    ) -> (
        tuple[
            tuple[FigureOperationState, ...],
            tuple[FigureSourceState, ...],
            dict[str, xr.DataArray],
            dict[str, xr.DataArray],
        ]
        | None
    ):
        clipboard = self._clipboard()
        if clipboard is None:
            return None
        return _step_clipboard_payload(clipboard.mimeData())

    def _write_selected_operations_to_clipboard(
        self, *, cut: bool = False
    ) -> tuple[int, ...] | None:
        indices = self._selected_operation_indices()
        if not indices:
            return None
        operations = tuple(self._document.recipe.operations[index] for index in indices)
        source_names = tuple(
            dict.fromkeys(
                source_name
                for operation in operations
                for source_name in self._document.operation_source_dependency_names(
                    operation
                )
            )
        )
        source_by_name = {
            source.name: source for source in self._document.recipe.sources
        }
        sources = tuple(
            source_by_name.get(source_name, FigureSourceState(name=source_name))
            for source_name in source_names
        )
        source_data = {
            source_name: self._document.source_data[source_name].copy(deep=False)
            for source_name in source_names
            if source_name in self._document.source_data
        }
        selection_base_data = {
            source_name: self._document.source_selection_base_data[source_name].copy(
                deep=False
            )
            for source_name in source_names
            if source_name in self._document.source_selection_base_data
        }
        clipboard = self._clipboard()
        if clipboard is None:
            return None
        clipboard.setMimeData(
            _FigureComposerStepMimeData(
                _step_clipboard_payload_text(operations, sources),
                _step_clipboard_code_text(self, operations),
                source_data,
                selection_base_data,
                cut_source_tool_id=self._step_clipboard_tool_id if cut else None,
            )
        )
        self._update_step_action_buttons()
        return indices

    @QtCore.Slot()
    def _copy_selected_operations(self) -> None:
        self._write_selected_operations_to_clipboard()

    @QtCore.Slot()
    def _cut_selected_operations(self) -> None:
        indices = self._write_selected_operations_to_clipboard(cut=True)
        if indices is None:
            return
        self._remove_operations_at_indices(indices)

    @QtCore.Slot()
    def _paste_operations_from_clipboard(self) -> None:
        clipboard = self._clipboard()
        if clipboard is None:
            return
        mime = clipboard.mimeData()
        payload = _step_clipboard_payload(mime)
        if payload is None:
            return
        operations, sources, source_data, selection_base_data = payload
        indices = self._selected_operation_indices()
        current = self._current_operation()
        if indices:
            insert_index = max(indices) + 1
        elif current is not None:
            insert_index = current[0] + 1
        else:
            insert_index = len(self._document.recipe.operations)
        result = self._document.paste_operations(
            insert_index,
            operations,
            sources,
            source_data,
            selection_base_data,
            preserve_existing=(
                getattr(mime, "figure_composer_cut_source_tool_id", None)
                == self._step_clipboard_tool_id
            ),
        )
        normalization_data_changed = self._normalize_operation_source_selections()
        self._refresh_source_list()
        if result.source_data_changed or normalization_data_changed:
            self.sigDataChanged.emit()
        self._finish_operation_structure_change(
            set(result.operation_ids),
            result.operation_ids[0],
        )

    def _remove_operations_at_indices(self, indices: Sequence[int]) -> None:
        if not indices:
            return
        self._document.remove_operation_indices(indices)
        operations = self._document.recipe.operations
        selected_ids: set[str] = set()
        current_id: str | None = None
        if operations:
            current_index = min(min(indices), len(operations) - 1)
            current_id = operations[current_index].operation_id
            selected_ids = {current_id}
        self._finish_operation_structure_change(selected_ids, current_id)

    @QtCore.Slot()
    def _remove_current_operation(self) -> None:
        self._remove_operations_at_indices(self._selected_operation_indices())

    @QtCore.Slot()
    def _duplicate_current_operation(self) -> None:
        indices = self._selected_operation_indices()
        if not indices:
            return
        duplicate_ids = self._document.duplicate_operations(indices)
        self._finish_operation_structure_change(
            set(duplicate_ids),
            duplicate_ids[0],
        )

    @QtCore.Slot(object, object, object)
    def _operation_list_reordered(
        self,
        operation_ids: object,
        selected_ids: object,
        current_id: object,
    ) -> None:
        if not isinstance(operation_ids, (tuple, list)):
            self._refresh_operation_list()
            return
        ordered_ids = tuple(
            operation_id
            for operation_id in operation_ids
            if isinstance(operation_id, str)
        )
        operation_by_id = {
            operation.operation_id: operation
            for operation in self._document.recipe.operations
        }
        if (
            len(ordered_ids) != len(operation_ids)
            or len(ordered_ids) != len(operation_by_id)
            or set(ordered_ids) != set(operation_by_id)
        ):
            self._refresh_operation_list()
            return
        current_order = tuple(
            operation.operation_id for operation in self._document.recipe.operations
        )
        if ordered_ids == current_order:
            return
        selected_id_set: set[str] = set()
        if isinstance(selected_ids, (set, frozenset, tuple, list)):
            selected_id_set = {
                operation_id
                for operation_id in selected_ids
                if isinstance(operation_id, str) and operation_id in operation_by_id
            }
        current_operation_id = (
            current_id
            if isinstance(current_id, str) and current_id in operation_by_id
            else None
        )
        if not selected_id_set and current_operation_id is not None:
            selected_id_set = {current_operation_id}
        if current_operation_id is None:
            current_operation_id = next(
                iter(selected_id_set),
                ordered_ids[0] if ordered_ids else None,
            )

        self._document.reorder_operations(ordered_ids)
        self._finish_operation_structure_change(selected_id_set, current_operation_id)

    def _move_current_operation(self, offset: int) -> None:
        indices = self._selected_operation_indices()
        if not indices:
            return
        selected_ids = {
            self._document.recipe.operations[index].operation_id for index in indices
        }
        current_id = self.operation_panel.current_id()
        if current_id not in selected_ids:
            current_id = None
        if not self._document.move_operations(selected_ids, offset):
            return
        operations = self._document.recipe.operations
        if current_id is None:
            current_id = next(
                operation.operation_id
                for operation in operations
                if operation.operation_id in selected_ids
            )
        self._finish_operation_structure_change(selected_ids, current_id)

    def _finish_operation_structure_change(
        self, selected_ids: set[str], current_id: str | None
    ) -> None:
        self._refresh_operation_list()
        if selected_ids:
            self._set_selected_operation_ids_silent(selected_ids)
        if current_id is not None:
            for row, operation in enumerate(self._document.recipe.operations):
                if operation.operation_id == current_id:
                    self._set_current_operation_row_silent(row)
                    break
        self._sync_axes_selector()
        self._update_operation_editor()
        self._maybe_redraw_plot()
        self.sigInfoChanged.emit()
        self._write_state()

    @QtCore.Slot()
    def _move_current_operation_up(self) -> None:
        self._move_current_operation(-1)

    @QtCore.Slot()
    def _move_current_operation_down(self) -> None:
        self._move_current_operation(1)

    def add_operation(self, operation: FigureOperationState) -> None:
        """Append an operation to the recipe."""
        index = self._document.append_operation(operation)
        data_changed = self._normalize_operation_source_selections()
        self._apply_recipe_to_controls()
        self.operation_panel.select_row(index)
        self._maybe_redraw_plot()
        if data_changed:
            self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()

    def add_sources(
        self,
        sources: Sequence[FigureSourceState],
        source_data: Mapping[str, xr.DataArray],
    ) -> FigureSourceAddResult:
        """Add or update source data without changing existing recipe steps.

        This supports appending operations and the manager's source-only workflow. The
        source list, backing data, preview, persistent state, and data-dirty signals are
        updated together so workspace saves include the new source data. The result
        identifies accepted additions and linked-source updates, including the stored
        recipe name chosen for each requested source.
        """
        result = self._document.add_sources(sources, source_data)
        skipped_details = tuple(detail for _name, detail in result.skipped)
        if not result:
            if result.skipped:
                self._set_source_panel_status(
                    "Could not update source data for: " + ", ".join(skipped_details)
                )
            return result
        self._normalize_operation_source_selections()
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()
        self._set_source_panel_status(
            "Could not update source data for: " + ", ".join(skipped_details)
            if result.skipped
            else None
        )
        return result

    def replace_source(
        self,
        alias: str,
        source: FigureSourceState,
        data: xr.DataArray,
    ) -> bool:
        """Replace source data while preserving the recipe-facing source name.

        The incoming source metadata and data replace the stored source slot, but
        recipe steps and generated code keep referring to the stored source name.
        Returns ``False`` when no matching stored source or backing data slot exists.
        """
        result = self._document.replace_source(alias, source, data)
        if not result:
            detail = result.skipped[0][1] if result.skipped else "source unavailable"
            self._set_source_panel_status(
                f"Could not refresh source “{alias}”: {detail}"
            )
            return False
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()
        self._set_source_panel_status(None)
        return True

    def remove_source(self, name: str) -> bool:
        """Remove an unused source from this figure."""
        if not self._document.remove_source(name):
            return False
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()
        return True

    def _set_step_sections(self, sections: Sequence[StepSection]) -> None:
        self.operation_editor.replace_sections(
            sections,
            summaries=self._step_section_summaries(
                tuple(section.key for section in sections)
            ),
        )

    def _refresh_step_section_button_texts(self) -> None:
        self.operation_editor.set_section_summaries(
            self._step_section_summaries(self.operation_editor.section_keys)
        )

    def _step_section_summaries(self, keys: Sequence[str]) -> dict[str, str]:
        current = self._current_operation()
        if current is None:
            return {}
        _index, operation = current
        return {
            key: summary
            for key in keys
            if (summary := self._section_summary(key, operation))
        }

    def _section_summary(self, key: str, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).section_summary(self, key, operation)

    def _selected_sources_for_operation(
        self, operation: FigureOperationState
    ) -> tuple[str, ...]:
        return declared_operation_source_names(operation)

    def _update_source_status(self, operation: FigureOperationState | None) -> None:
        if operation is None:
            self.operation_editor.set_source_status(
                "Select a step to choose data sources."
            )
            return
        if not _registry.spec_for(operation.kind).uses_source_section(operation):
            self.operation_editor.set_source_status(None)
            return
        if (
            input_error := self.operation_editor.input_error_text(operation)
        ) is not None:
            self.operation_editor.set_source_status(f"Invalid input: {input_error}")
            return
        if (
            render_error := self._operation_render_errors.get(operation.operation_id)
        ) is not None:
            self.operation_editor.set_source_status(f"Render error: {render_error}")
            return
        selected_sources = self._selected_sources_for_operation(operation)
        missing = [
            source
            for source in selected_sources
            if source not in self._document.source_data
        ]
        if missing:
            self.operation_editor.set_source_status(
                "Missing sources: " + ", ".join(self._source_display_names(missing))
            )
        elif selected_sources:
            self.operation_editor.set_source_status(None)
        else:
            self.operation_editor.set_source_status(
                "This step does not read a data source."
            )

    def _update_source_section(self) -> None:
        self.operation_editor.clear_source_controls()
        current = self._current_operation()
        if current is None:
            self._update_source_status(None)
            return
        _index, operation = current

        _registry.spec_for(operation.kind).build_source_editor(
            self.operation_editor, operation
        )
        self._update_source_status(operation)

    def _update_operation_editor(self) -> None:
        self.operation_editor.prepare_rebuild()
        self._update_source_section()
        current = self._current_operation()
        self._update_step_action_buttons()
        selected_indices = self._selected_operation_indices()
        if len(selected_indices) > 1 and not self._selected_operations_are_compatible():
            page, layout = self.operation_editor.new_form_page(
                "figureComposerIncompatibleBatchPage"
            )
            label = QtWidgets.QLabel(
                "Select steps with the same editor type to batch edit them.",
                page,
            )
            label.setWordWrap(True)
            label.setEnabled(False)
            layout.addRow(QtWidgets.QLabel("Batch edit", page), label)
            self._set_step_sections(
                (
                    StepSection(
                        "batch",
                        "Batch",
                        page,
                        "The selected steps do not share one editor schema.",
                    ),
                )
            )
            return
        if current is None:
            page, layout = self.operation_editor.new_form_page(
                "figureComposerNoStepPage"
            )
            layout.addRow(QtWidgets.QLabel("Select a recipe step to edit.", page))
            self._set_step_sections(
                (StepSection("empty", "Step", page, "Select a recipe step."),)
            )
            return
        _index, operation = current

        spec = _registry.spec_for(operation.kind)
        sections: list[StepSection] = []
        if spec.uses_source_section(operation):
            sections.append(
                StepSection(
                    "sources",
                    "Sources",
                    self.operation_editor.source_page,
                    COMMON_SOURCE_SECTION_TOOLTIP,
                )
            )
        if spec.uses_axes(operation):
            sections.append(
                StepSection(
                    "axes",
                    "Axes",
                    self.target_axes_page,
                    COMMON_AXES_SECTION_TOOLTIP,
                )
            )
        sections.extend(spec.build_editor_sections(self.operation_editor, operation))
        self._set_step_sections(sections)

    def _update_operation_editor_safely(self) -> None:
        if (
            self.operation_editor.rebuild_must_wait()
            or self._operation_editor_sender_requires_deferred_rebuild()
        ):
            self._queue_operation_editor_update()
            return
        self._update_operation_editor()

    def _operation_editor_sender_requires_deferred_rebuild(self) -> bool:
        sender = self.operation_editor.active_signal_widget
        if sender is None:
            qt_sender = self.sender()
            sender = qt_sender if isinstance(qt_sender, QtWidgets.QWidget) else None
        if sender is None or not erlab.interactive.utils.qt_is_valid(sender):
            return False
        return self.operation_editor.contains_widget(
            sender
        ) or self.source_panel.source_list.isAncestorOf(sender)

    def _queue_operation_editor_update(self) -> None:
        if self._operation_editor_update_pending or self._closing:
            return
        self._operation_editor_update_pending = True
        self._schedule_queued_operation_editor_update()

    def _schedule_queued_operation_editor_update(self) -> None:
        erlab.interactive.utils.single_shot(
            self,
            _OPERATION_EDITOR_UPDATE_DELAY_MS,
            self._run_queued_operation_editor_update,
        )

    def _run_queued_operation_editor_update(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        if self._closing:
            self._operation_editor_update_pending = False
            return
        if self.operation_editor.rebuild_must_wait():
            self._schedule_queued_operation_editor_update()
            return
        self._operation_editor_update_pending = False
        self._update_operation_editor()

    def generated_code(self) -> str:
        self._flush_restore_work()
        self.operation_editor.flush_pending_commits()
        with self._figure_options_context():
            return erlab.interactive._figurecomposer._codegen.generated_code(self)

    @QtCore.Slot()
    def copy_code(self) -> None:
        self._flush_restore_work()
        if self._warn_invalid_operation_targets():
            return
        try:
            code = self.generated_code()
        except ValueError as exc:
            erlab.interactive.utils.MessageDialog(
                self,
                title="Cannot Copy Figure Code",
                text=str(exc),
                detailed_text=erlab.interactive.utils._format_traceback(
                    traceback.format_exc()
                ),
                buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
                icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
            ).exec()
            return
        erlab.interactive.utils.copy_to_clipboard(code)

    @QtCore.Slot()
    def export_figure(self) -> None:
        self._flush_restore_work()
        if self._warn_invalid_operation_targets():
            return
        self.operation_editor.flush_pending_commits()
        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Figure",
            "",
            "Images (*.png *.pdf *.svg);;All files (*)",
        )
        if not filename:
            return
        with (
            self._figure_options_context(),
            _rendered_output_figure(self) as figure,
            _figure_draw_context(),
        ):
            figure.savefig(
                filename,
                dpi=self._document.recipe.export.dpi,
                transparent=self._document.recipe.export.transparent,
                bbox_inches=self._document.recipe.export.bbox_inches,
            )

    @property
    def tool_data(self) -> xr.DataArray:
        """Return the live primary source; call ``touch_source_data`` after edits."""
        if self._document.recipe.primary_source in self._document.source_data:
            return self._document.source_data[self._document.recipe.primary_source]
        return next(iter(self._document.source_data.values()))

    @property
    def tool_status(self) -> FigureRecipeState:
        return self._document.recipe

    @tool_status.setter
    def tool_status(self, status: FigureRecipeState) -> None:
        operations = tuple(
            _registry.spec_for(operation.kind).loaded_operation(operation)
            for operation in status.operations
        )
        sources = FigureDocument.normalized_source_states(status.sources)
        self._document.replace_recipe(
            status.model_copy(update={"sources": sources, "operations": operations})
        )
        self._ensure_primary_source_data()
        self._normalize_operation_source_selections()
        self._apply_recipe_to_controls()
        self._sync_figure_window_to_recipe_setup()
        if self._dataset_restore_in_progress:
            self._mark_preview_pixmap_stale()
            return
        _render_preview(self)

    def set_source_data(self, source_data: Mapping[str, xr.DataArray]) -> None:
        self._document.replace_source_payloads(source_data, {})
        self._refresh_source_list()
        self._mark_preview_pixmap_stale()

    def touch_source_data(self) -> None:
        """Invalidate prepared selections after editing source arrays in place.

        .. versionadded:: 3.25.0
        """
        self._document.touch_source_payloads()
        self._mark_preview_pixmap_stale()
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()

    def rebase_source_node_uids(self, uid_map: Mapping[str, str]) -> None:
        if not uid_map:
            return
        changed = False
        sources: list[FigureSourceState] = []
        for source in self._document.recipe.sources:
            updates: dict[str, typing.Any] = {}
            if source.node_uid is not None and source.node_uid in uid_map:
                updates["node_uid"] = uid_map[source.node_uid]
            if source.provenance_spec is not None:
                try:
                    rebased = rebase_script_input_node_uids(
                        source.provenance_spec, uid_map
                    )
                except (TypeError, ValueError):
                    pass
                else:
                    provenance_spec = rebased.model_dump(mode="json")
                    if provenance_spec != source.provenance_spec:
                        updates["provenance_spec"] = provenance_spec
            if updates:
                changed = True
                sources.append(source.model_copy(update=updates))
            else:
                sources.append(source)
        if not changed:
            return
        self._document.replace_recipe(
            self._document.recipe.model_copy(update={"sources": tuple(sources)})
        )
        self.sigInfoChanged.emit()

    def _ensure_primary_source_data(self) -> None:
        if (
            self._document.recipe.primary_source in self._document.source_data
            or not self._document.source_data
        ):
            return
        recipe_sources = {source.name for source in self._document.recipe.sources}
        source_data = dict(self._document.source_data)
        fallback_name, fallback_data = next(iter(self._document.source_data.items()))
        source_data[self._document.recipe.primary_source] = fallback_data
        if fallback_name not in recipe_sources:
            del source_data[fallback_name]
        self._document.replace_source_payloads(
            source_data, self._document.source_selection_base_data
        )

    def _recipe_source(self, source_name: str) -> FigureSourceState | None:
        for source in self._document.recipe.sources:
            if source.name == source_name:
                return source
        return None

    def _source_reference_payload(
        self, source_name: str
    ) -> dict[str, typing.Any] | None:
        if (
            not self._save_tool_data_references
            or source_name not in self._document.source_data
        ):
            return None
        source = self._recipe_source(source_name)
        if source is None or source.node_uid is None:
            return None
        allowed_uids = self._save_tool_data_reference_node_uids
        if allowed_uids is not None and source.node_uid not in allowed_uids:
            return None
        payload: dict[str, typing.Any] = {
            "kind": "manager_node",
            "source": source.name,
            "node_uid": source.node_uid,
        }
        if source.node_snapshot_token is not None:
            payload["node_snapshot_token"] = source.node_snapshot_token
        payload["data_role"] = source.data_role
        return payload

    def _tool_data_reference_payload(
        self, variable_name: str, data: xr.DataArray
    ) -> dict[str, typing.Any] | None:
        del data
        if variable_name == erlab.interactive.utils._SAVED_TOOL_DATA_NAME:
            return self._source_reference_payload(self._document.recipe.primary_source)
        return self._source_reference_payload(variable_name)

    @classmethod
    def _missing_saved_tool_data_reference_optional(
        cls,
        variable_name: str,
        reference: Mapping[str, typing.Any],
        ds: xr.Dataset,
    ) -> bool:
        del ds
        return (
            variable_name != erlab.interactive.utils._SAVED_TOOL_DATA_NAME
            and reference.get("kind") == "manager_node"
        )

    def _persistence_data_items(self) -> Mapping[str, xr.DataArray]:
        primary_source = self._document.recipe.primary_source
        if primary_source in self._document.source_data:
            primary_data, _already_selected = self._persistence_source_data(
                primary_source
            )
        else:
            primary_data = self.tool_data
        items = {erlab.interactive.utils._SAVED_TOOL_DATA_NAME: primary_data}
        for source_name in self._document.source_data:
            if source_name == self._document.recipe.primary_source:
                continue
            if source_name == erlab.interactive.utils._SAVED_TOOL_DATA_NAME:
                raise ValueError(
                    "Figure source names cannot use the reserved saved-tool data name"
                )
            data, _already_selected = self._persistence_source_data(source_name)
            items[source_name] = data
        return items

    def _persistence_source_data(self, source_name: str) -> tuple[xr.DataArray, bool]:
        data = self._document.source_data[source_name]
        source = self._recipe_source(source_name)
        if source is None or not _source_has_selection(source):
            return data, False
        base_data = self._document.source_selection_base_data.get(source_name)
        if base_data is not None:
            return base_data, False
        if (
            source.selection_source is not None
            and source.selection_source != source_name
        ):
            base_data = self._document.source_data.get(source.selection_source)
            if base_data is not None:
                return base_data, False
        return data, True

    def _embedded_selected_source_names(self, ds: xr.Dataset) -> tuple[str, ...]:
        references = self._saved_tool_data_references(ds)
        selected_names: list[str] = []
        for source in self._document.recipe.sources:
            if source.name not in self._document.source_data:
                continue
            _data, already_selected = self._persistence_source_data(source.name)
            variable_name = (
                erlab.interactive.utils._SAVED_TOOL_DATA_NAME
                if source.name == self._document.recipe.primary_source
                else source.name
            )
            if already_selected and variable_name not in references:
                selected_names.append(source.name)
        return tuple(selected_names)

    @staticmethod
    def _persisted_selected_source_names(ds: xr.Dataset) -> frozenset[str]:
        payload = ds.attrs.get(_PERSISTED_SELECTED_SOURCE_DATA_ATTR)
        if payload is None:
            return frozenset()
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            logger.debug("Ignoring invalid persisted selected-source metadata")
            return frozenset()
        if not isinstance(decoded, list) or not all(
            isinstance(name, str) for name in decoded
        ):
            logger.debug("Ignoring invalid persisted selected-source metadata")
            return frozenset()
        return frozenset(decoded)

    def _persistence_reference_node_uids(self) -> frozenset[str]:
        return frozenset(
            source.node_uid
            for source in self._document.recipe.sources
            if source.node_uid is not None
        )

    def _saved_tool_status(self) -> FigureRecipeState:
        self._flush_pending_figure_resize_history_write()
        status = self.tool_status
        allowed_uids = self._save_tool_data_reference_node_uids
        if allowed_uids is None:
            return status
        sources = tuple(
            source.model_copy(update={"node_uid": None, "node_snapshot_token": None})
            if source.node_uid is not None and source.node_uid not in allowed_uids
            else source
            for source in status.sources
        )
        if sources == status.sources:
            return status
        return status.model_copy(update={"sources": sources})

    def _restore_persistence_data_items(
        self, data_items: Mapping[str, xr.DataArray], ds: xr.Dataset
    ) -> None:
        source_data = dict(self._document.source_data)
        selection_base_data = dict(self._document.source_selection_base_data)
        embedded_selected_names = self._persisted_selected_source_names(ds)
        source_by_name = self._document.source_by_name()
        persisted_inputs: dict[str, tuple[xr.DataArray, bool]] = {}
        changed = False

        def persisted_data(source: FigureSourceState) -> xr.DataArray | None:
            variable_name = (
                erlab.interactive.utils._SAVED_TOOL_DATA_NAME
                if source.name == self._document.recipe.primary_source
                else source.name
            )
            data = data_items.get(variable_name)
            if data is None:
                return None
            if source.name != self._document.recipe.primary_source:
                return data
            current_primary = source_data.get(source.name)
            if current_primary is not None:
                return data.rename(current_primary.name)
            tool_data_name = ds.attrs.get("tool_data_name", "<none-value>")
            if tool_data_name == "<none-value>":
                tool_data_name = None
            return data.rename(tool_data_name)

        for source in self._document.recipe.sources:
            data = persisted_data(source)
            if data is None:
                continue
            changed = True
            source_data.pop(source.name, None)
            selection_base_data.pop(source.name, None)
            already_selected = source.name in embedded_selected_names
            if source.selection_source is not None:
                persisted_inputs[source.name] = (data, already_selected)
                continue
            if not _source_has_selection(source) or already_selected:
                source_data[source.name] = data
                continue
            try:
                source_data[source.name] = FigureDocument.source_data_from_selection(
                    data, source
                )
            except (IndexError, KeyError, TypeError, ValueError):
                logger.debug(
                    "Could not apply saved Figure Composer source selection for %s",
                    source.name,
                    exc_info=True,
                )
                source_data[source.name] = data
            else:
                selection_base_data[source.name] = data

        resolved = {
            source.name
            for source in self._document.recipe.sources
            if source.selection_source is None and source.name in source_data
        }
        queue: collections.deque[str] = collections.deque(resolved)

        def rebuild_descendants() -> None:
            nonlocal changed
            while queue:
                parent_name = queue.popleft()
                parent_data = source_data.get(parent_name)
                if parent_data is None:
                    continue
                for source in self._document.recipe.sources:
                    if (
                        source.name in resolved
                        or source.selection_source != parent_name
                        or source.name == parent_name
                    ):
                        continue
                    source_data.pop(source.name, None)
                    selection_base_data.pop(source.name, None)
                    try:
                        selected = FigureDocument.source_data_from_selection(
                            parent_data, source
                        )
                    except (IndexError, KeyError, TypeError, ValueError):
                        logger.debug(
                            "Could not rebuild saved Figure Composer source %s from %s",
                            source.name,
                            parent_name,
                            exc_info=True,
                        )
                        continue
                    source_data[source.name] = selected
                    if _source_has_selection(source):
                        selection_base_data[source.name] = parent_data
                    resolved.add(source.name)
                    queue.append(source.name)
                    changed = True

        rebuild_descendants()

        pending = [
            source
            for source in self._document.recipe.sources
            if source.name not in resolved and source.name in persisted_inputs
        ]
        while pending:
            next_pending: list[FigureSourceState] = []
            restored_this_pass = False
            for source in pending:
                if source.name in resolved:
                    continue
                parent_name = source.selection_source
                if parent_name in persisted_inputs and parent_name not in resolved:
                    next_pending.append(source)
                    continue
                if parent_name is not None and parent_name in source_data:
                    # A resolved parent would already have visited this source.  Do
                    # not mask a failed current selection with an older payload.
                    continue
                try:
                    FigureDocument.source_lineage_names(source.name, source_by_name)
                except ValueError:
                    continue
                data, already_selected = persisted_inputs[source.name]
                if already_selected or not _source_has_selection(source):
                    selected = data
                else:
                    try:
                        selected = FigureDocument.source_data_from_selection(
                            data, source
                        )
                    except (IndexError, KeyError, TypeError, ValueError):
                        logger.debug(
                            "Could not restore saved Figure Composer source %s "
                            "without %s",
                            source.name,
                            parent_name,
                            exc_info=True,
                        )
                        continue
                    selection_base_data[source.name] = data
                source_data[source.name] = selected
                resolved.add(source.name)
                queue.append(source.name)
                changed = True
                restored_this_pass = True
                rebuild_descendants()
            if not restored_this_pass:
                break
            pending = next_pending

        unresolved = tuple(
            source.name
            for source in self._document.recipe.sources
            if source.selection_source is not None and source.name not in source_data
        )
        if unresolved:
            logger.debug(
                "Could not resolve saved Figure Composer source dependencies: %s",
                ", ".join(unresolved),
            )
        if not changed:
            self._restore_persisted_preview_cache(ds)
            self._queue_post_restore_redraw_if_needed(ds)
            return
        self._document.replace_source_payloads(source_data, selection_base_data)
        self._refresh_source_list()
        self._mark_preview_pixmap_stale()
        self._normalize_operation_source_selections()
        self._apply_recipe_to_controls()
        self._restore_persisted_preview_cache(ds)
        self._queue_post_restore_redraw_if_needed(ds)

    @staticmethod
    def _saved_tool_window_visible(ds: xr.Dataset) -> bool:
        state = _qt_state.parse_qt_window_state(ds.attrs.get("tool_window_state"))
        if state is not None:
            return state.visible
        return bool(ds.attrs.get("tool_visible", False))

    def _queue_post_restore_redraw_if_needed(self, ds: xr.Dataset) -> None:
        if not self._saved_tool_window_visible(ds) or not self._auto_redraw_enabled():
            return
        if self._defer_restore_work(
            self._redraw_restored_plot,
            key=_RESTORE_REDRAW_KEY,
            run_on_show=True,
        ):
            return
        erlab.interactive.utils.single_shot(
            self,
            0,
            functools.partial(self._redraw_plot, show_window=True),
        )

    def _redraw_restored_plot(self) -> None:
        self._redraw_plot(show_window=True)

    def _flush_restore_work_for_save(self) -> None:
        self._flush_restore_work(
            skip=(_RESTORE_OPERATION_EDITOR_KEY, _RESTORE_REDRAW_KEY)
        )

    def _persisted_preview_cache_pixmap(self) -> QtGui.QPixmap | None:
        preview = self._preview_pixmap_cache
        if preview is None or preview.isNull():
            return None
        if (
            preview.width() > _PERSISTED_PREVIEW_CACHE_SIZE[0]
            or preview.height() > _PERSISTED_PREVIEW_CACHE_SIZE[1]
        ):
            return preview.scaled(
                QtCore.QSize(*_PERSISTED_PREVIEW_CACHE_SIZE),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        return QtGui.QPixmap(preview)

    def _append_persistence_payload(self, ds: xr.Dataset) -> xr.Dataset:
        selected_names = self._embedded_selected_source_names(ds)
        if selected_names:
            ds = ds.copy(deep=False)
            ds.attrs[_PERSISTED_SELECTED_SOURCE_DATA_ATTR] = json.dumps(selected_names)

        preview = self._persisted_preview_cache_pixmap()
        if preview is None:
            return ds

        data = QtCore.QByteArray()
        buffer = QtCore.QBuffer(data)
        if not buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
            return ds
        try:
            if not preview.save(buffer, "PNG"):
                return ds
        finally:
            buffer.close()

        png_bytes = data.data()
        if len(png_bytes) > _PERSISTED_PREVIEW_CACHE_MAX_BYTES:
            return ds

        ds = ds.copy(deep=False)
        ds.attrs[_PERSISTED_PREVIEW_CACHE_ATTR] = base64.b64encode(png_bytes).decode(
            "ascii"
        )
        ds.attrs[_PERSISTED_PREVIEW_CACHE_STALE_ATTR] = bool(self._preview_pixmap_stale)
        return ds

    def _restore_persisted_preview_cache(self, ds: xr.Dataset) -> None:
        encoded = ds.attrs.get(_PERSISTED_PREVIEW_CACHE_ATTR)
        if not isinstance(encoded, str) or not encoded:
            return
        try:
            png_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (binascii.Error, ValueError):
            return
        if len(png_bytes) > _PERSISTED_PREVIEW_CACHE_MAX_BYTES:
            return
        preview = QtGui.QPixmap()
        if not preview.loadFromData(png_bytes, "PNG"):
            return
        self._preview_pixmap_cache = preview
        self._preview_pixmap_generation += 1
        self._preview_thumbnail_cache.clear()
        self._preview_pixmap_stale = bool(
            ds.attrs.get(_PERSISTED_PREVIEW_CACHE_STALE_ATTR, False)
        )

    def _restore_persistence_payload(self, ds: xr.Dataset) -> None:
        self._restore_persisted_preview_cache(ds)

    @property
    def preview_pixmap(self) -> QtGui.QPixmap | None:
        return self._preview_pixmap_cache

    @property
    def preview_pixmap_generation(self) -> int:
        return self._preview_pixmap_generation

    @property
    def preview_pixmap_stale(self) -> bool:
        return self._preview_pixmap_stale

    def _clear_preview_pixmap_cache(self, *, stale: bool) -> None:
        self._preview_pixmap_cache = None
        self._preview_pixmap_generation += 1
        self._preview_thumbnail_cache.clear()
        self._preview_pixmap_stale = stale

    def _mark_preview_pixmap_stale(self) -> None:
        self._preview_pixmap_stale = True
        self._preview_thumbnail_cache.clear()

    def request_preview_pixmap_update(
        self, *, delay_ms: int = _PREVIEW_PIXMAP_UPDATE_DELAY_MS
    ) -> None:
        if self._closing:
            return
        if not self._document.recipe.operations:
            self._clear_preview_pixmap_cache(stale=False)
            return
        if self._preview_pixmap_update_pending or not self._preview_pixmap_stale:
            return
        self._preview_pixmap_update_pending = True
        self._preview_pixmap_update_generation += 1
        generation = self._preview_pixmap_update_generation
        erlab.interactive.utils.single_shot(
            self,
            delay_ms,
            functools.partial(self._run_queued_preview_pixmap_update, generation),
        )

    def _run_queued_preview_pixmap_update(self, generation: int) -> None:
        if (
            generation != self._preview_pixmap_update_generation
            or self._closing
            or not erlab.interactive.utils.qt_is_valid(self)
        ):
            return
        self._preview_pixmap_update_pending = False
        if self._rendering:
            self.request_preview_pixmap_update()
            return
        previous_generation = self._preview_pixmap_generation
        previous_stale = self._preview_pixmap_stale
        self.refresh_preview_pixmap()
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        if (
            self._preview_pixmap_generation != previous_generation
            or self._preview_pixmap_stale != previous_stale
        ):
            self.sigInfoChanged.emit()

    def _cache_live_canvas_preview(self, *, redraw: bool) -> bool:
        if (
            self._closing
            or not erlab.interactive.utils.qt_is_valid(self)
            or not self._document.recipe.operations
        ):
            return False
        window = self._figure_window
        if window is None or not erlab.interactive.utils.qt_is_valid(window):
            return False
        if not window.isVisible():
            return False

        try:
            canvas = window.canvas
            if not erlab.interactive.utils.qt_is_valid(canvas):
                return False
            if redraw:
                with self._figure_options_context(), _figure_style_context():
                    canvas.draw()
            width, height = canvas.get_width_height(physical=True)
            if width <= 0 or height <= 0:
                return False
            image = QtGui.QImage(
                canvas.buffer_rgba(),
                width,
                height,
                QtGui.QImage.Format.Format_RGBA8888,
            )
            preview = QtGui.QPixmap.fromImage(image.copy())
        except Exception:
            return False
        if preview.isNull():
            return False
        self._preview_pixmap_cache = preview
        self._preview_pixmap_generation += 1
        self._preview_thumbnail_cache.clear()
        self._preview_pixmap_stale = False
        return True

    def _fallback_preview_pixmap(self) -> QtGui.QPixmap | None:
        if self._closing or not erlab.interactive.utils.qt_is_valid(self):
            return None
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        with self._figure_options_context(), _figure_style_context():
            figure = Figure(
                figsize=self._document.recipe.setup.figsize,
                dpi=self._document.recipe.setup.dpi,
                layout=typing.cast("typing.Any", self._document.recipe.setup.layout),
            )
            try:
                canvas = FigureCanvasAgg(figure)
                try:
                    _render_into_figure(self, figure, sync_visible=False)
                    with _figure_draw_context():
                        canvas.draw()
                    width, height = canvas.get_width_height()
                except Exception:
                    return None
                if width <= 0 or height <= 0:
                    return None
                image = QtGui.QImage(
                    canvas.buffer_rgba(),
                    width,
                    height,
                    QtGui.QImage.Format.Format_RGBA8888,
                )
                return QtGui.QPixmap.fromImage(image.copy())
            finally:
                figure.clear()

    def refresh_preview_pixmap(
        self, *, allow_offscreen: bool = False
    ) -> QtGui.QPixmap | None:
        if self._closing or not erlab.interactive.utils.qt_is_valid(self):
            self._clear_preview_pixmap_cache(stale=False)
            return None
        if not self._document.recipe.operations:
            self._clear_preview_pixmap_cache(stale=False)
            return None
        if self._cache_live_canvas_preview(redraw=True):
            return self._preview_pixmap_cache
        preview = self._fallback_preview_pixmap() if allow_offscreen else None
        if preview is None:
            return self._preview_pixmap_cache
        self._preview_pixmap_cache = preview
        self._preview_pixmap_generation += 1
        self._preview_thumbnail_cache.clear()
        self._preview_pixmap_stale = False
        return preview

    def _preview_thumbnail_pixmap(self, size: QtCore.QSize) -> QtGui.QPixmap | None:
        if self._closing or not erlab.interactive.utils.qt_is_valid(self):
            return None
        preview = self._preview_pixmap_cache
        if preview is None or preview.isNull() or not size.isValid() or size.isEmpty():
            return None
        cache_key = (size.width(), size.height())
        cached = self._preview_thumbnail_cache.get(cache_key)
        if cached is not None and cached[0] == self._preview_pixmap_generation:
            return QtGui.QPixmap(cached[1])
        thumbnail = preview.scaled(
            size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_thumbnail_cache[cache_key] = (
            self._preview_pixmap_generation,
            thumbnail,
        )
        return QtGui.QPixmap(thumbnail)

    def source_states(self) -> tuple[FigureSourceState, ...]:
        return self._document.recipe.sources

    def source_data(self) -> dict[str, xr.DataArray]:
        """Return live sources; call ``touch_source_data`` after in-place edits."""
        return dict(self._document.source_data)

    @staticmethod
    def _source_selection_replay_operations(
        source: FigureSourceState,
    ) -> tuple[ToolProvenanceOperation, ...]:
        operations: list[ToolProvenanceOperation] = []
        if source.isel:
            operations.append(
                IselOperation(
                    kwargs=typing.cast(
                        "dict[Hashable, typing.Any]",
                        dict(source.isel),
                    )
                )
            )
        if source.qsel:
            operations.append(
                QSelOperation(
                    kwargs=typing.cast(
                        "dict[Hashable, typing.Any]",
                        dict(source.qsel),
                    )
                )
            )
        if source.mean_dims:
            operations.append(
                QSelAggregationOperation(
                    dims=tuple(source.mean_dims),
                    func="mean",
                )
            )
        return tuple(operations)

    @staticmethod
    def _source_code_name_candidate(text: str) -> str | None:
        text = re.sub(r"^\s*ImageTool\s+\d+\s*:\s*", "", text.strip())
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()
        text = re.sub(r"_+", "_", text)
        if text.startswith("data_"):
            text = f"source_{text.removeprefix('data_')}"
        if not text:
            return None
        if text[0].isdigit():
            text = f"source_{text}"
        if not text.isidentifier() or keyword.iskeyword(text):
            return None
        return text

    def _source_display_code_name(
        self,
        source: FigureSourceState,
        *,
        used_names: set[str],
    ) -> str:
        candidates: list[str] = []
        data = self._document.source_data.get(source.name)
        if data is not None and isinstance(data.name, str):
            candidates.append(data.name)
        candidates.append(source.name)
        for candidate in candidates:
            name = self._source_code_name_candidate(candidate)
            if name is not None:
                break
        else:
            name = "source"

        return _source_unique_name(name, used_names)

    @staticmethod
    def _script_input_with_name(
        script_input: ScriptInput,
        name: str,
    ) -> ScriptInput:
        if script_input.name == name:
            return script_input
        updates = {"name": name}
        if script_input.label == script_input.name:
            updates["label"] = name
        return script_input.model_copy(update=updates)

    def _selected_source_script_input(
        self,
        source: FigureSourceState,
        *,
        display_name: str,
        source_by_name: Mapping[str, FigureSourceState],
    ) -> ScriptInput | None:
        lineage: list[FigureSourceState] = []
        seen: set[str] = set()
        current = source
        while True:
            if current.name in seen:
                return None
            seen.add(current.name)
            lineage.append(current)
            parent_name = current.selection_source
            if parent_name is None or parent_name == current.name:
                break
            parent = source_by_name.get(parent_name)
            if parent is None:
                break
            current = parent

        base_input: ScriptInput | None = None
        base_index = -1
        try:
            for candidate_index in range(len(lineage) - 1, -1, -1):
                candidate_input = lineage[candidate_index].to_script_input()
                if candidate_input is None:
                    continue
                candidate_spec = candidate_input.parsed_provenance_spec()
                if candidate_spec is None:
                    continue
                if to_replay_provenance_spec(candidate_spec) is None:
                    continue
                base_input = candidate_input
                base_index = candidate_index
                break
            if base_input is None:
                for candidate_index in range(len(lineage) - 1, -1, -1):
                    candidate_input = lineage[candidate_index].to_script_input()
                    if candidate_input is None or not candidate_input.node_uid:
                        continue
                    base_input = candidate_input
                    base_index = candidate_index
                    break
            if base_input is None:
                return None
            source_label = " ".join(source.label.split())
            selected_spec = script(
                start_label=f"Select data for {source_label or source.name}",
                seed_code=(
                    None
                    if base_input.name == display_name
                    else f"{display_name} = {base_input.name}"
                ),
                active_name=display_name,
                script_inputs=(base_input,),
            )
            for selected_source in reversed(lineage[: base_index + 1]):
                operations = self._source_selection_replay_operations(selected_source)
                if operations:
                    selected_spec = selected_spec.append_replay_stage(
                        public_data(*operations)
                    )
        except (TypeError, ValueError, pydantic.ValidationError):
            return None
        return ScriptInput(
            name=display_name,
            label=(
                source_label
                if source_label and source_label != source.name
                else display_name
            ),
            provenance_spec=selected_spec.model_dump(mode="json"),
        )

    def _display_code_source_plan(
        self,
    ) -> tuple[
        tuple[ScriptInput, ...],
        frozenset[str],
        dict[str, str],
    ]:
        source_by_name = self._document.source_by_name()
        used_sources = self._document.direct_sources_used_by_recipe(
            enabled_only=True, executable_only=True
        )
        used_code_names = set(_FIGURE_CODE_RESERVED_NAMES)
        used_code_names.update(
            bound_name
            for operation in self._document.recipe.operations
            if operation.enabled
            and operation.kind == FigureOperationKind.CUSTOM
            and operation.trusted
            for bound_name in _custom_code_bound_names(operation.code)
        )
        if self._document.recipe.setup.layout_mode == "gridspec":
            source_names = self._document.source_names()
            used_code_names.update(
                _gridspec_reserved_axis_code_names(
                    self._document.recipe.setup, reserved_names=source_names
                )
            )
            used_code_names.update(
                _gridspec_axis_code_names(
                    self._document.recipe.setup, reserved_names=source_names
                ).values()
            )
        script_inputs: list[ScriptInput] = []
        script_input_names: set[str] = set()
        skip_source_selection_names: set[str] = set()
        source_name_map: dict[str, str] = {}

        def append_script_input(
            script_input: ScriptInput | None,
        ) -> None:
            if script_input is None or script_input.name in script_input_names:
                return
            script_inputs.append(script_input)
            script_input_names.add(script_input.name)

        for source in self._document.recipe.sources:
            if source.name not in used_sources:
                continue
            display_name = self._source_display_code_name(
                source,
                used_names=used_code_names,
            )
            script_input: ScriptInput | None = None
            if _source_has_selection(source):
                script_input = self._selected_source_script_input(
                    source,
                    display_name=display_name,
                    source_by_name=source_by_name,
                )
                if script_input is not None:
                    skip_source_selection_names.add(source.name)
                    source_name_map[source.name] = display_name
                else:
                    used_code_names.discard(display_name)
                    base_source = source_by_name.get(
                        source.selection_source or source.name
                    )
                    append_script_input(
                        None if base_source is None else base_source.to_script_input()
                    )
            else:
                script_input = source.to_script_input()
                if script_input is not None:
                    script_input = self._script_input_with_name(
                        script_input,
                        display_name,
                    )
                    source_name_map[source.name] = display_name
            append_script_input(script_input)

        return (
            tuple(script_inputs),
            frozenset(skip_source_selection_names),
            source_name_map,
        )

    def current_provenance_spec(
        self, *, flush_deferred_restore: bool = True
    ) -> ToolProvenanceSpec | None:
        del flush_deferred_restore
        script_inputs, skip_source_selection_names, source_name_map = (
            self._display_code_source_plan()
        )
        if not script_inputs:
            return None
        return script(
            erlab.interactive._figurecomposer._provenance._figure_build_operation(
                self,
                skip_source_selection_names=skip_source_selection_names,
                source_name_map=source_name_map,
            ),
            start_label="Figure",
            active_name="fig",
            script_inputs=script_inputs,
        )

    def set_missing_sources(self, names: set[str]) -> None:
        if not self._document.discard_source_data(names):
            return
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()

    def refresh_from_sources(self, source_data: Mapping[str, xr.DataArray]) -> None:
        result = self._document.refresh_sources(source_data)
        if result.skipped:
            self._set_source_panel_status(
                "Could not refresh source data for: "
                + ", ".join(f"{name} ({detail})" for name, detail in result.skipped)
            )
        elif result:
            self._set_source_panel_status(None)
        if not result:
            self._refresh_source_controls()
            return
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
