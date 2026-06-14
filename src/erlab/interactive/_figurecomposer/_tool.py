"""Manager-facing Figure Composer tool window."""

from __future__ import annotations

import collections
import contextlib
import functools
import json
import math
import textwrap
import typing
import uuid
import weakref

# Matplotlib's Qt backend should see the qtpy-selected binding first.
# isort: off
from qtpy import QtCore, QtGui, QtWidgets

from matplotlib.figure import Figure
# isort: on

import erlab
import erlab.interactive._figurecomposer._codegen
import erlab.interactive._figurecomposer._provenance
import erlab.interactive._figurecomposer._toolbar_dialogs
from erlab.interactive._figurecomposer._axes import _all_axes
from erlab.interactive._figurecomposer._defaults import (
    _MM_PER_INCH,
    _figure_draw_context,
    _figure_style_context,
)
from erlab.interactive._figurecomposer._editor_controls import (
    MIXED_VALUE as _MIXED_VALUE,
)
from erlab.interactive._figurecomposer._editor_controls import (
    MIXED_VALUES_TEXT as _MIXED_VALUES_TEXT,
)
from erlab.interactive._figurecomposer._editor_controls import (
    CheckBoxControlAdapter,
    ComboBoxControlAdapter,
    ComboBoxDataControlAdapter,
    LineEditControlAdapter,
    PlainTextControlAdapter,
    SignalValueControlAdapter,
)
from erlab.interactive._figurecomposer._gridspec import (
    _gridspec_all_axes_ids,
    _gridspec_axes_subplot_targets,
    _gridspec_axis_code_names,
    _gridspec_axis_display_name,
    _gridspec_axis_display_names,
    _gridspec_axis_variable_name_error,
    _gridspec_grid_by_id,
    _gridspec_grid_display_name,
    _gridspec_grid_display_names,
    _gridspec_grid_path,
    _gridspec_has_invalid_regions,
    _gridspec_invalid_axes_ids,
    _gridspec_region_label,
    _gridspec_region_overlaps,
    _gridspec_region_valid,
    _gridspec_remove_region,
    _gridspec_replace_grid,
    _gridspec_setup_from_subplots,
    _gridspec_update_axis_variable_name,
    _gridspec_valid_axes_ids,
    _subplots_setup_from_gridspec,
)
from erlab.interactive._figurecomposer._operations import _registry
from erlab.interactive._figurecomposer._operations._base import (
    COMMON_AXES_SECTION_TOOLTIP,
    COMMON_SOURCE_SECTION_TOOLTIP,
    StepSection,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    _PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
    _plot_slices_panel_keys,
)
from erlab.interactive._figurecomposer._rendering import (
    _live_layout_axes,
    _render_into_figure,
    _render_preview,
    _rendered_output_figure,
)
from erlab.interactive._figurecomposer._sources import (
    _default_plot_operation,
    _default_setup_for_data,
    _public_source_data,
    _source_display_label,
    _source_display_tooltip,
    _source_duplicate_labels,
    _source_label,
    _source_name,
)
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureGridSpecAxesState,
    FigureGridSpecGridState,
    FigureGridSpecSpanState,
    FigureMethodFamily,
    FigureOperationKind,
    FigureOperationState,
    FigurePlotSlicesPanelStyleState,
    FigureRecipeState,
    FigureSourceState,
    FigureSubplotsState,
)
from erlab.interactive._figurecomposer._text import (
    FigureComposerInputError,
    _format_axes_tuple,
    _format_tuple,
    _literal_sequence_from_text,
)
from erlab.interactive._figurecomposer._widgets import (
    _AxesSelectorWidget,
    _FigureComposerDisplayWindow,
    _GridSpecRegionInfo,
    _GridSpecViewWidget,
    _step_toolbar_button,
)
from erlab.interactive.imagetool import provenance

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import xarray as xr
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


_OPERATION_EDITOR_UPDATE_DELAY_MS = 25
_RETIRED_EDITOR_DRAIN_DELAY_MS = 100
_PREVIEW_RENDER_UPDATE_DELAY_MS = 50
_FIGURE_RESIZE_RENDER_DELAY_MS = 120
_PREVIEW_PIXMAP_UPDATE_DELAY_MS = 250
_COMBO_POPUP_REBUILD_GRACE_MS = 150
_COMBO_INTERACTION_REBUILD_GRACE_MS = 250
_COMBO_TRACKED_PROPERTY = "figure_composer_combo_tracked"
_COMBO_POPUP_GUARD_ID_PROPERTY = "figure_composer_combo_popup_guard_id"
_STEPS_CLIPBOARD_MIME = "application/x-erlab-figure-composer-steps+json"
_STEPS_CLIPBOARD_PAYLOAD_TYPE = "erlab.figure_composer.steps"
_STEPS_CLIPBOARD_PAYLOAD_VERSION = 1


class _FigureComposerStepMimeData(QtCore.QMimeData):
    """Clipboard payload that can also carry live source data in one process."""

    def __init__(
        self,
        payload_text: str,
        step_code_text: str,
        source_data: Mapping[str, xr.DataArray],
        *,
        cut_source_tool_id: str | None = None,
    ) -> None:
        super().__init__()
        self.figure_composer_source_data: dict[str, xr.DataArray] = dict(source_data)
        self.figure_composer_cut_source_tool_id = cut_source_tool_id
        self.setData(_STEPS_CLIPBOARD_MIME, payload_text.encode("utf-8"))
        self.setText(step_code_text)


class _FigureComposerOperationList(QtWidgets.QListWidget):
    copy_requested = QtCore.Signal()
    cut_requested = QtCore.Signal()
    paste_requested = QtCore.Signal()
    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_requested.emit()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Cut):
            self.cut_requested.emit()
            event.accept()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            self.paste_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


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
    return operations, sources, dict(source_data)


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
        self._updating_controls = False
        self._rendering = False
        self._operation_editor_update_pending = False
        self._preview_render_update_pending = False
        self._preview_render_update_generation = 0
        self._step_tab_order_update_pending = False
        self._retired_editor_drain_pending = False
        self._combo_popup_guard_tokens: set[int] = set()
        self._next_combo_popup_guard_token = 0
        self._tracked_combo_refs: list[weakref.ReferenceType[QtWidgets.QComboBox]] = []
        self._operation_multi_select_event = False
        self._operation_list_viewport: QtWidgets.QWidget | None = None
        self._retired_editor_widgets: list[QtWidgets.QWidget] = []
        self._operation_render_errors: dict[str, str] = {}
        self._operation_input_errors: dict[str, dict[str, str]] = {}
        self._operation_editor_generation = 0
        self._active_editor_signal_widget: QtWidgets.QWidget | None = None
        self._figure_resize_render_generation = 0
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
        self._section_tab_stop_refs: dict[
            str, weakref.ReferenceType[QtWidgets.QWidget]
        ] = {}
        self._source_data: dict[str, xr.DataArray] = {}
        self._recipe = recipe or self._default_recipe(data)
        self._active_gridspec_grid_id = self._recipe.setup.gridspec.root.grid_id
        self._gridspec_breadcrumb_buttons: list[QtWidgets.QToolButton] = []
        self._figure_window: _FigureComposerDisplayWindow | None = None
        self._subplot_adjust_dialog: QtWidgets.QDialog | None = None
        self._axes_customize_dialog: QtWidgets.QDialog | None = None
        self._operation_context_menu: QtWidgets.QMenu | None = None
        self._connected_step_clipboard: QtGui.QClipboard | None = None
        self._step_clipboard_tool_id = uuid.uuid4().hex
        self._prev_source_data_states: collections.deque[dict[str, xr.DataArray]] = (
            collections.deque(maxlen=self._prev_states.maxlen)
        )
        self._next_source_data_states: collections.deque[dict[str, xr.DataArray]] = (
            collections.deque(maxlen=self._next_states.maxlen)
        )

        if source_data is not None:
            self.set_source_data(source_data)
        elif self._recipe.primary_source in {
            source.name for source in self._recipe.sources
        }:
            self._source_data[self._recipe.primary_source] = data
        else:
            source_name = _source_name(data)
            self._source_data[source_name] = data

        if self._recipe.primary_source not in self._source_data:
            self._source_data[self._recipe.primary_source] = data

        self._current_step_section_key = "sources"
        self._build_ui()
        self._apply_recipe_to_controls()
        self._write_state()

    @staticmethod
    def _default_recipe(data: xr.DataArray) -> FigureRecipeState:
        source_name = _source_name(data)
        setup = _default_setup_for_data(data)
        source = FigureSourceState(name=source_name, label=_source_label(data))
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
                self._recipe.setup,
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
        self.figure_window.show_for_setup(
            self._recipe.setup, self._figure_window_title(), activate=activate
        )
        self._cancel_preview_render_update()
        _render_preview(self, show_window=True)

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
            width_inches, height_inches, draw=False, emit_info=False
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

        operations = list(self._recipe.operations)
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
        operations = list(self._recipe.operations)
        changed_operation_ids: set[str] = set()
        changed = False
        for mappable, clim in changes.items():
            target = self._plot_slices_mappable_target(mappable, operations)
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
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._refresh_operation_list()
        if current is not None and current[0] < len(operations):
            self._set_current_operation_row_silent(current[0])
        current_operation = self._current_operation()
        self._sync_axes_selector()
        self._update_step_action_buttons()
        self._refresh_step_section_button_texts()
        self._update_source_status(current_operation[1] if current_operation else None)
        if current_operation is not None and (
            current_operation[1].operation_id in changed_operation_ids
        ):
            self._update_operation_editor_safely()
        self._mark_preview_pixmap_stale()
        self.sigInfoChanged.emit()
        self._write_state()

    def _plot_slices_mappable_target(
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
            if (
                operation.operation_id == operation_id
                and operation.kind == FigureOperationKind.PLOT_SLICES
            ):
                return index, operation, typing.cast("tuple[int, int]", panel_key)
        return None

    def _operation_with_colorbar_clim(
        self,
        operation: FigureOperationState,
        panel_key: tuple[int, int],
        clim: tuple[float, float],
    ) -> FigureOperationState:
        panel_keys = _plot_slices_panel_keys(self, operation)
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

    def _set_recipe_figsize_from_canvas(
        self,
        width_inches: float,
        height_inches: float,
        *,
        draw: bool,
        emit_info: bool,
    ) -> bool:
        setup = self._recipe.setup
        if math.isclose(width_inches, setup.figsize[0], abs_tol=0.005) and math.isclose(
            height_inches, setup.figsize[1], abs_tol=0.005
        ):
            return False
        figsize = (round(width_inches, 4), round(height_inches, 4))
        self._recipe = self._recipe.model_copy(
            update={"setup": setup.model_copy(update={"figsize": figsize})}
        )
        self._mark_preview_pixmap_stale()
        self.figure.set_size_inches(figsize, forward=False)
        self._updating_controls = True
        try:
            self.width_spin.setValue(figsize[0])
            self.height_spin.setValue(figsize[1])
            self._sync_size_mm_controls(figsize)
        finally:
            self._updating_controls = False
        if draw:
            self.canvas.draw()
        if emit_info:
            self.sigInfoChanged.emit()
        self._write_state()
        return True

    def _show_subplot_adjust_dialog(self) -> None:
        erlab.interactive._figurecomposer._toolbar_dialogs.show_subplot_adjust_dialog(
            self
        )

    def _show_axes_customize_dialog(self) -> None:
        erlab.interactive._figurecomposer._toolbar_dialogs.show_axes_customize_dialog(
            self
        )

    def _hide_figure_window(self) -> None:
        if self._figure_window is not None and erlab.interactive.utils.qt_is_valid(
            self._figure_window
        ):
            self._figure_window.hide()

    def _close_figure_window(self) -> None:
        erlab.interactive._figurecomposer._toolbar_dialogs.close_toolbar_dialogs(self)
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
        self._request_show_figure_window(activate=False)

    def hideEvent(self, event: QtGui.QHideEvent | None) -> None:
        self._cancel_queued_show_figure_window()
        self._hide_figure_window()
        if event is not None:
            super().hideEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._closing = True
        self._cancel_queued_show_figure_window()
        self._figure_resize_render_generation += 1
        self._preview_render_update_generation += 1
        self._preview_render_update_pending = False
        self._preview_pixmap_update_generation += 1
        self._preview_pixmap_update_pending = False
        self._clear_preview_pixmap_cache(stale=False)
        self._remove_operation_list_event_filter()
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

        recipe_page = QtWidgets.QWidget(self.editor_tabs)
        recipe_page.setObjectName("figureComposerRecipePage")
        self.recipe_page = recipe_page
        recipe_layout = QtWidgets.QVBoxLayout(recipe_page)
        recipe_layout.setContentsMargins(6, 6, 6, 6)
        recipe_layout.setSpacing(4)

        action_layout = QtWidgets.QVBoxLayout()
        action_layout.setSpacing(2)

        self.add_step_button = _step_toolbar_button(
            recipe_page,
            "figureComposerAddStepButton",
            "Add Step ▾",
            "Add a plotting, ERLab method, Axes method, or Python step.",
        )
        self.add_step_button.setProperty("uses_inline_menu_arrow", True)
        self.add_step_menu = QtWidgets.QMenu(self.add_step_button)
        self.add_step_menu.setObjectName("figureComposerAddStepMenu")
        for action_spec in _registry.add_step_actions():
            action = QtWidgets.QAction(action_spec.text, self.add_step_menu)
            action.setData(action_spec.action_id)
            action.setToolTip(action_spec.tooltip)
            action.triggered.connect(
                lambda _checked=False, action_id=action_spec.action_id: (
                    self._add_operation(action_id)
                )
            )
            self.add_step_menu.addAction(action)
        self.add_step_button.clicked.connect(self._show_add_step_menu)
        self.remove_operation_button = _step_toolbar_button(
            recipe_page,
            "figureComposerDeleteStepButton",
            "Delete",
            "Remove the selected recipe step or steps.",
        )
        self.remove_operation_button.clicked.connect(self._remove_current_operation)
        self.duplicate_operation_button = _step_toolbar_button(
            recipe_page,
            "figureComposerDuplicateStepButton",
            "Duplicate",
            "Copy the selected step or steps after the last selected step.",
        )
        self.duplicate_operation_button.clicked.connect(
            self._duplicate_current_operation
        )
        self.copy_operation_button = _step_toolbar_button(
            recipe_page,
            "figureComposerCopyStepButton",
            "Copy",
            "Copy the selected recipe step or steps.",
        )
        self.copy_operation_button.clicked.connect(self._copy_selected_operations)
        self.cut_operation_button = _step_toolbar_button(
            recipe_page,
            "figureComposerCutStepButton",
            "Cut",
            "Cut the selected recipe step or steps.",
        )
        self.cut_operation_button.clicked.connect(self._cut_selected_operations)
        self.paste_operation_button = _step_toolbar_button(
            recipe_page,
            "figureComposerPasteStepButton",
            "Paste",
            "Paste copied recipe steps after the current selection.",
        )
        self.paste_operation_button.clicked.connect(
            self._paste_operations_from_clipboard
        )
        self.move_operation_up_button = _step_toolbar_button(
            recipe_page,
            "figureComposerMoveStepUpButton",
            "Up",
            "Move the selected recipe step or steps earlier.",
        )
        self.move_operation_up_button.clicked.connect(self._move_current_operation_up)
        self.move_operation_down_button = _step_toolbar_button(
            recipe_page,
            "figureComposerMoveStepDownButton",
            "Down",
            "Move the selected recipe step or steps later.",
        )
        self.move_operation_down_button.clicked.connect(
            self._move_current_operation_down
        )
        self.show_figure_button = QtWidgets.QPushButton("Show Plot Window", root)
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
        action_layout.addLayout(output_action_layout)
        root_layout.addLayout(action_layout)
        root_layout.addWidget(self.editor_tabs, 1)

        selected_step_action_layout = QtWidgets.QHBoxLayout()
        selected_step_action_layout.setSpacing(4)
        selected_step_action_layout.addWidget(self.add_step_button)
        selected_step_action_layout.addWidget(self.copy_operation_button)
        selected_step_action_layout.addWidget(self.cut_operation_button)
        selected_step_action_layout.addWidget(self.paste_operation_button)
        selected_step_action_layout.addWidget(self.duplicate_operation_button)
        selected_step_action_layout.addWidget(self.move_operation_up_button)
        selected_step_action_layout.addWidget(self.move_operation_down_button)
        selected_step_action_layout.addWidget(self.remove_operation_button)
        selected_step_action_layout.addStretch(1)
        recipe_layout.addLayout(selected_step_action_layout)

        self.recipe_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.recipe_splitter.setObjectName("figureComposerRecipeSplitter")
        self.recipe_splitter.setChildrenCollapsible(False)
        recipe_layout.addWidget(self.recipe_splitter, 1)

        self.operation_list = _FigureComposerOperationList(recipe_page)
        self.operation_list.setObjectName("figureComposerOperationList")
        self.operation_list.copy_requested.connect(self._copy_selected_operations)
        self.operation_list.cut_requested.connect(self._cut_selected_operations)
        self.operation_list.paste_requested.connect(
            self._paste_operations_from_clipboard
        )
        self.operation_list.context_menu_requested.connect(
            self._show_operation_context_menu
        )
        self._operation_list_viewport = self.operation_list.viewport()
        if self._operation_list_viewport is not None:
            self._operation_list_viewport.installEventFilter(self)
        self.operation_list.currentRowChanged.connect(self._operation_selection_changed)
        self.operation_list.itemSelectionChanged.connect(
            self._operation_selection_changed
        )
        self.operation_list.itemChanged.connect(self._operation_item_changed)
        self.operation_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.operation_list.setMinimumHeight(72)
        self.operation_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.operation_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.operation_list.setToolTip(
            "Checked steps run from top to bottom to build the figure."
        )
        self.recipe_splitter.addWidget(self.operation_list)

        self.step_inspector = QtWidgets.QWidget(recipe_page)
        self.step_inspector.setObjectName("figureComposerStepInspector")
        step_inspector_layout = QtWidgets.QHBoxLayout(self.step_inspector)
        step_inspector_layout.setContentsMargins(0, 0, 0, 0)
        step_inspector_layout.setSpacing(6)
        self.recipe_splitter.addWidget(self.step_inspector)
        self.recipe_splitter.setStretchFactor(0, 0)
        self.recipe_splitter.setStretchFactor(1, 1)
        self.recipe_splitter.setSizes((130, 420))

        self.step_navigator = QtWidgets.QWidget(self.step_inspector)
        self.step_navigator.setObjectName("figureComposerStepNavigator")
        self.step_navigator.setFixedWidth(150)
        self.step_navigator_layout = QtWidgets.QVBoxLayout(self.step_navigator)
        self.step_navigator_layout.setContentsMargins(0, 0, 0, 0)
        self.step_navigator_layout.setSpacing(3)
        self.step_section_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.step_section_keys: list[str] = []
        step_inspector_layout.addWidget(self.step_navigator)

        self.step_editor_stack = QtWidgets.QStackedWidget(self.step_inspector)
        self.step_editor_stack.setObjectName("figureComposerStepSectionStack")
        step_inspector_layout.addWidget(self.step_editor_stack, 1)
        self._operation_editor_pages: list[QtWidgets.QWidget] = []

        self.step_sources_page = QtWidgets.QWidget(self.step_editor_stack)
        self.step_sources_page.setObjectName("figureComposerStepSourcesPage")
        sources_layout = QtWidgets.QVBoxLayout(self.step_sources_page)
        sources_layout.setContentsMargins(6, 6, 6, 6)
        sources_layout.setSpacing(4)
        self.step_source_controls = QtWidgets.QWidget(self.step_sources_page)
        self.step_source_controls_layout = QtWidgets.QFormLayout(
            self.step_source_controls
        )
        self.step_source_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.step_source_controls_layout.setSpacing(4)
        self.step_source_controls_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        sources_layout.addWidget(self.step_source_controls)
        self.source_status_label = QtWidgets.QLabel(self.step_sources_page)
        self.source_status_label.setObjectName("figureComposerSourceStatus")
        self.source_status_label.setWordWrap(True)
        sources_layout.addWidget(self.source_status_label)
        self.source_list = QtWidgets.QTreeWidget(self.step_sources_page)
        self.source_list.setObjectName("figureComposerSourceList")
        self.source_list.setColumnCount(2)
        self.source_list.setHeaderLabels(("Source", "Shape"))
        self.source_list.setRootIsDecorated(False)
        self.source_list.setIndentation(0)
        self.source_list.setUniformRowHeights(True)
        self.source_list.setAlternatingRowColors(True)
        self.source_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.source_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.source_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.source_list.setToolTip("Data available to the selected step.")
        source_header = self.source_list.header()
        if source_header is not None:
            source_header.setStretchLastSection(False)
            source_header.setSectionResizeMode(
                0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            source_header.setSectionResizeMode(
                1, QtWidgets.QHeaderView.ResizeMode.Stretch
            )
        sources_layout.addWidget(self.source_list, 1)

        self.target_axes_page = QtWidgets.QWidget(self.step_editor_stack)
        self.target_axes_page.setObjectName("figureComposerTargetAxesPage")
        target_axes_layout = QtWidgets.QVBoxLayout(self.target_axes_page)
        target_axes_layout.setContentsMargins(6, 6, 6, 6)
        target_axes_layout.setSpacing(4)
        self.axes_selector = _AxesSelectorWidget(self.target_axes_page)
        self.axes_selector.sigSelectionChanged.connect(self._axes_selection_changed)
        self.axes_selector.sigAddRowRequested.connect(self._add_subplot_row)
        self.axes_selector.sigAddColumnRequested.connect(self._add_subplot_column)
        target_axes_layout.addWidget(self.axes_selector)
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
        self.operation_editor = QtWidgets.QWidget(self.step_editor_stack)
        self.operation_editor_layout = QtWidgets.QFormLayout(self.operation_editor)

        layout_page = QtWidgets.QWidget(self.editor_tabs)
        layout_page.setObjectName("figureComposerLayoutPage")
        self.layout_page = layout_page
        setup_layout = QtWidgets.QGridLayout(layout_page)
        setup_layout.setContentsMargins(6, 6, 6, 6)
        setup_layout.setHorizontalSpacing(8)
        setup_layout.setVerticalSpacing(6)
        setup_layout.setColumnStretch(2, 1)
        setup_layout.setColumnStretch(4, 1)
        self.nrows_spin = QtWidgets.QSpinBox(layout_page)
        self.nrows_spin.setRange(1, 12)
        self.ncols_spin = QtWidgets.QSpinBox(layout_page)
        self.ncols_spin.setRange(1, 12)
        self.width_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.width_spin.setRange(0.25, 100.0)
        self.width_spin.setDecimals(3)
        self.width_spin.setSingleStep(0.25)
        self.height_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.height_spin.setRange(0.25, 100.0)
        self.height_spin.setDecimals(3)
        self.height_spin.setSingleStep(0.25)
        self.width_mm_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.width_mm_spin.setRange(0.25 * _MM_PER_INCH, 100.0 * _MM_PER_INCH)
        self.width_mm_spin.setDecimals(2)
        self.width_mm_spin.setSingleStep(1.0)
        self.height_mm_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.height_mm_spin.setRange(0.25 * _MM_PER_INCH, 100.0 * _MM_PER_INCH)
        self.height_mm_spin.setDecimals(2)
        self.height_mm_spin.setSingleStep(1.0)
        self.dpi_spin = QtWidgets.QDoubleSpinBox(layout_page)
        self.dpi_spin.setObjectName("figureComposerDpiSpin")
        self.dpi_spin.setRange(1.0, 10000.0)
        self.dpi_spin.setDecimals(1)
        self.dpi_spin.setSingleStep(10.0)
        self.layout_combo = QtWidgets.QComboBox(layout_page)
        self.layout_combo.addItems(
            ["default", "constrained", "compressed", "tight", "none"]
        )
        self.sharex_combo = QtWidgets.QComboBox(layout_page)
        self.sharex_combo.addItems(["False", "True", "row", "col", "all"])
        self.sharey_combo = QtWidgets.QComboBox(layout_page)
        self.sharey_combo.addItems(["False", "True", "row", "col", "all"])
        self.width_ratios_edit = QtWidgets.QLineEdit(layout_page)
        self.width_ratios_edit.setObjectName("figureComposerWidthRatiosEdit")
        self.height_ratios_edit = QtWidgets.QLineEdit(layout_page)
        self.height_ratios_edit.setObjectName("figureComposerHeightRatiosEdit")
        self.layout_mode_combo = QtWidgets.QComboBox(layout_page)
        self.layout_mode_combo.setObjectName("figureComposerLayoutModeCombo")
        self.layout_mode_combo.addItems(["subplots", "gridspec"])
        self.layout_mode_combo.setToolTip(
            "Choose regular plt.subplots layout or custom GridSpec regions."
        )

        def add_grid_pair_row(
            row: int,
            row_label_text: str,
            row_object_name: str,
            row_tooltip: str,
            first_label: str,
            first_widget: QtWidgets.QWidget,
            first_tooltip: str,
            second_label: str,
            second_widget: QtWidgets.QWidget,
            second_tooltip: str,
        ) -> None:
            row_label = QtWidgets.QLabel(row_label_text, layout_page)
            row_label.setObjectName(row_object_name)
            row_label.setToolTip(row_tooltip)
            row_label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            setup_layout.addWidget(row_label, row, 0)
            for column, label_text, widget, tooltip in (
                (1, first_label, first_widget, first_tooltip),
                (3, second_label, second_widget, second_tooltip),
            ):
                label = QtWidgets.QLabel(label_text, layout_page)
                label.setBuddy(widget)
                label.setToolTip(tooltip)
                widget.setToolTip(tooltip)
                setup_layout.addWidget(label, row, column)
                setup_layout.addWidget(widget, row, column + 1)

        mode_label = QtWidgets.QLabel("Layout mode", layout_page)
        mode_label.setObjectName("figureComposerLayoutModeControls")
        mode_label.setBuddy(self.layout_mode_combo)
        mode_label.setToolTip(self.layout_mode_combo.toolTip())
        mode_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        setup_layout.addWidget(mode_label, 0, 0, 1, 2)
        setup_layout.addWidget(self.layout_mode_combo, 0, 2, 1, 3)

        add_grid_pair_row(
            1,
            "Grid",
            "figureComposerGridControls",
            "Subplot grid or active GridSpec grid.",
            "Rows",
            self.nrows_spin,
            "Number of rows in the subplot grid or active GridSpec grid.",
            "Columns",
            self.ncols_spin,
            "Number of columns in the subplot grid or active GridSpec grid.",
        )

        self.gridspec_editor_container = QtWidgets.QWidget(layout_page)
        self.gridspec_editor_container.setObjectName(
            "figureComposerGridSpecEditorContainer"
        )
        gridspec_container_layout = QtWidgets.QVBoxLayout(
            self.gridspec_editor_container
        )
        gridspec_container_layout.setContentsMargins(0, 2, 0, 2)
        gridspec_container_layout.setSpacing(4)
        self.gridspec_editor_top_line = QtWidgets.QFrame(self.gridspec_editor_container)
        self.gridspec_editor_top_line.setObjectName(
            "figureComposerGridSpecEditorTopLine"
        )
        self.gridspec_editor_top_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.gridspec_editor_top_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        gridspec_container_layout.addWidget(self.gridspec_editor_top_line)

        self.gridspec_editor_widget = QtWidgets.QWidget(self.gridspec_editor_container)
        self.gridspec_editor_widget.setObjectName("figureComposerGridSpecEditor")
        gridspec_editor_layout = QtWidgets.QVBoxLayout(self.gridspec_editor_widget)
        gridspec_editor_layout.setContentsMargins(0, 0, 0, 0)
        gridspec_editor_layout.setSpacing(4)
        self.gridspec_breadcrumb_widget = QtWidgets.QWidget(self.gridspec_editor_widget)
        self.gridspec_breadcrumb_layout = QtWidgets.QHBoxLayout(
            self.gridspec_breadcrumb_widget
        )
        self.gridspec_breadcrumb_layout.setContentsMargins(0, 0, 0, 0)
        self.gridspec_breadcrumb_layout.setSpacing(2)
        gridspec_editor_layout.addWidget(self.gridspec_breadcrumb_widget)

        gridspec_action_layout = QtWidgets.QHBoxLayout()
        self.gridspec_region_kind_combo = QtWidgets.QComboBox(
            self.gridspec_editor_widget
        )
        self.gridspec_region_kind_combo.setObjectName(
            "figureComposerGridSpecRegionKindCombo"
        )
        self.gridspec_region_kind_combo.addItem("Axes", "axes")
        self.gridspec_region_kind_combo.addItem("Nested Grid", "grid")
        self.gridspec_region_kind_combo.setToolTip(
            "Kind of region created by the next drag gesture."
        )
        self.gridspec_parent_grid_button = _step_toolbar_button(
            self.gridspec_editor_widget,
            "figureComposerGridSpecParentButton",
            "Parent",
            "Return to the parent GridSpec grid.",
        )
        self.gridspec_parent_grid_button.clicked.connect(
            self._gridspec_open_parent_grid
        )
        draw_label = QtWidgets.QLabel("Draw", self.gridspec_editor_widget)
        draw_label.setBuddy(self.gridspec_region_kind_combo)
        gridspec_action_layout.addWidget(draw_label)
        gridspec_action_layout.addWidget(self.gridspec_region_kind_combo)
        gridspec_action_layout.addWidget(self.gridspec_parent_grid_button)
        gridspec_action_layout.addStretch(1)
        gridspec_editor_layout.addLayout(gridspec_action_layout)

        self.gridspec_layout_widget = _GridSpecViewWidget(
            self.gridspec_editor_widget, mode="edit"
        )
        self.gridspec_layout_widget.set_creation_kind("axes")
        self.gridspec_layout_widget.sigRegionCreated.connect(
            self._gridspec_region_created
        )
        self.gridspec_layout_widget.sigRegionChanged.connect(
            self._gridspec_region_changed
        )
        self.gridspec_layout_widget.sigRegionSelected.connect(
            self._gridspec_region_selected
        )
        self.gridspec_layout_widget.sigNestedGridActivated.connect(
            self._gridspec_open_grid
        )
        self.gridspec_region_kind_combo.currentIndexChanged.connect(
            self._gridspec_region_kind_changed
        )
        gridspec_editor_layout.addWidget(self.gridspec_layout_widget)

        gridspec_region_layout = QtWidgets.QHBoxLayout()
        self.gridspec_region_label_edit = QtWidgets.QLineEdit(
            self.gridspec_editor_widget
        )
        self.gridspec_region_label_edit.setObjectName(
            "figureComposerGridSpecRegionLabelEdit"
        )
        self.gridspec_region_label_edit.setPlaceholderText("Autogenerated")
        self.gridspec_region_label_edit.setToolTip(
            "Optional Python variable name used in generated code.\n"
            "Leave blank to use the autogenerated name."
        )
        self.gridspec_region_label_edit.editingFinished.connect(
            self._gridspec_region_label_changed
        )
        self.gridspec_open_grid_button = _step_toolbar_button(
            self.gridspec_editor_widget,
            "figureComposerGridSpecOpenButton",
            "Open",
            "Edit the selected nested grid.",
        )
        self.gridspec_open_grid_button.clicked.connect(
            self._gridspec_open_selected_grid
        )
        self.gridspec_delete_region_button = _step_toolbar_button(
            self.gridspec_editor_widget,
            "figureComposerGridSpecDeleteButton",
            "Delete",
            "Delete the selected axes or nested grid region.",
        )
        self.gridspec_delete_region_button.clicked.connect(
            self._gridspec_delete_selected_region
        )
        self.gridspec_region_name_label = QtWidgets.QLabel(
            "Variable name", self.gridspec_editor_widget
        )
        self.gridspec_region_name_label.setBuddy(self.gridspec_region_label_edit)
        gridspec_region_layout.addWidget(self.gridspec_region_name_label)
        gridspec_region_layout.addWidget(self.gridspec_region_label_edit, 1)
        gridspec_region_layout.addWidget(self.gridspec_open_grid_button)
        gridspec_region_layout.addWidget(self.gridspec_delete_region_button)
        gridspec_editor_layout.addLayout(gridspec_region_layout)

        self.gridspec_status_label = QtWidgets.QLabel(self.gridspec_editor_widget)
        self.gridspec_status_label.setObjectName("figureComposerGridSpecStatus")
        self.gridspec_status_label.setWordWrap(True)
        gridspec_editor_layout.addWidget(self.gridspec_status_label)
        gridspec_container_layout.addWidget(self.gridspec_editor_widget)
        self.gridspec_editor_bottom_line = QtWidgets.QFrame(
            self.gridspec_editor_container
        )
        self.gridspec_editor_bottom_line.setObjectName(
            "figureComposerGridSpecEditorBottomLine"
        )
        self.gridspec_editor_bottom_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.gridspec_editor_bottom_line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        gridspec_container_layout.addWidget(self.gridspec_editor_bottom_line)
        setup_layout.addWidget(self.gridspec_editor_container, 2, 0, 1, 5)

        add_grid_pair_row(
            3,
            "Size (in)",
            "figureComposerSizeControls",
            "Figure size in inches. The plot window follows this size.",
            "Width",
            self.width_spin,
            "Figure width in inches. The plot window follows this size.",
            "Height",
            self.height_spin,
            "Figure height in inches. The plot window follows this size.",
        )
        add_grid_pair_row(
            4,
            "Size (mm)",
            "figureComposerSizeMmControls",
            "Figure size in millimeters, synced with the inch controls.",
            "Width",
            self.width_mm_spin,
            "Figure width in millimeters. Converted to inches for Matplotlib.",
            "Height",
            self.height_mm_spin,
            "Figure height in millimeters. Converted to inches for Matplotlib.",
        )
        dpi_tooltip = "Figure dots per inch passed to Matplotlib and generated code."
        dpi_label = QtWidgets.QLabel("DPI", layout_page)
        dpi_label.setObjectName("figureComposerDpiControls")
        dpi_label.setBuddy(self.dpi_spin)
        dpi_label.setToolTip(dpi_tooltip)
        dpi_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.dpi_spin.setToolTip(dpi_tooltip)
        setup_layout.addWidget(dpi_label, 5, 0, 1, 2)
        setup_layout.addWidget(self.dpi_spin, 5, 2, 1, 3)
        layout_row_label = QtWidgets.QLabel("Layout engine", layout_page)
        layout_row_label.setObjectName("figureComposerLayoutControls")
        layout_row_label.setToolTip(
            "Creation-time Matplotlib layout engine.\n"
            "Default omits the layout argument.\n"
            "None passes layout='none'."
        )
        layout_row_label.setBuddy(self.layout_combo)
        layout_row_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.layout_combo.setToolTip(
            "Creation-time Matplotlib layout engine.\n"
            "Default omits the layout argument.\n"
            "None passes layout='none'.",
        )
        setup_layout.addWidget(layout_row_label, 6, 0, 1, 2)
        setup_layout.addWidget(self.layout_combo, 6, 2, 1, 3)
        add_grid_pair_row(
            7,
            "Share axes",
            "figureComposerShareControls",
            "Matplotlib shared-axis settings passed to plt.subplots.",
            "x",
            self.sharex_combo,
            "Matplotlib sharex setting passed to plt.subplots.",
            "y",
            self.sharey_combo,
            "Matplotlib sharey setting passed to plt.subplots.",
        )
        add_grid_pair_row(
            8,
            "Ratios",
            "figureComposerRatioControls",
            "Optional width and height ratios for subplots or active GridSpec grid.",
            "Widths",
            self.width_ratios_edit,
            "Optional width ratios, one positive number per column.",
            "Heights",
            self.height_ratios_edit,
            "Optional height ratios, one positive number per row.",
        )
        setup_layout.setRowStretch(9, 1)

        layout_index = self.editor_tabs.addTab(layout_page, "Layout")
        self.editor_tabs.setTabToolTip(
            layout_index, "Subplot grid, figure size, and shared axes."
        )
        recipe_index = self.editor_tabs.addTab(recipe_page, "Recipe")
        self.editor_tabs.setTabToolTip(
            recipe_index,
            "Ordered plotting steps and controls for the selected step.",
        )
        self.editor_tabs.setCurrentWidget(recipe_page)

        for widget in (
            self.nrows_spin,
            self.ncols_spin,
            self.width_spin,
            self.height_spin,
            self.dpi_spin,
        ):
            typing.cast("QtWidgets.QAbstractSpinBox", widget).editingFinished.connect(
                self._setup_controls_changed
            )
        self.nrows_spin.valueChanged.connect(self._setup_controls_changed)
        self.ncols_spin.valueChanged.connect(self._setup_controls_changed)
        for widget in (self.width_mm_spin, self.height_mm_spin):
            typing.cast("QtWidgets.QAbstractSpinBox", widget).editingFinished.connect(
                self._size_mm_controls_changed
            )
        for widget in (self.width_ratios_edit, self.height_ratios_edit):
            widget.editingFinished.connect(self._setup_controls_changed)
        self.layout_mode_combo.currentTextChanged.connect(self._layout_mode_changed)
        self.layout_combo.currentTextChanged.connect(self._setup_controls_changed)
        self.sharex_combo.currentTextChanged.connect(self._setup_controls_changed)
        self.sharey_combo.currentTextChanged.connect(self._setup_controls_changed)
        for combo in (
            self.layout_mode_combo,
            self.layout_combo,
            self.sharex_combo,
            self.sharey_combo,
            self.gridspec_region_kind_combo,
        ):
            self._track_combo_interaction(combo)
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            clipboard = application.clipboard()
            if clipboard is not None:
                clipboard.dataChanged.connect(self._update_step_action_buttons)
                self._connected_step_clipboard = clipboard

        self.setCentralWidget(root)
        self.setWindowTitle("Figure Composer")

    def _apply_recipe_to_controls(self) -> None:
        self._updating_controls = True
        try:
            setup = self._recipe.setup
            self._set_combo_value(self.layout_mode_combo, setup.layout_mode)
            self._sync_active_grid_controls(setup)
            self.width_spin.setValue(setup.figsize[0])
            self.height_spin.setValue(setup.figsize[1])
            self.dpi_spin.setValue(setup.dpi)
            self._sync_size_mm_controls(setup.figsize)
            self._set_combo_value(self.layout_combo, setup.layout or "default")
            self._set_combo_value(self.sharex_combo, str(setup.sharex))
            self._set_combo_value(self.sharey_combo, str(setup.sharey))
            self._refresh_source_list()
            self._rebuild_axes_grid()
            self._refresh_operation_list()
            if self.operation_list.count() and self.operation_list.currentRow() < 0:
                self.operation_list.setCurrentRow(0)
            self._update_operation_editor()
        finally:
            self._updating_controls = False

    @staticmethod
    def _set_combo_value(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findText(value)
        combo.setCurrentIndex(max(index, 0))

    def _sync_size_mm_controls(self, figsize: tuple[float, float]) -> None:
        self.width_mm_spin.setValue(figsize[0] * _MM_PER_INCH)
        self.height_mm_spin.setValue(figsize[1] * _MM_PER_INCH)

    def _sync_active_grid_controls(self, setup: FigureSubplotsState) -> None:
        if setup.layout_mode == "gridspec":
            active_grid = _gridspec_grid_by_id(setup, self._active_gridspec_grid_id)
            if active_grid is None:
                self._active_gridspec_grid_id = setup.gridspec.root.grid_id
                active_grid = setup.gridspec.root
            self.nrows_spin.setValue(active_grid.nrows)
            self.ncols_spin.setValue(active_grid.ncols)
            self.width_ratios_edit.setText(_format_tuple(active_grid.width_ratios))
            self.height_ratios_edit.setText(_format_tuple(active_grid.height_ratios))
        else:
            self.nrows_spin.setValue(setup.nrows)
            self.ncols_spin.setValue(setup.ncols)
            self.width_ratios_edit.setText(_format_tuple(setup.width_ratios))
            self.height_ratios_edit.setText(_format_tuple(setup.height_ratios))
        show_gridspec_editor = setup.layout_mode == "gridspec"
        self.gridspec_editor_container.setVisible(show_gridspec_editor)
        self.gridspec_editor_widget.setVisible(show_gridspec_editor)
        self.sharex_combo.setEnabled(setup.layout_mode == "subplots")
        self.sharey_combo.setEnabled(setup.layout_mode == "subplots")

    @staticmethod
    def _ratio_tuple_from_text(text: str) -> tuple[float, ...]:
        values = tuple(float(value) for value in _literal_sequence_from_text(text))
        if any(value <= 0.0 for value in values):
            raise ValueError("subplot ratios must be positive")
        return values

    def _refresh_source_list(self) -> None:
        self.source_list.clear()
        source_by_name = {source.name: source for source in self._recipe.sources}
        duplicate_labels = _source_duplicate_labels(self._recipe.sources)
        for name, data in self._source_data.items():
            source = source_by_name.get(name)
            display = _source_display_label(
                source,
                name,
                disambiguate=(
                    source is not None
                    and (source.label.strip() or name) in duplicate_labels
                ),
            )
            self._add_source_list_row(
                display,
                name,
                _source_display_tooltip(source, name),
                data=data,
                missing=source is None,
            )

        missing = [
            source
            for source in self._recipe.sources
            if source.name not in self._source_data
        ]
        for source in missing:
            display = _source_display_label(
                source,
                source.name,
                disambiguate=(source.label.strip() or source.name) in duplicate_labels,
            )
            self._add_source_list_row(
                display,
                source.name,
                _source_display_tooltip(source, source.name),
                missing=True,
            )
        self.source_list.resizeColumnToContents(0)

    def _add_source_list_row(
        self,
        display: str,
        name: str,
        tooltip: str,
        *,
        data: xr.DataArray | None = None,
        missing: bool = False,
    ) -> None:
        item = QtWidgets.QTreeWidgetItem([display, "missing" if data is None else ""])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, name)
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        for column in range(self.source_list.columnCount()):
            item.setToolTip(column, tooltip)
        if missing:
            item.setForeground(0, QtGui.QBrush(QtGui.QColor("darkRed")))
            item.setForeground(1, QtGui.QBrush(QtGui.QColor("darkRed")))
        self.source_list.addTopLevelItem(item)
        if data is None:
            return

        shape_label = QtWidgets.QLabel(
            erlab.interactive.utils._apply_qt_accent_color(
                erlab.utils.formatting.format_darr_shape_html(
                    _public_source_data(data).rename(None),
                    show_size=False,
                )
            ),
            self.source_list,
        )
        shape_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        shape_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.NoTextInteraction
        )
        shape_label.setContentsMargins(4, 0, 4, 0)
        shape_label.setToolTip(tooltip)
        if missing:
            palette = shape_label.palette()
            palette.setColor(
                QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("darkRed")
            )
            shape_label.setPalette(palette)
        item.setSizeHint(1, shape_label.sizeHint())
        self.source_list.setItemWidget(item, 1, shape_label)

    def _rebuild_axes_grid(self) -> None:
        self.axes_selector.set_grid(
            self._recipe.setup.nrows,
            self._recipe.setup.ncols,
        )
        self._refresh_gridspec_editor()
        self._sync_axes_selector()

    def _refresh_gridspec_editor(self) -> None:
        setup = self._recipe.setup
        if setup.layout_mode != "gridspec":
            self.gridspec_editor_widget.setVisible(False)
            return
        self.gridspec_editor_widget.setVisible(True)
        grid = _gridspec_grid_by_id(setup, self._active_gridspec_grid_id)
        if grid is None:
            self._active_gridspec_grid_id = setup.gridspec.root.grid_id
            grid = setup.gridspec.root
        reserved_names = self._source_names()
        grid_names = _gridspec_grid_display_names(setup)
        regions = [
            _GridSpecRegionInfo(
                axis.axes_id,
                "axes",
                axis.span,
                _gridspec_axis_display_name(
                    setup, axis.axes_id, reserved_names=reserved_names
                ),
                _gridspec_region_valid(grid, axis.span),
            )
            for axis in grid.axes
        ]
        regions.extend(
            _GridSpecRegionInfo(
                child.grid_id,
                "grid",
                child.span,
                grid_names.get(child.grid_id, child.grid_id),
                _gridspec_region_valid(grid, child.span),
            )
            for child in grid.child_grids
            if child.span is not None
        )
        self.gridspec_layout_widget.set_edit_grid(
            grid,
            regions,
            {
                axes_id: _gridspec_axis_display_name(
                    setup, axes_id, reserved_names=reserved_names
                )
                for axes_id in _gridspec_all_axes_ids(setup)
            },
        )
        self._refresh_gridspec_breadcrumbs()
        self._refresh_gridspec_region_controls()
        self._refresh_gridspec_axes_selector()
        self._refresh_gridspec_status(grid)

    def _refresh_gridspec_status(self, grid: FigureGridSpecGridState) -> None:
        reserved_names = self._source_names()
        invalid_regions = [
            _gridspec_region_label(
                self._recipe.setup,
                grid,
                axis.axes_id,
                reserved_names=reserved_names,
            )
            for axis in grid.axes
            if not _gridspec_region_valid(grid, axis.span)
        ]
        invalid_regions.extend(
            _gridspec_region_label(
                self._recipe.setup,
                grid,
                child.grid_id,
                reserved_names=reserved_names,
            )
            for child in grid.child_grids
            if child.span is None or not _gridspec_region_valid(grid, child.span)
        )
        if invalid_regions:
            self.gridspec_status_label.setText(
                "Some regions are outside the active grid: "
                + ", ".join(invalid_regions)
                + ". Increase rows/columns or delete and redraw them."
            )
            return
        if _gridspec_has_invalid_regions(self._recipe.setup.gridspec.root):
            self.gridspec_status_label.setText(
                "A nested grid contains regions outside its bounds."
            )
            return
        kind = self.gridspec_region_kind_combo.currentText()
        self.gridspec_status_label.setText(
            f"Drag cells to create {kind.lower()} regions. "
            "Double-click a nested grid to edit it."
        )

    def _refresh_gridspec_breadcrumbs(self) -> None:
        for button in self._gridspec_breadcrumb_buttons:
            button.deleteLater()
        self._gridspec_breadcrumb_buttons.clear()
        while self.gridspec_breadcrumb_layout.count():
            item = self.gridspec_breadcrumb_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        path = _gridspec_grid_path(self._recipe.setup, self._active_gridspec_grid_id)
        for index, grid in enumerate(path):
            if index:
                separator = QtWidgets.QLabel(">", self.gridspec_breadcrumb_widget)
                self.gridspec_breadcrumb_layout.addWidget(separator)
            button = QtWidgets.QToolButton(self.gridspec_breadcrumb_widget)
            button.setText(
                _gridspec_grid_display_name(self._recipe.setup, grid.grid_id)
            )
            button.setToolTip("Open this GridSpec grid.")
            button.clicked.connect(
                lambda _checked=False, grid_id=grid.grid_id: self._gridspec_open_grid(
                    grid_id
                )
            )
            self._gridspec_breadcrumb_buttons.append(button)
            self.gridspec_breadcrumb_layout.addWidget(button)
        self.gridspec_breadcrumb_layout.addStretch(1)
        self.gridspec_parent_grid_button.setEnabled(len(path) > 1)

    def _refresh_gridspec_region_controls(self) -> None:
        region_id = self.gridspec_layout_widget.selected_region_id()
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        label = ""
        kind = ""
        placeholder = "Autogenerated"
        if grid is not None and region_id:
            code_names = _gridspec_axis_code_names(
                self._recipe.setup, reserved_names=self._source_names()
            )
            for axis in grid.axes:
                if axis.axes_id == region_id:
                    label = axis.label
                    kind = "axes"
                    placeholder = (
                        f"Autogenerated: {code_names.get(region_id, region_id)}"
                    )
                    break
            if not kind:
                for child in grid.child_grids:
                    if child.grid_id == region_id:
                        kind = "grid"
                        break
        blocker = QtCore.QSignalBlocker(self.gridspec_region_label_edit)
        self.gridspec_region_label_edit.setText(label)
        del blocker
        has_region = bool(kind)
        self.gridspec_region_label_edit.setPlaceholderText(placeholder)
        self._set_gridspec_variable_name_invalid(False)
        self.gridspec_region_name_label.setVisible(kind != "grid")
        self.gridspec_region_label_edit.setVisible(kind != "grid")
        self.gridspec_region_label_edit.setEnabled(kind == "axes")
        self.gridspec_delete_region_button.setEnabled(has_region)
        self.gridspec_open_grid_button.setEnabled(kind == "grid")
        self.gridspec_open_grid_button.setToolTip(
            "Edit the selected nested grid."
            if kind == "grid"
            else "Select a nested grid region to edit it."
        )
        self.gridspec_delete_region_button.setToolTip(
            "Delete the selected axes or nested grid region."
            if has_region
            else "Select an axes or nested grid region to delete it."
        )

    def _refresh_gridspec_axes_selector(self) -> None:
        selected_ids: set[str] = set()
        current = self._current_operation()
        if current is not None and _registry.spec_for(current[1].kind).uses_axes(
            current[1]
        ):
            selected_ids = set(current[1].axes.axes_ids)
        blocker = QtCore.QSignalBlocker(self.gridspec_axes_selector)
        axes_ids = _gridspec_all_axes_ids(self._recipe.setup)
        self.gridspec_axes_selector.set_layout(
            self._recipe.setup.gridspec.root,
            {
                axes_id: _gridspec_axis_display_name(
                    self._recipe.setup,
                    axes_id,
                    reserved_names=self._source_names(),
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
                if self._recipe.setup.layout_mode == "gridspec":
                    selected_axes_ids = _gridspec_valid_axes_ids(
                        self._recipe.setup, operation.axes.axes_ids
                    )
                    invalid_axes_ids = _gridspec_invalid_axes_ids(
                        self._recipe.setup, operation.axes.axes_ids
                    )
                else:
                    selected_axes = set(operation.axes.valid_axes(self._recipe.setup))
                    invalid_axes = operation.axes.invalid_axes(self._recipe.setup)
                expression = operation.axes.expression

        self._updating_controls = True
        try:
            grid_mode = self._recipe.setup.layout_mode == "gridspec"
            self.axes_selector.setVisible(not grid_mode)
            self.axes_expression_edit.setVisible(not grid_mode)
            self.gridspec_axes_selector.setVisible(grid_mode)
            self.axes_selector.set_selected_axes(tuple(sorted(selected_axes)))
            self._refresh_gridspec_axes_selector()
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
                            self._recipe.setup,
                            selected_axes_ids,
                            reserved_names=self._source_names(),
                        )
                    )
                else:
                    axes_text = _format_axes_tuple(
                        tuple(sorted(selected_axes)),
                        nrows=self._recipe.setup.nrows,
                        ncols=self._recipe.setup.ncols,
                    )
                self.target_axes_status_label.setText(f"Targets: {axes_text}")
        finally:
            self._updating_controls = False

    def _refresh_operation_list(self) -> None:
        current_id = None
        current = self._current_operation()
        if current is not None:
            current_id = current[1].operation_id
        selected_ids = self._selected_operation_ids()
        if not selected_ids and current_id is not None:
            selected_ids = {current_id}
        self.operation_list.blockSignals(True)
        try:
            self.operation_list.clear()
            for operation in self._recipe.operations:
                render_error = self._operation_render_errors.get(operation.operation_id)
                input_error = self._operation_input_error_text(operation)
                text = self._operation_display_text(operation)
                if input_error is not None:
                    text = f"{text} (invalid input)"
                if render_error is not None:
                    text = f"{text} (render error)"
                item = QtWidgets.QListWidgetItem(text)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if operation.enabled
                    else QtCore.Qt.CheckState.Unchecked
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, operation.operation_id)
                tooltip = self._operation_tooltip(operation)
                if input_error is not None:
                    tooltip = f"{tooltip}\n\nInvalid input: {input_error}"
                if render_error is not None:
                    tooltip = f"{tooltip}\n\nRender error: {render_error}"
                item.setToolTip(tooltip)
                if (
                    self._operation_has_invalid_axes(operation)
                    or input_error is not None
                    or render_error is not None
                ):
                    item.setForeground(QtGui.QBrush(QtGui.QColor("darkRed")))
                self.operation_list.addItem(item)
                if operation.operation_id in selected_ids:
                    item.setSelected(True)
                if operation.operation_id == current_id:
                    self.operation_list.setCurrentItem(item)
        finally:
            self.operation_list.blockSignals(False)

    def _set_current_operation_row_silent(
        self, index: int, *, preserve_selection: bool = True
    ) -> None:
        selected_ids = self._selected_operation_ids() if preserve_selection else set()
        was_blocked = self.operation_list.blockSignals(True)
        try:
            self.operation_list.setCurrentRow(index)
            if preserve_selection and selected_ids:
                self._set_selected_operation_ids_silent(selected_ids)
        finally:
            self.operation_list.blockSignals(was_blocked)

    def _operation_id_for_item(self, item: QtWidgets.QListWidgetItem) -> str | None:
        operation_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return operation_id if isinstance(operation_id, str) else None

    def _selected_operation_ids(self) -> set[str]:
        return {
            operation_id
            for item in self.operation_list.selectedItems()
            if (operation_id := self._operation_id_for_item(item)) is not None
        }

    def _set_selected_operation_ids_silent(self, operation_ids: set[str]) -> None:
        was_blocked = self.operation_list.blockSignals(True)
        try:
            for row in range(self.operation_list.count()):
                item = self.operation_list.item(row)
                if item is None:
                    continue
                item.setSelected(self._operation_id_for_item(item) in operation_ids)
        finally:
            self.operation_list.blockSignals(was_blocked)

    def _operation_display_text(self, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).display_text(self, operation)

    def _operation_tooltip(self, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).tooltip(self, operation)

    def _set_operation_render_errors(self, errors: Mapping[str, str]) -> None:
        render_errors = dict(errors)
        if render_errors == self._operation_render_errors:
            return
        self._operation_render_errors = render_errors
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        current = self._current_operation()
        self._update_source_status(current[1] if current is not None else None)

    def _set_operation_input_errors(
        self, errors: Mapping[str, Mapping[str, str]]
    ) -> None:
        input_errors = {
            operation_id: dict(operation_errors)
            for operation_id, operation_errors in errors.items()
            if operation_errors
        }
        if input_errors == self._operation_input_errors:
            return
        self._operation_input_errors = input_errors
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        current = self._current_operation()
        self._update_source_status(current[1] if current is not None else None)

    def _operation_has_invalid_input(self, operation: FigureOperationState) -> bool:
        return operation.operation_id in self._operation_input_errors

    def _operation_input_error_text(
        self, operation: FigureOperationState
    ) -> str | None:
        operation_errors = self._operation_input_errors.get(operation.operation_id)
        if not operation_errors:
            return None
        messages = tuple(dict.fromkeys(operation_errors.values()))
        if len(messages) == 1:
            return messages[0]
        return f"{len(messages)} invalid inputs: " + "; ".join(messages)

    def _record_editor_input_error(
        self, widget: QtWidgets.QWidget, error: FigureComposerInputError
    ) -> None:
        operation_ids = self._editable_operation_ids_for_error()
        if not operation_ids:
            return
        key = self._editor_input_error_key(widget)
        self._record_editor_input_error_for_key(key, operation_ids, error)

    def _record_editor_input_error_for_key(
        self,
        key: str,
        operation_ids: Sequence[str],
        error: FigureComposerInputError,
    ) -> None:
        if not operation_ids:
            return
        errors = {
            operation_id: dict(operation_errors)
            for operation_id, operation_errors in self._operation_input_errors.items()
        }
        for operation_id in operation_ids:
            operation_errors = errors.setdefault(operation_id, {})
            operation_errors[key] = str(error)
        self._set_operation_input_errors(errors)

    def _clear_editor_input_error(self, widget: QtWidgets.QWidget) -> None:
        if not self._operation_input_errors:
            return
        operation_ids = self._editable_operation_ids_for_error()
        if not operation_ids:
            return
        key = self._editor_input_error_key(widget)
        self._clear_editor_input_error_for_key(key, operation_ids)

    def _clear_editor_input_error_for_key(
        self, key: str, operation_ids: Sequence[str]
    ) -> None:
        if not self._operation_input_errors or not operation_ids:
            return
        errors = {
            operation_id: dict(operation_errors)
            for operation_id, operation_errors in self._operation_input_errors.items()
        }
        changed = False
        for operation_id in operation_ids:
            operation_errors = errors.get(operation_id)
            if operation_errors is None or key not in operation_errors:
                continue
            del operation_errors[key]
            changed = True
        if changed:
            self._set_operation_input_errors(errors)

    @staticmethod
    def _editor_input_error_key(widget: QtWidgets.QWidget) -> str:
        if not erlab.interactive.utils.qt_is_valid(widget):
            return f"anonymous:{id(widget)}"
        object_name = widget.objectName()
        if object_name:
            return object_name
        return f"anonymous:{id(widget)}"

    def _editable_operation_ids_for_error(self) -> tuple[str, ...]:
        editable = self._editable_operations()
        if editable:
            return tuple(operation.operation_id for _index, operation in editable)
        current = self._current_operation()
        return () if current is None else (current[1].operation_id,)

    def _clear_operation_input_errors(self, operation_ids: Sequence[str]) -> None:
        if not operation_ids or not self._operation_input_errors:
            return
        errors = dict(self._operation_input_errors)
        for operation_id in operation_ids:
            errors.pop(operation_id, None)
        self._set_operation_input_errors(errors)

    def _axes_target_text(self, selection: FigureAxesSelectionState) -> str:
        if selection.expression:
            return selection.expression
        if self._recipe.setup.layout_mode == "gridspec":
            invalid_ids = _gridspec_invalid_axes_ids(
                self._recipe.setup, selection.axes_ids
            )
            if invalid_ids:
                return _removed_axes_summary_text(len(invalid_ids))
            valid_ids = _gridspec_valid_axes_ids(self._recipe.setup, selection.axes_ids)
            if not valid_ids:
                return "none"
            return ", ".join(
                _gridspec_axis_display_names(
                    self._recipe.setup,
                    valid_ids,
                    reserved_names=self._source_names(),
                )
            )
        invalid_axes = selection.invalid_axes(self._recipe.setup)
        if invalid_axes:
            return f"removed axes {_format_axes_tuple(invalid_axes)}"
        valid_axes = selection.valid_axes(self._recipe.setup)
        if not valid_axes:
            return "none"
        return _format_axes_tuple(
            valid_axes,
            nrows=self._recipe.setup.nrows,
            ncols=self._recipe.setup.ncols,
        )

    def _operation_has_invalid_axes(self, operation: FigureOperationState) -> bool:
        return _registry.spec_for(operation.kind).has_invalid_target(self, operation)

    def _axes_selection_has_invalid_target(
        self, selection: FigureAxesSelectionState
    ) -> bool:
        if selection.expression:
            return False
        if self._recipe.setup.layout_mode == "gridspec":
            if not _gridspec_valid_axes_ids(self._recipe.setup, selection.axes_ids):
                return True
            return bool(
                _gridspec_invalid_axes_ids(self._recipe.setup, selection.axes_ids)
            )
        return bool(selection.invalid_axes(self._recipe.setup)) or not bool(
            selection.valid_axes(self._recipe.setup)
        )

    def _invalid_operation_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, operation in enumerate(self._recipe.operations)
            if operation.enabled
            and (
                self._operation_has_invalid_axes(operation)
                or self._operation_has_invalid_input(operation)
            )
        )

    def _invalid_operation_target_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, operation in enumerate(self._recipe.operations)
            if operation.enabled and self._operation_has_invalid_axes(operation)
        )

    def _warn_invalid_operation_targets(self) -> bool:
        indices = self._invalid_operation_target_indices()
        if not indices:
            return False
        self.editor_tabs.setCurrentWidget(self.recipe_page)
        self.operation_list.setCurrentRow(indices[0])
        self._select_step_section("axes")
        QtWidgets.QMessageBox.warning(
            self,
            "Retarget Figure Steps",
            "Some enabled steps target axes that no longer exist in the current "
            "figure layout. Use All Axes or Drop Removed Axes before copying, "
            "exporting, or rendering the full recipe.",
        )
        return True

    def _current_operation(self) -> tuple[int, FigureOperationState] | None:
        row = self.operation_list.currentRow()
        if row < 0 or row >= len(self._recipe.operations):
            return None
        return row, self._recipe.operations[row]

    def _selected_operation_indices(self) -> tuple[int, ...]:
        selected_ids = self._selected_operation_ids()
        if not selected_ids:
            current = self._current_operation()
            return () if current is None else (current[0],)
        return tuple(
            index
            for index, operation in enumerate(self._recipe.operations)
            if operation.operation_id in selected_ids
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
            self._operation_editor_schema_key(self._recipe.operations[index])
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
            (index, self._recipe.operations[index])
            for index in self._editable_operation_indices()
        )

    def _batch_value(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
    ) -> typing.Any:
        editable = self._editable_operations()
        if len(editable) <= 1:
            return getter(operation)
        values = [getter(target) for _index, target in editable]
        first = values[0]
        if all(value == first for value in values[1:]):
            return first
        return _MIXED_VALUE

    def _batch_is_mixed(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
    ) -> bool:
        return self._batch_value(operation, getter) is _MIXED_VALUE

    def _batch_text(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
        formatter: Callable[[typing.Any], str],
    ) -> tuple[str, bool]:
        value = self._batch_value(operation, getter)
        if value is _MIXED_VALUE:
            return "", True
        return formatter(value), False

    def _batch_combo_text(
        self,
        operation: FigureOperationState,
        getter: Callable[[FigureOperationState], typing.Any],
        formatter: Callable[[typing.Any], str] = str,
    ) -> str | None:
        value = self._batch_value(operation, getter)
        if value is _MIXED_VALUE:
            return None
        return formatter(value)

    def _batch_options_match(
        self,
        operation: FigureOperationState,
        options_getter: Callable[[FigureOperationState], Sequence[str]],
    ) -> bool:
        editable = self._editable_operations()
        if len(editable) <= 1:
            return True
        expected = tuple(options_getter(operation))
        return all(
            tuple(options_getter(target)) == expected for _index, target in editable
        )

    @staticmethod
    def _line_edit_batch_unchanged(edit: QtWidgets.QLineEdit) -> bool:
        return LineEditControlAdapter(edit).unchanged_mixed()

    @staticmethod
    def _apply_mixed_line_edit(edit: QtWidgets.QLineEdit, mixed: bool) -> None:
        LineEditControlAdapter(edit).set_mixed(mixed)

    @staticmethod
    def _apply_mixed_plain_text_edit(
        edit: QtWidgets.QPlainTextEdit, mixed: bool
    ) -> None:
        PlainTextControlAdapter(edit).set_mixed(mixed)

    @staticmethod
    def _set_combo_mixed_placeholder(combo: QtWidgets.QComboBox) -> None:
        ComboBoxControlAdapter(combo).set_mixed(True)

    @staticmethod
    def _mixed_combo_text(text: str) -> bool:
        return text == _MIXED_VALUES_TEXT

    def _update_operations(
        self,
        updater: Callable[[int, FigureOperationState], FigureOperationState],
        *,
        render: bool = True,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> None:
        editable = self._editable_operations()
        if not editable:
            return
        current = self._current_operation()
        operations = list(self._recipe.operations)
        for index, operation in editable:
            operations[index] = updater(index, operation)
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
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
        if render:
            _render_preview(self)
            self.sigInfoChanged.emit()
        self._write_state()

    def _replace_operation(
        self,
        index: int,
        operation: FigureOperationState,
        *,
        render: bool = True,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> None:
        operations = list(self._recipe.operations)
        previous_operation = operations[index]
        operations[index] = operation
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._clear_operation_input_errors(
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
        if render:
            _render_preview(self)
            self.sigInfoChanged.emit()
        self._write_state()

    def _update_current_operation(self, **updates: typing.Any) -> None:
        if self._updating_controls:
            return
        self._update_operations(
            lambda _index, operation: operation.model_copy(update=updates)
        )

    def _update_current_operation_rebuild(self, **updates: typing.Any) -> None:
        if self._updating_controls:
            return
        self._update_operations(
            lambda _index, operation: operation.model_copy(update=updates),
            rebuild_editor=True,
        )

    def _queue_preview_render_update(self) -> None:
        if self._closing:
            return
        self._preview_render_update_generation += 1
        generation = self._preview_render_update_generation
        self._preview_render_update_pending = True
        erlab.interactive.utils.single_shot(
            self,
            _PREVIEW_RENDER_UPDATE_DELAY_MS,
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
        if self._rendering:
            self._queue_preview_render_update()
            return
        _render_preview(self)
        self.sigInfoChanged.emit()

    def _update_step_action_buttons(self) -> None:
        indices = self._selected_operation_indices()
        can_paste = self._clipboard_step_payload() is not None
        if not indices:
            self.remove_operation_button.setEnabled(False)
            self.duplicate_operation_button.setEnabled(False)
            self.copy_operation_button.setEnabled(False)
            self.cut_operation_button.setEnabled(False)
            self.paste_operation_button.setEnabled(can_paste)
            self.move_operation_up_button.setEnabled(False)
            self.move_operation_down_button.setEnabled(False)
            return
        index_set = set(indices)
        self.remove_operation_button.setEnabled(True)
        self.duplicate_operation_button.setEnabled(True)
        self.copy_operation_button.setEnabled(True)
        self.cut_operation_button.setEnabled(True)
        self.paste_operation_button.setEnabled(can_paste)
        self.move_operation_up_button.setEnabled(
            any(index > 0 and index - 1 not in index_set for index in indices)
        )
        self.move_operation_down_button.setEnabled(
            any(
                index < len(self._recipe.operations) - 1 and index + 1 not in index_set
                for index in indices
            )
        )

    @QtCore.Slot()
    @QtCore.Slot(int)
    @QtCore.Slot(str)
    def _setup_controls_changed(self, _value: object | None = None) -> None:
        if self._updating_controls:
            return
        try:
            width_ratios = self._ratio_tuple_from_text(self.width_ratios_edit.text())
            height_ratios = self._ratio_tuple_from_text(self.height_ratios_edit.text())
            if self._recipe.setup.layout_mode == "gridspec":
                setup = self._recipe.setup.model_copy(
                    update={
                        "figsize": (self.width_spin.value(), self.height_spin.value()),
                        "dpi": self.dpi_spin.value(),
                        "layout": self._layout_combo_value(),
                    }
                )

                def update_grid(
                    grid: FigureGridSpecGridState,
                ) -> FigureGridSpecGridState:
                    return grid.model_copy(
                        update={
                            "nrows": self.nrows_spin.value(),
                            "ncols": self.ncols_spin.value(),
                            "width_ratios": width_ratios,
                            "height_ratios": height_ratios,
                        }
                    )

                setup = _gridspec_replace_grid(
                    setup, self._active_gridspec_grid_id, update_grid
                )
            else:
                setup = FigureSubplotsState(
                    layout_mode="subplots",
                    nrows=self.nrows_spin.value(),
                    ncols=self.ncols_spin.value(),
                    figsize=(self.width_spin.value(), self.height_spin.value()),
                    dpi=self.dpi_spin.value(),
                    layout=self._layout_combo_value(),
                    sharex=self._combo_bool_or_text(self.sharex_combo),
                    sharey=self._combo_bool_or_text(self.sharey_combo),
                    width_ratios=width_ratios,
                    height_ratios=height_ratios,
                    gridspec=self._recipe.setup.gridspec,
                )
        except ValueError:
            return
        if setup == self._recipe.setup:
            return
        self._recipe = self._recipe.model_copy(update={"setup": setup})
        self._updating_controls = True
        try:
            self._sync_size_mm_controls(setup.figsize)
            self._sync_active_grid_controls(setup)
        finally:
            self._updating_controls = False
        if (
            self._figure_window is not None
            and erlab.interactive.utils.qt_is_valid(self._figure_window)
            and self._figure_window.isVisible()
        ):
            self._figure_window.resize_to_setup(setup)
            self.canvas.flush_events()
            self._sync_recipe_figsize_to_canvas(draw=False, emit_info=False)
        self._rebuild_axes_grid()
        self._refresh_operation_list()
        self._update_operation_editor()
        _render_preview(self)
        self.sigInfoChanged.emit()
        self._write_state()

    @QtCore.Slot()
    def _add_subplot_row(self) -> None:
        self._grow_subplot_grid("row")

    @QtCore.Slot()
    def _add_subplot_column(self) -> None:
        self._grow_subplot_grid("column")

    def _grow_subplot_grid(self, direction: typing.Literal["row", "column"]) -> bool:
        if self._recipe.setup.layout_mode != "subplots":
            return False
        if direction == "row":
            if self.nrows_spin.value() >= self.nrows_spin.maximum():
                return False
            self.nrows_spin.setValue(self.nrows_spin.value() + 1)
        else:
            if self.ncols_spin.value() >= self.ncols_spin.maximum():
                return False
            self.ncols_spin.setValue(self.ncols_spin.value() + 1)
        return True

    @QtCore.Slot()
    def _size_mm_controls_changed(self) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            self.width_spin.setValue(self.width_mm_spin.value() / _MM_PER_INCH)
            self.height_spin.setValue(self.height_mm_spin.value() / _MM_PER_INCH)
        finally:
            self._updating_controls = False
        self._setup_controls_changed()

    def _layout_combo_value(
        self,
    ) -> typing.Literal["constrained", "compressed", "tight", "none"] | None:
        text = self.layout_combo.currentText()
        if text == "default":
            return None
        return typing.cast(
            'typing.Literal["constrained", "compressed", "tight", "none"]',
            text,
        )

    @QtCore.Slot()
    def _layout_mode_changed(self) -> None:
        if self._updating_controls:
            return
        mode = self.layout_mode_combo.currentText()
        if mode == self._recipe.setup.layout_mode:
            return
        if mode == "gridspec":
            setup = _gridspec_setup_from_subplots(self._recipe.setup)
            axes_ids = _gridspec_all_axes_ids(setup)
            axis_id_by_tuple = {
                (row, col): axes_ids[row * setup.ncols + col]
                for row in range(setup.nrows)
                for col in range(setup.ncols)
                if row * setup.ncols + col < len(axes_ids)
            }
            operations = tuple(
                operation.model_copy(
                    update={
                        "axes": operation.axes.model_copy(
                            update={
                                "axes_ids": tuple(
                                    axis_id_by_tuple[axis]
                                    for axis in operation.axes.axes
                                    if axis in axis_id_by_tuple
                                ),
                                "expression": "",
                            }
                        )
                    }
                )
                if _registry.spec_for(operation.kind).uses_axes(operation)
                else operation
                for operation in self._recipe.operations
            )
            self._active_gridspec_grid_id = setup.gridspec.root.grid_id
        else:
            setup = _subplots_setup_from_gridspec(self._recipe.setup)
            operations = tuple(
                operation.model_copy(
                    update={
                        "axes": operation.axes.model_copy(
                            update={
                                "axes": _gridspec_axes_subplot_targets(
                                    self._recipe.setup, operation.axes.axes_ids
                                )
                                or operation.axes.axes
                                or ((0, 0),),
                                "expression": "",
                            }
                        )
                    }
                )
                if _registry.spec_for(operation.kind).uses_axes(operation)
                else operation
                for operation in self._recipe.operations
            )
        self._recipe = self._recipe.model_copy(
            update={"setup": setup, "operations": operations}
        )
        self._updating_controls = True
        try:
            self._sync_active_grid_controls(setup)
        finally:
            self._updating_controls = False
        self._rebuild_axes_grid()
        self._refresh_operation_list()
        self._update_operation_editor()
        _render_preview(self)
        self.sigInfoChanged.emit()
        self._write_state()

    @staticmethod
    def _combo_bool_or_text(
        combo: QtWidgets.QComboBox,
    ) -> bool | typing.Literal["none", "all", "row", "col"]:
        text = combo.currentText()
        if text == "True":
            return True
        if text == "False":
            return False
        return typing.cast('typing.Literal["none", "all", "row", "col"]', text)

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
            render=False,
            sync_axes=False,
        )
        self.sigInfoChanged.emit()
        self._queue_preview_render_update()
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
            render=False,
            sync_axes=False,
        )
        self.sigInfoChanged.emit()
        self._queue_preview_render_update()
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _gridspec_region_kind_changed(self, _index: int | None = None) -> None:
        kind = typing.cast(
            'typing.Literal["axes", "grid"]',
            self.gridspec_region_kind_combo.currentData(),
        )
        self.gridspec_layout_widget.set_creation_kind(kind)
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        if grid is not None:
            self._refresh_gridspec_status(grid)

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
            render=False,
            sync_axes=False,
        )
        self.sigInfoChanged.emit()
        self._queue_preview_render_update()
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _operation_selection_changed(self, _row: int | None = None) -> None:
        if _row is not None and not self._operation_multi_select_event:
            item = self.operation_list.item(_row)
            if item is not None:
                operation_id = self._operation_id_for_item(item)
                if operation_id is not None:
                    self._set_selected_operation_ids_silent({operation_id})
        self._sync_axes_selector()
        self._update_operation_editor()

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        self._handle_combo_interaction_event(watched, event)
        operation_list_viewport = self._operation_list_viewport
        if (
            operation_list_viewport is not None
            and watched is operation_list_viewport
            and event is not None
            and event.type()
            in {
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.MouseButtonDblClick,
                QtCore.QEvent.Type.KeyPress,
            }
        ):
            input_event = typing.cast("QtGui.QInputEvent", event)
            self._operation_multi_select_event = (
                self._operation_modifiers_enable_multi_selection(
                    input_event.modifiers()
                )
            )
            erlab.interactive.utils.single_shot(
                self, 0, self._clear_operation_multi_select_event
            )
        return super().eventFilter(watched, event)

    def _handle_combo_interaction_event(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> None:
        if event is None or watched is None:
            return
        event_type = event.type()
        if isinstance(watched, QtWidgets.QComboBox):
            if event_type in {
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.KeyPress,
                QtCore.QEvent.Type.Wheel,
            }:
                self._begin_transient_combo_popup_guard()
            return
        if not isinstance(watched, QtWidgets.QAbstractItemView):
            return
        if self._combo_for_popup_view(watched) is None:
            return
        if event_type in {QtCore.QEvent.Type.Show, QtCore.QEvent.Type.ShowToParent}:
            self._begin_combo_popup_view_guard(watched)
        elif event_type in {
            QtCore.QEvent.Type.Hide,
            QtCore.QEvent.Type.HideToParent,
            QtCore.QEvent.Type.Close,
        }:
            self._end_combo_popup_view_guard(watched)

    def _track_combo_interaction(self, combo: QtWidgets.QComboBox) -> None:
        if combo.property(_COMBO_TRACKED_PROPERTY):
            return
        combo.setProperty(_COMBO_TRACKED_PROPERTY, True)
        combo.installEventFilter(self)
        view = combo.view()
        if view is not None and erlab.interactive.utils.qt_is_valid(view):
            view.installEventFilter(self)
        self._tracked_combo_refs.append(weakref.ref(combo))

    def _combo_for_popup_view(
        self, view: QtWidgets.QAbstractItemView
    ) -> QtWidgets.QComboBox | None:
        matched_combo: QtWidgets.QComboBox | None = None
        for combo in self._live_tracked_combos():
            combo_view = combo.view()
            if combo_view is not None and combo_view is view:
                matched_combo = combo
        return matched_combo

    def _live_tracked_combos(self) -> tuple[QtWidgets.QComboBox, ...]:
        live_refs: list[weakref.ReferenceType[QtWidgets.QComboBox]] = []
        live_combos: list[QtWidgets.QComboBox] = []
        for combo_ref in self._tracked_combo_refs:
            combo = combo_ref()
            if combo is None or not erlab.interactive.utils.qt_is_valid(combo):
                continue
            live_refs.append(combo_ref)
            live_combos.append(combo)
        self._tracked_combo_refs = live_refs
        return tuple(live_combos)

    def _tracked_combo_popup_is_visible(self) -> bool:
        for combo in self._live_tracked_combos():
            view = combo.view()
            if view is not None and view.isVisible():
                return True
        return False

    def _begin_transient_combo_popup_guard(self) -> None:
        token = self._begin_combo_popup_guard()
        self._schedule_combo_popup_guard_end(token, _COMBO_INTERACTION_REBUILD_GRACE_MS)

    def _begin_combo_popup_view_guard(self, view: QtWidgets.QAbstractItemView) -> None:
        if isinstance(view.property(_COMBO_POPUP_GUARD_ID_PROPERTY), int):
            return
        view.setProperty(
            _COMBO_POPUP_GUARD_ID_PROPERTY,
            self._begin_combo_popup_guard(),
        )

    def _end_combo_popup_view_guard(self, view: QtWidgets.QAbstractItemView) -> None:
        token = view.property(_COMBO_POPUP_GUARD_ID_PROPERTY)
        if not isinstance(token, int):
            return
        view.setProperty(_COMBO_POPUP_GUARD_ID_PROPERTY, None)
        self._schedule_combo_popup_guard_end(token, _COMBO_POPUP_REBUILD_GRACE_MS)

    def _begin_combo_popup_guard(self) -> int:
        token = self._next_combo_popup_guard_token
        self._next_combo_popup_guard_token += 1
        self._combo_popup_guard_tokens.add(token)
        return token

    def _schedule_combo_popup_guard_end(self, token: int, delay_ms: int) -> None:
        def discard_guard() -> None:
            self._combo_popup_guard_tokens.discard(token)

        erlab.interactive.utils.single_shot(
            self,
            delay_ms,
            discard_guard,
        )

    def _remove_operation_list_event_filter(self) -> None:
        viewport = self._operation_list_viewport
        self._operation_list_viewport = None
        self._operation_multi_select_event = False
        if viewport is not None and erlab.interactive.utils.qt_is_valid(self, viewport):
            viewport.removeEventFilter(self)

    def _disconnect_step_clipboard(self) -> None:
        clipboard = self._connected_step_clipboard
        self._connected_step_clipboard = None
        if clipboard is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                clipboard.dataChanged.disconnect(self._update_step_action_buttons)

    def _source_data_history_state(self) -> dict[str, xr.DataArray]:
        return dict(self._source_data)

    def _restore_source_data_history_state(
        self, source_data: Mapping[str, xr.DataArray]
    ) -> None:
        self._source_data = dict(source_data)
        self._mark_preview_pixmap_stale()

    def _reset_history_stack(self) -> None:
        self._prev_states.clear()
        self._next_states.clear()
        self._prev_source_data_states.clear()
        self._next_source_data_states.clear()
        self._prev_states.append(self.tool_status)
        self._prev_source_data_states.append(self._source_data_history_state())
        self._update_history_actions()

    @QtCore.Slot()
    def _write_state(self, *_args: typing.Any) -> None:
        if not self._write_history:
            return
        curr_state = self.tool_status
        last_state = self._prev_states[-1] if self._prev_states else None
        if not self._history_state_equal(last_state, curr_state):
            self._prev_states.append(curr_state)
            self._prev_source_data_states.append(self._source_data_history_state())
            self._next_states.clear()
            self._next_source_data_states.clear()
            self._update_history_actions()

    @QtCore.Slot()
    def _replace_last_state(self, *_args: typing.Any) -> None:
        if not self._write_history:
            return
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

    def _clear_operation_multi_select_event(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        self._operation_multi_select_event = False

    @staticmethod
    def _operation_modifiers_enable_multi_selection(
        modifiers: QtCore.Qt.KeyboardModifier,
    ) -> bool:
        multi_modifiers = (
            QtCore.Qt.KeyboardModifier.ShiftModifier
            | QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        )
        return bool(modifiers & multi_modifiers)

    @QtCore.Slot(QtWidgets.QListWidgetItem)
    def _operation_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._updating_controls:
            return
        operation_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        for index, operation in enumerate(self._recipe.operations):
            if operation.operation_id == operation_id:
                updated = operation.model_copy(
                    update={
                        "enabled": item.checkState() == QtCore.Qt.CheckState.Checked,
                    }
                )
                operations = list(self._recipe.operations)
                operations[index] = updated
                self._recipe = self._recipe.model_copy(
                    update={"operations": tuple(operations)}
                )
                if index == self.operation_list.currentRow():
                    self._sync_axes_selector()
                    self._update_source_status(updated)
                self._refresh_step_section_button_texts()
                _render_preview(self)
                self.sigInfoChanged.emit()
                self._write_state()
                return

    @QtCore.Slot()
    def _target_current_operation_all_axes(self) -> None:
        current = self._current_operation()
        if current is None:
            return
        index, operation = current
        if not _registry.spec_for(operation.kind).uses_axes(operation):
            return
        if self._recipe.setup.layout_mode == "gridspec":
            selection = operation.axes.model_copy(
                update={
                    "axes_ids": _gridspec_all_axes_ids(self._recipe.setup),
                    "expression": "",
                }
            )
            self._replace_operation(
                index, operation.model_copy(update={"axes": selection})
            )
            return
        selection = operation.axes.model_copy(
            update={"axes": _all_axes(self._recipe.setup), "expression": ""}
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
        if self._recipe.setup.layout_mode == "gridspec":
            axes_ids = _gridspec_valid_axes_ids(
                self._recipe.setup, operation.axes.axes_ids
            )
            if not axes_ids:
                axes_ids = _gridspec_all_axes_ids(self._recipe.setup)[:1]
            selection = operation.axes.model_copy(
                update={"axes_ids": axes_ids, "expression": ""}
            )
            self._replace_operation(
                index, operation.model_copy(update={"axes": selection})
            )
            return
        axes = operation.axes.valid_axes(self._recipe.setup)
        if not axes:
            axes = ((0, 0),)
        selection = operation.axes.model_copy(update={"axes": axes, "expression": ""})
        self._replace_operation(index, operation.model_copy(update={"axes": selection}))

    def _source_names(self) -> tuple[str, ...]:
        names = tuple(source.name for source in self._recipe.sources)
        if names:
            return names
        return tuple(self._source_data)

    def _source_by_name(self) -> dict[str, FigureSourceState]:
        return {source.name: source for source in self._recipe.sources}

    def _source_display_name(self, name: str) -> str:
        sources = self._source_by_name()
        source = sources.get(name)
        duplicate_labels = _source_duplicate_labels(tuple(sources.values()))
        return _source_display_label(
            source,
            name,
            disambiguate=source is not None
            and (source.label.strip() or name) in duplicate_labels,
        )

    def _source_display_names(self, names: Sequence[str]) -> tuple[str, ...]:
        return tuple(self._source_display_name(name) for name in names)

    def _source_tooltip(self, name: str) -> str:
        return _source_display_tooltip(self._source_by_name().get(name), name)

    def _selected_axes_state(self) -> FigureAxesSelectionState:
        if self._recipe.setup.layout_mode == "gridspec":
            axes_ids = self.gridspec_axes_selector.selected_axes_ids()
            if not axes_ids:
                axes_ids = _gridspec_all_axes_ids(self._recipe.setup)[:1]
            return FigureAxesSelectionState(axes_ids=axes_ids)
        axes = self.axes_selector.selected_axes()
        if not axes:
            axes = ((0, 0),)
        return FigureAxesSelectionState(
            axes=axes,
            expression=self.axes_expression_edit.text().strip(),
        )

    @QtCore.Slot(object, str)
    def _gridspec_region_created(self, span_obj: object, kind_text: str) -> None:
        kind = typing.cast(
            'typing.Literal["axes", "grid"]',
            kind_text,
        )
        self._add_gridspec_region(
            typing.cast("FigureGridSpecSpanState", span_obj), kind
        )

    @QtCore.Slot(str, object)
    def _gridspec_region_changed(self, region_id: str, span_obj: object) -> None:
        span = typing.cast("FigureGridSpecSpanState", span_obj)
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        if grid is None or not _gridspec_region_valid(grid, span):
            self.gridspec_status_label.setText("Region is outside the active grid.")
            return
        if _gridspec_region_overlaps(
            grid,
            span,
            ignore_axes_id=region_id,
            ignore_grid_id=region_id,
        ):
            self.gridspec_status_label.setText("Regions cannot overlap.")
            return

        def update_grid(grid_state: FigureGridSpecGridState) -> FigureGridSpecGridState:
            axes = tuple(
                axis.model_copy(update={"span": span})
                if axis.axes_id == region_id
                else axis
                for axis in grid_state.axes
            )
            children = tuple(
                child.model_copy(update={"span": span})
                if child.grid_id == region_id
                else child
                for child in grid_state.child_grids
            )
            return grid_state.model_copy(update={"axes": axes, "child_grids": children})

        setup = _gridspec_replace_grid(
            self._recipe.setup, self._active_gridspec_grid_id, update_grid
        )
        self._apply_gridspec_setup(setup, selected_region_id=region_id)

    @QtCore.Slot(str, str)
    def _gridspec_region_selected(self, region_id: str, _kind: str) -> None:
        self.gridspec_layout_widget.set_selected_region(region_id)
        self._refresh_gridspec_region_controls()

    @QtCore.Slot(str)
    def _gridspec_open_grid(self, grid_id: str) -> None:
        if _gridspec_grid_by_id(self._recipe.setup, grid_id) is None:
            return
        self._active_gridspec_grid_id = grid_id
        self._updating_controls = True
        try:
            self._sync_active_grid_controls(self._recipe.setup)
        finally:
            self._updating_controls = False
        self._refresh_gridspec_editor()

    @QtCore.Slot()
    def _gridspec_open_selected_grid(self) -> None:
        region_id = self.gridspec_layout_widget.selected_region_id()
        if region_id:
            self._gridspec_open_grid(region_id)

    @QtCore.Slot()
    def _gridspec_open_parent_grid(self) -> None:
        path = _gridspec_grid_path(self._recipe.setup, self._active_gridspec_grid_id)
        if len(path) > 1:
            self._gridspec_open_grid(path[-2].grid_id)

    @QtCore.Slot()
    def _gridspec_delete_selected_region(self) -> None:
        region_id = self.gridspec_layout_widget.selected_region_id()
        if not region_id:
            return
        selected_region_id = self._nearest_gridspec_axes_after_delete(region_id)
        setup = _gridspec_remove_region(
            self._recipe.setup, self._active_gridspec_grid_id, region_id
        )
        self._apply_gridspec_setup(setup, selected_region_id=selected_region_id)

    def _nearest_gridspec_axes_after_delete(self, region_id: str) -> str:
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        if grid is None:
            return ""
        deleted_span: FigureGridSpecSpanState | None = None
        for axis in grid.axes:
            if axis.axes_id == region_id:
                deleted_span = axis.span
                break
        if deleted_span is None:
            for child in grid.child_grids:
                if child.grid_id == region_id:
                    deleted_span = child.span
                    break
        if deleted_span is None:
            return ""

        def span_center(span: FigureGridSpecSpanState) -> tuple[float, float]:
            return (
                (span.row_start + span.row_stop - 1) / 2.0,
                (span.col_start + span.col_stop - 1) / 2.0,
            )

        deleted_center = span_center(deleted_span)
        candidates = [axis for axis in grid.axes if axis.axes_id != region_id]
        if not candidates:
            return ""

        def sort_key(
            axis: FigureGridSpecAxesState,
        ) -> tuple[float, int, int, int, int]:
            center = span_center(axis.span)
            distance = abs(center[0] - deleted_center[0]) + abs(
                center[1] - deleted_center[1]
            )
            return (
                distance,
                axis.span.row_start,
                axis.span.col_start,
                axis.span.row_stop,
                axis.span.col_stop,
            )

        return min(candidates, key=sort_key).axes_id

    @QtCore.Slot()
    def _gridspec_region_label_changed(self) -> None:
        region_id = self.gridspec_layout_widget.selected_region_id()
        if not region_id:
            return
        name = self.gridspec_region_label_edit.text().strip()
        error = _gridspec_axis_variable_name_error(
            self._recipe.setup,
            region_id,
            name,
            reserved_names=self._source_names(),
        )
        if error:
            self._set_gridspec_variable_name_invalid(True)
            self.gridspec_status_label.setText(error)
            return
        self._set_gridspec_variable_name_invalid(False)
        setup = _gridspec_update_axis_variable_name(self._recipe.setup, region_id, name)
        self._apply_gridspec_setup(setup, selected_region_id=region_id)

    def _set_gridspec_variable_name_invalid(self, invalid: bool) -> None:
        self.gridspec_region_label_edit.setProperty("invalid", invalid)

    def _add_gridspec_region(
        self,
        span: FigureGridSpecSpanState,
        kind: typing.Literal["axes", "grid"],
    ) -> None:
        grid = _gridspec_grid_by_id(self._recipe.setup, self._active_gridspec_grid_id)
        if grid is None:
            return
        if not _gridspec_region_valid(grid, span):
            self.gridspec_status_label.setText("Region is outside the active grid.")
            return
        if _gridspec_region_overlaps(grid, span):
            self.gridspec_status_label.setText("Regions cannot overlap.")
            return

        if kind == "axes":
            region = FigureGridSpecAxesState(span=span)
            selected_region_id = region.axes_id

            def update_grid(
                grid_state: FigureGridSpecGridState,
            ) -> FigureGridSpecGridState:
                return grid_state.model_copy(
                    update={"axes": (*grid_state.axes, region)}
                )

        else:
            region = FigureGridSpecGridState(span=span)
            selected_region_id = region.grid_id

            def update_grid(
                grid_state: FigureGridSpecGridState,
            ) -> FigureGridSpecGridState:
                return grid_state.model_copy(
                    update={"child_grids": (*grid_state.child_grids, region)}
                )

        setup = _gridspec_replace_grid(
            self._recipe.setup, self._active_gridspec_grid_id, update_grid
        )
        self._apply_gridspec_setup(setup, selected_region_id=selected_region_id)

    def _apply_gridspec_setup(
        self, setup: FigureSubplotsState, *, selected_region_id: str
    ) -> None:
        self._recipe = self._recipe.model_copy(update={"setup": setup})
        self._updating_controls = True
        try:
            self._sync_active_grid_controls(setup)
        finally:
            self._updating_controls = False
        self._refresh_gridspec_editor()
        self.gridspec_layout_widget.set_selected_region(selected_region_id)
        self._refresh_gridspec_region_controls()
        self._sync_axes_selector()
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._update_operation_editor()
        _render_preview(self)
        self.sigInfoChanged.emit()
        self._write_state()

    @QtCore.Slot()
    def _show_add_step_menu(self) -> None:
        self.add_step_menu.popup(
            self.add_step_button.mapToGlobal(
                QtCore.QPoint(0, self.add_step_button.height())
            )
        )

    @QtCore.Slot(QtCore.QPoint)
    def _show_operation_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu("Recipe Steps", self.operation_list)
        self._operation_context_menu = menu

        copy_action = QtGui.QAction("Copy", menu)
        copy_action.setObjectName("figureComposerContextCopyStepsAction")
        copy_action.setEnabled(bool(self._selected_operation_indices()))
        copy_action.triggered.connect(self._copy_selected_operations)
        menu.addAction(copy_action)

        cut_action = QtGui.QAction("Cut", menu)
        cut_action.setObjectName("figureComposerContextCutStepsAction")
        cut_action.setEnabled(bool(self._selected_operation_indices()))
        cut_action.triggered.connect(self._cut_selected_operations)
        menu.addAction(cut_action)

        paste_action = QtGui.QAction("Paste", menu)
        paste_action.setObjectName("figureComposerContextPasteStepsAction")
        paste_action.setEnabled(self._clipboard_step_payload() is not None)
        paste_action.triggered.connect(self._paste_operations_from_clipboard)
        menu.addAction(paste_action)

        menu.addSeparator()
        duplicate_action = QtGui.QAction("Duplicate", menu)
        duplicate_action.setObjectName("figureComposerContextDuplicateStepAction")
        duplicate_action.setEnabled(self.duplicate_operation_button.isEnabled())
        duplicate_action.triggered.connect(self._duplicate_current_operation)
        menu.addAction(duplicate_action)

        move_up_action = QtGui.QAction("Up", menu)
        move_up_action.setObjectName("figureComposerContextMoveStepUpAction")
        move_up_action.setEnabled(self.move_operation_up_button.isEnabled())
        move_up_action.triggered.connect(self._move_current_operation_up)
        menu.addAction(move_up_action)

        move_down_action = QtGui.QAction("Down", menu)
        move_down_action.setObjectName("figureComposerContextMoveStepDownAction")
        move_down_action.setEnabled(self.move_operation_down_button.isEnabled())
        move_down_action.triggered.connect(self._move_current_operation_down)
        menu.addAction(move_down_action)

        remove_action = QtGui.QAction("Delete", menu)
        remove_action.setObjectName("figureComposerContextDeleteStepAction")
        remove_action.setEnabled(self.remove_operation_button.isEnabled())
        remove_action.triggered.connect(self._remove_current_operation)
        menu.addAction(remove_action)

        viewport = self.operation_list.viewport()
        if viewport is not None:  # pragma: no branch
            menu.popup(viewport.mapToGlobal(position))

    def _add_operation(self, action_id: str) -> None:
        operation = _registry.spec_for_action(action_id).create_operation(self)
        operations = (*self._recipe.operations, operation)
        self._recipe = self._recipe.model_copy(update={"operations": operations})
        self._refresh_operation_list()
        self.operation_list.setCurrentRow(len(operations) - 1)
        _render_preview(self)
        self.sigInfoChanged.emit()
        self._write_state()

    @staticmethod
    def _operation_source_names(operation: FigureOperationState) -> tuple[str, ...]:
        names: list[str] = []
        for source_name in operation.sources:
            if source_name not in names:
                names.append(source_name)
        for selection in operation.map_selections:
            if selection.source not in names:
                names.append(selection.source)
        if operation.line_source is not None and operation.line_source not in names:
            names.append(operation.line_source)
        return tuple(names)

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
        operations = tuple(self._recipe.operations[index] for index in indices)
        source_names = tuple(
            dict.fromkeys(
                source_name
                for operation in operations
                for source_name in self._operation_source_names(operation)
            )
        )
        source_by_name = {source.name: source for source in self._recipe.sources}
        sources = tuple(
            source_by_name.get(
                source_name, FigureSourceState(name=source_name, label=source_name)
            )
            for source_name in source_names
        )
        source_data = {
            source_name: self._source_data[source_name].copy(deep=False)
            for source_name in source_names
            if source_name in self._source_data
        }
        clipboard = self._clipboard()
        if clipboard is None:
            return None
        clipboard.setMimeData(
            _FigureComposerStepMimeData(
                _step_clipboard_payload_text(operations, sources),
                _step_clipboard_code_text(self, operations),
                source_data,
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

    def _renamed_pasted_sources(
        self,
        sources: Sequence[FigureSourceState],
        source_data: Mapping[str, xr.DataArray],
        *,
        preserve_existing: bool = False,
    ) -> tuple[
        tuple[FigureSourceState, ...],
        dict[str, str],
        dict[str, xr.DataArray],
    ]:
        existing_source_names = {source.name for source in self._recipe.sources}
        reserved = {source.name for source in self._recipe.sources}
        reserved.update(self._source_data)
        rename_map: dict[str, str] = {}
        renamed_sources: list[FigureSourceState] = []
        renamed_source_data: dict[str, xr.DataArray] = {}
        seen_sources: set[str] = set()
        for source in sources:
            if source.name in seen_sources:
                continue
            seen_sources.add(source.name)
            if preserve_existing and source.name in reserved:
                rename_map[source.name] = source.name
                if source.name not in existing_source_names:
                    renamed_sources.append(source)
                    existing_source_names.add(source.name)
                if source.name in source_data and source.name not in self._source_data:
                    renamed_source_data[source.name] = source_data[source.name]
                continue
            pasted_name = source.name
            if pasted_name in reserved:
                stem = f"{source.name}_copy"
                pasted_name = stem
                suffix = 2
                while pasted_name in reserved:
                    pasted_name = f"{stem}_{suffix}"
                    suffix += 1
            reserved.add(pasted_name)
            rename_map[source.name] = pasted_name
            renamed_sources.append(source.model_copy(update={"name": pasted_name}))
            if source.name in source_data:
                renamed_source_data[pasted_name] = source_data[source.name]
        return tuple(renamed_sources), rename_map, renamed_source_data

    @staticmethod
    def _operation_with_renamed_sources(
        operation: FigureOperationState, rename_map: Mapping[str, str]
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {}
        if operation.sources:
            updates["sources"] = tuple(
                rename_map.get(source_name, source_name)
                for source_name in operation.sources
            )
        if operation.map_selections:
            updates["map_selections"] = tuple(
                selection.model_copy(
                    update={
                        "source": rename_map.get(selection.source, selection.source),
                    }
                )
                for selection in operation.map_selections
            )
        if operation.line_source is not None:
            updates["line_source"] = rename_map.get(
                operation.line_source, operation.line_source
            )
        if not updates:
            return operation
        return operation.model_copy(update=updates)

    @QtCore.Slot()
    def _paste_operations_from_clipboard(self) -> None:
        clipboard = self._clipboard()
        if clipboard is None:
            return
        mime = clipboard.mimeData()
        payload = _step_clipboard_payload(mime)
        if payload is None:
            return
        operations, sources, source_data = payload
        renamed_sources, rename_map, renamed_source_data = self._renamed_pasted_sources(
            sources,
            source_data,
            preserve_existing=(
                getattr(mime, "figure_composer_cut_source_tool_id", None)
                == self._step_clipboard_tool_id
            ),
        )
        pasted_operations = tuple(
            self._operation_with_renamed_sources(operation, rename_map).model_copy(
                update={"operation_id": uuid.uuid4().hex},
                deep=True,
            )
            for operation in operations
        )
        operation_list = list(self._recipe.operations)
        indices = self._selected_operation_indices()
        current = self._current_operation()
        if indices:
            insert_index = max(indices) + 1
        elif current is not None:
            insert_index = current[0] + 1
        else:
            insert_index = len(operation_list)
        operation_list[insert_index:insert_index] = pasted_operations

        source_names = {source.name for source in self._recipe.sources}
        source_list = list(self._recipe.sources)
        for source in renamed_sources:
            if source.name in source_names:
                continue
            source_names.add(source.name)
            source_list.append(source)
        self._source_data.update(renamed_source_data)
        self._recipe = self._recipe.model_copy(
            update={
                "sources": tuple(source_list),
                "operations": tuple(operation_list),
            }
        )
        self._refresh_source_list()
        self._finish_operation_structure_change(
            {operation.operation_id for operation in pasted_operations},
            pasted_operations[0].operation_id,
        )

    def _remove_operations_at_indices(self, indices: Sequence[int]) -> None:
        if not indices:
            return
        index_set = set(indices)
        operations = [
            operation
            for index, operation in enumerate(self._recipe.operations)
            if index not in index_set
        ]
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
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
        operations = list(self._recipe.operations)
        duplicates = [
            operations[index].model_copy(
                update={"operation_id": uuid.uuid4().hex},
                deep=True,
            )
            for index in indices
        ]
        insert_index = max(indices) + 1
        operations[insert_index:insert_index] = duplicates
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._finish_operation_structure_change(
            {operation.operation_id for operation in duplicates},
            duplicates[0].operation_id,
        )

    def _move_current_operation(self, offset: int) -> None:
        indices = self._selected_operation_indices()
        if not indices:
            return
        operations = list(self._recipe.operations)
        index_set = set(indices)
        selected_ids = {operations[index].operation_id for index in indices}
        moved = False
        if offset < 0:
            for index in indices:
                if index > 0 and index - 1 not in index_set:
                    operations[index - 1], operations[index] = (
                        operations[index],
                        operations[index - 1],
                    )
                    index_set.remove(index)
                    index_set.add(index - 1)
                    moved = True
        else:
            for index in reversed(indices):
                if index < len(operations) - 1 and index + 1 not in index_set:
                    operations[index + 1], operations[index] = (
                        operations[index],
                        operations[index + 1],
                    )
                    index_set.remove(index)
                    index_set.add(index + 1)
                    moved = True
        if not moved:
            return
        current = self._current_operation()
        current_id = (
            current[1].operation_id
            if current is not None and current[1].operation_id in selected_ids
            else next(
                operation.operation_id
                for operation in operations
                if operation.operation_id in selected_ids
            )
        )
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
        self._finish_operation_structure_change(selected_ids, current_id)

    def _finish_operation_structure_change(
        self, selected_ids: set[str], current_id: str | None
    ) -> None:
        self._refresh_operation_list()
        if selected_ids:
            self._set_selected_operation_ids_silent(selected_ids)
        if current_id is not None:
            for row, operation in enumerate(self._recipe.operations):
                if operation.operation_id == current_id:
                    self._set_current_operation_row_silent(row)
                    break
        self._sync_axes_selector()
        self._update_operation_editor()
        _render_preview(self)
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
        self._recipe = self._recipe.model_copy(
            update={"operations": (*self._recipe.operations, operation)}
        )
        self._apply_recipe_to_controls()
        self.operation_list.setCurrentRow(len(self._recipe.operations) - 1)
        _render_preview(self)
        self.sigInfoChanged.emit()
        self._write_state()

    def add_sources(
        self,
        sources: Sequence[FigureSourceState],
        source_data: Mapping[str, xr.DataArray],
    ) -> None:
        """Add or refresh recipe sources used by appended operations."""
        existing = {source.name: source for source in self._recipe.sources}
        for source in sources:
            existing[source.name] = source
        self._source_data.update(source_data)
        ordered_sources = tuple(existing[name] for name in existing)
        self._recipe = self._recipe.model_copy(update={"sources": ordered_sources})
        self._refresh_source_list()
        self._update_source_section()
        _render_preview(self)
        self.sigInfoChanged.emit()
        self._write_state()

    @staticmethod
    def _remove_posted_events_recursive(widget: QtWidgets.QWidget) -> None:
        for child in widget.findChildren(QtCore.QObject):
            QtCore.QCoreApplication.removePostedEvents(child)
        QtCore.QCoreApplication.removePostedEvents(widget)

    @staticmethod
    def _block_signals_recursive(widget: QtWidgets.QWidget) -> None:
        widget.blockSignals(True)
        for child in widget.findChildren(QtCore.QObject):
            if erlab.interactive.utils.qt_is_valid(child):
                child.blockSignals(True)

    def _retire_editor_widget(self, widget: QtWidgets.QWidget) -> None:
        if not erlab.interactive.utils.qt_is_valid(widget):
            return
        self._block_signals_recursive(widget)
        widget.setEnabled(False)
        widget.hide()
        if widget not in self._retired_editor_widgets:
            self._retired_editor_widgets.append(widget)
        self._queue_retired_editor_drain()

    def _queue_retired_editor_drain(self) -> None:
        if self._retired_editor_drain_pending:
            return
        self._retired_editor_drain_pending = True
        erlab.interactive.utils.single_shot(
            self,
            _RETIRED_EDITOR_DRAIN_DELAY_MS,
            self._drain_retired_editor_widgets,
        )

    def _drain_retired_editor_widgets(self) -> None:
        self._retired_editor_drain_pending = False
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        if (
            not self._retired_editor_widgets
            or self._operation_editor_rebuild_must_wait()
            or self._retired_editor_has_focus()
        ):
            if self._retired_editor_widgets:
                self._queue_retired_editor_drain()
            return

        retired_widgets = self._retired_editor_widgets
        self._retired_editor_widgets = []
        for widget in retired_widgets:
            if not erlab.interactive.utils.qt_is_valid(widget):
                continue
            self._remove_posted_events_recursive(widget)
            widget.setParent(None)
            widget.deleteLater()

    def _retired_editor_has_focus(self) -> bool:
        focus_widget = QtWidgets.QApplication.focusWidget()
        return (
            focus_widget is not None
            and erlab.interactive.utils.qt_is_valid(focus_widget)
            and any(
                widget is focus_widget or widget.isAncestorOf(focus_widget)
                for widget in self._retired_editor_widgets
                if erlab.interactive.utils.qt_is_valid(widget)
            )
        )

    def _clear_form_layout(self, layout: QtWidgets.QFormLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                self._retire_editor_widget(widget)

    def _clear_operation_editor(self) -> None:
        for page in self._operation_editor_pages:
            self.step_editor_stack.removeWidget(page)
            self._retire_editor_widget(page)
        self._operation_editor_pages.clear()

    def _clear_step_source_controls(self) -> None:
        self._clear_form_layout(self.step_source_controls_layout)

    def _new_step_form_page(
        self, object_name: str
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        page = QtWidgets.QWidget(self.step_editor_stack)
        page.setObjectName(object_name)
        layout = QtWidgets.QFormLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._operation_editor_pages.append(page)
        return page, layout

    def _set_step_sections(self, sections: Sequence[StepSection]) -> None:
        self._section_tab_stop_refs.clear()
        while self.step_navigator_layout.count():
            item = self.step_navigator_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                self._retire_editor_widget(widget)
        self.step_section_buttons.clear()
        self.step_section_keys = [section.key for section in sections]

        while self.step_editor_stack.count():
            self.step_editor_stack.removeWidget(self.step_editor_stack.widget(0))

        for index, section in enumerate(sections):
            self.step_editor_stack.addWidget(section.page)
            button = QtWidgets.QToolButton(self.step_navigator)
            button.setText(self._section_button_text(section.key, section.title))
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button.setObjectName(f"figureComposerSection_{section.key}")
            button.setProperty("section_title", section.title)
            button.setProperty("section_tooltip", section.tooltip)
            button.setToolTip(section.tooltip)
            button.clicked.connect(
                lambda _checked=False, key=section.key: self._select_step_section(key)
            )
            self.step_navigator_layout.insertWidget(index, button)
            self.step_section_buttons[section.key] = button
        self.step_navigator_layout.addStretch(1)

        key = (
            self._current_step_section_key
            if self._current_step_section_key in self.step_section_keys
            else self.step_section_keys[0]
            if self.step_section_keys
            else ""
        )
        if key:
            self._select_step_section(key)

    def _select_step_section(self, key: str) -> None:
        if key not in self.step_section_keys:
            return
        self._current_step_section_key = key
        index = self.step_section_keys.index(key)
        self.step_editor_stack.setCurrentIndex(index)
        for button_key, button in self.step_section_buttons.items():
            selected = button_key == key
            font = button.font()
            if font.bold() != selected:
                font.setBold(selected)
                button.setFont(font)
        self._queue_step_tab_order_refresh()

    def _refresh_step_section_button_texts(self) -> None:
        for key, button in self.step_section_buttons.items():
            title = button.property("section_title")
            if not isinstance(title, str):
                title = key
            button.setText(self._section_button_text(key, title))

    def _section_button_text(self, key: str, title: str) -> str:
        current = self._current_operation()
        if current is None:
            return title
        _index, operation = current
        summary = self._section_summary(key, operation)
        return f"{title}: {summary}" if summary else title

    def _section_summary(self, key: str, operation: FigureOperationState) -> str:
        return _registry.spec_for(operation.kind).section_summary(self, key, operation)

    @staticmethod
    def _accepts_tab_focus(widget: QtWidgets.QWidget) -> bool:
        return bool(widget.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus)

    def _first_editor_tab_stop(self) -> QtWidgets.QWidget | None:
        current_page = self.step_editor_stack.currentWidget()
        if current_page is None:
            return None
        cache_key = self._current_step_section_key
        cached_ref = self._section_tab_stop_refs.get(cache_key)
        if cached_ref is not None:
            cached_widget = cached_ref()
            if (
                cached_widget is not None
                and erlab.interactive.utils.qt_is_valid(cached_widget)
                and cached_widget.isEnabled()
                and self._accepts_tab_focus(cached_widget)
                and (
                    cached_widget is current_page
                    or current_page.isAncestorOf(cached_widget)
                )
            ):
                return cached_widget
            self._section_tab_stop_refs.pop(cache_key, None)
        if (
            erlab.interactive.utils.qt_is_valid(current_page)
            and current_page.isEnabled()
            and self._accepts_tab_focus(current_page)
        ):
            self._section_tab_stop_refs[cache_key] = weakref.ref(current_page)
            return current_page
        for widget in current_page.findChildren(QtWidgets.QWidget):
            if (
                erlab.interactive.utils.qt_is_valid(widget)
                and widget.isEnabled()
                and self._accepts_tab_focus(widget)
            ):
                self._section_tab_stop_refs[cache_key] = weakref.ref(widget)
                return widget
        return None

    def _refresh_step_tab_order(self) -> None:
        buttons = list(self.step_section_buttons.values())
        if not buttons:
            return
        preceding_widgets = (
            self.add_step_button,
            self.copy_operation_button,
            self.cut_operation_button,
            self.paste_operation_button,
            self.duplicate_operation_button,
            self.move_operation_up_button,
            self.move_operation_down_button,
            self.remove_operation_button,
            self.operation_list,
        )
        tab_chain = [*preceding_widgets, *buttons]
        next_widget = self._first_editor_tab_stop()
        if next_widget is not None:
            tab_chain.append(next_widget)
        for index, widget in enumerate(tab_chain[:-1]):
            QtWidgets.QWidget.setTabOrder(widget, tab_chain[index + 1])

    def _queue_step_tab_order_refresh(self) -> None:
        if self._step_tab_order_update_pending or self._closing:
            return
        self._step_tab_order_update_pending = True
        erlab.interactive.utils.single_shot(
            self,
            0,
            self._run_queued_step_tab_order_refresh,
        )

    def _run_queued_step_tab_order_refresh(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        self._step_tab_order_update_pending = False
        self._refresh_step_tab_order()

    def _selected_sources_for_operation(
        self, operation: FigureOperationState
    ) -> tuple[str, ...]:
        return _registry.spec_for(operation.kind).source_names(operation)

    def _update_source_status(self, operation: FigureOperationState | None) -> None:
        if operation is None:
            self.source_status_label.setText("Select a step to choose data sources.")
            return
        if (input_error := self._operation_input_error_text(operation)) is not None:
            self.source_status_label.setText(f"Invalid input: {input_error}")
            return
        if (
            render_error := self._operation_render_errors.get(operation.operation_id)
        ) is not None:
            self.source_status_label.setText(f"Render error: {render_error}")
            return
        selected_sources = self._selected_sources_for_operation(operation)
        missing = [
            source for source in selected_sources if source not in self._source_data
        ]
        if missing:
            self.source_status_label.setText(
                "Missing sources: " + ", ".join(self._source_display_names(missing))
            )
        elif selected_sources:
            self.source_status_label.setText(
                "Selected sources: "
                + ", ".join(self._source_display_names(selected_sources))
            )
        else:
            self.source_status_label.setText("This step does not read a data source.")

    def _update_source_section(self) -> None:
        self._clear_step_source_controls()
        current = self._current_operation()
        if current is None:
            self._update_source_status(None)
            return
        _index, operation = current

        _registry.spec_for(operation.kind).build_source_editor(self, operation)
        self._update_source_status(operation)

    def _update_operation_editor(self) -> None:
        self._operation_editor_generation += 1
        section_key = self._current_step_section_key
        self._clear_operation_editor()
        self._update_source_section()
        current = self._current_operation()
        self._update_step_action_buttons()
        selected_indices = self._selected_operation_indices()
        if len(selected_indices) > 1 and not self._selected_operations_are_compatible():
            page, layout = self._new_step_form_page(
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
            page, layout = self._new_step_form_page("figureComposerNoStepPage")
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
                    self.step_sources_page,
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
        sections.extend(spec.build_editor_sections(self, operation))
        self._current_step_section_key = section_key
        self._set_step_sections(sections)

    def _update_operation_editor_safely(self) -> None:
        if (
            self._operation_editor_rebuild_must_wait()
            or self._operation_editor_sender_requires_deferred_rebuild()
        ):
            self._queue_operation_editor_update()
            return
        self._update_operation_editor()

    def _operation_editor_sender_requires_deferred_rebuild(self) -> bool:
        sender = self._active_editor_signal_widget
        if sender is None:
            qt_sender = self.sender()
            sender = qt_sender if isinstance(qt_sender, QtWidgets.QWidget) else None
        if sender is None or not erlab.interactive.utils.qt_is_valid(sender):
            return False
        editor_roots = (
            self.step_editor_stack,
            self.step_source_controls,
            self.source_list,
            self.target_axes_page,
            self.step_navigator,
        )
        return any(root is sender or root.isAncestorOf(sender) for root in editor_roots)

    def _queue_operation_editor_update(self) -> None:
        if self._operation_editor_update_pending:
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
        if self._operation_editor_rebuild_must_wait():
            self._schedule_queued_operation_editor_update()
            return
        self._operation_editor_update_pending = False
        self._update_operation_editor()

    def _operation_editor_rebuild_must_wait(self) -> bool:
        if self._combo_popup_guard_tokens or self._tracked_combo_popup_is_visible():
            return True
        popup = QtWidgets.QApplication.activePopupWidget()
        return popup is not None and erlab.interactive.utils.qt_is_valid(popup)

    def _line_edit(
        self, text: str, *, parent: QtWidgets.QWidget | None = None
    ) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(parent or self.operation_editor)
        self._mark_editor_control(edit)
        edit.setText(text)
        return edit

    def _mark_editor_control(self, widget: QtWidgets.QWidget) -> None:
        widget.setProperty(
            "figure_composer_editor_generation",
            self._operation_editor_generation,
        )
        if isinstance(widget, QtWidgets.QComboBox):
            self._track_combo_interaction(widget)
        for combo in widget.findChildren(QtWidgets.QComboBox):
            self._track_combo_interaction(combo)

    def _editor_control_signal_allowed(self, widget: QtWidgets.QWidget) -> bool:
        return (
            not self._updating_controls
            and erlab.interactive.utils.qt_is_valid(widget)
            and widget.property("figure_composer_editor_generation")
            == self._operation_editor_generation
        )

    def _connect_editor_signal(
        self,
        widget: QtWidgets.QWidget,
        signal: typing.Any,
        callback: Callable[..., None],
    ) -> None:
        """Connect a recipe-editor signal with generation and lifetime guards."""
        self._mark_editor_control(widget)
        widget_ref = weakref.ref(widget)

        def guarded_callback(*args: typing.Any) -> None:
            guarded_widget = widget_ref()
            if guarded_widget is None or not self._editor_control_signal_allowed(
                guarded_widget
            ):
                return
            input_error_key = self._editor_input_error_key(guarded_widget)
            operation_ids = self._editable_operation_ids_for_error()
            previous_widget = self._active_editor_signal_widget
            self._active_editor_signal_widget = guarded_widget
            try:
                callback(*args)
            except FigureComposerInputError as exc:
                self._record_editor_input_error_for_key(
                    input_error_key, operation_ids, exc
                )
            else:
                self._clear_editor_input_error_for_key(input_error_key, operation_ids)
            finally:
                self._active_editor_signal_widget = previous_widget

        signal.connect(guarded_callback)

    def _connect_line_edit_finished(
        self,
        edit: QtWidgets.QLineEdit,
        callback: Callable[[str], None],
    ) -> None:
        """Connect an editable text control with mixed-value protection."""
        LineEditControlAdapter(edit).connect_commit(
            self._connect_editor_signal,
            callback,
        )

    def _connect_plain_text_changed(
        self,
        edit: QtWidgets.QPlainTextEdit,
        callback: Callable[[str], None],
    ) -> None:
        """Connect an editable plain-text control with mixed-value protection."""
        PlainTextControlAdapter(edit).connect_commit(
            self._connect_editor_signal,
            callback,
        )

    def _connect_value_signal(
        self,
        widget: QtWidgets.QWidget,
        signal: typing.Any,
        value_getter: Callable[..., typing.Any],
        callback: Callable[[typing.Any], None],
        *,
        unchanged_mixed: Callable[[], bool] | None = None,
    ) -> None:
        """Connect a custom editor widget signal through the adapter contract."""
        SignalValueControlAdapter(
            widget,
            signal,
            value_getter,
            unchanged_mixed=unchanged_mixed,
        ).connect_commit(self._connect_editor_signal, callback)

    def _mixed_value_widget(
        self,
        widget: QtWidgets.QWidget,
        *,
        mixed: bool,
        parent: QtWidgets.QWidget | None = None,
    ) -> QtWidgets.QWidget:
        """Wrap non-placeholder controls with the standard mixed-value marker."""
        if not mixed:
            return widget
        container = QtWidgets.QWidget(parent or widget.parentWidget())
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        marker = QtWidgets.QLabel(_MIXED_VALUES_TEXT, container)
        marker.setObjectName("figureComposerMixedValueMarker")
        marker.setEnabled(False)
        layout.addWidget(widget, 1)
        layout.addWidget(marker)
        return container

    def _combo(
        self,
        values: Sequence[str],
        current: str | None,
        changed: Callable[[str], None],
        *,
        parent: QtWidgets.QWidget | None = None,
        mixed: bool = False,
        enabled: bool = True,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self.operation_editor)
        self._mark_editor_control(combo)
        combo.addItems(list(values))
        adapter = ComboBoxControlAdapter(combo)
        if mixed:
            adapter.set_mixed(True)
        elif current is not None:
            self._set_combo_value(combo, current)
        combo.setEnabled(enabled)
        adapter.connect_commit(
            self._connect_editor_signal,
            changed,
        )
        return combo

    def _source_combo(
        self,
        values: Sequence[str],
        current: str | None,
        changed: Callable[[str | None], None],
        *,
        parent: QtWidgets.QWidget | None = None,
        mixed: bool = False,
        enabled: bool = True,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self.operation_editor)
        self._mark_editor_control(combo)
        adapter = ComboBoxDataControlAdapter(combo)
        if mixed:
            combo.addItem(_MIXED_VALUES_TEXT, _MIXED_VALUE)
        for value in values:
            combo.addItem(self._source_display_name(value), value)
            combo.setItemData(
                combo.count() - 1,
                self._source_tooltip(value),
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        if current is not None and current not in values and not mixed:
            combo.addItem(self._source_display_name(current), current)
            combo.setItemData(
                combo.count() - 1,
                self._source_tooltip(current),
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        if mixed:
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
            combo.setCurrentIndex(0)
        elif current is not None:
            for index in range(combo.count()):
                if combo.itemData(index) == current:
                    combo.setCurrentIndex(index)
                    break
        combo.setEnabled(enabled)
        adapter.connect_commit(
            self._connect_editor_signal,
            lambda value: changed(typing.cast("str | None", value)),
        )
        return combo

    def _optional_name_combo(
        self,
        values: Sequence[str],
        current: str | None,
        none_label: str,
        changed: Callable[[str | None], None],
        *,
        parent: QtWidgets.QWidget | None = None,
        mixed: bool = False,
        enabled: bool = True,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent or self.operation_editor)
        self._mark_editor_control(combo)
        adapter = ComboBoxDataControlAdapter(combo)
        if mixed:
            combo.addItem(_MIXED_VALUES_TEXT, _MIXED_VALUE)
        combo.addItem(none_label, None)
        for value in values:
            combo.addItem(value, value)
        if current is not None and current not in values and not mixed:
            combo.addItem(current, current)
        if mixed:
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
            combo.setCurrentIndex(0)
        else:
            for index in range(combo.count()):
                if combo.itemData(index) == current:
                    combo.setCurrentIndex(index)
                    break
        combo.setEnabled(enabled)
        adapter.connect_commit(
            self._connect_editor_signal,
            lambda value: changed(typing.cast("str | None", value)),
        )
        return combo

    def _check_box(
        self,
        checked: bool,
        changed: Callable[[bool], None],
        *,
        parent: QtWidgets.QWidget | None = None,
        mixed: bool = False,
    ) -> QtWidgets.QCheckBox:
        check = QtWidgets.QCheckBox(parent or self.operation_editor)
        self._mark_editor_control(check)
        adapter = CheckBoxControlAdapter(check)
        if mixed:
            adapter.set_mixed(True)
            adapter.connect_commit(self._connect_editor_signal, changed)
        else:
            check.setChecked(checked)
            adapter.connect_commit(self._connect_editor_signal, changed)
        return check

    @staticmethod
    def _wrapped_tooltip(tooltip: str) -> str:
        if "\n" in tooltip:
            return tooltip
        return "\n".join(textwrap.wrap(tooltip, width=58, break_long_words=False))

    @staticmethod
    def _add_form_row(
        layout: QtWidgets.QFormLayout,
        label: str,
        widget: QtWidgets.QWidget,
        tooltip: str,
    ) -> None:
        tooltip = FigureComposerTool._wrapped_tooltip(tooltip)
        widget.setToolTip(tooltip)
        layout.addRow(label, widget)
        label_widget = layout.labelForField(widget)
        if label_widget is not None:
            label_widget.setToolTip(tooltip)

    @staticmethod
    def _add_compound_form_row(
        layout: QtWidgets.QFormLayout,
        label: str,
        controls: Sequence[tuple[str, QtWidgets.QWidget, str]],
        tooltip: str,
    ) -> QtWidgets.QWidget:
        row_widget = QtWidgets.QWidget(layout.parentWidget())
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for control_label, widget, control_tooltip in controls:
            control_tooltip = FigureComposerTool._wrapped_tooltip(control_tooltip)
            label_widget = QtWidgets.QLabel(control_label, row_widget)
            label_widget.setBuddy(widget)
            label_widget.setToolTip(control_tooltip)
            widget.setToolTip(control_tooltip)
            row_layout.addWidget(label_widget)
            row_layout.addWidget(widget, 1)
        row_tooltip = FigureComposerTool._wrapped_tooltip(tooltip)
        row_widget.setToolTip(row_tooltip)
        layout.addRow(label, row_widget)
        label_widget = layout.labelForField(row_widget)
        if label_widget is not None:
            label_widget.setToolTip(row_tooltip)
        return row_widget

    def generated_code(self) -> str:
        return erlab.interactive._figurecomposer._codegen.generated_code(self)

    @QtCore.Slot()
    def copy_code(self) -> None:
        if self._warn_invalid_operation_targets():
            return
        try:
            code = self.generated_code()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot Copy Figure Code", str(exc))
            return
        erlab.interactive.utils.copy_to_clipboard(code)

    @QtCore.Slot()
    def export_figure(self) -> None:
        if self._warn_invalid_operation_targets():
            return
        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Figure",
            "",
            "Images (*.png *.pdf *.svg);;All files (*)",
        )
        if not filename:
            return
        with _rendered_output_figure(self) as figure, _figure_draw_context():
            figure.savefig(
                filename,
                dpi=self._recipe.export.dpi,
                transparent=self._recipe.export.transparent,
                bbox_inches=self._recipe.export.bbox_inches,
            )

    @property
    def tool_data(self) -> xr.DataArray:
        if self._recipe.primary_source in self._source_data:
            return self._source_data[self._recipe.primary_source]
        return next(iter(self._source_data.values()))

    @property
    def tool_status(self) -> FigureRecipeState:
        return self._recipe

    @tool_status.setter
    def tool_status(self, status: FigureRecipeState) -> None:
        operations = tuple(
            _registry.spec_for(operation.kind).loaded_operation(operation)
            for operation in status.operations
        )
        self._recipe = status.model_copy(update={"operations": operations})
        self._ensure_primary_source_data()
        self._apply_recipe_to_controls()
        _render_preview(self)

    def set_source_data(self, source_data: Mapping[str, xr.DataArray]) -> None:
        self._source_data = dict(source_data)
        self._mark_preview_pixmap_stale()

    def rebase_source_node_uids(self, uid_map: Mapping[str, str]) -> None:
        if not uid_map:
            return
        changed = False
        sources: list[FigureSourceState] = []
        for source in self._recipe.sources:
            updates: dict[str, typing.Any] = {}
            if source.node_uid is not None and source.node_uid in uid_map:
                updates["node_uid"] = uid_map[source.node_uid]
            if source.provenance_spec is not None:
                try:
                    rebased = provenance.rebase_script_input_node_uids(
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
        self._recipe = self._recipe.model_copy(update={"sources": tuple(sources)})
        self.sigInfoChanged.emit()

    def _ensure_primary_source_data(self) -> None:
        if self._recipe.primary_source in self._source_data or not self._source_data:
            return
        recipe_sources = {source.name for source in self._recipe.sources}
        fallback_name, fallback_data = next(iter(self._source_data.items()))
        self._source_data[self._recipe.primary_source] = fallback_data
        if fallback_name not in recipe_sources:
            del self._source_data[fallback_name]

    def _recipe_source(self, source_name: str) -> FigureSourceState | None:
        for source in self._recipe.sources:
            if source.name == source_name:
                return source
        return None

    def _source_reference_payload(
        self, source_name: str
    ) -> dict[str, typing.Any] | None:
        if not self._save_tool_data_references or source_name not in self._source_data:
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
        return payload

    def _tool_data_reference_payload(
        self, variable_name: str, data: xr.DataArray
    ) -> dict[str, typing.Any] | None:
        del data
        if variable_name == erlab.interactive.utils._SAVED_TOOL_DATA_NAME:
            return self._source_reference_payload(self._recipe.primary_source)
        return self._source_reference_payload(variable_name)

    def _persistence_data_items(self) -> Mapping[str, xr.DataArray]:
        items = {erlab.interactive.utils._SAVED_TOOL_DATA_NAME: self.tool_data}
        for source_name, data in self._source_data.items():
            if source_name == self._recipe.primary_source:
                continue
            if source_name == erlab.interactive.utils._SAVED_TOOL_DATA_NAME:
                raise ValueError(
                    "Figure source aliases cannot use the reserved saved-tool data name"
                )
            items[source_name] = data
        return items

    def _restore_persistence_data_items(
        self, data_items: Mapping[str, xr.DataArray], ds: xr.Dataset
    ) -> None:
        del ds
        source_data = dict(self._source_data)
        primary_data = data_items.get(erlab.interactive.utils._SAVED_TOOL_DATA_NAME)
        changed = False
        if primary_data is not None and self._recipe.primary_source not in source_data:
            source_data[self._recipe.primary_source] = primary_data
            changed = True
        for source in self._recipe.sources:
            if source.name == self._recipe.primary_source:
                continue
            source_data_item = data_items.get(source.name)
            if source_data_item is None:
                continue
            source_data[source.name] = source_data_item
            changed = True
        if not changed:
            return
        self.set_source_data(source_data)
        self._apply_recipe_to_controls()
        _render_preview(self, show_window=False)

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
        if not self._recipe.operations:
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

    def _canvas_preview_pixmap(self) -> QtGui.QPixmap | None:
        if self._closing or not erlab.interactive.utils.qt_is_valid(self):
            return None
        window = self._figure_window
        if window is None or not erlab.interactive.utils.qt_is_valid(window):
            return None
        if not window.isVisible():
            return None

        try:
            canvas = window.canvas
            if not erlab.interactive.utils.qt_is_valid(canvas):
                return None
            with _figure_style_context():
                canvas.draw()
            width, height = canvas.get_width_height(physical=True)
            if width <= 0 or height <= 0:
                return None
            image = QtGui.QImage(
                canvas.buffer_rgba(),
                width,
                height,
                QtGui.QImage.Format.Format_RGBA8888,
            )
            return QtGui.QPixmap.fromImage(image.copy())
        except Exception:
            return None

    def _fallback_preview_pixmap(self) -> QtGui.QPixmap | None:
        if self._closing or not erlab.interactive.utils.qt_is_valid(self):
            return None
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        with _figure_style_context():
            figure = Figure(
                figsize=self._recipe.setup.figsize,
                dpi=self._recipe.setup.dpi,
                layout=typing.cast("typing.Any", self._recipe.setup.layout),
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
        if not self._recipe.operations:
            self._clear_preview_pixmap_cache(stale=False)
            return None
        preview = self._canvas_preview_pixmap()
        if preview is None and allow_offscreen:
            preview = self._fallback_preview_pixmap()
        if preview is None:
            return self._preview_pixmap_cache
        self._preview_pixmap_cache = preview
        self._preview_pixmap_generation += 1
        self._preview_thumbnail_cache.clear()
        self._preview_pixmap_stale = False
        return preview

    def preview_thumbnail_pixmap(self, size: QtCore.QSize) -> QtGui.QPixmap | None:
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
        return self._recipe.sources

    def source_data(self) -> dict[str, xr.DataArray]:
        return dict(self._source_data)

    def current_provenance_spec(self) -> provenance.ToolProvenanceSpec | None:
        script_inputs = tuple(
            script_input
            for source in self._recipe.sources
            if (script_input := source.to_script_input()) is not None
        )
        if not script_inputs:
            return None
        return provenance.script(
            erlab.interactive._figurecomposer._provenance._figure_build_operation(self),
            start_label="Figure",
            active_name="fig",
            script_inputs=script_inputs,
        )

    def set_missing_sources(self, names: set[str]) -> None:
        if not names:
            return
        for source in self._recipe.sources:
            if source.name in names:
                self._source_data.pop(source.name, None)
        self._refresh_source_list()
        self._update_source_section()
        _render_preview(self)

    def refresh_from_sources(self, source_data: Mapping[str, xr.DataArray]) -> None:
        self._source_data.update(source_data)
        self._refresh_source_list()
        self._update_source_section()
        _render_preview(self)
