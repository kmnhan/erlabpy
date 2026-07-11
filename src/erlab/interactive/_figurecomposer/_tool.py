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
import textwrap
import traceback
import typing
import unicodedata
import uuid
import weakref

# Matplotlib's Qt backend should see the qtpy-selected binding first.
# isort: off
from qtpy import QtCore, QtGui, QtWidgets

from matplotlib.figure import Figure
# isort: on

import numpy as np
import pydantic

import erlab
import erlab.interactive._figurecomposer._codegen
import erlab.interactive._figurecomposer._provenance
import erlab.interactive._figurecomposer._toolbar_dialogs
import erlab.interactive._qt_state as _qt_state
from erlab.interactive._figurecomposer._axes import _all_axes, _axes_expression_value
from erlab.interactive._figurecomposer._defaults import (
    _MM_PER_INCH,
    _figure_draw_context,
    _figure_style_context,
    figure_options_context,
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
    _gridspec_reserved_axis_code_names,
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
from erlab.interactive._figurecomposer._operations._custom_code import (
    _custom_code_names,
    _renamed_source_loads,
)
from erlab.interactive._figurecomposer._operations._plot_slices import (
    _PLOT_SLICES_MAPPABLE_OPERATION_ID_ATTR,
    _PLOT_SLICES_MAPPABLE_PANEL_KEY_ATTR,
    _effective_extra_kwargs,
    _effective_slice_kwargs,
    _is_slice_kwarg_key,
    _operation_dim_names,
    _plot_slices_panel_keys,
    _selection_updates_from_kwargs,
    _selection_values,
    _selection_width,
)
from erlab.interactive._figurecomposer._operations._source_selection import (
    selection_dim_mode,
    selection_dim_value_text,
    selection_dim_width_text,
    selection_has_effect,
    selection_value_from_text,
    selection_width_from_text,
    selection_with_dimension,
    shared_selection,
)
from erlab.interactive._figurecomposer._rendering import (
    _live_layout_axes,
    _render_into_figure,
    _render_preview,
    _rendered_output_figure,
)
from erlab.interactive._figurecomposer._source_inspector import (
    SourceInspectorWidget,
    source_metadata_tooltip,
)
from erlab.interactive._figurecomposer._sources import (
    _FIGURE_CODE_RESERVED_NAMES,
    _default_plot_operation,
    _default_setup_for_data,
    _public_source_data,
    _selected_data,
    _selected_source_data,
    _source_alias_error,
    _source_display_label,
    _source_has_selection,
    _source_name,
    _source_selection,
    _source_unique_name,
    _source_with_selection,
)
from erlab.interactive._figurecomposer._state import (
    FigureAxesSelectionState,
    FigureDataSelectionState,
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
    _AxesTargetItemDelegate,
    _FigureComposerDisplayWindow,
    _gridspec_target_preview_descriptor,
    _GridSpecRegionInfo,
    _GridSpecViewWidget,
    _step_toolbar_button,
    _subplot_target_preview_descriptor,
)
from erlab.interactive.imagetool import provenance

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
_RETIRED_EDITOR_DRAIN_DELAY_MS = 100
_PREVIEW_RENDER_UPDATE_DELAY_MS = 50
_EDITOR_CONTROL_RENDER_UPDATE_DELAY_MS = 300
_FIGURE_RESIZE_RENDER_DELAY_MS = 120
_FIGURE_RESIZE_HISTORY_DELAY_MS = 250
_PREVIEW_PIXMAP_UPDATE_DELAY_MS = 250
_PERSISTED_PREVIEW_CACHE_ATTR = "figure_composer_preview_cache_png"
_PERSISTED_PREVIEW_CACHE_STALE_ATTR = "figure_composer_preview_cache_stale"
_PERSISTED_SELECTED_SOURCE_DATA_ATTR = "figure_composer_selected_source_data"
_PERSISTED_PREVIEW_CACHE_SIZE = QtCore.QSize(512, 384)
_PERSISTED_PREVIEW_CACHE_MAX_BYTES = 384_000
_COMBO_POPUP_REBUILD_GRACE_MS = 150
_COMBO_INTERACTION_REBUILD_GRACE_MS = 250
_COMBO_TRACKED_PROPERTY = "figure_composer_combo_tracked"
_COMBO_POPUP_GUARD_ID_PROPERTY = "figure_composer_combo_popup_guard_id"
_RESTORE_OPERATION_EDITOR_KEY = "figure_composer_operation_editor"
_RESTORE_REDRAW_KEY = "figure_composer_restored_redraw"
_OPERATION_LIST_STEP_COLUMN = 0
_OPERATION_LIST_TARGET_COLUMN = 1
_OPERATION_LIST_STATUS_COLUMN = 2
_OPERATION_LIST_TARGET_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_OPERATION_LIST_STATUS_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2
_SOURCE_LIST_SOURCE_COLUMN = 0
_SOURCE_LIST_SHAPE_COLUMN = 1
_SOURCE_LIST_USED_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
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


def _event_requests_context_menu(event: QtGui.QKeyEvent) -> bool:
    return event.key() == QtCore.Qt.Key.Key_Menu or (
        event.key() == QtCore.Qt.Key.Key_F10
        and bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
    )


class _FigureComposerStepEditorScroll(QtWidgets.QScrollArea):
    """Scroll vertically without allowing the editor content to clip horizontally."""

    def minimumSizeHint(self) -> QtCore.QSize:
        hint = super().minimumSizeHint()
        content = self.widget()
        if content is None:
            return hint
        width = content.minimumSizeHint().width() + 2 * self.frameWidth()
        if (
            self.verticalScrollBarPolicy()
            != QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        ):
            scrollbar = self.verticalScrollBar()
            if scrollbar is not None:  # pragma: no branch
                width += scrollbar.sizeHint().width()
        return QtCore.QSize(max(hint.width(), width), hint.height())


class _FigureComposerStepEditorPage(QtWidgets.QWidget):
    """Editor page that clears itself with the enclosing tab pane background."""

    def __init__(
        self,
        editor_tabs: QtWidgets.QTabWidget,
        parent: QtWidgets.QWidget,
    ) -> None:
        super().__init__(parent)
        self._editor_tabs = editor_tabs
        self._background_color = self.palette().color(QtGui.QPalette.ColorRole.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def _refresh_background(self) -> None:
        source = self._editor_tabs
        sample_point = source.rect().center()
        background = QtGui.QPixmap(1, 1)
        background.fill(self._background_color)
        source.render(
            background,
            QtCore.QPoint(),
            QtGui.QRegion(QtCore.QRect(sample_point, QtCore.QSize(1, 1))),
            QtWidgets.QWidget.RenderFlag.DrawWindowBackground,
        )
        color = background.toImage().pixelColor(0, 0)
        if color.isValid() and color.alpha() != 0:
            self._background_color = color
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        painter = QtGui.QPainter(self)
        try:
            painter.fillRect(self.rect(), self._background_color)
        finally:
            painter.end()
        if event is not None:
            super().paintEvent(event)

    def changeEvent(self, event: QtCore.QEvent | None) -> None:
        if event is not None:
            super().changeEvent(event)
        if event is not None and event.type() in {
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.StyleChange,
        }:
            self._background_color = self.palette().color(
                QtGui.QPalette.ColorRole.Window
            )
            erlab.interactive.utils.single_shot(self, 0, self._refresh_background)


class _FigureComposerReorderList(QtWidgets.QTreeWidget):
    rows_reordered = QtCore.Signal(object, object, object)

    def __init__(self, id_column: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._reorder_id_column = id_column
        self._rows_reordered_pending = False
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)

    def _row_ids(self) -> tuple[str, ...]:
        row_ids: list[str] = []
        for row in range(self.topLevelItemCount()):
            item = self.topLevelItem(row)
            row_id = (
                None
                if item is None
                else item.data(
                    self._reorder_id_column,
                    QtCore.Qt.ItemDataRole.UserRole,
                )
            )
            if not isinstance(row_id, str):
                return ()
            row_ids.append(row_id)
        return tuple(row_ids)

    def _queue_rows_reordered(self, *_args: object) -> None:
        if self._rows_reordered_pending:
            return
        self._rows_reordered_pending = True
        # Let QTreeWidget finish transferring ownership of the dragged items before
        # the recipe refreshes this view.
        erlab.interactive.utils.single_shot(self, 0, self._emit_rows_reordered)

    def _emit_rows_reordered(self) -> None:
        self._rows_reordered_pending = False
        row_ids = self._row_ids()
        if not row_ids or len(set(row_ids)) != len(row_ids):
            return
        current_id = None
        current_item = self.currentItem()
        if current_item is not None:
            candidate = current_item.data(
                self._reorder_id_column,
                QtCore.Qt.ItemDataRole.UserRole,
            )
            if isinstance(candidate, str):
                current_id = candidate
        selected_ids = frozenset(
            row_id
            for item in self.selectedItems()
            if isinstance(
                row_id := item.data(
                    self._reorder_id_column,
                    QtCore.Qt.ItemDataRole.UserRole,
                ),
                str,
            )
        )
        self.rows_reordered.emit(row_ids, selected_ids, current_id)

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        if event is None:
            return
        if event.source() is not self:
            event.ignore()
            return
        super().dropEvent(event)
        if event.isAccepted():
            self._queue_rows_reordered()


class _FigureComposerOperationList(_FigureComposerReorderList):
    copy_requested = QtCore.Signal()
    cut_requested = QtCore.Signal()
    paste_requested = QtCore.Signal()
    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(_OPERATION_LIST_STEP_COLUMN, parent)
        self.setColumnCount(3)
        self.setHeaderLabels(("Step", "Target", "Status"))
        self.setRootIsDecorated(False)
        self.setItemsExpandable(False)
        self.setIndentation(0)
        self.setUniformRowHeights(True)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def _operation_ids(self) -> tuple[str, ...]:
        return self._row_ids()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if _event_requests_context_menu(event):
            item = self.currentItem()
            rect = self.visualItemRect(item) if item is not None else QtCore.QRect()
            self.context_menu_requested.emit(rect.center())
            event.accept()
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


class _FigureComposerSourceList(_FigureComposerReorderList):
    context_menu_requested = QtCore.Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(_SOURCE_LIST_SOURCE_COLUMN, parent)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

    def _source_names(self) -> tuple[str, ...]:
        return self._row_ids()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return
        if _event_requests_context_menu(event):
            item = self.currentItem()
            rect = self.visualItemRect(item) if item is not None else QtCore.QRect()
            self.context_menu_requested.emit(rect.center())
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
        self._operation_axes_sync_pending = False
        self._preview_render_update_pending = False
        self._preview_render_update_generation = 0
        self._step_tab_order_update_pending = False
        self._retired_editor_drain_pending = False
        self._combo_popup_guard_tokens: set[int] = set()
        self._next_combo_popup_guard_token = 0
        self._tracked_combo_refs: list[weakref.ReferenceType[QtWidgets.QComboBox]] = []
        self._operation_multi_select_event = False
        self._operation_selection_input_event = False
        self._operation_selection_state: tuple[str | None, frozenset[str]] | None = None
        self._operation_list_viewport: QtWidgets.QWidget | None = None
        self._retired_editor_widgets: list[QtWidgets.QWidget] = []
        self._operation_render_errors: dict[str, str] = {}
        self._operation_input_errors: dict[str, dict[str, str]] = {}
        self._plot_slices_selection_cache: (
            MutableMapping[Hashable, tuple[xr.DataArray, ...]] | None
        ) = None
        self._operation_editor_generation = 0
        self._active_editor_signal_widget: QtWidgets.QWidget | None = None
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
        self._section_tab_stop_refs: dict[
            str, weakref.ReferenceType[QtWidgets.QWidget]
        ] = {}
        self._source_data: dict[str, xr.DataArray] = {}
        self._source_selection_base_data: dict[str, xr.DataArray] = {}
        self._source_refresh_available_callback: Callable[[str], bool] | None = None
        self._source_refresh_callback: Callable[[str], bool] | None = None
        self._source_refresh_label_callback: Callable[[str], str | None] | None = None
        self._source_add_available_callback: Callable[[], bool] | None = None
        self._source_add_callback: Callable[[], bool] | None = None
        self._source_drop_available_callback: (
            Callable[[QtCore.QMimeData], bool] | None
        ) = None
        self._source_drop_callback: Callable[[QtCore.QMimeData], bool] | None = None
        self._source_inspector_target: str | None = None
        self._updating_source_selection = False
        self._recipe = recipe or self._default_recipe(data)
        self._active_gridspec_grid_id = self._recipe.setup.gridspec.root.grid_id
        self._gridspec_breadcrumb_buttons: list[QtWidgets.QToolButton] = []
        self._figure_window: _FigureComposerDisplayWindow | None = None
        self._subplot_adjust_dialog: QtWidgets.QDialog | None = None
        self._axes_customize_dialog: QtWidgets.QDialog | None = None
        self._operation_context_menu: QtWidgets.QMenu | None = None
        self._source_context_menu: QtWidgets.QMenu | None = None
        self._connected_step_clipboard: QtGui.QClipboard | None = None
        self._step_clipboard_tool_id = uuid.uuid4().hex
        self._prev_source_data_states: collections.deque[
            tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
        ] = collections.deque(maxlen=self._prev_states.maxlen)
        self._next_source_data_states: collections.deque[
            tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]
        ] = collections.deque(maxlen=self._next_states.maxlen)

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

        self._normalize_operation_source_selections()
        self._current_step_section_key = "sources"
        self._build_ui()
        self.setAcceptDrops(True)
        self._apply_recipe_to_controls()
        self._write_state()

    def set_options_getter(self, getter: Callable[[], AppOptions] | None) -> None:
        self._options_getter = getter

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
            self._recipe.setup, self._figure_window_title(), activate=activate
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
        self._recipe = self._recipe.model_copy(update={"operations": tuple(operations)})
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

    def _sync_figure_window_to_recipe_setup(self) -> None:
        window = self._figure_window
        if window is None or not erlab.interactive.utils.qt_is_valid(window):
            return
        window.resize_to_setup(self._recipe.setup)
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
        if history == "deferred":
            self._queue_figure_resize_history_write()
        else:
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
        current_page = self.step_editor_stack.currentWidget()
        if isinstance(current_page, _FigureComposerStepEditorPage):
            current_page._refresh_background()
        self._request_show_figure_window(activate=False)

    def hideEvent(self, event: QtGui.QHideEvent | None) -> None:
        self._cancel_queued_show_figure_window()
        self._hide_figure_window()
        if event is not None:
            super().hideEvent(event)

    def _flush_pending_editor_commits(self, *, render: bool = False) -> None:
        for widget in self.findChildren(QtWidgets.QWidget):
            flush = getattr(
                widget,
                "_figure_composer_custom_code_flush_pending_commit",
                None,
            )
            if callable(flush):
                with contextlib.suppress(RuntimeError):
                    flush(render=render)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._flush_pending_editor_commits()
        self._flush_pending_figure_resize_history_write()
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

        selected_step_action_layout = QtWidgets.QHBoxLayout()
        selected_step_action_layout.setSpacing(4)
        selected_step_action_layout.addWidget(self.add_step_button)
        selected_step_action_layout.addWidget(self.copy_operation_button)
        selected_step_action_layout.addWidget(self.cut_operation_button)
        selected_step_action_layout.addWidget(self.paste_operation_button)
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
        self.operation_list.rows_reordered.connect(self._operation_list_reordered)
        self.operation_target_delegate = _AxesTargetItemDelegate(
            int(_OPERATION_LIST_TARGET_ROLE), self.operation_list
        )
        self.operation_list.setItemDelegateForColumn(
            _OPERATION_LIST_TARGET_COLUMN, self.operation_target_delegate
        )
        self._operation_list_viewport = self.operation_list.viewport()
        if self._operation_list_viewport is not None:
            self._operation_list_viewport.installEventFilter(self)
        self.operation_list.currentItemChanged.connect(
            self._operation_current_item_changed
        )
        self.operation_list.itemSelectionChanged.connect(
            self._operation_selection_changed
        )
        self.operation_list.itemChanged.connect(self._operation_item_changed)
        self.operation_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.operation_list.setMinimumHeight(140)
        self.operation_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.operation_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        operation_header = self.operation_list.header()
        if operation_header is not None:  # pragma: no branch
            operation_header.setStretchLastSection(False)
            operation_header.setMinimumSectionSize(48)
            operation_header.setSectionResizeMode(
                _OPERATION_LIST_STEP_COLUMN,
                QtWidgets.QHeaderView.ResizeMode.Stretch,
            )
            operation_header.setSectionResizeMode(
                _OPERATION_LIST_TARGET_COLUMN,
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
            operation_header.setSectionResizeMode(
                _OPERATION_LIST_STATUS_COLUMN,
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
            operation_header.resizeSection(_OPERATION_LIST_TARGET_COLUMN, 72)
            operation_header.resizeSection(_OPERATION_LIST_STATUS_COLUMN, 88)
        self.recipe_splitter.addWidget(self.operation_list)

        self.step_inspector = QtWidgets.QWidget(recipe_page)
        self.step_inspector.setObjectName("figureComposerStepInspector")
        self.step_inspector.setAutoFillBackground(False)
        step_inspector_layout = QtWidgets.QHBoxLayout(self.step_inspector)
        step_inspector_layout.setContentsMargins(0, 0, 0, 0)
        step_inspector_layout.setSpacing(6)
        self.recipe_splitter.addWidget(self.step_inspector)
        self.recipe_splitter.setStretchFactor(0, 0)
        self.recipe_splitter.setStretchFactor(1, 1)
        self.recipe_splitter.setSizes((140, 410))

        self.step_navigator = QtWidgets.QWidget(self.step_inspector)
        self.step_navigator.setObjectName("figureComposerStepNavigator")
        self.step_navigator.setFixedWidth(150)
        self.step_navigator_layout = QtWidgets.QVBoxLayout(self.step_navigator)
        self.step_navigator_layout.setContentsMargins(0, 0, 0, 0)
        self.step_navigator_layout.setSpacing(3)
        self.step_section_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.step_section_keys: list[str] = []
        step_inspector_layout.addWidget(self.step_navigator)

        self.step_editor_scroll = _FigureComposerStepEditorScroll(self.step_inspector)
        self.step_editor_scroll.setObjectName("figureComposerStepEditorScroll")
        self.step_editor_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.step_editor_scroll.setWidgetResizable(True)
        self.step_editor_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.step_editor_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.step_editor_scroll.setAutoFillBackground(False)
        step_editor_viewport = self.step_editor_scroll.viewport()
        if step_editor_viewport is not None:  # pragma: no branch
            step_editor_viewport.setObjectName("figureComposerStepEditorViewport")
            step_editor_viewport.setAutoFillBackground(False)
        step_inspector_layout.addWidget(self.step_editor_scroll, 1)

        self.step_editor_stack = QtWidgets.QStackedWidget()
        self.step_editor_stack.setObjectName("figureComposerStepSectionStack")
        self.step_editor_scroll.setWidget(self.step_editor_stack)
        self.step_editor_stack.setAutoFillBackground(False)
        self._operation_editor_pages: list[QtWidgets.QWidget] = []

        self.step_sources_page = _FigureComposerStepEditorPage(
            self.editor_tabs, self.step_editor_stack
        )
        self.step_sources_page.setObjectName("figureComposerStepSourcesPage")
        step_sources_layout = QtWidgets.QVBoxLayout(self.step_sources_page)
        step_sources_layout.setContentsMargins(6, 6, 6, 6)
        step_sources_layout.setSpacing(4)
        self.step_source_controls = QtWidgets.QWidget(self.step_sources_page)
        self.step_source_controls_layout = QtWidgets.QFormLayout(
            self.step_source_controls
        )
        self.step_source_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.step_source_controls_layout.setSpacing(4)
        self.step_source_controls_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        step_sources_layout.addWidget(self.step_source_controls)
        self.step_source_status_label = QtWidgets.QLabel(self.step_sources_page)
        self.step_source_status_label.setObjectName("figureComposerStepSourceStatus")
        self.step_source_status_label.setWordWrap(True)
        self.step_source_status_label.setVisible(False)
        step_sources_layout.addWidget(self.step_source_status_label)

        self.sources_page = QtWidgets.QWidget(self.editor_tabs)
        self.sources_page.setObjectName("figureComposerSourcesPage")
        sources_layout = QtWidgets.QVBoxLayout(self.sources_page)
        sources_layout.setContentsMargins(6, 6, 6, 6)
        sources_layout.setSpacing(4)
        self.source_status_label = QtWidgets.QLabel(self.sources_page)
        self.source_status_label.setObjectName("figureComposerSourceStatus")
        self.source_status_label.setWordWrap(True)
        self.source_status_label.setVisible(False)
        self.source_actions = QtWidgets.QWidget(self.sources_page)
        self.source_actions.setObjectName("figureComposerSourceActions")
        source_actions_layout = QtWidgets.QHBoxLayout(self.source_actions)
        source_actions_layout.setContentsMargins(0, 0, 0, 0)
        source_actions_layout.setSpacing(4)
        self.add_source_button = _step_toolbar_button(
            self.source_actions,
            "figureComposerAddSourceButton",
            "Add…",
            "Add ImageTool data from the manager as figure sources.",
        )
        self.add_source_button.clicked.connect(self._request_add_sources_from_button)
        source_actions_layout.addWidget(self.add_source_button)
        self.remove_selected_source_button = _step_toolbar_button(
            self.source_actions,
            "figureComposerRemoveSelectedSourceButton",
            "Remove",
            "Remove the selected unused source or sources.",
        )
        self.remove_selected_source_button.clicked.connect(
            self._remove_selected_sources
        )
        source_actions_layout.addWidget(self.remove_selected_source_button)
        source_actions_layout.addStretch(1)
        self.refresh_sources_button = QtWidgets.QToolButton(self.source_actions)
        self.refresh_sources_button.setObjectName("figureComposerRefreshSourcesButton")
        self.refresh_sources_button.setText("Refresh")
        self.refresh_sources_button.setAccessibleName("Refresh Selected Sources")
        self.refresh_sources_button.setIcon(QtGui.QIcon.fromTheme("view-refresh"))
        self.refresh_sources_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.refresh_sources_button.setAutoRaise(True)
        self.refresh_sources_button.clicked.connect(
            self._refresh_selected_sources_from_button
        )
        source_actions_layout.addWidget(self.refresh_sources_button)
        sources_layout.addWidget(self.source_actions)
        sources_layout.addWidget(self.source_status_label)

        self.source_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal, self.sources_page
        )
        self.source_splitter.setObjectName("figureComposerSourceSplitter")
        self.source_splitter.setChildrenCollapsible(False)
        self.source_list = _FigureComposerSourceList(self.source_splitter)
        self.source_list.setObjectName("figureComposerSourceList")
        self.source_list.setAccessibleName("Figure Sources")
        self.source_list.setColumnCount(2)
        self.source_list.setHeaderLabels(("Alias", "Shape"))
        self.source_list.context_menu_requested.connect(self._show_source_context_menu)
        self.source_list.rows_reordered.connect(self._source_list_reordered)
        self.source_list.setRootIsDecorated(False)
        self.source_list.setIndentation(0)
        self.source_list.setUniformRowHeights(True)
        self.source_list.setAlternatingRowColors(True)
        self.source_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.source_list.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.source_list.currentItemChanged.connect(
            self._source_list_current_item_changed
        )
        self.source_list.itemSelectionChanged.connect(
            self._source_list_selection_changed
        )
        self.source_list.itemDoubleClicked.connect(
            self._source_list_item_double_clicked
        )
        self.rename_source_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_F2), self.source_list
        )
        self.rename_source_shortcut.setObjectName("figureComposerRenameSourceShortcut")
        self.rename_source_shortcut.activated.connect(self._focus_source_alias_editor)
        self.source_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.source_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        source_header = self.source_list.header()
        if source_header is not None:
            source_header.setStretchLastSection(False)
            source_header.setMinimumSectionSize(80)
            source_header.setSectionResizeMode(
                _SOURCE_LIST_SOURCE_COLUMN,
                QtWidgets.QHeaderView.ResizeMode.Stretch,
            )
            source_header.setSectionResizeMode(
                _SOURCE_LIST_SHAPE_COLUMN,
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
            source_header.resizeSection(_SOURCE_LIST_SHAPE_COLUMN, 150)
        self.source_splitter.addWidget(self.source_list)

        self.source_detail_scroll = QtWidgets.QScrollArea(self.source_splitter)
        self.source_detail_scroll.setObjectName("figureComposerSourceDetailScroll")
        self.source_detail_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.source_detail_scroll.setWidgetResizable(True)
        self.source_detail_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.source_detail_content = QtWidgets.QWidget(self.source_detail_scroll)
        self.source_detail_content.setObjectName("figureComposerSourceDetailContent")
        source_detail_layout = QtWidgets.QVBoxLayout(self.source_detail_content)
        source_detail_layout.setContentsMargins(8, 0, 0, 0)
        source_detail_layout.setSpacing(6)

        self.source_editor_state_label = QtWidgets.QLabel(
            "Select a source to inspect or edit it.", self.source_detail_content
        )
        self.source_editor_state_label.setObjectName("figureComposerSourceEditorState")
        self.source_editor_state_label.setWordWrap(True)
        source_detail_layout.addWidget(self.source_editor_state_label)

        self.source_alias_controls = QtWidgets.QWidget(self.source_detail_content)
        self.source_alias_controls.setObjectName("figureComposerSourceAliasControls")
        source_alias_layout = QtWidgets.QFormLayout(self.source_alias_controls)
        source_alias_layout.setContentsMargins(0, 0, 0, 0)
        source_alias_layout.setSpacing(4)
        source_alias_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.source_alias_edit = QtWidgets.QLineEdit(self.source_alias_controls)
        self.source_alias_edit.setObjectName("figureComposerSourceAliasEdit")
        self.source_alias_edit.setToolTip("Rename this source variable.")
        self.source_alias_edit.editingFinished.connect(self._commit_source_alias_edit)
        source_alias_layout.addRow("Alias", self.source_alias_edit)
        source_detail_layout.addWidget(self.source_alias_controls)

        self.source_inspector = SourceInspectorWidget(self.source_detail_content)
        self.source_inspector.setAcceptDrops(True)
        source_detail_layout.addWidget(self.source_inspector)
        self.source_selection_controls = QtWidgets.QWidget(self.source_detail_content)
        self.source_selection_controls.setObjectName(
            "figureComposerSourceSelectionControls"
        )
        self.source_selection_controls_layout = QtWidgets.QFormLayout(
            self.source_selection_controls
        )
        self.source_selection_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.source_selection_controls_layout.setSpacing(4)
        self.source_selection_controls_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.source_selection_controls.setAcceptDrops(True)
        source_detail_layout.addWidget(self.source_selection_controls)
        self.source_validation_label = QtWidgets.QLabel(self.source_detail_content)
        self.source_validation_label.setObjectName(
            "figureComposerSourceValidationStatus"
        )
        self.source_validation_label.setWordWrap(True)
        self.source_validation_label.setVisible(False)
        source_detail_layout.addWidget(self.source_validation_label)
        source_detail_layout.addStretch(1)
        self.source_detail_scroll.setWidget(self.source_detail_content)
        self.source_detail_content.setAutoFillBackground(False)
        source_detail_viewport = self.source_detail_scroll.viewport()
        if source_detail_viewport is not None:  # pragma: no branch
            source_detail_viewport.setAutoFillBackground(False)
        self.source_splitter.addWidget(self.source_detail_scroll)
        self.source_splitter.setStretchFactor(0, 2)
        self.source_splitter.setStretchFactor(1, 3)
        self.source_splitter.setSizes((400, 600))
        sources_layout.addWidget(self.source_splitter, 1)
        for drop_target in (
            self.sources_page,
            self.source_list,
            self.source_list.viewport(),
            self.source_detail_scroll,
            self.source_detail_scroll.viewport(),
            self.source_detail_content,
            self.source_inspector,
            self.source_selection_controls,
        ):
            if drop_target is not None:
                drop_target.setAcceptDrops(True)
                drop_target.installEventFilter(self)

        self.target_axes_page = _FigureComposerStepEditorPage(
            self.editor_tabs, self.step_editor_stack
        )
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
        self.nrows_spin = erlab.interactive.utils.BetterSpinBox(
            layout_page, integer=True, minimum=1
        )
        self.ncols_spin = erlab.interactive.utils.BetterSpinBox(
            layout_page, integer=True, minimum=1
        )
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
        for spinbox in (
            self.nrows_spin,
            self.ncols_spin,
            self.width_spin,
            self.height_spin,
            self.width_mm_spin,
            self.height_mm_spin,
            self.dpi_spin,
        ):
            spinbox.setKeyboardTracking(False)
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

        sources_index = self.editor_tabs.addTab(self.sources_page, "Sources")
        self.editor_tabs.setTabToolTip(
            sources_index, "Named data variables captured for this figure."
        )
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
            if (
                self.operation_list.topLevelItemCount()
                and self.operation_list.currentItem() is None
            ):
                self.operation_list.setCurrentItem(self.operation_list.topLevelItem(0))
            self._refresh_operation_editor()
        finally:
            self._updating_controls = False

    @staticmethod
    def _set_combo_value(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findText(value)
        combo.setCurrentIndex(max(index, 0))

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
        inspector = getattr(self, "source_inspector", None)
        if isinstance(inspector, SourceInspectorWidget):
            inspector.invalidate_details()
        selected_names = set(self._selected_source_names())
        current_name = self._source_name_from_list_item(self.source_list.currentItem())
        self._clear_source_list_widgets()
        source_by_name = {source.name: source for source in self._recipe.sources}
        used_sources = self._sources_used_by_recipe()
        shown: set[str] = set()
        for source in self._recipe.sources:
            name = source.name
            shown.add(name)
            display = _source_display_label(source, name)
            self._add_source_list_row(
                display,
                name,
                data=self._source_data.get(name),
                missing=name not in self._source_data,
                used=name in used_sources,
            )

        for name, data in self._source_data.items():
            if name in shown:
                continue
            display = _source_display_label(source_by_name.get(name), name)
            self._add_source_list_row(
                display,
                name,
                data=data,
                missing=name not in source_by_name,
                used=name in used_sources,
            )
        available_names = set(self._source_names())
        selected_names.intersection_update(available_names)
        if current_name not in available_names:
            current_name = (
                self._source_inspector_target
                if self._source_inspector_target in available_names
                else self._default_source_inspector_target()
            )
        if not selected_names and current_name is not None:
            selected_names.add(current_name)
        self._set_selected_source_names_silent(selected_names, current_name)
        self._source_inspector_target = current_name
        self._refresh_source_controls()
        self._refresh_source_detail_panel()
        self._refresh_source_selection_editor()

    def _clear_source_list_widgets(self) -> None:
        for row in range(self.source_list.topLevelItemCount()):
            item = self.source_list.topLevelItem(row)
            if item is None:  # pragma: no cover
                # Qt can report stale rows during teardown.
                continue
            for column in range(self.source_list.columnCount()):
                widget = self.source_list.itemWidget(item, column)
                if widget is None:
                    continue
                self.source_list.removeItemWidget(item, column)
                widget.setParent(None)
                widget.deleteLater()
        self.source_list.clear()

    def _add_source_list_row(
        self,
        display: str,
        name: str,
        *,
        data: xr.DataArray | None = None,
        missing: bool = False,
        used: bool = False,
    ) -> None:
        item = QtWidgets.QTreeWidgetItem(
            [display, "Data unavailable" if data is None else ""]
        )
        item.setData(_SOURCE_LIST_SOURCE_COLUMN, QtCore.Qt.ItemDataRole.UserRole, name)
        item.setData(_SOURCE_LIST_SOURCE_COLUMN, _SOURCE_LIST_USED_ROLE, used)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        if missing:
            item.setForeground(
                _SOURCE_LIST_SOURCE_COLUMN, QtGui.QBrush(QtGui.QColor("darkRed"))
            )
            item.setForeground(
                _SOURCE_LIST_SHAPE_COLUMN, QtGui.QBrush(QtGui.QColor("darkRed"))
            )
        self.source_list.addTopLevelItem(item)
        self._update_source_list_item(item)
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
        shape_label.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        shape_label.setContentsMargins(4, 0, 4, 0)
        if missing:
            palette = shape_label.palette()
            palette.setColor(
                QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("darkRed")
            )
            shape_label.setPalette(palette)
        item.setSizeHint(_SOURCE_LIST_SHAPE_COLUMN, shape_label.sizeHint())
        self.source_list.setItemWidget(item, _SOURCE_LIST_SHAPE_COLUMN, shape_label)

    def _update_source_list_item(self, item: QtWidgets.QTreeWidgetItem) -> None:
        name = self._source_name_from_list_item(item)
        if name is None:
            return
        item.setIcon(_SOURCE_LIST_SOURCE_COLUMN, QtGui.QIcon())
        tooltip = self._source_tooltip(name)
        item.setToolTip(_SOURCE_LIST_SOURCE_COLUMN, tooltip)
        item.setToolTip(_SOURCE_LIST_SHAPE_COLUMN, tooltip)
        item.setData(
            _SOURCE_LIST_SOURCE_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            tooltip,
        )

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, QtWidgets.QTreeWidgetItem)
    def _source_list_current_item_changed(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        if self._updating_source_selection:
            return
        source_name = self._source_name_from_list_item(current)
        self._source_inspector_target = source_name
        self._set_source_validation_text(None)
        self._refresh_source_detail_panel()
        self._refresh_source_selection_editor()

    @QtCore.Slot()
    def _source_list_selection_changed(self) -> None:
        if self._updating_source_selection:
            return
        self._set_source_validation_text(None)
        self._refresh_source_controls()
        self._refresh_source_detail_panel()
        self._refresh_source_selection_editor()

    @staticmethod
    def _source_name_from_list_item(
        item: QtWidgets.QTreeWidgetItem | None,
    ) -> str | None:
        if item is None:
            return None
        source_name = item.data(
            _SOURCE_LIST_SOURCE_COLUMN, QtCore.Qt.ItemDataRole.UserRole
        )
        return source_name if isinstance(source_name, str) else None

    def _selected_source_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for item in self.source_list.selectedItems():
            source_name = self._source_name_from_list_item(item)
            if source_name is not None and source_name not in names:
                names.append(source_name)
        return tuple(names)

    def _selected_source_indices(self) -> tuple[int, ...]:
        selected = set(self._selected_source_names())
        return tuple(
            index
            for index, source in enumerate(self._recipe.sources)
            if source.name in selected
        )

    def _source_move_possible(self, offset: int) -> bool:
        indices = self._selected_source_indices()
        if not indices:
            return False
        index_set = set(indices)
        if offset < 0:
            return any(index > 0 and index - 1 not in index_set for index in indices)
        return any(
            index < len(self._recipe.sources) - 1 and index + 1 not in index_set
            for index in indices
        )

    def _source_duplicate_possible(self) -> bool:
        return bool(self._selected_source_indices())

    def _set_selected_source_names_silent(
        self, selected_names: set[str], current_name: str | None
    ) -> None:
        self._updating_source_selection = True
        try:
            self.source_list.clearSelection()
            if current_name is None:
                self.source_list.setCurrentIndex(QtCore.QModelIndex())
            current_item: QtWidgets.QTreeWidgetItem | None = None
            for row in range(self.source_list.topLevelItemCount()):
                item = self.source_list.topLevelItem(row)
                if item is None:  # pragma: no cover
                    continue
                source_name = self._source_name_from_list_item(item)
                if source_name == current_name:
                    current_item = item
            if current_item is not None:
                self.source_list.setCurrentItem(current_item)
            for row in range(self.source_list.topLevelItemCount()):
                item = self.source_list.topLevelItem(row)
                if item is None:  # pragma: no cover
                    continue
                item.setSelected(
                    self._source_name_from_list_item(item) in selected_names
                )
        finally:
            self._updating_source_selection = False

    def _source_refresh_available(self, name: str) -> bool:
        if (
            self._source_refresh_available_callback is None
            or self._source_refresh_callback is None
        ):
            return False
        with contextlib.suppress(LookupError, RuntimeError, ValueError):
            return bool(self._source_refresh_available_callback(name))
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
            for name in self._source_names()
            if self._source_refresh_available(name)
        )

    def _refresh_source_controls(self) -> None:
        selected_names = self._selected_source_names()
        add_enabled = self._source_add_available()
        self.add_source_button.setEnabled(add_enabled)
        add_tip = (
            "Add ImageTool data from the manager as figure sources"
            if add_enabled
            else "Open this Figure Composer from ImageTool Manager to add sources"
        )
        self.add_source_button.setToolTip(add_tip)
        self.add_source_button.setStatusTip(add_tip)
        selected_refreshable = tuple(
            name for name in selected_names if self._source_refresh_available(name)
        )
        selected_tip = (
            "Refresh selected sources from their ImageTools"
            if selected_refreshable
            else "No selected sources can be refreshed"
        )
        self.refresh_sources_button.setEnabled(bool(selected_refreshable))
        self.refresh_sources_button.setToolTip(selected_tip)
        self.refresh_sources_button.setStatusTip(self.refresh_sources_button.toolTip())

        removable_selected = [
            name for name in selected_names if self._source_removable(name)
        ]
        remove_enabled = bool(removable_selected)
        self.remove_selected_source_button.setEnabled(remove_enabled)
        remove_tip = (
            "Remove the selected unused source or sources"
            if remove_enabled
            else "Selected sources are in use or cannot be removed"
        )
        self.remove_selected_source_button.setToolTip(remove_tip)
        self.remove_selected_source_button.setStatusTip(remove_tip)

    def refresh_source_controls(self) -> None:
        for row in range(self.source_list.topLevelItemCount()):
            item = self.source_list.topLevelItem(row)
            if item is not None:
                self._update_source_list_item(item)
        self._refresh_source_controls()
        self._refresh_source_detail_panel()

    def _set_source_status_text(self, text: str | None) -> None:
        self.source_status_label.setText("" if text is None else text)
        self.source_status_label.setVisible(bool(text))

    def _set_source_validation_text(self, text: str | None) -> None:
        self.source_validation_label.setText("" if text is None else text)
        self.source_validation_label.setVisible(bool(text))

    def _set_step_source_status_text(self, text: str | None) -> None:
        self.step_source_status_label.setText("" if text is None else text)
        self.step_source_status_label.setVisible(bool(text))

    @QtCore.Slot()
    def _refresh_selected_sources_from_button(self) -> None:
        self._refresh_source_names(self._selected_source_names())

    @QtCore.Slot()
    def _refresh_all_sources_from_button(self) -> None:
        self._refresh_source_names(self._source_names())

    def _refresh_source_names(self, source_names: Sequence[str]) -> None:
        callback = self._source_refresh_callback
        requested = tuple(dict.fromkeys(source_names))
        refreshable = tuple(
            name for name in requested if self._source_refresh_available(name)
        )
        unavailable = tuple(name for name in requested if name not in refreshable)
        if callback is None or not refreshable:
            if unavailable:
                self._set_source_status_text(
                    "Unavailable: " + ", ".join(unavailable) + "."
                )
            self._refresh_source_controls()
            return

        refreshed: list[str] = []
        failed: list[str] = []
        failure_messages: list[str] = []
        self._set_source_status_text(None)
        for name in refreshable:
            self._set_source_status_text(None)
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
            if message := self.source_status_label.text().strip():
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
        self._set_source_status_text(status or None)
        self._refresh_source_controls()
        self._refresh_source_detail_panel()

    def _operation_source_names(
        self, operation: FigureOperationState
    ) -> tuple[str, ...]:
        names = list(_registry.spec_for(operation.kind).source_names(operation))
        if operation.kind == FigureOperationKind.CUSTOM:
            loaded_names = _custom_code_names(operation.code)
            names.extend(
                source_name
                for source_name in self._source_names()
                if source_name in loaded_names
            )
        return tuple(dict.fromkeys(names))

    def _source_dependency_names(self, names: Iterable[str]) -> tuple[str, ...]:
        source_by_name = self._source_by_name()
        ordered: list[str] = []
        resolved: set[str] = set()
        resolving: set[str] = set()

        def add_dependencies(name: str) -> None:
            if name in resolved or name in resolving:
                return
            resolving.add(name)
            source = source_by_name.get(name)
            if source is not None:
                base_name = source.selection_source
                if base_name is not None and base_name != name:
                    add_dependencies(base_name)
            resolving.remove(name)
            resolved.add(name)
            ordered.append(name)

        for name in names:
            add_dependencies(name)
        return tuple(ordered)

    def _operation_source_dependency_names(
        self, operation: FigureOperationState
    ) -> tuple[str, ...]:
        return self._source_dependency_names(self._operation_source_names(operation))

    def _direct_sources_used_by_recipe(
        self, *, enabled_only: bool = False, executable_only: bool = False
    ) -> set[str]:
        return {
            source_name
            for operation in self._recipe.operations
            if not enabled_only or operation.enabled
            if not executable_only
            or operation.kind != FigureOperationKind.CUSTOM
            or operation.trusted
            for source_name in self._operation_source_names(operation)
        }

    def _source_used_by_operation(self, name: str) -> bool:
        return any(
            name in self._operation_source_dependency_names(operation)
            for operation in self._recipe.operations
        )

    def _source_usage_count(self, name: str) -> int:
        return sum(
            name in self._operation_source_dependency_names(operation)
            for operation in self._recipe.operations
        )

    def _sources_used_by_recipe(self) -> set[str]:
        return {
            source_name
            for operation in self._recipe.operations
            for source_name in self._operation_source_dependency_names(operation)
        }

    def _source_removable(self, name: str) -> bool:
        return (
            name in self._source_by_name()
            and len(self._recipe.sources) > 1
            and not self._source_used_by_operation(name)
            and not any(
                source.name != name and source.selection_source == name
                for source in self._recipe.sources
            )
        )

    @QtCore.Slot()
    def _remove_selected_sources(self) -> None:
        removed = False
        for name in tuple(self._selected_source_names()):
            if self.remove_source(name):
                removed = True
        if not removed:
            self._refresh_source_controls()

    def _source_alias_error(
        self, alias: str, *, current: str | None = None
    ) -> str | None:
        if error := _source_alias_error(alias):
            return error
        if alias != current and (
            alias in {source.name for source in self._recipe.sources}
            or alias in self._source_data
        ):
            return f"Source alias {alias!r} is already in use."
        return None

    @staticmethod
    def _source_with_name(source: FigureSourceState, name: str) -> FigureSourceState:
        if source.name == name:
            return source
        updates = {"name": name}
        if source.label == source.name:
            updates["label"] = name
        return source.model_copy(update=updates)

    def _source_unique_alias(self, source_name: str, reserved: set[str]) -> str:
        return _source_unique_name(source_name, reserved)

    def _source_copy_alias(self, source_name: str, reserved: set[str]) -> str:
        stem = f"{source_name}_copy"
        alias = stem
        suffix = 2
        while self._source_alias_error(alias) is not None or alias in reserved:
            alias = f"{stem}_{suffix}"
            suffix += 1
        reserved.add(alias)
        return alias

    @QtCore.Slot()
    def _commit_source_alias_edit(self) -> None:
        edit = self.sender()
        if not isinstance(edit, QtWidgets.QLineEdit):
            return
        original = edit.property("figure_composer_source_alias_original")
        if not isinstance(original, str):
            return
        alias = edit.text().strip()
        if alias == original:
            edit.setText(original)
            self._set_source_validation_text(None)
            return
        error = self._source_alias_error(alias, current=original)
        if error is not None:
            self._set_source_validation_text(error)
            return
        self._set_source_validation_text(None)
        if not self._rename_source_alias(original, alias):
            edit.setText(original)

    @QtCore.Slot()
    def _focus_source_alias_editor(self) -> None:
        if len(self._selected_source_names()) != 1:
            return
        if not self.source_alias_edit.isEnabled():
            return
        self.source_alias_edit.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        self.source_alias_edit.selectAll()

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _source_list_item_double_clicked(
        self, _item: QtWidgets.QTreeWidgetItem, column: int
    ) -> None:
        if column == _SOURCE_LIST_SOURCE_COLUMN:
            self._focus_source_alias_editor()

    def _finish_source_structure_change(
        self, selected_names: set[str], current_name: str | None
    ) -> None:
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._refresh_source_list()
        self._set_selected_source_names_silent(selected_names, current_name)
        self._refresh_source_controls()
        self._source_inspector_target = current_name
        self._refresh_source_detail_panel()
        self._refresh_source_selection_editor()
        self._update_source_section()
        self._maybe_redraw_plot()
        self._set_source_status_text(None)
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()

    def _rename_source_alias(self, old_name: str, new_name: str) -> bool:
        self._flush_pending_editor_commits()
        rename_map = {old_name: new_name}
        sources: list[FigureSourceState] = []
        changed = False
        for source in self._recipe.sources:
            updated_source = source
            if source.name == old_name:
                updated_source = self._source_with_name(source, new_name)
            if source.selection_source == old_name:
                updated_source = updated_source.model_copy(
                    update={"selection_source": new_name}
                )
            if updated_source is not source:
                changed = True
                sources.append(updated_source)
            else:
                sources.append(source)
        if not changed:
            self._refresh_source_controls()
            return False

        operations: list[FigureOperationState] = []
        try:
            for operation in self._recipe.operations:
                updated = self._operation_with_renamed_sources(operation, rename_map)
                if (
                    operation.kind == FigureOperationKind.CUSTOM
                    and re.search(rf"\b{re.escape(old_name)}\b", operation.code)
                    is not None
                ):
                    updated = updated.model_copy(
                        update={
                            "code": _renamed_source_loads(operation.code, rename_map)
                        }
                    )
                operations.append(updated)
        except ValueError as exc:
            self._set_source_validation_text(
                f"Could not rename source “{old_name}”: {operation.label}: {exc}."
            )
            self._refresh_source_selection_editor()
            return False

        self._source_data = {
            rename_map.get(name, name): data for name, data in self._source_data.items()
        }
        self._source_selection_base_data = {
            rename_map.get(name, name): data
            for name, data in self._source_selection_base_data.items()
        }
        updates: dict[str, typing.Any] = {
            "sources": tuple(sources),
            "operations": tuple(operations),
        }
        if self._recipe.primary_source == old_name:
            updates["primary_source"] = new_name
        self._recipe = self._recipe.model_copy(update=updates)
        self._finish_source_structure_change({new_name}, new_name)
        return True

    @QtCore.Slot()
    def _duplicate_selected_sources(self) -> None:
        indices = self._selected_source_indices()
        if not indices:
            return
        sources = list(self._recipe.sources)
        reserved = {source.name for source in sources}
        reserved.update(self._source_data)
        duplicates: list[FigureSourceState] = []
        duplicated_names: set[str] = set()
        for index in indices:
            source = sources[index]
            alias = self._source_copy_alias(source.name, reserved)
            duplicates.append(self._source_with_name(source, alias))
            duplicated_names.add(alias)
            if source.name in self._source_data:
                self._source_data[alias] = self._source_data[source.name].copy(
                    deep=False
                )
            if source.name in self._source_selection_base_data:
                self._source_selection_base_data[alias] = (
                    self._source_selection_base_data[source.name].copy(deep=False)
                )
        insert_index = max(indices) + 1
        sources[insert_index:insert_index] = duplicates
        self._recipe = self._recipe.model_copy(update={"sources": tuple(sources)})
        self._finish_source_structure_change(duplicated_names, duplicates[0].name)

    def _move_selected_sources(self, offset: int) -> None:
        indices = self._selected_source_indices()
        if not indices:
            return
        sources = list(self._recipe.sources)
        index_set = set(indices)
        selected_names = {sources[index].name for index in indices}
        moved = False
        if offset < 0:
            for index in indices:
                if index > 0 and index - 1 not in index_set:
                    sources[index - 1], sources[index] = (
                        sources[index],
                        sources[index - 1],
                    )
                    index_set.remove(index)
                    index_set.add(index - 1)
                    moved = True
        else:
            for index in reversed(indices):
                if index < len(sources) - 1 and index + 1 not in index_set:
                    sources[index + 1], sources[index] = (
                        sources[index],
                        sources[index + 1],
                    )
                    index_set.remove(index)
                    index_set.add(index + 1)
                    moved = True
        if not moved:
            self._refresh_source_controls()
            return
        current = self._source_name_from_list_item(self.source_list.currentItem())
        current_name = (
            current if current in selected_names else next(iter(selected_names))
        )
        self._recipe = self._recipe.model_copy(update={"sources": tuple(sources)})
        self._finish_source_structure_change(selected_names, current_name)

    @QtCore.Slot()
    def _move_selected_sources_up(self) -> None:
        self._move_selected_sources(-1)

    @QtCore.Slot()
    def _move_selected_sources_down(self) -> None:
        self._move_selected_sources(1)

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
        source_by_name = {source.name: source for source in self._recipe.sources}
        if (
            len(ordered_names) != len(source_names)
            or len(ordered_names) != len(source_by_name)
            or set(ordered_names) != set(source_by_name)
        ):
            self._refresh_source_list()
            return
        current_order = tuple(source.name for source in self._recipe.sources)
        if ordered_names == current_order:
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
                iter(selected_name_set),
                ordered_names[0] if ordered_names else None,
            )

        sources = tuple(source_by_name[name] for name in ordered_names)
        self._recipe = self._recipe.model_copy(update={"sources": sources})
        self._finish_source_structure_change(selected_name_set, current_source_name)

    @QtCore.Slot(QtCore.QPoint)
    def _show_source_context_menu(self, position: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu("Sources", self.source_list)
        self._source_context_menu = menu
        add_action = QtGui.QAction("Add…", menu)
        add_action.setObjectName("figureComposerContextAddSourceAction")
        add_action.setEnabled(self.add_source_button.isEnabled())
        add_action.triggered.connect(self._request_add_sources_from_button)
        menu.addAction(add_action)

        menu.addSeparator()
        rename_action = QtGui.QAction("Rename Alias", menu)
        rename_action.setObjectName("figureComposerContextRenameSourceAction")
        rename_action.setEnabled(len(self._selected_source_indices()) == 1)
        rename_action.triggered.connect(self._focus_source_alias_editor)
        menu.addAction(rename_action)

        duplicate_action = QtGui.QAction("Duplicate", menu)
        duplicate_action.setObjectName("figureComposerContextDuplicateSourceAction")
        duplicate_action.setEnabled(self._source_duplicate_possible())
        duplicate_action.triggered.connect(self._duplicate_selected_sources)
        menu.addAction(duplicate_action)

        move_up_action = QtGui.QAction("Move Up", menu)
        move_up_action.setObjectName("figureComposerContextMoveSourceUpAction")
        move_up_action.setEnabled(self._source_move_possible(-1))
        move_up_action.triggered.connect(self._move_selected_sources_up)
        menu.addAction(move_up_action)

        move_down_action = QtGui.QAction("Move Down", menu)
        move_down_action.setObjectName("figureComposerContextMoveSourceDownAction")
        move_down_action.setEnabled(self._source_move_possible(1))
        move_down_action.triggered.connect(self._move_selected_sources_down)
        menu.addAction(move_down_action)

        menu.addSeparator()
        refresh_action = QtGui.QAction("Refresh Selected", menu)
        refresh_action.setObjectName("figureComposerContextRefreshSourceAction")
        selected_names = self._selected_source_names()
        refresh_action.setEnabled(
            any(self._source_refresh_available(name) for name in selected_names)
        )
        refresh_action.triggered.connect(self._refresh_selected_sources_from_button)
        menu.addAction(refresh_action)

        refresh_all_action = QtGui.QAction("Refresh All", menu)
        refresh_all_action.setObjectName("figureComposerContextRefreshAllSourcesAction")
        refresh_all_action.setEnabled(bool(self._refreshable_source_names()))
        refresh_all_action.triggered.connect(self._refresh_all_sources_from_button)
        menu.addAction(refresh_all_action)

        remove_action = QtGui.QAction("Remove", menu)
        remove_action.setObjectName("figureComposerContextRemoveSourceAction")
        remove_action.setEnabled(self.remove_selected_source_button.isEnabled())
        remove_action.triggered.connect(self._remove_selected_sources)
        menu.addAction(remove_action)

        viewport = self.source_list.viewport()
        if viewport is not None:  # pragma: no branch
            menu.popup(viewport.mapToGlobal(position))

    def _set_source_list_row_used(
        self, item: QtWidgets.QTreeWidgetItem, used: bool
    ) -> None:
        item.setData(_SOURCE_LIST_SOURCE_COLUMN, _SOURCE_LIST_USED_ROLE, used)
        self._update_source_list_item(item)

    def _sync_source_list_used_state(self) -> None:
        used_sources = self._sources_used_by_recipe()
        for row in range(self.source_list.topLevelItemCount()):
            item = self.source_list.topLevelItem(row)
            if item is None:  # pragma: no cover
                # Qt can report stale rows during teardown.
                continue
            source_name = item.data(
                _SOURCE_LIST_SOURCE_COLUMN, QtCore.Qt.ItemDataRole.UserRole
            )
            self._set_source_list_row_used(item, source_name in used_sources)

    def _default_source_inspector_target(self) -> str | None:
        source_name = self._source_name_from_list_item(self.source_list.currentItem())
        if source_name is not None:
            return source_name
        if self._recipe.primary_source in self._source_data:
            return self._recipe.primary_source
        if self._recipe.sources:
            return self._recipe.sources[0].name
        return next(iter(self._source_data), None)

    def _refresh_source_detail_panel(self) -> None:
        inspector = getattr(self, "source_inspector", None)
        if not isinstance(inspector, SourceInspectorWidget):
            return
        selected_names = self._selected_source_names()
        if not selected_names:
            self.source_detail_content.setProperty(
                "figureComposerSourceEditorMode", "empty"
            )
            self.source_detail_content.setProperty(
                "figureComposerSourceSelectionCount", 0
            )
            self.source_detail_content.setProperty("figureComposerSourceUsageCount", 0)
            self.source_detail_content.setProperty("figureComposerSourceOrigin", "")
            self.source_editor_state_label.setText(
                "Select a source to inspect or edit it."
            )
            self.source_editor_state_label.setVisible(True)
            self.source_alias_controls.setVisible(False)
            inspector.setVisible(False)
            self.source_selection_controls.setVisible(False)
            inspector.set_context(source_name=None, data=None)
            return
        if len(selected_names) > 1:
            self.source_detail_content.setProperty(
                "figureComposerSourceEditorMode", "multiple"
            )
            self.source_detail_content.setProperty(
                "figureComposerSourceSelectionCount", len(selected_names)
            )
            self.source_detail_content.setProperty("figureComposerSourceUsageCount", 0)
            self.source_detail_content.setProperty("figureComposerSourceOrigin", "")
            self.source_editor_state_label.setText(
                f"{len(selected_names)} sources selected\n"
                "Selection changes apply to all compatible selected sources."
            )
            self.source_editor_state_label.setVisible(True)
            self.source_alias_controls.setVisible(False)
            inspector.setVisible(False)
            self.source_selection_controls.setVisible(True)
            inspector.set_context(source_name=None, data=None)
            return

        target = selected_names[0]
        source_states = self._source_by_name()
        self._source_inspector_target = target
        source = source_states.get(target)
        self.source_detail_content.setProperty(
            "figureComposerSourceEditorMode", "single"
        )
        self.source_detail_content.setProperty("figureComposerSourceSelectionCount", 1)
        self.source_detail_content.setProperty(
            "figureComposerSourceUsageCount", self._source_usage_count(target)
        )
        self.source_detail_content.setProperty(
            "figureComposerSourceOrigin", self._source_refresh_label(target) or ""
        )
        self.source_editor_state_label.setVisible(False)
        self.source_alias_controls.setVisible(True)
        self.source_alias_edit.setEnabled(source is not None)
        self.source_alias_edit.setText(target)
        self.source_alias_edit.setProperty(
            "figure_composer_source_alias_original", target
        )
        inspector.setVisible(True)
        self.source_selection_controls.setVisible(True)
        inspector.set_context(
            source_name=target,
            data=self._source_data.get(target),
            context_lines=self._source_detail_context_lines(target),
        )

    def _refresh_source_selection_editor(self) -> None:
        layout = getattr(self, "source_selection_controls_layout", None)
        if not isinstance(layout, QtWidgets.QFormLayout):
            return
        self._clear_form_layout(layout)
        selected_names = self._selected_source_names()
        if not selected_names:
            return

        source_by_name = self._source_by_name()
        available_names = tuple(
            name
            for name in selected_names
            if name in source_by_name and name in self._source_data
        )
        self._add_form_section(
            layout,
            "Selection",
            object_name="figureComposerSourceSelectionSection",
        )
        if not available_names:
            label = QtWidgets.QLabel("No selected source data is available.", self)
            label.setObjectName("figureComposerSourceSelectionMessage")
            label.setWordWrap(True)
            self._add_form_row(
                layout,
                "Dimensions",
                label,
                "Source selections can be edited after source data is available.",
            )
            return

        dimensions = self._common_source_selection_dims(available_names)
        if not dimensions:
            label = QtWidgets.QLabel("No common dimensions.", self)
            label.setObjectName("figureComposerSourceSelectionMessage")
            self._add_form_row(
                layout,
                "Dimensions",
                label,
                "The selected source data has no common editable dimensions.",
            )
            return

        for dim_index, dim_name in enumerate(dimensions):
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

            row = QtWidgets.QWidget(self.source_selection_controls)
            row.setObjectName(f"figureComposerSourceSelectionDimRow{dim_index}")
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            mode_combo = self._source_selection_mode_combo(
                current=current_mode,
                mixed=mode_mixed,
                parent=row,
            )
            mode_combo.setObjectName(
                f"figureComposerSourceSelectionModeCombo{dim_index}"
            )
            mode_combo.setProperty("figure_composer_source_selection_dim", dim_name)
            mode_combo.setToolTip(
                f"{dimension_context}\n\nChoose None, isel, qsel, or Mean for this "
                "dimension."
            )
            value_edit = QtWidgets.QLineEdit(row)
            value_edit.setObjectName(
                f"figureComposerSourceSelectionValueEdit{dim_index}"
            )
            value_edit.setProperty("figure_composer_source_selection_dim", dim_name)
            value_edit.setProperty("figure_composer_source_selection_field", "value")
            value_edit.setPlaceholderText("value")
            value_edit.setToolTip(
                f"{dimension_context}\n\nisel accepts integer positions and index "
                "slices. qsel accepts coordinate values and coordinate slices."
            )
            value_edit.setText("" if value_mixed else value_texts[0])
            self._apply_mixed_line_edit(value_edit, value_mixed)
            width_edit = QtWidgets.QLineEdit(row)
            width_edit.setObjectName(
                f"figureComposerSourceSelectionWidthEdit{dim_index}"
            )
            width_edit.setProperty("figure_composer_source_selection_dim", dim_name)
            width_edit.setProperty("figure_composer_source_selection_field", "width")
            width_edit.setPlaceholderText("width")
            width_edit.setToolTip(
                f"{dimension_context}\n\nOptional qsel width for centered averaging. "
                "Leave blank to select the nearest coordinate."
            )
            width_edit.setText("" if width_mixed else width_texts[0])
            self._apply_mixed_line_edit(width_edit, width_mixed)
            value_edit.setVisible(current_mode in {"isel", "qsel"})
            width_edit.setVisible(current_mode == "qsel")
            self._connect_source_selection_dimension_controls(
                dim_name,
                mode_combo,
                value_edit,
                width_edit,
            )
            row_layout.addWidget(mode_combo)
            row_layout.addWidget(value_edit, 1)
            row_layout.addWidget(width_edit, 1)
            self._add_form_row(
                layout,
                dim_name,
                row,
                dimension_context,
            )

    def _source_selection_mode_combo(
        self,
        *,
        current: str | None,
        mixed: bool,
        parent: QtWidgets.QWidget,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent)
        if mixed:
            combo.addItem(_MIXED_VALUES_TEXT, _MIXED_VALUE)
        for mode, label, tooltip in (
            ("keep", "None", "Leave this dimension unchanged."),
            (
                "isel",
                "isel",
                "Select this dimension by integer position or index slice.",
            ),
            (
                "qsel",
                "qsel",
                "Select this dimension by coordinate value or coordinate slice.",
            ),
            ("mean", "Mean", "Average over and remove this dimension."),
        ):
            combo.addItem(label, mode)
            combo.setItemData(
                combo.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        if mixed:
            item = typing.cast("typing.Any", combo.model()).item(0)
            if item is not None:
                item.setEnabled(False)
            combo.setCurrentIndex(0)
        elif current is not None:
            index = combo.findData(current)
            combo.setCurrentIndex(max(index, 0))
        self._track_combo_interaction(combo)
        return combo

    def _source_selection_dimension_tooltip(
        self, dim: str, source_names: Sequence[str]
    ) -> str:
        source_by_name = self._source_by_name()
        sizes: set[int] = set()
        dtypes: set[str] = set()
        endpoints: set[tuple[str, str]] = set()
        for source_name in source_names:
            data = self._source_selection_input_data(
                source_name, source_by_name.get(source_name)
            )
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

    def _connect_source_selection_dimension_controls(
        self,
        dim: str,
        mode_combo: QtWidgets.QComboBox,
        value_edit: QtWidgets.QLineEdit,
        width_edit: QtWidgets.QLineEdit,
    ) -> None:
        def set_control_visibility(mode: str) -> None:
            value_edit.setVisible(mode in {"isel", "qsel"})
            width_edit.setVisible(mode == "qsel")

        def update_from_controls() -> None:
            mode = mode_combo.currentData()
            if not isinstance(mode, str) or mode not in {
                "keep",
                "isel",
                "qsel",
                "mean",
            }:
                return
            if LineEditControlAdapter(value_edit).unchanged_mixed() and mode in {
                "isel",
                "qsel",
            }:
                return
            if LineEditControlAdapter(width_edit).unchanged_mixed() and mode == "qsel":
                return
            self._update_selected_source_dimension(
                dim,
                mode,
                value_edit.text(),
                width_edit.text(),
            )

        def mode_changed(_index: int) -> None:
            mode = mode_combo.currentData()
            mode_text = mode if isinstance(mode, str) else ""
            set_control_visibility(mode_text)
            previous_mode = mode_combo.property(
                "figure_composer_source_selection_previous_mode"
            )
            mode_changed_from_previous = previous_mode != mode_text
            if mode_changed_from_previous or mode_text not in {"isel", "qsel"}:
                value_edit.clear()
                value_edit.setModified(False)
            if mode_changed_from_previous or mode_text != "qsel":
                width_edit.clear()
                width_edit.setModified(False)
            mode_combo.setProperty(
                "figure_composer_source_selection_previous_mode", mode_text
            )
            if mode_text in {"isel", "qsel"} and not value_edit.text().strip():
                return
            update_from_controls()

        mode_combo.setProperty(
            "figure_composer_source_selection_previous_mode",
            mode_combo.currentData(),
        )
        mode_combo.activated.connect(mode_changed)
        value_edit.editingFinished.connect(update_from_controls)
        width_edit.editingFinished.connect(update_from_controls)

    def _common_source_selection_dims(
        self, source_names: Sequence[str]
    ) -> tuple[str, ...]:
        source_by_name = self._source_by_name()
        common: list[str] | None = None
        for source_name in source_names:
            data = self._source_selection_input_data(
                source_name, source_by_name.get(source_name)
            )
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
        dim: str,
        mode: str,
        value_text: str,
        width_text: str,
    ) -> None:
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

        changed = False
        skipped: list[str] = []
        source_by_name = self._source_by_name()
        source_list = list(self._recipe.sources)
        for source_name in self._selected_source_names():
            source = source_by_name.get(source_name)
            if source is None or source_name not in self._source_data:
                skipped.append(source_name)
                continue
            try:
                selection = selection_with_dimension(
                    _source_selection(source),
                    dim,
                    mode,
                    value,
                    width,
                )
                raw_data = self._source_selection_input_data(source_name, source)
                if raw_data is None:
                    skipped.append(source_name)
                    continue
                updated_source = _source_with_selection(source, selection)
                selected_data = self._source_data_from_selection(
                    source_name,
                    raw_data,
                    updated_source,
                )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                message = str(exc) or exc.__class__.__name__
                skipped.append(f"{source_name} ({message})")
                continue
            if selection_has_effect(selection):
                self._source_selection_base_data[source_name] = raw_data
            else:
                self._source_selection_base_data.pop(source_name, None)
            self._source_data[source_name] = selected_data
            for index, candidate in enumerate(source_list):
                if candidate.name == source_name:
                    source_list[index] = updated_source
                    break
            changed = True

        status_text = (
            "Selection was not applied to: " + ", ".join(skipped) if skipped else None
        )
        if not changed:
            self._set_source_validation_text(status_text)
            self._refresh_source_selection_editor()
            return
        self._recipe = self._recipe.model_copy(update={"sources": tuple(source_list)})
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
        self._set_source_validation_text(status_text)
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()

    @staticmethod
    def _source_data_from_selection(
        source_name: str,
        data: xr.DataArray,
        source: FigureSourceState,
    ) -> xr.DataArray:
        selected = _selected_source_data(data, source)
        return selected.rename(data.name).copy(deep=False)

    def _source_selection_input_data(
        self, source_name: str, source: FigureSourceState | None
    ) -> xr.DataArray | None:
        data = self._source_selection_base_data.get(source_name)
        if data is not None:
            return data
        selection_source = None if source is None else source.selection_source
        if selection_source is not None and selection_source != source_name:
            data = self._source_data.get(selection_source)
            if data is not None:
                return data
        return self._source_data.get(source_name)

    def _normalize_operation_source_selections(self) -> None:
        operations: list[FigureOperationState] = []
        source_list = list(self._recipe.sources)
        source_by_name = {source.name: source for source in source_list}
        reserved = set(source_by_name)
        reserved.update(self._source_data)
        changed = False
        for operation in self._recipe.operations:
            if not operation.map_selections:
                operations.append(operation)
                continue
            updated = self._operation_with_legacy_source_selections(
                operation,
                source_list=source_list,
                source_by_name=source_by_name,
                reserved=reserved,
            )
            operations.append(updated)
            changed = changed or updated != operation
        if not changed:
            return
        self._recipe = self._recipe.model_copy(
            update={"sources": tuple(source_list), "operations": tuple(operations)}
        )

    def _operation_with_legacy_source_selections(
        self,
        operation: FigureOperationState,
        *,
        source_list: list[FigureSourceState],
        source_by_name: dict[str, FigureSourceState],
        reserved: set[str],
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
        )
        return self._operation_without_map_selections(operation, alias)

    def _plot_slices_operation_with_shared_legacy_selection(
        self, operation: FigureOperationState
    ) -> FigureOperationState | None:
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
            return self._operation_without_map_selections(operation, fallback)
        if selection.isel or selection.mean_dims:
            return None

        dims = _operation_dim_names(self, operation)
        if not dims:
            return self._plot_slices_operation_with_legacy_qsel(
                operation, selection.qsel
            )
        if any(not _is_slice_kwarg_key(key, dims) for key in selection.qsel):
            return None
        updates = _selection_updates_from_kwargs(
            self,
            operation,
            {**_effective_slice_kwargs(self, operation), **selection.qsel},
            _effective_extra_kwargs(self, operation),
        )
        updates["map_selections"] = ()
        return operation.model_copy(update=updates)

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
    ) -> FigureOperationState:
        selection_by_source = {
            selection.source: selection for selection in operation.map_selections
        }
        sources = operation.sources or tuple(
            selection.source for selection in operation.map_selections
        )
        updated_sources: list[str] = []
        for source_name in sources:
            selection = selection_by_source.get(source_name)
            if selection is None or not selection_has_effect(selection):
                updated_sources.append(source_name)
                continue
            updated_sources.append(
                self._source_alias_for_legacy_selection(
                    selection,
                    source_list=source_list,
                    source_by_name=source_by_name,
                    reserved=reserved,
                )
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
                if source.name not in self._source_data:
                    base_data = self._source_data.get(source_name)
                    if base_data is not None:
                        self._source_selection_base_data[source.name] = base_data
                        try:
                            selected = self._source_data_from_selection(
                                source.name, base_data, source
                            )
                        except (IndexError, KeyError, TypeError, ValueError):
                            pass
                        else:
                            self._source_data[source.name] = selected
                return source.name

        alias = self._selected_source_alias(source_name, reserved)
        reserved.add(alias)
        base_source = source_by_name.get(
            source_name, FigureSourceState(name=source_name)
        )
        selected_source = _source_with_selection(
            self._source_with_name(base_source, alias).model_copy(
                update={"selection_source": source_name}
            ),
            selection.model_copy(update={"source": alias}),
        )
        source_by_name[alias] = selected_source
        source_list.append(selected_source)
        base_data = self._source_data.get(source_name)
        if base_data is None:
            return alias
        self._source_selection_base_data[alias] = base_data
        try:
            selected = _selected_data(self._source_data, selection)
        except (IndexError, KeyError, TypeError, ValueError):
            return alias
        if selected is not None:
            self._source_data[alias] = selected.copy(deep=False)
        return alias

    @staticmethod
    def _operation_without_map_selections(
        operation: FigureOperationState,
        source_name: str | None,
    ) -> FigureOperationState:
        updates: dict[str, typing.Any] = {"map_selections": ()}
        if operation.kind == FigureOperationKind.LINE:
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

    def _select_source_list_row_silent(self, source_name: str | None) -> None:
        self._updating_source_selection = True
        try:
            self._source_inspector_target = source_name
            if source_name is None:
                self.source_list.clearSelection()
                self.source_list.setCurrentIndex(QtCore.QModelIndex())
                return
            for row in range(self.source_list.topLevelItemCount()):
                item = self.source_list.topLevelItem(row)
                if self._source_name_from_list_item(item) == source_name:
                    self.source_list.setCurrentItem(item)
                    return
        finally:
            self._updating_source_selection = False

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
        operation_ids = tuple(
            operation.operation_id for operation in self._recipe.operations
        )
        current_item_ids = self.operation_list._operation_ids()
        reuse_items = current_item_ids == operation_ids
        self.operation_list.blockSignals(True)
        try:
            if not reuse_items:
                self.operation_list.clear()
                self.operation_list.addTopLevelItems(
                    [
                        QtWidgets.QTreeWidgetItem()
                        for _operation in self._recipe.operations
                    ]
                )
            for row, operation in enumerate(self._recipe.operations):
                item = self.operation_list.topLevelItem(row)
                if item is not None:  # pragma: no branch
                    self._update_operation_list_item(item, operation)
            if not reuse_items:
                self._set_selected_operation_ids_silent(selected_ids)
                if current_id is not None:
                    for row, operation in enumerate(self._recipe.operations):
                        if operation.operation_id == current_id:
                            self.operation_list.setCurrentItem(
                                self.operation_list.topLevelItem(row)
                            )
                            break
        finally:
            self.operation_list.blockSignals(False)
        self._sync_source_list_used_state()

    def _update_operation_list_item(
        self,
        item: QtWidgets.QTreeWidgetItem,
        operation: FigureOperationState,
    ) -> None:
        issues = self._operation_issues(operation)
        item.setText(
            _OPERATION_LIST_STEP_COLUMN, self._operation_display_text(operation)
        )
        item.setText(
            _OPERATION_LIST_STATUS_COLUMN,
            self._operation_status_text(issues),
        )
        item.setFlags(
            (
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsDragEnabled
            )
            & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled
        )
        item.setCheckState(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.CheckState.Checked
            if operation.enabled
            else QtCore.Qt.CheckState.Unchecked,
        )
        item.setData(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.ItemDataRole.UserRole,
            operation.operation_id,
        )
        item.setData(
            _OPERATION_LIST_TARGET_COLUMN,
            _OPERATION_LIST_TARGET_ROLE,
            self._operation_target_preview_descriptor(operation),
        )
        item.setData(
            _OPERATION_LIST_STATUS_COLUMN,
            _OPERATION_LIST_STATUS_ROLE,
            tuple(code for code, _detail in issues),
        )
        item.setSizeHint(_OPERATION_LIST_STEP_COLUMN, QtCore.QSize(0, 22))
        tooltip = self._operation_tooltip(operation)
        item.setToolTip(_OPERATION_LIST_STEP_COLUMN, tooltip)
        item.setData(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            tooltip,
        )
        target_text = _registry.spec_for(operation.kind).target_text(self, operation)
        target_description = (
            "No target" if target_text.casefold() == "none" else target_text
        )
        item.setToolTip(_OPERATION_LIST_TARGET_COLUMN, target_description)
        item.setData(
            _OPERATION_LIST_TARGET_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            target_description,
        )
        status_tooltip = "\n".join(
            f"{_OPERATION_STATUS_LABELS[code]}: {detail}" for code, detail in issues
        )
        item.setToolTip(_OPERATION_LIST_STATUS_COLUMN, status_tooltip)
        item.setData(
            _OPERATION_LIST_STATUS_COLUMN,
            QtCore.Qt.ItemDataRole.AccessibleDescriptionRole,
            status_tooltip,
        )
        item.setForeground(
            _OPERATION_LIST_STATUS_COLUMN,
            QtGui.QBrush(QtGui.QColor("darkRed")) if issues else QtGui.QBrush(),
        )

    def _set_current_operation_row_silent(
        self, index: int, *, preserve_selection: bool = True
    ) -> None:
        selected_ids = self._selected_operation_ids() if preserve_selection else set()
        was_blocked = self.operation_list.blockSignals(True)
        try:
            item = self.operation_list.topLevelItem(index)
            if item is None:
                self.operation_list.setCurrentIndex(QtCore.QModelIndex())
            else:
                self.operation_list.setCurrentItem(item)
            if preserve_selection and selected_ids:
                self._set_selected_operation_ids_silent(selected_ids)
        finally:
            self.operation_list.blockSignals(was_blocked)

    @staticmethod
    def _operation_id_for_item(
        item: QtWidgets.QTreeWidgetItem,
    ) -> str | None:
        operation_id = item.data(
            _OPERATION_LIST_STEP_COLUMN,
            QtCore.Qt.ItemDataRole.UserRole,
        )
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
            for row in range(self.operation_list.topLevelItemCount()):
                item = self.operation_list.topLevelItem(row)
                if item is None:
                    continue
                item.setSelected(self._operation_id_for_item(item) in operation_ids)
        finally:
            self.operation_list.blockSignals(was_blocked)

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
        setup = self._recipe.setup
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
        if spec.has_invalid_target(self, operation):
            issues.append(("invalid_target", spec.target_text(self, operation)))
        missing_sources = tuple(
            source
            for source in spec.source_names(operation)
            if source not in self._source_data
        )
        if missing_sources:
            issues.append(
                (
                    "missing_source",
                    ", ".join(self._source_display_names(missing_sources)),
                )
            )
        input_error = self._operation_input_error_text(operation)
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
        self.operation_list.setCurrentItem(self.operation_list.topLevelItem(indices[0]))
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
        row = self._current_operation_index()
        if row < 0 or row >= len(self._recipe.operations):
            return None
        return row, self._recipe.operations[row]

    def _current_operation_index(self) -> int:
        item = self.operation_list.currentItem()
        return -1 if item is None else self.operation_list.indexOfTopLevelItem(item)

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

    def _operation_duplicate_possible(self) -> bool:
        return bool(self._selected_operation_indices())

    def _operation_move_possible(self, offset: int) -> bool:
        indices = self._selected_operation_indices()
        if not indices:
            return False
        index_set = set(indices)
        if offset < 0:
            return any(index > 0 and index - 1 not in index_set for index in indices)
        return any(
            index < len(self._recipe.operations) - 1 and index + 1 not in index_set
            for index in indices
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
        options_getter: Callable[[FigureOperationState], Sequence[typing.Any]],
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
        defer_render: bool = False,
        rebuild_editor: bool = False,
        defer_editor_rebuild: bool = False,
        sync_axes: bool = True,
    ) -> bool:
        editable = self._editable_operations()
        if not editable:
            return False
        return self._update_operations_by_ids(
            (operation.operation_id for _index, operation in editable),
            updater,
            render=render,
            defer_render=defer_render,
            rebuild_editor=rebuild_editor,
            defer_editor_rebuild=defer_editor_rebuild,
            sync_axes=sync_axes,
        )

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
        operations = list(self._recipe.operations)
        changed = False
        preview_affected = False
        for index, operation in enumerate(operations):
            if operation.operation_id not in operation_id_set:
                continue
            updated = updater(index, operation)
            operation_changed = updated != operation
            changed = changed or operation_changed
            if operation_changed and self._operation_change_affects_preview(
                operation, updated
            ):
                preview_affected = True
            operations[index] = updated
        if not changed:
            return False
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
        if defer or self._active_editor_signal_widget is not None:
            delay_ms = (
                _EDITOR_CONTROL_RENDER_UPDATE_DELAY_MS
                if self._active_editor_signal_widget is not None
                else _PREVIEW_RENDER_UPDATE_DELAY_MS
            )
            self._queue_preview_render_update(delay_ms=delay_ms)
            return False
        self._cancel_preview_render_update()
        self._redraw_plot()
        self.sigInfoChanged.emit()
        return True

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
        can_paste = self._clipboard_step_payload() is not None
        if not indices:
            self.remove_operation_button.setEnabled(False)
            self.copy_operation_button.setEnabled(False)
            self.cut_operation_button.setEnabled(False)
            self.paste_operation_button.setEnabled(can_paste)
            return
        self.remove_operation_button.setEnabled(True)
        self.copy_operation_button.setEnabled(True)
        self.cut_operation_button.setEnabled(True)
        self.paste_operation_button.setEnabled(can_paste)

    @QtCore.Slot()
    @QtCore.Slot(int)
    @QtCore.Slot(str)
    @QtCore.Slot(object)
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
        self._maybe_redraw_plot()
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
        self._maybe_redraw_plot()
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
            defer_render=True,
            sync_axes=False,
        )
        erlab.interactive.utils.single_shot(self, 0, self._sync_axes_selector)

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, QtWidgets.QTreeWidgetItem)
    def _operation_current_item_changed(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        if current is not None and not self._operation_multi_select_event:
            operation_id = self._operation_id_for_item(current)
            if operation_id is not None:
                self._set_selected_operation_ids_silent({operation_id})
        self._operation_selection_changed()

    @QtCore.Slot()
    def _operation_selection_changed(self) -> None:
        current = self.operation_list.currentItem()
        current_id = None if current is None else self._operation_id_for_item(current)
        state = (current_id, frozenset(self._selected_operation_ids()))
        if state == self._operation_selection_state:
            return
        self._operation_selection_state = state
        if self._operation_selection_input_event:
            self._operation_axes_sync_pending = True
            self._queue_operation_editor_update()
            return
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

    def eventFilter(
        self, watched: QtCore.QObject | None, event: QtCore.QEvent | None
    ) -> bool:
        if self._handle_source_drag_event(event):
            return True
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
            self._operation_selection_input_event = True
            self._operation_multi_select_event = (
                self._operation_modifiers_enable_multi_selection(
                    input_event.modifiers()
                )
            )
            erlab.interactive.utils.single_shot(
                self, 0, self._clear_operation_selection_input_state
            )
        return super().eventFilter(watched, event)

    def _handle_source_drag_event(self, event: QtCore.QEvent | None) -> bool:
        if event is None or event.type() not in {
            QtCore.QEvent.Type.DragEnter,
            QtCore.QEvent.Type.DragMove,
            QtCore.QEvent.Type.Drop,
        }:
            return False
        if not isinstance(
            event, (QtGui.QDragEnterEvent, QtGui.QDragMoveEvent, QtGui.QDropEvent)
        ):
            return False
        mime = event.mimeData()
        if mime is None:
            return False
        if not self._source_drop_available(mime):
            return False
        if event.type() == QtCore.QEvent.Type.Drop and not self._add_sources_from_mime(
            mime
        ):
            return False
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()
        return True

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent | None) -> None:
        if self._handle_source_drag_event(event):
            return
        if event is not None:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent | None) -> None:
        if self._handle_source_drag_event(event):
            return
        if event is not None:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent | None) -> None:
        if self._handle_source_drag_event(event):
            return
        if event is not None:
            super().dropEvent(event)

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
        self._operation_selection_input_event = False
        self._operation_axes_sync_pending = False
        if viewport is not None and erlab.interactive.utils.qt_is_valid(self, viewport):
            viewport.removeEventFilter(self)

    def _disconnect_step_clipboard(self) -> None:
        clipboard = self._connected_step_clipboard
        self._connected_step_clipboard = None
        if clipboard is not None:
            with contextlib.suppress(TypeError, RuntimeError):
                clipboard.dataChanged.disconnect(self._update_step_action_buttons)

    def _source_data_history_state(
        self,
    ) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
        return dict(self._source_data), dict(self._source_selection_base_data)

    def _restore_source_data_history_state(
        self,
        state: tuple[Mapping[str, xr.DataArray], Mapping[str, xr.DataArray]],
    ) -> None:
        source_data, selection_base_data = state
        self._source_data = dict(source_data)
        self._source_selection_base_data = dict(selection_base_data)
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

    def _clear_operation_selection_input_state(self) -> None:
        if not erlab.interactive.utils.qt_is_valid(self):
            return
        self._operation_multi_select_event = False
        self._operation_selection_input_event = False

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

    @QtCore.Slot(QtWidgets.QTreeWidgetItem, int)
    def _operation_item_changed(
        self, item: QtWidgets.QTreeWidgetItem, column: int
    ) -> None:
        if self._updating_controls or column != _OPERATION_LIST_STEP_COLUMN:
            return
        operation_id = self._operation_id_for_item(item)
        for index, operation in enumerate(self._recipe.operations):
            if operation.operation_id == operation_id:
                updated = operation.model_copy(
                    update={
                        "enabled": item.checkState(_OPERATION_LIST_STEP_COLUMN)
                        == QtCore.Qt.CheckState.Checked,
                    }
                )
                if updated == operation:
                    return
                operations = list(self._recipe.operations)
                operations[index] = updated
                self._recipe = self._recipe.model_copy(
                    update={"operations": tuple(operations)}
                )
                if index == self._current_operation_index():
                    self._sync_axes_selector()
                    self._update_source_status(updated)
                self._refresh_step_section_button_texts()
                if self._operation_change_affects_preview(operation, updated):
                    self._notify_operation_preview_changed()
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
        return _source_display_label(source, name)

    def _source_display_names(self, names: Sequence[str]) -> tuple[str, ...]:
        return tuple(self._source_display_name(name) for name in names)

    def _source_detail_context_lines(self, name: str) -> tuple[str, ...]:
        source = self._source_by_name().get(name)
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
        usage_count = self._source_usage_count(name)
        if usage_count:
            suffix = "step" if usage_count == 1 else "steps"
            lines.append(f"Used by {usage_count} recipe {suffix}")
        else:
            lines.append("Not used by any recipe steps")
        return tuple(lines)

    def _source_tooltip(self, name: str) -> str:
        lines = [
            source_metadata_tooltip(
                self._source_by_name().get(name), name, self._source_data.get(name)
            ),
            *self._source_detail_context_lines(name),
        ]
        if self._source_refresh_available(name):
            lines.append("Can refresh from the linked ImageTool")
        return "\n".join(lines)

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
        self._maybe_redraw_plot()
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
        duplicate_action.setEnabled(self._operation_duplicate_possible())
        duplicate_action.triggered.connect(self._duplicate_current_operation)
        menu.addAction(duplicate_action)

        move_up_action = QtGui.QAction("Move Up", menu)
        move_up_action.setObjectName("figureComposerContextMoveStepUpAction")
        move_up_action.setEnabled(self._operation_move_possible(-1))
        move_up_action.triggered.connect(self._move_current_operation_up)
        menu.addAction(move_up_action)

        move_down_action = QtGui.QAction("Move Down", menu)
        move_down_action.setObjectName("figureComposerContextMoveStepDownAction")
        move_down_action.setEnabled(self._operation_move_possible(1))
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
        self.operation_list.setCurrentItem(
            self.operation_list.topLevelItem(len(operations) - 1)
        )
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
        operations = tuple(self._recipe.operations[index] for index in indices)
        source_names = tuple(
            dict.fromkeys(
                source_name
                for operation in operations
                for source_name in self._operation_source_dependency_names(operation)
            )
        )
        source_by_name = {source.name: source for source in self._recipe.sources}
        sources = tuple(
            source_by_name.get(source_name, FigureSourceState(name=source_name))
            for source_name in source_names
        )
        source_data = {
            source_name: self._source_data[source_name].copy(deep=False)
            for source_name in source_names
            if source_name in self._source_data
        }
        selection_base_data = {
            source_name: self._source_selection_base_data[source_name].copy(deep=False)
            for source_name in source_names
            if source_name in self._source_selection_base_data
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
        unique_sources: list[FigureSourceState] = []
        seen_sources: set[str] = set()
        for source in sources:
            if source.name in seen_sources:
                continue
            seen_sources.add(source.name)
            unique_sources.append(source)

        rename_map: dict[str, str] = {}
        for source in unique_sources:
            if preserve_existing and source.name in reserved:
                rename_map[source.name] = source.name
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

        renamed_sources: list[FigureSourceState] = []
        renamed_source_data: dict[str, xr.DataArray] = {}
        for source in unique_sources:
            pasted_name = rename_map[source.name]
            renamed_source = self._source_with_name(source, pasted_name)
            if source.selection_source is not None:
                renamed_source = renamed_source.model_copy(
                    update={
                        "selection_source": rename_map.get(
                            source.selection_source, source.selection_source
                        )
                    }
                )
            if pasted_name not in existing_source_names:
                renamed_sources.append(renamed_source)
                existing_source_names.add(pasted_name)
            if source.name in source_data and (
                pasted_name not in self._source_data or not preserve_existing
            ):
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
        for field in (
            "method_plot_x",
            "method_plot_y",
            "method_plot_xerr",
            "method_plot_yerr",
        ):
            state = getattr(operation, field)
            if state is not None:
                updates[field] = state.model_copy(
                    update={"source": rename_map.get(state.source, state.source)}
                )
        if operation.hv_overlay_source is not None:
            updates["hv_overlay_source"] = rename_map.get(
                operation.hv_overlay_source, operation.hv_overlay_source
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
        operations, sources, source_data, selection_base_data = payload
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
        renamed_selection_base_data = {
            rename_map.get(source_name, source_name): data
            for source_name, data in selection_base_data.items()
            if rename_map.get(source_name, source_name) in source_names
        }
        self._source_selection_base_data.update(renamed_selection_base_data)
        self._recipe = self._recipe.model_copy(
            update={
                "sources": tuple(source_list),
                "operations": tuple(operation_list),
            }
        )
        self._normalize_operation_source_selections()
        self._refresh_source_list()
        if renamed_source_data or renamed_selection_base_data:
            self.sigDataChanged.emit()
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
            operation.operation_id: operation for operation in self._recipe.operations
        }
        if (
            len(ordered_ids) != len(operation_ids)
            or len(ordered_ids) != len(operation_by_id)
            or set(ordered_ids) != set(operation_by_id)
        ):
            self._refresh_operation_list()
            return
        current_order = tuple(
            operation.operation_id for operation in self._recipe.operations
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

        operations = tuple(
            operation_by_id[operation_id] for operation_id in ordered_ids
        )
        self._recipe = self._recipe.model_copy(update={"operations": operations})
        self._finish_operation_structure_change(selected_id_set, current_operation_id)

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
        self._recipe = self._recipe.model_copy(
            update={"operations": (*self._recipe.operations, operation)}
        )
        self._normalize_operation_source_selections()
        self._apply_recipe_to_controls()
        self.operation_list.setCurrentItem(
            self.operation_list.topLevelItem(len(self._recipe.operations) - 1)
        )
        self._maybe_redraw_plot()
        self.sigInfoChanged.emit()
        self._write_state()

    def add_sources(
        self,
        sources: Sequence[FigureSourceState],
        source_data: Mapping[str, xr.DataArray],
    ) -> None:
        """Add or update source data without changing existing recipe steps.

        This supports appending operations and the manager's source-only workflow. The
        source list, backing data, preview, persistent state, and data-dirty signals are
        updated together so workspace saves include the new source data.
        """
        existing = {source.name: source for source in self._recipe.sources}
        reserved = set(existing)
        reserved.update(self._source_data)
        source_data_updates: dict[str, xr.DataArray] = {}
        selection_base_updates: dict[str, xr.DataArray] = {}
        clear_selection_bases: set[str] = set()
        skipped: list[str] = []
        for source in sources:
            incoming_name = source.name
            data = source_data.get(incoming_name)
            if data is None:
                continue
            linked_matches = (
                [
                    candidate
                    for candidate in existing.values()
                    if candidate.node_uid == source.node_uid
                ]
                if source.node_uid is not None
                else []
            )
            target_name = incoming_name
            existing_source = existing.get(target_name)
            same_linked_source = (
                existing_source is not None
                and existing_source.node_uid is not None
                and existing_source.node_uid == source.node_uid
            )
            if not same_linked_source and len(linked_matches) == 1:
                existing_source = linked_matches[0]
                target_name = existing_source.name
                same_linked_source = True
            if existing_source is not None and not same_linked_source:
                target_name = self._source_unique_alias(target_name, reserved)
                source = source.model_copy(update={"name": target_name})
                selected_data = data
            else:
                reserved.add(target_name)
                if existing_source is None:
                    selected_data = data
                else:
                    try:
                        source, selected_data = self._replacement_source_data(
                            target_name,
                            source,
                            data,
                            existing_source,
                            keep_selection_source=True,
                        )
                    except (IndexError, KeyError, TypeError, ValueError) as exc:
                        message = str(exc) or exc.__class__.__name__
                        skipped.append(f"{target_name} ({message})")
                        continue
            existing[target_name] = source
            source_data_updates[target_name] = selected_data
            if same_linked_source and _source_has_selection(source):
                selection_base_updates[target_name] = data
            else:
                clear_selection_bases.add(target_name)
        if not source_data_updates:
            if skipped:
                self._set_source_status_text(
                    "Could not update source data for: " + ", ".join(skipped)
                )
            return
        self._source_data.update(source_data_updates)
        self._source_selection_base_data.update(selection_base_updates)
        for source_name in clear_selection_bases - selection_base_updates.keys():
            self._source_selection_base_data.pop(source_name, None)
        ordered_sources = tuple(existing[name] for name in existing)
        self._recipe = self._recipe.model_copy(update={"sources": ordered_sources})
        self._normalize_operation_source_selections()
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()
        self._set_source_status_text(
            "Could not update source data for: " + ", ".join(skipped)
            if skipped
            else None
        )

    def _replacement_source_data(
        self,
        alias: str,
        source: FigureSourceState,
        data: xr.DataArray,
        existing_source: FigureSourceState | None,
        *,
        keep_selection_source: bool,
    ) -> tuple[FigureSourceState, xr.DataArray]:
        replacement = self._source_with_name(source, alias)
        if source.selection_source == source.name:
            replacement = replacement.model_copy(update={"selection_source": alias})
        if (
            existing_source is not None
            and _source_has_selection(existing_source)
            and not _source_has_selection(replacement)
        ):
            replacement = _source_with_selection(
                replacement.model_copy(
                    update={
                        "selection_source": (
                            existing_source.selection_source
                            if keep_selection_source
                            else alias
                        )
                    }
                ),
                _source_selection(existing_source),
            )
        return replacement, self._source_data_from_selection(alias, data, replacement)

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
        source_list = list(self._recipe.sources)
        existing_source: FigureSourceState | None = None
        existing_index: int | None = None
        for candidate_index, candidate_source in enumerate(source_list):
            if candidate_source.name == alias:
                existing_source = candidate_source
                existing_index = candidate_index
                break
        else:
            if alias not in self._source_data:
                return False
            existing_source = None
            source_list.append(self._source_with_name(source, alias))
            existing_index = len(source_list) - 1

        if existing_index is None:  # pragma: no cover
            raise RuntimeError("source replacement index was not resolved")
        try:
            replacement, selected_data = self._replacement_source_data(
                alias,
                source,
                data,
                existing_source,
                keep_selection_source=False,
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            message = str(exc) or exc.__class__.__name__
            self._set_source_status_text(
                f"Could not refresh source “{alias}”: {message}"
            )
            return False
        source_list[existing_index] = replacement

        self._source_data[alias] = selected_data
        if _source_has_selection(replacement):
            self._source_selection_base_data[alias] = data
        else:
            self._source_selection_base_data.pop(alias, None)
        self._recipe = self._recipe.model_copy(update={"sources": tuple(source_list)})
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()
        self._set_source_status_text(None)
        return True

    def remove_source(self, name: str) -> bool:
        """Remove an unused source from this figure."""
        if not self._source_removable(name):
            return False

        source_list = tuple(
            source for source in self._recipe.sources if source.name != name
        )
        updates: dict[str, typing.Any] = {"sources": source_list}
        if self._recipe.primary_source == name:
            updates["primary_source"] = source_list[0].name
        self._recipe = self._recipe.model_copy(update=updates)
        self._source_data.pop(name, None)
        self._source_selection_base_data.pop(name, None)
        self._refresh_operation_list()
        self._refresh_step_section_button_texts()
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
        self.sigDataChanged.emit()
        self.sigInfoChanged.emit()
        self._write_state()
        return True

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
        self._flush_pending_editor_commits()
        for page in self._operation_editor_pages:
            self.step_editor_stack.removeWidget(page)
            self._retire_editor_widget(page)
        self._operation_editor_pages.clear()

    def _clear_step_source_controls(self) -> None:
        self._clear_form_layout(self.step_source_controls_layout)

    def _new_step_form_page(
        self, object_name: str
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        page = _FigureComposerStepEditorPage(self.editor_tabs, self.step_editor_stack)
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
            page = self.step_editor_stack.widget(0)
            if page is None:  # pragma: no cover - guarded by count()
                break
            page.hide()
            self.step_editor_stack.removeWidget(page)

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
        self.step_editor_scroll.updateGeometry()
        self.step_inspector.updateGeometry()

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
        current_page = self.step_editor_stack.currentWidget()
        if isinstance(current_page, _FigureComposerStepEditorPage):
            current_page._refresh_background()
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
            self._set_step_source_status_text("Select a step to choose data sources.")
            return
        if not _registry.spec_for(operation.kind).uses_source_section(operation):
            self._set_step_source_status_text(None)
            return
        if (input_error := self._operation_input_error_text(operation)) is not None:
            self._set_step_source_status_text(f"Invalid input: {input_error}")
            return
        if (
            render_error := self._operation_render_errors.get(operation.operation_id)
        ) is not None:
            self._set_step_source_status_text(f"Render error: {render_error}")
            return
        selected_sources = self._selected_sources_for_operation(operation)
        missing = [
            source for source in selected_sources if source not in self._source_data
        ]
        if missing:
            self._set_step_source_status_text(
                "Missing sources: " + ", ".join(self._source_display_names(missing))
            )
        elif selected_sources:
            self._set_step_source_status_text(None)
        else:
            self._set_step_source_status_text("This step does not read a data source.")

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
            self._operation_axes_sync_pending = False
            return
        if self._operation_editor_rebuild_must_wait():
            self._schedule_queued_operation_editor_update()
            return
        self._operation_editor_update_pending = False
        if self._operation_axes_sync_pending:
            self._operation_axes_sync_pending = False
            self._sync_axes_selector()
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
    def _add_form_section(
        layout: QtWidgets.QFormLayout,
        title: str,
        *,
        object_name: str | None = None,
    ) -> QtWidgets.QWidget:
        """Add a lightweight native section header to a form layout."""
        section = QtWidgets.QWidget(layout.parentWidget())
        if object_name:
            section.setObjectName(object_name)
        section.setProperty("figureComposerSectionHeader", True)
        section_layout = QtWidgets.QHBoxLayout(section)
        top_margin = 8 if layout.rowCount() else 0
        section_layout.setContentsMargins(0, top_margin, 0, 2)
        section_layout.setSpacing(6)

        label = QtWidgets.QLabel(title, section)
        label.setProperty("figureComposerSectionHeaderLabel", True)
        label_font = label.font()
        label_font.setBold(True)
        label.setFont(label_font)
        line = QtWidgets.QFrame(section)
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        if object_name:
            line.setObjectName(f"{object_name}Line")

        section_layout.addWidget(label)
        section_layout.addWidget(line, 1)
        layout.addRow(section)
        return section

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
        self._flush_restore_work()
        self._flush_pending_editor_commits()
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
        self._flush_pending_editor_commits()
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
        self._normalize_operation_source_selections()
        self._apply_recipe_to_controls()
        self._sync_figure_window_to_recipe_setup()
        if self._dataset_restore_in_progress:
            self._mark_preview_pixmap_stale()
            return
        _render_preview(self)

    def set_source_data(self, source_data: Mapping[str, xr.DataArray]) -> None:
        self._source_data = dict(source_data)
        self._source_selection_base_data.clear()
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
        primary_source = self._recipe.primary_source
        if primary_source in self._source_data:
            primary_data, _already_selected = self._persistence_source_data(
                primary_source
            )
        else:
            primary_data = self.tool_data
        items = {erlab.interactive.utils._SAVED_TOOL_DATA_NAME: primary_data}
        for source_name in self._source_data:
            if source_name == self._recipe.primary_source:
                continue
            if source_name == erlab.interactive.utils._SAVED_TOOL_DATA_NAME:
                raise ValueError(
                    "Figure source names cannot use the reserved saved-tool data name"
                )
            data, _already_selected = self._persistence_source_data(source_name)
            items[source_name] = data
        return items

    def _persistence_source_data(self, source_name: str) -> tuple[xr.DataArray, bool]:
        data = self._source_data[source_name]
        source = self._recipe_source(source_name)
        if source is None or not _source_has_selection(source):
            return data, False
        base_data = self._source_selection_base_data.get(source_name)
        if base_data is not None:
            return base_data, False
        if (
            source.selection_source is not None
            and source.selection_source != source_name
        ):
            base_data = self._source_data.get(source.selection_source)
            if base_data is not None:
                return base_data, False
        return data, True

    def _embedded_selected_source_names(self, ds: xr.Dataset) -> tuple[str, ...]:
        references = self._saved_tool_data_references(ds)
        selected_names: list[str] = []
        for source in self._recipe.sources:
            if source.name not in self._source_data:
                continue
            _data, already_selected = self._persistence_source_data(source.name)
            variable_name = (
                erlab.interactive.utils._SAVED_TOOL_DATA_NAME
                if source.name == self._recipe.primary_source
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
            for source in self._recipe.sources
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
        source_data = dict(self._source_data)
        selection_base_data: dict[str, xr.DataArray] = {}
        embedded_selected_names = self._persisted_selected_source_names(ds)

        def restored_source_data(
            source: FigureSourceState, data: xr.DataArray
        ) -> xr.DataArray:
            if (
                not _source_has_selection(source)
                or source.name in embedded_selected_names
            ):
                return data
            try:
                selected = self._source_data_from_selection(source.name, data, source)
            except (IndexError, KeyError, TypeError, ValueError):
                logger.debug(
                    "Could not apply saved Figure Composer source selection for %s",
                    source.name,
                    exc_info=True,
                )
                return data
            selection_base_data[source.name] = data
            return selected

        primary_data = data_items.get(erlab.interactive.utils._SAVED_TOOL_DATA_NAME)
        changed = False
        if (
            primary_data is not None
            and source_data.get(self._recipe.primary_source) is not primary_data
        ):
            current_primary = source_data.get(self._recipe.primary_source)
            if current_primary is not None:
                primary_data = primary_data.rename(current_primary.name)
            else:
                tool_data_name = ds.attrs.get("tool_data_name", "<none-value>")
                if tool_data_name == "<none-value>":
                    tool_data_name = None
                primary_data = primary_data.rename(tool_data_name)
            primary_source = self._recipe_source(self._recipe.primary_source)
            if primary_source is not None:
                primary_data = restored_source_data(primary_source, primary_data)
            source_data[self._recipe.primary_source] = primary_data
            changed = True
        for source in self._recipe.sources:
            if source.name == self._recipe.primary_source:
                continue
            source_data_item = data_items.get(source.name)
            if source_data_item is None:
                continue
            source_data[source.name] = restored_source_data(source, source_data_item)
            changed = True
        for source in self._recipe.sources:
            if source.name in source_data or not _source_has_selection(source):
                continue
            if source.selection_source is None:
                continue
            source_data_item = source_data.get(source.selection_source)
            if source_data_item is None:
                continue
            source_data[source.name] = restored_source_data(source, source_data_item)
            changed = True
        if not changed:
            self._restore_persisted_preview_cache(ds)
            self._queue_post_restore_redraw_if_needed(ds)
            return
        self.set_source_data(source_data)
        self._source_selection_base_data.update(selection_base_data)
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
            preview.width() > _PERSISTED_PREVIEW_CACHE_SIZE.width()
            or preview.height() > _PERSISTED_PREVIEW_CACHE_SIZE.height()
        ):
            return preview.scaled(
                _PERSISTED_PREVIEW_CACHE_SIZE,
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
            with self._figure_options_context(), _figure_style_context():
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

        with self._figure_options_context(), _figure_style_context():
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

    @staticmethod
    def _source_selection_replay_operations(
        source: FigureSourceState,
    ) -> tuple[provenance.ToolProvenanceOperation, ...]:
        operations: list[provenance.ToolProvenanceOperation] = []
        if source.isel:
            operations.append(
                provenance.IselOperation(
                    kwargs=typing.cast(
                        "dict[Hashable, typing.Any]",
                        dict(source.isel),
                    )
                )
            )
        if source.qsel:
            operations.append(
                provenance.QSelOperation(
                    kwargs=typing.cast(
                        "dict[Hashable, typing.Any]",
                        dict(source.qsel),
                    )
                )
            )
        if source.mean_dims:
            operations.append(
                provenance.QSelAggregationOperation(
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
        data = self._source_data.get(source.name)
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
        script_input: provenance.ScriptInput,
        name: str,
    ) -> provenance.ScriptInput:
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
    ) -> provenance.ScriptInput | None:
        base_source = source_by_name.get(source.selection_source or source.name)
        base_input = None if base_source is None else base_source.to_script_input()
        if base_input is None:
            base_input = source.to_script_input()
        if base_input is None:
            return None
        base_spec = base_input.parsed_provenance_spec()
        if base_spec is None:
            return None

        operations = self._source_selection_replay_operations(source)
        if not operations:
            return self._script_input_with_name(base_input, display_name)
        try:
            base_replay_spec = provenance.to_replay_provenance_spec(base_spec)
            if base_replay_spec is None:
                return None
            selected_spec = base_replay_spec.append_replay_stage(
                provenance.public_data(*operations)
            )
        except (TypeError, ValueError, pydantic.ValidationError):
            return None
        return provenance.ScriptInput(
            name=display_name,
            node_uid=base_input.node_uid,
            node_snapshot_token=base_input.node_snapshot_token,
            provenance_spec=selected_spec.model_dump(mode="json"),
        )

    def _display_code_source_plan(
        self,
    ) -> tuple[
        tuple[provenance.ScriptInput, ...],
        frozenset[str],
        dict[str, str],
    ]:
        source_by_name = self._source_by_name()
        used_sources = self._direct_sources_used_by_recipe(
            enabled_only=True, executable_only=True
        )
        used_code_names = set(_FIGURE_CODE_RESERVED_NAMES)
        if self._recipe.setup.layout_mode == "gridspec":
            source_names = self._source_names()
            used_code_names.update(
                _gridspec_reserved_axis_code_names(
                    self._recipe.setup, reserved_names=source_names
                )
            )
            used_code_names.update(
                _gridspec_axis_code_names(
                    self._recipe.setup, reserved_names=source_names
                ).values()
            )
        script_inputs: list[provenance.ScriptInput] = []
        script_input_names: set[str] = set()
        skip_source_selection_names: set[str] = set()
        source_name_map: dict[str, str] = {}

        def append_script_input(script_input: provenance.ScriptInput | None) -> None:
            if script_input is None or script_input.name in script_input_names:
                return
            script_inputs.append(script_input)
            script_input_names.add(script_input.name)

        for source in self._recipe.sources:
            if source.name not in used_sources:
                continue
            display_name = self._source_display_code_name(
                source,
                used_names=used_code_names,
            )
            script_input: provenance.ScriptInput | None = None
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
    ) -> provenance.ToolProvenanceSpec | None:
        del flush_deferred_restore
        script_inputs, skip_source_selection_names, source_name_map = (
            self._display_code_source_plan()
        )
        if not script_inputs:
            return None
        return provenance.script(
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
        if not names:
            return
        for source in self._recipe.sources:
            if source.name in names:
                self._source_data.pop(source.name, None)
                self._source_selection_base_data.pop(source.name, None)
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()

    def refresh_from_sources(self, source_data: Mapping[str, xr.DataArray]) -> None:
        source_by_name = self._source_by_name()
        skipped: list[str] = []
        changed = False
        for source_name, data in source_data.items():
            source = source_by_name.get(source_name)
            if source is None:
                self._source_data[source_name] = data
                self._source_selection_base_data.pop(source_name, None)
                changed = True
                continue
            try:
                selected_data = self._source_data_from_selection(
                    source_name, data, source
                )
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                message = str(exc) or exc.__class__.__name__
                skipped.append(f"{source_name} ({message})")
                continue
            self._source_data[source_name] = selected_data
            if _source_has_selection(source):
                self._source_selection_base_data[source_name] = data
            else:
                self._source_selection_base_data.pop(source_name, None)
            changed = True
        if skipped:
            self._set_source_status_text(
                "Could not refresh source data for: " + ", ".join(skipped)
            )
        elif changed:
            self._set_source_status_text(None)
        if not changed:
            self._refresh_source_controls()
            return
        self._refresh_source_list()
        self._update_source_section()
        self._maybe_redraw_plot()
