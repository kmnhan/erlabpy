"""Interactive tool for setting initial conditions for curve fitting."""

from __future__ import annotations

import collections
import contextlib
import importlib
import threading
import time
import traceback
import typing

import numpy as np
import pydantic
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import erlab.interactive.utils

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    import lmfit
    import varname
    import xarray as xr
else:
    import lazy_loader as _lazy

    varname = _lazy.load("varname")
    lmfit = _lazy.load("lmfit")


class _PythonCodeEditor(QtWidgets.QTextEdit):
    TAB_SPACES = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        self.highlighter = erlab.interactive.utils.PythonHighlighter(self.document())

    def keyPressEvent(self, e: QtGui.QKeyEvent | None) -> None:
        if e is not None:
            key = e.key()
            mods = e.modifiers()

            is_tab = key == QtCore.Qt.Key.Key_Tab
            is_backtab = key == QtCore.Qt.Key.Key_Backtab
            shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)

            if (is_tab or is_backtab) and not (
                mods & QtCore.Qt.KeyboardModifier.ControlModifier
            ):
                cursor = self.textCursor()
                if is_backtab or (is_tab and shift):
                    self._unindent(cursor, self.TAB_SPACES)
                else:
                    self._indent(cursor, self.TAB_SPACES)
                e.accept()
                return

        super().keyPressEvent(e)

    def _indent(self, cursor: QtGui.QTextCursor, n: int) -> None:
        text = " " * n
        cursor.beginEditBlock()
        if cursor.hasSelection():
            self._for_each_selected_block(cursor, lambda c: c.insertText(text))
        else:
            cursor.insertText(text)
        cursor.endEditBlock()

    def _unindent(self, cursor: QtGui.QTextCursor, n: int) -> None:
        cursor.beginEditBlock()

        def unindent_block(c: QtGui.QTextCursor):
            c.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)

            # Remove one literal tab if present
            c.movePosition(
                QtGui.QTextCursor.MoveOperation.Right,
                QtGui.QTextCursor.MoveMode.KeepAnchor,
                1,
            )
            if c.selectedText() == "\t":
                c.removeSelectedText()
                return
            c.clearSelection()

            # Otherwise remove up to n leading spaces
            removed = 0
            while removed < n:
                c.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)
                c.movePosition(
                    QtGui.QTextCursor.MoveOperation.Right,
                    QtGui.QTextCursor.MoveMode.KeepAnchor,
                    1,
                )
                if c.selectedText() == " ":
                    c.removeSelectedText()
                    removed += 1
                else:
                    c.clearSelection()
                    break

        if cursor.hasSelection():
            self._for_each_selected_block(cursor, unindent_block)
        else:
            unindent_block(cursor)

        cursor.endEditBlock()

    def _for_each_selected_block(self, cursor: QtGui.QTextCursor, fn):
        start = cursor.selectionStart()
        end = cursor.selectionEnd()

        c = QtGui.QTextCursor(cursor.document())
        c.setPosition(start)
        start_block = c.blockNumber()

        c.setPosition(end)
        end_block = c.blockNumber()

        c.setPosition(start)
        c.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)

        for _ in range(start_block, end_block + 1):
            fn(c)
            c.movePosition(QtGui.QTextCursor.MoveOperation.NextBlock)


class _ExpressionInitScriptDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setWindowTitle("Edit Expression Model Init Script")

        layout = QtWidgets.QVBoxLayout(self)

        self.text_edit = _PythonCodeEditor(self)
        self.text_edit.setAcceptRichText(False)
        self.text_edit.setPlaceholderText("# Write Python code here")
        self.text_edit.setToolTip(
            "Write Python code here to define functions or variables "
            "that will be available when evaluating the model expression."
        )
        layout.addWidget(self.text_edit)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_script(self) -> str | None:
        txt = self.text_edit.toPlainText().strip()
        if txt:
            return txt
        return None


class _SnapCursorLine(pg.InfiniteLine):
    _sigDragStarted = QtCore.Signal(object)  # :meta private:

    def value(self) -> float:
        return float(super().value())

    def mouseDragEvent(self, ev) -> None:
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
                self._sigDragStarted.emit(self)
            ev.accept()

            if self.moving:
                new_position = self.cursorOffset + self.mapToParent(ev.pos())
                if self.angle % 180 == 0:
                    self.temp_value = new_position.y()
                elif self.angle % 180 == 90:
                    self.temp_value = new_position.x()

                self.sigDragged.emit(self)
                if ev.isFinish():
                    self.moving = False
                self.sigPositionChangeFinished.emit(self)


class _PeakPositionLine(pg.InfiniteLine):
    def __init__(self, tool: Fit1DTool, peak_index: int, *args, **kwargs) -> None:
        self._tool = tool
        self._peak_index = peak_index
        self._start_height: float | None = None
        self._start_width: float | None = None
        kwargs["movable"] = True
        super().__init__(*args, **kwargs)
        self.addMarker("o", -0.5)

    def mouseDragEvent(self, ev) -> None:
        if not self.movable:
            ev.ignore()
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
                self._start_height = self._tool._get_peak_param_value(
                    self._peak_index, "height"
                )
            ev.accept()

            if not self.moving:
                return

            pos = self.cursorOffset + self.mapToParent(ev.pos())
            self._tool._set_peak_param_value(self._peak_index, "center", pos.x())
            if self._start_height is not None:
                delta = self.mapToParent(ev.pos() - ev.buttonDownPos()).y()
                self._tool._set_peak_param_value(
                    self._peak_index, "height", self._start_height + delta
                )

        elif QtCore.Qt.MouseButton.RightButton in ev.buttons():
            if ev.isStart():
                self._start_width = self._tool._get_peak_param_value(
                    self._peak_index, "width"
                )
            ev.accept()

            if self._start_width is not None:
                val = self.mapToParent(ev.buttonDownPos() - ev.pos()).y()
                y0, _, y1, _ = self.boundingRect().getCoords()
                span = abs(y1 - y0)
                if span > 0:
                    amount = val / span
                    self._tool._set_peak_param_value(
                        self._peak_index, "width", self._start_width + amount
                    )
        if ev.isFinish():
            self.moving = False
            self.sigPositionChangeFinished.emit(self)

        self.setMouseHover(self.moving)

    def refresh_pos(self) -> None:
        center = self._tool._get_peak_param_value(self._peak_index, "center")
        if center is not None and np.isfinite(center):
            self.setPos(center)
            self.show()
        else:
            self.hide()


class _ParameterEditDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, model: _ParameterTableModel, parent=None) -> None:
        super().__init__(parent)
        self._model = model

    def createEditor(self, parent, option, index):
        return QtWidgets.QLineEdit(parent)

    def setEditorData(self, editor, index):
        if isinstance(editor, QtWidgets.QLineEdit):
            text = self._model.edit_value_string(index.row(), index.column())
            editor.setText(text)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QtWidgets.QLineEdit):
            model.setData(index, editor.text(), QtCore.Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)


class _ParameterTableModel(QtCore.QAbstractTableModel):
    sigParamsChanged = QtCore.Signal()

    _COLUMN_NAMES = ("Parameter", "Value", "StdErr", "Min", "Max", "Vary")

    def __init__(
        self,
        params: lmfit.Parameters,
        params_from_coord: dict[str, str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._params = params
        self._params_from_coord: dict[str, str] = params_from_coord
        self._param_names = list(params.keys()) if params is not None else []

    @property
    def params(self) -> lmfit.Parameters:
        return self._params

    def set_params(
        self, params: lmfit.Parameters, params_from_coord: dict[str, str]
    ) -> None:
        self.beginResetModel()
        self._params = params
        self._params_from_coord = params_from_coord
        self._param_names = list(params.keys()) if params is not None else []
        self.endResetModel()
        self.sigParamsChanged.emit()

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return len(self._param_names)

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return len(self._COLUMN_NAMES)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> typing.Any:
        if (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and orientation == QtCore.Qt.Orientation.Horizontal
        ):
            return self._COLUMN_NAMES[section]
        return None

    def data(
        self,
        index: QtCore.QModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> typing.Any:
        if not index.isValid():
            return None
        param = self._params[self._param_names[index.row()]]
        col = index.column()

        if role in (
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ):
            if col == 0:
                return param.name
            if col == 1:
                return self._format_value(param.value)
            if col == 2:
                if param.stderr is None:
                    return ""
                return self._format_scientific(param.stderr)
            if col == 3:
                return "-inf" if not np.isfinite(param.min) else float(param.min)
            if col == 4:
                return "inf" if not np.isfinite(param.max) else float(param.max)
            if col == 5:
                return None

        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return self._param_tooltip(param)

        if role == QtCore.Qt.ItemDataRole.CheckStateRole and col == 5:
            if self._is_expr_param(param):
                return None
            return (
                QtCore.Qt.CheckState.Checked
                if param.vary
                else QtCore.Qt.CheckState.Unchecked
            )
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        param = self._params[self._param_names[index.row()]]
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if (
            index.column() == 1
            and not self._is_expr_param(param)
            and not self._is_param_from_coord(param)
        ):
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        if index.column() in (3, 4) and not self._is_expr_param(param):
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        if index.column() == 5 and not self._is_expr_param(param):
            flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
        if index.column() == 5 and self._is_expr_param(param):
            flags &= ~QtCore.Qt.ItemFlag.ItemIsEnabled
        return flags

    def setData(
        self,
        index: QtCore.QModelIndex,
        value: typing.Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False
        param = self._params[self._param_names[index.row()]]
        if self._is_expr_param(param):
            return False
        col = index.column()

        if col == 5 and role in (
            QtCore.Qt.ItemDataRole.CheckStateRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ):
            try:
                check_state = QtCore.Qt.CheckState(value)
            except Exception:
                check_state = (
                    QtCore.Qt.CheckState.Checked
                    if bool(value)
                    else QtCore.Qt.CheckState.Unchecked
                )
            param.vary = check_state == QtCore.Qt.CheckState.Checked
            if self._is_param_from_coord(param) and param.vary:
                del self._params_from_coord[param.name]
            self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.CheckStateRole])
            self.sigParamsChanged.emit()
            return True

        if role != QtCore.Qt.ItemDataRole.EditRole:
            return False

        try:
            if col == 1:
                param.value = float(value)
            elif col == 3:
                param.min = self._parse_bound(value, default=-np.inf)
            elif col == 4:
                param.max = self._parse_bound(value, default=np.inf)
            else:
                return False
        except (TypeError, ValueError):
            return False

        if param.min > param.max:
            param.max = param.min
        if param.value < param.min:
            param.value = param.min
        if param.value > param.max:
            param.value = param.max

        top_left = self.index(index.row(), 1)
        bottom_right = self.index(index.row(), 4)
        self.dataChanged.emit(
            top_left,
            bottom_right,
            [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole],
        )
        self.sigParamsChanged.emit()
        return True

    def edit_value_string(self, row: int, column: int) -> str:
        param = self._params[self._param_names[row]]
        if column == 1:
            return self._format_value(param.value)
        if column == 2:
            if param.stderr is None:
                return ""
            return self._format_scientific(param.stderr)
        if column == 3:
            return self._format_bound(param.min, default="-inf")
        if column == 4:
            return self._format_bound(param.max, default="inf")
        return ""

    def param_name(self, row: int) -> str:
        return self._param_names[row]

    def param_at(self, row: int) -> lmfit.Parameter:
        return self._params[self._param_names[row]]

    def _is_param_from_coord(self, param: lmfit.Parameter) -> bool:
        return str(param.name) in self._params_from_coord

    @staticmethod
    def _is_expr_param(param: lmfit.Parameter) -> bool:
        return bool(param.expr)

    @staticmethod
    def _param_tooltip(param: lmfit.Parameter) -> str:
        expr = f"expr: {param.expr}\n" if param.expr else ""
        value_line = f"value: {param.value}"
        if param.stderr is not None and np.isfinite(param.stderr):
            uncertainty = _ParameterTableModel._format_uncertainty(
                param.value, param.stderr
            )
            value_line = f"value: {uncertainty}"
        return (
            f"name: {param.name}\n"
            f"{expr}{value_line}\n"
            f"min: {param.min}\n"
            f"max: {param.max}\n"
            f"vary: {param.vary}"
        )

    @staticmethod
    def _parse_bound(value: typing.Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return default
            if stripped.lower() in {"inf", "+inf", "infinity", "+infinity"}:
                return np.inf
            if stripped.lower() in {"-inf", "-infinity"}:
                return -np.inf
        return float(value)

    @staticmethod
    def _format_float(value: float) -> str:
        return f"{value:.16g}"

    @staticmethod
    def _format_value(value: float) -> str:
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:.16g}"

    @staticmethod
    def _format_scientific(value: float) -> str:
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:.3e}"

    @staticmethod
    def _format_uncertainty(value: float, stderr: float) -> str:
        if stderr == 0:
            return f"{value:.16g}±0"
        import uncertainties

        return f"{uncertainties.ufloat(value, stderr):P}"

    @staticmethod
    def _format_bound(value: float, default: str) -> str:
        if not np.isfinite(value):
            return default
        return f"{value:.16g}"


class _State2D(pydantic.BaseModel):
    current_idx: int
    data_name_full: str
    params_full: list[list[tuple[typing.Any, ...]] | None]
    params_from_coord_full: list[dict[str, str]]
    fill_mode: typing.Literal["previous", "extrapolate", "none"]


class _FitWorker(QtCore.QObject):
    sigFinished = QtCore.Signal(object)
    sigTimedOut = QtCore.Signal()
    sigErrored = QtCore.Signal(str)
    sigCancelled = QtCore.Signal()

    def __init__(
        self,
        fit_data: xr.DataArray,
        coord_name: Hashable,
        model: lmfit.Model,
        params: lmfit.Parameters,
        *,
        max_nfev: int,
        method: str,
        timeout: float,
    ) -> None:
        super().__init__()
        self._fit_data = fit_data
        self._coord_name = coord_name
        self._model = model
        self._params = params.copy()
        self._max_nfev = max_nfev
        self._method = method
        self._timeout = timeout
        self._cancelled = False
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        t0 = time.perf_counter()
        timed_out = False
        cancelled = False
        self._cancel.clear()

        def _callback(*args, **kwargs) -> bool | None:
            nonlocal timed_out, cancelled

            curr_thread = QtCore.QThread.currentThread()
            if self._cancel.is_set() or (
                curr_thread and curr_thread.isInterruptionRequested()
            ):
                cancelled = True
                return True
            if self._timeout > 0 and (time.perf_counter() - t0) >= self._timeout:
                timed_out = True
                return True
            return None

        try:
            result_ds = self._fit_data.xlm.modelfit(
                self._coord_name,
                model=self._model,
                params=self._params,
                max_nfev=self._max_nfev if self._max_nfev > 0 else None,
                method=self._method,
                iter_cb=_callback,
            )
        except Exception:
            if timed_out:
                self.sigTimedOut.emit()
            elif cancelled:
                self.sigCancelled.emit()
            else:
                self.sigErrored.emit(traceback.format_exc())
            return

        if cancelled:
            self.sigCancelled.emit()
        elif timed_out:
            self.sigTimedOut.emit()
        else:
            self.sigFinished.emit(result_ds)


class Fit1DTool(erlab.interactive.utils.ToolWindow):
    """GUI for selecting initial parameters and bounds for 1D curve fitting.

    Parameters
    ----------
    model
        The lmfit model to fit. If `None`, uses :class:`MultiPeakModel
        <erlab.analysis.fit.models.MultiPeakModel>`, and displays options for model
        initialization.
    data
        The 1D data to fit.
    data_name
        Optional display name for the dataset.

    Signals
    -------
    sigFitFinished(lmfit.Parameters)
        Emitted after a successful fit with the latest parameters.
    """

    tool_name = "ftool_1d"

    class StateModel(pydantic.BaseModel):
        data_name: str
        model_name: str
        model_state: tuple[str, str]
        model_load_path: str | None
        domain: tuple[float, float] | None = None
        normalize_mean: bool = False
        show_components: bool
        timeout: float
        max_nfev: int
        method: str
        slider_widths: dict[str, float]
        params: list[tuple[typing.Any, ...]]
        params_from_coord: dict[str, str]
        state2d: _State2D | None = None

    @property
    def tool_data(self) -> xr.DataArray:
        return self._data

    PLOT_RESAMPLE: int = 5
    FIT_COLOR: str = "c"
    BOUNDS_COLOR: str = "#adadad"
    HIGHLIGHT_COLOR: str = "#d62728"

    MODEL_CHOICES: typing.ClassVar[dict[str, type[lmfit.Model]]] = {
        "MultiPeakModel": erlab.analysis.fit.models.MultiPeakModel,
        "FermiEdgeModel": erlab.analysis.fit.models.FermiEdgeModel,
        "StepEdgeModel": erlab.analysis.fit.models.StepEdgeModel,
        "PolynomialModel": erlab.analysis.fit.models.PolynomialModel,
        "TLLModel": erlab.analysis.fit.models.TLLModel,
        "SymmetrizedGapModel": erlab.analysis.fit.models.SymmetrizedGapModel,
        "ExpressionModel": lmfit.models.ExpressionModel,
    }

    sigFitFinished = QtCore.Signal(object)

    def __init__(
        self,
        data: xr.DataArray,
        model: lmfit.Model | None = None,
        params: lmfit.Parameters | Mapping[str, typing.Any] = None,
        *,
        data_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self._reset_fit_state(
            data,
            model,
            params,
            data_name=data_name,
            model_name=model_name,
        )

        self._build_ui()
        self._undo_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence.StandardKey.Undo, self
        )
        self._undo_shortcut.setContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._undo_shortcut.activated.connect(self.undo)
        self._redo_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence.StandardKey.Redo, self
        )
        self._redo_shortcut.setContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._redo_shortcut.activated.connect(self.redo)
        self._update_fit_curve()
        self._write_history: bool = True
        self._write_state()

    def _reset_fit_state(
        self,
        data: xr.DataArray,
        model: lmfit.Model | None,
        params: lmfit.Parameters | Mapping[str, typing.Any] | None,
        *,
        data_name: str | None,
        model_name: str | None,
    ) -> None:
        data = erlab.interactive.utils.parse_data(data)
        if data.ndim != 1:
            raise ValueError("`data` must be a 1D DataArray")

        self._data: xr.DataArray = data
        self._coord_name: Hashable = data.dims[0]
        self._user_model: lmfit.Model | None = model
        if model is None:
            self._model: lmfit.Model = self._make_default_model()
        else:
            self._model = model

        self._model_load_path: str | None = None
        if data_name is None:
            data_name = "data"
        if model_name is None:
            model_name = "model"
        self._data_name: str = data_name
        self._model_name: str = model_name

        if params is None:
            params = self._model.make_params()
        if not isinstance(params, lmfit.Parameters):
            params = self._model.make_params(**params)

        self._params: lmfit.Parameters = typing.cast("lmfit.Parameters", params)
        self._initial_params: lmfit.Parameters = self._params.copy()
        self._params_from_coord: dict[str, str] = {}
        self._current_row: int | None = None
        self._slider_steps: int = 10000
        self._slider_updating: bool = False
        self._slider_dragging: bool = False
        self._slider_widths: dict[str, float] = {}
        self._last_fit_y: np.ndarray | None = None
        self._last_residual: np.ndarray | None = None
        self._last_result_ds: xr.Dataset | None = None
        self._slider_drag_range: tuple[float, float] | None = None
        self._fit_is_current: bool = False
        self._table_widths_initialized: bool = False
        self._fit_thread: QtCore.QThread | None = None
        self._fit_worker: _FitWorker | None = None
        self._fit_start_time: float | None = None
        self._fit_running_multi: bool = False
        self._pending_fit_action: Callable[[], None] | None = None
        self._peak_lines: list[_PeakPositionLine] = []
        self._fit_multi_total: int | None = None
        self._fit_multi_step: int = 0
        self._fit_multi_fit_data: xr.DataArray | None = None
        self._fit_multi_params: lmfit.Parameters | None = None
        self._prev_states: collections.deque[Fit1DTool.StateModel] = collections.deque(
            maxlen=1000
        )
        self._next_states: collections.deque[Fit1DTool.StateModel] = collections.deque(
            maxlen=1000
        )
        self._write_history = False

    def _build_ui(self) -> None:
        self.resize(987, 610)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        right_container = QtWidgets.QWidget(central)
        right_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._right_layout = QtWidgets.QVBoxLayout(right_container)
        self._right_tabs = QtWidgets.QTabWidget(right_container)
        self._right_tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)
        self._setup_tab = QtWidgets.QWidget()
        self._fit_tab = QtWidgets.QWidget()
        self._setup_tab_layout = QtWidgets.QVBoxLayout(self._setup_tab)
        self._fit_tab_layout = QtWidgets.QVBoxLayout(self._fit_tab)
        self._right_tabs.addTab(self._setup_tab, "Setup")
        self._right_tabs.addTab(self._fit_tab, "Fit")
        self._right_layout.addWidget(self._right_tabs, stretch=1)
        layout.addWidget(right_container)

        self.main_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical, central
        )
        layout.addWidget(self.main_splitter, stretch=1)

        plot_container = QtWidgets.QWidget(self.main_splitter)
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.GraphicsLayoutWidget()
        plot_layout.addWidget(self.plot_widget)
        self.residual_plot = self.plot_widget.addPlot(row=0, col=0)
        self.main_plot = self.plot_widget.addPlot(row=1, col=0)
        self.residual_plot.setXLink(self.main_plot)
        self.residual_plot.hideAxis("bottom")
        self.plot_widget.ci.layout.setRowStretchFactor(0, 1)
        self.plot_widget.ci.layout.setRowStretchFactor(1, 3)
        self.plot_widget.ci.layout.setRowSpacing(0, 20)
        self.main_plot.getAxis("bottom").autoSIPrefix = False
        self.residual_plot.getAxis("left").autoSIPrefix = False

        self.main_plot.getAxis("left").setWidth(50)
        self.residual_plot.getAxis("left").setWidth(50)

        for plot_item in (self.main_plot, self.residual_plot):
            plot_item.setDefaultPadding(0)
            font = QtGui.QFont()
            font.setPointSizeF(11.0)
            for ax in ("left", "top", "right", "bottom"):
                plot_item.getAxis(ax).setTickFont(font)
                plot_item.getAxis(ax).setStyle(
                    autoExpandTextSpace=True, autoReduceTextSpace=True
                )

        self.legend: pg.LegendItem = self.main_plot.addLegend(offset=(5, 5))

        self.data_curve = self.main_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(30, 30, 30),
            symbolPen=pg.mkPen(120, 120, 120),
            name="Data",
        )
        self.fit_curve = self.main_plot.plot(
            pen=pg.mkPen(self.FIT_COLOR, width=2), name="Fit"
        )
        self.residual_curve = self.residual_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(30, 30, 30),
            symbolPen=pg.mkPen(120, 120, 120),
        )
        self.residual_zero = self.residual_plot.addLine(
            y=0.0, pen=pg.mkPen(self.FIT_COLOR, width=2)
        )
        self.component_curves: dict[str, pg.PlotCurveItem] = {}
        self.main_plot.showGrid(x=True, y=True, alpha=0.2)
        self.residual_plot.showGrid(x=True, y=True, alpha=0.2)
        self.main_plot.setLabel("bottom", self._coord_name)
        self.main_plot.setLabel("left", self._data.name)
        self.residual_plot.setLabel("left", "Residual")

        xvals = self._x_values()
        x_min = float(np.nanmin(xvals))
        x_max = float(np.nanmax(xvals))
        self.domain_min_line = _SnapCursorLine(
            pos=x_min,
            angle=90,
            movable=True,
            bounds=(x_min, x_max),
            pen=pg.mkPen(self.BOUNDS_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.domain_max_line = _SnapCursorLine(
            pos=x_max,
            angle=90,
            movable=True,
            bounds=(x_min, x_max),
            pen=pg.mkPen(self.BOUNDS_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.main_plot.addItem(self.domain_min_line)
        self.main_plot.addItem(self.domain_max_line)
        self.domain_min_line.sigDragged.connect(self._domain_min_line_dragged)
        self.domain_max_line.sigDragged.connect(self._domain_max_line_dragged)

        components_container = QtWidgets.QWidget(self.main_splitter)
        components_layout = QtWidgets.QHBoxLayout(components_container)
        components_layout.setContentsMargins(0, 0, 0, 0)
        self.components_check = QtWidgets.QCheckBox("Plot components")
        self.components_check.setChecked(False)
        self.components_check.toggled.connect(self._update_fit_curve)
        components_layout.addWidget(self.components_check)
        components_layout.addStretch(1)
        components_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        components_container.setMaximumHeight(components_container.sizeHint().height())

        table_container = QtWidgets.QWidget(self.main_splitter)
        self._table_layout = QtWidgets.QHBoxLayout(table_container)
        self._table_layout.setContentsMargins(0, 0, 0, 0)
        self.table_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal, table_container
        )
        self.table_splitter.setChildrenCollapsible(False)

        self.param_model: _ParameterTableModel = _ParameterTableModel(
            self._params, self._params_from_coord, self
        )
        self.param_model.sigParamsChanged.connect(self._update_fit_curve)
        self.param_model.sigParamsChanged.connect(self._refresh_slider_from_model)
        self.param_model.sigParamsChanged.connect(self._mark_fit_stale)
        self.param_model.sigParamsChanged.connect(self._write_state)
        self.sigFitFinished.connect(self._replace_last_state)

        self.param_view = QtWidgets.QTableView()
        self.param_view.setModel(self.param_model)
        self.param_view.setCornerButtonEnabled(False)
        vert_header = typing.cast(
            "QtWidgets.QHeaderView", self.param_view.verticalHeader()
        )
        horiz_header = typing.cast(
            "QtWidgets.QHeaderView", self.param_view.horizontalHeader()
        )
        vert_header.setVisible(False)
        self.param_view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.param_view.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.param_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        horiz_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self._schedule_table_width_init()
        value_delegate = _ParameterEditDelegate(self.param_model, self.param_view)
        for col in (1, 3, 4):
            self.param_view.setItemDelegateForColumn(col, value_delegate)
        self.param_view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.param_view.resizeColumnsToContents()
        self.param_view.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.param_view.customContextMenuRequested.connect(self._show_param_menu)
        self.table_splitter.addWidget(self.param_view)
        self._table_layout.addWidget(self.table_splitter, stretch=1)

        self.guess_button = QtWidgets.QPushButton("Guess")
        self.guess_button.clicked.connect(self._guess_params)
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.fit_button.clicked.connect(self._run_fit)
        self.fit_multi_button = QtWidgets.QPushButton("Fit ×20")
        self.fit_multi_button.clicked.connect(lambda: self._run_fit_multiple(20))
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_params)
        self.copy_button = QtWidgets.QPushButton("Copy code")
        self.copy_button.clicked.connect(self.copy_code)
        self.copy_button.setEnabled(False)
        self.save_button = QtWidgets.QPushButton("Save fit")
        self.save_button.clicked.connect(self._save_fit)
        self.save_button.setEnabled(False)
        self.cancel_fit_button = QtWidgets.QPushButton("Cancel")
        self.cancel_fit_button.setToolTip("Abort the running fit sequence.")
        self.cancel_fit_button.setEnabled(False)
        self.cancel_fit_button.clicked.connect(self._cancel_fit)
        self._build_model_group()
        for model_options in self._model_option_registry().values():
            group = model_options["controls"]()
            model_options["visible"](False)
            self._setup_tab_layout.addWidget(group)

        x_vals = self._x_values()
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        x_decimals = erlab.utils.array.effective_decimals(x_vals)
        self.preprocess_group = QtWidgets.QGroupBox("Preprocess")
        preprocess_layout = QtWidgets.QVBoxLayout(self.preprocess_group)
        domain_layout = QtWidgets.QHBoxLayout()
        self.domain_minmax_label = QtWidgets.QLabel("X range")
        self.domain_min_spin = QtWidgets.QDoubleSpinBox()
        self.domain_max_spin = QtWidgets.QDoubleSpinBox()
        self.domain_min_spin.setKeyboardTracking(False)
        self.domain_max_spin.setKeyboardTracking(False)

        for spin in (self.domain_min_spin, self.domain_max_spin):
            spin.setRange(x_min, x_max)
            spin.setDecimals(x_decimals)
            spin.setSingleStep(10**-x_decimals if x_decimals > 0 else 1.0)
        self.domain_min_spin.setValue(x_min)
        self.domain_max_spin.setValue(x_max)
        self.domain_min_spin.valueChanged.connect(self._domain_changed)
        self.domain_max_spin.valueChanged.connect(self._domain_changed)
        domain_layout.addWidget(self.domain_minmax_label)
        domain_layout.addStretch(1)
        domain_layout.addWidget(self.domain_min_spin)
        domain_layout.addWidget(self.domain_max_spin)
        preprocess_layout.addLayout(domain_layout)

        self.normalize_check = QtWidgets.QCheckBox("Normalize by mean")
        self.normalize_check.setChecked(False)
        self.normalize_check.setToolTip(
            "Normalize the data by its mean value before fitting."
        )
        self.normalize_check.toggled.connect(self._mark_fit_stale)
        self.normalize_check.toggled.connect(self._write_state)
        self.normalize_check.toggled.connect(self._populate_data_curve)
        self.normalize_check.toggled.connect(self._reset_slider_widths)
        preprocess_layout.addWidget(self.normalize_check)
        self._setup_tab_layout.addWidget(self.preprocess_group)

        self.current_param_group = QtWidgets.QGroupBox("Parameter")
        param_layout = QtWidgets.QGridLayout(self.current_param_group)

        self.expr_label = QtWidgets.QLabel("")
        self.expr_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.expr_label.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        self.expr_label.setWordWrap(True)

        self.slider_width_label = QtWidgets.QLabel("Range")
        self.slider_width_spin = pg.SpinBox(dec=True, compactHeight=False, finite=False)
        self.slider_width_spin.setOpts(decimals=6, step=0.1)

        self.param_value_label = QtWidgets.QLabel("Value")
        self.param_value_spin = pg.SpinBox(dec=True, compactHeight=False, finite=False)
        self.param_value_spin.setOpts(decimals=6, step=0.1)

        self.param_mode_label = QtWidgets.QLabel("Mode")
        self.param_mode_combo = QtWidgets.QComboBox()
        self.param_mode_combo.addItem("Manual", userData="__fixed")
        for k, v in self._data.coords.items():
            if v.size == 1:
                self.param_mode_combo.addItem(f"Take from '{k}'", userData=k)

        self.param_value_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.param_value_slider.setRange(0, self._slider_steps)

        self.slider_width_label.setToolTip("The range of the slider.")
        self.slider_width_spin.setToolTip("The range of the slider.")
        self.param_value_label.setToolTip("Parameter value used for the fit.")
        self.param_value_spin.setToolTip("Parameter value used for the fit.")
        self.param_value_slider.setToolTip("Drag to adjust the parameter value.")

        param_layout.addWidget(self.expr_label, 0, 0, 1, 2)
        param_layout.addWidget(self.param_mode_label, 1, 0)
        param_layout.addWidget(self.param_mode_combo, 1, 1)
        param_layout.addWidget(self.param_value_label, 2, 0)
        param_layout.addWidget(self.param_value_spin, 2, 1)
        param_layout.addWidget(self.param_value_slider, 3, 0, 1, 2)
        param_layout.addWidget(self.slider_width_label, 4, 0)
        param_layout.addWidget(self.slider_width_spin, 4, 1)

        fit_group = QtWidgets.QGroupBox("Fit options")
        fit_layout = QtWidgets.QFormLayout(fit_group)

        self.timeout_spin = QtWidgets.QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 1e6)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setValue(1.0)
        self.timeout_spin.setSingleStep(1.0)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setToolTip("Timeout for the fit evaluation.")

        self.nfev_spin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self.nfev_spin.setRange(0, 10_000_000)
        self.nfev_spin.setValue(100)
        self.nfev_spin.setSingleStep(100)
        self.nfev_spin.setToolTip("Maximum number of function evaluations.")

        self.method_combo: QtWidgets.QComboBox = QtWidgets.QComboBox()
        self.method_combo.addItems(
            [
                "leastsq",
                "least_squares",
                "nelder",
                "powell",
                "cobyla",
                "trust-constr",
            ]
        )
        self.method_combo.setCurrentText("least_squares")

        fit_layout.addRow("Timeout", self.timeout_spin)
        fit_layout.addRow("Max nfev", self.nfev_spin)
        fit_layout.addRow("Method", self.method_combo)
        self._fit_tab_layout.addWidget(fit_group)

        self.fit_buttons = QtWidgets.QGridLayout()
        self.fit_buttons.setContentsMargins(0, 0, 0, 0)
        self.fit_buttons.addWidget(self.fit_button, 0, 0)
        self.fit_buttons.addWidget(self.fit_multi_button, 0, 1)
        self.fit_buttons.addWidget(self.cancel_fit_button, 2, 0, 1, 2)

        stats_group = QtWidgets.QGroupBox()
        stats_layout = QtWidgets.QGridLayout(stats_group)
        stats_layout.setContentsMargins(8, 3, 8, 3)

        self.elapsed_label = QtWidgets.QLabel("Elapsed")
        self.elapsed_value = QtWidgets.QLabel("—")
        self.nfev_out_label = QtWidgets.QLabel("nfev")
        self.nfev_out_value = QtWidgets.QLabel("—")
        self.redchi_label = QtWidgets.QLabel("redchi")
        self.redchi_value = QtWidgets.QLabel("—")
        self.rsq_label = QtWidgets.QLabel("R²")
        self.rsq_value = QtWidgets.QLabel("—")
        self.aic_label = QtWidgets.QLabel("AIC")
        self.aic_value = QtWidgets.QLabel("—")
        self.bic_label = QtWidgets.QLabel("BIC")
        self.bic_value = QtWidgets.QLabel("—")

        self.elapsed_label.setToolTip("Wall-clock time for the last fit run.")
        self.elapsed_value.setToolTip("Wall-clock time for the last fit run.")
        self.nfev_out_label.setToolTip("Number of function evaluations used.")
        self.nfev_out_value.setToolTip("Number of function evaluations used.")
        self.redchi_label.setToolTip(
            "Reduced chi-square (chi-square per degree of freedom)."
        )
        self.redchi_value.setToolTip(
            "Reduced chi-square (chi-square per degree of freedom)."
        )
        self.rsq_label.setToolTip("Coefficient of determination (R²).")
        self.rsq_value.setToolTip("Coefficient of determination (R²).")
        self.aic_label.setToolTip("Akaike Information Criterion; lower is better.")
        self.aic_value.setToolTip("Akaike Information Criterion; lower is better.")
        self.bic_label.setToolTip("Bayesian Information Criterion; lower is better.")
        self.bic_value.setToolTip("Bayesian Information Criterion; lower is better.")

        stats_layout.addWidget(self.elapsed_label, 0, 0)
        stats_layout.addWidget(self.elapsed_value, 0, 1)
        stats_layout.addWidget(self.nfev_out_label, 0, 2)
        stats_layout.addWidget(self.nfev_out_value, 0, 3)
        stats_layout.addWidget(self.redchi_label, 1, 0)
        stats_layout.addWidget(self.redchi_value, 1, 1)
        stats_layout.addWidget(self.rsq_label, 1, 2)
        stats_layout.addWidget(self.rsq_value, 1, 3)
        stats_layout.addWidget(self.aic_label, 2, 0)
        stats_layout.addWidget(self.aic_value, 2, 1)
        stats_layout.addWidget(self.bic_label, 2, 2)
        stats_layout.addWidget(self.bic_value, 2, 3)

        self.parameters_group = QtWidgets.QGroupBox("Parameters")
        self.parameters_layout = QtWidgets.QVBoxLayout(self.parameters_group)
        self.parameters_button_layout = QtWidgets.QHBoxLayout()
        self.parameters_button_layout.addWidget(self.guess_button)
        self.parameters_button_layout.addWidget(self.reset_button)
        self.parameters_layout.addLayout(self.parameters_button_layout)
        self._fit_tab_layout.addWidget(self.parameters_group)
        self._fit_tab_layout.addWidget(self.current_param_group)

        self.copy_layout = QtWidgets.QGridLayout()
        self.copy_layout.addWidget(self.copy_button, 0, 0)
        self.copy_layout.addWidget(self.save_button, 0, 1)
        self._setup_tab_layout.addStretch()
        self._fit_tab_layout.addStretch()
        self._right_layout.addStretch(1)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self._right_layout.addWidget(separator)
        self._right_layout.addLayout(self.fit_buttons)
        self._right_layout.addWidget(stats_group)
        self._right_layout.addLayout(self.copy_layout)

        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 3)

        selection_model = typing.cast(
            "QtCore.QItemSelectionModel", self.param_view.selectionModel()
        )
        selection_model.currentChanged.connect(self._on_param_selected)

        self.slider_width_spin.valueChanged.connect(self._on_slider_width_changed)
        self.param_value_spin.valueChanged.connect(self._on_slider_value_changed)
        self.param_mode_combo.currentIndexChanged.connect(self._on_param_mode_changed)
        self.param_value_slider.valueChanged.connect(self._on_slider_moved)
        self.param_value_slider.sliderPressed.connect(self._on_slider_pressed)
        self.param_value_slider.sliderReleased.connect(self._on_slider_released)

        self._populate_data_curve()
        if self.param_model.rowCount() > 0:
            self.param_view.selectRow(0)
        self._update_history_actions()

        self._sync_model_display()
        self._sync_model_specific_controls()

    def _infer_model_choice(self, model: lmfit.Model) -> str | None:
        for name, model_cls in self.MODEL_CHOICES.items():
            if isinstance(model, model_cls):
                return name
        return None

    def _make_default_model(self) -> lmfit.Model:
        return erlab.analysis.fit.models.MultiPeakModel(
            npeaks=1,
            peak_shapes="lorentzian",
            fd=False,
            background="none",
            convolve=True,
            segmented=self._auto_segmented(convolve=True),
        )

    def _model_option_registry(self) -> dict[str, dict[str, Callable]]:
        return {
            "MultiPeakModel": {
                "kwargs": self._make_multipeak_model_kwargs,
                "controls": self._build_multipeak_group,
                "sync": self._sync_multipeak_controls,
                "visible": self._set_multipeak_group_visible,
            },
            "PolynomialModel": {
                "kwargs": self._make_polynomial_model_kwargs,
                "controls": self._build_polynomial_group,
                "sync": self._sync_polynomial_controls,
                "visible": self._set_polynomial_group_visible,
            },
            "ExpressionModel": {
                "kwargs": self._make_expression_model_kwargs,
                "controls": self._build_expression_group,
                "sync": self._sync_expression_controls,
                "visible": self._set_expression_group_visible,
            },
        }

    def _make_model_from_choice(self, model_choice: str) -> lmfit.Model:
        registry = self._model_option_registry()
        kwargs: dict[str, typing.Any] = {}
        if model_choice in registry:
            kwargs = registry[model_choice]["kwargs"]()
        model_cls = self.MODEL_CHOICES[model_choice]
        return model_cls(**kwargs)

    def _build_model_group(self) -> None:
        self.model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(self.model_group)

        self.model_combo = QtWidgets.QComboBox()
        for name in self.MODEL_CHOICES:
            self.model_combo.addItem(name, userData=name)
        model_choice = self._infer_model_choice(self._model)
        if not model_choice and self._model_load_path is None:
            self.model_combo.addItem("User", userData="__user")
        self.model_combo.addItem("From file", userData="__file")
        self.model_combo.currentIndexChanged.connect(self._on_model_choice_changed)

        self.model_repr_label = QtWidgets.QLabel("")
        self.model_repr_label.setWordWrap(True)
        self.model_repr_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        model_layout.addWidget(self.model_repr_label)
        model_layout.addWidget(self.model_combo)
        self._setup_tab_layout.addWidget(self.model_group)

    def _sync_model_display(self) -> None:
        self.model_repr_label.setText(f"Current: {self._model!r}")
        with QtCore.QSignalBlocker(self.model_combo):
            label = self._infer_model_choice(self._model)
            if label is None:
                if self._model_load_path is not None:
                    idx = self.model_combo.findData("__file")
                else:
                    idx = self.model_combo.findData("__user")
                self.model_combo.setCurrentIndex(idx)
            else:
                self.model_combo.setCurrentIndex(self.model_combo.findData(label))

    def _sync_model_specific_controls(self) -> None:
        registry = self._model_option_registry()
        model_choice = self._infer_model_choice(self._model)

        for name, handlers in registry.items():
            is_active = name == model_choice
            handlers["visible"](is_active)
            if is_active:
                handlers["sync"]()

    def _make_multipeak_model_kwargs(self) -> dict[str, typing.Any]:
        convolve = self.convolve_check.isChecked()
        oversample = self.oversample_spin.value()
        return dict(
            npeaks=self.npeaks_spin.value(),
            peak_shapes=typing.cast(
                "typing.Literal['lorentzian', 'gaussian']",
                self.peak_shape_combo.currentText(),
            ),
            fd=self.fd_check.isChecked(),
            background=typing.cast(
                "typing.Literal['none', 'constant', 'linear', 'polynomial']",
                self.background_combo.currentText(),
            ),
            degree=self.degree_spin.value(),
            convolve=convolve,
            segmented=self._auto_segmented(convolve),
            **({"oversample": oversample} if convolve else {}),
        )

    @QtCore.Slot()
    def _refresh_multipeak_model(self) -> None:
        self.set_model(
            self._make_model_from_choice("MultiPeakModel"), merge_params=True
        )

    def _build_multipeak_group(self) -> QtWidgets.QWidget:
        self.multipeak_group = QtWidgets.QGroupBox("MultiPeakModel options")
        multipeak_layout = QtWidgets.QGridLayout(self.multipeak_group)

        self.npeaks_label = QtWidgets.QLabel("# Peaks")
        self.npeaks_spin = QtWidgets.QSpinBox()
        self.npeaks_spin.setRange(1, 20)
        self.npeaks_spin.setValue(1)

        self.peak_shape_label = QtWidgets.QLabel("Peak shape")
        self.peak_shape_combo = QtWidgets.QComboBox()
        self.peak_shape_combo.addItems(["lorentzian", "gaussian", "voigt"])
        self.peak_shape_combo.setCurrentText("lorentzian")

        self.fd_check = QtWidgets.QCheckBox("Fermi-Dirac")
        self.fd_check.setChecked(False)
        self.fd_check.setToolTip("Multiply with a Fermi-Dirac distribution.")

        self.background_label = QtWidgets.QLabel("Background")
        self.background_combo = QtWidgets.QComboBox()
        self.background_combo.addItems(["none", "constant", "linear", "polynomial"])
        self.background_combo.setCurrentText("none")
        self.background_combo.setToolTip("Type of background to use.")

        self.degree_label = QtWidgets.QLabel("Degree")
        self.degree_spin = QtWidgets.QSpinBox()
        self.degree_spin.setRange(0, 10)
        self.degree_spin.setValue(2)
        self.degree_spin.setToolTip("Degree of the polynomial background.")

        self.convolve_check = QtWidgets.QCheckBox("Convolve")
        self.convolve_check.setChecked(True)

        self.oversample_label = QtWidgets.QLabel("Oversample")
        self.oversample_spin = QtWidgets.QSpinBox()
        self.oversample_spin.setRange(1, 64)
        self.oversample_spin.setToolTip("Factor to oversample x during convolution.")
        self.oversample_spin.setValue(3)
        self._set_oversample_enabled(self.convolve_check.isChecked())

        multipeak_layout.addWidget(self.npeaks_label, 0, 0)
        multipeak_layout.addWidget(self.npeaks_spin, 0, 1)
        multipeak_layout.addWidget(self.peak_shape_label, 1, 0)
        multipeak_layout.addWidget(self.peak_shape_combo, 1, 1)
        multipeak_layout.addWidget(self.background_label, 2, 0)
        multipeak_layout.addWidget(self.background_combo, 2, 1)
        multipeak_layout.addWidget(self.degree_label, 3, 0)
        multipeak_layout.addWidget(self.degree_spin, 3, 1)
        multipeak_layout.addWidget(self.fd_check, 4, 0)
        multipeak_layout.addWidget(self.convolve_check, 4, 1)
        multipeak_layout.addWidget(self.oversample_label, 5, 0)
        multipeak_layout.addWidget(self.oversample_spin, 5, 1)

        self.npeaks_spin.valueChanged.connect(self._refresh_multipeak_model)
        self.peak_shape_combo.currentTextChanged.connect(self._refresh_multipeak_model)
        self.fd_check.toggled.connect(self._refresh_multipeak_model)
        self.background_combo.currentTextChanged.connect(self._on_bkg_changed)
        self.degree_spin.valueChanged.connect(self._refresh_multipeak_model)
        self.convolve_check.toggled.connect(self._refresh_multipeak_model)
        self.convolve_check.toggled.connect(self._set_oversample_enabled)
        self.oversample_spin.valueChanged.connect(self._refresh_multipeak_model)

        self._set_multipeak_group_visible(False)
        return self.multipeak_group

    def _on_bkg_changed(self, value: str) -> None:
        self._set_degree_visibility(value == "polynomial")
        self._refresh_multipeak_model()

    def _set_degree_visibility(self, visible: bool) -> None:
        self.degree_label.setVisible(visible)
        self.degree_spin.setVisible(visible)

    def _set_oversample_enabled(self, enabled: bool) -> None:
        self.oversample_label.setEnabled(enabled)
        self.oversample_spin.setEnabled(enabled)

    def _sync_multipeak_controls(self) -> None:
        if not isinstance(self._model, erlab.analysis.fit.models.MultiPeakModel):
            return
        with (
            QtCore.QSignalBlocker(self.npeaks_spin),
            QtCore.QSignalBlocker(self.peak_shape_combo),
            QtCore.QSignalBlocker(self.fd_check),
            QtCore.QSignalBlocker(self.background_combo),
            QtCore.QSignalBlocker(self.degree_spin),
            QtCore.QSignalBlocker(self.convolve_check),
            QtCore.QSignalBlocker(self.oversample_spin),
        ):
            self.npeaks_spin.setValue(self._model.func.npeaks)
            self.peak_shape_combo.setCurrentText(self._model.func._peak_shapes[0])
            self.fd_check.setChecked(self._model.func.fd)
            self.background_combo.setCurrentText(self._model.func.background)
            degree = (
                self._model.func.bkg_degree
                if self._model.func.background == "polynomial"
                else 2
            )
            self.degree_spin.setValue(degree)
            self.convolve_check.setChecked(self._model.func.convolve)
            self.oversample_spin.setValue(getattr(self._model.func, "oversample", 1))
        self._set_degree_visibility(self.background_combo.currentText() == "polynomial")
        self._set_oversample_enabled(self.convolve_check.isChecked())

    def _set_multipeak_group_visible(self, visible: bool) -> None:
        self.multipeak_group.setVisible(visible)
        self.multipeak_group.setEnabled(visible)

    @QtCore.Slot()
    def _refresh_polynomial_model(self) -> None:
        degree = int(self.poly_degree_spin.value())
        if degree == self._model.func.degree:
            return
        self.set_model(
            self._make_model_from_choice("PolynomialModel"), merge_params=True
        )

    def _make_polynomial_model_kwargs(self) -> dict[str, typing.Any]:
        return {"degree": int(self.poly_degree_spin.value())}

    def _build_polynomial_group(self) -> QtWidgets.QWidget:
        self.polynomial_group = QtWidgets.QGroupBox("PolynomialModel options")
        poly_layout = QtWidgets.QGridLayout(self.polynomial_group)

        self.poly_degree_label = QtWidgets.QLabel("Degree")
        self.poly_degree_spin = QtWidgets.QSpinBox()
        self.poly_degree_spin.setRange(0, 20)
        self.poly_degree_spin.setValue(9)

        poly_layout.addWidget(self.poly_degree_label, 0, 0)
        poly_layout.addWidget(self.poly_degree_spin, 0, 1)

        self.poly_degree_spin.valueChanged.connect(self._refresh_polynomial_model)
        return self.polynomial_group

    def _sync_polynomial_controls(self) -> None:
        if not isinstance(self._model, erlab.analysis.fit.models.PolynomialModel):
            return
        with QtCore.QSignalBlocker(self.poly_degree_spin):
            self.poly_degree_spin.setValue(self._model.func.degree)

    def _set_polynomial_group_visible(self, visible: bool) -> None:
        self.polynomial_group.setVisible(visible)
        self.polynomial_group.setEnabled(visible)

    @QtCore.Slot()
    def _refresh_expression_model(self) -> None:
        try:
            model = self._make_model_from_choice("ExpressionModel")
        except Exception:
            self._show_error(
                "Expression error",
                "While creating the ExpressionModel from the given expression, "
                "an error occurred. Please check the expression syntax.",
            )
        else:
            self.set_model(model, merge_params=True)

    def _make_expression_model_kwargs(self) -> dict[str, typing.Any]:
        kwargs = {
            "expr": self.expr_edit.toPlainText().strip().replace("\n", " "),
            "independent_vars": [self.indep_var_edit.text().strip()],
        }
        init_script: str | None = self.expr_init_script_dialog.get_script()
        if init_script:
            # Check if the init script executes without errors
            dummy_model = lmfit.models.ExpressionModel("x", independent_vars=["x"])
            dummy_model.asteval.eval(init_script, raise_errors=True)
            kwargs["init_script"] = init_script

        return kwargs

    def _build_expression_group(self) -> QtWidgets.QWidget:
        self.expression_group = QtWidgets.QGroupBox("ExpressionModel options")
        expr_layout = QtWidgets.QVBoxLayout(self.expression_group)

        indep_var_layout = QtWidgets.QHBoxLayout()
        self.indep_var_edit = erlab.interactive.utils.ResizingLineEdit()
        self.indep_var_edit.setText("x")
        indep_var_layout.setSpacing(0)
        indep_var_layout.addWidget(QtWidgets.QLabel("f("))
        indep_var_layout.addWidget(self.indep_var_edit)
        indep_var_layout.addWidget(QtWidgets.QLabel(")"))
        indep_var_layout.addStretch(1)

        expr_edit_layout = QtWidgets.QHBoxLayout()
        self.expr_edit = _PythonCodeEditor()
        self.expr_edit.setMinimumWidth(200)
        self.expr_edit.setAcceptRichText(False)
        self.expr_edit.setPlainText("a * x + b")
        expr_edit_layout.addWidget(QtWidgets.QLabel("="))
        expr_edit_layout.addWidget(self.expr_edit)

        button_layout = QtWidgets.QHBoxLayout()
        self.expr_apply_button = QtWidgets.QPushButton("Apply")
        self.expr_init_script_button = QtWidgets.QPushButton("Edit init script...")
        button_layout.addWidget(self.expr_init_script_button)
        button_layout.addWidget(self.expr_apply_button)

        self.expr_init_script_dialog = _ExpressionInitScriptDialog(self)

        expr_layout.addLayout(indep_var_layout)
        expr_layout.addLayout(expr_edit_layout)
        expr_layout.addLayout(button_layout)

        self.expr_apply_button.clicked.connect(self._refresh_expression_model)
        self.expr_init_script_button.clicked.connect(self._show_init_script_editor)
        return self.expression_group

    @QtCore.Slot()
    def _show_init_script_editor(self) -> None:
        if not isinstance(self._model, lmfit.models.ExpressionModel):
            return
        self.expr_init_script_dialog.show()
        self.expr_init_script_dialog.raise_()
        self.expr_init_script_dialog.activateWindow()

    def _sync_expression_controls(self) -> None:
        if not isinstance(self._model, lmfit.models.ExpressionModel):
            return
        self.indep_var_edit.setText(self._model.independent_vars[0])
        self.expr_edit.setText(self._model.expr)

    def _set_expression_group_visible(self, visible: bool) -> None:
        self.expression_group.setVisible(visible)
        self.expression_group.setEnabled(visible)
        if not visible:
            self.expr_init_script_dialog.hide()

    def set_model(
        self,
        model: lmfit.Model,
        *,
        model_load_path: str | None = None,
        merge_params: bool = False,
        reset_params_from_coord: bool = False,
    ) -> None:
        prev_params = self._params
        prev_widths = dict(self._slider_widths)
        self._model = model
        self._model_load_path = model_load_path

        if reset_params_from_coord:
            self._params_from_coord = {}
        self._params = self._model.make_params()
        if merge_params and prev_params is not None:
            self._merge_params(prev_params, self._params)
        for k in list(self._params_from_coord.keys()):
            if k not in self._params:
                del self._params_from_coord[k]
        self._initial_params = self._params.copy()
        self._slider_widths = {
            name: width for name, width in prev_widths.items() if name in self._params
        }
        self.param_model.set_params(self._params, self._params_from_coord)
        self._sync_model_display()
        self._sync_model_specific_controls()
        self._update_fit_curve()
        self._mark_fit_stale()
        if self.param_model.rowCount() > 0:
            self.param_view.selectRow(0)

    def _reset_history_stack(self) -> None:
        self._prev_states.clear()
        self._next_states.clear()
        self._prev_states.append(self.tool_status)
        self._update_history_actions()

    @QtCore.Slot(int)
    def _on_model_choice_changed(self, _index: int) -> None:
        label = self.model_combo.currentData(role=QtCore.Qt.ItemDataRole.UserRole)
        set_model_kw: dict[str, typing.Any] = {
            "merge_params": True,
            "reset_params_from_coord": True,
        }
        try:
            match label:
                case "__file":
                    path, _ = QtWidgets.QFileDialog.getOpenFileName(
                        self,
                        "Load lmfit model",
                        "",
                        "lmfit Model Files (*.sav *.json *.model);;All files (*)",
                    )
                    if not path:
                        self._sync_model_display()
                        return
                    model = lmfit.model.load_model(path)
                    set_model_kw["model_load_path"] = path
                case "__user":
                    model = self._user_model
                case _:
                    model = self._make_model_from_choice(label)
        except Exception:
            self._show_error(
                "Model creation failed",
                f"Failed to create model '{label}'. Reverting to previous model.",
            )
            self._sync_model_display()
            return
        self.set_model(model, **set_model_kw)

    @property
    def undoable(self) -> bool:
        return len(self._prev_states) > 1

    @property
    def redoable(self) -> bool:
        return len(self._next_states) > 0

    @contextlib.contextmanager
    def _history_suppressed(self):
        original = bool(self._write_history)
        self._write_history = False
        try:
            yield
        finally:
            self._write_history = original

    @QtCore.Slot()
    def _write_state(self) -> None:
        if not self._write_history:
            return
        curr_state = self.tool_status
        last_state = self._prev_states[-1] if self._prev_states else None
        if last_state is None or last_state.model_dump() != curr_state.model_dump():
            self._prev_states.append(curr_state)
            self._next_states.clear()
            self._update_history_actions()

    @QtCore.Slot()
    def _replace_last_state(self) -> None:
        if not self._write_history:
            return
        curr_state = self.tool_status
        if self._prev_states:
            self._prev_states[-1] = curr_state
            self._update_history_actions()

    def _update_history_actions(self) -> None:
        if hasattr(self, "_undo_shortcut"):
            self._undo_shortcut.setEnabled(self.undoable)
        if hasattr(self, "_redo_shortcut"):
            self._redo_shortcut.setEnabled(self.redoable)

    @QtCore.Slot()
    def undo(self) -> None:
        if not self.undoable:
            return
        with self._history_suppressed():
            self._next_states.append(self._prev_states.pop())
            self.tool_status = self._prev_states[-1]
        self._update_history_actions()

    @QtCore.Slot()
    def redo(self) -> None:
        if not self.redoable:
            return
        with self._history_suppressed():
            next_state = self._next_states.pop()
            self._prev_states.append(next_state)
            self.tool_status = next_state
        self._update_history_actions()

    @property
    def tool_status(self) -> StateModel:
        model_cls = self._model.__class__
        return self.StateModel(
            data_name=self._data_name,
            model_name=self._model_name,
            model_state=(
                f"{model_cls.__module__}:{model_cls.__qualname__}",
                self._model.dumps(),
            ),
            model_load_path=self._model_load_path,
            domain=self._fit_domain(),
            normalize_mean=self.normalize_check.isChecked(),
            show_components=self.components_check.isChecked(),
            timeout=self.timeout_spin.value(),
            max_nfev=self.nfev_spin.value(),
            method=self.method_combo.currentText(),
            slider_widths=dict(self._slider_widths),
            params=self._serialize_params(self._params),
            params_from_coord=self._params_from_coord.copy(),
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        with self._history_suppressed():
            self._data_name = status.data_name
            self._model_name = status.model_name
            try:
                model = lmfit.model.Model(lambda x: x)
                cls_info, model_state = status.model_state
                model = model.loads(model_state)
                mod_name, qual = cls_info.split(":")
            except Exception:  # pragma: no cover
                self._show_error("Model restore failed", "Failed to restore model.")
                model = self._model
            else:
                with contextlib.suppress(Exception):
                    mod = importlib.import_module(mod_name)
                    cls_obj = mod
                    for attr in qual.split("."):
                        cls_obj = getattr(cls_obj, attr)
                    model.__class__ = cls_obj

            self.set_model(model, model_load_path=status.model_load_path)

            self.components_check.setChecked(status.show_components)
            self.timeout_spin.setValue(status.timeout)
            self.nfev_spin.setValue(status.max_nfev)
            self.method_combo.setCurrentText(status.method)
            with QtCore.QSignalBlocker(self.normalize_check):
                self.normalize_check.setChecked(status.normalize_mean)

            self._slider_widths = dict(status.slider_widths)
            self._params_from_coord = dict(status.params_from_coord)

            if status.params:
                self._params = self._deserialize_params(status.params)
                self._initial_params = self._params.copy()
                self.param_model.set_params(self._params, self._params_from_coord)
            self._refresh_slider_from_model()
            self._mark_fit_stale()

            if status.domain is not None:
                with (
                    QtCore.QSignalBlocker(self.domain_min_spin),
                    QtCore.QSignalBlocker(self.domain_max_spin),
                ):
                    self.domain_min_spin.setValue(status.domain[0])
                    self.domain_max_spin.setValue(status.domain[1])
            self._domain_changed()

    def _schedule_table_width_init(self) -> None:
        if self._table_widths_initialized:
            return
        QtCore.QTimer.singleShot(0, self._init_table_column_widths)

    def _init_table_column_widths(self) -> None:
        if self._table_widths_initialized:
            return
        self.param_view.resizeColumnsToContents()
        viewport = self.param_view.viewport()
        if viewport:  # pragma: no branch
            viewport_width = viewport.width()
            if viewport_width <= 0:
                self._schedule_table_width_init()
                return
        total = 0
        for col in range(self.param_model.columnCount()):
            total += self.param_view.columnWidth(col)
        stderr_col = 2
        stderr_min = max(self.param_view.sizeHintForColumn(stderr_col), 90)
        if self.param_view.columnWidth(stderr_col) < stderr_min:
            total += stderr_min - self.param_view.columnWidth(stderr_col)
            self.param_view.setColumnWidth(stderr_col, stderr_min)
        value_col = 1
        delta = viewport_width - total
        if delta != 0:
            min_width = self.param_view.sizeHintForColumn(value_col)

            target = max(self.param_view.columnWidth(value_col) + delta, min_width)

            style = self.param_view.style()
            if style:  # pragma: no branch
                target = target - style.pixelMetric(
                    QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent
                )
            self.param_view.setColumnWidth(value_col, target)
        self._table_widths_initialized = True

    def _domain_brushes(self, xvals: np.ndarray) -> list[QtGui.QBrush] | None:
        if xvals.size == 0:
            return None
        domain = self._fit_domain()
        if domain is None:
            mask = np.isfinite(xvals)
        else:
            lo, hi = domain
            mask = np.isfinite(xvals) & (xvals >= lo) & (xvals <= hi)
        base = pg.mkBrush(30, 30, 30)
        tint_color = pg.mkColor(self.FIT_COLOR)
        tint_color.setAlpha(50)
        tint = pg.mkBrush(tint_color)
        return [tint if inside else base for inside in mask]

    def _update_domain_brushes(self) -> None:
        xvals = self._x_values()
        brushes = self._domain_brushes(xvals)
        if brushes is None:
            return
        self.data_curve.setData(
            xvals, self._normalized_data_values(), symbolBrush=brushes
        )
        if self._last_residual is not None:
            self.residual_curve.setData(xvals, self._last_residual, symbolBrush=brushes)

    def _populate_data_curve(self) -> None:
        self._update_domain_brushes()

    def _x_values(self) -> np.ndarray:
        if self._coord_name in self._data.coords:
            coords = self._data.coords[self._coord_name]
            return np.asarray(coords.values)
        return np.arange(self._data.size, dtype=float)

    @QtCore.Slot()
    def _domain_changed(self) -> None:
        x0 = float(self.domain_min_spin.value())
        x1 = float(self.domain_max_spin.value())
        if x0 > x1:
            with (
                QtCore.QSignalBlocker(self.domain_min_spin),
                QtCore.QSignalBlocker(self.domain_max_spin),
            ):
                self.domain_min_spin.setValue(x1)
                self.domain_max_spin.setValue(x0)

        x_vals = self._x_values()
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        lo, hi = sorted((x0, x1))
        lo = float(np.clip(lo, x_min, x_max))
        hi = float(np.clip(hi, x_min, x_max))
        self.domain_min_line.setBounds((x_min, hi))
        self.domain_max_line.setBounds((lo, x_max))
        self.domain_min_line.setPos(lo)
        self.domain_max_line.setPos(hi)
        self._update_domain_brushes()
        self._mark_fit_stale()

    @QtCore.Slot(object)
    def _domain_min_line_dragged(self, line: _SnapCursorLine) -> None:
        x_vals = self._x_values()
        pos = line.temp_value
        idx = np.abs(x_vals - pos).argmin()
        snapped_pos = x_vals[idx]
        self.domain_min_spin.setValue(
            round(snapped_pos, self.domain_min_spin.decimals())
        )

    @QtCore.Slot(object)
    def _domain_max_line_dragged(self, line: _SnapCursorLine) -> None:
        x_vals = self._x_values()
        pos = line.temp_value
        idx = np.abs(x_vals - pos).argmin()
        snapped_pos = x_vals[idx]
        self.domain_max_spin.setValue(
            round(snapped_pos, self.domain_max_spin.decimals())
        )

    def _fit_domain(self) -> tuple[float, float] | None:
        x0 = float(self.domain_min_line.value())
        x1 = float(self.domain_max_line.value())
        x_vals = self._x_values()
        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
        x0, x1 = sorted((float(x0), float(x1)))
        if np.isclose(x0, x_min) and np.isclose(x1, x_max):
            return None
        x_decimals = erlab.utils.array.effective_decimals(x_vals)
        return round(x0, x_decimals), round(x1, x_decimals)

    def _fit_data_raw(self) -> xr.DataArray:
        domain = self._fit_domain()
        if domain is None:
            return self._data
        try:
            return self._data.sel({self._coord_name: slice(*domain)})
        except Exception:
            if self._coord_name in self._data.coords:
                coord = np.asarray(self._data.coords[self._coord_name].values)
                lo, hi = domain
                mask = np.isfinite(coord) & (coord >= lo) & (coord <= hi)
                if mask.any():
                    return self._data.isel({self._coord_name: np.where(mask)[0]})
            return self._data

    def _fit_normalization_factor(self) -> float | None:
        if not self.normalize_check.isChecked():
            return None
        mean_value = float(np.nanmean(self._fit_data_raw().values))
        if not np.isfinite(mean_value) or np.isclose(mean_value, 0.0):
            return None
        return mean_value

    def _fit_data(self) -> xr.DataArray:
        data = self._fit_data_raw()
        norm = self._fit_normalization_factor()
        if norm is None:
            return data
        return data / norm

    def _normalized_data_values(self) -> np.ndarray:
        values = self._data.values
        norm = self._fit_normalization_factor()
        if norm is None:
            return values
        return values / norm

    def _guess_params(self) -> None:
        if not hasattr(self._model, "guess"):
            self._show_warning("Guess not supported", "Model does not support guess().")
            return
        try:
            fit_data = self._fit_data()
            if self._coord_name in fit_data.coords:
                x_vals = fit_data.coords[self._coord_name]
            else:
                x_vals = np.arange(fit_data.size, dtype=float)
            params = self._model.guess(fit_data, **{self._independent_var(): x_vals})
        except Exception:  # pragma: no cover - GUI feedback
            self._show_error("Guess failed", "Failed to estimate initial parameters.")
            return
        self._params = typing.cast("lmfit.Parameters", params)
        self._params_from_coord = {}
        self.param_model.set_params(self._params, self._params_from_coord)
        self._mark_fit_stale()

    def _reset_params(self) -> None:
        match QtWidgets.QMessageBox.question(
            self,
            "Reset",
            "All parameters will be reset to their default values. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        ):
            case QtWidgets.QMessageBox.StandardButton.No:
                return
            case _:
                pass
        self._params_from_coord = {}
        self._params = self._initial_params.copy()
        self.param_model.set_params(self._params, self._params_from_coord)
        self._set_fit_stats(None)
        self._mark_fit_stale()

    def _independent_var(self) -> str:
        if self._model.independent_vars:
            return str(self._model.independent_vars[0])
        return str(self._coord_name)

    def _update_fit_curve(self) -> None:
        xvals = self._x_values()
        if self._has_non_finite_params():
            self.fit_curve.setData([], [])
            self.residual_curve.setData([], [])
            self._last_fit_y = None
            self._last_residual = None
            self._update_component_curves(np.array([]))
            self._update_peak_lines(xvals)
            return

        x_min = float(np.nanmin(xvals)) if xvals.size else 0.0
        x_max = float(np.nanmax(xvals)) if xvals.size else 0.0
        fine_count = max(int(xvals.size) * self.PLOT_RESAMPLE, 2)
        x_fine = np.linspace(x_min, x_max, fine_count, dtype=float)
        if "y" in self._model.independent_vars:
            x_fine = xvals
        try:
            indep_var_kwargs = {self._independent_var(): x_fine}
            if "y" in self._model.independent_vars:
                indep_var_kwargs["y"] = self._normalized_data_values()
            y_fine = self._model.eval(params=self._params, **indep_var_kwargs)
        except Exception:  # pragma: no cover - GUI feedback
            self._show_error("Evaluation failed", "Failed to evaluate model.")
            return
        self.fit_curve.setData(x_fine, y_fine)
        residuals = self._residuals_from_result(xvals)
        brushes = self._domain_brushes(xvals)
        if brushes is None:
            self.residual_curve.setData(xvals, residuals)
        else:
            self.residual_curve.setData(xvals, residuals, symbolBrush=brushes)
        self._last_fit_y = y_fine
        self._last_residual = residuals
        self._update_component_curves(x_fine)
        self._update_peak_lines(xvals)

    @staticmethod
    def _serialize_params(
        params: lmfit.Parameters,
    ) -> list[tuple[typing.Any, ...]]:
        return [p.__getstate__() for p in params.values()]

    @typing.overload
    @staticmethod
    def _deserialize_params(
        state: None,
    ) -> None: ...

    @typing.overload
    @staticmethod
    def _deserialize_params(
        state: list[tuple[typing.Any, ...]],
    ) -> lmfit.Parameters: ...

    @staticmethod
    def _deserialize_params(
        state: list[tuple[typing.Any, ...]] | None,
    ) -> lmfit.Parameters | None:
        if state is None:
            return None
        params = lmfit.Parameters()

        param_list = []
        for pstate in state:
            param = lmfit.Parameter(name="")
            param.__setstate__(pstate)
            param_list.append(param)
        params.add_many(*param_list)
        return params

    @staticmethod
    def _merge_params(
        old_params: lmfit.Parameters, new_params: lmfit.Parameters
    ) -> None:
        for name in new_params:
            if name not in old_params:
                continue
            old_param: lmfit.Parameter = old_params[name]
            new_param: lmfit.Parameter = new_params[name]
            if new_param.expr:
                continue
            new_param.set(
                value=old_param.value,
                min=old_param.min,
                max=old_param.max,
                vary=old_param.vary,
            )

    def _auto_segmented(self, convolve: bool) -> bool:
        if not convolve:
            return False
        xvals = self._x_values()
        if xvals.size < 3:
            return False
        diffs = np.diff(xvals)
        return not np.allclose(diffs, diffs[0])

    def _update_component_curves(self, xvals: np.ndarray) -> None:
        if (
            xvals.size == 0
            or not self.components_check.isChecked()
            or not hasattr(self._model, "eval_components")
        ):
            for curve in self.component_curves.values():
                curve.hide()
            self._remove_component_legend(self.component_curves.keys())
            return
        try:
            comps = self._model.eval_components(
                params=self._params, **{self._independent_var(): xvals}
            )
        except Exception:  # pragma: no cover - GUI feedback
            return
        pens = [
            pg.intColor(i, hues=max(len(comps), 1), sat=128) for i in range(len(comps))
        ]
        for idx, (name, values) in enumerate(comps.items()):
            if name not in self.component_curves:
                curve = self.main_plot.plot(pen=pg.mkPen(pens[idx], width=1), name=name)
                self.component_curves[name] = curve
            curve = self.component_curves[name]
            curve.setData(xvals, values)
            curve.show()
            if not self._legend_has_name(name):
                self.legend.addItem(curve, name)
        for name, curve in list(self.component_curves.items()):
            if name not in comps:
                curve.hide()
        self._remove_component_legend(
            name for name in self.component_curves if name not in comps
        )

    def _clear_peak_lines(self) -> None:
        if not hasattr(self, "main_plot"):
            self._peak_lines.clear()
            return
        for line in self._peak_lines:
            self.main_plot.removeItem(line)
        self._peak_lines.clear()

    def _update_peak_lines(self, xvals: np.ndarray) -> None:
        if (
            not isinstance(self._model, erlab.analysis.fit.models.MultiPeakModel)
            or not self.components_check.isChecked()
        ):
            self._clear_peak_lines()
            return
        if not hasattr(self, "main_plot"):
            return

        if xvals.size == 0:
            xvals = self._x_values()
        x_min = float(np.nanmin(xvals)) if xvals.size else 0.0
        x_max = float(np.nanmax(xvals)) if xvals.size else 0.0
        npeaks = self._model.func.npeaks

        while len(self._peak_lines) > npeaks:
            line = self._peak_lines.pop()
            self.main_plot.removeItem(line)

        for i in range(npeaks):
            key = getattr(self._model, "_prefix", "")
            if len(key) < 1:
                key = getattr(self._model, "_name", "")
            comp_name = f"{key}_p{i}"
            curve = self.component_curves.get(comp_name)
            comp_pen = curve.opts.get("pen") if curve is not None else None
            comp_color = (
                comp_pen.color()
                if isinstance(comp_pen, QtGui.QPen)
                else pg.intColor(i, hues=max(len(self.component_curves), 1), sat=128)
            )
            if i >= len(self._peak_lines):
                line = _PeakPositionLine(self, i, angle=90, movable=True)
                self.main_plot.addItem(line)
                self._peak_lines.append(line)
            line = self._peak_lines[i]
            line.setPen(pg.mkPen(comp_color, width=1))
            line.setHoverPen(pg.mkPen(comp_color, width=2))
            line.setBounds((x_min, x_max))
            line.refresh_pos()

    def _residuals_from_result(self, xvals: np.ndarray) -> np.ndarray:
        if (
            self._fit_is_current
            and self._last_result_ds is not None
            and self._params is not None
            and self._params_match_result(self._last_result_ds, self._params)
        ):
            best_fit = getattr(
                self._last_result_ds.modelfit_results.compute().item(), "best_fit", None
            )
            if best_fit is not None and best_fit.size == self._data.size:
                return self._normalized_data_values() - best_fit
        indep_var_kwargs = {self._independent_var(): xvals}
        if "y" in self._model.independent_vars:
            indep_var_kwargs["y"] = self._normalized_data_values()
        yvals = self._model.eval(params=self._params, **indep_var_kwargs)
        return self._normalized_data_values() - yvals

    @staticmethod
    def _params_match_result(ds: xr.Dataset, params: lmfit.Parameters) -> bool:
        result: lmfit.model.ModelResult = ds.modelfit_results.compute().item()
        for name, param in params.items():
            if name not in result.params:
                return False
            if not np.isclose(param.value, result.params[name].value, rtol=0, atol=0):
                return False
        return True

    def _legend_has_name(self, name: str) -> bool:
        return any(item[1].text == name for item in self.legend.items)

    def _remove_component_legend(self, names: typing.Iterable[str]) -> None:
        for name in names:
            if self._legend_has_name(name):
                self.legend.removeItem(name)

    def _set_fit_ds(self, result_ds: xr.Dataset, t0: float) -> lmfit.Parameters:
        self._last_result_ds = result_ds.copy()
        result = self._last_result_ds.modelfit_results.compute().item()
        self._params = result.params.copy()
        self.param_model.set_params(self._params, self._params_from_coord)
        self._update_fit_curve()
        self._refresh_slider_from_model()
        elapsed = time.perf_counter() - t0
        self._set_fit_stats(result, elapsed=elapsed)
        self._mark_fit_fresh()
        self.sigFitFinished.emit(self._params.copy())

        viewport = self.param_view.viewport()
        if viewport:  # pragma: no branch
            viewport.update()
        return self._params

    def _fit_running(self) -> bool:
        return self._fit_thread is not None and self._fit_thread.isRunning()

    def _fit_timed_out(self, start_time: float) -> None:
        elapsed = time.perf_counter() - start_time
        self._show_error(
            "Fit timed out",
            f"Fit timed out in {elapsed:.2f} s.",
        )
        self._set_fit_stats(None)
        self._set_elapsed_status(elapsed, timed_out=True)
        self._mark_fit_stale()

    def _fit_errored(self, detailed_text: str | None = None) -> None:
        self._show_error(
            "Fit failed", "Fit failed to complete.", detailed_text=detailed_text
        )
        self._set_fit_stats(None)
        self._mark_fit_stale()

    def _start_fit_worker(
        self,
        fit_data: xr.DataArray,
        params: lmfit.Parameters,
        *,
        multi: bool,
        step: int = 0,
        total: int = 0,
        on_success: Callable[[xr.Dataset], None],
        on_timeout: Callable[[], None],
        on_error: Callable[[str], None],
    ) -> bool:
        if self._fit_running():
            self._show_warning("Fit running", "A fit is already running.")
            return False

        self._fit_start_time = time.perf_counter()
        self._fit_running_multi = multi
        self._set_fit_running(True, multi=multi, step=step, total=total)

        worker = _FitWorker(
            fit_data,
            self._coord_name,
            self._model,
            params,
            max_nfev=self.nfev_spin.value(),
            method=self.method_combo.currentText(),
            timeout=self.timeout_spin.value(),
        )
        thread = QtCore.QThread(self)
        if hasattr(thread, "setServiceLevel"):
            thread.setServiceLevel(QtCore.QThread.QualityOfService.High)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        worker.sigFinished.connect(thread.quit)
        worker.sigTimedOut.connect(thread.quit)
        worker.sigErrored.connect(thread.quit)
        worker.sigCancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finalize_fit_thread)

        def _queue_success(result_ds: xr.Dataset) -> None:
            self._queue_fit_action(lambda: on_success(result_ds))

        def _queue_error(message: str) -> None:
            self._queue_fit_action(
                lambda: on_error(erlab.interactive.utils._format_traceback(message))
            )

        worker.sigFinished.connect(_queue_success)
        worker.sigTimedOut.connect(lambda: self._queue_fit_action(on_timeout))
        worker.sigErrored.connect(_queue_error)
        worker.sigCancelled.connect(lambda: self._queue_fit_action(self._fit_cancelled))

        self._fit_thread = thread
        self._fit_worker = worker
        thread.start()
        return True

    def _queue_fit_action(self, action: Callable[[], None]) -> None:
        self._pending_fit_action = action
        if self._fit_thread is None:
            self._pending_fit_action = None
            action()

    def _finalize_fit_thread(self) -> None:
        action = self._pending_fit_action
        self._pending_fit_action = None
        self._fit_thread = None
        self._fit_worker = None
        if action is not None:
            action()

    def _fit_cancelled(self) -> None:
        if self._fit_multi_total is not None:
            self._finish_multi_fit()
        else:
            self._set_fit_running(False, multi=self._fit_running_multi)

    def _run_fit(self) -> bool:
        def _on_success(result_ds: xr.Dataset) -> None:
            if self._fit_start_time is None:
                return
            self._set_fit_ds(result_ds, self._fit_start_time)
            self._set_fit_running(False, multi=False)

        def _on_timeout() -> None:
            if self._fit_start_time is None:
                return
            self._fit_timed_out(self._fit_start_time)
            self._set_fit_running(False, multi=False)

        def _on_error(message) -> None:
            self._fit_errored(message)
            self._set_fit_running(False, multi=False)

        return self._start_fit_worker(
            self._fit_data(),
            self._params,
            multi=False,
            on_success=_on_success,
            on_timeout=_on_timeout,
            on_error=_on_error,
        )

    def _run_fit_multiple(self, count: int) -> None:
        if self._fit_running():
            self._show_warning("Fit running", "A fit is already running.")
            return

        self._fit_multi_total = count
        self._fit_multi_step = 0
        self._fit_multi_fit_data = self._fit_data()
        self._fit_multi_params = self._params
        self._start_next_multi_fit()

    def _start_next_multi_fit(self) -> None:
        if (
            self._fit_multi_total is None
            or self._fit_multi_fit_data is None
            or self._fit_multi_params is None
        ):
            return
        if self._fit_multi_step >= self._fit_multi_total:
            self._finish_multi_fit()
            return

        self._fit_multi_step += 1

        def _on_success(result_ds: xr.Dataset) -> None:
            if self._fit_start_time is None:
                return
            self._fit_multi_params = self._set_fit_ds(result_ds, self._fit_start_time)
            if self._fit_multi_step >= (self._fit_multi_total or 0):
                self._finish_multi_fit()
            else:
                self._start_next_multi_fit()

        def _on_timeout() -> None:
            if self._fit_start_time is None:
                return
            self._fit_timed_out(self._fit_start_time)
            self._finish_multi_fit()

        def _on_error(message) -> None:
            self._fit_errored(message)
            self._finish_multi_fit()

        started = self._start_fit_worker(
            self._fit_multi_fit_data,
            self._fit_multi_params,
            multi=True,
            step=self._fit_multi_step,
            total=self._fit_multi_total,
            on_success=_on_success,
            on_timeout=_on_timeout,
            on_error=_on_error,
        )
        if not started:
            self._finish_multi_fit()

    def _finish_multi_fit(self) -> None:
        self._set_fit_running(False, multi=True)
        self._fit_running_multi = False
        self._fit_multi_total = None
        self._fit_multi_step = 0
        self._fit_multi_fit_data = None
        self._fit_multi_params = None

    def _set_fit_running(
        self, running: bool, *, multi: bool, step: int = 0, total: int = 0
    ) -> None:
        if running:
            self.fit_button.setEnabled(False)
            self.fit_multi_button.setEnabled(False)
            self.fit_button.setText("Fitting...")
            if multi and total > 0:
                self.fit_multi_button.setText(f"Fit {step}/{total}")
            else:
                self.fit_multi_button.setText("Fit ×20")
            self.cancel_fit_button.setEnabled(True)
        else:
            self.fit_button.setEnabled(True)
            self.fit_multi_button.setEnabled(True)
            self.fit_button.setText("Fit")
            self.fit_multi_button.setText("Fit ×20")
            self.cancel_fit_button.setEnabled(False)

    def _make_model_code(self, data_name: str) -> tuple[str, str, list[str]]:
        lines: list[str] = []

        data_name = str(data_name)
        if not data_name.isidentifier():
            lines.append(f"target = {data_name}")
            data_name = "target"
        model_name = str(self._model_name)
        if not model_name.isidentifier():
            model_name = "model"

        model_choice = self._infer_model_choice(self._model)
        if model_choice:
            model_cls = self.MODEL_CHOICES.get(model_choice, None)
            if model_cls is not None:
                registry = self._model_option_registry()
                if model_choice in registry:
                    model_kwargs = registry[model_choice]["kwargs"]()
                else:
                    model_kwargs = {}

                lines.append(
                    erlab.interactive.utils.generate_code(
                        model_cls,
                        args=[],
                        kwargs=model_kwargs,
                        module=model_cls.__module__.replace("erlab.analysis", "era"),
                        assign=model_name,
                    )
                )
        elif self._model_load_path:
            lines.append(
                erlab.interactive.utils.generate_code(
                    lmfit.model.load_model,
                    args=[self._model_load_path],
                    kwargs={},
                    module="lmfit.model",
                    assign=model_name,
                )
            )

        return data_name, model_name, lines

    @QtCore.Slot()
    def copy_code(self) -> str:
        data_name, model_name, lines = self._make_model_code(self._data_name)

        fit_domain = self._fit_domain()
        if fit_domain is not None:
            sel_kw = erlab.interactive.utils.format_kwargs(
                {self._coord_name: slice(*fit_domain)}
            )
            lines.append(f"{data_name}_crop = {data_name}.sel({sel_kw})")
            data_name = f"{data_name}_crop"

        if self.normalize_check.isChecked():
            lines.append(f"{data_name}_norm = {data_name} / {data_name}.mean()")
            data_name = f"{data_name}_norm"

        param_entries: list[str] = []
        param_kwargs: dict[str, typing.Any] = {}
        needs_dict = False
        for name, param in self._params.items():
            entry_kwargs: dict[str, typing.Any] = {}
            if param.expr:
                if self._param_expr_from_hint(param) and not self._param_is_func_arg(
                    param
                ):
                    continue
                entry_kwargs["expr"] = param.expr
                can_be_float: bool = False
            else:
                if np.isfinite(param.min):
                    entry_kwargs["min"] = param.min
                if np.isfinite(param.max):
                    entry_kwargs["max"] = param.max
                if not param.vary:
                    entry_kwargs["vary"] = False
                can_be_float = not bool(entry_kwargs)

                entry_kwargs["value"] = param.value

            entry_value: float | dict[str, float] = (
                param.value if can_be_float else entry_kwargs
            )
            param_kwargs[name] = entry_value
            if not name.isidentifier() or needs_dict:
                needs_dict = True
                continue
            param_entries.append(f"{name}={entry_value!r}")

        if needs_dict:
            lines.append(f"params = {model_name}.make_params(**{param_kwargs!r})")
        else:
            joined = ",\n    ".join(param_entries)
            if joined:
                lines.append(f"params = {model_name}.make_params(\n    {joined},\n)")
            else:
                lines.append(f"params = {model_name}.make_params()")

        lines.append(
            erlab.interactive.utils.generate_code(
                self._data.xlm.modelfit,
                args=[self._coord_name],
                kwargs={
                    "model": f"|{model_name}|",
                    "params": "|params|",
                    "method": self.method_combo.currentText(),
                },
                name="modelfit",
                module=f"{data_name}.xlm",
                assign="result",
            )
        )
        return erlab.interactive.utils.copy_to_clipboard(lines)

    @QtCore.Slot()
    def _save_fit(self) -> None:
        if self._last_result_ds is None:
            self._show_warning("No fit result", "There is no fit result to save.")
            return
        erlab.interactive.utils.save_fit_ui(self._last_result_ds, parent=self)

    def _on_param_selected(
        self, current: QtCore.QModelIndex, previous: QtCore.QModelIndex
    ) -> None:
        if not current.isValid():
            self._current_row = None
            self.current_param_group.setTitle("Parameter")
            self._show_slider_message("")
            return
        self._current_row = current.row()
        self._refresh_slider_from_model()

    def _set_slider_enabled(self, enabled: bool) -> None:
        for widget in (
            self.slider_width_spin,
            self.param_value_spin,
            self.param_value_slider,
        ):
            widget.setEnabled(enabled)

    @QtCore.Slot()
    def _reset_slider_widths(self) -> None:
        self._slider_widths = {}
        self._refresh_slider_from_model()

    def _set_slider_widget_visibility(self, visible: bool) -> None:
        self.slider_width_label.setVisible(visible)
        self.slider_width_spin.setVisible(visible)
        self.param_value_label.setVisible(visible)
        self.param_value_spin.setVisible(visible)
        self.param_value_slider.setVisible(visible)
        self.param_mode_label.setVisible(visible)
        self.param_mode_combo.setVisible(visible)
        self.expr_label.setVisible(not visible)

    def _show_slider_message(self, message: str) -> None:
        self.expr_label.setText(message)
        self._set_slider_enabled(False)
        self._set_slider_widget_visibility(False)

    def _get_current_param(self) -> lmfit.Parameter:
        return self.param_model.param_at(typing.cast("int", self._current_row))

    def _show_param_menu(self, pos: QtCore.QPoint) -> None:
        index = self.param_view.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        param = self.param_model.param_at(row)
        menu = QtWidgets.QMenu(self)
        set_action = typing.cast("QtGui.QAction", menu.addAction("Set expression..."))
        clear_action = typing.cast("QtGui.QAction", menu.addAction("Clear expression"))
        can_edit = self._can_edit_expr(param)
        set_action.setEnabled(can_edit)
        clear_action.setEnabled(can_edit and bool(param.expr))
        set_action.triggered.connect(lambda: self._prompt_param_expr(row))
        clear_action.triggered.connect(lambda: self._clear_param_expr(row))
        self.param_view.selectRow(row)
        menu.exec(
            typing.cast("QtWidgets.QWidget", self.param_view.viewport()).mapToGlobal(
                pos
            )
        )

    def _can_edit_expr(self, param: lmfit.Parameter) -> bool:
        if str(param.name) not in self._model.param_names:
            return False
        return not (
            self._param_expr_from_hint(param) and not self._param_is_func_arg(param)
        )

    def _param_is_func_arg(self, param: lmfit.Parameter) -> bool:
        func_args = getattr(self._model, "_func_allargs", None)
        if not func_args:
            return False
        param_name = str(param.name)
        prefix = getattr(self._model, "_prefix", "")
        basename = (
            param_name[len(prefix) :]
            if prefix and param_name.startswith(prefix)
            else param_name
        )
        return basename in func_args and basename not in getattr(
            self._model, "independent_vars", []
        )

    def _param_expr_from_hint(self, param: lmfit.Parameter) -> bool:
        hints = getattr(self._model, "param_hints", {})
        if not hints:
            return False
        param_name = str(param.name)
        prefix = getattr(self._model, "_prefix", "")
        basename = (
            param_name[len(prefix) :]
            if prefix and param_name.startswith(prefix)
            else param_name
        )
        for key in (param_name, basename):
            hint = hints.get(key)
            if isinstance(hint, dict) and hint.get("expr"):
                return True
        return False

    def _prompt_param_expr(self, row: int) -> None:
        param = self.param_model.param_at(row)
        if not self._can_edit_expr(param):
            if self._param_expr_from_hint(param) and not self._param_is_func_arg(param):
                self._show_warning(
                    "Expression Locked",
                    "This expression is defined by the model and cannot be edited.",
                )
            return
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Set Expression",
            f"Expression for {param.name}:",
            text=param.expr or "",
        )
        if not ok:
            return
        expr = text.strip()
        if not expr:
            self._clear_param_expr(row)
            return
        self._set_param_expr(row, expr)

    def _clear_param_expr(self, row: int) -> None:
        param = self.param_model.param_at(row)
        if not self._can_edit_expr(param) or not param.expr:
            return
        param.set(expr="")
        self._notify_param_change(row)

    def _set_param_expr(self, row: int, expr: str) -> None:
        param = self.param_model.param_at(row)
        if not self._can_edit_expr(param):
            return
        is_valid, error = self._validate_param_expr(param, expr)
        if not is_valid:
            details = f" due to the following reason:\n\n{error}" if error else "."
            self._show_warning(
                "Invalid Expression",
                f"Expression could not be parsed or evaluated{details}",
            )
            return
        if str(param.name) in self._params_from_coord:
            del self._params_from_coord[str(param.name)]
        param.set(expr=expr)
        param.vary = False
        self._notify_param_change(row)

    def _validate_param_expr(
        self, param: lmfit.Parameter, expr: str
    ) -> tuple[bool, str]:
        expr_eval = getattr(param, "_expr_eval", None)
        if expr_eval is None:
            expr_eval = getattr(self._params, "_asteval", None)
        if expr_eval is None:
            return True, ""
        expr_eval.error = []
        expr_eval.error_msg = None
        try:
            expr_eval.parse(expr)
            lmfit.parameter.check_ast_errors(expr_eval)
            expr_eval.eval(expr)
            lmfit.parameter.check_ast_errors(expr_eval)
        except Exception as exc:
            return False, str(exc)
        return True, ""

    def _notify_param_change(self, row: int) -> None:
        top_left = self.param_model.index(row, 0)
        bottom_right = self.param_model.index(row, self.param_model.columnCount() - 1)
        self.param_model.dataChanged.emit(
            top_left,
            bottom_right,
            [
                QtCore.Qt.ItemDataRole.DisplayRole,
                QtCore.Qt.ItemDataRole.EditRole,
                QtCore.Qt.ItemDataRole.CheckStateRole,
            ],
        )
        self.param_model.sigParamsChanged.emit()

    def _param_row(self, name: str) -> int | None:
        for row in range(self.param_model.rowCount()):
            if self.param_model.param_name(row) == name:
                return row
        return None

    def _set_param_value_by_name(self, name: str, value: float) -> bool:
        row = self._param_row(name)
        if row is None:
            return False
        index = self.param_model.index(row, 1)
        return self.param_model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)

    def _param_is_editable(self, name: str) -> bool:
        if name not in self._params:
            return False
        param = self._params[name]
        if param.expr:
            return False
        return str(param.name) not in self._params_from_coord

    def _peak_param_name(self, peak_index: int, suffix: str) -> str:
        prefix = getattr(self._model, "prefix", "")
        if self.peak_shape_combo.currentText().casefold() == "voigt":
            if suffix == "width":
                suffix = "gamma"
            elif suffix == "height":
                suffix = "amplitude"
        return f"{prefix}p{peak_index}_{suffix}"

    def _get_peak_param_value(self, peak_index: int, suffix: str) -> float | None:
        name = self._peak_param_name(peak_index, suffix)
        if name not in self._params:
            return None
        value = self._params[name].value
        if value is None:
            return None
        return float(value)

    def _set_peak_param_value(self, peak_index: int, suffix: str, value: float) -> bool:
        name = self._peak_param_name(peak_index, suffix)
        if not self._param_is_editable(name):
            return False
        return self._set_param_value_by_name(name, value)

    def _refresh_slider_from_model(self) -> None:
        if (
            self._current_row is None
            or self._current_row < 0
            or self._current_row >= self.param_model.rowCount()
        ):
            self.current_param_group.setTitle("Parameter")
            self._show_slider_message("")
            return
        param = self._get_current_param()
        if self._slider_dragging and not param.expr:
            self._slider_updating = True
            with (
                QtCore.QSignalBlocker(self.param_value_spin),
                QtCore.QSignalBlocker(self.param_value_slider),
            ):
                self.param_value_spin.setValue(param.value)
                if self._slider_drag_range is None:
                    slider_min, slider_max, _ = self._slider_range(param.value, param)
                    self._slider_drag_range = (slider_min, slider_max)
                else:
                    slider_min, slider_max = self._slider_drag_range
                self._set_slider_position(param.value, slider_min, slider_max)
            self._slider_updating = False
            return

        self.current_param_group.setTitle(f"Parameter: {param.name}")

        is_from_coord: bool = str(param.name) in self._params_from_coord

        if param.expr:
            self._show_slider_message(f"expr: {param.expr}")
            self._set_slider_values(param.value, None, None, None)
            return
        if not np.isfinite(param.value):
            self._show_slider_message(
                f"value: {_ParameterTableModel._format_value(param.value)}"
            )
            return

        self.expr_label.setText("")
        self._set_slider_enabled(not is_from_coord)
        self._set_slider_widget_visibility(True)

        slider_min, slider_max, width = self._slider_range(param.value, param)
        self._set_slider_values(param.value, width, slider_min, slider_max)
        self.param_value_spin.setReadOnly(is_from_coord)

        with QtCore.QSignalBlocker(self.param_mode_combo):
            if is_from_coord:
                k = self._params_from_coord[str(param.name)]
                self.param_mode_combo.setCurrentText(f"Take from '{k}'")
            else:
                self.param_mode_combo.setCurrentText("Manual")

    def _slider_range(
        self, value: float, param: lmfit.Parameter
    ) -> tuple[float, float, float]:
        width = self._slider_widths.get(param.name)
        if width is None or not np.isfinite(width) or width <= 0:
            if np.isfinite(param.min) and np.isfinite(param.max):
                width = float(param.max - param.min)
            else:
                width = float(self._default_slider_width(value))
            self._slider_widths[param.name] = width
        half_span = width / 2.0
        vmin = float(value - half_span)
        vmax = float(value + half_span)
        if np.isfinite(param.min):
            vmin = max(vmin, float(param.min))
        if np.isfinite(param.max):
            vmax = min(vmax, float(param.max))
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return vmin, vmax, width

    def _default_slider_width(self, value: float) -> float:
        y_std = float(np.nanstd(self._fit_data().values))
        x_std = float(np.nanstd(self._x_values()))
        base = max(y_std, x_std, 1.0)
        if np.isfinite(value):
            base = max(base, abs(value) * 0.5, 1.0)
        return base * 0.05

    def _set_slider_values(
        self,
        value: float,
        width: float | None,
        slider_min: float | None,
        slider_max: float | None,
    ) -> None:
        if not np.isfinite(value):
            return
        self._slider_updating = True
        try:
            if width is not None and np.isfinite(width):
                self.slider_width_spin.setValue(width)
            if slider_min is not None and slider_max is not None:
                if slider_min > slider_max:
                    slider_min, slider_max = slider_max, slider_min
                self._set_slider_position(value, slider_min, slider_max)
            self.param_value_spin.setValue(value)
        finally:
            self._slider_updating = False

    def _set_slider_position(self, value: float, vmin: float, vmax: float) -> None:
        if not np.isfinite(vmin) or not np.isfinite(vmax) or not np.isfinite(value):
            self.param_value_slider.setEnabled(False)
            return
        if vmax <= vmin:
            self.param_value_slider.setEnabled(False)
            return
        self.param_value_slider.setEnabled(
            not self.param_model._is_param_from_coord(self._get_current_param())
        )
        clamped = min(max(value, vmin), vmax)
        ratio = (clamped - vmin) / (vmax - vmin)
        self.param_value_slider.setValue(round(ratio * self._slider_steps))

    def _on_slider_moved(self, slider_value: int) -> None:
        if self._slider_updating or self._current_row is None:
            return
        param = self._get_current_param()
        if self._slider_dragging and self._slider_drag_range is not None:
            vmin, vmax = self._slider_drag_range
        else:
            vmin, vmax, _ = self._slider_range(self.param_value_spin.value(), param)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return
        if vmax <= vmin:
            return
        value = vmin + (vmax - vmin) * (slider_value / self._slider_steps)
        self._slider_updating = True
        with QtCore.QSignalBlocker(self.param_value_spin):
            self.param_value_spin.setValue(value)
        self._slider_updating = False
        self._update_param_value(value)

    def _on_slider_value_changed(self, value: float) -> None:
        if self._slider_updating or self._current_row is None:
            return
        if not np.isfinite(value):
            self.param_value_slider.setEnabled(False)
            return
        param = self._get_current_param()
        vmin, vmax, _ = self._slider_range(value, param)
        self._slider_updating = True
        with (
            QtCore.QSignalBlocker(self.param_value_spin),
            QtCore.QSignalBlocker(self.param_value_slider),
        ):
            self._set_slider_position(value, vmin, vmax)
        self._slider_updating = False
        self._update_param_value(value)

    def _on_slider_width_changed(self) -> None:
        if self._slider_updating or self._current_row is None:
            return
        param = self._get_current_param()
        width = self.slider_width_spin.value()
        if not np.isfinite(width) or width <= 0:
            return
        self._slider_widths[param.name] = width
        vmin, vmax, _ = self._slider_range(self.param_value_spin.value(), param)
        with (
            QtCore.QSignalBlocker(self.param_value_spin),
            QtCore.QSignalBlocker(self.param_value_slider),
        ):
            self._set_slider_position(self.param_value_spin.value(), vmin, vmax)
        self._slider_updating = False

    @QtCore.Slot()
    def _on_param_mode_changed(self) -> None:
        if self._current_row is None:
            return
        param = self._get_current_param()
        current_name = str(param.name)
        param_coord = typing.cast(
            "Hashable",
            self.param_mode_combo.currentData(role=QtCore.Qt.ItemDataRole.UserRole),
        )
        if param_coord == "__fixed":
            if current_name in self._params_from_coord:
                del self._params_from_coord[current_name]
            self._update_param_value(float(param.value))
        else:
            self._params_from_coord[current_name] = str(param_coord)
            self._update_param_value(float(self._data[param_coord].values))
            self._update_param_vary(False)

    def _update_param_value(self, value: float) -> None:
        if self._current_row is None:
            return
        index = self.param_model.index(self._current_row, 1)
        self.param_model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)

    def _update_param_vary(self, vary: bool) -> None:
        if self._current_row is None:
            return
        index = self.param_model.index(self._current_row, 5)
        self.param_model.setData(index, vary, QtCore.Qt.ItemDataRole.EditRole)

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        if self._current_row is None:
            self._slider_drag_range = None
            return
        param = self._get_current_param()
        vmin, vmax, _ = self._slider_range(self.param_value_spin.value(), param)
        self._slider_drag_range = (vmin, vmax)

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self._slider_drag_range = None
        self._refresh_slider_from_model()

    def _set_fit_stats(
        self, result: lmfit.model.ModelResult | None, *, elapsed: float | None = None
    ) -> None:
        if result is None:
            self.elapsed_value.setText("—")
            self._set_elapsed_status(None, timed_out=False)
            self.nfev_out_value.setText("—")
            self.redchi_value.setText("—")
            self.rsq_value.setText("—")
            self.aic_value.setText("—")
            self.bic_value.setText("—")
            return
        if elapsed is None:
            elapsed = float("nan")
        self._set_elapsed_status(elapsed, timed_out=False)
        nfev_text = str(result.nfev)
        if self.nfev_spin.value() > 0 and result.nfev >= self.nfev_spin.value():
            nfev_text = (
                f'<span style="color:{self.HIGHLIGHT_COLOR}; font-weight:600;">'
                f"{nfev_text}</span>"
            )
            self.nfev_out_label.setText(
                f'<span style="color:{self.HIGHLIGHT_COLOR}; font-weight:600;">'
                "nfev</span>"
            )
        else:
            self.nfev_out_label.setText("nfev")
        self.nfev_out_value.setText(nfev_text)
        self.redchi_value.setText(f"{result.redchi:.4g}")
        r_squared = result.rsquared if result.rsquared is not None else float("nan")
        self.rsq_value.setText(f"{r_squared:.4g}")
        self.aic_value.setText(f"{result.aic:.4g}")
        self.bic_value.setText(f"{result.bic:.4g}")

    def _set_elapsed_status(self, elapsed: float | None, timed_out: bool) -> None:
        if elapsed is None or not np.isfinite(elapsed):
            self.elapsed_value.setText("—")
        else:
            self.elapsed_value.setText(f"{elapsed:.2f} s")
        if timed_out:
            self.elapsed_label.setText(
                f'<span style="color:{self.HIGHLIGHT_COLOR}; font-weight:600;">'
                "Elapsed</span>"
            )
            self.elapsed_value.setText(
                f'<span style="color:{self.HIGHLIGHT_COLOR}; font-weight:600;">'
                f"{elapsed:.2f} s</span>"
            )
        else:
            self.elapsed_label.setText("Elapsed")

    def _has_non_finite_params(self) -> bool:
        return any(not np.isfinite(param.value) for param in self._params.values())

    def _mark_fit_stale(self) -> None:
        self._fit_is_current = False
        self.save_button.setEnabled(False)
        self.copy_button.setEnabled(False)

    def _mark_fit_fresh(self) -> None:
        self._fit_is_current = True
        self.save_button.setEnabled(True)
        self.copy_button.setEnabled(True)

    def _show_warning(self, title: str, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, title, text)

    def _show_error(
        self, title: str, text: str, detailed_text: str | None = None
    ) -> None:
        erlab.interactive.utils.MessageDialog.critical(
            self, title, text, detailed_text=detailed_text
        )

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:
        self._cancel_fit()
        super().closeEvent(event)

    def _cancel_fit(self) -> None:
        if self._fit_worker is not None:
            self._fit_worker.cancel()
        if self._fit_thread is not None:
            self._fit_thread.requestInterruption()
            self._fit_thread.quit()
            self._fit_thread.wait()
        self._pending_fit_action = None
        self._fit_thread = None
        self._fit_worker = None
        self._fit_running_multi = False
