"""State and input parsing helpers for ImageTool viewer widgets."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import logging
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool._provenance._model import (
    FileDataSelection,
    ToolProvenanceOperation,
)

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import qtawesome

    from erlab.interactive.imagetool.slicer import ArraySlicer, ArraySlicerState
    from erlab.interactive.imagetool.viewer import ImageSlicerArea
else:
    import lazy_loader as _lazy

    qtawesome = _lazy.load("qtawesome")

logger = logging.getLogger(__name__)

_SerializedFileDataSelection: typing.TypeAlias = int | dict[str, typing.Any]


@dataclasses.dataclass(frozen=True)
class _PreparedInputData:
    data: xr.DataArray
    selection: FileDataSelection
    source_ndim: int
    source_dtype: np.dtype[typing.Any]
    operations: tuple[ToolProvenanceOperation, ...] = ()


class ColorMapState(typing.TypedDict):
    """A dictionary containing the colormap state of an `ImageSlicerArea` instance."""

    cmap: str | pg.ColorMap
    gamma: float
    reverse: bool
    high_contrast: bool
    zero_centered: bool
    levels_locked: bool
    levels: typing.NotRequired[tuple[float, float]]


class GuidelineState(typing.TypedDict):
    """A dictionary containing the state of rotation guidelines."""

    count: typing.Literal[1, 2, 3]
    angle: float
    offset: tuple[float, float]
    follow_cursor: bool


class PlotItemState(typing.TypedDict):
    """A dictionary containing the state of a `PlotItem` instance."""

    vb_aspect_locked: bool | float
    vb_x_inverted: typing.NotRequired[bool]
    vb_y_inverted: typing.NotRequired[bool]
    vb_autorange: typing.NotRequired[tuple[bool, bool]]
    roi_states: typing.NotRequired[list[dict[str, typing.Any]]]
    guideline_state: typing.NotRequired[GuidelineState]


class ImageSlicerState(typing.TypedDict):
    """A dictionary containing the state of an `ImageSlicerArea` instance."""

    color: ColorMapState
    slice: ArraySlicerState
    current_cursor: int
    manual_limits: dict[str, list[float]]
    axis_inversions: typing.NotRequired[dict[str, bool]]
    filter_operation: typing.NotRequired[dict[str, typing.Any] | None]
    cursor_colors: list[str]
    controls_visible: typing.NotRequired[bool]
    file_path: typing.NotRequired[str | None]
    load_func: typing.NotRequired[
        tuple[str, dict[str, typing.Any], _SerializedFileDataSelection] | None
    ]
    splitter_sizes: typing.NotRequired[list[list[int]]]
    plotitem_states: typing.NotRequired[list[PlotItemState]]


class _SuppressNanWarning(contextlib.ContextDecorator):
    def __init__(self) -> None:
        self._contexts: list[typing.Any] = []

    def __enter__(self) -> typing.Self:
        context = warnings.catch_warnings()
        context.__enter__()
        warnings.filterwarnings(
            "ignore",
            r"All-NaN (slice|axis) encountered",
            RuntimeWarning,
        )
        self._contexts.append(context)
        return self

    def __exit__(self, *exc_info: object) -> bool | None:
        return self._contexts.pop().__exit__(*exc_info)


suppressnanwarning = _SuppressNanWarning()


def _processed_ndim(darr: xr.DataArray) -> int:
    if darr.ndim == 1:
        nd = 2
    elif darr.ndim > 4:
        nd = len(tuple(s for s in darr.shape if s != 1))
    else:
        nd = darr.ndim
    return nd


def _supported_shape(darr: xr.DataArray) -> bool:
    return _processed_ndim(darr) in (2, 3, 4)


def _reducible_shape(darr: xr.DataArray) -> bool:
    return _processed_ndim(darr) >= 2


def _datatree_dataarray_selection(
    source_path: str, variable_name: Hashable
) -> tuple[str, Hashable]:
    node_path = source_path.rstrip("/") or "/"
    return node_path, variable_name


class _SelectDataArraysDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        data: xr.Dataset | xr.DataTree,
    ) -> None:
        super().__init__(parent)
        self._data_arrays: list[xr.DataArray] = []
        self._selections: list[FileDataSelection] = []
        self._source_paths: list[str] = []
        variable_names: list[str] = []

        if isinstance(data, xr.Dataset):
            for variable_name, darr in data.data_vars.items():
                if _reducible_shape(darr):
                    self._data_arrays.append(darr)
                    self._selections.append(
                        FileDataSelection(
                            kind="dataset_variable",
                            value=variable_name,
                        )
                    )
                    self._source_paths.append("/")
                    variable_names.append(str(variable_name))
        else:
            for leaf in data.leaves:
                source_path = str(leaf.path)
                for variable_name, darr in leaf.dataset.data_vars.items():
                    if _reducible_shape(darr):
                        self._data_arrays.append(darr)
                        self._selections.append(
                            FileDataSelection(
                                kind="datatree_variable",
                                value=_datatree_dataarray_selection(
                                    source_path, variable_name
                                ),
                            )
                        )
                        self._source_paths.append(source_path)
                        variable_names.append(str(variable_name))

        self.setWindowTitle("Select Data Variables")
        self.resize(820, 420)

        layout = QtWidgets.QVBoxLayout(self)

        show_path_tree = any(source_path != "/" for source_path in self._source_paths)
        name_column = 2 if show_path_tree else 1
        ndim_column = 3 if show_path_tree else 2
        shape_column = 4 if show_path_tree else 3
        size_column = 5 if show_path_tree else 4

        self._tree_widget = QtWidgets.QTreeWidget(self)
        if show_path_tree:
            self._tree_widget.setColumnCount(6)
            self._tree_widget.setHeaderLabels(
                ["", "Path", "Name", "ndim", "Shape", "Size"]
            )
        else:
            self._tree_widget.setColumnCount(5)
            self._tree_widget.setHeaderLabels(["", "Name", "ndim", "Shape", "Size"])
        self._tree_widget.setRootIsDecorated(False)
        self._tree_widget.setIndentation(0)
        self._tree_widget.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._tree_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        header = self._tree_widget.header()
        if header is not None:
            header.setStretchLastSection(False)
        self._tree_widget.setAlternatingRowColors(True)
        self._tree_widget.setUniformRowHeights(True)

        self._path_tree: QtWidgets.QTreeWidget | None = None
        if show_path_tree:
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
            self._path_tree = QtWidgets.QTreeWidget(splitter)
            self._path_tree.setColumnCount(1)
            self._path_tree.setHeaderHidden(True)
            self._path_tree.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
            )
            self._path_tree.setUniformRowHeights(True)
            self._path_tree.setMinimumWidth(160)
            splitter.addWidget(self._path_tree)
            splitter.addWidget(self._tree_widget)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([180, 640])
            layout.addWidget(splitter)
        else:
            layout.addWidget(self._tree_widget)

        for source_index, (darr, source_path, variable_name) in enumerate(
            zip(self._data_arrays, self._source_paths, variable_names, strict=True)
        ):
            row_text = [""] * self._tree_widget.columnCount()
            if show_path_tree:
                row_text[1] = source_path.strip("/")
            row_text[name_column] = variable_name
            row_text[ndim_column] = str(darr.ndim)
            row_text[size_column] = erlab.utils.formatting.format_nbytes(darr.nbytes)
            if not _supported_shape(darr):
                row_text[ndim_column] = f"{darr.ndim} -> reduce"
            item = QtWidgets.QTreeWidgetItem(row_text)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, source_index)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            item.setTextAlignment(
                ndim_column,
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
            )
            item.setTextAlignment(
                size_column,
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
            )
            item.setToolTip(0, source_path)
            for column in range(1, self._tree_widget.columnCount()):
                item.setToolTip(column, repr(darr))
            self._tree_widget.addTopLevelItem(item)

            checkbox_container = QtWidgets.QWidget(self._tree_widget)
            checkbox_layout = QtWidgets.QHBoxLayout(checkbox_container)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setSpacing(0)
            checkbox_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            checkbox = QtWidgets.QCheckBox(checkbox_container)
            checkbox.setChecked(True)
            checkbox.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            checkbox.setToolTip("Load this variable")
            checkbox.toggled.connect(self._update_ok_enabled)
            checkbox.clicked.connect(
                lambda _checked=False, item=item: self._tree_widget.setCurrentItem(item)
            )
            checkbox_layout.addWidget(checkbox)
            item.setSizeHint(0, checkbox_container.sizeHint())
            self._tree_widget.setItemWidget(item, 0, checkbox_container)

            label = QtWidgets.QLabel(
                erlab.interactive.utils._apply_qt_accent_color(
                    erlab.utils.formatting.format_darr_shape_html(
                        darr.rename(None),
                        show_size=False,
                    )
                ),
                self,
            )
            label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.NoTextInteraction
            )
            label.setToolTip(repr(darr))
            label.setContentsMargins(4, 0, 4, 0)
            item.setSizeHint(shape_column, label.sizeHint())
            self._tree_widget.setItemWidget(item, shape_column, label)

        if self._path_tree is not None:
            self._populate_path_tree()
        for column in range(self._tree_widget.columnCount()):
            self._tree_widget.resizeColumnToContents(column)
        header = self._tree_widget.header()
        if header is not None:
            header.setSectionResizeMode(
                shape_column, QtWidgets.QHeaderView.ResizeMode.Stretch
            )

        self._details = QtWidgets.QTextBrowser(self)
        self._details.setOpenExternalLinks(False)
        self._details.setMinimumHeight(150)
        self._details.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._tree_widget.currentItemChanged.connect(self._update_details)
        layout.addWidget(self._details)

        button_row = QtWidgets.QHBoxLayout()
        self._clear_path_filter_button: QtWidgets.QPushButton | None = None
        if self._path_tree is not None:
            self._clear_path_filter_button = QtWidgets.QPushButton(
                "Clear Path Filter", self
            )
            self._clear_path_filter_button.setEnabled(False)
            self._clear_path_filter_button.clicked.connect(self._clear_path_filter)
            button_row.addWidget(self._clear_path_filter_button)
        select_all_button = QtWidgets.QPushButton("Select All", self)
        deselect_all_button = QtWidgets.QPushButton("Deselect All", self)
        select_all_button.clicked.connect(self._check_all)
        deselect_all_button.clicked.connect(self._uncheck_all)
        button_row.addWidget(select_all_button)
        button_row.addWidget(deselect_all_button)
        button_row.addStretch()

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        button_row.addWidget(self._button_box)
        layout.addLayout(button_row)

        self._tree_widget.itemChanged.connect(self._update_ok_enabled)
        if self._path_tree is not None:
            self._path_tree.currentItemChanged.connect(self._filter_for_path)
            self._tree_widget.setCurrentItem(next(self._items(), None))
        else:
            self._tree_widget.setCurrentItem(next(self._items(), None))
        self._update_ok_enabled()

    def _populate_path_tree(self) -> None:
        if self._path_tree is None:  # pragma: no cover - guarded by caller
            return

        data_paths = {
            source_path for source_path in self._source_paths if source_path != "/"
        }
        path_children: dict[str, set[str]] = {}
        for source_path in data_paths:
            parent_path = ""
            for part in source_path.strip("/").split("/"):
                path_children.setdefault(parent_path, set()).add(part)
                parent_path = f"{parent_path}/{part}"

        path_items: dict[str, QtWidgets.QTreeWidgetItem] = {}

        def compressed_child_path(parent_path: str, part: str) -> tuple[str, str]:
            label_parts = [part]
            child_path = f"{parent_path}/{part}" if parent_path else f"/{part}"
            while (
                child_path not in data_paths
                and len(path_children.get(child_path, ())) == 1
            ):
                next_part = sorted(path_children[child_path])[0]
                label_parts.append(next_part)
                child_path = f"{child_path}/{next_part}"
            return child_path, "/".join(label_parts)

        for source_path in sorted(data_paths):
            parent_item: QtWidgets.QTreeWidgetItem | None = None
            current_path = ""
            while current_path != source_path:
                remaining_path = source_path.removeprefix(current_path).strip("/")
                part = remaining_path.split("/", maxsplit=1)[0]
                current_path, label = compressed_child_path(current_path, part)
                item = path_items.get(current_path)
                if item is None:
                    item = QtWidgets.QTreeWidgetItem([label])
                    item.setData(0, QtCore.Qt.ItemDataRole.UserRole, current_path)
                    item.setToolTip(0, current_path)
                    path_items[current_path] = item
                    if parent_item is None:
                        self._path_tree.addTopLevelItem(item)
                    else:
                        parent_item.addChild(item)
                parent_item = item

        self._path_tree.expandAll()

    def _item_checkbox(self, item: QtWidgets.QTreeWidgetItem) -> QtWidgets.QCheckBox:
        widget = self._tree_widget.itemWidget(item, 0)
        checkbox = widget.findChild(QtWidgets.QCheckBox) if widget is not None else None
        if checkbox is None:  # pragma: no cover - only possible if the widget is lost
            raise RuntimeError("Missing variable selection checkbox")
        return checkbox

    def _items(self) -> collections.abc.Iterator[QtWidgets.QTreeWidgetItem]:
        stack = [
            self._tree_widget.topLevelItem(row)
            for row in reversed(range(self._tree_widget.topLevelItemCount()))
        ]
        while stack:
            item = stack.pop()
            if item is None:  # pragma: no cover - only possible if Qt drops an item
                continue
            stack.extend(item.child(row) for row in reversed(range(item.childCount())))
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) is not None:
                yield item

    @QtCore.Slot()
    def _check_all(self) -> None:
        for item in self._items():
            if not item.isHidden():
                self._item_checkbox(item).setChecked(True)

    @QtCore.Slot()
    def _uncheck_all(self) -> None:
        for item in self._items():
            if not item.isHidden():
                self._item_checkbox(item).setChecked(False)

    @QtCore.Slot()
    def _clear_path_filter(self) -> None:
        if self._path_tree is None:  # pragma: no cover - guarded by caller
            return
        self._path_tree.clearSelection()
        self._path_tree.setCurrentIndex(QtCore.QModelIndex())
        self._filter_for_path(None)

    def selected_dataarrays(self) -> tuple[_PreparedInputData, ...]:
        selected: list[_PreparedInputData] = []
        for item in self._items():
            if self._item_checkbox(item).isChecked():
                index = typing.cast(
                    "int", item.data(0, QtCore.Qt.ItemDataRole.UserRole)
                )
                data = self._data_arrays[index]
                selected.append(
                    _PreparedInputData(
                        data=data,
                        selection=self._selections[index],
                        source_ndim=data.ndim,
                        source_dtype=np.dtype(data.dtype),
                    )
                )
        return tuple(selected)

    def _update_ok_enabled(self, *_args: object) -> None:
        button = self._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if button is not None:  # pragma: no branch
            button.setEnabled(bool(self.selected_dataarrays()))

    def _filter_for_path(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None = None,
    ) -> None:
        selected_path = ""
        if current is not None:
            selected_path = typing.cast(
                "str", current.data(0, QtCore.Qt.ItemDataRole.UserRole)
            )

        if self._clear_path_filter_button is not None:
            self._clear_path_filter_button.setEnabled(selected_path != "")

        first_visible: QtWidgets.QTreeWidgetItem | None = None
        for item in self._items():
            source_path = self._source_paths[
                typing.cast("int", item.data(0, QtCore.Qt.ItemDataRole.UserRole))
            ]
            visible = (
                selected_path == ""
                or source_path == selected_path
                or source_path.startswith(f"{selected_path}/")
            )
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item

        current_item = self._tree_widget.currentItem()
        if first_visible is None:
            self._tree_widget.setCurrentItem(None)
            self._details.clear()
        elif current_item is None or current_item.isHidden():
            self._tree_widget.setCurrentItem(first_visible)
        else:
            self._update_details(current_item)

    def _update_details(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None = None,
    ) -> None:
        if current is None:
            self._details.clear()
            return

        candidate_index = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if candidate_index is None:
            self._details.clear()
            return

        data_array = self._data_arrays[typing.cast("int", candidate_index)]
        self._details.setHtml(
            erlab.interactive.utils._apply_qt_accent_color(
                erlab.utils.formatting.format_darr_html(
                    data_array,
                    show_size=False,
                    show_summary=False,
                )
            )
        )

    def accept(self) -> None:
        if not self.selected_dataarrays():
            self.reject()
            return
        super().accept()


def _select_input_dataarrays(
    data: Sequence[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
    parent: QtWidgets.QWidget | None = None,
) -> tuple[_PreparedInputData, ...] | None:
    parsed_data = _parse_input_data(data)
    if len(parsed_data) > 1 and isinstance(data, xr.Dataset | xr.DataTree):
        dialog = _SelectDataArraysDialog(parent, data)
        if not dialog.exec():
            return None
        selected = dialog.selected_dataarrays()
        if not selected:
            return None
        return _prepare_parsed_input_data(selected, parent, allow_dialog=True)
    return _prepare_parsed_input_data(parsed_data, parent, allow_dialog=True)


def _make_cursor_colors(
    clr: QtGui.QColor,
) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor, QtGui.QColor, QtGui.QColor]:
    """Given a color, return a tuple of colors used for cursors and spans.

    This function generates a set of colors based on the input color `clr` with
    pre-defined transparency levels.
    """
    clr_cursor = pg.mkColor(clr)
    clr_cursor_hover = pg.mkColor(clr)
    clr_span = pg.mkColor(clr)
    clr_span_edge = pg.mkColor(clr)

    clr_cursor.setAlphaF(0.75)
    clr_cursor_hover.setAlphaF(0.95)
    clr_span.setAlphaF(0.15)
    clr_span_edge.setAlphaF(0.35)

    return clr, clr_cursor, clr_cursor_hover, clr_span, clr_span_edge


def _plotted_associated_coord_names(
    array_slicer: ArraySlicer,
) -> tuple[Hashable, ...]:
    plotted = array_slicer.twin_coord_names
    return tuple(name for name in array_slicer.associated_coord_dims if name in plotted)


def _associated_coord_color(
    slicer_area: ImageSlicerArea, coord_name: Hashable
) -> QtGui.QColor:
    coord_names = tuple(slicer_area.array_slicer.associated_coord_dims)
    return slicer_area.TWIN_COLORS[
        coord_names.index(coord_name) % len(slicer_area.TWIN_COLORS)
    ]


def _associated_coord_icon(color: QtGui.QColor) -> QtGui.QIcon:
    img = QtGui.QImage(16, 16, QtGui.QImage.Format.Format_RGBA64)
    img.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(img)
    painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawEllipse(2, 2, 12, 12)
    painter.end()
    return QtGui.QIcon(QtGui.QPixmap.fromImage(img))


def _parse_input(
    data: Sequence[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
) -> list[xr.DataArray]:
    parsed = _prepare_parsed_input_data(
        _parse_input_data(data),
        parent=None,
        allow_dialog=False,
    )
    if parsed is None:  # pragma: no cover - cancellation is impossible here
        return []
    return [prepared.data for prepared in parsed]


def _prepare_input_data(
    data: Sequence[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
    parent: QtWidgets.QWidget | None = None,
    *,
    allow_dialog: bool = True,
) -> tuple[_PreparedInputData, ...] | None:
    return _prepare_parsed_input_data(
        _parse_input_data(data),
        parent,
        allow_dialog=allow_dialog,
    )


def _prepare_parsed_input_data(
    parsed_data: collections.abc.Iterable[_PreparedInputData],
    parent: QtWidgets.QWidget | None,
    *,
    allow_dialog: bool,
) -> tuple[_PreparedInputData, ...] | None:
    prepared_data: list[_PreparedInputData] = []
    for prepared in parsed_data:
        if _supported_shape(prepared.data):
            prepared_data.append(prepared)
            continue
        if not allow_dialog or QtWidgets.QApplication.instance() is None:
            raise ValueError(
                f"Data with {prepared.data.ndim} dimensions has more than four "
                "non-singleton dimensions. Reduce it to four or fewer dimensions "
                "before opening it in ImageTool."
            )

        from erlab.interactive.imagetool import _highdim

        dialog = _highdim._HighDimensionalReductionDialog(parent, prepared.data)
        if not dialog.exec():
            return None
        prepared_data.append(
            dataclasses.replace(
                prepared,
                data=dialog.result_data,
                operations=tuple(dialog.source_operations()),
            )
        )
    return tuple(prepared_data)


def _parse_input_data(
    data: Sequence[xr.DataArray | npt.NDArray]
    | xr.DataArray
    | npt.NDArray
    | xr.Dataset
    | xr.DataTree,
) -> list[_PreparedInputData]:
    input_cls: str = data.__class__.__name__
    if isinstance(data, np.ndarray | xr.DataArray):
        data_array = xr.DataArray(data) if not isinstance(data, xr.DataArray) else data
        if not _reducible_shape(data_array):
            raise ValueError(f"No valid data for ImageTool found in {input_cls}")
        parsed = (
            _PreparedInputData(
                data=data_array,
                selection=FileDataSelection(kind="dataarray"),
                source_ndim=data_array.ndim,
                source_dtype=np.dtype(data_array.dtype),
            ),
        )
    elif isinstance(data, xr.Dataset):
        parsed = tuple(
            _PreparedInputData(
                data=darr,
                selection=FileDataSelection(
                    kind="dataset_variable",
                    value=variable_name,
                ),
                source_ndim=darr.ndim,
                source_dtype=np.dtype(darr.dtype),
            )
            for variable_name, darr in data.data_vars.items()
            if _reducible_shape(darr)
        )
    elif isinstance(data, xr.DataTree):
        parsed = tuple(
            _PreparedInputData(
                data=darr,
                selection=FileDataSelection(
                    kind="datatree_variable",
                    value=_datatree_dataarray_selection(str(leaf.path), variable_name),
                ),
                source_ndim=darr.ndim,
                source_dtype=np.dtype(darr.dtype),
            )
            for leaf in data.leaves
            for variable_name, darr in leaf.dataset.data_vars.items()
            if _reducible_shape(darr)
        )
    else:
        if not isinstance(data, collections.abc.Sequence):
            raise TypeError(
                f"Unsupported input type {input_cls}. Expected DataArray, Dataset, "
                "DataTree, numpy array, or a list of DataArray or numpy arrays."
            )
        parsed_list: list[_PreparedInputData] = []
        for index, d in enumerate(data):
            if not isinstance(d, xr.DataArray | np.ndarray):
                raise TypeError(
                    f"Unsupported input type {input_cls}. Expected DataArray, "
                    "Dataset, DataTree, numpy array, or a list of DataArray or "
                    "numpy arrays."
                )
            data_array = xr.DataArray(d) if not isinstance(d, xr.DataArray) else d
            if not _reducible_shape(data_array):
                continue
            parsed_list.append(
                _PreparedInputData(
                    data=data_array,
                    selection=FileDataSelection(kind="sequence_index", value=index),
                    source_ndim=data_array.ndim,
                    source_dtype=np.dtype(data_array.dtype),
                )
            )
        parsed = tuple(parsed_list)

    if len(parsed) == 0:
        raise ValueError(f"No valid data for ImageTool found in {input_cls}")

    return list(parsed)
