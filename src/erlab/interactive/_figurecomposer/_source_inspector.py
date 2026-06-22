"""Data-source inspection widgets for Figure Composer."""

from __future__ import annotations

import contextlib
import typing

import numpy as np
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._figurecomposer._sources import (
    _public_source_data,
    _source_display_label,
)
from erlab.interactive._figurecomposer._state import (
    FigureMethodFamily,
    FigureMethodPlotValueState,
    FigureOperationKind,
    FigureOperationState,
    FigureSourceState,
)

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import xarray as xr


_MAX_REPR_CHARS = 120
_MAX_SAMPLE_VALUES = 5
_MAX_COORD_VALUES_FOR_STATS = 200_000


def _clip_text(text: str, limit: int = _MAX_REPR_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _dims_text(dims: Sequence[typing.Hashable]) -> str:
    return ", ".join(str(dim) for dim in dims) or "scalar"


def _shape_text(shape: Sequence[int]) -> str:
    return " × ".join(str(size) for size in shape) or "scalar"


def _sizes_text(data: xr.DataArray) -> str:
    return ", ".join(f"{dim}: {size}" for dim, size in data.sizes.items()) or "scalar"


def _array_is_lazy(value: object) -> bool:
    return callable(getattr(value, "__dask_graph__", None))


def _safe_coord_values(coord: xr.DataArray) -> np.ndarray | None:
    if coord.size > _MAX_COORD_VALUES_FOR_STATS or _array_is_lazy(coord.data):
        return None
    with contextlib.suppress(Exception):
        return np.asarray(coord.values)
    return None


def _format_scalar(value: typing.Any) -> str:
    with contextlib.suppress(Exception):
        scalar = np.asarray(value).item()
        if isinstance(scalar, float):
            return f"{scalar:g}"
        return str(scalar)
    return _clip_text(repr(value), 48)


def _sample_text(values: np.ndarray | None, size: int) -> str:
    if values is None:
        return "not loaded" if size else ""
    flat = values.ravel()
    if flat.size == 0:
        return ""
    if flat.size <= _MAX_SAMPLE_VALUES:
        return ", ".join(_format_scalar(value) for value in flat)
    head = ", ".join(_format_scalar(value) for value in flat[: _MAX_SAMPLE_VALUES - 1])
    return f"{head}, …, {_format_scalar(flat[-1])}"


def _numeric_range_text(values: np.ndarray | None) -> str:
    if values is None or values.size == 0:
        return ""
    if not np.issubdtype(values.dtype, np.number):
        return ""
    with contextlib.suppress(Exception):
        numeric = values.astype(float, copy=False).ravel()
        finite = numeric[np.isfinite(numeric)]
        if finite.size:
            return f"{np.nanmin(finite):g} … {np.nanmax(finite):g}"
    return ""


def _coord_order_text(values: np.ndarray | None) -> str:
    if values is None or values.ndim != 1 or values.size < 2:
        return ""
    if not np.issubdtype(values.dtype, np.number):
        return ""
    with contextlib.suppress(Exception):
        numeric = values.astype(float, copy=False)
        if not np.all(np.isfinite(numeric)):
            return ""
        diff = np.diff(numeric)
        if np.all(diff >= 0):
            direction = "ascending"
        elif np.all(diff <= 0):
            direction = "descending"
        else:
            return "nonmonotonic"
        uniform = np.allclose(diff, diff[0]) if diff.size else True
        return f"{direction}, uniform" if uniform else direction
    return ""


def _attr_value_text(value: typing.Any) -> tuple[str, str]:
    if isinstance(value, np.ndarray):
        full = f"array(shape={value.shape}, dtype={value.dtype})"
        return full, full
    text = repr(value)
    return _clip_text(text), text


def source_metadata_tooltip(
    source: FigureSourceState | None,
    name: str,
    data: xr.DataArray | None,
) -> str:
    """Return a compact source tooltip with public DataArray metadata."""
    lines = [_source_display_label(source, name, disambiguate=False)]
    if source is not None and (source.label.strip() or name) != name:
        lines.append(f"Alias: {name}")
    if data is None:
        lines.append("DataArray: unavailable")
        return "\n".join(lines)
    public = _public_source_data(data)
    lines.extend(
        (
            f"Dims: {_sizes_text(public)}",
            f"Shape: {_shape_text(public.shape)}",
            f"Dtype: {public.dtype}",
            f"Size: {erlab.utils.formatting.format_nbytes(public.nbytes)}",
        )
    )
    return "\n".join(lines)


def source_value_tooltip(
    data: xr.DataArray | None,
    value: tuple[str, str | None],
    *,
    axis: str,
) -> str:
    """Return a compact tooltip for an ax.plot picked-value row."""
    axis_name = axis.upper()
    kind, name = value
    if data is None:
        return f"{axis_name} values are unavailable because the source is missing."
    public = _public_source_data(data)
    if kind == "data":
        squeezed = public.squeeze(drop=True)
        return "\n".join(
            (
                f"Use DataArray values as {axis_name} values.",
                f"Dims: {_sizes_text(squeezed)}",
                f"Shape: {_shape_text(squeezed.shape)}",
                f"Dtype: {squeezed.dtype}",
            )
        )
    coord = _coord_by_name(public, name)
    if coord is None:
        return f"Coordinate {name!r} is not available."
    return "\n".join(
        (
            f"Use coordinate {name!r} as {axis_name} values.",
            f"Coord dims: {_dims_text(coord.dims)}",
            f"Shape: {_shape_text(coord.shape)}",
            f"Dtype: {coord.dtype}",
        )
    )


def _coord_by_name(data: xr.DataArray, name: str | None) -> xr.DataArray | None:
    if name is None:
        return None
    coord = data.coords.get(name)
    if coord is not None:
        return coord
    for coord_name, coord_data in data.coords.items():
        if str(coord_name) == name:
            return coord_data
    return None


class SourceInspectorWidget(QtWidgets.QWidget):
    """Modeless inspector for Figure Composer source DataArrays."""

    followSelectionChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureComposerSourceInspector")
        self._source_name: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        title_layout = QtWidgets.QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(1)
        self.title_label = QtWidgets.QLabel("No source selected", header)
        self.title_label.setObjectName("figureComposerSourceInspectorTitle")
        self.title_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.subtitle_label = QtWidgets.QLabel("", header)
        self.subtitle_label.setObjectName("figureComposerSourceInspectorSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setEnabled(False)
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)
        header_layout.addLayout(title_layout, 1)
        self.follow_button = QtWidgets.QToolButton(header)
        self.follow_button.setObjectName("figureComposerSourceInspectorFollowButton")
        self.follow_button.setAccessibleName("Follow Selection")
        self.follow_button.setCheckable(True)
        self.follow_button.setChecked(True)
        self.follow_button.setAutoRaise(True)
        self.follow_button.setToolTip(
            "Follow the selected source or the source used by the selected step."
        )
        self.follow_button.setIcon(
            erlab.interactive.utils.qtawesome.icon("mdi6.pin-off")
        )
        self.follow_button.toggled.connect(self._follow_toggled)
        header_layout.addWidget(self.follow_button)
        layout.addWidget(header)

        self.plot_table = self._new_table(
            "figureComposerSourceInspectorPlotTable",
            ("Plotted data", "Value"),
        )
        self.plot_table.setMaximumHeight(112)
        layout.addWidget(self.plot_table)

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.setObjectName("figureComposerSourceInspectorTabs")
        self.summary_table = self._new_table(
            "figureComposerSourceInspectorSummaryTable",
            ("Property", "Value"),
        )
        self.coord_table = self._new_table(
            "figureComposerSourceInspectorCoordTable",
            ("Coordinate", "Dims", "Shape", "Dtype", "Range", "Order", "Sample"),
        )
        self.attr_table = self._new_table(
            "figureComposerSourceInspectorAttrTable",
            ("Attribute", "Value"),
        )
        self.tabs.addTab(self.summary_table, "Summary")
        self.tabs.addTab(self.coord_table, "Coordinates")
        self.tabs.addTab(self.attr_table, "Attributes")
        layout.addWidget(self.tabs, 1)

    @staticmethod
    def _new_table(object_name: str, labels: Sequence[str]) -> QtWidgets.QTreeWidget:
        table = QtWidgets.QTreeWidget()
        table.setObjectName(object_name)
        table.setRootIsDecorated(False)
        table.setAlternatingRowColors(True)
        table.setUniformRowHeights(True)
        table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        table.setHeaderLabels(tuple(labels))
        table.header().setStretchLastSection(True)
        return table

    @QtCore.Slot(bool)
    def _follow_toggled(self, checked: bool) -> None:
        icon = "mdi6.pin-off" if checked else "mdi6.pin"
        self.follow_button.setIcon(erlab.interactive.utils.qtawesome.icon(icon))
        self.followSelectionChanged.emit(checked)

    def source_name(self) -> str | None:
        return self._source_name

    def follows_selection(self) -> bool:
        return self.follow_button.isChecked()

    def set_follows_selection(self, checked: bool) -> None:
        self.follow_button.setChecked(checked)

    def set_context(
        self,
        *,
        source_name: str | None,
        source_state: FigureSourceState | None,
        data: xr.DataArray | None,
        operation: FigureOperationState | None,
        operation_source_names: Sequence[str],
        source_data: Mapping[str, xr.DataArray],
        source_states: Mapping[str, FigureSourceState],
    ) -> None:
        self._source_name = source_name
        self._populate_header(source_name, source_state, data, operation_source_names)
        self._populate_summary(source_name, source_state, data)
        self._populate_coords(data)
        self._populate_attrs(data)
        self._populate_plot_summary(operation, source_data, source_states)

    def _populate_header(
        self,
        source_name: str | None,
        source_state: FigureSourceState | None,
        data: xr.DataArray | None,
        operation_source_names: Sequence[str],
    ) -> None:
        if source_name is None:
            self.title_label.setText("No source selected")
            self.subtitle_label.setText("Select a source or a data-reading step.")
            return
        self.title_label.setText(_source_display_label(source_state, source_name))
        if data is None:
            self.subtitle_label.setText(f"Alias: {source_name} · unavailable")
            return
        public = _public_source_data(data)
        used_text = (
            " · used by selected step"
            if source_name in set(operation_source_names)
            else ""
        )
        self.subtitle_label.setText(
            f"Alias: {source_name} · {_sizes_text(public)} · {public.dtype}"
            f" · {erlab.utils.formatting.format_nbytes(public.nbytes)}{used_text}"
        )

    def _populate_summary(
        self,
        source_name: str | None,
        source_state: FigureSourceState | None,
        data: xr.DataArray | None,
    ) -> None:
        rows: list[tuple[str, str]] = []
        if source_name is not None:
            rows.append(("Alias", source_name))
        if source_state is not None:
            display_name = source_state.label.strip() or source_name or ""
            rows.append(("Display name", display_name))
            if source_state.node_uid:
                rows.append(("Workspace node", source_state.node_uid))
        if data is None:
            rows.append(("Status", "Unavailable"))
            self._set_rows(self.summary_table, rows)
            return
        public = _public_source_data(data)
        rows.extend(
            (
                ("DataArray name", "" if public.name is None else str(public.name)),
                ("Dims", _sizes_text(public)),
                ("Shape", _shape_text(public.shape)),
                ("Dtype", str(public.dtype)),
                ("Size", erlab.utils.formatting.format_nbytes(public.nbytes)),
                ("Coordinates", str(len(public.coords))),
                ("Attributes", str(len(public.attrs))),
            )
        )
        self._set_rows(self.summary_table, rows)

    def _populate_coords(self, data: xr.DataArray | None) -> None:
        self.coord_table.clear()
        if data is None:
            return
        public = _public_source_data(data)
        for coord_name, coord in public.coords.items():
            values = _safe_coord_values(coord)
            item = QtWidgets.QTreeWidgetItem(
                (
                    str(coord_name),
                    _dims_text(coord.dims),
                    _shape_text(coord.shape),
                    str(coord.dtype),
                    _numeric_range_text(values),
                    _coord_order_text(values),
                    _sample_text(values, coord.size),
                )
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(coord_name))
            item.setData(
                0,
                QtCore.Qt.ItemDataRole.UserRole + 1,
                tuple(str(dim) for dim in coord.dims),
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 2, str(coord.dtype))
            self.coord_table.addTopLevelItem(item)
        self.coord_table.resizeColumnToContents(0)
        self.coord_table.resizeColumnToContents(1)
        self.coord_table.resizeColumnToContents(2)

    def _populate_attrs(self, data: xr.DataArray | None) -> None:
        self.attr_table.clear()
        if data is None:
            return
        public = _public_source_data(data)
        for key, value in public.attrs.items():
            display, full = _attr_value_text(value)
            item = QtWidgets.QTreeWidgetItem((str(key), display))
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(key))
            item.setData(1, QtCore.Qt.ItemDataRole.UserRole, full)
            item.setToolTip(1, full)
            self.attr_table.addTopLevelItem(item)
        self.attr_table.resizeColumnToContents(0)

    def _populate_plot_summary(
        self,
        operation: FigureOperationState | None,
        source_data: Mapping[str, xr.DataArray],
        source_states: Mapping[str, FigureSourceState],
    ) -> None:
        self._set_rows(
            self.plot_table,
            _operation_summary_rows(operation, source_data, source_states),
        )

    @staticmethod
    def _set_rows(
        table: QtWidgets.QTreeWidget, rows: Sequence[tuple[str, str]]
    ) -> None:
        table.clear()
        for key, value in rows:
            item = QtWidgets.QTreeWidgetItem((key, value))
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, key)
            item.setData(1, QtCore.Qt.ItemDataRole.UserRole, value)
            table.addTopLevelItem(item)
        table.resizeColumnToContents(0)


def _operation_summary_rows(
    operation: FigureOperationState | None,
    source_data: Mapping[str, xr.DataArray],
    source_states: Mapping[str, FigureSourceState],
) -> tuple[tuple[str, str], ...]:
    if operation is None:
        return (("Step", "No recipe step selected"),)
    if operation.kind == FigureOperationKind.METHOD:
        return _method_summary_rows(operation, source_data, source_states)
    if operation.kind == FigureOperationKind.LINE:
        return _line_summary_rows(operation, source_data)
    if operation.kind == FigureOperationKind.PLOT_SLICES:
        return _plot_slices_summary_rows(operation, source_data)
    return (("Step", operation.kind.value),)


def _method_summary_rows(
    operation: FigureOperationState,
    source_data: Mapping[str, xr.DataArray],
    source_states: Mapping[str, FigureSourceState],
) -> tuple[tuple[str, str], ...]:
    if not (
        operation.method_family == FigureMethodFamily.AXES
        and operation.method_name == "plot"
        and operation.method_plot_data_mode == "from_data"
    ):
        return (("Step", f"{operation.method_family.value}.{operation.method_name}"),)

    rows: list[tuple[str, str]] = [("Step", "ax.plot picked data")]
    lengths: dict[str, int] = {}
    for axis, state in (("X", operation.method_plot_x), ("Y", operation.method_plot_y)):
        if state is None:
            rows.append((axis, "default positions" if axis == "X" else "not set"))
            continue
        summary, length = _plot_value_summary(state, source_data, source_states)
        rows.append((axis, summary))
        if length is not None:
            lengths[axis] = length
    if {"X", "Y"} <= set(lengths) and lengths["X"] != lengths["Y"]:
        rows.append(("Status", "X and Y lengths differ"))
    elif operation.method_plot_y is None:
        rows.append(("Status", "Y values are required"))
    return tuple(rows)


def _plot_value_summary(
    state: FigureMethodPlotValueState,
    source_data: Mapping[str, xr.DataArray],
    source_states: Mapping[str, FigureSourceState],
) -> tuple[str, int | None]:
    data = source_data.get(state.source)
    source = _source_display_label(source_states.get(state.source), state.source)
    if data is None:
        return f"{source}: unavailable", None
    public = _public_source_data(data)
    if state.kind == "data":
        value = public.squeeze(drop=True)
        if value.ndim != 1:
            return f"{source}: data values are {value.ndim}D", None
        return f"{source}: data values, length {value.size}, {value.dtype}", value.size
    if state.name is None:
        return f"{source}: coordinate not set", None
    coord = _coord_by_name(public, state.name)
    if coord is None:
        return f"{source}: missing coordinate {state.name!r}", None
    value = coord.squeeze(drop=True)
    if value.ndim != 1:
        return f"{source}: coord {state.name!r} is {value.ndim}D", None
    summary = f"{source}: coord {state.name!r}, length {value.size}, {value.dtype}"
    return summary, value.size


def _line_summary_rows(
    operation: FigureOperationState,
    source_data: Mapping[str, xr.DataArray],
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = [("Step", "line/profile")]
    source = operation.line_source
    rows.append(("Source", "not set" if source is None else source))
    if source is not None and (data := source_data.get(source)) is not None:
        public = _public_source_data(data)
        rows.append(("Input", f"{_sizes_text(public)}, {public.dtype}"))
        if operation.line_iter_dim:
            count = public.sizes.get(operation.line_iter_dim)
            rows.append(
                (
                    "Profiles",
                    (
                        f"{operation.line_iter_dim}: {count}"
                        if count is not None
                        else f"{operation.line_iter_dim}: unavailable"
                    ),
                )
            )
        if operation.line_x or operation.line_y:
            rows.append(
                (
                    "Axes",
                    (
                        f"x={operation.line_x or 'index'}, "
                        f"y={operation.line_y or 'values'}"
                    ),
                )
            )
    return tuple(rows)


def _plot_slices_summary_rows(
    operation: FigureOperationState,
    source_data: Mapping[str, xr.DataArray],
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = [("Step", "plot_slices")]
    rows.append(("Sources", str(len(operation.sources))))
    if operation.sources:
        rows.append(("Source aliases", ", ".join(operation.sources)))
    if operation.slice_dim:
        if operation.slice_values_mode == "all":
            counts: list[int] = []
            for source in operation.sources:
                data = source_data.get(source)
                if data is None:
                    continue
                public = _public_source_data(data)
                if operation.slice_dim in public.sizes:
                    counts.append(public.sizes[operation.slice_dim])
            if counts:
                count = max(counts)
                plotted = (count + operation.slice_values_thin - 1) // (
                    operation.slice_values_thin
                )
                rows.append(
                    ("Slice values", f"{operation.slice_dim}: {plotted} plotted")
                )
            else:
                rows.append(("Slice values", f"{operation.slice_dim}: unavailable"))
        elif operation.slice_values:
            rows.append(
                (
                    "Slice values",
                    f"{operation.slice_dim}: {len(operation.slice_values)} entered",
                )
            )
        else:
            rows.append(("Slice values", f"{operation.slice_dim}: current selection"))
    elif operation.slice_kwargs:
        rows.append(("Selection", f"{len(operation.slice_kwargs)} selectors"))
    else:
        rows.append(("Selection", "current view"))
    return tuple(rows)
