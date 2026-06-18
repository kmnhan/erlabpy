"""Dialog for reducing high-dimensional data before opening ImageTool."""

from __future__ import annotations

import typing

from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool import provenance
from erlab.interactive.imagetool.dialogs import (
    _current_reducer,
    _populate_reducer_combo,
    _ScalarSelectionControls,
)

if typing.TYPE_CHECKING:
    from collections.abc import Hashable

    import xarray as xr


class _ReduceDimensionRow:
    def __init__(
        self,
        dialog: _HighDimensionalReductionDialog,
        axis: int,
        dim: Hashable,
        row: int,
        *,
        require_choice: bool,
    ) -> None:
        self.axis = axis
        self.dim = dim

        data = dialog.data
        dim_label = QtWidgets.QLabel(str(dim))
        dim_label.setObjectName(f"reduce_dimension_name_{axis}")
        size_label = QtWidgets.QLabel(str(data.sizes[dim]))
        size_label.setObjectName(f"reduce_dimension_size_{axis}")
        size_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        self.action_combo = QtWidgets.QComboBox()
        self.action_combo.setObjectName(f"reduce_dimension_action_{axis}")
        self.action_combo.addItem("Choose...", None)
        self.action_combo.addItem("Keep", "keep")
        self.action_combo.addItem("Select", "select")
        self.action_combo.addItem("Aggregate", "aggregate")
        self.action_combo.setCurrentIndex(0 if require_choice else 1)

        self.scalar_controls = _ScalarSelectionControls(
            data,
            dim,
            axis,
            object_name_prefix="reduce_dimension",
            current_index=data.sizes[dim] // 2,
            include_width=False,
            default_method="isel",
        )
        self.reducer_combo = QtWidgets.QComboBox()
        self.reducer_combo.setObjectName(f"reduce_dimension_reducer_{axis}")
        _populate_reducer_combo(self.reducer_combo)

        for column, widget in enumerate(
            (
                dim_label,
                size_label,
                self.action_combo,
                self.scalar_controls.method_combo,
                self.scalar_controls.stack,
                self.reducer_combo,
            )
        ):
            dialog.grid_layout.addWidget(widget, row, column)

        self.action_combo.currentIndexChanged.connect(self.sync_widgets)
        self.action_combo.currentIndexChanged.connect(dialog.update_preview)
        self.scalar_controls.connect_changed(dialog.update_preview)
        self.reducer_combo.currentIndexChanged.connect(dialog.update_preview)
        self.sync_widgets()

    @property
    def action(
        self,
    ) -> typing.Literal["keep", "select", "aggregate"] | None:
        return typing.cast(
            "typing.Literal['keep', 'select', 'aggregate'] | None",
            self.action_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    def sync_widgets(self) -> None:
        action = self.action
        self.scalar_controls.method_combo.setEnabled(action == "select")
        self.scalar_controls.stack.setEnabled(action == "select")
        self.reducer_combo.setEnabled(action == "aggregate")

    def selection_indexer(self) -> tuple[Hashable, typing.Any] | None:
        if self.action != "select":
            return None
        return self.scalar_controls.indexer()

    def aggregate_operation(self) -> provenance.QSelAggregationOperation | None:
        if self.action != "aggregate":
            return None
        return provenance.QSelAggregationOperation(
            dims=(self.dim,),
            func=_current_reducer(self.reducer_combo),
        )


class _HighDimensionalReductionDialog(QtWidgets.QDialog):
    """Dialog used before opening data with more than four effective dimensions."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        data: xr.DataArray,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reduce Dimensions to Open")
        self.setModal(True)
        if parent is not None:
            self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.data = data
        self._result_data: xr.DataArray | None = None

        layout = QtWidgets.QVBoxLayout(self)

        visible_dimension_count = sum(size != 1 for size in data.shape)
        heading = QtWidgets.QLabel(
            "ImageTool opens data when 2 to 4 dimensions have size greater "
            f"than 1. This data has {visible_dimension_count}. Select or "
            "aggregate dimensions to create a non-empty 2D, 3D, or 4D result.",
            self,
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(8)
        self.grid_layout.setVerticalSpacing(4)
        for column, label in enumerate(
            ("Dimension", "Size", "Action", "Method", "Selection", "Reducer")
        ):
            header = QtWidgets.QLabel(label)
            header.setObjectName(f"reduce_dimension_header_{column}")
            self.grid_layout.addWidget(header, 0, column)
        layout.addLayout(self.grid_layout)

        non_singleton_dims = [dim for dim in data.dims if data.sizes[dim] != 1]
        keep_without_choice = set(non_singleton_dims[:4])
        self.rows: list[_ReduceDimensionRow] = []
        for axis, dim in enumerate(data.dims):
            self.rows.append(
                _ReduceDimensionRow(
                    self,
                    axis,
                    dim,
                    axis + 1,
                    require_choice=(
                        data.sizes[dim] != 1 and dim not in keep_without_choice
                    ),
                )
            )

        self.preview_label = QtWidgets.QLabel(self)
        self.preview_label.setObjectName("reduce_dimension_result_preview")
        layout.addWidget(self.preview_label)

        self.code_preview = QtWidgets.QPlainTextEdit(self)
        self.code_preview.setObjectName("reduce_dimension_code_preview")
        self.code_preview.setReadOnly(True)
        self.code_preview.setMaximumBlockCount(4)
        self.code_preview.setMaximumHeight(
            4 * QtGui.QFontMetrics(self.code_preview.font()).height()
        )
        layout.addWidget(self.code_preview)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Open
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.copy_button = QtWidgets.QPushButton("Copy Code", self)
        self.copy_button.clicked.connect(self._copy_code)
        self.button_box.addButton(
            self.copy_button, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.resize(780, self.sizeHint().height())
        self.update_preview()

    @property
    def result_data(self) -> xr.DataArray:
        if self._result_data is None:
            raise RuntimeError("No reduced data is available")
        return self._result_data

    def source_operations(self) -> list[provenance.ToolProvenanceOperation]:
        isel_kwargs: dict[Hashable, typing.Any] = {}
        sel_kwargs: dict[Hashable, typing.Any] = {}
        qsel_kwargs: dict[Hashable, typing.Any] = {}
        aggregate_operations: list[provenance.ToolProvenanceOperation] = []
        selection_target = {
            "isel": isel_kwargs,
            "sel": sel_kwargs,
            "qsel": qsel_kwargs,
        }

        for row in self.rows:
            indexer = row.selection_indexer()
            if indexer is not None:
                dim, value = indexer
                selection_target[row.scalar_controls.method][dim] = value
                continue
            aggregate_operation = row.aggregate_operation()
            if aggregate_operation is not None:
                aggregate_operations.append(aggregate_operation)

        operations: list[provenance.ToolProvenanceOperation] = []
        if isel_kwargs:
            operations.append(provenance.IselOperation(kwargs=isel_kwargs))
        if sel_kwargs:
            operations.append(provenance.SelOperation(kwargs=sel_kwargs))
        if qsel_kwargs:
            operations.append(provenance.QSelOperation(kwargs=qsel_kwargs))
        operations.extend(aggregate_operations)
        return operations

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        for operation in self.source_operations():
            data = operation.apply(data, parent_data=data)
        return data

    def make_code(self) -> str:
        try:
            return provenance.operations_expression_code(
                self.source_operations(),
                "data",
            )
        except Exception:
            return ""

    def _choices_complete(self) -> bool:
        return all(row.action is not None for row in self.rows)

    @QtCore.Slot()
    @QtCore.Slot(int)
    @QtCore.Slot(bool)
    @QtCore.Slot(object)
    def update_preview(self, *_args: object) -> None:
        code = self.make_code()
        self.code_preview.setPlainText(code)
        ok_button = self.button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Open
        )

        if not self._choices_complete():
            self._result_data = None
            self.preview_label.setText("Choose an action for each required dimension.")
            if ok_button is not None:
                ok_button.setEnabled(False)
            return

        try:
            selected = self.process_data(self.data)
        except Exception as exc:
            self._result_data = None
            self.preview_label.setText(f"Invalid reduction: {exc}")
            if ok_button is not None:
                ok_button.setEnabled(False)
            return

        if selected.ndim == 1:
            processed_ndim = 2
        elif selected.ndim > 4:
            processed_ndim = len(tuple(size for size in selected.shape if size != 1))
        else:
            processed_ndim = selected.ndim

        shape = " x ".join(str(size) for size in selected.shape)
        dims = ", ".join(str(dim) for dim in selected.dims)
        if processed_ndim != selected.ndim:
            ndim_text = f"{selected.ndim}D, opens as {processed_ndim}D"
        else:
            ndim_text = f"{selected.ndim}D"
        self.preview_label.setText(f"{ndim_text} ({shape}) [{dims}]")
        valid = 2 <= processed_ndim <= 4 and selected.size > 0
        self._result_data = selected if valid else None
        if ok_button is not None:
            ok_button.setEnabled(valid)

    @QtCore.Slot()
    def _copy_code(self) -> None:
        code = self.make_code()
        if code:
            erlab.interactive.utils.copy_to_clipboard(code)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to Copy", "Generated code is empty."
            )

    @QtCore.Slot()
    def accept(self) -> None:
        self.update_preview()
        if self._result_data is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Reduction",
                "Choose reductions that leave data with 2 to 4 dimensions.",
            )
            return
        super().accept()
