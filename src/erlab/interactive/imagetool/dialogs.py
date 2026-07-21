"""Dialogs for data manipulation found in the menu bar."""

from __future__ import annotations

import ast
import contextlib
import math
import operator
import traceback
import typing
import weakref
from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool import _kspace_conversion
from erlab.interactive.imagetool._dialog_widgets import (
    CoordinateEditorWidget,
    CoordinateGridWidget,
)
from erlab.interactive.imagetool._provenance._code import _provenance_value_code
from erlab.interactive.imagetool._provenance._model import (
    ToolProvenanceOperation,
    ToolProvenanceSpec,
    compose_full_provenance,
    full_data,
    operations_expression_code,
    public_data,
    require_live_source_spec,
)
from erlab.interactive.imagetool._provenance._operations import (
    AffineCoordOperation,
    AssignAttrsOperation,
    AssignCoord1DOperation,
    AssignCoordsOperation,
    AssignScalarCoordOperation,
    AverageOperation,
    BoxcarFilterOperation,
    CoarsenOperation,
    CorrectWithEdgeOperation,
    DivideByCoordOperation,
    GaussianFilterOperation,
    ImageDerivativeOperation,
    InterpolationOperation,
    IselOperation,
    KspaceConfigurationOperation,
    KspaceConvertOperation,
    KspaceInnerPotentialOperation,
    KspaceSetNormalOperation,
    KspaceWorkFunctionOperation,
    LeadingEdgeOperation,
    MaskWithPolygonOperation,
    NormalizeOperation,
    QSelAggregationOperation,
    QSelOperation,
    RemoveMeshOperation,
    RenameDimsCoordsOperation,
    RestoreNonuniformDimsOperation,
    RotateOperation,
    SelOperation,
    SliceAlongPathOperation,
    SortByOperation,
    SqueezeOperation,
    SwapDimsOperation,
    SymmetrizeNfoldOperation,
    SymmetrizeOperation,
    ThinOperation,
    UniformInterpolationOperation,
)

__all__ = [
    "AggregateDialog",
    "AssignAttrsDialog",
    "AssignCoordsDialog",
    "CoarsenDialog",
    "CropDialog",
    "CropToViewDialog",
    "DataFilterDialog",
    "DataTransformDialog",
    "DivideByCoordDialog",
    "EdgeCorrectionDialog",
    "GaussianFilterDialog",
    "InterpolationDialog",
    "KspaceConversionDialog",
    "LeadingEdgeDialog",
    "NormalizeDialog",
    "ROIMaskDialog",
    "ROIPathDialog",
    "RenameDimsCoordsDialog",
    "RotationDialog",
    "SelectionDialog",
    "SortByDialog",
    "SqueezeDialog",
    "SwapDimsDialog",
    "SymmetrizeDialog",
    "SymmetrizeNfoldDialog",
    "ThinDialog",
]

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import xarray as xr

    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.plot_items import ItoolPolyLineROI
    from erlab.interactive.imagetool.slicer import ArraySlicer
    from erlab.interactive.imagetool.viewer import ImageSlicerArea
    from erlab.interactive.imagetool.viewer_state import ColorMapState

_GAUSSIAN_FWHM_FACTOR: float = 2 * math.sqrt(2 * math.log(2))


def _show_warning_with_traceback(
    parent: QtWidgets.QWidget,
    title: str,
    text: str,
) -> None:
    erlab.interactive.utils.MessageDialog(
        parent,
        title=title,
        text=text,
        detailed_text=erlab.interactive.utils._format_traceback(traceback.format_exc()),
        buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
        icon_pixmap=QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
    ).exec()


def _set_combo_data(combo: QtWidgets.QComboBox, value: typing.Any) -> bool:
    index = combo.findData(value, QtCore.Qt.ItemDataRole.UserRole)
    if index < 0:
        return False
    combo.setCurrentIndex(index)
    return True


def _set_combo_text(combo: QtWidgets.QComboBox, value: str) -> bool:
    index = combo.findText(value)
    if index < 0:
        return False
    combo.setCurrentIndex(index)
    return True


def _set_spin_value(
    spin: (
        QtWidgets.QSpinBox
        | QtWidgets.QDoubleSpinBox
        | erlab.interactive.utils.BetterSpinBox
    ),
    value: float,
    *,
    label: str,
) -> None:
    """Restore a spin value without silently clamping the operation model."""
    if not spin.minimum() <= value <= spin.maximum():
        raise ValueError(f"{label} {value!r} is outside the editor range")
    if isinstance(spin, QtWidgets.QSpinBox):
        spin.setValue(int(value))
    else:
        spin.setValue(float(value))


_AGGREGATION_REDUCERS: dict[str, str] = {
    "mean": "Mean",
    "min": "Minimum",
    "max": "Maximum",
    "sum": "Sum",
}


def _populate_reducer_combo(combo: QtWidgets.QComboBox) -> None:
    for reducer, label in _AGGREGATION_REDUCERS.items():
        combo.addItem(label, userData=reducer)


def _current_reducer(
    combo: QtWidgets.QComboBox,
) -> typing.Literal["mean", "min", "max", "sum"]:
    return typing.cast(
        "typing.Literal['mean', 'min', 'max', 'sum']",
        combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
    )


class _ScalarSelectionControls:
    """Reusable widgets for selecting one scalar value along a dimension."""

    def __init__(
        self,
        data: xr.DataArray,
        dim: Hashable,
        axis: int,
        *,
        object_name_prefix: str,
        current_index: int | None = None,
        bin_value: float | None = None,
        is_binned: bool = False,
        include_width: bool = True,
        default_method: typing.Literal["qsel", "sel", "isel"] = "qsel",
    ) -> None:
        self.dim = dim
        self.axis = axis
        self._include_width = include_width

        dim_size = data.sizes[dim]
        index_minimum = -dim_size if dim_size > 0 else 0
        coord = np.asarray(data[dim].values)
        try:
            coord_float = np.asarray(coord, dtype=float)
        except (TypeError, ValueError):
            coord_float = np.arange(dim_size, dtype=float)
            self._has_numeric_coord = False
        else:
            self._has_numeric_coord = True
        if coord_float.size == 0:
            coord_float = np.array([0.0])

        self.coord = coord_float
        self.coord_ascending = bool(coord_float[0] <= coord_float[-1])
        if current_index is None:
            current_index = dim_size // 2
        current_index = max(0, min(int(current_index), max(dim_size - 1, 0)))
        current_value = float(coord_float[current_index])

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.setObjectName(f"{object_name_prefix}_method_{axis}")
        self.method_combo.setToolTip(
            "Choose how this dimension is selected: qsel selects the nearest "
            "coordinate value, sel uses coordinate labels, and isel uses "
            "integer indices."
        )
        if self._has_numeric_coord:
            for method in ("qsel", "sel", "isel"):
                self.method_combo.addItem(method, method)
        else:
            self.method_combo.addItem("isel", "isel")
            self.method_combo.setToolTip(
                "This coordinate is not numeric, so scalar selection uses integer "
                "indices."
            )
        _set_combo_data(self.method_combo, default_method)

        self.index_spin = erlab.interactive.utils.BetterSpinBox(
            integer=True,
            compact=False,
            minimum=index_minimum,
            maximum=max(dim_size - 1, 0),
            value=current_index,
        )
        self.index_spin.setObjectName(f"{object_name_prefix}_index_{axis}")

        self.value_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            decimals=6,
            exact_float=True,
            significant=True,
            minimum=float(np.nanmin(coord_float)),
            maximum=float(np.nanmax(coord_float)),
            value=current_value,
        )
        self.value_spin.setObjectName(f"{object_name_prefix}_value_{axis}")

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.value_spin)
        self.stack.addWidget(self.index_spin)

        self.width_widget = QtWidgets.QWidget()
        self.width_widget.setToolTip(
            "For point qsel selections, include nearby coordinate values within "
            "this width."
        )
        width_layout = QtWidgets.QHBoxLayout(self.width_widget)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.setSpacing(3)
        self.width_check = QtWidgets.QCheckBox()
        self.width_check.setObjectName(f"{object_name_prefix}_width_enabled_{axis}")
        self.width_check.setToolTip(self.width_widget.toolTip())
        self.width_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            decimals=6,
            exact_float=True,
            significant=True,
            minimum=0.0,
            value=0.0 if bin_value is None else float(bin_value),
        )
        self.width_spin.setObjectName(f"{object_name_prefix}_width_{axis}")
        self.width_spin.setToolTip(self.width_widget.toolTip())
        self.width_check.setChecked(is_binned and bin_value is not None)
        width_layout.addWidget(self.width_check)
        width_layout.addWidget(self.width_spin)
        self.width_widget.setVisible(include_width)

        self.method_combo.currentIndexChanged.connect(self.sync_widgets)
        self.width_check.toggled.connect(self.sync_widgets)
        self.sync_widgets()

    @property
    def method(self) -> typing.Literal["qsel", "sel", "isel"]:
        return typing.cast(
            "typing.Literal['qsel', 'sel', 'isel']",
            self.method_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    def connect_changed(self, slot: Callable[..., object]) -> None:
        self.method_combo.currentIndexChanged.connect(slot)
        self.index_spin.valueChanged.connect(slot)
        self.value_spin.valueChanged.connect(slot)
        self.width_check.toggled.connect(slot)
        self.width_spin.valueChanged.connect(slot)

    def sync_widgets(self) -> None:
        is_index = self.method == "isel"
        is_qsel = self.method == "qsel"
        self.stack.setCurrentIndex(1 if is_index else 0)
        self.width_widget.setEnabled(self._include_width and is_qsel)
        self.width_spin.setEnabled(
            self._include_width and is_qsel and self.width_check.isChecked()
        )

    def indexer(self) -> tuple[Hashable, typing.Any]:
        if self.method == "isel":
            return self.dim, int(self.index_spin.value())
        return self.dim, float(self.value_spin.value())

    def qsel_width_indexer(self) -> tuple[str, float] | None:
        if (
            not self._include_width
            or self.method != "qsel"
            or not self.width_check.isChecked()
        ):
            return None
        width = float(self.width_spin.value())
        if not np.isfinite(width) or width <= 0.0:
            return None
        return f"{self.dim}_width", width


class _DataManipulationDialog(QtWidgets.QDialog):
    """Parent class for a dialog that manipulates data.

    In practice, use child classes `DataTransformDialog` and `DataFilterDialog`.
    """

    whatsthis: str | None = None
    """The whatsthis text for the dialog window."""

    title: str | None = None
    """The title of the dialog window."""

    enable_copy: bool = False
    """Whether to show a button to copy the code to the clipboard.

    If True, the button will be shown in the dialog box. The `make_code` method must be
    overridden to provide the code to be copied.
    """

    operation_types: typing.ClassVar[
        tuple[
            type[ToolProvenanceOperation],
            ...,
        ]
    ] = ()
    """Operation classes this dialog can emit directly."""

    _sigCodeCopied = QtCore.Signal(str)

    def __init__(
        self,
        slicer_area: ImageSlicerArea | None,
        *,
        batch_manager: ImageToolManager | None = None,
        provenance_edit_mode: bool = False,
        dialog_parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(slicer_area if dialog_parent is None else dialog_parent)
        if self.title is not None:
            self.setWindowTitle(self.title)

        self._slicer_area: Callable[[], ImageSlicerArea | None]
        self.slicer_area = slicer_area
        self._provenance_edit_mode = provenance_edit_mode
        self._batch_manager = (
            None if batch_manager is None else weakref.ref(batch_manager)
        )

        self._layout = QtWidgets.QFormLayout()
        self.setLayout(self._layout)

        if self.whatsthis is not None:
            self.setWhatsThis(self.whatsthis)
            self.setWindowFlags(
                self.windowFlags() | QtCore.Qt.WindowType.WindowContextHelpButtonHint
            )

        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        if self.enable_copy:
            self.copy_button = QtWidgets.QPushButton("Copy Code")
            self.copy_button.clicked.connect(self._copy)
            self.buttonBox.addButton(
                self.copy_button, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            )

        self.setup_widgets()

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout_.addRow(self.buttonBox)

    @property
    def layout_(self) -> QtWidgets.QFormLayout:
        return self._layout

    @property
    def slicer_area(self) -> ImageSlicerArea:
        slicer_area = self._slicer_area()
        if slicer_area:
            return slicer_area
        raise LookupError("Parent was destroyed")

    @slicer_area.setter
    def slicer_area(self, value: ImageSlicerArea | None) -> None:
        if value is None:
            self._slicer_area = lambda: None
            return
        self._slicer_area = weakref.ref(value)

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    @property
    def batch_manager(self) -> ImageToolManager | None:
        if self._batch_manager is None:
            return None
        return self._batch_manager()

    @property
    def is_batch_mode(self) -> bool:
        return self.batch_manager is not None

    @property
    def is_provenance_edit_mode(self) -> bool:
        return self._provenance_edit_mode

    def provenance_edit_operations(
        self,
    ) -> list[ToolProvenanceOperation]:
        raise NotImplementedError

    @QtCore.Slot()
    def _accept_provenance_edit(self) -> None:
        try:
            if not self.provenance_edit_operations():
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Operation",
                    "Select at least one operation before applying.",
                )
                return
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self, "Error", "An error occurred while editing provenance."
            )
            return
        super().accept()

    @QtCore.Slot()
    def accept(self) -> None:
        if self.is_provenance_edit_mode:
            self._accept_provenance_edit()
            return
        super().accept()

    @QtCore.Slot()
    def _copy(self) -> None:
        code = self.make_code()
        if code:
            self._sigCodeCopied.emit(erlab.interactive.utils.copy_to_clipboard(code))
        else:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to Copy", "Generated code is empty."
            )

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        """Run checks before opening the dialog.

        Reimplement this method in subclasses to perform checks before showing the
        dialog.

        Returns
        -------
        QDialog.DialogCode
            Anything other than `QDialog.DialogCode.Accepted` will prevent the dialog
            from showing.
        """
        return QtWidgets.QDialog.DialogCode.Accepted

    def show(self) -> None:
        """Run checks and (asynchronously) open the dialog if they pass.

        If checks fail, the dialog is never shown; reject() is scheduled so that
        finished(Rejected) still fires (allowing uniform handling).
        """
        code = self._validate()
        if code != QtWidgets.QDialog.DialogCode.Accepted:
            erlab.interactive.utils.single_shot(self, 0, self.reject)
            return
        super().show()

    def setup_widgets(self) -> None:
        # Overridden by subclasses
        pass

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        # Overridden by subclasses
        return data

    def make_code(self) -> str:
        # Overridden by subclasses
        return ""

    def _copy_data_name(self) -> str:
        slicer_area = self._slicer_area()
        if slicer_area is None:
            return "data"
        return slicer_area.watched_data_name or "data"


class DataTransformDialog(_DataManipulationDialog):
    """Parent class for implementing data changes that affect both shape and values.

    These changes are destructive and cannot be undone. The user can choose to open the
    transformed data in a new window or replace the current data.

    - Override method `setup_widgets` to add widgets to the dialog.

    - Override method `process_data` to implement the data transformation.

    - Override `source_transform_operation` or `source_operations` to generate
      operation-backed copy code, or override `make_code` only for non-operation code.

    - Override attribute `title` to set the title of the dialog window.

    - Override attribute `enable_copy` to show or hide the copy button.

    - Override attribute `keep_colors` and `keep_color_limits` to control which
      color-related settings are migrated when opening in a new window.

    """

    keep_colors: bool = True
    """Whether to keep the color settings when opening in a new window.

    If True, the same colormap and normalization is used in the new window.
    """

    keep_color_limits: bool = True
    """Whether to also keep manual color limits when opening in a new window."""

    apply_on_nonuniform_data: bool = False
    """Whether to apply the transform on data with non-uniform dimensions.

    Set to `True` for transforms that can handle coordinates that are not evenly spaced.
    """

    copy_output_suffix: typing.ClassVar[str] = "_transformed"
    """Suffix for copied-code output variables when operations require statements."""

    _LAUNCH_MODES: typing.ClassVar[tuple[tuple[str, str, str], ...]] = (
        (
            "replace",
            "Replace Current",
            "Replace the data in this ImageTool and keep working in the same window.",
        ),
        (
            "detach",
            "Open Top-Level Window",
            "Create a separate top-level ImageTool that is detached from refresh "
            "propagation.",
        ),
        (
            "nest",
            "Open Child Window",
            "Create a child ImageTool under this node and keep its derivation for "
            "later refreshes.",
        ),
    )

    _BATCH_LAUNCH_MODES: typing.ClassVar[tuple[tuple[str, str, str], ...]] = (
        (
            "replace",
            "Replace Selected",
            "Replace the data in each selected ImageTool.",
        ),
        (
            "nest",
            "Open Child Window Under Each",
            "Create one child ImageTool under each selected ImageTool.",
        ),
        (
            "detach",
            "Open Top-Level Copies",
            "Create one detached top-level ImageTool for each selected ImageTool.",
        ),
    )

    def __init__(
        self,
        slicer_area: ImageSlicerArea | None,
        *,
        batch_manager: ImageToolManager | None = None,
        provenance_edit_mode: bool = False,
        dialog_parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            slicer_area,
            batch_manager=batch_manager,
            provenance_edit_mode=provenance_edit_mode,
            dialog_parent=dialog_parent,
        )
        if self.is_provenance_edit_mode:
            return
        self.launch_mode_combo = QtWidgets.QComboBox()
        for value, label, tooltip in self._available_launch_modes():
            self.launch_mode_combo.addItem(label, userData=value)
            idx = self.launch_mode_combo.count() - 1
            self.launch_mode_combo.setItemData(
                idx, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        self.launch_mode_combo.setCurrentIndex(
            self.launch_mode_combo.findData(
                self._default_launch_mode(), QtCore.Qt.ItemDataRole.UserRole
            )
        )
        self.launch_mode_combo.currentIndexChanged.connect(
            self._sync_launch_mode_tooltip
        )
        self._sync_launch_mode_tooltip()
        self.layout_.insertRow(-1, "Result Placement", self.launch_mode_combo)

    @property
    def launch_mode(self) -> typing.Literal["replace", "detach", "nest"]:
        return typing.cast(
            "typing.Literal['replace', 'detach', 'nest']",
            self.launch_mode_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    def _available_launch_modes(self) -> tuple[tuple[str, str, str], ...]:
        if self.is_batch_mode:
            return self._BATCH_LAUNCH_MODES
        if self._manager_target()[1] is not None:
            return self._LAUNCH_MODES
        return tuple(mode for mode in self._LAUNCH_MODES if mode[0] != "nest")

    def _default_launch_mode(self) -> typing.Literal["replace", "detach", "nest"]:
        if self.is_batch_mode:
            return "replace"
        if any(mode[0] == "nest" for mode in self._available_launch_modes()):
            return "replace"
        return "detach"

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _sync_launch_mode_tooltip(self, index: int | None = None) -> None:
        if index is None:
            index = self.launch_mode_combo.currentIndex()
        self.launch_mode_combo.setToolTip(
            str(
                self.launch_mode_combo.itemData(
                    index, QtCore.Qt.ItemDataRole.ToolTipRole
                )
                or ""
            )
        )

    def source_operations(
        self,
    ) -> list[ToolProvenanceOperation]:
        operation = self.source_transform_operation()
        return [] if operation is None else [operation]

    def provenance_edit_operations(
        self,
    ) -> list[ToolProvenanceOperation]:
        return self.source_operations()

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation | None:
        return None

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        """Restore widgets from a transform operation when supported."""
        del operation

    grouped_operation_only: typing.ClassVar[bool] = False
    """Whether this dialog only edits matched operation groups."""

    operation_group_kind: typing.ClassVar[str | None] = None
    """Optional operation-group kind this dialog edits as one unit."""

    @classmethod
    def operation_group_for_edit(
        cls,
        operations: Sequence[ToolProvenanceOperation],
        operation_index: int,
    ) -> tuple[int, int] | None:
        """Return the operation range edited together with ``operation_index``."""
        del operations, operation_index
        return None

    def restore_transform_operations(
        self,
        operations: Sequence[ToolProvenanceOperation],
    ) -> None:
        """Restore widgets from one or more transform operations."""
        if len(operations) != 1:
            raise ValueError("This dialog can only restore one provenance operation")
        self.restore_transform_operation(operations[0])

    def focus_operation_group_control(self, focus: str | None) -> None:
        """Focus a control associated with a grouped operation row."""
        del focus

    def source_spec_for_data(
        self,
        data: xr.DataArray,
        new_name: str | None = None,
    ) -> ToolProvenanceSpec:
        del new_name
        operations = self.source_operations()
        builder = public_data if self.apply_on_nonuniform_data else full_data
        if not self.apply_on_nonuniform_data:
            dimension_mapping = erlab.utils.array._nonuniform_dim_mapping(data)
            if dimension_mapping:
                operations.append(
                    RestoreNonuniformDimsOperation(dimension_mapping=dimension_mapping)
                )
        return builder(*operations)

    def source_spec(self, new_name: str | None = None) -> ToolProvenanceSpec:
        return self.source_spec_for_data(self.slicer_area.data, new_name)

    def _detached_provenance_spec(
        self,
        parent_provenance: ToolProvenanceSpec | None,
        source_spec: ToolProvenanceSpec,
        new_name: str,
    ) -> ToolProvenanceSpec:
        return self._compose_transform_provenance(
            parent_provenance,
            source_spec,
            new_name,
        )

    @staticmethod
    def _compose_transform_provenance(
        base_spec: ToolProvenanceSpec | None,
        source_spec: ToolProvenanceSpec,
        new_name: str,
    ) -> ToolProvenanceSpec:
        del new_name
        if base_spec is None:
            return source_spec
        with contextlib.suppress(TypeError):
            live_parent = require_live_source_spec(base_spec)
            if live_parent is not None:
                return live_parent.append_replacement_operations(
                    *source_spec.operations
                )
        composed = compose_full_provenance(
            base_spec,
            source_spec,
        )
        if composed is None:
            raise RuntimeError("Could not compose ImageTool transform provenance.")
        return composed

    def _compose_replace_source_spec(
        self,
        existing_spec: ToolProvenanceSpec,
        new_name: str,
    ) -> ToolProvenanceSpec:
        return self._compose_transform_provenance(
            existing_spec,
            self.source_spec(new_name),
            new_name,
        )

    def _rewrite_target_provenance(
        self,
        target: int | str,
        new_name: str,
        fallback_spec: ToolProvenanceSpec | None,
    ) -> bool:
        manager, _ = self._manager_target()
        if manager is None:
            return False
        node = manager._node_for_target(target)
        source_spec = node.displayed_source_spec
        if source_spec is not None:
            node.set_source_binding(
                self._compose_replace_source_spec(source_spec, new_name),
                auto_update=node.source_auto_update,
                state=node.source_state,
            )
            return True
        displayed_provenance = node.displayed_provenance_spec
        replay_source_data = node.resolved_replay_source_data()
        if displayed_provenance is not None:
            node.set_detached_provenance(
                self._compose_replace_source_spec(displayed_provenance, new_name),
                replay_source_data=replay_source_data,
            )
            return True
        if fallback_spec is not None:
            node.set_detached_provenance(
                fallback_spec,
                replay_source_data=replay_source_data,
            )
            return True
        return False

    def _set_current_tool_provenance(
        self,
        provenance_spec: ToolProvenanceSpec | None,
    ) -> None:
        parent = self.slicer_area.parent()
        if parent is not None and hasattr(parent, "set_provenance_spec"):
            typing.cast("typing.Any", parent).set_provenance_spec(provenance_spec)

    def _apply_source_transform(self, data: xr.DataArray) -> xr.DataArray:
        operation = self.source_transform_operation()
        if operation is None:
            return data
        return operation.apply(data)

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return self._apply_source_transform(data)

    def preflight_data(self, data: xr.DataArray) -> None:
        """Validate a target before processing it."""
        del data

    def _handle_process_error(self, exc: BaseException) -> bool:
        """Handle a transform error before the generic error dialog is shown."""
        del exc
        return False

    def make_code(self) -> str:
        try:
            operations = self.source_operations()
            input_name = self._copy_data_name()
            if not any(operation.statement_mutates_input for operation in operations):
                try:
                    return operations_expression_code(
                        operations,
                        input_name,
                    )
                except NotImplementedError:
                    if len(operations) != 1:
                        raise
                    operation = operations[0]
                    output_name = (
                        operation.preferred_replay_output_name()
                        or f"{input_name}{self.copy_output_suffix}"
                    )
                    return operation.replay_code(
                        input_name,
                        output_name=output_name,
                    )

            if not erlab.utils.misc._is_valid_identifier(input_name):
                input_name = "data"
            output_name = f"{input_name}{self.copy_output_suffix}"
            current_name = output_name
            lines = [f"{output_name} = {input_name}.copy(deep=False)"]
            for index, operation in enumerate(operations):
                if operation.statement_mutates_input:
                    lines.append(
                        operation.replay_code(current_name, output_name=current_name)
                    )
                    continue
                replay_output_name = (
                    output_name
                    if lines
                    or any(
                        later_operation.statement_mutates_input
                        for later_operation in operations[index + 1 :]
                    )
                    else current_name
                )
                lines.append(
                    operation.replay_code(
                        current_name,
                        output_name=replay_output_name,
                    )
                )
                current_name = replay_output_name
            return "\n".join(lines)
        except Exception:
            return ""

    def _itool_kwargs(
        self,
        processed,
        slicer_area: ImageSlicerArea | None = None,
    ) -> dict[str, typing.Any]:
        itool_kw: dict[str, typing.Any] = {
            "data": processed,
            "execute": False,
        }

        if slicer_area is None:
            slicer_area = self.slicer_area
        itool_kw["options_model"] = getattr(slicer_area, "_options_model", None)

        if self.keep_colors:
            color_props: ColorMapState = slicer_area.colormap_properties

            itool_kw["cmap"] = color_props["cmap"]
            itool_kw["gamma"] = color_props["gamma"]
            itool_kw["high_contrast"] = color_props["high_contrast"]
            itool_kw["zero_centered"] = color_props["zero_centered"]

            if color_props["levels_locked"] and self.keep_color_limits:
                itool_kw["vmin"], itool_kw["vmax"] = color_props["levels"]
        return itool_kw

    def _manager_target(
        self,
    ) -> tuple[
        erlab.interactive.imagetool.manager.ImageToolManager | None,
        int | str | None,
    ]:
        manager = self.slicer_area._manager_instance
        if manager is None:
            return None, None
        return manager, manager.target_from_slicer_area(self.slicer_area)

    def _manager_target_for_nest(
        self,
    ) -> tuple[erlab.interactive.imagetool.manager.ImageToolManager, int | str]:
        manager, target = self._manager_target()
        if manager is None or target is None:
            raise RuntimeError(
                "Open Child Window is only available for manager-backed ImageTools"
            )
        return manager, target

    @QtCore.Slot()
    def accept(self) -> None:
        if self.is_provenance_edit_mode:
            super().accept()
            return

        manager = self.batch_manager
        if manager is not None:
            try:
                if not manager.apply_batch_transform_dialog(self, self.launch_mode):
                    return
            except Exception:
                erlab.interactive.utils.MessageDialog.critical(
                    self, "Error", "An error occurred while processing data."
                )
                return
            super().accept()
            return

        input_name = self.slicer_area.data.name
        new_name = ""
        if input_name is not None:
            new_name = str(input_name)

        try:
            if self.apply_on_nonuniform_data:
                input_data = erlab.utils.array._restore_nonuniform_dims(
                    self.slicer_area.data
                )
                self.preflight_data(input_data)
                processed = self.process_data(input_data)
            else:
                self.preflight_data(self.slicer_area.data)
                dimension_mapping = erlab.utils.array._nonuniform_dim_mapping(
                    self.slicer_area.data
                )
                processed = erlab.utils.array._restore_nonuniform_dims(
                    self.process_data(self.slicer_area.data), dimension_mapping
                )
            processed = processed.rename(input_name)

            manager, target = self._manager_target()
            source_spec = self.source_spec(new_name)
            parent_provenance = self.slicer_area.displayed_provenance_spec()
            if manager is not None and target is not None:
                with contextlib.suppress(Exception):
                    parent_provenance = manager._node_for_target(
                        target
                    ).displayed_provenance_spec
            nested_provenance_spec = compose_full_provenance(
                parent_provenance,
                source_spec,
            )
            detached_provenance_spec = self._detached_provenance_spec(
                parent_provenance,
                source_spec,
                new_name,
            )

            if self.launch_mode == "replace":
                if (
                    manager is not None
                    and target is not None
                    and manager._is_imagetool_target(target)
                ):
                    self._rewrite_target_provenance(
                        target,
                        new_name,
                        detached_provenance_spec,
                    )
                else:
                    self._set_current_tool_provenance(detached_provenance_spec)
                self.slicer_area.replace_source_data(processed, emit_edited=True)
            else:
                itool_kw = self._itool_kwargs(processed)
                if self.launch_mode == "nest":
                    manager, target = self._manager_target_for_nest()
                    tool = typing.cast(
                        "QtWidgets.QWidget | None",
                        erlab.interactive.itool(manager=False, **itool_kw),
                    )
                    if tool is not None:  # pragma: no branch
                        typing.cast(
                            "erlab.interactive.imagetool.ImageTool", tool
                        ).set_provenance_spec(nested_provenance_spec)
                        manager.add_imagetool_child(
                            typing.cast("erlab.interactive.imagetool.ImageTool", tool),
                            target,
                            source_spec=source_spec,
                        )
                else:
                    if manager is not None and self.slicer_area._in_manager:
                        tool = typing.cast(
                            "erlab.interactive.imagetool.ImageTool | None",
                            erlab.interactive.itool(manager=False, **itool_kw),
                        )
                        if tool is not None:  # pragma: no branch
                            tool.set_provenance_spec(detached_provenance_spec)
                            replay_source_data = self.slicer_area.data
                            if target is not None:
                                replay_source_data = manager._node_for_target(
                                    target
                                ).resolved_replay_source_data()
                            manager.add_imagetool(
                                tool,
                                activate=True,
                                provenance_spec=detached_provenance_spec,
                                replay_source_data=replay_source_data,
                            )
                    else:
                        tool = typing.cast(
                            "erlab.interactive.imagetool.ImageTool | None",
                            erlab.interactive.itool(**itool_kw),
                        )
                        if tool is not None:
                            tool.set_provenance_spec(detached_provenance_spec)

            del processed

        except Exception as exc:
            if self._handle_process_error(exc):
                return
            erlab.interactive.utils.MessageDialog.critical(
                self, "Error", "An error occurred while processing data."
            )
            return

        super().accept()


class DataFilterDialog(_DataManipulationDialog):
    """Parent class for implementing data changes that only affects the appearance.

    These changes are not destructive and can be undone from the menu bar. Only one kind
    of filter can be applied at a time, and applying a new kind of filter will replace
    the previous one.

    - Override method `setup_widgets` to add widgets to the dialog.

    - Override `filter_operation` to generate operation-backed copy code and provenance
      for the displayed filtered data. Accepted filters must be operation-backed.

    - Override attribute `title` to set the title of the dialog window.

    - Override attribute `enable_copy` to show or hide the copy button.

    - Override attributes `enable_preview` to show or hide the preview button.

    """

    enable_preview: bool = True
    """Whether to show a preview button."""

    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        *,
        batch_manager: ImageToolManager | None = None,
        provenance_edit_mode: bool = False,
        dialog_parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            slicer_area,
            batch_manager=batch_manager,
            provenance_edit_mode=provenance_edit_mode,
            dialog_parent=dialog_parent,
        )
        self._previewed: bool = False
        self._starting_applied_func = self.slicer_area._applied_func
        self._starting_filter_operation = (
            self.slicer_area._accepted_filter_provenance_operation
        )

        if (
            self.enable_preview
            and not self.is_batch_mode
            and not self.is_provenance_edit_mode
        ):
            self.preview_button = QtWidgets.QPushButton("Preview")
            self.preview_button.clicked.connect(self._preview)
            self.buttonBox.addButton(
                self.preview_button, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            )
        self._restore_current_filter_operation()

    def _restore_current_filter_operation(self) -> None:
        operation = self.slicer_area._accepted_filter_provenance_operation
        if operation is not None:
            self.restore_filter_operation(operation)

    def restore_filter_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        """Restore widgets from an active filter operation when supported."""
        del operation

    def _uses_process_only_filter(self) -> bool:
        return (
            type(self).filter_operation is DataFilterDialog.filter_operation
            and type(self).process_data is not _DataManipulationDialog.process_data
        )

    def _apply_current_filter(
        self,
        *,
        update: bool = True,
        emit_edited: bool = False,
        preview: bool = False,
    ) -> None:
        operation = self.filter_operation()
        if operation is not None:
            self.slicer_area.apply_filter_operation(
                operation,
                update=update,
                emit_edited=emit_edited,
                preview=preview,
            )
            return
        if self._uses_process_only_filter():
            raise NotImplementedError(
                "DataFilterDialog subclasses must implement filter_operation() "
                "to apply accepted filters."
            )
        self.slicer_area.apply_filter_operation(
            None,
            update=update,
            emit_edited=emit_edited,
            preview=preview,
        )

    @QtCore.Slot()
    def _preview(self):
        self._previewed = True
        self._apply_current_filter(preview=True)

    @QtCore.Slot()
    def reject(self) -> None:
        if self._previewed:
            if self._starting_filter_operation is not None:
                self.slicer_area.apply_filter_operation(
                    self._starting_filter_operation,
                    preview=True,
                )
            else:
                self.slicer_area.apply_func(
                    self._starting_applied_func,
                    preview=True,
                )
        super().reject()

    @QtCore.Slot()
    def accept(self) -> None:
        if self.is_provenance_edit_mode:
            super().accept()
            return

        manager = self.batch_manager
        if manager is not None:
            try:
                if not manager.apply_batch_filter_dialog(self):
                    return
            except Exception:
                erlab.interactive.utils.MessageDialog.critical(
                    self, "Error", "An error occurred while processing data."
                )
                return
            super().accept()
            return

        try:
            operation = self.filter_operation()
            emit_edited = (
                operation is not None
                or self._starting_filter_operation is not None
                or self._starting_applied_func is not None
                or self.slicer_area.has_active_filter
            )
            self.slicer_area.record_history_mutation(
                None,
                lambda: self._apply_current_filter(emit_edited=emit_edited),
            )
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self, "Error", "An error occurred while processing data."
            )
            return
        super().accept()

    def filter_operation(
        self,
    ) -> ToolProvenanceOperation | None:
        return None

    def filter_operations(
        self,
    ) -> list[ToolProvenanceOperation]:
        operation = self.filter_operation()
        return [] if operation is None else [operation]

    def provenance_edit_operations(
        self,
    ) -> list[ToolProvenanceOperation]:
        return self.filter_operations()

    def make_code(self) -> str:
        try:
            return operations_expression_code(
                self.filter_operations(),
                self._copy_data_name(),
            )
        except Exception:
            return ""


class KspaceConversionDialog(DataTransformDialog):
    title = "Convert to kspace"
    enable_copy = True
    apply_on_nonuniform_data = True
    grouped_operation_only = True
    operation_group_kind = _kspace_conversion.KSPACE_CONVERSION_GROUP_KIND
    copy_output_suffix = "_kconv"
    operation_types = (
        KspaceConfigurationOperation,
        KspaceWorkFunctionOperation,
        KspaceInnerPotentialOperation,
        KspaceSetNormalOperation,
        KspaceConvertOperation,
    )

    _OFFSET_LABELS: typing.ClassVar[dict[str, str]] = {"V0": "V₀", "wf": "𝜙"}
    _OFFSET_UNITS: typing.ClassVar[dict[str, str]] = {"V0": " eV", "wf": " eV"}
    _NORMAL_EMISSION_LABELS: typing.ClassVar[dict[str, str]] = {
        "alpha": "𝛼",
        "beta": "𝛽",
    }

    @classmethod
    def operation_group_for_edit(
        cls,
        operations: Sequence[ToolProvenanceOperation],
        operation_index: int,
    ) -> tuple[int, int] | None:
        return _kspace_conversion.is_kspace_conversion_group(
            operations,
            operation_index,
        )

    def setup_widgets(self) -> None:
        self.setObjectName("kspaceConversionDialog")
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )
        self._compatible = bool(self._source_data.kspace._interactive_compatible)
        self._control_data = self._source_data
        self._normal_delta: float | None = None

        if not self._compatible:
            self.layout_.addRow(
                QtWidgets.QLabel("Momentum conversion is not available for this data.")
            )
            return

        self._source_configuration = int(self._source_data.kspace.configuration)
        self.configuration_combo = QtWidgets.QComboBox()
        for configuration in erlab.constants.AxesConfiguration:
            self.configuration_combo.addItem(
                _kspace_conversion.configuration_text(configuration),
                int(configuration),
            )
        self.configuration_combo.currentIndexChanged.connect(
            self._handle_configuration_changed
        )
        self.layout_.addRow("Configuration", self.configuration_combo)

        self.parameters_group = QtWidgets.QGroupBox("Parameters")
        self.parameters_group.setLayout(QtWidgets.QFormLayout())
        self.layout_.addRow(self.parameters_group)

        self.normal_emission_group = QtWidgets.QGroupBox("Normal Emission")
        self.normal_emission_group.setLayout(QtWidgets.QFormLayout())
        self.layout_.addRow(self.normal_emission_group)

        self.bounds_supergroup = QtWidgets.QGroupBox("Bounds")
        self.bounds_supergroup.setCheckable(True)
        self.bounds_supergroup.setChecked(False)
        self.bounds_supergroup.setLayout(QtWidgets.QFormLayout())
        self.bounds_supergroup.toggled.connect(self._update_memory_estimate)
        self.bounds_group = self.bounds_supergroup
        self.layout_.addRow(self.bounds_supergroup)

        self.resolution_supergroup = QtWidgets.QGroupBox("Resolution")
        self.resolution_supergroup.setCheckable(True)
        self.resolution_supergroup.setChecked(False)
        self.resolution_supergroup.setLayout(QtWidgets.QFormLayout())
        self.resolution_supergroup.toggled.connect(self._update_memory_estimate)
        self.resolution_group = self.resolution_supergroup
        self.layout_.addRow(self.resolution_supergroup)

        self._set_configuration_combo(self._source_configuration)
        self._rebuild_kspace_controls()
        if not self._seed_from_newest_ktool():
            self._seed_from_current_view()

    @staticmethod
    def _clear_layout(layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _set_configuration_combo(self, configuration: int) -> None:
        index = self.configuration_combo.findData(configuration)
        if index < 0:
            raise ValueError(f"Invalid kspace configuration: {configuration!r}")
        with QtCore.QSignalBlocker(self.configuration_combo):
            self.configuration_combo.setCurrentIndex(index)

    @property
    def current_configuration(self) -> erlab.constants.AxesConfiguration:
        value = self.configuration_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
        if value is None:
            return self._control_data.kspace.configuration
        return erlab.constants.AxesConfiguration(int(value))

    def _set_control_configuration(self, configuration: int) -> None:
        self._control_data = self._source_data.kspace.as_configuration(configuration)
        self._set_configuration_combo(configuration)

    @QtCore.Slot(int)
    def _handle_configuration_changed(self, _index: int = -1) -> None:
        if not self._compatible:
            return
        normal = (
            self.normal_emission if hasattr(self, "_normal_emission_spins") else None
        )
        delta = self._normal_delta
        self._control_data = self._source_data.kspace.as_configuration(
            self.current_configuration
        )
        self._rebuild_kspace_controls(
            initial_normal_emission=normal,
            initial_delta=delta,
        )

    def _rebuild_kspace_controls(
        self,
        *,
        initial_normal_emission: tuple[float, float] | None = None,
        initial_delta: float | None = None,
    ) -> None:
        parameters_layout = typing.cast(
            "QtWidgets.QFormLayout", self.parameters_group.layout()
        )
        normal_emission_layout = typing.cast(
            "QtWidgets.QFormLayout", self.normal_emission_group.layout()
        )
        bounds_layout = typing.cast("QtWidgets.QFormLayout", self.bounds_group.layout())
        resolution_layout = typing.cast(
            "QtWidgets.QFormLayout", self.resolution_group.layout()
        )
        self._clear_layout(parameters_layout)
        self._clear_layout(normal_emission_layout)
        self._clear_layout(bounds_layout)
        self._clear_layout(resolution_layout)

        self.bounds_btn = QtWidgets.QPushButton("Calculate")
        self.bounds_btn.clicked.connect(self.calculate_bounds)
        self.res_btn = QtWidgets.QPushButton("Calculate")
        self.res_btn.clicked.connect(self.calculate_resolution)
        self.res_npts_check = QtWidgets.QCheckBox("From number of points")
        self.res_npts_check.toggled.connect(self.calculate_resolution)
        self.res_npts_check.toggled.connect(self._update_memory_estimate)

        self._offset_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        if self._control_data.kspace._has_hv:
            v0_spin = QtWidgets.QDoubleSpinBox()
            v0_spin.setRange(0, 100)
            v0_spin.setSingleStep(1)
            v0_spin.setDecimals(1)
            v0_spin.setSuffix(self._OFFSET_UNITS["V0"])
            v0_spin.setToolTip("Inner potential of the sample.")
            v0_spin.setValue(
                _kspace_conversion.kspace_inner_potential(self._control_data)
            )
            v0_spin.valueChanged.connect(self._update_memory_estimate)
            self._offset_spins["V0"] = v0_spin
            parameters_layout.addRow(self._OFFSET_LABELS["V0"], v0_spin)

        wf_spin = QtWidgets.QDoubleSpinBox()
        wf_spin.setRange(0.0, 9.999)
        wf_spin.setSingleStep(0.01)
        wf_spin.setDecimals(4)
        wf_spin.setSuffix(self._OFFSET_UNITS["wf"])
        wf_spin.setToolTip("Work function of the system.")
        wf_spin.setValue(_kspace_conversion.kspace_work_function(self._control_data))
        wf_spin.valueChanged.connect(self._update_memory_estimate)
        self._offset_spins["wf"] = wf_spin
        parameters_layout.addRow(self._OFFSET_LABELS["wf"], wf_spin)

        self._normal_emission_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for axis, label in self._NORMAL_EMISSION_LABELS.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-360, 360)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setSuffix("°")
            spin.setKeyboardTracking(False)
            spin.setToolTip("Angle corresponding to sample normal emission.")
            spin.valueChanged.connect(self._update_memory_estimate)
            self._normal_emission_spins[axis] = spin
            normal_emission_layout.addRow(label, spin)

        if initial_normal_emission is None:
            initial_normal_emission = self._normal_emission_from_data(
                self._control_data
            )
        if initial_delta is None:
            initial_delta = float(self._control_data.kspace.offsets["delta"])
        self._normal_delta = initial_delta
        self._set_normal_emission_spins(initial_normal_emission)

        self._bound_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._resolution_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for axis in self._control_data.kspace.momentum_axes:
            for index in range(2):
                name = f"{axis}{index}"
                spin = QtWidgets.QDoubleSpinBox()
                if axis == "kz":
                    spin.setRange(0, 100)
                else:
                    spin.setRange(-10, 10)
                spin.setSingleStep(0.01)
                spin.setDecimals(4)
                spin.setSuffix(" Å⁻¹")
                spin.valueChanged.connect(self._update_memory_estimate)
                self._bound_spins[name] = spin
                bounds_layout.addRow(name, spin)

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0001, 10)
            spin.setSingleStep(0.001)
            spin.setDecimals(5)
            spin.setSuffix(" Å⁻¹")
            spin.valueChanged.connect(self._update_memory_estimate)
            self._resolution_spins[axis] = spin
            resolution_layout.addRow(axis, spin)

        bounds_layout.addRow(self.bounds_btn)
        resolution_layout.addRow(self.res_npts_check)
        resolution_layout.addRow(self.res_btn)
        self._memory_estimate_label = QtWidgets.QLabel()
        self._memory_estimate_label.setObjectName("kspaceConversionMemoryEstimate")
        self._memory_estimate_label.setMinimumWidth(0)
        self._memory_estimate_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self._memory_estimate_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self._memory_estimate_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._memory_estimate_label.setWordWrap(True)
        self._memory_estimate_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        resolution_layout.addRow(self._memory_estimate_label)
        self.calculate_bounds()
        self.calculate_resolution()
        self._update_memory_estimate()

    def _normal_emission_from_data(self, data: xr.DataArray) -> tuple[float, float]:
        return _kspace_conversion.normal_emission_angles(data, data.kspace.offsets)

    def _set_normal_emission_spins(self, values: tuple[float, float]) -> None:
        for axis, value in {"alpha": values[0], "beta": values[1]}.items():
            spin = self._normal_emission_spins[axis]
            with QtCore.QSignalBlocker(spin):
                spin.setValue(value)

    def _seed_from_current_view(self) -> None:
        initial_normal_emission, initial_delta = (
            _kspace_conversion.initial_normal_emission_from_slicer_area(
                self.slicer_area
            )
        )
        if initial_normal_emission is not None:
            if initial_delta is None:
                initial_delta = float(self._control_data.kspace.offsets["delta"])
            self._normal_delta = initial_delta
            self._set_normal_emission_spins(initial_normal_emission)
            self.calculate_bounds()
            self.calculate_resolution()

    def _seed_from_newest_ktool(self) -> bool:
        ktool = self._newest_child_ktool()
        if ktool is None:
            return False
        configuration = int(ktool.current_configuration)
        self._set_control_configuration(configuration)
        self._control_data.kspace.angle_scales = ktool.data.kspace.angle_scales
        self._rebuild_kspace_controls(
            initial_normal_emission=ktool._current_normal_emission_angles(),
            initial_delta=ktool.offset_dict["delta"],
        )

        self._offset_spins["wf"].setValue(ktool._work_function)
        if "V0" in self._offset_spins and ktool.data.kspace._has_hv:
            self._offset_spins["V0"].setValue(ktool._inner_potential)

        self.bounds_supergroup.setChecked(ktool.bounds_supergroup.isChecked())
        if ktool.bounds is not None:
            for axis, values in ktool.bounds.items():
                for index, value in enumerate(values):
                    name = f"{axis}{index}"
                    if name in self._bound_spins:
                        self._bound_spins[name].setValue(value)

        self.resolution_supergroup.setChecked(ktool.resolution_supergroup.isChecked())
        if ktool.resolution is not None:
            for axis, value in ktool.resolution.items():
                if axis in self._resolution_spins:
                    self._resolution_spins[axis].setValue(value)
        with QtCore.QSignalBlocker(self.res_npts_check):
            self.res_npts_check.setChecked(ktool.res_npts_check.isChecked())
        return True

    def _newest_child_ktool(self) -> typing.Any | None:
        from erlab.interactive.kspace import KspaceTool

        manager = self.slicer_area._manager_instance
        if manager is not None:
            target = manager.target_from_slicer_area(self.slicer_area)
            if target is not None:
                node = manager._node_for_target(target)
                for uid in reversed(node._childtool_indices):
                    with contextlib.suppress(Exception):
                        child = manager.get_childtool(uid)
                        if isinstance(child, KspaceTool):
                            return child

        for child in reversed(self.slicer_area._associated_tools_list):
            if isinstance(child, KspaceTool):
                return child
        return None

    @property
    def _work_function(self) -> float:
        return _kspace_conversion.rounded_spin_value(self._offset_spins["wf"])

    @property
    def _inner_potential(self) -> float | None:
        if "V0" not in self._offset_spins:
            return None
        return _kspace_conversion.rounded_spin_value(self._offset_spins["V0"])

    @property
    def normal_emission(self) -> tuple[float, float]:
        return (
            _kspace_conversion.rounded_spin_value(self._normal_emission_spins["alpha"]),
            _kspace_conversion.rounded_spin_value(self._normal_emission_spins["beta"]),
        )

    @property
    def bounds(self) -> dict[str, tuple[float, float]] | None:
        if self.bounds_supergroup.isChecked():
            return {
                axis: (
                    float(np.round(self._bound_spins[f"{axis}0"].value(), 5)),
                    float(np.round(self._bound_spins[f"{axis}1"].value(), 5)),
                )
                for axis in self._control_data.kspace.momentum_axes
            }
        return None

    @property
    def resolution(self) -> dict[str, float] | None:
        if self.resolution_supergroup.isChecked():
            return {
                axis: float(np.round(self._resolution_spins[axis].value(), 5))
                for axis in self._control_data.kspace.momentum_axes
            }
        return None

    def _parameterized_data(
        self,
        data: xr.DataArray,
        *,
        source_data: xr.DataArray | None = None,
    ) -> xr.DataArray:
        return _kspace_conversion.apply_kspace_parameters(
            data,
            source_data=self._control_data if source_data is None else source_data,
            work_function=self._work_function,
            inner_potential=self._inner_potential,
            force_work_function=True,
            force_inner_potential=True,
            normal_emission=self.normal_emission,
            delta=self._normal_delta,
            alpha_scale=self._control_data.kspace.alpha_scale,
            beta_scale=self._control_data.kspace.beta_scale,
        )

    def _conversion_input_for_data(self, data: xr.DataArray) -> xr.DataArray:
        source_data = erlab.utils.array._restore_nonuniform_dims(data)
        if int(source_data.kspace.configuration) != int(self.current_configuration):
            source_data = source_data.kspace.as_configuration(
                self.current_configuration
            )
        return self._parameterized_data(
            source_data.copy(deep=False),
            source_data=source_data,
        )

    def conversion_estimate_for_data(
        self,
        data: xr.DataArray,
    ) -> _kspace_conversion.KspaceConversionEstimate:
        return _kspace_conversion.estimate_kspace_conversion(
            self._conversion_input_for_data(data),
            bounds=self.bounds,
            resolution=self.resolution,
        )

    def _set_memory_estimate(
        self,
        estimate: _kspace_conversion.KspaceConversionEstimate,
    ) -> None:
        if not hasattr(self, "_memory_estimate_label"):
            return
        self._memory_estimate_label.setText(
            _kspace_conversion.kspace_conversion_estimate_text(estimate)
        )
        self._memory_estimate_label.updateGeometry()
        self._memory_estimate_label.setProperty(
            "kspaceMemoryUnsafe",
            not estimate.is_safe,
        )
        style = self._memory_estimate_label.style()
        if style is not None:
            style.unpolish(self._memory_estimate_label)
            style.polish(self._memory_estimate_label)

    @QtCore.Slot()
    @QtCore.Slot(int)
    @QtCore.Slot(bool)
    @QtCore.Slot(float)
    def _update_memory_estimate(self, *args: object) -> None:
        del args
        if not self._compatible or not hasattr(self, "_memory_estimate_label"):
            return
        try:
            estimate = self.conversion_estimate_for_data(self._source_data)
        except Exception as exc:
            self._memory_estimate_label.setText(str(exc))
            self._memory_estimate_label.setProperty("kspaceMemoryUnsafe", True)
            return
        self._set_memory_estimate(estimate)

    def preflight_data(self, data: xr.DataArray) -> None:
        estimate = self.conversion_estimate_for_data(data)
        self._set_memory_estimate(estimate)
        if not estimate.is_safe:
            raise _kspace_conversion.KspaceConversionMemoryError(estimate)

    def _handle_process_error(self, exc: BaseException) -> bool:
        if not isinstance(exc, _kspace_conversion.KspaceConversionMemoryError):
            return False
        erlab.interactive.utils.MessageDialog.critical(
            self,
            _kspace_conversion.kspace_conversion_memory_dialog_title(),
            _kspace_conversion.kspace_conversion_memory_dialog_text(),
            informative_text=_kspace_conversion.kspace_conversion_memory_dialog_info(
                exc.estimate
            ),
            detailed_text=_kspace_conversion.kspace_conversion_memory_dialog_details(
                exc.estimate
            ),
            buttons=QtWidgets.QDialogButtonBox.StandardButton.Ok,
        )
        return True

    @QtCore.Slot()
    def calculate_bounds(self) -> None:
        data = self._parameterized_data(self._control_data.copy(deep=False))
        data.kspace._check_kinetic_energy(
            context="estimating momentum bounds in kspace conversion dialog"
        )
        bounds = data.kspace.estimate_bounds()
        for axis in data.kspace.momentum_axes:
            for index, value in enumerate(bounds[axis]):
                spin = self._bound_spins[f"{axis}{index}"]
                with QtCore.QSignalBlocker(spin):
                    spin.setValue(value)
        self._update_memory_estimate()

    @QtCore.Slot()
    def calculate_resolution(self) -> None:
        data = self._parameterized_data(self._control_data.copy(deep=False))
        data.kspace._check_kinetic_energy(
            context="estimating momentum resolution in kspace conversion dialog"
        )
        for axis, spin in self._resolution_spins.items():
            with QtCore.QSignalBlocker(spin):
                spin.setValue(
                    data.kspace.estimate_resolution(
                        axis,
                        from_numpoints=self.res_npts_check.isChecked(),
                    )
                )
        self._update_memory_estimate()

    def _operations_for_data(
        self,
        data: xr.DataArray,
    ) -> tuple[ToolProvenanceOperation, ...]:
        source_data = erlab.utils.array._restore_nonuniform_dims(data)
        return _kspace_conversion.kspace_conversion_operations(
            source_data,
            target_configuration=self.current_configuration,
            source_configuration=source_data.kspace.configuration,
            work_function=self._work_function,
            inner_potential=self._inner_potential,
            normal_emission=self.normal_emission,
            delta=self._normal_delta,
            alpha_scale=self._control_data.kspace.alpha_scale,
            beta_scale=self._control_data.kspace.beta_scale,
            bounds=self.bounds,
            resolution=self.resolution,
            force_scalars=True,
        )

    def source_operations(
        self,
    ) -> list[ToolProvenanceOperation]:
        return list(self._operations_for_data(self.slicer_area.data))

    def source_spec_for_data(
        self,
        data: xr.DataArray,
        new_name: str | None = None,
    ) -> ToolProvenanceSpec:
        del new_name
        return public_data(*self._operations_for_data(data))

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return self.source_spec_for_data(data).apply(data)

    def focus_operation_group_control(self, focus: str | None) -> None:
        match focus:
            case "configuration":
                self.configuration_combo.setFocus(
                    QtCore.Qt.FocusReason.OtherFocusReason
                )
            case "work_function":
                self._offset_spins["wf"].setFocus(
                    QtCore.Qt.FocusReason.OtherFocusReason
                )
                self._offset_spins["wf"].selectAll()
            case "inner_potential":
                spin = self._offset_spins.get("V0")
                if spin is not None:
                    spin.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                    spin.selectAll()
            case "normal_emission":
                self._normal_emission_spins["alpha"].setFocus(
                    QtCore.Qt.FocusReason.OtherFocusReason
                )
                self._normal_emission_spins["alpha"].selectAll()
            case "bounds_resolution":
                self.bounds_supergroup.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def restore_transform_operations(
        self,
        operations: Sequence[ToolProvenanceOperation],
    ) -> None:
        group = _kspace_conversion.is_kspace_conversion_group(operations, 0)
        if group != (0, len(operations)):
            raise ValueError("Expected a complete kspace conversion operation group")

        normal: tuple[float, float] | None = None
        delta: float | None = None
        bounds: dict[str, tuple[float, float]] | None = None
        resolution: dict[str, float] | None = None
        restored_angle_scales = False
        for operation in operations:
            if isinstance(
                operation,
                KspaceConfigurationOperation,
            ):
                self._set_control_configuration(operation.configuration)
                self._rebuild_kspace_controls()
                break
        for operation in operations:
            if isinstance(
                operation,
                KspaceWorkFunctionOperation,
            ):
                self._offset_spins["wf"].setValue(operation.work_function)
            elif isinstance(
                operation,
                KspaceInnerPotentialOperation,
            ):
                if "V0" in self._offset_spins:
                    self._offset_spins["V0"].setValue(operation.inner_potential)
            elif isinstance(
                operation,
                KspaceSetNormalOperation,
            ):
                normal = (operation.alpha, operation.beta)
                delta = operation.delta
                if operation.alpha_scale is not None:
                    self._control_data.kspace.alpha_scale = operation.alpha_scale
                    restored_angle_scales = True
                if operation.beta_scale is not None:
                    self._control_data.kspace.beta_scale = operation.beta_scale
                    restored_angle_scales = True
            elif isinstance(
                operation,
                KspaceConvertOperation,
            ):
                bounds = operation.bounds
                resolution = operation.resolution

        if normal is not None:
            self._normal_delta = delta
            self._set_normal_emission_spins(normal)
        if restored_angle_scales:
            self.calculate_bounds()
            self.calculate_resolution()
        self.bounds_supergroup.setChecked(bounds is not None)
        if bounds is not None:
            for axis, values in bounds.items():
                for index, value in enumerate(values):
                    name = f"{axis}{index}"
                    if name in self._bound_spins:
                        self._bound_spins[name].setValue(value)
        self.resolution_supergroup.setChecked(resolution is not None)
        if resolution is not None:
            for axis, value in resolution.items():
                if axis in self._resolution_spins:
                    self._resolution_spins[axis].setValue(value)

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        self.restore_transform_operations((operation,))

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if self._compatible:
            return QtWidgets.QDialog.DialogCode.Accepted
        QtWidgets.QMessageBox.warning(
            self,
            "Momentum Conversion Unavailable",
            "Momentum conversion is not available for this data.",
        )
        return QtWidgets.QDialog.DialogCode.Rejected


class RotationDialog(DataTransformDialog):
    enable_copy = True
    operation_types = (RotateOperation,)

    @property
    def _rotate_params(self) -> dict[str, typing.Any]:
        return {
            "angle": float(
                np.round(self.angle_spin.value(), self.angle_spin.decimals())
            ),
            "axes": typing.cast(
                "tuple[str, str]", tuple(self.slicer_area.main_image.axis_dims_uniform)
            ),
            "center": typing.cast(
                "tuple[float, float]", tuple(spin.value() for spin in self.center_spins)
            ),
            "reshape": self.reshape_check.isChecked(),
            "order": self.order_spin.value(),
        }

    def setup_widgets(self) -> None:
        main_image = self.slicer_area.main_image
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(-360, 360)
        self.angle_spin.setSingleStep(1)
        self.angle_spin.setValue(0)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setSuffix("°")
        self.layout_.addRow("Angle", self.angle_spin)

        self.center_spins = (
            pg.SpinBox(dec=True, compactHeight=False),
            pg.SpinBox(dec=True, compactHeight=False),
        )
        for i in range(2):
            axis: int = main_image.display_axis[i]
            dim: str = str(main_image.axis_dims_uniform[i])
            self.center_spins[i].setDecimals(
                self.array_slicer.get_significant(axis, uniform=True)
            )
            self.center_spins[i].setSingleStep(float(self.array_slicer.incs_uniform[i]))
            self.center_spins[i].setValue(0.0)

            self.layout_.addRow(f"Center {dim}", self.center_spins[i])

        self.order_spin: QtWidgets.QSpinBox = QtWidgets.QSpinBox()
        self.order_spin.setRange(0, 5)
        self.order_spin.setValue(1)
        self.layout_.addRow("Spline Order", self.order_spin)

        self.reshape_check: QtWidgets.QCheckBox = QtWidgets.QCheckBox("Reshape")
        self.reshape_check.setChecked(True)
        self.layout_.addRow(self.reshape_check)

        if main_image.is_guidelines_visible:
            # Fill values from guideline
            self.angle_spin.setValue(-main_image._guideline_angle)
            for spin, val in zip(
                self.center_spins, main_image._guideline_offset, strict=True
            ):
                spin.setValue(val)

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        return RotateOperation(**self._rotate_params)

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            RotateOperation,
        ):
            return
        if tuple(operation.axes) != tuple(
            self.slicer_area.main_image.axis_dims_uniform
        ):
            raise ValueError("Rotation axes are not currently visible")
        self.angle_spin.setValue(float(operation.angle))
        for spin, value in zip(self.center_spins, operation.center, strict=True):
            spin.setValue(float(value))
        self.reshape_check.setChecked(operation.reshape)
        self.order_spin.setValue(int(operation.order))


class AggregateDialog(DataTransformDialog):
    title = "Aggregate Over Dimensions"
    enable_copy = True
    operation_types = (
        AverageOperation,
        QSelAggregationOperation,
    )

    def setup_widgets(self) -> None:
        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QVBoxLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}

        for d in self.slicer_area.data.dims:
            self.dim_checks[d] = QtWidgets.QCheckBox(str(d))
            dim_layout.addWidget(self.dim_checks[d])

        self.layout_.addRow(dim_group)

        self.reducer_combo = QtWidgets.QComboBox()
        _populate_reducer_combo(self.reducer_combo)
        self.layout_.addRow("Reducer", self.reducer_combo)

    @property
    def _target_dims(self) -> tuple[Hashable, ...]:
        return tuple(k for k, v in self.dim_checks.items() if v.isChecked())

    @property
    def _reducer(self) -> typing.Literal["mean", "min", "max", "sum"]:
        return _current_reducer(self.reducer_combo)

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        if not self._target_dims:
            raise ValueError("No dimensions selected")
        return QSelAggregationOperation(
            dims=self._target_dims,
            func=self._reducer,
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if isinstance(
            operation,
            AverageOperation,
        ):
            dims = operation.dims
            func = "mean"
        elif isinstance(
            operation,
            QSelAggregationOperation,
        ):
            dims = operation.dims
            func = operation.func
        else:
            return
        for check in self.dim_checks.values():
            check.setChecked(False)
        for dim in dims:
            if dim not in self.dim_checks:
                raise ValueError(f"Dimension {dim!r} is not available")
            self.dim_checks[dim].setChecked(True)
        if not _set_combo_data(self.reducer_combo, func):
            raise ValueError(f"Reducer {func!r} is not available")

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._target_dims:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension.",
            )
            return

        if (self.slicer_area.data.ndim - len(self._target_dims)) < 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Not Enough Dimensions Left",
                "Data must have at least 1 dimension after aggregation to be "
                "displayed.",
            )
            return

        super().accept()


AverageDialog = AggregateDialog


class _SelectionRow:
    def __init__(
        self,
        dialog: SelectionDialog,
        axis: int,
        dim: Hashable,
        row: int,
        *,
        active: bool,
    ) -> None:
        self.dialog = dialog
        self.axis = axis
        self.dim = dim

        data = dialog.public_data
        current_index = dialog._current_selection_index(axis)
        bin_values = dialog._selection_bin_values()
        binned = dialog._selection_binned()
        scalar_controls = _ScalarSelectionControls(
            data,
            dim,
            axis,
            object_name_prefix="selection",
            current_index=current_index,
            bin_value=bin_values[axis],
            is_binned=binned[axis],
        )
        self._scalar_controls = scalar_controls
        self._coord_ascending = scalar_controls.coord_ascending
        self._dim_size = data.sizes[dim]
        stop_index = min(current_index + 1, data.sizes[dim] - 1)
        stop_value = float(scalar_controls.coord[stop_index])

        self.use_check = QtWidgets.QCheckBox(str(dim))
        self.use_check.setObjectName(f"selection_use_{axis}")
        self.use_check.setChecked(active)

        self.method_combo = scalar_controls.method_combo

        self.kind_combo = QtWidgets.QComboBox()
        self.kind_combo.setObjectName(f"selection_kind_{axis}")
        self.kind_combo.addItem("Point", "point")
        self.kind_combo.addItem("Range", "range")

        self.index_start_spin = scalar_controls.index_spin
        self.index_stop_spin = erlab.interactive.utils.BetterSpinBox(
            integer=True,
            compact=False,
            minimum=-self._dim_size if self._dim_size > 0 else 0,
            maximum=data.sizes[dim],
            value=min(current_index + 1, data.sizes[dim]),
        )

        self.value_start_spin = scalar_controls.value_spin
        self.value_stop_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            decimals=6,
            exact_float=True,
            significant=True,
            minimum=float(np.nanmin(scalar_controls.coord)),
            maximum=float(np.nanmax(scalar_controls.coord)),
            value=stop_value,
        )

        self.start_stack = scalar_controls.stack

        self.start_widget = QtWidgets.QWidget()
        start_layout = QtWidgets.QHBoxLayout(self.start_widget)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(3)
        self.start_none_check = QtWidgets.QCheckBox("None")
        self.start_none_check.setObjectName(f"selection_start_none_{axis}")
        self.start_none_check.setToolTip(
            "Leave the range start open, equivalent to slice(None, stop)."
        )
        start_layout.addWidget(self.start_none_check)
        start_layout.addWidget(self.start_stack)

        self.stop_stack = QtWidgets.QStackedWidget()
        self.stop_stack.addWidget(self.value_stop_spin)
        self.stop_stack.addWidget(self.index_stop_spin)

        self.stop_widget = QtWidgets.QWidget()
        stop_layout = QtWidgets.QHBoxLayout(self.stop_widget)
        stop_layout.setContentsMargins(0, 0, 0, 0)
        stop_layout.setSpacing(3)
        self.stop_none_check = QtWidgets.QCheckBox("None")
        self.stop_none_check.setObjectName(f"selection_stop_none_{axis}")
        self.stop_none_check.setToolTip(
            "Leave the range stop open, equivalent to slice(start, None)."
        )
        stop_layout.addWidget(self.stop_none_check)
        stop_layout.addWidget(self.stop_stack)

        self.step_widget = QtWidgets.QWidget()
        self.step_widget.setToolTip(
            "For range selections, include every Nth item in the selected range."
        )
        step_layout = QtWidgets.QHBoxLayout(self.step_widget)
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_layout.setSpacing(3)
        self.step_check = QtWidgets.QCheckBox()
        self.step_check.setObjectName(f"selection_step_enabled_{axis}")
        self.step_check.setToolTip(self.step_widget.toolTip())
        self.step_spin = erlab.interactive.utils.BetterSpinBox(
            integer=True,
            compact=False,
            minimum=1,
            maximum=max(1, data.sizes[dim]),
            value=1,
        )
        self.step_spin.setObjectName(f"selection_step_{axis}")
        self.step_spin.setToolTip(self.step_widget.toolTip())
        step_layout.addWidget(self.step_check)
        step_layout.addWidget(self.step_spin)

        self.width_widget = scalar_controls.width_widget
        self.width_check = scalar_controls.width_check
        self.width_spin = scalar_controls.width_spin

        widgets: tuple[QtWidgets.QWidget, ...] = (
            self.use_check,
            self.method_combo,
            self.kind_combo,
            self.start_widget,
            self.stop_widget,
            self.step_widget,
            self.width_widget,
        )
        for column, widget in enumerate(widgets):
            dialog.grid_layout.addWidget(widget, row, column)

        for widget in (
            self.use_check,
            self.method_combo,
            self.kind_combo,
            self.index_start_spin,
            self.index_stop_spin,
            self.value_start_spin,
            self.value_stop_spin,
            self.start_none_check,
            self.stop_none_check,
            self.step_check,
            self.step_spin,
            self.width_check,
            self.width_spin,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(dialog.update_preview)
            elif isinstance(widget, QtWidgets.QAbstractButton):
                widget.toggled.connect(dialog.update_preview)
            else:
                widget.valueChanged.connect(dialog.update_preview)

        self.method_combo.currentIndexChanged.connect(self.sync_widgets)
        self.kind_combo.currentIndexChanged.connect(self.sync_widgets)
        self.start_none_check.toggled.connect(self.sync_widgets)
        self.stop_none_check.toggled.connect(self.sync_widgets)
        self.step_check.toggled.connect(self.sync_widgets)
        self.sync_widgets()

    @property
    def method(self) -> typing.Literal["qsel", "sel", "isel"]:
        return typing.cast(
            "typing.Literal['qsel', 'sel', 'isel']",
            self.method_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    @property
    def kind(self) -> typing.Literal["point", "range"]:
        return typing.cast(
            "typing.Literal['point', 'range']",
            self.kind_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    def sync_widgets(self) -> None:
        is_index = self.method == "isel"
        is_range = self.kind == "range"
        is_qsel = self.method == "qsel"
        start_is_open = is_range and self.start_none_check.isChecked()
        stop_is_open = is_range and self.stop_none_check.isChecked()

        self.start_stack.setCurrentIndex(1 if is_index else 0)
        self.stop_stack.setCurrentIndex(1 if is_index else 0)
        self.start_none_check.setVisible(is_range)
        self.stop_none_check.setVisible(is_range)
        self.start_none_check.setEnabled(is_range)
        self.stop_none_check.setEnabled(is_range)
        self.start_stack.setEnabled(not start_is_open)
        self.stop_stack.setEnabled(is_range and not stop_is_open)
        self.step_widget.setEnabled(is_range)
        self.step_spin.setEnabled(is_range and self.step_check.isChecked())
        self.width_widget.setEnabled(is_qsel and not is_range)
        self.width_spin.setEnabled(
            is_qsel and not is_range and self.width_check.isChecked()
        )

    def _step_value(self) -> int | None:
        if self.kind != "range" or not self.step_check.isChecked():
            return None
        return int(self.step_spin.value())

    def _isel_slice(self, start: int, stop: int) -> slice:
        def _normalized_stop(value: int) -> int:
            if value < 0:
                return max(value + self._dim_size, 0)
            return min(value, self._dim_size)

        start_position = _normalized_stop(start)
        stop_position = _normalized_stop(stop)
        if start_position <= stop_position:
            return slice(start, stop, self._step_value())
        return slice(stop, start, self._step_value())

    def indexer(self) -> tuple[Hashable, typing.Any]:
        if self.method == "isel":
            start = int(self.index_start_spin.value())
            if self.kind == "point":
                return self.dim, start
            range_start = None if self.start_none_check.isChecked() else start
            range_stop = (
                None
                if self.stop_none_check.isChecked()
                else int(self.index_stop_spin.value())
            )
            if range_start is None or range_stop is None:
                return self.dim, slice(range_start, range_stop, self._step_value())
            return self.dim, self._isel_slice(range_start, range_stop)

        start = float(self.value_start_spin.value())
        if self.kind == "point":
            return self.dim, start

        range_start = None if self.start_none_check.isChecked() else start
        range_stop = (
            None
            if self.stop_none_check.isChecked()
            else float(self.value_stop_spin.value())
        )
        if range_start is None or range_stop is None:
            return self.dim, slice(range_start, range_stop, self._step_value())
        if self._coord_ascending:
            return self.dim, slice(
                min(range_start, range_stop),
                max(range_start, range_stop),
                self._step_value(),
            )
        return self.dim, slice(
            max(range_start, range_stop),
            min(range_start, range_stop),
            self._step_value(),
        )

    def qsel_width_indexer(self) -> tuple[str, float] | None:
        if (
            self.method != "qsel"
            or self.kind != "point"
            or not self.width_check.isChecked()
        ):
            return None
        return self._scalar_controls.qsel_width_indexer()

    def restore_indexer(
        self,
        *,
        method: typing.Literal["qsel", "sel", "isel"],
        indexer: typing.Any,
        width: float | None = None,
    ) -> None:
        self.use_check.setChecked(True)
        if not _set_combo_data(self.method_combo, method):
            raise ValueError(f"Selection method {method!r} is not available")
        if isinstance(indexer, slice):
            if indexer.step is None:
                self.step_check.setChecked(False)
            else:
                step = self._validate_restored_step(indexer.step)
                self.step_check.setChecked(True)
                self.step_spin.setValue(step)
            _set_combo_data(self.kind_combo, "range")
            self.start_none_check.setChecked(indexer.start is None)
            self.stop_none_check.setChecked(indexer.stop is None)
            if method == "isel":
                if indexer.start is not None:
                    self.index_start_spin.setValue(int(indexer.start))
                if indexer.stop is not None:
                    self.index_stop_spin.setValue(int(indexer.stop))
            else:
                if indexer.start is not None:
                    self.value_start_spin.setValue(float(indexer.start))
                if indexer.stop is not None:
                    self.value_stop_spin.setValue(float(indexer.stop))
        else:
            _set_combo_data(self.kind_combo, "point")
            self.step_check.setChecked(False)
            self.start_none_check.setChecked(False)
            self.stop_none_check.setChecked(False)
            if method == "isel":
                self.index_start_spin.setValue(int(indexer))
            else:
                self.value_start_spin.setValue(float(indexer))

        if method == "qsel" and width is not None:
            self.width_check.setChecked(True)
            self.width_spin.setValue(float(width))
        else:
            self.width_check.setChecked(False)
        self.sync_widgets()

    def _validate_restored_step(self, step: typing.Any) -> int:
        try:
            step_value = operator.index(step)
        except TypeError as exc:
            raise ValueError("Selection slice steps must be integer strides") from exc
        if step_value <= 0:
            raise ValueError(
                "Reverse or zero-step selections cannot be edited in this dialog"
            )
        return step_value


class SelectionDialog(DataTransformDialog):
    title = "Select Data"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (
        IselOperation,
        SelOperation,
        QSelOperation,
    )

    def __init__(
        self,
        slicer_area: ImageSlicerArea | None = None,
        *,
        batch_manager: ImageToolManager | None = None,
        provenance_edit_mode: bool = False,
        dialog_parent: QtWidgets.QWidget | None = None,
        source_data: xr.DataArray | None = None,
    ) -> None:
        if slicer_area is None:
            if source_data is None:
                raise ValueError(
                    "SelectionDialog requires a slicer area or source data"
                )
            if not provenance_edit_mode:
                raise ValueError(
                    "Source-data-only SelectionDialog requires provenance edit mode"
                )
        self._source_data = source_data
        super().__init__(
            slicer_area,
            batch_manager=batch_manager,
            provenance_edit_mode=provenance_edit_mode,
            dialog_parent=dialog_parent,
        )

    def setup_widgets(self) -> None:
        source_data = self._source_data
        if source_data is None:
            source_data = self.slicer_area.data
        self.public_data = erlab.utils.array._restore_nonuniform_dims(source_data)

        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(3)

        for column, label in enumerate(
            (
                "Use",
                "Method",
                "Selection",
                "Start",
                "Stop",
                "Step",
                "Width",
            )
        ):
            header = QtWidgets.QLabel(label)
            header.setObjectName(f"selection_header_{column}")
            self.grid_layout.addWidget(header, 0, column)

        self.rows: list[_SelectionRow] = []
        default_axis = self.public_data.ndim - 1 if self.public_data.ndim == 4 else None
        for axis, dim in enumerate(self.public_data.dims):
            self.rows.append(
                _SelectionRow(
                    self,
                    axis,
                    dim,
                    axis + 1,
                    active=axis == default_axis,
                )
            )

        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setObjectName("selection_result_preview")
        self.code_preview = QtWidgets.QPlainTextEdit()
        self.code_preview.setObjectName("selection_code_preview")
        self.code_preview.setReadOnly(True)
        self.code_preview.setMaximumBlockCount(4)
        self.code_preview.setMaximumHeight(
            4 * QtGui.QFontMetrics(self.code_preview.font()).height()
        )

        self.layout_.addRow(self.grid_layout)
        self.layout_.addRow("Result", self.preview_label)
        self.layout_.addRow("Code", self.code_preview)
        self.update_preview()

    def _current_selection_index(self, axis: int) -> int:
        if self._source_data is None:
            return self.array_slicer.get_index(self.slicer_area.current_cursor, axis)
        dim = self.public_data.dims[axis]
        size = self.public_data.sizes[dim]
        return max(0, min(size // 2, size - 1))

    def _selection_bin_values(self) -> tuple[float, ...]:
        if self._source_data is None:
            return self.array_slicer.get_bin_values(self.slicer_area.current_cursor)
        return (0.0,) * self.public_data.ndim

    def _selection_binned(self) -> tuple[bool, ...]:
        if self._source_data is None:
            return self.array_slicer.get_binned(self.slicer_area.current_cursor)
        return (False,) * self.public_data.ndim

    def _selection_kwargs(
        self,
    ) -> tuple[
        dict[Hashable, typing.Any],
        dict[Hashable, typing.Any],
        dict[Hashable, typing.Any],
    ]:
        isel_kwargs: dict[Hashable, typing.Any] = {}
        sel_kwargs: dict[Hashable, typing.Any] = {}
        qsel_kwargs: dict[Hashable, typing.Any] = {}

        target: dict[str, dict[Hashable, typing.Any]] = {
            "isel": isel_kwargs,
            "sel": sel_kwargs,
            "qsel": qsel_kwargs,
        }
        for row in self.rows:
            if not row.use_check.isChecked():
                continue
            dim, indexer = row.indexer()
            target[row.method][dim] = indexer
            width_indexer = row.qsel_width_indexer()
            if width_indexer is not None:
                width_dim, width = width_indexer
                qsel_kwargs[width_dim] = width

        return isel_kwargs, sel_kwargs, qsel_kwargs

    def source_operations(
        self,
    ) -> list[ToolProvenanceOperation]:
        isel_kwargs, sel_kwargs, qsel_kwargs = self._selection_kwargs()
        operations: list[ToolProvenanceOperation] = []
        if isel_kwargs:
            operations.append(IselOperation(kwargs=isel_kwargs))
        if sel_kwargs:
            operations.append(SelOperation(kwargs=sel_kwargs))
        if qsel_kwargs:
            operations.append(QSelOperation(kwargs=qsel_kwargs))
        return operations

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        isel_kwargs, sel_kwargs, qsel_kwargs = self._selection_kwargs()
        if isel_kwargs:
            data = data.isel(isel_kwargs)
        if sel_kwargs:
            data = data.sel(sel_kwargs)
        if qsel_kwargs:
            data = data.qsel(qsel_kwargs)
        return data

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if isinstance(operation, IselOperation):
            self._restore_selection_operation("isel", operation.decoded_kwargs)
        elif isinstance(operation, SelOperation):
            self._restore_selection_operation("sel", operation.decoded_kwargs)
        elif isinstance(operation, QSelOperation):
            self._restore_selection_operation("qsel", operation.decoded_kwargs)

    def _restore_selection_operation(
        self,
        method: typing.Literal["qsel", "sel", "isel"],
        kwargs: dict[Hashable, typing.Any],
    ) -> None:
        for row in self.rows:
            row.use_check.setChecked(False)
            row.start_none_check.setChecked(False)
            row.stop_none_check.setChecked(False)
            row.step_check.setChecked(False)
            row.width_check.setChecked(False)

        row_by_dim = {row.dim: row for row in self.rows}
        for dim, indexer in kwargs.items():
            if method == "qsel" and isinstance(dim, str) and dim.endswith("_width"):
                continue
            if dim not in row_by_dim:
                raise ValueError(f"Dimension {dim!r} is not available")
            width = kwargs.get(f"{dim}_width") if method == "qsel" else None
            row_by_dim[dim].restore_indexer(
                method=method,
                indexer=indexer,
                width=None if width is None else float(width),
            )
        self.update_preview()

    @QtCore.Slot()
    @QtCore.Slot(int)
    @QtCore.Slot(bool)
    @QtCore.Slot(object)
    def update_preview(self, *args) -> None:
        try:
            selected = self.process_data(self.public_data)
        except Exception as exc:
            self.preview_label.setText(f"Invalid selection: {exc}")
            self.code_preview.setPlainText(self.make_code())
            ok_button = self.buttonBox.button(
                QtWidgets.QDialogButtonBox.StandardButton.Ok
            )
            if ok_button is not None:
                ok_button.setEnabled(False)
            return

        shape = " × ".join(str(size) for size in selected.shape)
        dims = ", ".join(str(dim) for dim in selected.dims)
        self.preview_label.setText(f"{selected.ndim}D ({shape}) [{dims}]")
        self.code_preview.setPlainText(self.make_code())
        ok_button = self.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(
                bool(self.source_operations())
                and 2 <= selected.ndim <= 4
                and selected.size > 0
            )

    @QtCore.Slot()
    def accept(self) -> None:
        if not self.source_operations():
            QtWidgets.QMessageBox.warning(
                self,
                "No Selection",
                "Select at least one dimension before applying.",
            )
            return

        try:
            selected = self.process_data(self.public_data)
        except Exception:
            erlab.interactive.utils.MessageDialog.critical(
                self, "Error", "An error occurred while selecting data."
            )
            return

        if not (2 <= selected.ndim <= 4) or selected.size == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Selection",
                "Selected data must have 2 to 4 dimensions and contain at least "
                "one value.",
            )
            return

        super().accept()


class InterpolationDialog(DataTransformDialog):
    title = "Interpolate"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (InterpolationOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )

        self.dim_combo = QtWidgets.QComboBox()
        for dim in self._source_data.dims:
            self.dim_combo.addItem(str(dim), userData=dim)
        self.dim_combo.currentIndexChanged.connect(self._dimension_changed)
        self.layout_.addRow("Dimension", self.dim_combo)

        self.coord_widget = CoordinateGridWidget(
            np.array([0.0, 1.0]),
            editable_count=True,
            preserve_shape=False,
            require_complete=True,
            numeric_reference=True,
            disable_singleton_controls=False,
            reset_table_to_reference=False,
            update_table_on_mode_changed=True,
        )
        self.layout_.addRow("Target Coordinates", self.coord_widget)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["linear", "nearest"])
        self.layout_.addRow("Method", self.method_combo)

        self._dimension_changed()

    @property
    def _selected_dim(self) -> Hashable | None:
        if self.dim_combo.currentIndex() < 0:
            return None
        return typing.cast(
            "Hashable",
            self.dim_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    def _source_coord_values(self, dim: Hashable) -> npt.NDArray:
        if dim not in self._source_data.coords:
            return np.arange(self._source_data.sizes[dim], dtype=np.float64)
        return np.asarray(self._source_data.coords[dim].values)

    def _source_coord_error(self, dim: Hashable) -> str | None:
        coord = self._source_coord_values(dim)
        if coord.ndim != 1:
            return "The selected dimension coordinate must be one-dimensional."
        if not np.issubdtype(coord.dtype, np.number) or np.issubdtype(
            coord.dtype, np.complexfloating
        ):
            return "The selected dimension coordinate must contain real numbers."
        numeric = coord.astype(np.float64, copy=False)
        if not np.all(np.isfinite(numeric)):
            return "The selected dimension coordinate must be finite."
        if np.unique(numeric).size != numeric.size:
            return "The selected dimension coordinate values must be unique."
        return None

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _dimension_changed(self, _index: int | None = None) -> None:
        dim = self._selected_dim
        if dim is None:
            return
        coord = self._source_coord_values(dim)
        if (
            coord.ndim != 1
            or not np.issubdtype(coord.dtype, np.number)
            or np.issubdtype(coord.dtype, np.complexfloating)
        ):
            coord = np.arange(self._source_data.sizes[dim], dtype=np.float64)
        self.coord_widget.set_reference_coord(coord)

    def _target_values(self) -> npt.NDArray:
        values = np.asarray(self.coord_widget.new_coord, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Target coordinates must be one-dimensional.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Target coordinates must be finite.")
        return values

    def source_transform_operation(
        self,
    ) -> InterpolationOperation:
        dim = self._selected_dim
        if dim is None:
            raise ValueError("No dimension selected")
        source_error = self._source_coord_error(dim)
        if source_error is not None:
            raise ValueError(source_error)
        return InterpolationOperation(
            dim=dim,
            values=self._target_values(),
            method=typing.cast(
                "typing.Literal['linear', 'nearest']",
                self.method_combo.currentText(),
            ),
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(operation, InterpolationOperation):
            return
        if not _set_combo_data(self.dim_combo, operation.dim):
            raise ValueError(f"Dimension {operation.dim!r} is not available")
        values = operation.decoded_values
        self.coord_widget.count_spin.setValue(int(values.size))
        self.coord_widget._set_table_values(values)
        if not _set_combo_text(self.method_combo, operation.method):
            raise ValueError(
                f"Interpolation method {operation.method!r} is not available"
            )

    @QtCore.Slot()
    def accept(self) -> None:
        dim = self._selected_dim
        if dim is None:
            QtWidgets.QMessageBox.warning(
                self, "No Dimension Selected", "Choose a dimension to interpolate."
            )
            return

        source_error = self._source_coord_error(dim)
        if source_error is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Source Coordinate",
                source_error,
            )
            return

        try:
            self._target_values()
        except Exception as exc:
            _show_warning_with_traceback(
                self,
                "Invalid Target Coordinates",
                str(exc),
            )
            return

        super().accept()


class _UniformInterpolationDialog(DataTransformDialog):
    """Parameter editor for interpolation over current coordinate bounds."""

    title = "Interpolate Uniform Grid"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (UniformInterpolationOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )
        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        self.size_spins: dict[Hashable, QtWidgets.QSpinBox] = {}

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout(dim_group)
        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Point Count"), 0, 1)
        for row, dim in enumerate(self._source_data.dims, start=1):
            check = QtWidgets.QCheckBox(str(dim))
            size_spin = QtWidgets.QSpinBox()
            size_spin.setRange(1, 10_000_000)
            size_spin.setValue(self._source_data.sizes[dim])
            size_spin.setEnabled(False)
            check.toggled.connect(size_spin.setEnabled)
            self.dim_checks[dim] = check
            self.size_spins[dim] = size_spin
            dim_layout.addWidget(check, row, 0)
            dim_layout.addWidget(size_spin, row, 1)
        self.layout_.addRow(dim_group)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["linear", "nearest"])
        self.layout_.addRow("Method", self.method_combo)

    @property
    def _sizes(self) -> dict[Hashable, int]:
        return {
            dim: self.size_spins[dim].value()
            for dim, check in self.dim_checks.items()
            if check.isChecked()
        }

    def source_transform_operation(self) -> UniformInterpolationOperation:
        if not (sizes := self._sizes):
            raise ValueError("No dimensions selected")
        return UniformInterpolationOperation(
            sizes=sizes,
            method=typing.cast(
                "typing.Literal['linear', 'nearest']",
                self.method_combo.currentText(),
            ),
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(operation, UniformInterpolationOperation):
            return
        for check in self.dim_checks.values():
            check.setChecked(False)
        for dim, size in operation.sizes.items():
            if dim not in self.dim_checks:
                raise ValueError(f"Dimension {dim!r} is not available")
            self.dim_checks[dim].setChecked(True)
            _set_spin_value(
                self.size_spins[dim],
                size,
                label=f"Interpolation size for {dim!r}",
            )
        if not _set_combo_text(self.method_combo, operation.method):
            raise ValueError(
                f"Interpolation method {operation.method!r} is not available"
            )


class _ImageDerivativeDialog(DataTransformDialog):
    """Parameter editor for image-derivative provenance."""

    title = "Image Derivative"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (ImageDerivativeOperation,)
    _METHODS: typing.ClassVar[tuple[tuple[str, str], ...]] = (
        ("diffn", "Second Derivative"),
        ("scaled_laplace", "Scaled Laplace"),
        ("curvature1d", "1D Curvature"),
        ("curvature", "2D Curvature"),
        ("minimum_gradient", "Minimum Gradient"),
    )

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )
        self.method_combo = QtWidgets.QComboBox()
        for method, label in self._METHODS:
            self.method_combo.addItem(label, userData=method)
        self.method_combo.currentIndexChanged.connect(self._sync_method_controls)
        self.layout_.addRow("Method", self.method_combo)

        self.dimension_label = QtWidgets.QLabel("Dimension")
        self.dimension_combo = QtWidgets.QComboBox()
        for dim in self._source_data.dims:
            self.dimension_combo.addItem(str(dim), userData=dim)
        self.layout_.addRow(self.dimension_label, self.dimension_combo)

        self.order_label = QtWidgets.QLabel("Order")
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 9)
        self.order_spin.setValue(2)
        self.layout_.addRow(self.order_label, self.order_spin)

        self.a0_label = QtWidgets.QLabel("a₀")
        self.a0_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            exact_float=True,
        )
        self.a0_spin.setRange(1e-12, 1e12)
        self.a0_spin.setDecimals(8)
        self.a0_spin.setValue(1.0)
        self.layout_.addRow(self.a0_label, self.a0_spin)

        self.factor_label = QtWidgets.QLabel("Factor")
        self.factor_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            exact_float=True,
        )
        self.factor_spin.setRange(-1e12, 1e12)
        self.factor_spin.setDecimals(8)
        self.factor_spin.setValue(1.0)
        self.layout_.addRow(self.factor_label, self.factor_spin)
        self._sync_method_controls()

    @property
    def _method(self) -> str:
        return str(self.method_combo.currentData(QtCore.Qt.ItemDataRole.UserRole))

    @property
    def _dimension(self) -> Hashable:
        if self.dimension_combo.currentIndex() < 0:
            raise ValueError("Derivative input has no dimensions")
        return typing.cast(
            "Hashable",
            self.dimension_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _sync_method_controls(self, _index: int | None = None) -> None:
        method = self._method
        uses_dimension = method in {"diffn", "curvature1d"}
        uses_order = method == "diffn"
        uses_a0 = method in {"curvature1d", "curvature"}
        uses_factor = method in {"scaled_laplace", "curvature"}
        self.dimension_label.setVisible(uses_dimension)
        self.dimension_combo.setVisible(uses_dimension)
        self.order_label.setVisible(uses_order)
        self.order_spin.setVisible(uses_order)
        self.a0_label.setVisible(uses_a0)
        self.a0_spin.setVisible(uses_a0)
        self.factor_label.setVisible(uses_factor)
        self.factor_spin.setVisible(uses_factor)

    def source_transform_operation(self) -> ImageDerivativeOperation:
        method = self._method
        kwargs: dict[str, typing.Any]
        if method == "diffn":
            kwargs = {"coord": self._dimension, "order": self.order_spin.value()}
        elif method == "scaled_laplace":
            kwargs = {"factor": self.factor_spin.value()}
        elif method == "curvature1d":
            kwargs = {"along": self._dimension, "a0": self.a0_spin.value()}
        elif method == "curvature":
            kwargs = {
                "a0": self.a0_spin.value(),
                "factor": self.factor_spin.value(),
            }
        else:
            kwargs = {}
        return ImageDerivativeOperation(
            method=typing.cast(
                "typing.Literal['diffn', 'scaled_laplace', 'curvature1d', "
                "'curvature', 'minimum_gradient']",
                method,
            ),
            kwargs=typing.cast("dict[Hashable, typing.Any]", kwargs),
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(operation, ImageDerivativeOperation):
            return
        if not _set_combo_data(self.method_combo, operation.method):
            raise ValueError(f"Derivative method {operation.method!r} is not available")
        kwargs = typing.cast("dict[str, typing.Any]", dict(operation.kwargs))
        dimension = kwargs.get("coord", kwargs.get("along"))
        if dimension is not None and not _set_combo_data(
            self.dimension_combo, dimension
        ):
            raise ValueError(f"Dimension {dimension!r} is not available")
        if "order" in kwargs:
            _set_spin_value(
                self.order_spin,
                int(kwargs["order"]),
                label="Derivative order",
            )
        if "a0" in kwargs:
            _set_spin_value(
                self.a0_spin,
                float(kwargs["a0"]),
                label="Derivative a0",
            )
        if "factor" in kwargs:
            _set_spin_value(
                self.factor_spin,
                float(kwargs["factor"]),
                label="Derivative factor",
            )
        self._sync_method_controls()


class _RemoveMeshDialog(DataTransformDialog):
    """Parameter editor for mesh-removal provenance."""

    title = "Remove Mesh"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (RemoveMeshOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )

        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.addItem("Corrected Data", userData="corrected")
        self.output_combo.addItem("Extracted Mesh", userData="mesh")
        self.layout_.addRow("Output", self.output_combo)

        self.peak_spins: list[tuple[QtWidgets.QSpinBox, QtWidgets.QSpinBox]] = []
        peak_group = QtWidgets.QGroupBox("First-order Peaks")
        peak_layout = QtWidgets.QGridLayout(peak_group)
        peak_layout.addWidget(QtWidgets.QLabel("Point"), 0, 0)
        peak_layout.addWidget(QtWidgets.QLabel("alpha index"), 0, 1)
        peak_layout.addWidget(QtWidgets.QLabel("eV index"), 0, 2)
        alpha_max = max(0, self._source_data.sizes.get("alpha", 1) - 1)
        energy_max = max(0, self._source_data.sizes.get("eV", 1) - 1)
        for row, label in enumerate(("Center", "Peak 1", "Peak 2"), start=1):
            alpha_spin = QtWidgets.QSpinBox()
            energy_spin = QtWidgets.QSpinBox()
            alpha_spin.setRange(0, alpha_max)
            energy_spin.setRange(0, energy_max)
            self.peak_spins.append((alpha_spin, energy_spin))
            peak_layout.addWidget(QtWidgets.QLabel(label), row, 0)
            peak_layout.addWidget(alpha_spin, row, 1)
            peak_layout.addWidget(energy_spin, row, 2)
        self.layout_.addRow(peak_group)

        self.method_combo = QtWidgets.QComboBox()
        for method in ("constant", "gaussian", "circular"):
            self.method_combo.addItem(method.title(), userData=method)
        self.layout_.addRow("Mask Method", self.method_combo)

        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(0, 9)
        self.order_spin.setValue(3)
        self.layout_.addRow("Order", self.order_spin)

        self.n_pad_spin = QtWidgets.QSpinBox()
        self.n_pad_spin.setRange(0, 9999)
        self.n_pad_spin.setValue(90)
        self.layout_.addRow("Padding", self.n_pad_spin)

        self.roi_hw_spin = QtWidgets.QSpinBox()
        self.roi_hw_spin.setRange(0, 9999)
        self.roi_hw_spin.setValue(25)
        self.layout_.addRow("ROI Half-width", self.roi_hw_spin)

        self.k_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            exact_float=True,
        )
        self.k_spin.setRange(-1e12, 1e12)
        self.k_spin.setDecimals(8)
        self.k_spin.setValue(0.5)
        self.layout_.addRow("Threshold k", self.k_spin)

        self.feather_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            exact_float=True,
        )
        self.feather_spin.setRange(0.0, 1e12)
        self.feather_spin.setDecimals(8)
        self.feather_spin.setValue(3.0)
        self.layout_.addRow("Feather", self.feather_spin)

        self.undo_edge_check = QtWidgets.QCheckBox()
        self.layout_.addRow("Undo Edge Correction", self.undo_edge_check)

    @property
    def _first_order_peaks(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        values = tuple(
            (alpha_spin.value(), energy_spin.value())
            for alpha_spin, energy_spin in self.peak_spins
        )
        return typing.cast(
            "tuple[tuple[int, int], tuple[int, int], tuple[int, int]]",
            values,
        )

    def preflight_data(self, data: xr.DataArray) -> None:
        missing = {"alpha", "eV"} - set(data.dims)
        if missing:
            raise ValueError("Mesh removal requires 'alpha' and 'eV' dimensions")

    def source_transform_operation(self) -> RemoveMeshOperation:
        return RemoveMeshOperation(
            first_order_peaks=self._first_order_peaks,
            order=self.order_spin.value(),
            n_pad=self.n_pad_spin.value(),
            roi_hw=self.roi_hw_spin.value(),
            k=self.k_spin.value(),
            feather=self.feather_spin.value(),
            undo_edge_correction=self.undo_edge_check.isChecked(),
            method=typing.cast(
                "typing.Literal['constant', 'gaussian', 'circular']",
                self.method_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
            ),
            output=typing.cast(
                "typing.Literal['corrected', 'mesh']",
                self.output_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
            ),
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(operation, RemoveMeshOperation):
            return
        if not _set_combo_data(self.output_combo, operation.output):
            raise ValueError(f"Mesh output {operation.output!r} is not available")
        if not _set_combo_data(self.method_combo, operation.method):
            raise ValueError(f"Mesh method {operation.method!r} is not available")
        for spins, peak in zip(
            self.peak_spins, operation.first_order_peaks, strict=True
        ):
            for spin, value in zip(spins, peak, strict=True):
                if not spin.minimum() <= value <= spin.maximum():
                    raise ValueError(
                        f"Mesh peak index {value} is outside the input data"
                    )
                spin.setValue(value)
        _set_spin_value(self.order_spin, operation.order, label="Mesh order")
        _set_spin_value(self.n_pad_spin, operation.n_pad, label="Mesh padding")
        _set_spin_value(
            self.roi_hw_spin,
            operation.roi_hw,
            label="Mesh ROI half-width",
        )
        _set_spin_value(self.k_spin, operation.k, label="Mesh threshold k")
        _set_spin_value(
            self.feather_spin,
            operation.feather,
            label="Mesh feather",
        )
        self.undo_edge_check.setChecked(operation.undo_edge_correction)


class SortByDialog(DataTransformDialog):
    title = "Sort By"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (SortByOperation,)

    def setup_widgets(self) -> None:
        source_data = erlab.utils.array._restore_nonuniform_dims(self.slicer_area.data)
        public_dims = tuple(
            dim.removesuffix("_idx")
            if isinstance(dim, str) and dim.endswith("_idx")
            else dim
            for dim in self.slicer_area.data.dims
        )
        if tuple(source_data.dims) != public_dims and set(source_data.dims) == set(
            public_dims
        ):
            source_data = source_data.transpose(*public_dims)

        sort_keys: list[Hashable] = list(source_data.dims)
        for name, coord in source_data.coords.items():
            if name in sort_keys:
                continue
            if (
                coord.ndim == 1
                and len(coord.dims) == 1
                and coord.dims[0] in source_data.dims
            ):
                sort_keys.append(name)

        key_group = QtWidgets.QGroupBox("Sort Keys")
        key_layout = QtWidgets.QVBoxLayout()
        key_group.setLayout(key_layout)

        self.key_table = QtWidgets.QTableWidget()
        self.key_table.setObjectName("sortby_key_table")
        self.key_table.setColumnCount(1)
        self.key_table.setHorizontalHeaderLabels(["Coordinate"])
        self.key_table.setRowCount(len(sort_keys))
        self.key_table.setAlternatingRowColors(True)
        self.key_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.key_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.key_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        item_flags = (
            QtCore.Qt.ItemFlag.ItemIsUserCheckable
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        for row, key in enumerate(sort_keys):
            item = QtWidgets.QTableWidgetItem(str(key))
            item.setFlags(item_flags)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            self.key_table.setItem(row, 0, item)
        # ImageTool data always contributes at least one dimension-backed sort key.
        if self.key_table.rowCount() != 0:  # pragma: no branch
            self.key_table.selectRow(0)

        header = typing.cast("QtWidgets.QHeaderView", self.key_table.horizontalHeader())
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.key_table.resizeRowsToContents()

        move_layout = QtWidgets.QHBoxLayout()
        self.move_up_button = QtWidgets.QPushButton("Move Up")
        self.move_up_button.setObjectName("sortby_move_up_button")
        self.move_down_button = QtWidgets.QPushButton("Move Down")
        self.move_down_button.setObjectName("sortby_move_down_button")
        self.move_up_button.clicked.connect(lambda: self._move_selected_row(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected_row(1))
        move_layout.addWidget(self.move_up_button)
        move_layout.addWidget(self.move_down_button)

        key_layout.addWidget(self.key_table)
        key_layout.addLayout(move_layout)

        self.ascending_combo = QtWidgets.QComboBox()
        self.ascending_combo.setObjectName("sortby_direction_combo")
        self.ascending_combo.addItem("Ascending", userData=True)
        self.ascending_combo.addItem("Descending", userData=False)

        self.layout_.addRow(key_group)
        self.layout_.addRow("Direction", self.ascending_combo)

    @property
    def _sort_keys(self) -> tuple[Hashable, ...]:
        keys: list[Hashable] = []
        for row in range(self.key_table.rowCount()):
            item = self.key_table.item(row, 0)
            if item is None:
                continue
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                keys.append(
                    typing.cast(
                        "Hashable",
                        item.data(QtCore.Qt.ItemDataRole.UserRole),
                    )
                )
        return tuple(keys)

    @QtCore.Slot()
    def _move_selected_row(self, offset: int) -> None:
        row = self.key_table.currentRow()
        target_row = row + offset
        if row < 0 or target_row < 0 or target_row >= self.key_table.rowCount():
            return
        item = self.key_table.takeItem(row, 0)
        target_item = self.key_table.takeItem(target_row, 0)
        if item is None or target_item is None:
            return
        self.key_table.setItem(row, 0, target_item)
        self.key_table.setItem(target_row, 0, item)
        self.key_table.selectRow(target_row)

    def source_transform_operation(
        self,
    ) -> SortByOperation:
        sort_keys = self._sort_keys
        if not sort_keys:
            raise ValueError("No sort keys selected")
        return SortByOperation(
            variables=sort_keys,
            ascending=bool(
                self.ascending_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
            ),
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            SortByOperation,
        ):
            return
        row_items: list[tuple[Hashable, QtWidgets.QTableWidgetItem]] = []
        for row in range(self.key_table.rowCount()):
            item = self.key_table.item(row, 0)
            if item is None:
                continue
            key = item.data(QtCore.Qt.ItemDataRole.UserRole)
            row_items.append((key, item))

        item_by_key = dict(row_items)
        missing = [key for key in operation.variables if key not in item_by_key]
        if missing:
            raise ValueError(f"Sort keys {missing!r} are not available")

        selected_keys = set(operation.variables)
        ordered_keys = [
            *operation.variables,
            *(key for key, _item in row_items if key not in selected_keys),
        ]
        for row in range(self.key_table.rowCount()):
            self.key_table.takeItem(row, 0)
        for row, key in enumerate(ordered_keys):
            item = item_by_key[key]
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if key in selected_keys
                else QtCore.Qt.CheckState.Unchecked
            )
            self.key_table.setItem(row, 0, item)
        if ordered_keys:
            self.key_table.selectRow(0)
        if not _set_combo_data(self.ascending_combo, operation.ascending):
            raise ValueError("Sort direction is not available")

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._sort_keys:
            QtWidgets.QMessageBox.warning(
                self,
                "No Sort Keys Selected",
                "Choose at least one coordinate to sort by.",
            )
            return
        super().accept()


class LeadingEdgeDialog(DataTransformDialog):
    title = "Leading Edge"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (LeadingEdgeOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )

        self.dim_combo = QtWidgets.QComboBox()
        for dim in self._source_data.dims:
            self.dim_combo.addItem(str(dim), userData=dim)
        ev_index = self.dim_combo.findData("eV", QtCore.Qt.ItemDataRole.UserRole)
        if ev_index >= 0:
            self.dim_combo.setCurrentIndex(ev_index)
        self.layout_.addRow("Dimension", self.dim_combo)

        self.fraction_spin = QtWidgets.QDoubleSpinBox()
        self.fraction_spin.setDecimals(6)
        self.fraction_spin.setRange(0.000001, 1.0)
        self.fraction_spin.setSingleStep(0.05)
        self.fraction_spin.setValue(0.5)
        self.layout_.addRow("Fraction", self.fraction_spin)

        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItem("Positive", userData="positive")
        self.direction_combo.addItem("Negative", userData="negative")
        self.layout_.addRow("Direction", self.direction_combo)

    @property
    def _selected_dim(self) -> Hashable | None:
        if self.dim_combo.currentIndex() < 0:
            return None
        return typing.cast(
            "Hashable",
            self.dim_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    @property
    def _direction(self) -> typing.Literal["positive", "negative"]:
        return typing.cast(
            "typing.Literal['positive', 'negative']",
            self.direction_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    def _source_coord_error(self, dim: Hashable) -> str | None:
        coord = np.asarray(self._source_data[dim].values)
        if coord.ndim != 1:
            return "The selected dimension coordinate must be one-dimensional."
        if not np.issubdtype(coord.dtype, np.number) or np.issubdtype(
            coord.dtype, np.complexfloating
        ):
            return "The selected dimension coordinate must contain real numbers."
        numeric = coord.astype(np.float64, copy=False)
        if not np.all(np.isfinite(numeric)):
            return "The selected dimension coordinate must be finite."
        if np.unique(numeric).size != numeric.size:
            return "The selected dimension coordinate values must be unique."
        return None

    def source_transform_operation(
        self,
    ) -> LeadingEdgeOperation:
        dim = self._selected_dim
        if dim is None:
            raise ValueError("No dimension selected")
        source_error = self._source_coord_error(dim)
        if source_error is not None:
            raise ValueError(source_error)
        return LeadingEdgeOperation(
            fraction=float(self.fraction_spin.value()),
            dim=dim,
            direction=self._direction,
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            LeadingEdgeOperation,
        ):
            return
        if not _set_combo_data(self.dim_combo, operation.dim):
            raise ValueError(f"Dimension {operation.dim!r} is not available")
        self.fraction_spin.setValue(float(operation.fraction))
        if not _set_combo_data(self.direction_combo, operation.direction):
            raise ValueError(f"Direction {operation.direction!r} is not available")

    @QtCore.Slot()
    def accept(self) -> None:
        dim = self._selected_dim
        if dim is None:
            QtWidgets.QMessageBox.warning(
                self, "No Dimension Selected", "Choose a dimension to process."
            )
            return

        source_error = self._source_coord_error(dim)
        if source_error is not None:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Source Coordinate",
                source_error,
            )
            return

        super().accept()


class CoarsenDialog(DataTransformDialog):
    title = "Coarsen"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (CoarsenOperation,)

    _REDUCERS: tuple[str, ...] = (
        "all",
        "any",
        "count",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "std",
        "sum",
        "var",
    )

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Window Size"), 0, 1)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        self.window_spins: dict[Hashable, QtWidgets.QSpinBox] = {}

        for row, dim in enumerate(self._source_data.dims, start=1):
            check = QtWidgets.QCheckBox(str(dim))
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 2_147_483_647)
            spin.setValue(3)
            spin.setEnabled(False)
            check.toggled.connect(spin.setEnabled)

            self.dim_checks[dim] = check
            self.window_spins[dim] = spin

            dim_layout.addWidget(check, row, 0)
            dim_layout.addWidget(spin, row, 1)

        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QFormLayout()
        options_group.setLayout(options_layout)

        self.boundary_combo = QtWidgets.QComboBox()
        self.boundary_combo.addItems(["exact", "trim", "pad"])
        self.boundary_combo.setCurrentText("trim")
        options_layout.addRow("Boundary", self.boundary_combo)

        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItems(["left", "right"])
        options_layout.addRow("Side", self.side_combo)

        self.coord_func_combo = QtWidgets.QComboBox()
        self.coord_func_combo.addItems(
            ["mean", "median", "min", "max", "first", "last"]
        )
        self.coord_func_combo.setCurrentText("mean")
        options_layout.addRow("Coordinate Function", self.coord_func_combo)

        self.reducer_combo = QtWidgets.QComboBox()
        self.reducer_combo.addItems(list(self._REDUCERS))
        self.reducer_combo.setCurrentText("mean")
        options_layout.addRow("Reducer", self.reducer_combo)

        self.layout_.addRow(dim_group)
        self.layout_.addRow(options_group)

    @property
    def _selected_windows(self) -> dict[Hashable, int]:
        return {
            dim: self.window_spins[dim].value()
            for dim, check in self.dim_checks.items()
            if check.isChecked()
        }

    @property
    def _coord_func(self) -> str:
        return self.coord_func_combo.currentText().strip()

    @property
    def _reducer(self) -> str:
        return self.reducer_combo.currentText()

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        if not self._selected_windows:
            raise ValueError("No dimensions selected")
        return CoarsenOperation(
            dim=self._selected_windows,
            boundary=self.boundary_combo.currentText(),
            side=self.side_combo.currentText(),
            coord_func=self._coord_func,
            reducer=self._reducer,
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            CoarsenOperation,
        ):
            return
        for check in self.dim_checks.values():
            check.setChecked(False)
        for dim, window in operation.dim.items():
            if dim not in self.dim_checks:
                raise ValueError(f"Dimension {dim!r} is not available")
            self.dim_checks[dim].setChecked(True)
            self.window_spins[dim].setValue(int(window))
        for combo, value in (
            (self.boundary_combo, operation.boundary),
            (self.side_combo, operation.side),
            (self.coord_func_combo, operation.coord_func),
            (self.reducer_combo, operation.reducer),
        ):
            if not _set_combo_text(combo, value):
                raise ValueError(f"Option {value!r} is not available")

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._selected_windows:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension.",
            )
            return

        if self.boundary_combo.currentText() == "exact":
            invalid_dims = [
                str(dim)
                for dim, window in self._selected_windows.items()
                if self._source_data.sizes[dim] % window != 0
            ]
            if invalid_dims:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Incompatible Window Size",
                    "Window sizes must evenly divide the selected dimensions when "
                    "boundary is exact: "
                    f"{', '.join(invalid_dims)}. Try trim or pad instead.",
                )
                return

        if self.boundary_combo.currentText() == "trim":
            empty_dims = [
                str(dim)
                for dim, window in self._selected_windows.items()
                if self._source_data.sizes[dim] // window == 0
            ]
            if empty_dims:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Output Blocks",
                    "Trim boundary would remove all data along: "
                    f"{', '.join(empty_dims)}.",
                )
                return

        super().accept()


class ThinDialog(DataTransformDialog):
    title = "Thin Data"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (ThinOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )

        mode_group = QtWidgets.QGroupBox("Mode")
        mode_layout = QtWidgets.QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.per_dim_radio = QtWidgets.QRadioButton("Thin selected dimensions")
        self.global_radio = QtWidgets.QRadioButton("Thin all dimensions equally")
        self.per_dim_radio.setChecked(True)

        mode_layout.addWidget(self.per_dim_radio)
        mode_layout.addWidget(self.global_radio)

        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.addButton(self.per_dim_radio)
        self._mode_group.addButton(self.global_radio)

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Factor"), 0, 1)

        self.dim_group = dim_group
        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        self.factor_spins: dict[Hashable, QtWidgets.QSpinBox] = {}

        for row, dim in enumerate(self._source_data.dims, start=1):
            check = QtWidgets.QCheckBox(str(dim))
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 2_147_483_647)
            spin.setValue(2)
            spin.setEnabled(False)
            check.toggled.connect(spin.setEnabled)

            self.dim_checks[dim] = check
            self.factor_spins[dim] = spin

            dim_layout.addWidget(check, row, 0)
            dim_layout.addWidget(spin, row, 1)

        global_group = QtWidgets.QGroupBox("Global Factor")
        global_layout = QtWidgets.QFormLayout()
        global_group.setLayout(global_layout)

        self.global_group = global_group
        self.global_spin = QtWidgets.QSpinBox()
        self.global_spin.setRange(1, 2_147_483_647)
        self.global_spin.setValue(2)
        global_layout.addRow("Factor", self.global_spin)

        self.per_dim_radio.toggled.connect(self._update_mode)
        self.global_radio.toggled.connect(self._update_mode)

        self.layout_.addRow(mode_group)
        self.layout_.addRow(dim_group)
        self.layout_.addRow(global_group)
        self._update_mode()

    @property
    def _use_global_mode(self) -> bool:
        return self.global_radio.isChecked()

    @property
    def _selected_factors(self) -> dict[Hashable, int]:
        return {
            dim: self.factor_spins[dim].value()
            for dim, check in self.dim_checks.items()
            if check.isChecked()
        }

    @property
    def _effective_factors(self) -> dict[Hashable, int]:
        if self._use_global_mode:
            return {
                dim: self.global_spin.value()
                for dim in self._source_data.dims
                if self.global_spin.value() > 1
            }
        return {
            dim: factor for dim, factor in self._selected_factors.items() if factor > 1
        }

    @QtCore.Slot()
    def _update_mode(self) -> None:
        self.dim_group.setEnabled(not self._use_global_mode)
        self.global_group.setEnabled(self._use_global_mode)

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        if self._use_global_mode:
            if self.global_spin.value() <= 1:
                raise ValueError("No thinning requested")
            return ThinOperation(mode="global", factor=self.global_spin.value())
        if not self._effective_factors:
            raise ValueError("No thinning requested")
        return ThinOperation(mode="per_dim", factors=self._effective_factors)

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(operation, ThinOperation):
            return
        for check in self.dim_checks.values():
            check.setChecked(False)
        if operation.mode == "global":
            self.global_radio.setChecked(True)
            self.global_spin.setValue(int(typing.cast("int", operation.factor)))
        else:
            self.per_dim_radio.setChecked(True)
            for dim, factor in operation.factors.items():
                if dim not in self.dim_checks:
                    raise ValueError(f"Dimension {dim!r} is not available")
                self.dim_checks[dim].setChecked(True)
                self.factor_spins[dim].setValue(int(factor))
        self._update_mode()

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._use_global_mode and not self._selected_factors:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension.",
            )
            return

        if not self._effective_factors:
            QtWidgets.QMessageBox.warning(
                self,
                "No Thinning Requested",
                "Choose at least one thinning factor greater than 1.",
            )
            return

        super().accept()


class SqueezeDialog(DataTransformDialog):
    title = "Squeeze Dimensions"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (SqueezeOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_group.setObjectName("squeezeDimensionsGroup")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        singleton_dims = [
            dim for dim in self._source_data.dims if self._source_data.sizes[dim] == 1
        ]
        if singleton_dims:
            dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
            dim_layout.addWidget(QtWidgets.QLabel("Size"), 0, 1)
            for row, dim in enumerate(singleton_dims, start=1):
                check = QtWidgets.QCheckBox(str(dim))
                check.setObjectName(f"squeezeDimCheck{row}")
                check.setChecked(True)
                self.dim_checks[dim] = check
                dim_layout.addWidget(check, row, 0)
                dim_layout.addWidget(QtWidgets.QLabel("1"), row, 1)
        else:
            dim_layout.addWidget(QtWidgets.QLabel("No singleton dimensions"), 0, 0)

        self.drop_check = QtWidgets.QCheckBox("Drop squeezed coordinates")
        self.drop_check.setObjectName("squeezeDropCheck")

        self.layout_.addRow(dim_group)
        self.layout_.addRow("Coordinates", self.drop_check)

    @property
    def _selected_dims(self) -> tuple[Hashable, ...]:
        return tuple(dim for dim, check in self.dim_checks.items() if check.isChecked())

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if not self.dim_checks:
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing to Squeeze",
                "The data has no dimensions of size 1.",
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._selected_dims:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "Choose at least one dimension to squeeze.",
            )
            return
        if len(self._selected_dims) >= self._source_data.ndim:
            QtWidgets.QMessageBox.warning(
                self,
                "No Displayable Dimensions",
                "Squeezing all dimensions would leave no axes to display.",
            )
            return
        super().accept()

    def preflight_data(self, data: xr.DataArray) -> None:
        selected_dims = self._selected_dims
        missing_dims = [dim for dim in selected_dims if dim not in data.dims]
        if missing_dims:
            raise ValueError(f"Dimensions are not available: {missing_dims!r}")
        nonsingleton_dims = [dim for dim in selected_dims if data.sizes[dim] != 1]
        if nonsingleton_dims:
            raise ValueError(f"Dimensions are not size 1: {nonsingleton_dims!r}")
        if data.ndim - len(selected_dims) < 1:
            raise ValueError("Squeeze would remove all dimensions")

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        if not self._selected_dims:
            raise ValueError("No dimensions selected")
        return SqueezeOperation(
            dims=self._selected_dims,
            drop=self.drop_check.isChecked(),
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            SqueezeOperation,
        ):
            return
        for check in self.dim_checks.values():
            check.setChecked(False)
        dims = (
            tuple(self.dim_checks) if operation.dims is None else tuple(operation.dims)
        )
        for dim in dims:
            if dim not in self.dim_checks:
                raise ValueError(f"Dimension {dim!r} is not available")
            self.dim_checks[dim].setChecked(True)
        self.drop_check.setChecked(operation.drop)


class SymmetrizeDialog(DataTransformDialog):
    title = "Symmetrize"
    enable_copy = True
    operation_types = (SymmetrizeOperation,)

    def setup_widgets(self) -> None:
        dim_group = QtWidgets.QGroupBox("Mirror plane")
        dim_layout = QtWidgets.QHBoxLayout()
        dim_group.setLayout(dim_layout)

        self._dim_combo = QtWidgets.QComboBox()
        self._dim_combo.addItems([str(d) for d in self.slicer_area.data.dims])
        dim_layout.addWidget(self._dim_combo)

        self._center_spin = QtWidgets.QDoubleSpinBox()
        dim_layout.addWidget(self._center_spin)
        self._dim_combo.currentIndexChanged.connect(self._update_spin)
        self._update_spin()

        option_group = QtWidgets.QGroupBox("Options")
        option_group_layout = QtWidgets.QVBoxLayout()
        option_group.setLayout(option_group_layout)

        self.subtract_check = QtWidgets.QCheckBox("Subtract")
        self.subtract_check.setChecked(False)
        self.subtract_check.setToolTip(
            "Subtract the reflected part from the data instead of adding it."
        )
        option_group_layout.addWidget(self.subtract_check)

        option_mode = QtWidgets.QWidget()
        option_group_layout.addWidget(option_mode)
        option_mode_layout = QtWidgets.QHBoxLayout()
        option_mode_layout.setContentsMargins(0, 0, 0, 0)
        option_mode.setLayout(option_mode_layout)
        option_mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.opt_mode: list[QtWidgets.QRadioButton] = []
        self.opt_mode.append(QtWidgets.QRadioButton("full"))
        self.opt_mode.append(QtWidgets.QRadioButton("valid"))
        self.opt_mode[0].setChecked(True)
        for opt in self.opt_mode:
            option_mode_layout.addWidget(opt)

        option_part = QtWidgets.QWidget()
        option_group_layout.addWidget(option_part)
        option_part_layout = QtWidgets.QHBoxLayout()
        option_part_layout.setContentsMargins(0, 0, 0, 0)
        option_part.setLayout(option_part_layout)
        option_part_layout.addWidget(QtWidgets.QLabel("Part to Keep:"))
        self.opt_part: list[QtWidgets.QRadioButton] = []
        self.opt_part.append(QtWidgets.QRadioButton("both"))
        self.opt_part.append(QtWidgets.QRadioButton("below"))
        self.opt_part.append(QtWidgets.QRadioButton("above"))
        self.opt_part[0].setChecked(True)
        for opt in self.opt_part:
            option_part_layout.addWidget(opt)

        self.layout_.addRow(dim_group)
        self.layout_.addRow(option_group)

    @QtCore.Slot()
    def _update_spin(self) -> None:
        axis = self._dim_combo.currentIndex()
        self._center_spin.setRange(*map(float, self.array_slicer.lims_uniform[axis]))
        self._center_spin.setSingleStep(float(self.array_slicer.incs_uniform[axis]))
        self._center_spin.setDecimals(
            self.array_slicer.get_significant(axis, uniform=True)
        )
        self._center_spin.setValue(
            self.slicer_area.current_values_uniform[self._dim_combo.currentIndex()]
        )

    @property
    def _params(self) -> dict[str, typing.Any]:
        return {
            "dim": self._dim_combo.currentText(),
            "center": float(
                np.round(self._center_spin.value(), self._center_spin.decimals())
            ),
            "subtract": self.subtract_check.isChecked(),
            "mode": ("full", "valid")[
                next(i for i, opt in enumerate(self.opt_mode) if opt.isChecked())
            ],
            "part": ("both", "below", "above")[
                next(i for i, opt in enumerate(self.opt_part) if opt.isChecked())
            ],
        }

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        return SymmetrizeOperation(**self._params)

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            SymmetrizeOperation,
        ):
            return
        if not _set_combo_text(self._dim_combo, str(operation.dim)):
            raise ValueError(f"Dimension {operation.dim!r} is not available")
        self._center_spin.setValue(float(operation.center))
        self.subtract_check.setChecked(operation.subtract)
        self.opt_mode[0 if operation.mode == "full" else 1].setChecked(True)
        part_index = {"both": 0, "below": 1, "above": 2}[operation.part]
        self.opt_part[part_index].setChecked(True)


class SymmetrizeNfoldDialog(DataTransformDialog):
    title = "Rotational Symmetrize"
    enable_copy = True
    operation_types = (SymmetrizeNfoldOperation,)

    @property
    def _params(self) -> dict[str, typing.Any]:
        return {
            "fold": self.fold_spin.value(),
            "axes": self._axes,
            "center": {
                dim: float(np.round(spin.value(), spin.decimals()))
                for dim, spin in zip(self._axes, self.center_spins, strict=True)
            },
            "reshape": self.reshape_check.isChecked(),
            "order": self.order_spin.value(),
        }

    def setup_widgets(self) -> None:
        main_image = self.slicer_area.main_image

        self._axes: tuple[str, str] = (
            str(main_image.axis_dims_uniform[0]),
            str(main_image.axis_dims_uniform[1]),
        )

        plane_group = QtWidgets.QGroupBox("Rotation plane")
        plane_layout = QtWidgets.QHBoxLayout()
        plane_group.setLayout(plane_layout)
        plane_dims = ", ".join(self._axes)
        plane_label = QtWidgets.QLabel(plane_dims)
        plane_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        plane_layout.addWidget(plane_label)
        self.layout_.addRow(plane_group)

        self.fold_spin = QtWidgets.QSpinBox()
        self.fold_spin.setRange(2, 99)
        self.fold_spin.setValue(4)
        self.layout_.addRow("Fold", self.fold_spin)

        self.center_spins = (
            erlab.interactive.utils.BetterSpinBox(compact=False),
            erlab.interactive.utils.BetterSpinBox(compact=False),
        )
        for i in range(2):
            axis = main_image.display_axis[i]
            dim = str(main_image.axis_dims_uniform[i])
            self.center_spins[i].setDecimals(
                self.array_slicer.get_significant(axis, uniform=True)
            )
            self.center_spins[i].setSingleStep(float(self.array_slicer.incs_uniform[i]))
            self.center_spins[i].setValue(self.slicer_area.current_values_uniform[axis])
            self.layout_.addRow(f"Center {dim}", self.center_spins[i])

        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(0, 5)
        self.order_spin.setValue(1)
        self.layout_.addRow("Spline Order", self.order_spin)

        self.reshape_check = QtWidgets.QCheckBox("Reshape")
        self.reshape_check.setChecked(True)
        self.layout_.addRow(self.reshape_check)

        if main_image.is_guidelines_visible:
            for spin, val in zip(
                self.center_spins, main_image._guideline_offset, strict=True
            ):
                spin.setValue(val)
            self.fold_spin.setValue(2 * (len(main_image._guidelines_items) - 1))

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        return SymmetrizeNfoldOperation(**self._params)

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            SymmetrizeNfoldOperation,
        ):
            return
        if tuple(operation.axes) != self._axes:
            raise ValueError("Rotational symmetrize axes are not currently visible")
        self.fold_spin.setValue(int(operation.fold))
        self.reshape_check.setChecked(operation.reshape)
        self.order_spin.setValue(int(operation.order))
        center = operation.center
        if isinstance(center, Mapping):
            values = [center[dim] for dim in self._axes]
        else:
            values = center
        for spin, value in zip(self.center_spins, values, strict=True):
            spin.setValue(float(value))


class EdgeCorrectionDialog(DataTransformDialog):
    title = "Edge Correction"
    enable_copy = False

    def setup_widgets(self) -> None:
        self.shift_coord_check = QtWidgets.QCheckBox("Shift Coordinates")
        self.shift_coord_check.setChecked(True)

        self.layout_.addRow(self.shift_coord_check)

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        edge_fit = getattr(self, "_edge_fit", None)
        if edge_fit is None:
            raise RuntimeError("Edge correction fit data has not been loaded.")
        return CorrectWithEdgeOperation(
            edge_fit=edge_fit,
            shift_coords=self.shift_coord_check.isChecked(),
        )

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if "eV" not in self.slicer_area.data.dims:
            QtWidgets.QMessageBox.warning(
                self,
                "No Energy Dimension",
                "Edge correction requires an energy dimension (eV) in the data.",
            )
            return QtWidgets.QDialog.DialogCode.Rejected

        self._edge_fit: xr.Dataset | None = erlab.interactive.utils.load_fit_ui()
        if self._edge_fit is None:
            # User canceled the fit dialog
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()


class _BaseCropDialog(DataTransformDialog):
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (
        SelOperation,
        IselOperation,
    )

    @property
    def _slice_kwargs(self) -> dict[Hashable, slice]:
        raise NotImplementedError

    @staticmethod
    def _nonuniform_isel_slice(selector: slice) -> slice:
        start = None if selector.start is None else int(selector.start)
        stop = None if selector.stop is None else int(selector.stop)
        step = selector.step
        if stop is not None:
            if step is not None and step < 0:
                stop -= 1
            else:
                stop += 1
        return slice(start, stop, step)

    def source_operations(
        self,
    ) -> list[ToolProvenanceOperation]:
        sel_kwargs: dict[Hashable, slice] = dict(self._slice_kwargs)
        isel_kwargs: dict[Hashable, slice] = {}
        operations: list[ToolProvenanceOperation] = []

        for key in list(sel_kwargs.keys()):
            if isinstance(key, str) and key.endswith("_idx"):
                isel_kwargs[key.removesuffix("_idx")] = self._nonuniform_isel_slice(
                    sel_kwargs.pop(key)
                )

        if sel_kwargs:
            operations.append(SelOperation(kwargs=sel_kwargs))
        if isel_kwargs:
            operations.append(IselOperation(kwargs=isel_kwargs))
        return operations

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        for operation in self.source_operations():
            data = operation.apply(data)
        return data


class CropToViewDialog(_BaseCropDialog):
    title = "Crop to View"
    whatsthis = "Crop the data to the currently visible area."

    def setup_widgets(self) -> None:
        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QVBoxLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}

        for i, d in enumerate(self.slicer_area.data.dims):
            self.dim_checks[d] = QtWidgets.QCheckBox(str(d))
            dim_layout.addWidget(self.dim_checks[d])
            if i < 2:
                # Enable first 2 dimensions by default
                self.dim_checks[d].setChecked(True)

            if d not in self.slicer_area.manual_limits:
                # Disable dimensions without manual limits
                self.dim_checks[d].setChecked(False)
                self.dim_checks[d].setDisabled(True)

        self.layout_.addRow(dim_group)

    @property
    def _slice_kwargs(self) -> dict[Hashable, slice]:
        return {
            k: v
            for k, v in self.slicer_area.make_slice_dict().items()
            if self.dim_checks[k].isChecked()
        }

    @QtCore.Slot()
    def accept(self) -> None:
        if self._slice_kwargs == {}:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension with manual limits.",
            )
            return
        super().accept()

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if len(self.slicer_area.manual_limits) == 0:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to Crop", "Manually zoom in to define the crop area."
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()


class CropDialog(_BaseCropDialog):
    title = "Crop Between Cursors"

    @property
    def _enabled_dims(self) -> list[Hashable]:
        return [k for k, v in self.dim_checks.items() if v.isChecked()]

    @property
    def _cursor_indices(self) -> tuple[int, int]:
        return typing.cast(
            "tuple[int, int]",
            tuple(combo.currentIndex() for combo in self.cursor_combos),
        )

    @property
    def _slice_kwargs(self) -> dict[Hashable, slice]:
        c0, c1 = self._cursor_indices

        slice_dict: dict[Hashable, slice] = {}

        for k in self._enabled_dims:
            ax_idx = self.slicer_area.data.dims.index(k)
            sig_digits = self.array_slicer.get_significant(ax_idx, uniform=True)

            start = self.array_slicer.get_value(cursor=c0, axis=ax_idx, uniform=True)
            end = self.array_slicer.get_value(cursor=c1, axis=ax_idx, uniform=True)

            if start > end:
                start, end = end, start

            start = float(np.round(start, sig_digits))
            end = float(np.round(end, sig_digits))

            if sig_digits == 0:
                start, end = int(start), int(end)

            slice_dict[k] = slice(start, end)

        return slice_dict

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if self.slicer_area.n_cursors == 1:
            QtWidgets.QMessageBox.warning(
                self, "Only 1 Cursor", "You need at least 2 cursors to crop the data."
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()

    @QtCore.Slot()
    def accept(self) -> None:
        if self._slice_kwargs == {}:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Selected",
                "You need to select at least one dimension.",
            )
            return
        super().accept()

    def setup_widgets(self) -> None:
        if self.slicer_area.n_cursors == 1:
            return

        self._cursors_group = erlab.interactive.utils.ExclusiveComboGroup(self)

        self.cursor_combos: list[QtWidgets.QComboBox] = []

        cursor_group = QtWidgets.QGroupBox("Between")
        cursor_layout = QtWidgets.QHBoxLayout()
        cursor_group.setLayout(cursor_layout)

        for i in range(2):
            combo = QtWidgets.QComboBox()
            for cursor_idx in range(self.slicer_area.n_cursors):
                combo.addItem(
                    self.slicer_area._cursor_icon(cursor_idx),
                    self.slicer_area._cursor_name(cursor_idx),
                )
            combo.setCurrentIndex(i)
            combo.setMaximumHeight(QtGui.QFontMetrics(combo.font()).height() + 3)
            combo.setIconSize(QtCore.QSize(10, 10))
            cursor_layout.addWidget(combo)
            self._cursors_group.addCombo(combo)
            self.cursor_combos.append(combo)

        self.layout_.addRow(cursor_group)

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QVBoxLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}

        for d in self.slicer_area.data.dims:
            self.dim_checks[d] = QtWidgets.QCheckBox(str(d))
            dim_layout.addWidget(self.dim_checks[d])

        self.layout_.addRow(dim_group)


class NormalizeDialog(DataFilterDialog):
    title = "Normalize"
    enable_copy = True
    operation_types = (NormalizeOperation,)
    denominator_rtol: float = 1e-12
    _MODES: typing.ClassVar[
        tuple[typing.Literal["area", "minmax", "min", "min_area"], ...]
    ] = ("area", "minmax", "min", "min_area")

    def setup_widgets(self) -> None:
        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QVBoxLayout()
        dim_group.setLayout(dim_layout)

        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}

        for d in self.slicer_area.data.dims:
            self.dim_checks[d] = QtWidgets.QCheckBox(str(d))
            dim_layout.addWidget(self.dim_checks[d])

        option_group = QtWidgets.QGroupBox("Options")
        option_layout = QtWidgets.QVBoxLayout()
        option_group.setLayout(option_layout)

        self.opts: list[QtWidgets.QRadioButton] = []
        self.opts.append(QtWidgets.QRadioButton("Data/Area"))
        self.opts.append(QtWidgets.QRadioButton("(Data−m)/(M−m)"))
        self.opts.append(QtWidgets.QRadioButton("Data−m"))
        self.opts.append(QtWidgets.QRadioButton("(Data−m)/Area"))

        self.opts[0].setChecked(True)
        for opt in self.opts:
            option_layout.addWidget(opt)

        self.layout_.addRow(dim_group)
        self.layout_.addRow(option_group)

    @property
    def _norm_dims(self) -> tuple[Hashable, ...]:
        return tuple(k for k, v in self.dim_checks.items() if v.isChecked())

    @property
    def _mode(self) -> typing.Literal["area", "minmax", "min", "min_area"]:
        return self._MODES[
            next(i for i, opt in enumerate(self.opts) if opt.isChecked())
        ]

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        operation = self.filter_operation()
        if operation is None:
            return data
        return operation.apply(data)

    def filter_operation(
        self,
    ) -> ToolProvenanceOperation | None:
        norm_dims = self._norm_dims
        if not norm_dims:
            return None
        return NormalizeOperation(
            dims=norm_dims,
            mode=self._mode,
            denominator_rtol=self.denominator_rtol,
        )

    def restore_filter_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            NormalizeOperation,
        ):
            return
        for check in self.dim_checks.values():
            check.setChecked(False)
        for dim in operation.dims:
            if dim in self.dim_checks:
                self.dim_checks[dim].setChecked(True)
        self.opts[self._MODES.index(operation.mode)].setChecked(True)
        self.denominator_rtol = operation.denominator_rtol


class DivideByCoordDialog(DataTransformDialog):
    title = "Divide by Coordinate"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (DivideByCoordOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )
        self.coord_combo = QtWidgets.QComboBox()
        self.coord_dims_label = QtWidgets.QLabel()

        data_dims = set(self._source_data.dims)
        for name, coord in self._source_data.coords.items():
            coord_dims = tuple(coord.dims)
            if (
                set(coord_dims).issubset(data_dims)
                and np.issubdtype(coord.dtype, np.number)
                and not np.issubdtype(coord.dtype, np.complexfloating)
            ):
                self.coord_combo.addItem(str(name), userData=name)

        self.coord_combo.currentIndexChanged.connect(self._update_coord_dims_label)
        self.layout_.addRow("Coordinate", self.coord_combo)
        self.layout_.addRow("Coordinate Dims", self.coord_dims_label)
        self._update_coord_dims_label()

    @property
    def _selected_coord_name(self) -> Hashable | None:
        if self.coord_combo.currentIndex() < 0:
            return None
        return typing.cast(
            "Hashable",
            self.coord_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _update_coord_dims_label(self, index: int | None = None) -> None:
        coord_name = self._selected_coord_name
        if coord_name is None:
            self.coord_dims_label.setText("")
            return
        coord_dims = self._source_data.coords[coord_name].dims
        self.coord_dims_label.setText(
            "scalar" if not coord_dims else ", ".join(str(dim) for dim in coord_dims)
        )

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if self.coord_combo.count() == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "No Coordinates",
                "No numeric coordinates that can be broadcast to the data were found.",
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()

    @QtCore.Slot()
    def accept(self) -> None:
        coord_name = self._selected_coord_name
        if coord_name is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Coordinate Selected",
                "Choose a coordinate to divide by.",
            )
            return
        coord = self._source_data.coords[coord_name]
        try:
            DivideByCoordOperation._raise_if_zero(coord)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Zero Coordinate Values",
                "The selected coordinate contains zero values and cannot be used as a "
                "divisor.",
            )
            return
        super().accept()

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        coord_name = self._selected_coord_name
        if coord_name is None:
            raise ValueError("No coordinate selected")
        return DivideByCoordOperation(coord_name=coord_name)

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            DivideByCoordOperation,
        ):
            return
        if not _set_combo_data(self.coord_combo, operation.coord_name):
            raise ValueError(f"Coordinate {operation.coord_name!r} is not available")


class GaussianFilterDialog(DataFilterDialog):
    title = "Gaussian Filter"
    enable_copy = True
    operation_types = (GaussianFilterOperation,)

    def setup_widgets(self) -> None:
        self._source_data = self.slicer_area._data
        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        self.sigma_spins: dict[Hashable, erlab.interactive.utils.BetterSpinBox] = {}
        self.fwhm_spins: dict[Hashable, erlab.interactive.utils.BetterSpinBox] = {}

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Sigma"), 0, 1)
        dim_layout.addWidget(QtWidgets.QLabel("FWHM"), 0, 2)

        for row, dim in enumerate(self._source_data.dims, start=1):
            check = QtWidgets.QCheckBox(str(dim))
            sigma_spin = erlab.interactive.utils.BetterSpinBox(
                compact=False, exact_float=True
            )
            fwhm_spin = erlab.interactive.utils.BetterSpinBox(
                compact=False, exact_float=True
            )

            sigma_spin.setMinimum(0.0)
            fwhm_spin.setMinimum(0.0)

            support_reason = self._unsupported_reason(dim)
            if support_reason is None:
                step = self._coord_step(dim)
                sigma_spin.setSingleStep(step)
                fwhm_spin.setSingleStep(step * _GAUSSIAN_FWHM_FACTOR)

                sigma_spin.setDecimals(self._sigma_decimals(dim))
                fwhm_spin.setDecimals(self._fwhm_decimals(dim, step))

                sigma_spin.setValue(step)
                fwhm_spin.setValue(step * _GAUSSIAN_FWHM_FACTOR)

                sigma_spin.setEnabled(False)
                fwhm_spin.setEnabled(False)

                check.toggled.connect(sigma_spin.setEnabled)
                check.toggled.connect(fwhm_spin.setEnabled)
                sigma_spin.valueChanged.connect(
                    lambda _value, dim=dim: self._sync_from_sigma(dim)
                )
                fwhm_spin.valueChanged.connect(
                    lambda _value, dim=dim: self._sync_from_fwhm(dim)
                )
            else:
                for widget in (check, sigma_spin, fwhm_spin):
                    widget.setToolTip(support_reason)
                    widget.setEnabled(False)

            self.dim_checks[dim] = check
            self.sigma_spins[dim] = sigma_spin
            self.fwhm_spins[dim] = fwhm_spin

            dim_layout.addWidget(check, row, 0)
            dim_layout.addWidget(sigma_spin, row, 1)
            dim_layout.addWidget(fwhm_spin, row, 2)

        self.layout_.addRow(dim_group)

    def _unsupported_reason(self, dim: Hashable) -> str | None:
        coord = np.asarray(self._source_data[dim].values, dtype=np.float64)
        if coord.size < 2:
            return "Gaussian filtering requires at least two coordinate values."
        if np.allclose(np.diff(coord), 0.0):
            return "Gaussian filtering does not support constant coordinates."
        if not erlab.utils.array.is_uniform_spaced(coord):
            return "Gaussian filtering requires uniformly spaced coordinates."
        return None

    def _coord_step(self, dim: Hashable) -> float:
        coord = np.asarray(self._source_data[dim].values, dtype=np.float64)
        return float(np.abs(coord[1] - coord[0]))

    def _sigma_decimals(self, dim: Hashable) -> int:
        return erlab.utils.array.effective_decimals(
            np.asarray(self._source_data[dim].values, dtype=np.float64)
        )

    def _fwhm_decimals(self, dim: Hashable, sigma: float) -> int:
        sigma_decimals = self._sigma_decimals(dim)
        fwhm_decimals = max(
            sigma_decimals,
            erlab.utils.array.effective_decimals(sigma * _GAUSSIAN_FWHM_FACTOR),
        )

        sigma_literal = self._format_literal(sigma, sigma_decimals)
        while (
            fwhm_decimals < 15
            and self._roundtrip_sigma_literal(
                sigma_literal, sigma_decimals, fwhm_decimals
            )
            != sigma_literal
        ):
            fwhm_decimals += 1
        return fwhm_decimals

    def _format_literal(self, value: float, decimals: int) -> str:
        return np.format_float_positional(
            value, precision=decimals, unique=False, fractional=True, trim="k"
        )

    def _roundtrip_sigma_literal(
        self, sigma_literal: str, sigma_decimals: int, fwhm_decimals: int
    ) -> str:
        fwhm_literal = self._format_literal(
            float(sigma_literal) * _GAUSSIAN_FWHM_FACTOR, fwhm_decimals
        )
        return self._format_literal(
            float(fwhm_literal) / _GAUSSIAN_FWHM_FACTOR, sigma_decimals
        )

    def _spin_literal(self, spin: erlab.interactive.utils.BetterSpinBox) -> str:
        return spin.text().removeprefix(spin.prefix())

    def _spin_value(self, spin: erlab.interactive.utils.BetterSpinBox) -> float:
        return float(self._spin_literal(spin))

    def _set_synced_exact_value(
        self, spin: erlab.interactive.utils.BetterSpinBox, value: float
    ) -> None:
        with QtCore.QSignalBlocker(spin):
            spin.setValue(value)
            line = spin.lineEdit()
            if line is None:
                return
            line.setText(str(float(value)))
            spin.editingFinishedEvent()
            line.setModified(False)

    def _commit_numeric_inputs(self) -> None:
        for spin in (*self.sigma_spins.values(), *self.fwhm_spins.values()):
            line = spin.lineEdit()
            if line is None or not line.isModified():
                continue
            spin.editingFinishedEvent()
            line.setModified(False)

    @QtCore.Slot()
    def _sync_from_sigma(self, dim: Hashable) -> None:
        value = self._spin_value(self.sigma_spins[dim]) * _GAUSSIAN_FWHM_FACTOR
        with QtCore.QSignalBlocker(self.fwhm_spins[dim]):
            self.fwhm_spins[dim].setValue(value)

    @QtCore.Slot()
    def _sync_from_fwhm(self, dim: Hashable) -> None:
        value = self._spin_value(self.fwhm_spins[dim]) / _GAUSSIAN_FWHM_FACTOR
        self._set_synced_exact_value(self.sigma_spins[dim], value)

    def _sigma_values(self) -> tuple[dict[Hashable, float], dict[Hashable, str]]:
        self._commit_numeric_inputs()

        sigma_values: dict[Hashable, float] = {}
        sigma_literals: dict[Hashable, str] = {}
        for dim in self._source_data.dims:
            if not self.dim_checks[dim].isChecked():
                continue
            sigma_literal = self._spin_literal(self.sigma_spins[dim])
            sigma_literals[dim] = sigma_literal
            sigma_values[dim] = float(sigma_literal)
        return sigma_values, sigma_literals

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        sigma_values, _ = self._sigma_values()
        if not sigma_values:
            return data
        return erlab.analysis.image.gaussian_filter(data, sigma=sigma_values)

    def filter_operation(
        self,
    ) -> ToolProvenanceOperation | None:
        sigma_values, _ = self._sigma_values()
        if not sigma_values:
            return None
        return GaussianFilterOperation(sigma=sigma_values)

    def restore_filter_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            GaussianFilterOperation,
        ):
            return
        for check in self.dim_checks.values():
            check.setChecked(False)
        for dim, sigma in operation.sigma.items():
            if dim not in self.dim_checks or not self.dim_checks[dim].isEnabled():
                continue
            self.dim_checks[dim].setChecked(True)
            self._set_synced_exact_value(self.sigma_spins[dim], float(sigma))
            self._sync_from_sigma(dim)


class _BoxcarFilterDialog(DataFilterDialog):
    """Parameter editor for boxcar-filter provenance."""

    title = "Boxcar Filter"
    enable_copy = True
    operation_types = (BoxcarFilterOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )
        self.dim_checks: dict[Hashable, QtWidgets.QCheckBox] = {}
        self.size_spins: dict[Hashable, QtWidgets.QSpinBox] = {}

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout(dim_group)
        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Window Size"), 0, 1)
        for row, dim in enumerate(self._source_data.dims, start=1):
            check = QtWidgets.QCheckBox(str(dim))
            size_spin = QtWidgets.QSpinBox()
            size_spin.setRange(1, 9999)
            size_spin.setValue(3)
            size_spin.setEnabled(False)
            check.toggled.connect(size_spin.setEnabled)
            self.dim_checks[dim] = check
            self.size_spins[dim] = size_spin
            dim_layout.addWidget(check, row, 0)
            dim_layout.addWidget(size_spin, row, 1)
        self.layout_.addRow(dim_group)

        self.mode_combo = QtWidgets.QComboBox()
        for mode in ("nearest", "reflect", "constant", "mirror", "wrap"):
            self.mode_combo.addItem(mode.title(), userData=mode)
        self.layout_.addRow("Boundary Mode", self.mode_combo)

        self.cval_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            exact_float=True,
        )
        self.cval_spin.setRange(-1e12, 1e12)
        self.cval_spin.setDecimals(8)
        self.layout_.addRow("Constant Value", self.cval_spin)

    @property
    def _sizes(self) -> dict[Hashable, int]:
        return {
            dim: self.size_spins[dim].value()
            for dim, check in self.dim_checks.items()
            if check.isChecked()
        }

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        operation = self.filter_operation()
        if operation is None:
            return data
        return operation.apply(data)

    def filter_operation(self) -> ToolProvenanceOperation | None:
        if not (sizes := self._sizes):
            return None
        return BoxcarFilterOperation(
            size=sizes,
            mode=typing.cast(
                "typing.Literal['reflect', 'constant', 'nearest', 'mirror', 'wrap']",
                self.mode_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
            ),
            cval=self.cval_spin.value(),
        )

    def restore_filter_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(operation, BoxcarFilterOperation):
            return
        for check in self.dim_checks.values():
            check.setChecked(False)
        for dim, size in operation.size.items():
            if dim not in self.dim_checks:
                raise ValueError(f"Dimension {dim!r} is not available")
            self.dim_checks[dim].setChecked(True)
            _set_spin_value(
                self.size_spins[dim],
                size,
                label=f"Boxcar size for {dim!r}",
            )
        if not _set_combo_data(self.mode_combo, operation.mode):
            raise ValueError(f"Boxcar mode {operation.mode!r} is not available")
        _set_spin_value(
            self.cval_spin,
            operation.cval,
            label="Boxcar constant value",
        )


class SwapDimsDialog(DataTransformDialog):
    title = "Swap Dimensions"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (SwapDimsOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )
        public_dims = tuple(
            dim.removesuffix("_idx")
            if isinstance(dim, str) and dim.endswith("_idx")
            else dim
            for dim in self.slicer_area.data.dims
        )
        if tuple(self._source_data.dims) != public_dims and set(
            self._source_data.dims
        ) == set(public_dims):
            self._source_data = self._source_data.transpose(*public_dims)

        dim_group = QtWidgets.QGroupBox("Dimensions")
        dim_layout = QtWidgets.QGridLayout()
        dim_group.setLayout(dim_layout)

        dim_layout.addWidget(QtWidgets.QLabel("Dimension"), 0, 0)
        dim_layout.addWidget(QtWidgets.QLabel("Target"), 0, 1)

        self.source_labels: dict[Hashable, QtWidgets.QLabel] = {}
        self.target_combos: dict[Hashable, QtWidgets.QComboBox] = {}
        self.target_options: dict[Hashable, tuple[Hashable, ...]] = {}

        for row, dim in enumerate(self._source_data.dims, start=1):
            label = QtWidgets.QLabel(str(dim))
            combo = QtWidgets.QComboBox()

            targets = [dim]
            targets.extend(
                name
                for name, coord in self._source_data.coords.items()
                if name != dim and coord.ndim == 1 and coord.dims == (dim,)
            )
            self.target_options[dim] = tuple(targets)

            for target in self.target_options[dim]:
                combo.addItem(str(target), userData=target)

            if len(self.target_options[dim]) == 1:
                tooltip = (
                    "No compatible 1D coordinates are available for this dimension."
                )
                label.setToolTip(tooltip)
                combo.setToolTip(tooltip)
                combo.setEnabled(False)

            self.source_labels[dim] = label
            self.target_combos[dim] = combo

            dim_layout.addWidget(label, row, 0)
            dim_layout.addWidget(combo, row, 1)

        self.layout_.addRow(dim_group)

    @property
    def _swap_mapping(self) -> dict[Hashable, Hashable]:
        mapping: dict[Hashable, Hashable] = {}
        for dim, combo in self.target_combos.items():
            if not combo.isEnabled():
                continue
            target = typing.cast(
                "Hashable | None",
                combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
            )
            if target is not None and target != dim:
                mapping[dim] = target
        return mapping

    def _validate(self) -> QtWidgets.QDialog.DialogCode:
        if not any(combo.isEnabled() for combo in self.target_combos.values()):
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing to Swap",
                "No compatible 1D coordinates are available for any dimension.",
            )
            return QtWidgets.QDialog.DialogCode.Rejected
        return super()._validate()

    @QtCore.Slot()
    def accept(self) -> None:
        if not self._swap_mapping:
            QtWidgets.QMessageBox.warning(
                self,
                "No Dimensions Changed",
                "Choose at least one dimension to swap.",
            )
            return
        super().accept()

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        if not self._swap_mapping:
            raise ValueError("No dimensions changed")
        return SwapDimsOperation(mapping=self._swap_mapping)

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            SwapDimsOperation,
        ):
            return
        remaining = set(operation.mapping)
        for dim, combo in self.target_combos.items():
            target = operation.mapping.get(dim, dim)
            if not _set_combo_data(combo, target):
                raise ValueError(f"Target {target!r} is not available for {dim!r}")
            remaining.discard(dim)
        if remaining:
            unavailable = sorted(str(dim) for dim in remaining)
            raise ValueError(f"Dimensions {unavailable!r} are not available")


class RenameDimsCoordsDialog(DataTransformDialog):
    title = "Rename Coordinates and Dimensions"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (RenameDimsCoordsOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.utils.array._restore_nonuniform_dims(
            self.slicer_area.data
        )
        public_dims = tuple(
            dim.removesuffix("_idx")
            if isinstance(dim, str) and dim.endswith("_idx")
            else dim
            for dim in self.slicer_area.data.dims
        )
        if tuple(self._source_data.dims) != public_dims and set(
            self._source_data.dims
        ) == set(public_dims):
            self._source_data = self._source_data.transpose(*public_dims)

        self._rename_sources: list[Hashable] = [*self._source_data.dims]
        self._rename_sources.extend(
            name
            for name in self._source_data.coords
            if name not in self._source_data.dims
        )

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Current", "New Name"])
        self.table.setRowCount(len(self._rename_sources))
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSortingEnabled(False)

        readonly_flags = (
            QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        for row, name in enumerate(self._rename_sources):
            current_item = QtWidgets.QTableWidgetItem(str(name))
            current_item.setFlags(readonly_flags)
            current_item.setData(QtCore.Qt.ItemDataRole.UserRole, name)

            new_item = QtWidgets.QTableWidgetItem(str(name))

            self.table.setItem(row, 0, current_item)
            self.table.setItem(row, 1, new_item)

        header = typing.cast("QtWidgets.QHeaderView", self.table.horizontalHeader())
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.resizeRowsToContents()
        self.layout_.addRow(self.table)

    def _new_name_for_row(self, row: int) -> str:
        item = self.table.item(row, 1)
        if item is None:
            return ""
        return item.text().strip()

    @property
    def _rename_mapping(self) -> dict[Hashable, str]:
        mapping: dict[Hashable, str] = {}
        for row, name in enumerate(self._rename_sources):
            new_name = self._new_name_for_row(row)
            if new_name != str(name):
                mapping[name] = new_name
        return mapping

    @property
    def _final_names(self) -> tuple[Hashable, ...]:
        mapping = self._rename_mapping
        return tuple(mapping.get(name, name) for name in self._rename_sources)

    def _duplicate_final_names(self) -> list[Hashable]:
        seen: set[Hashable] = set()
        duplicates: list[Hashable] = []
        for name in self._final_names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        return duplicates

    @QtCore.Slot()
    def accept(self) -> None:
        mapping = self._rename_mapping
        if not mapping:
            QtWidgets.QMessageBox.warning(
                self,
                "No Names Changed",
                "Edit at least one coordinate or dimension name.",
            )
            return

        empty_names = [
            str(name) for name, new_name in mapping.items() if new_name == ""
        ]
        if empty_names:
            QtWidgets.QMessageBox.warning(
                self,
                "Empty Name",
                "Names cannot be empty.",
            )
            return

        duplicates = self._duplicate_final_names()
        if duplicates:
            duplicate_text = ", ".join(str(name) for name in duplicates)
            QtWidgets.QMessageBox.warning(
                self,
                "Duplicate Names",
                f"Names must be unique after renaming: {duplicate_text}.",
            )
            return

        super().accept()

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        if not self._rename_mapping:
            raise ValueError("No names changed")
        return RenameDimsCoordsOperation(
            mapping=typing.cast("dict[Hashable, Hashable]", self._rename_mapping)
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            RenameDimsCoordsOperation,
        ):
            return
        remaining = set(operation.mapping)
        for row, name in enumerate(self._rename_sources):
            item = self.table.item(row, 1)
            if item is not None:
                item.setText(str(operation.mapping.get(name, name)))
            remaining.discard(name)
        if remaining:
            unavailable = sorted(str(name) for name in remaining)
            raise ValueError(f"Names {unavailable!r} are not available")


class AssignCoordsDialog(DataTransformDialog):
    title = "Coordinate Editor"
    operation_types = (
        AffineCoordOperation,
        AssignCoordsOperation,
        AssignScalarCoordOperation,
        AssignCoord1DOperation,
    )

    def setup_widgets(self) -> None:
        self._mode_tabs = QtWidgets.QTabWidget(self)
        self.layout_.addRow(self._mode_tabs)

        existing_widget = QtWidgets.QWidget(self)
        existing_layout = QtWidgets.QVBoxLayout(existing_widget)

        existing_coord_names: list[str] = [str(d) for d in self.slicer_area.data.dims]
        existing_coord_names.extend(
            str(k)
            for k in self.slicer_area.data.coords
            if k not in self.slicer_area.data.dims
        )
        self._coord_combo = QtWidgets.QComboBox()
        self._coord_combo.addItems(existing_coord_names)
        existing_layout.addWidget(self._coord_combo)

        self._coord_combo.currentTextChanged.connect(self._coord_selection_changed)

        self.coord_widget = CoordinateEditorWidget(np.array([0, 1]))
        self._coord_selection_changed()
        existing_layout.addWidget(self.coord_widget)
        self._mode_tabs.addTab(existing_widget, "Edit Existing")

        add_widget = QtWidgets.QWidget(self)
        add_layout = QtWidgets.QFormLayout(add_widget)
        add_widget.setLayout(add_layout)

        self._add_name_edit = QtWidgets.QLineEdit()
        add_layout.addRow("Name", self._add_name_edit)

        self._add_kind_combo = QtWidgets.QComboBox()
        self._add_kind_combo.addItems(["Scalar", "1D Along Coordinate"])
        self._add_kind_combo.currentTextChanged.connect(self._sync_add_widgets)
        add_layout.addRow("Kind", self._add_kind_combo)

        self._add_ref_combo = QtWidgets.QComboBox()
        self._populate_add_reference_combo()
        self._add_ref_combo.currentIndexChanged.connect(self._add_reference_changed)
        add_layout.addRow("Reference", self._add_ref_combo)

        self._add_value_mode_combo = QtWidgets.QComboBox()
        self._add_value_mode_combo.addItems(["Numeric Values", "Python Literal"])
        self._add_value_mode_combo.currentTextChanged.connect(self._sync_add_widgets)
        add_layout.addRow("Value Mode", self._add_value_mode_combo)

        self._add_value_stack = QtWidgets.QStackedWidget()
        self._add_coord_widget = CoordinateEditorWidget(np.array([0.0, 1.0]))
        self._add_value_stack.addWidget(self._add_coord_widget)

        literal_widget = QtWidgets.QWidget(self)
        literal_layout = QtWidgets.QFormLayout(literal_widget)
        literal_widget.setLayout(literal_layout)
        self._add_literal_edit = QtWidgets.QLineEdit("0.0")
        literal_layout.addRow("Value", self._add_literal_edit)
        self._add_value_stack.addWidget(literal_widget)
        add_layout.addRow(self._add_value_stack)

        self._mode_tabs.addTab(add_widget, "Add Coordinate")
        self._add_reference_changed()
        self._sync_add_widgets()

    @property
    def current_coord_name(self) -> str:
        """Get the name of the currently selected coordinate."""
        return self._coord_combo.currentText()

    @QtCore.Slot()
    def _coord_selection_changed(self) -> None:
        self.coord_widget.set_old_coord(
            self.slicer_area.data[self.current_coord_name].values
        )

    def _populate_add_reference_combo(self) -> None:
        self._add_ref_combo.clear()
        added_dims: set[Hashable] = set()
        for dim in self.slicer_area.data.dims:
            if dim in self.slicer_area.data.coords:
                coord = self.slicer_area.data.coords[dim]
                if coord.ndim == 1:
                    self._add_ref_combo.addItem(f"{dim} (dimension)", userData=dim)
                    added_dims.add(dim)
        for coord_name, coord in self.slicer_area.data.coords.items():
            if coord_name in self.slicer_area.data.dims or coord.ndim != 1:
                continue
            dim = coord.dims[0]
            if dim in self.slicer_area.data.dims:
                self._add_ref_combo.addItem(
                    f"{coord_name} ({dim})",
                    userData=dim,
                )
                added_dims.add(dim)
        for dim in self.slicer_area.data.dims:
            if dim not in added_dims:
                self._add_ref_combo.addItem(str(dim), userData=dim)

    @QtCore.Slot()
    @QtCore.Slot(str)
    def _sync_add_widgets(self, _text: str | None = None) -> None:
        is_scalar = self._add_kind_combo.currentText() == "Scalar"
        self._add_ref_combo.setEnabled(not is_scalar)
        self._add_value_mode_combo.setEnabled(not is_scalar)
        use_literal = (
            is_scalar or self._add_value_mode_combo.currentText() == "Python Literal"
        )
        self._add_value_stack.setCurrentIndex(1 if use_literal else 0)

    @QtCore.Slot()
    @QtCore.Slot(int)
    def _add_reference_changed(self, _index: int | None = None) -> None:
        if self._add_ref_combo.count() == 0:
            return
        dim = typing.cast(
            "Hashable",
            self._add_ref_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )
        label = self._add_ref_combo.currentText()
        coord_name = label.split(" (", 1)[0]
        if coord_name in self.slicer_area.data.coords:
            values = self.slicer_area.data.coords[coord_name].values
        else:
            values = np.arange(self.slicer_area.data.sizes[dim], dtype=float)
        self._add_coord_widget.set_old_coord(values)

    def _add_coord_values(self) -> tuple[typing.Any, Hashable | None]:
        if self._add_kind_combo.currentText() == "Scalar":
            value = ast.literal_eval(self._add_literal_edit.text().strip())
            if isinstance(value, (list, tuple, dict)):
                raise ValueError("Scalar coordinates must use a scalar value.")
            return value, None

        dim = typing.cast(
            "Hashable",
            self._add_ref_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )
        if self._add_value_mode_combo.currentText() == "Numeric Values":
            values = (
                self._add_coord_widget.affine_coord
                if self._add_coord_widget.use_affine_transform
                else self._add_coord_widget.new_coord
            )
        else:
            values = np.asarray(ast.literal_eval(self._add_literal_edit.text().strip()))

        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("1D coordinates must use a one-dimensional value.")
        if values.size != self.slicer_area.data.sizes[dim]:
            raise ValueError(
                f"Coordinate length {values.size} does not match dimension "
                f"{dim!r} length {self.slicer_area.data.sizes[dim]}."
            )
        return values, dim

    @QtCore.Slot()
    def accept(self) -> None:
        if self._mode_tabs.currentIndex() != 1:
            super().accept()
            return

        name = self._add_name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(
                self,
                "Empty Name",
                "Coordinate names cannot be empty.",
            )
            return

        if name in self.slicer_area.data.dims or name in self.slicer_area.data.coords:
            QtWidgets.QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A coordinate or dimension named {name!r} already exists.",
            )
            return

        try:
            self._add_coord_values()
        except Exception as exc:
            _show_warning_with_traceback(
                self,
                "Invalid Coordinate Value",
                str(exc),
            )
            return

        super().accept()

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        if self._mode_tabs.currentIndex() == 1:
            values, dim = self._add_coord_values()
            name = self._add_name_edit.text().strip()
            if dim is None:
                return AssignScalarCoordOperation(
                    coord_name=name,
                    value=values,
                )
            return AssignCoord1DOperation(
                coord_name=name,
                dim=dim,
                values=values,
            )
        if self.coord_widget.use_affine_transform:
            return AffineCoordOperation(
                coord_name=self.current_coord_name,
                scale=self.coord_widget.affine_scale,
                offset=self.coord_widget.affine_offset,
            )
        return AssignCoordsOperation(
            coord_name=self.current_coord_name,
            values=self.coord_widget.new_coord,
        )

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if isinstance(
            operation,
            AffineCoordOperation,
        ):
            if not _set_combo_text(self._coord_combo, operation.coord_name):
                raise ValueError(
                    f"Coordinate {operation.coord_name!r} is not available"
                )
            self._mode_tabs.setCurrentIndex(0)
            self.coord_widget.edit_mode_tabs.setCurrentIndex(1)
            self.coord_widget.scale_spin.setValue(float(operation.scale))
            self.coord_widget.offset_spin.setValue(float(operation.offset))
            self.coord_widget.update_affine_preview()
            return
        if isinstance(
            operation,
            AssignCoordsOperation,
        ):
            if not _set_combo_text(self._coord_combo, operation.coord_name):
                raise ValueError(
                    f"Coordinate {operation.coord_name!r} is not available"
                )
            self._mode_tabs.setCurrentIndex(0)
            self.coord_widget.edit_mode_tabs.setCurrentIndex(0)
            self.coord_widget._set_table_values(operation.decoded_values)
            return
        if isinstance(
            operation,
            AssignScalarCoordOperation,
        ):
            self._mode_tabs.setCurrentIndex(1)
            self._add_name_edit.setText(str(operation.coord_name))
            self._add_kind_combo.setCurrentText("Scalar")
            self._add_literal_edit.setText(
                _provenance_value_code(operation.decoded_value)
            )
            self._sync_add_widgets()
            return
        if isinstance(
            operation,
            AssignCoord1DOperation,
        ):
            self._mode_tabs.setCurrentIndex(1)
            self._add_name_edit.setText(str(operation.coord_name))
            self._add_kind_combo.setCurrentText("1D Along Coordinate")
            if not _set_combo_data(self._add_ref_combo, operation.dim):
                raise ValueError(f"Dimension {operation.dim!r} is not available")
            self._add_value_mode_combo.setCurrentText("Numeric Values")
            self._add_reference_changed()
            self._add_coord_widget._set_table_values(operation.decoded_values)
            self._sync_add_widgets()


_ATTR_TYPE_NAMES: tuple[str, ...] = (
    "String",
    "Int",
    "Float",
    "Bool",
    "Python literal",
)


def _attr_display_value(value: typing.Any) -> tuple[str, str]:
    value = erlab.utils.misc._convert_to_native(value)
    text = (
        value
        if isinstance(value, str)
        else erlab.interactive.utils._parse_single_arg(value)
    )
    if isinstance(value, bool):
        return "Bool", text
    if isinstance(value, int):
        return "Int", text
    if isinstance(value, float):
        return "Float", text
    if isinstance(value, str):
        return "String", text
    return "Python literal", text


def _parse_attr_value(type_name: str, text: str) -> typing.Any:
    stripped = text.strip()
    match type_name:
        case "String":
            return text
        case "Int":
            return int(stripped)
        case "Float":
            return float(stripped)
        case "Bool":
            lowered = stripped.casefold()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
            raise ValueError("Bool attributes must be True or False.")
        case "Python literal":
            return ast.literal_eval(stripped)
        case _:
            raise ValueError(f"Unknown attribute type: {type_name}.")


def _attr_values_equal(left: typing.Any, right: typing.Any) -> bool:
    try:
        equal = left == right
    except Exception:
        return False
    if isinstance(equal, np.ndarray):
        return bool(np.all(equal))
    return bool(equal)


class AssignAttrsDialog(DataTransformDialog):
    title = "Attribute Editor"
    operation_types = (AssignAttrsOperation,)

    def setup_widgets(self) -> None:
        self._original_attrs = dict(self.slicer_area.data.attrs)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Value"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSortingEnabled(False)
        self.layout_.addRow(self.table)

        self.add_attr_button = QtWidgets.QPushButton("Add Attribute")
        self.add_attr_button.clicked.connect(self._add_empty_row)
        self.layout_.addRow(self.add_attr_button)

        for key, value in self._original_attrs.items():
            self._add_attr_row(key, value, editable_name=False)

        header = typing.cast("QtWidgets.QHeaderView", self.table.horizontalHeader())
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.resizeRowsToContents()

    def _add_attr_row(
        self,
        key: Hashable | str,
        value: typing.Any,
        *,
        editable_name: bool,
    ) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        key_item = QtWidgets.QTableWidgetItem("" if editable_name else str(key))
        key_item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
        if not editable_name:
            key_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
        self.table.setItem(row, 0, key_item)

        type_name, value_text = _attr_display_value(value)
        type_combo = QtWidgets.QComboBox()
        type_combo.addItems(_ATTR_TYPE_NAMES)
        type_combo.setCurrentText(type_name)
        self.table.setCellWidget(row, 1, type_combo)
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(value_text))

    @QtCore.Slot()
    def _add_empty_row(self) -> None:
        self._add_attr_row("", "", editable_name=True)

    def _row_key(self, row: int) -> Hashable:
        item = self.table.item(row, 0)
        if item is None:
            return ""
        original = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if original in self._original_attrs:
            return typing.cast("Hashable", original)
        return item.text().strip()

    def _row_type(self, row: int) -> str:
        combo = typing.cast("QtWidgets.QComboBox", self.table.cellWidget(row, 1))
        return combo.currentText()

    def _row_text(self, row: int) -> str:
        item = self.table.item(row, 2)
        return "" if item is None else item.text()

    def _attrs_from_table(self) -> dict[Hashable, typing.Any]:
        attrs: dict[Hashable, typing.Any] = {}
        for row in range(self.table.rowCount()):
            key = self._row_key(row)
            if key == "":
                raise ValueError("Attribute names cannot be empty.")
            attrs[key] = _parse_attr_value(self._row_type(row), self._row_text(row))
        return attrs

    @property
    def _changed_attrs(self) -> dict[Hashable, typing.Any]:
        attrs = self._attrs_from_table()
        return {
            key: value
            for key, value in attrs.items()
            if key not in self._original_attrs
            or not _attr_values_equal(value, self._original_attrs[key])
        }

    def _duplicate_names(self) -> list[Hashable]:
        seen: set[Hashable] = set()
        duplicates: list[Hashable] = []
        for row in range(self.table.rowCount()):
            key = self._row_key(row)
            if key in seen and key not in duplicates:
                duplicates.append(key)
            seen.add(key)
        return duplicates

    @QtCore.Slot()
    def accept(self) -> None:
        try:
            self._attrs_from_table()
        except Exception as exc:
            _show_warning_with_traceback(
                self,
                "Invalid Attribute Value",
                str(exc),
            )
            return

        duplicates = self._duplicate_names()
        if duplicates:
            duplicate_text = ", ".join(str(name) for name in duplicates)
            QtWidgets.QMessageBox.warning(
                self,
                "Duplicate Names",
                f"Attribute names must be unique: {duplicate_text}.",
            )
            return

        if not self._changed_attrs:
            QtWidgets.QMessageBox.warning(
                self,
                "No Attributes Changed",
                "Edit at least one attribute or add a new one.",
            )
            return

        super().accept()

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        return AssignAttrsOperation(attrs=self._changed_attrs)

    def restore_transform_operation(
        self,
        operation: ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            AssignAttrsOperation,
        ):
            return
        row_by_key = {self._row_key(row): row for row in range(self.table.rowCount())}
        for key, value in operation.attrs.items():
            row = row_by_key.get(key)
            if row is None:
                self._add_attr_row(key, value, editable_name=True)
                row = self.table.rowCount() - 1
                key_item = self.table.item(row, 0)
                if key_item is not None:
                    key_item.setText(str(key))
            type_name, value_text = _attr_display_value(value)
            type_combo = typing.cast(
                "QtWidgets.QComboBox", self.table.cellWidget(row, 1)
            )
            if not _set_combo_text(type_combo, type_name):
                raise ValueError(f"Attribute type {type_name!r} is not available")
            value_item = self.table.item(row, 2)
            if value_item is not None:
                value_item.setText(value_text)


class ROIPathDialog(DataTransformDialog):
    title = "Slice Along ROI Path"
    enable_copy = True
    apply_on_nonuniform_data = True

    def __init__(self, roi: ItoolPolyLineROI) -> None:
        self.roi = roi
        super().__init__(self.roi.plot_item.slicer_area)

    def setup_widgets(self) -> None:
        group = QtWidgets.QGroupBox()
        layout = QtWidgets.QFormLayout()
        group.setLayout(layout)

        decimals = max(
            self.array_slicer.get_significant(ax, uniform=False)
            for ax in self.roi.plot_item.display_axis
        )
        default_step = min(
            self.roi.slicer_area.array_slicer.incs[ax]
            for ax in self.roi.plot_item.display_axis
        )  # Reasonable default step size

        # TODO: add vertice customization

        self._step_spin = pg.SpinBox()
        self._step_spin.setDecimals(decimals)
        self._step_spin.setSingleStep(10 ** (-decimals))
        self._step_spin.setMinimum(10 ** (-decimals - 1))
        self._step_spin.setOpts(compactHeight=False)
        self._step_spin.setValue(round(default_step, decimals))

        layout.addRow("Step Size", self._step_spin)

        self._dim_name_line = QtWidgets.QLineEdit()
        self._dim_name_line.setText("path")
        layout.addRow("New Dim Name", self._dim_name_line)

        self.layout_.addRow(group)

    @property
    def _params(self) -> dict[str, typing.Any]:
        vert_dict = self.roi._get_vertices()

        if self.roi.closed:
            for k, v in dict(vert_dict).items():
                vert_dict[k] = [*v, v[0]]  # Close the path

        return {
            "vertices": vert_dict,
            "step_size": self._step_spin.value(),
            "dim_name": self._dim_name_line.text().strip(),
        }

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        return SliceAlongPathOperation(**self._params)


class ROIMaskDialog(DataTransformDialog):
    title = "Mask with ROI"
    enable_copy = True
    apply_on_nonuniform_data = True

    def __init__(self, roi: ItoolPolyLineROI) -> None:
        self.roi = roi
        super().__init__(self.roi.plot_item.slicer_area)

    def setup_widgets(self) -> None:
        group = QtWidgets.QGroupBox()
        layout = QtWidgets.QFormLayout()
        group.setLayout(layout)

        self._invert_check = QtWidgets.QCheckBox("Invert Mask")
        layout.addRow(self._invert_check)

        self._drop_check = QtWidgets.QCheckBox("Drop Masked Values")
        layout.addRow(self._drop_check)

        self.layout_.addRow(group)

    @property
    def _params(self) -> dict[str, typing.Any]:
        vert_dict = self.roi._get_vertices()
        return {
            "vertices": np.column_stack(tuple(vert_dict.values())),
            "dims": tuple(vert_dict.keys()),
            "invert": self._invert_check.isChecked(),
            "drop": self._drop_check.isChecked(),
        }

    def source_transform_operation(
        self,
    ) -> ToolProvenanceOperation:
        return MaskWithPolygonOperation(**self._params)
