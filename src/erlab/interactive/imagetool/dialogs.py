"""Dialogs for data manipulation found in the menu bar."""

from __future__ import annotations

import ast
import contextlib
import math
import typing
import weakref

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.interactive.imagetool import provenance
from erlab.interactive.imagetool._dialog_widgets import (
    CoordinateEditorWidget,
    CoordinateGridWidget,
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
    "LeadingEdgeDialog",
    "NormalizeDialog",
    "ROIMaskDialog",
    "ROIPathDialog",
    "RenameDimsCoordsDialog",
    "RotationDialog",
    "SelectionDialog",
    "SortByDialog",
    "SwapDimsDialog",
    "SymmetrizeDialog",
    "SymmetrizeNfoldDialog",
    "ThinDialog",
]

if typing.TYPE_CHECKING:
    from collections.abc import Hashable

    import xarray as xr

    from erlab.interactive.imagetool.manager import ImageToolManager
    from erlab.interactive.imagetool.plot_items import ItoolPolyLineROI
    from erlab.interactive.imagetool.slicer import ArraySlicer
    from erlab.interactive.imagetool.viewer import ImageSlicerArea
    from erlab.interactive.imagetool.viewer_state import ColorMapState

_GAUSSIAN_FWHM_FACTOR: float = 2 * math.sqrt(2 * math.log(2))


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
        tuple[type[provenance.ToolProvenanceOperation], ...]
    ] = ()
    """Operation classes this dialog can emit directly."""

    _sigCodeCopied = QtCore.Signal(str)

    def __init__(
        self,
        slicer_area: ImageSlicerArea,
        *,
        batch_manager: ImageToolManager | None = None,
    ) -> None:
        super().__init__(slicer_area)
        if self.title is not None:
            self.setWindowTitle(self.title)

        self.slicer_area = slicer_area
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
    def slicer_area(self, value: ImageSlicerArea) -> None:
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
        return self.slicer_area.watched_data_name or "data"


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
        slicer_area: ImageSlicerArea,
        *,
        batch_manager: ImageToolManager | None = None,
    ) -> None:
        super().__init__(slicer_area, batch_manager=batch_manager)
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
    ) -> list[provenance.ToolProvenanceOperation]:
        operation = self.source_transform_operation()
        return [] if operation is None else [operation]

    def source_transform_operation(
        self,
    ) -> provenance.ToolProvenanceOperation | None:
        return None

    def source_spec_for_data(
        self,
        data: xr.DataArray,
        new_name: str | None = None,
    ) -> provenance.ToolProvenanceSpec:
        del new_name
        operations = self.source_operations()
        builder = (
            provenance.public_data
            if self.apply_on_nonuniform_data
            else provenance.full_data
        )
        if not self.apply_on_nonuniform_data and any(
            str(dim).endswith("_idx") and str(dim).removesuffix("_idx") in data.coords
            for dim in data.dims
        ):
            operations.append(provenance.RestoreNonuniformDimsOperation())
        return builder(*operations)

    def source_spec(self, new_name: str | None = None) -> provenance.ToolProvenanceSpec:
        return self.source_spec_for_data(self.slicer_area.data, new_name)

    def _detached_provenance_spec(
        self,
        parent_provenance: provenance.ToolProvenanceSpec | None,
        source_spec: provenance.ToolProvenanceSpec,
        new_name: str,
    ) -> provenance.ToolProvenanceSpec:
        return self._compose_transform_provenance(
            parent_provenance,
            source_spec,
            new_name,
        )

    @staticmethod
    def _compose_transform_provenance(
        base_spec: provenance.ToolProvenanceSpec | None,
        source_spec: provenance.ToolProvenanceSpec,
        new_name: str,
    ) -> provenance.ToolProvenanceSpec:
        del new_name
        if base_spec is None:
            return source_spec
        with contextlib.suppress(TypeError):
            live_parent = provenance.require_live_source_spec(base_spec)
            if live_parent is not None:
                return live_parent.append_replacement_operations(
                    *source_spec.operations
                )
        composed = provenance.compose_full_provenance(
            base_spec,
            source_spec,
        )
        if composed is None:
            raise RuntimeError("Could not compose ImageTool transform provenance.")
        return composed

    def _compose_replace_source_spec(
        self,
        existing_spec: provenance.ToolProvenanceSpec,
        new_name: str,
    ) -> provenance.ToolProvenanceSpec:
        return self._compose_transform_provenance(
            existing_spec,
            self.source_spec(new_name),
            new_name,
        )

    def _rewrite_target_provenance(
        self,
        target: int | str,
        new_name: str,
        fallback_spec: provenance.ToolProvenanceSpec | None,
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
        if displayed_provenance is not None:
            node.set_detached_provenance(
                self._compose_replace_source_spec(displayed_provenance, new_name)
            )
            return True
        if fallback_spec is not None:
            node.set_detached_provenance(fallback_spec)
            return True
        return False

    def _set_current_tool_provenance(
        self,
        provenance_spec: provenance.ToolProvenanceSpec | None,
    ) -> None:
        parent = self.slicer_area.parent()
        if parent is not None and hasattr(parent, "set_provenance_spec"):
            typing.cast("typing.Any", parent).set_provenance_spec(provenance_spec)

    def _apply_source_transform(self, data: xr.DataArray) -> xr.DataArray:
        operation = self.source_transform_operation()
        if operation is None:
            return data
        return operation.apply(data, parent_data=data)

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        return self._apply_source_transform(data)

    def make_code(self) -> str:
        try:
            return provenance.operations_expression_code(
                self.source_operations(),
                self._copy_data_name(),
            )
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

        if self.keep_colors:
            if slicer_area is None:
                slicer_area = self.slicer_area
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
                processed = self.process_data(
                    erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
                        self.slicer_area.data
                    )
                )
            else:
                processed = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
                    self.process_data(self.slicer_area.data)
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
            nested_provenance_spec = provenance.compose_full_provenance(
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
                        target, new_name, detached_provenance_spec
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
                            manager.add_imagetool(
                                tool,
                                activate=True,
                                provenance_spec=detached_provenance_spec,
                            )
                    else:
                        tool = typing.cast(
                            "erlab.interactive.imagetool.ImageTool | None",
                            erlab.interactive.itool(**itool_kw),
                        )
                        if tool is not None:
                            tool.set_provenance_spec(detached_provenance_spec)

            del processed

        except Exception:
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
    ) -> None:
        super().__init__(slicer_area, batch_manager=batch_manager)
        self._previewed: bool = False
        self._starting_applied_func = self.slicer_area._applied_func
        self._starting_filter_operation = (
            self.slicer_area._accepted_filter_provenance_operation
        )

        if self.enable_preview and not self.is_batch_mode:
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
        operation: provenance.ToolProvenanceOperation,
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
    ) -> provenance.ToolProvenanceOperation | None:
        return None

    def filter_operations(
        self,
    ) -> list[provenance.ToolProvenanceOperation]:
        operation = self.filter_operation()
        return [] if operation is None else [operation]

    def make_code(self) -> str:
        try:
            return provenance.operations_expression_code(
                self.filter_operations(),
                self._copy_data_name(),
            )
        except Exception:
            return ""


class RotationDialog(DataTransformDialog):
    enable_copy = True
    operation_types = (provenance.RotateOperation,)

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
    ) -> provenance.ToolProvenanceOperation:
        return provenance.RotateOperation(**self._rotate_params)


class AggregateDialog(DataTransformDialog):
    title = "Aggregate Over Dimensions"
    enable_copy = True
    operation_types = (provenance.QSelAggregationOperation,)

    _REDUCERS: typing.ClassVar[dict[str, str]] = {
        "mean": "Mean",
        "min": "Minimum",
        "max": "Maximum",
        "sum": "Sum",
    }

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
        for reducer, label in self._REDUCERS.items():
            self.reducer_combo.addItem(label, userData=reducer)
        self.layout_.addRow("Reducer", self.reducer_combo)

    @property
    def _target_dims(self) -> tuple[Hashable, ...]:
        return tuple(k for k, v in self.dim_checks.items() if v.isChecked())

    @property
    def _reducer(self) -> typing.Literal["mean", "min", "max", "sum"]:
        return typing.cast(
            "typing.Literal['mean', 'min', 'max', 'sum']",
            self.reducer_combo.currentData(QtCore.Qt.ItemDataRole.UserRole),
        )

    def source_transform_operation(
        self,
    ) -> provenance.ToolProvenanceOperation:
        if not self._target_dims:
            raise ValueError("No dimensions selected")
        return provenance.QSelAggregationOperation(
            dims=self._target_dims,
            func=self._reducer,
        )

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
        coord = np.asarray(data[dim].values, dtype=float)
        self._coord_ascending = coord[0] <= coord[-1]
        current_index = dialog.array_slicer.get_index(
            dialog.slicer_area.current_cursor, axis
        )
        current_value = float(
            dialog.array_slicer.get_value(
                dialog.slicer_area.current_cursor, axis, uniform=False
            )
        )
        stop_index = min(current_index + 1, data.sizes[dim] - 1)
        stop_value = float(coord[stop_index])
        bin_value = dialog.array_slicer.get_bin_values(
            dialog.slicer_area.current_cursor
        )[axis]
        is_binned = dialog.array_slicer.get_binned(dialog.slicer_area.current_cursor)[
            axis
        ]

        self.use_check = QtWidgets.QCheckBox(str(dim))
        self.use_check.setObjectName(f"selection_use_{axis}")
        self.use_check.setChecked(active)

        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.setObjectName(f"selection_method_{axis}")
        self.method_combo.setToolTip(
            "Choose how this dimension is selected: qsel selects the nearest "
            "coordinate value, sel uses coordinate labels, and isel uses "
            "integer indices."
        )
        for method in ("qsel", "sel", "isel"):
            self.method_combo.addItem(method, method)

        self.kind_combo = QtWidgets.QComboBox()
        self.kind_combo.setObjectName(f"selection_kind_{axis}")
        self.kind_combo.addItem("Point", "point")
        self.kind_combo.addItem("Range", "range")

        self.index_start_spin = erlab.interactive.utils.BetterSpinBox(
            integer=True,
            compact=False,
            minimum=0,
            maximum=data.sizes[dim] - 1,
            value=current_index,
        )
        self.index_stop_spin = erlab.interactive.utils.BetterSpinBox(
            integer=True,
            compact=False,
            minimum=0,
            maximum=data.sizes[dim],
            value=min(current_index + 1, data.sizes[dim]),
        )

        self.value_start_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            decimals=6,
            exact_float=True,
            significant=True,
            minimum=float(np.nanmin(coord)),
            maximum=float(np.nanmax(coord)),
            value=current_value,
        )
        self.value_stop_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            decimals=6,
            exact_float=True,
            significant=True,
            minimum=float(np.nanmin(coord)),
            maximum=float(np.nanmax(coord)),
            value=stop_value,
        )

        self.start_stack = QtWidgets.QStackedWidget()
        self.start_stack.addWidget(self.value_start_spin)
        self.start_stack.addWidget(self.index_start_spin)

        self.stop_stack = QtWidgets.QStackedWidget()
        self.stop_stack.addWidget(self.value_stop_spin)
        self.stop_stack.addWidget(self.index_stop_spin)

        self.width_widget = QtWidgets.QWidget()
        self.width_widget.setToolTip(
            "For point qsel selections, include nearby coordinate values within "
            "this width. Ignored for range selections and for sel/isel."
        )
        width_layout = QtWidgets.QHBoxLayout(self.width_widget)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.setSpacing(3)
        self.width_check = QtWidgets.QCheckBox()
        self.width_check.setObjectName(f"selection_width_enabled_{axis}")
        self.width_check.setToolTip(self.width_widget.toolTip())
        self.width_spin = erlab.interactive.utils.BetterSpinBox(
            compact=False,
            decimals=6,
            exact_float=True,
            significant=True,
            minimum=0.0,
            value=0.0 if bin_value is None else float(bin_value),
        )
        self.width_spin.setObjectName(f"selection_width_{axis}")
        self.width_spin.setToolTip(self.width_widget.toolTip())
        self.width_check.setChecked(is_binned and bin_value is not None)
        width_layout.addWidget(self.width_check)
        width_layout.addWidget(self.width_spin)

        widgets: tuple[QtWidgets.QWidget, ...] = (
            self.use_check,
            self.method_combo,
            self.kind_combo,
            self.start_stack,
            self.stop_stack,
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
        self.width_check.toggled.connect(self.sync_widgets)
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

        self.start_stack.setCurrentIndex(1 if is_index else 0)
        self.stop_stack.setCurrentIndex(1 if is_index else 0)
        self.stop_stack.setEnabled(is_range)
        self.width_widget.setEnabled(is_qsel and not is_range)
        self.width_spin.setEnabled(
            is_qsel and not is_range and self.width_check.isChecked()
        )

    def indexer(self) -> tuple[Hashable, typing.Any]:
        if self.method == "isel":
            start = int(self.index_start_spin.value())
            if self.kind == "point":
                return self.dim, start
            stop = int(self.index_stop_spin.value())
            return self.dim, slice(min(start, stop), max(start, stop))

        start = float(self.value_start_spin.value())
        if self.kind == "point":
            return self.dim, start

        stop = float(self.value_stop_spin.value())
        if self._coord_ascending:
            return self.dim, slice(min(start, stop), max(start, stop))
        return self.dim, slice(max(start, stop), min(start, stop))

    def qsel_width_indexer(self) -> tuple[str, float] | None:
        if (
            self.method != "qsel"
            or self.kind != "point"
            or not self.width_check.isChecked()
        ):
            return None
        width = float(self.width_spin.value())
        if not np.isfinite(width) or width <= 0.0:
            return None
        return f"{self.dim}_width", width


class SelectionDialog(DataTransformDialog):
    title = "Select Data"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (
        provenance.IselOperation,
        provenance.SelOperation,
        provenance.QSelOperation,
    )

    def setup_widgets(self) -> None:
        self.public_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
            self.slicer_area.data
        )

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
    ) -> list[provenance.ToolProvenanceOperation]:
        isel_kwargs, sel_kwargs, qsel_kwargs = self._selection_kwargs()
        operations: list[provenance.ToolProvenanceOperation] = []
        if isel_kwargs:
            operations.append(provenance.IselOperation(kwargs=isel_kwargs))
        if sel_kwargs:
            operations.append(provenance.SelOperation(kwargs=sel_kwargs))
        if qsel_kwargs:
            operations.append(provenance.QSelOperation(kwargs=qsel_kwargs))
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
    operation_types = (provenance.InterpolationOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
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
    ) -> provenance.InterpolationOperation:
        dim = self._selected_dim
        if dim is None:
            raise ValueError("No dimension selected")
        source_error = self._source_coord_error(dim)
        if source_error is not None:
            raise ValueError(source_error)
        return provenance.InterpolationOperation(
            dim=dim,
            values=self._target_values(),
            method=typing.cast(
                "typing.Literal['linear', 'nearest']",
                self.method_combo.currentText(),
            ),
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
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Target Coordinates",
                str(exc),
            )
            return

        super().accept()


class SortByDialog(DataTransformDialog):
    title = "Sort By"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (provenance.SortByOperation,)

    def setup_widgets(self) -> None:
        source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
            self.slicer_area.data
        )
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
    ) -> provenance.SortByOperation:
        sort_keys = self._sort_keys
        if not sort_keys:
            raise ValueError("No sort keys selected")
        return provenance.SortByOperation(
            variables=sort_keys,
            ascending=bool(
                self.ascending_combo.currentData(QtCore.Qt.ItemDataRole.UserRole)
            ),
        )

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
    operation_types = (provenance.LeadingEdgeOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
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
    ) -> provenance.LeadingEdgeOperation:
        dim = self._selected_dim
        if dim is None:
            raise ValueError("No dimension selected")
        source_error = self._source_coord_error(dim)
        if source_error is not None:
            raise ValueError(source_error)
        return provenance.LeadingEdgeOperation(
            fraction=float(self.fraction_spin.value()),
            dim=dim,
            direction=self._direction,
        )

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
    operation_types = (provenance.CoarsenOperation,)

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
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
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
    ) -> provenance.ToolProvenanceOperation:
        if not self._selected_windows:
            raise ValueError("No dimensions selected")
        return provenance.CoarsenOperation(
            dim=self._selected_windows,
            boundary=self.boundary_combo.currentText(),
            side=self.side_combo.currentText(),
            coord_func=self._coord_func,
            reducer=self._reducer,
        )

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
    operation_types = (provenance.ThinOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
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
    ) -> provenance.ToolProvenanceOperation:
        if self._use_global_mode:
            if self.global_spin.value() <= 1:
                raise ValueError("No thinning requested")
            return provenance.ThinOperation(
                mode="global", factor=self.global_spin.value()
            )
        if not self._effective_factors:
            raise ValueError("No thinning requested")
        return provenance.ThinOperation(mode="per_dim", factors=self._effective_factors)

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


class SymmetrizeDialog(DataTransformDialog):
    title = "Symmetrize"
    enable_copy = True
    operation_types = (provenance.SymmetrizeOperation,)

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
    ) -> provenance.ToolProvenanceOperation:
        return provenance.SymmetrizeOperation(**self._params)


class SymmetrizeNfoldDialog(DataTransformDialog):
    title = "Rotational Symmetrize"
    enable_copy = True
    operation_types = (provenance.SymmetrizeNfoldOperation,)

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
    ) -> provenance.ToolProvenanceOperation:
        return provenance.SymmetrizeNfoldOperation(**self._params)


class EdgeCorrectionDialog(DataTransformDialog):
    title = "Edge Correction"
    enable_copy = False

    def setup_widgets(self) -> None:
        self.shift_coord_check = QtWidgets.QCheckBox("Shift Coordinates")
        self.shift_coord_check.setChecked(True)

        self.layout_.addRow(self.shift_coord_check)

    def source_transform_operation(
        self,
    ) -> provenance.ToolProvenanceOperation:
        edge_fit = getattr(self, "_edge_fit", None)
        if edge_fit is None:
            raise RuntimeError("Edge correction fit data has not been loaded.")
        return provenance.CorrectWithEdgeOperation(
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
    operation_types = (provenance.SelOperation, provenance.IselOperation)

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
    ) -> list[provenance.ToolProvenanceOperation]:
        sel_kwargs: dict[Hashable, slice] = dict(self._slice_kwargs)
        isel_kwargs: dict[Hashable, slice] = {}
        operations: list[provenance.ToolProvenanceOperation] = []

        for key in list(sel_kwargs.keys()):
            if isinstance(key, str) and key.endswith("_idx"):
                isel_kwargs[key.removesuffix("_idx")] = self._nonuniform_isel_slice(
                    sel_kwargs.pop(key)
                )

        if sel_kwargs:
            operations.append(provenance.SelOperation(kwargs=sel_kwargs))
        if isel_kwargs:
            operations.append(provenance.IselOperation(kwargs=isel_kwargs))
        return operations

    def process_data(self, data: xr.DataArray) -> xr.DataArray:
        for operation in self.source_operations():
            data = operation.apply(data, parent_data=data)
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
    operation_types = (provenance.NormalizeOperation,)
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
        return operation.apply(data, parent_data=data)

    def filter_operation(
        self,
    ) -> provenance.ToolProvenanceOperation | None:
        norm_dims = self._norm_dims
        if not norm_dims:
            return None
        return provenance.NormalizeOperation(
            dims=norm_dims,
            mode=self._mode,
            denominator_rtol=self.denominator_rtol,
        )

    def restore_filter_operation(
        self,
        operation: provenance.ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            provenance.NormalizeOperation,
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
    operation_types = (provenance.DivideByCoordOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
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
            provenance.DivideByCoordOperation._raise_if_zero(coord)
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
    ) -> provenance.ToolProvenanceOperation:
        coord_name = self._selected_coord_name
        if coord_name is None:
            raise ValueError("No coordinate selected")
        return provenance.DivideByCoordOperation(coord_name=coord_name)


class GaussianFilterDialog(DataFilterDialog):
    title = "Gaussian Filter"
    enable_copy = True
    operation_types = (provenance.GaussianFilterOperation,)

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
    ) -> provenance.ToolProvenanceOperation | None:
        sigma_values, _ = self._sigma_values()
        if not sigma_values:
            return None
        return provenance.GaussianFilterOperation(sigma=sigma_values)

    def restore_filter_operation(
        self,
        operation: provenance.ToolProvenanceOperation,
    ) -> None:
        if not isinstance(
            operation,
            provenance.GaussianFilterOperation,
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


class SwapDimsDialog(DataTransformDialog):
    title = "Swap Dimensions"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (provenance.SwapDimsOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
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
    ) -> provenance.ToolProvenanceOperation:
        if not self._swap_mapping:
            raise ValueError("No dimensions changed")
        return provenance.SwapDimsOperation(mapping=self._swap_mapping)


class RenameDimsCoordsDialog(DataTransformDialog):
    title = "Rename Coordinates and Dimensions"
    enable_copy = True
    apply_on_nonuniform_data = True
    operation_types = (provenance.RenameDimsCoordsOperation,)

    def setup_widgets(self) -> None:
        self._source_data = erlab.interactive.imagetool.slicer.restore_nonuniform_dims(
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
    ) -> provenance.ToolProvenanceOperation:
        if not self._rename_mapping:
            raise ValueError("No names changed")
        return provenance.RenameDimsCoordsOperation(
            mapping=typing.cast("dict[Hashable, Hashable]", self._rename_mapping)
        )


class AssignCoordsDialog(DataTransformDialog):
    title = "Coordinate Editor"
    operation_types = (
        provenance.AffineCoordOperation,
        provenance.AssignCoordsOperation,
        provenance.AssignScalarCoordOperation,
        provenance.AssignCoord1DOperation,
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
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Coordinate Value",
                str(exc),
            )
            return

        super().accept()

    def source_transform_operation(
        self,
    ) -> provenance.ToolProvenanceOperation:
        if self._mode_tabs.currentIndex() == 1:
            values, dim = self._add_coord_values()
            name = self._add_name_edit.text().strip()
            if dim is None:
                return provenance.AssignScalarCoordOperation(
                    coord_name=name,
                    value=values,
                )
            return provenance.AssignCoord1DOperation(
                coord_name=name,
                dim=dim,
                values=values,
            )
        if self.coord_widget.use_affine_transform:
            return provenance.AffineCoordOperation(
                coord_name=self.current_coord_name,
                scale=self.coord_widget.affine_scale,
                offset=self.coord_widget.affine_offset,
            )
        return provenance.AssignCoordsOperation(
            coord_name=self.current_coord_name,
            values=self.coord_widget.new_coord,
        )


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
    operation_types = (provenance.AssignAttrsOperation,)

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
            QtWidgets.QMessageBox.warning(
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
    ) -> provenance.ToolProvenanceOperation:
        return provenance.AssignAttrsOperation(attrs=self._changed_attrs)


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
    ) -> provenance.ToolProvenanceOperation:
        return provenance.SliceAlongPathOperation(**self._params)


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
    ) -> provenance.ToolProvenanceOperation:
        return provenance.MaskWithPolygonOperation(**self._params)
