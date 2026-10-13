"""Shared editing and rendering of momentum-space image and pocket overlays."""

from __future__ import annotations

import base64
import pathlib
import typing
import uuid
from functools import partial

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from erlab.interactive._momentum_overlays import _geometry
from erlab.interactive._momentum_overlays._state import (
    ImageState,
    MomentumOverlayState,
    PocketClipboard,
    PocketState,
)
from erlab.interactive._momentum_overlays._widgets import ImageROI, PocketROI
from erlab.interactive.utils import BetterSpinBox, qt_is_valid, single_shot

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    import numpy.typing as npt
    import PyQt6.QtWidgets
    import PySide6.QtWidgets

    CoordinateMap = Callable[
        [npt.ArrayLike, npt.ArrayLike], tuple[npt.ArrayLike, npt.ArrayLike]
    ]


_POCKET_MIME_TYPE = "application/x-erlab-momentum-pockets+json"


class MomentumOverlayEditor(QtWidgets.QWidget):
    """Edit overlays on a momentum plot and optionally project them into another view.

    The host supplies coordinate maps and lattice vectors. This widget does not
    interpret analyzer settings or change the host's measured data.
    """

    sigStateChanged = QtCore.Signal()
    sigDragStarted = QtCore.Signal()
    sigDragFinished = QtCore.Signal()

    def __init__(
        self,
        momentum_plot: pg.PlotItem,
        parent: QtWidgets.QWidget | None = None,
        *,
        projected_plot: pg.PlotItem | None = None,
        image_z: float = -10,
    ) -> None:
        super().__init__(parent)
        self.momentum_plot = momentum_plot
        self.projected_plot = projected_plot
        self.state = MomentumOverlayState()
        self._restoring = False
        self._updating = False
        self._active = True
        self._available = True
        self._pocket_rois: dict[str, PocketROI] = {}
        self._selected_pocket: str | None = None
        self._contours: list[tuple[pg.PlotDataItem, pg.PlotDataItem | None]] = []
        self._source_image: np.ndarray | None = None
        self._display_image: np.ndarray | None = None
        self._image_roi: ImageROI | None = None
        self._drag_roi: PocketROI | ImageROI | None = None
        self._drag_start_geometry: npt.NDArray | None = None
        self._drag_start_state: PocketState | ImageState | None = None
        self._grid_cache: dict[tuple, tuple[npt.ArrayLike, npt.ArrayLike]] = {}
        self._pocket_cache: tuple | None = None
        self._translations: npt.NDArray[np.float64] | None = None
        self._lattice = np.eye(2)
        self._repeats = (0, 0)
        self._momentum_to_display: CoordinateMap | None = None
        self._display_to_momentum: CoordinateMap | None = None
        self._projection_token: Hashable | None = None
        self._projection_max_step: tuple[float, float] | None = None
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(24)
        self._redraw_timer.timeout.connect(self.redraw)
        self._refine_timer = QtCore.QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(180)
        self._refine_timer.timeout.connect(self._refine_image)
        self.inputs: dict[str, QtWidgets.QDoubleSpinBox | BetterSpinBox] = {}
        self.momentum_image = pg.ImageItem(axisOrder="row-major")
        self.momentum_image.setZValue(image_z)
        # View-sized projected images must not feed back into auto-range.
        self.momentum_plot.addItem(self.momentum_image, ignoreBounds=True)
        self.projected_image: pg.ImageItem | None = None
        if projected_plot is not None:
            self.projected_image = pg.ImageItem(axisOrder="row-major")
            self.projected_image.setZValue(image_z)
            projected_plot.addItem(self.projected_image, ignoreBounds=True)
            projected_plot.getViewBox().sigRangeChanged.connect(self._schedule_redraw)
        self._build_ui()
        self.restore_state(self.state)
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.dataChanged.connect(self._update_pocket_actions)

    def _spin(
        self,
        name,
        lower,
        upper,
        value,
        suffix="",
        decimals=3,
        *,
        step=0.1,
        significant_digits: int | None = None,
    ):
        spin: QtWidgets.QDoubleSpinBox | BetterSpinBox
        if significant_digits is None:
            spin = QtWidgets.QDoubleSpinBox(self)
            spin.setDecimals(2 if suffix == "°" else decimals)
        else:
            spin = BetterSpinBox(
                self,
                compact=False,
                decimals=significant_digits,
                significant=True,
                trim="-",
            )
        spin.setObjectName(name)
        spin.setRange(lower, upper)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        self.inputs[name] = spin
        return spin

    def _build_ui(self):
        overlay_layout = QtWidgets.QFormLayout(self)
        overlay_layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        overlay_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.pocket_list = QtWidgets.QListWidget()
        self.pocket_list.setObjectName("momentum_pocket_list")
        self.pocket_list.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.pocket_list.setMaximumHeight(150)
        self.pocket_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.pocket_list.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.pocket_list.currentItemChanged.connect(self._pocket_selected)
        self.pocket_list.itemSelectionChanged.connect(self._pocket_selected)
        self.pocket_list.itemChanged.connect(self._pocket_renamed)
        self.pocket_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.pocket_list.customContextMenuRequested.connect(self._show_pocket_menu)
        self.pocket_menu = QtWidgets.QMenu(self.pocket_list)
        self.rename_pocket_action = QtGui.QAction("Rename…", self.pocket_list)
        self.copy_pockets_action = QtGui.QAction("Copy", self.pocket_list)
        self.paste_pockets_action = QtGui.QAction("Paste", self.pocket_list)
        for action, name, slot in (
            (self.rename_pocket_action, "rename_pocket", self._rename_pocket),
            (self.copy_pockets_action, "copy_pockets", self._copy_pockets),
            (self.paste_pockets_action, "paste_pockets", self._paste_pockets),
        ):
            action.setObjectName(name)
            action.triggered.connect(slot)
            action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
            self.pocket_list.addAction(action)
            self.pocket_menu.addAction(action)
        self.copy_pockets_action.setShortcuts(QtGui.QKeySequence.StandardKey.Copy)
        self.paste_pockets_action.setShortcuts(QtGui.QKeySequence.StandardKey.Paste)
        self.pocket_menu.aboutToShow.connect(self._update_pocket_actions)
        overlay_layout.addRow(self.pocket_list)
        self.add_button = QtWidgets.QPushButton("Add")
        self.add_button.clicked.connect(self.add_pocket)
        self.delete_button: (
            PyQt6.QtWidgets.QPushButton | PySide6.QtWidgets.QPushButton
        ) = QtWidgets.QPushButton("Remove")
        self.delete_button.clicked.connect(self._delete_selected)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.delete_button)
        overlay_layout.addRow(buttons)
        for name, label, lower, upper, value in (
            ("pocket_x", "Center kx", -100, 100, 0),
            ("pocket_y", "Center ky", -100, 100, 0),
            ("pocket_rx", "Radius 1", 0.00001, 100, 0.05),
            ("pocket_ry", "Radius 2", 0.00001, 100, 0.05),
            ("pocket_rotation", "Rotation", -360, 360, 0),
        ):
            spin = self._spin(
                name,
                lower,
                upper,
                value,
                "°" if name.endswith("rotation") else " Å⁻¹",
                5,
                step=0.1 if name.endswith("rotation") else 0.001,
            )
            spin.valueChanged.connect(partial(self._pocket_numbers_changed, name))
            overlay_layout.addRow(label, spin)
        self.group_combo = QtWidgets.QComboBox()
        self.group_combo.addItems(
            ["1", "2", "3", "4", "6", "m", "2mm", "3m", "4mm", "6mm"]
        )
        self.group_combo.currentTextChanged.connect(
            partial(self._pocket_numbers_changed, "point_group")
        )
        overlay_layout.addRow("Point group", self.group_combo)
        self.symmetry_spin = self._spin("symmetry_angle", -360, 360, 0, "°")
        self.symmetry_spin.valueChanged.connect(
            partial(self._pocket_numbers_changed, "symmetry_angle")
        )
        overlay_layout.addRow("Mirror axis", self.symmetry_spin)
        self.pocket_visible = QtWidgets.QCheckBox("Show pocket")
        self.pocket_symmetry = QtWidgets.QCheckBox("Apply point group")
        for check, field in (
            (self.pocket_visible, "visible"),
            (self.pocket_symmetry, "symmetry"),
        ):
            check.toggled.connect(partial(self._pocket_numbers_changed, field))
            overlay_layout.addRow(check)
        self.overlay_separator = QtWidgets.QFrame()
        self.overlay_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.overlay_separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        overlay_layout.addRow(self.overlay_separator)
        self.image_button = QtWidgets.QPushButton("Select…")
        self.image_button.clicked.connect(self._choose_image)
        self.image_name = QtWidgets.QLabel("No image")
        self.image_name.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.image_name.setWordWrap(True)
        self.image_name.setMinimumWidth(0)
        overlay_layout.addRow("Image", self.image_button)
        overlay_layout.addRow(self.image_name)
        self.image_locked = QtWidgets.QCheckBox("Lock image")
        self.image_locked.setObjectName("image_locked")
        self.image_locked.setToolTip(
            "Hide the image frame and its drag, scale, and rotation handles. "
            "You can still change the image with the numeric controls."
        )
        self.image_locked.toggled.connect(self._image_lock_changed)
        overlay_layout.addRow(self.image_locked)
        for name, label, lower, upper, value in (
            ("image_x", "Center kx", -100, 100, 0),
            ("image_y", "Center ky", -100, 100, 0),
            ("image_width", "Width", 0.00001, 1000, 0.3),
            ("image_rotation", "Rotation", -360, 360, 0),
            ("image_opacity", "Opacity", 0, 1, 0.65),
        ):
            spin = self._spin(
                name,
                lower,
                upper,
                value,
                "°"
                if name.endswith("rotation")
                else ""
                if name.endswith("opacity")
                else " Å⁻¹",
                5,
                step=0.1
                if name.endswith("rotation")
                else 0.01
                if name.endswith("opacity")
                else 0.001,
                significant_digits=4 if name == "image_width" else None,
            )
            spin.valueChanged.connect(partial(self._image_numbers_changed, name))
            overlay_layout.addRow(label, spin)
        self.image_invert = QtWidgets.QCheckBox("Invert colors")
        self.image_invert.setObjectName("image_invert")
        self.image_invert.toggled.connect(
            partial(self._image_numbers_changed, "invert")
        )
        overlay_layout.addRow(self.image_invert)
        self.image_black_transparent = QtWidgets.QCheckBox("Make black transparent")
        self.image_black_transparent.setObjectName("image_black_transparent")
        self.image_black_transparent.toggled.connect(
            partial(self._image_numbers_changed, "black_transparent")
        )
        overlay_layout.addRow(self.image_black_transparent)
        threshold = self._spin("image_black_threshold", 0, 255, 0, decimals=0, step=1)
        threshold.setToolTip(
            "After inversion, hide pixels whose red, green, and blue values are all "
            "at or below this threshold. Use 0 for pure black or 255 for all colors."
        )
        threshold.valueChanged.connect(
            partial(self._image_numbers_changed, "black_threshold")
        )
        overlay_layout.addRow("Black threshold", threshold)
        overlay_layout.setRowVisible(threshold, False)
        self.image_black_transparent.toggled.connect(
            lambda visible: overlay_layout.setRowVisible(threshold, visible)
        )
        self.image_visible = QtWidgets.QCheckBox("Show image")
        self.image_visible.toggled.connect(
            partial(self._image_numbers_changed, "visible")
        )
        overlay_layout.addRow(self.image_visible)
        for following_widget, name in (
            (self.inputs["pocket_x"], "pocket_selection_separator"),
            (self.group_combo, "pocket_symmetry_separator"),
        ):
            separator = QtWidgets.QFrame()
            separator.setObjectName(name)
            separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            row, _ = overlay_layout.getWidgetPosition(following_widget)
            overlay_layout.insertRow(row, separator)

    def add_pocket(self, *_args, state: PocketState | None = None):
        pocket = state or PocketState()
        if state is None:
            pocket.point_group = self.group_combo.currentText()
            pocket.symmetry_angle = self.symmetry_spin.value()
            colors = ("#e69f00", "#009e73", "#cc79a7", "#f0e442")
            pocket.color = colors[len(self.state.pockets) % len(colors)]
        if pocket.uid in self._pocket_rois:
            raise ValueError("Pocket identifiers must be unique.")
        if not pocket.name:
            names = {existing.name for existing in self.state.pockets}
            index = 1
            while f"Pocket {index}" in names:
                index += 1
            pocket.name = f"Pocket {index}"
        self.state.pockets.append(pocket)
        roi = PocketROI(
            pocket.center,
            pocket.radii,
            pocket.rotation,
            pen=pg.mkPen(pocket.color, width=2),
        )
        roi.setZValue(10)
        self._pocket_rois[pocket.uid] = roi
        roi.sigSelected.connect(self._roi_selected)
        roi.sigRegionChangeStarted.connect(self._start_drag)
        roi.sigRegionChanged.connect(self._pocket_dragged)
        roi.sigRegionChangeFinished.connect(self._finish_drag)
        roi.sigRemoveRequested.connect(self._remove_roi)
        self.momentum_plot.addItem(roi, ignoreBounds=True)
        item = QtWidgets.QListWidgetItem(pocket.name)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, pocket.uid)
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        self.pocket_list.addItem(item)
        self.pocket_list.setCurrentItem(
            item, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
        self._pocket_selected()
        self._changed()
        return roi

    def _pocket_selected(self, *_args):
        item = self.pocket_list.currentItem()
        self._selected_pocket = (
            None
            if item is None or not item.isSelected()
            else item.data(QtCore.Qt.ItemDataRole.UserRole)
        )
        self._sync_pocket_numbers()
        self._update_pocket_actions()

    def _pocket_renamed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._restoring:
            return
        pocket = next(
            p
            for p in self.state.pockets
            if p.uid == item.data(QtCore.Qt.ItemDataRole.UserRole)
        )
        name = item.text().strip() or pocket.name
        with QtCore.QSignalBlocker(self.pocket_list):
            item.setText(name)
        if name != pocket.name:
            self.finish_drag()
            pocket.name = name
            self._changed()

    def _rename_pocket(self) -> None:
        items = self.pocket_list.selectedItems()
        if self._available and len(items) == 1:
            self.pocket_list.editItem(items[0])

    def _show_pocket_menu(self, position: QtCore.QPoint) -> None:
        item = self.pocket_list.itemAt(position)
        if item is not None and not item.isSelected():
            self.pocket_list.setCurrentItem(
                item, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            )
        self.pocket_menu.popup(self.pocket_list.viewport().mapToGlobal(position))

    @QtCore.Slot()
    def _update_pocket_actions(self) -> None:
        selected = self.pocket_list.selectedItems()
        self.rename_pocket_action.setEnabled(self._available and len(selected) == 1)
        self.copy_pockets_action.setEnabled(self._available and bool(selected))
        self.paste_pockets_action.setEnabled(
            self._available and bool(self._clipboard_pockets())
        )
        self.delete_button.setEnabled(self._available and bool(selected))

    @staticmethod
    def _clipboard_pockets() -> list[PocketState]:
        clipboard = QtWidgets.QApplication.clipboard()
        mime = None if clipboard is None else clipboard.mimeData()
        if mime is None or not mime.hasFormat(_POCKET_MIME_TYPE):
            return []
        try:
            pockets = PocketClipboard.model_validate_json(
                mime.data(_POCKET_MIME_TYPE).data()
            ).pockets
            # Validate every color before adding any ROI to the destination plot.
            for pocket in pockets:
                if not pocket.color or (
                    pocket.color.startswith("#")
                    and len(pocket.color) not in (4, 5, 7, 9)
                ):
                    return []
                pg.mkColor(pocket.color)
        except (ValueError, TypeError):
            return []
        return pockets

    def _copy_pockets(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        selected = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self.pocket_list.selectedItems()
        }
        if not self._available or clipboard is None or not selected:
            return
        payload = PocketClipboard(
            version=1, pockets=[p for p in self.state.pockets if p.uid in selected]
        )
        mime = QtCore.QMimeData()
        mime.setData(_POCKET_MIME_TYPE, payload.model_dump_json().encode("utf-8"))
        clipboard.setMimeData(mime)

    def _paste_pockets(self) -> None:
        if not self._available or not (pockets := self._clipboard_pockets()):
            return
        self.finish_drag()
        first_row = self.pocket_list.count()
        with QtCore.QSignalBlocker(self):
            for pocket in pockets:
                self.add_pocket(
                    state=pocket.model_copy(update={"uid": uuid.uuid4().hex})
                )
            for row in range(first_row, self.pocket_list.count()):
                self.pocket_list.item(row).setSelected(True)
        self._changed()

    def _roi_selected(self, roi):
        uid = next(
            uid for uid, candidate in self._pocket_rois.items() if candidate is roi
        )
        for index in range(self.pocket_list.count()):
            item = self.pocket_list.item(index)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == uid:
                self.pocket_list.setCurrentItem(
                    item, QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
                )
                break

    def _sync_pocket_numbers(self):
        pocket = next(
            (p for p in self.state.pockets if p.uid == self._selected_pocket), None
        )
        self._updating = True
        try:
            for name in (
                "pocket_x",
                "pocket_y",
                "pocket_rx",
                "pocket_ry",
                "pocket_rotation",
            ):
                self.inputs[name].setEnabled(pocket is not None)
            for widget in (
                self.delete_button,
                self.pocket_visible,
                self.pocket_symmetry,
                self.group_combo,
                self.symmetry_spin,
            ):
                widget.setEnabled(pocket is not None)
            if pocket is not None:
                for name, value in zip(
                    (
                        "pocket_x",
                        "pocket_y",
                        "pocket_rx",
                        "pocket_ry",
                        "pocket_rotation",
                    ),
                    (*pocket.center, *pocket.radii, pocket.rotation),
                    strict=True,
                ):
                    self.inputs[name].setValue(value)
                self.pocket_visible.setChecked(pocket.visible)
                self.pocket_symmetry.setChecked(pocket.symmetry)
                self.group_combo.setCurrentText(pocket.point_group)
                self.symmetry_spin.setValue(pocket.symmetry_angle)
        finally:
            self._updating = False

    def _pocket_numbers_changed(self, field, value):
        if self._restoring or self._updating or self._selected_pocket is None:
            return
        self.finish_drag()
        pocket = next(p for p in self.state.pockets if p.uid == self._selected_pocket)
        match field:
            case "pocket_x":
                pocket.center = (value, pocket.center[1])
            case "pocket_y":
                pocket.center = (pocket.center[0], value)
            case "pocket_rx":
                pocket.radii = (value, pocket.radii[1])
            case "pocket_ry":
                pocket.radii = (pocket.radii[0], value)
            case "pocket_rotation":
                pocket.rotation = value
            case "visible":
                pocket.visible = value
            case "symmetry":
                pocket.symmetry = value
            case "point_group":
                pocket.point_group = value
            case "symmetry_angle":
                pocket.symmetry_angle = value
        if field.startswith("pocket_"):
            roi = self._pocket_rois[pocket.uid]
            with QtCore.QSignalBlocker(roi):
                roi.setGeometry(pocket.center, pocket.radii, pocket.rotation)
        self._changed()

    def _pocket_dragged(self, roi):
        if self._restoring:
            return
        pocket = next(p for p in self.state.pockets if self._pocket_rois[p.uid] is roi)
        pocket.center, pocket.radii, pocket.rotation = roi.geometry()
        pocket.center = tuple(np.clip(pocket.center, -100, 100))
        pocket.radii = tuple(np.clip(pocket.radii, 0.00001, 100))
        pocket.rotation = (pocket.rotation + 180) % 360 - 180
        with QtCore.QSignalBlocker(roi):
            roi.setGeometry(pocket.center, pocket.radii, pocket.rotation)
        self._roi_selected(roi)
        self._sync_pocket_numbers()
        self._changed()

    def _remove_roi(self, roi):
        uid = next(
            (uid for uid, candidate in self._pocket_rois.items() if candidate is roi),
            None,
        )
        if uid is not None:
            self._remove_pocket(uid)

    def _delete_selected(self):
        selected = self.pocket_list.selectedItems()
        if selected:
            self.finish_drag()
            with QtCore.QSignalBlocker(self):
                for item in selected:
                    self._remove_pocket(item.data(QtCore.Qt.ItemDataRole.UserRole))
            self._changed()

    def _remove_pocket(self, uid):
        if self._pocket_rois[uid] is self._drag_roi:
            self.finish_drag()
        roi = self._pocket_rois.pop(uid)
        roi.sigSelected.disconnect(self._roi_selected)
        roi.sigRegionChangeStarted.disconnect(self._start_drag)
        roi.sigRegionChanged.disconnect(self._pocket_dragged)
        roi.sigRegionChangeFinished.disconnect(self._finish_drag)
        roi.sigRemoveRequested.disconnect(self._remove_roi)
        roi.hide()
        plot = self.momentum_plot

        def remove_graphics() -> None:
            plot.removeItem(roi)
            roi.deleteLater()

        # Removing graphics inside a menu handler is unsafe. Keep this exact
        # ROI alive until teardown; Undo can restore the same UID before then.
        single_shot(roi, 0, remove_graphics, plot)
        self.state.pockets = [p for p in self.state.pockets if p.uid != uid]
        for index in range(self.pocket_list.count()):
            if (
                self.pocket_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
                == uid
            ):
                self.pocket_list.takeItem(index)
                break
        if not self._restoring and not self.pocket_list.selectedItems():
            self.pocket_list.setCurrentRow(
                min(index, self.pocket_list.count() - 1),
                QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
        self._pocket_selected()
        self._changed()

    def _choose_image(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open momentum image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if filename:
            try:
                self.load_image(filename)
            except ValueError as error:
                QtWidgets.QMessageBox.warning(self, "Cannot open image", str(error))

    def load_image(self, filename: str) -> None:
        image = QtGui.QImage(filename)
        if image.isNull():
            raise ValueError("The selected file is not a readable image.")
        self.finish_drag()
        data = QtCore.QByteArray()
        buffer = QtCore.QBuffer(data)
        buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        buffer.close()
        previous = self.state.image
        self.state.image = ImageState(
            png_base64=base64.b64encode(data.data()).decode("ascii"),
            name=pathlib.Path(filename).name,
        )
        if previous is not None:
            self.state.image.center = previous.center
            self.state.image.width = previous.width
            self.state.image.rotation = previous.rotation
            self.state.image.opacity = previous.opacity
            self.state.image.visible = previous.visible
            self.state.image.locked = previous.locked
            self.state.image.invert = previous.invert
            self.state.image.black_transparent = previous.black_transparent
            self.state.image.black_threshold = previous.black_threshold
        self._restore_image(image)
        self._changed()

    def _restore_image(self, image: QtGui.QImage | None):
        if self._image_roi is not None:
            self._image_roi.sigRegionChangeStarted.disconnect(self._start_drag)
            self._image_roi.sigRegionChanged.disconnect(self._image_dragged)
            self._image_roi.sigRegionChangeFinished.disconnect(self._finish_drag)
            self.momentum_plot.removeItem(self._image_roi)
            self._image_roi.deleteLater()
            self._image_roi = None
        self._source_image = None
        self._display_image = None
        self.momentum_image.clear()
        state = self.state.image
        if state is not None:
            if image is None:
                raise ValueError("An image is required to restore the image overlay.")
            image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            self._source_image = pg.functions.ndarray_from_qimage(image).copy()
            self._update_image_pixels()
            self._image_roi = ImageROI(
                state.center,
                state.width,
                image.height() / image.width(),
                rotation=state.rotation,
                pen=pg.mkPen("#999999"),
            )
            self._image_roi.setZValue(5)
            self._image_roi.sigRegionChangeStarted.connect(self._start_drag)
            self._image_roi.sigRegionChanged.connect(self._image_dragged)
            self._image_roi.sigRegionChangeFinished.connect(self._finish_drag)
            self.momentum_plot.addItem(self._image_roi, ignoreBounds=True)
        self._sync_image_numbers()

    def _update_image_pixels(self) -> None:
        image = self._source_image
        state = self.state.image
        if image is None or state is None:
            return
        if state.invert or state.black_transparent:
            image = image.copy()
        if state.invert:
            image[..., :3] = 255 - image[..., :3]
        if state.black_transparent:
            image[..., 3][np.all(image[..., :3] <= state.black_threshold, axis=-1)] = 0
        self._display_image = image
        self.momentum_image.setImage(np.flipud(image), autoLevels=False)

    def _sync_image_numbers(self):
        state = self.state.image
        self._updating = True
        try:
            self.image_name.setText(state.name if state else "No image")
            for name in (
                "image_x",
                "image_y",
                "image_width",
                "image_rotation",
                "image_opacity",
            ):
                self.inputs[name].setEnabled(state is not None)
            self.image_visible.setEnabled(state is not None)
            self.image_locked.setEnabled(state is not None)
            self.image_locked.setChecked(state.locked if state is not None else False)
            self.image_invert.setEnabled(state is not None)
            self.image_black_transparent.setEnabled(state is not None)
            self.image_invert.setChecked(state.invert if state is not None else False)
            self.image_black_transparent.setChecked(
                state.black_transparent if state is not None else False
            )
            self.inputs["image_black_threshold"].setEnabled(
                state is not None and state.black_transparent
            )
            self.inputs["image_black_threshold"].setValue(
                state.black_threshold if state is not None else 0
            )
            if state is not None:
                for name, value in zip(
                    (
                        "image_x",
                        "image_y",
                        "image_width",
                        "image_rotation",
                        "image_opacity",
                    ),
                    (*state.center, state.width, state.rotation, state.opacity),
                    strict=True,
                ):
                    self.inputs[name].setValue(value)
                self.image_visible.setChecked(state.visible)
        finally:
            self._updating = False
        self._sync_image_handles()

    def _sync_image_handles(self) -> None:
        state = self.state.image
        if self._image_roi is not None:
            self._image_roi.setVisible(
                self._available
                and state is not None
                and state.visible
                and not state.locked
                and self._display_image is not None
            )

    def _image_lock_changed(self, locked: bool) -> None:
        state = self.state.image
        if self._restoring or self._updating or state is None:
            return
        self.finish_drag()
        state.locked = locked
        self._sync_image_handles()
        self._changed()

    def _image_numbers_changed(self, field, value):
        state = self.state.image
        if self._restoring or self._updating or state is None:
            return
        self.finish_drag()
        match field:
            case "image_x":
                state.center = (value, state.center[1])
            case "image_y":
                state.center = (state.center[0], value)
            case "image_width":
                state.width = value
            case "image_rotation":
                state.rotation = value
            case "image_opacity":
                state.opacity = value
            case "visible":
                state.visible = value
            case "invert":
                state.invert = value
            case "black_transparent":
                state.black_transparent = value
            case "black_threshold":
                state.black_threshold = int(value)
        if field in ("invert", "black_transparent", "black_threshold"):
            self._update_image_pixels()
        self.inputs["image_black_threshold"].setEnabled(state.black_transparent)
        if (
            field in ("image_x", "image_y", "image_width", "image_rotation")
            and self._image_roi is not None
        ):
            with QtCore.QSignalBlocker(self._image_roi):
                self._image_roi.setGeometry(
                    state.center, state.width, rotation=state.rotation
                )
        self._changed()

    def _image_dragged(self, roi):
        if self.state.image is not None:
            (
                self.state.image.center,
                self.state.image.width,
                self.state.image.rotation,
            ) = roi.geometry()
            self.state.image.rotation = (self.state.image.rotation + 180) % 360 - 180
            self.state.image.center = tuple(np.clip(self.state.image.center, -100, 100))
            self.state.image.width = float(
                np.clip(self.state.image.width, 0.00001, 1000)
            )
            with QtCore.QSignalBlocker(roi):
                roi.setGeometry(
                    self.state.image.center,
                    self.state.image.width,
                    rotation=self.state.image.rotation,
                )
            self._sync_image_numbers()
            self._changed()

    def restore_state(self, state: MomentumOverlayState | dict) -> None:
        restored = MomentumOverlayState.model_validate(state).model_copy(deep=True)
        image = None
        if restored.image is not None:
            image = QtGui.QImage.fromData(
                base64.b64decode(restored.image.png_base64), "PNG"
            )
            if image.isNull():
                raise ValueError("The embedded image cannot be decoded.")
        self.finish_drag(notify=False)
        selected_uid = self._selected_pocket
        selected_uids = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self.pocket_list.selectedItems()
        }
        self._restoring = True
        try:
            for uid in tuple(self._pocket_rois):
                self._remove_pocket(uid)
            self.state = restored
            pockets = self.state.pockets
            self.state.pockets = []
            for pocket in pockets:
                self.add_pocket(state=pocket)
            if selected_uid is not None and selected_uid in self._pocket_rois:
                self._roi_selected(self._pocket_rois[selected_uid])
            if selected_uids.intersection(self._pocket_rois):
                for row in range(self.pocket_list.count()):
                    item = self.pocket_list.item(row)
                    item.setSelected(
                        item.data(QtCore.Qt.ItemDataRole.UserRole) in selected_uids
                    )
                current = self.pocket_list.currentItem()
                if current is None or not current.isSelected():
                    self.pocket_list.setCurrentItem(
                        self.pocket_list.selectedItems()[0],
                        QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
                    )
            self._pocket_selected()
            self._restore_image(image)
        finally:
            self._restoring = False
        self._update_pocket_actions()
        self.redraw()
        self._schedule_redraw()

    def set_lattice(
        self, basis: npt.ArrayLike, repeats: tuple[int, int] = (0, 0)
    ) -> None:
        lattice = np.asarray(basis, dtype=float)
        if lattice.shape != (2, 2) or not np.isfinite(lattice).all():
            raise ValueError("Overlay lattice vectors must form a finite 2 by 2 array.")
        if any(int(n) != n or n < 0 or n > 5 for n in repeats):
            raise ValueError(
                "Use zero to five neighboring cells per lattice direction."
            )
        if (
            self._translations is None
            and np.array_equal(lattice, self._lattice)
            and repeats == self._repeats
        ):
            return
        self._translations = None
        self._lattice = lattice.copy()
        self._repeats = repeats
        self._schedule_redraw()

    def set_translations(self, translations: npt.ArrayLike | None) -> None:
        """Set explicit momentum translations, or restore lattice repeats with None."""
        shifts = None if translations is None else np.asarray(translations, dtype=float)
        if shifts is not None and (
            shifts.ndim != 2 or shifts.shape[1] != 2 or not np.isfinite(shifts).all()
        ):
            raise ValueError("Overlay translations must form a finite N by 2 array.")
        if (shifts is None and self._translations is None) or (
            shifts is not None
            and self._translations is not None
            and np.array_equal(shifts, self._translations)
        ):
            return
        self._translations = None if shifts is None else shifts.copy()
        self._schedule_redraw()

    def set_projection(
        self,
        momentum_to_display: CoordinateMap | None,
        display_to_momentum: CoordinateMap | None,
        *,
        token: Hashable,
        max_step: tuple[float, float] | None = None,
    ) -> None:
        self._momentum_to_display = momentum_to_display
        self._display_to_momentum = display_to_momentum
        if token != self._projection_token or max_step != self._projection_max_step:
            self._projection_token = token
            self._projection_max_step = max_step
            self._grid_cache.clear()
            self._schedule_redraw()

    def set_available(self, available: bool) -> None:
        if available != self._available:
            if not available:
                self.finish_drag()
            self._available = available
            self.setEnabled(available)
            self._update_pocket_actions()
            self.redraw()
            self._schedule_redraw()

    def set_active(self, active: bool) -> None:
        """Enable rendering while preserving edit notifications when inactive."""
        self._active = active
        if active:
            self._schedule_redraw()
        else:
            self.finish_drag()
            self._redraw_timer.stop()
            self._refine_timer.stop()

    @property
    def dragging(self) -> bool:
        return self._drag_roi is not None

    def _start_drag(self, roi) -> None:
        if self._restoring or self._drag_roi is roi:
            return
        self.finish_drag()
        # The host records preceding edits before this gesture changes the state.
        self.sigDragStarted.emit()
        self._drag_roi = roi
        self._drag_start_geometry = np.r_[*roi.geometry()]
        if roi is self._image_roi:
            state = self.state.image
            self._drag_start_state = (
                None if state is None else state.model_copy(deep=True)
            )
        else:
            pocket = next(
                p for p in self.state.pockets if self._pocket_rois[p.uid] is roi
            )
            self._drag_start_state = pocket.model_copy(deep=True)

    def _finish_drag(self, roi) -> None:
        if self._drag_roi is roi:
            self.finish_drag()
        self._schedule_redraw()

    def finish_drag(self, *, notify: bool = True) -> None:
        roi = self._drag_roi
        if roi is None:
            return
        if qt_is_valid(roi):
            start = self._drag_start_geometry
            if notify and start is not None:
                geometry = np.r_[*roi.geometry()]
                geometry[-1] = start[-1] + (geometry[-1] - start[-1] + 180) % 360 - 180
                if np.allclose(geometry, start, rtol=0.0, atol=1e-12):
                    # Native cancellation can reconstruct coordinates with roundoff.
                    # Keep the original model geometry and the host's Redo branch.
                    saved = self._drag_start_state
                    if isinstance(saved, ImageState) and self.state.image is not None:
                        self.state.image.center = saved.center
                        self.state.image.width = saved.width
                        self.state.image.rotation = saved.rotation
                        self._sync_image_numbers()
                    elif isinstance(saved, PocketState):
                        pocket = next(
                            p for p in self.state.pockets if p.uid == saved.uid
                        )
                        pocket.center, pocket.radii, pocket.rotation = (
                            saved.center,
                            saved.radii,
                            saved.rotation,
                        )
                        self._sync_pocket_numbers()
                    self._changed()
            roi.isMoving = False
            roi.mouseDragHandler.dragMode = None
            for handle in roi.getHandles():
                handle.isMoving = False
        self._drag_roi = None
        self._drag_start_geometry = None
        self._drag_start_state = None
        if notify:
            self.sigDragFinished.emit()

    def _changed(self):
        if self._restoring:
            return
        self.sigStateChanged.emit()
        self._schedule_redraw()

    def _schedule_redraw(self, *_args):
        if self._restoring or not self._active or not qt_is_valid(self):
            return
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()
        self._refine_timer.start()

    @staticmethod
    def _set_curve(curve, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        curve.setData(points[:, 0], points[:, 1], connect="finite")

    def redraw(self) -> None:
        if (
            not self._active
            or not qt_is_valid(self)
            or not qt_is_valid(self.momentum_plot)
        ):
            return
        self._redraw_timer.stop()
        projected = (
            self.projected_plot is not None
            and qt_is_valid(self.projected_plot)
            and self.projected_plot.isVisible()
            and self._momentum_to_display is not None
        )
        pocket_key = (
            tuple(
                (
                    p.uid,
                    p.center,
                    p.radii,
                    p.rotation,
                    p.visible,
                    p.symmetry,
                    p.point_group,
                    p.symmetry_angle,
                    p.color,
                )
                for p in self.state.pockets
            ),
            tuple(self._lattice.ravel()),
            self._repeats,
            None if self._translations is None else tuple(self._translations.ravel()),
        )
        for pocket in self.state.pockets:
            self._pocket_rois[pocket.uid].setVisible(self._available and pocket.visible)
        if self._pocket_cache is None or self._pocket_cache[0] != pocket_key:
            contours = []
            for pocket in self.state.pockets:
                if pocket.visible:
                    copies = _geometry.expand_pocket(
                        pocket.center,
                        pocket.radii,
                        pocket.rotation,
                        pocket.point_group if pocket.symmetry else "1",
                        pocket.symmetry_angle,
                        self._lattice,
                        self._repeats,
                        translations=self._translations,
                    )
                    if not copies:
                        continue
                    contours.append(
                        (
                            np.vstack(
                                [
                                    np.vstack((points, [np.nan, np.nan]))
                                    for points in copies
                                ]
                            ),
                            pocket.color,
                        )
                    )
            self._pocket_cache = (pocket_key, contours)
        contours = self._pocket_cache[1]
        while len(self._contours) < len(contours):
            primary = pg.PlotDataItem()
            self.momentum_plot.addItem(primary, ignoreBounds=True)
            primary.setZValue(3)
            secondary = None
            if self.projected_plot is not None:
                secondary = pg.PlotDataItem()
                self.projected_plot.addItem(secondary, ignoreBounds=True)
                secondary.setZValue(3)
            self._contours.append((primary, secondary))
        for index, (primary, secondary) in enumerate(self._contours):
            visible = self._available and index < len(contours)
            primary.setVisible(visible)
            if secondary is not None:
                secondary.setVisible(visible and projected)
            if visible:
                points, color = contours[index]
                primary.setPen(pg.mkPen(color, width=1.5))
                self._set_curve(primary, points)
                if (
                    projected
                    and secondary is not None
                    and self._momentum_to_display is not None
                ):
                    secondary.setPen(pg.mkPen(color, width=1.5))
                    x, y = self._momentum_to_display(points[:, 0], points[:, 1])
                    projected_points = np.column_stack((x, y))
                    if self._projection_max_step is not None:
                        jumps = np.any(
                            np.abs(np.diff(projected_points, axis=0))
                            > self._projection_max_step,
                            axis=1,
                        )
                        projected_points[np.flatnonzero(jumps) + 1] = np.nan
                    self._set_curve(secondary, projected_points)
        self._draw_image(180)

    def _draw_image(self, resolution: int):
        state = self.state.image
        visible = (
            self._available
            and state is not None
            and state.visible
            and self._display_image is not None
        )
        self.momentum_image.setVisible(visible)
        self._sync_image_handles()
        projected_visible = (
            visible
            and self.projected_plot is not None
            and qt_is_valid(self.projected_plot)
            and self.projected_plot.isVisible()
            and self._display_to_momentum is not None
        )
        if self.projected_image is not None:
            self.projected_image.setVisible(projected_visible)
        if not visible or state is None or self._display_image is None:
            return
        image = self._display_image
        height = state.width * image.shape[0] / image.shape[1]
        transform = QtGui.QTransform()
        transform.translate(*state.center)
        transform.rotate(state.rotation)
        transform.translate(-state.width / 2, -height / 2)
        transform.scale(state.width / image.shape[1], height / image.shape[0])
        self.momentum_image.setTransform(transform)
        self.momentum_image.setOpacity(state.opacity)
        if (
            not projected_visible
            or self.projected_plot is None
            or self.projected_image is None
            or self._display_to_momentum is None
        ):
            return
        limits = tuple(tuple(axis) for axis in self.projected_plot.viewRange())
        key = (limits, resolution)
        if key not in self._grid_cache:
            self._grid_cache = {
                old_key: grid
                for old_key, grid in self._grid_cache.items()
                if old_key[0] == limits
            }
            # ImageItem rects describe pixel edges; sample the pixel centers.
            axes = [
                np.linspace(lower, upper, resolution, endpoint=False)
                + (upper - lower) / (2 * resolution)
                for lower, upper in limits
            ]
            x, y = np.meshgrid(*axes)
            self._grid_cache[key] = self._display_to_momentum(x, y)
        warped = _geometry.warp_image(
            image,
            *self._grid_cache[key],
            state.center,
            state.width,
            rotation=state.rotation,
        )
        self.projected_image.setImage(warped, autoLevels=False)
        (x0, x1), (y0, y1) = limits
        self.projected_image.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self.projected_image.setOpacity(state.opacity)

    def _refine_image(self):
        if self._active and qt_is_valid(self) and qt_is_valid(self.momentum_plot):
            self._draw_image(480)
