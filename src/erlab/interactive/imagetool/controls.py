"""Widgets for controlling :class:`erlab.interactive.imagetool.core.ImageSlicerArea`."""

from __future__ import annotations

import contextlib
import typing
import weakref

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    import xarray as xr

    from erlab.interactive.imagetool.core import ImageSlicerArea
    from erlab.interactive.imagetool.slicer import ArraySlicer


_ICON_ALIASES: dict[str, str] = {
    "invert": "mdi6.invert-colors",
    "invert_off": "mdi6.invert-colors-off",
    "contrast": "mdi6.contrast-box",
    "lock": "mdi6.lock",
    "unlock": "mdi6.lock-open-variant",
    "bright_auto": "mdi6.brightness-auto",
    "bright_percent": "mdi6.brightness-percent",
    "colorbar": "mdi6.gradient-vertical",
    "transpose_0": "mdi6.arrow-top-left-bottom-right",
    "transpose_1": "mdi6.arrow-up-down",
    "transpose_2": "mdi6.arrow-left-right",
    "transpose_3": "mdi6.axis-z-arrow",
    "snap": "mdi6.grid",
    "snap_off": "mdi6.grid-off",
    "palette": "mdi6.palette-advanced",
    "styles": "mdi6.palette-swatch",
    "layout": "mdi6.page-layout-body",
    "zero_center": "mdi6.format-vertical-align-center",
    "table_eye": "mdi6.table-eye",
    "plus": "mdi6.plus",
    "minus": "mdi6.minus",
    "reset": "mdi6.backup-restore",
    # all_cursors="mdi6.checkbox-multiple-outline",
    "all_cursors": "mdi6.sync",
}  #: Mapping from custom aliases to qtawesome icon names.


def clear_layout(layout: QtWidgets.QLayout | None) -> None:
    """Clear the given layout by removing all its child widgets and layouts recursively.

    Parameters
    ----------
    layout
        The layout to be cleared.
    """
    if layout is None:
        return
    while layout.count():
        child = layout.takeAt(0)
        if child is not None:
            wi, lo = child.widget(), child.layout()
            if wi:
                wi.deleteLater()
            elif lo:
                clear_layout(lo)
                lo.deleteLater()


class ItoolControlsBase(QtWidgets.QWidget):
    def __init__(
        self, slicer_area: ImageSlicerArea | ItoolControlsBase, *args, **kwargs
    ) -> None:
        self.sub_controls: list[ItoolControlsBase] = []
        super().__init__(*args, **kwargs)
        self._parent_control = weakref.ref(slicer_area)
        self.initialize_layout()
        self.initialize_widgets()
        self.connect_signals()
        self.update_content()

    @property
    def data(self) -> xr.DataArray:
        return self.slicer_area.data

    @property
    def array_slicer(self) -> ArraySlicer:
        return self.slicer_area.array_slicer

    @property
    def n_cursors(self) -> int:
        return self.slicer_area.n_cursors

    @property
    def current_cursor(self) -> int:
        return self.slicer_area.current_cursor

    def initialize_layout(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

    def initialize_widgets(self) -> None:
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.initialize_widgets()

    def connect_signals(self) -> None:
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.connect_signals()

    def disconnect_signals(self) -> None:
        # Multiple inheritance disconnection is broken
        # https://bugreports.qt.io/browse/PYSIDE-229
        # Will not work correctly until this is fixed
        for ctrl in self.sub_controls:
            if isinstance(ctrl, ItoolControlsBase):
                ctrl.disconnect_signals()

    def update_content(self) -> None:
        for ctrl in self.sub_controls:
            ctrl.update_content()

    def add_control(self, widget: ItoolControlsBase) -> ItoolControlsBase:
        self.sub_controls.append(widget)
        return widget

    @property
    def is_nested(self) -> bool:
        return isinstance(self.parent_control, ItoolControlsBase)

    @property
    def slicer_area(self) -> ImageSlicerArea:
        parent_control = self.parent_control
        if isinstance(parent_control, ItoolControlsBase):
            return parent_control.slicer_area
        return parent_control

    @property
    def parent_control(self) -> ImageSlicerArea | ItoolControlsBase:
        ref = self._parent_control()
        if ref is not None:
            return ref
        raise LookupError("Parent was destroyed")

    @parent_control.setter
    def parent_control(self, value: ImageSlicerArea | ItoolControlsBase) -> None:
        """Set the `ImageSlicerArea` instance for the control widget.

        Initially, the goal was to be able to control multiple `ImageSlicerArea`s with a
        single control widget, so the control widgets were designed with easy connection
        and disconnection of signals in mind. However, this has become largely
        unnecessary since we now have `SlicerLinkProxy` as a translation layer between
        multiple `ImageSlicerArea`s with individual control widgets. Hence, this method
        will remain unused for the time being.

        Also, in principle, most of the control widgets along with the menu bar should
        be re-written to use QActions...

        """
        # ignore until https://bugreports.qt.io/browse/PYSIDE-229 is fixed
        with contextlib.suppress(RuntimeError):
            self.disconnect_signals()
        self._parent_control = weakref.ref(value)
        clear_layout(self.layout())
        self.sub_controls = []
        self.initialize_widgets()
        self.update_content()
        self.connect_signals()


# class ItoolAAAAAControls(ItoolControlsBase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def initialize_layout(self):
#         pass

#     def initialize_widgets(self):
#         pass

#     def connect_signals(self):
#         pass

#     def disconnect_signals(self):
#         pass

#     def update_content(self):
#         pass


class ItoolCrosshairControls(ItoolControlsBase):
    def __init__(
        self, *args, orientation=QtCore.Qt.Orientation.Vertical, **kwargs
    ) -> None:
        if isinstance(orientation, QtCore.Qt.Orientation):
            self.orientation = orientation
        elif orientation == "vertical":
            self.orientation = QtCore.Qt.Orientation.Vertical
        elif orientation == "horizontal":
            self.orientation = QtCore.Qt.Orientation.Horizontal
        super().__init__(*args, **kwargs)

    def initialize_widgets(self) -> None:
        super().initialize_widgets()
        self.values_groups = tuple(
            QtWidgets.QWidget() for _ in range(self.data.ndim + 1)
        )
        self.values_layouts = tuple(
            QtWidgets.QGridLayout(g) for g in self.values_groups
        )
        for s in self.values_layouts:
            s.setContentsMargins(0, 0, 0, 0)

            s.setSpacing(3)
        # buttons for multicursor control
        self.btn_add = erlab.interactive.utils.IconButton(
            _ICON_ALIASES["plus"], toolTip="Add cursor"
        )
        self.btn_add.clicked.connect(self.slicer_area.add_cursor)

        self.btn_rem = erlab.interactive.utils.IconButton(
            _ICON_ALIASES["minus"], toolTip="Remove cursor"
        )
        self.btn_rem.clicked.connect(
            lambda: self.slicer_area.remove_cursor(self.cb_cursors.currentIndex())
        )

        self.btn_snap = erlab.interactive.utils.IconActionButton(
            self.array_slicer.snap_act,
            on=_ICON_ALIASES["snap"],
            off=_ICON_ALIASES["snap_off"],
        )

        # multicursor combobox
        self.cb_cursors = QtWidgets.QComboBox()
        self.cb_cursors.textActivated.connect(self.setActiveCursor)
        self.cb_cursors.setMaximumHeight(
            QtGui.QFontMetrics(self.cb_cursors.font()).height() + 3
        )
        self.cb_cursors.setIconSize(QtCore.QSize(10, 10))
        for i in range(self.n_cursors):
            self.cb_cursors.addItem(
                self.slicer_area._cursor_icon(i), self.slicer_area._cursor_name(i)
            )
        if self.n_cursors == 1:
            # can't remove more cursors
            self.cb_cursors.setDisabled(True)
            self.btn_rem.setDisabled(True)

        # current value widget
        self.spin_dat = erlab.interactive.utils.BetterSpinBox(
            self.values_groups[-1], discrete=False, scientific=True, readOnly=True
        )
        try:
            with np.errstate(divide="ignore"):
                self.spin_dat.setDecimals(
                    round(abs(np.log10(self.array_slicer.absnanmax)) + 1)
                )
        except OverflowError:
            self.spin_dat.setDecimals(4)

        # add multicursor widgets
        if self.orientation == QtCore.Qt.Orientation.Vertical:
            self.values_layouts[0].addWidget(self.btn_add, 0, 1, 1, 1)
            self.values_layouts[0].addWidget(self.btn_rem, 0, 2, 1, 1)
            self.values_layouts[0].addWidget(self.btn_snap, 0, 0, 1, 1)
            self.values_layouts[0].addWidget(self.cb_cursors, 1, 0, 1, 3)
            self.values_layouts[0].addWidget(self.spin_dat, 2, 0, 1, 3)
        else:
            self.values_layouts[0].addWidget(self.btn_add, 0, 1, 1, 1)
            self.values_layouts[0].addWidget(self.btn_rem, 0, 2, 1, 1)
            self.values_layouts[0].addWidget(self.btn_snap, 0, 3, 1, 1)
            self.values_layouts[0].addWidget(self.cb_cursors, 0, 0, 1, 1)
            self.values_layouts[0].addWidget(self.spin_dat, 0, 4, 1, 1)
        typing.cast("QtWidgets.QLayout", self.layout()).addWidget(self.values_groups[0])

        # info widgets
        self.label_dim = tuple(
            QtWidgets.QPushButton(grp) for grp in self.values_groups[1:]
        )
        for lab in self.label_dim:
            lab.setCheckable(True)

        self.spin_idx = tuple(
            erlab.interactive.utils.BetterSpinBox(
                grp,
                integer=True,
                singleStep=1,
                wrapping=False,
                minimumWidth=60,
                keyboardTracking=False,
            )
            for grp in self.values_groups[1:]
        )
        self.spin_val = tuple(
            erlab.interactive.utils.BetterSpinBox(
                grp,
                discrete=True,
                decimals=3,
                wrapping=False,
                minimumWidth=70,
                keyboardTracking=False,
            )
            for grp in self.values_groups[1:]
        )

        if self.data.ndim <= 3:
            icons = [_ICON_ALIASES[f"transpose_{i}"] for i in range(self.data.ndim)]
        else:
            icons = [_ICON_ALIASES[f"transpose_{i}"] for i in (0, 1, 3, 2)]
        self.btn_transpose = tuple(
            erlab.interactive.utils.IconButton(on=icon) for icon in icons
        )

        # add and connect info widgets
        for i in range(self.data.ndim):
            # TODO: implelemnt cursor locking
            # self.label_dim[i].toggled.connect()
            self.spin_idx[i].valueChanged.connect(
                lambda ind, axis=i: self.slicer_area.set_index(axis, ind)
            )
            self.spin_val[i].valueChanged.connect(
                lambda val, axis=i: self.slicer_area.set_value(axis, val, uniform=False)
            )
            self.btn_transpose[i].clicked.connect(
                lambda *, idx=i: self._transpose_axes(idx)
            )
            if self.orientation == QtCore.Qt.Orientation.Vertical:
                self.values_layouts[i + 1].addWidget(self.label_dim[i], 0, 0, 1, 1)
                self.values_layouts[i + 1].addWidget(self.btn_transpose[i], 0, 1, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_idx[i], 1, 0, 1, 2)
                self.values_layouts[i + 1].addWidget(self.spin_val[i], 2, 0, 1, 2)
            else:
                self.values_layouts[i + 1].addWidget(self.label_dim[i], 0, 0, 1, 1)
                self.values_layouts[i + 1].addWidget(self.btn_transpose[i], 0, 1, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_idx[i], 0, 2, 1, 1)
                self.values_layouts[i + 1].addWidget(self.spin_val[i], 0, 3, 1, 1)

            typing.cast("QtWidgets.QLayout", self.layout()).addWidget(
                self.values_groups[i + 1]
            )

    def _transpose_axes(self, idx: int) -> None:
        if self.data.ndim == 4:
            if idx == 3:
                self.slicer_area.swap_axes(0, 2)
            else:
                self.slicer_area.swap_axes(idx, (idx + 1) % 4)
        else:
            self.slicer_area.swap_axes(idx, (idx + 1) % self.data.ndim)

    def connect_signals(self) -> None:
        super().connect_signals()
        self.slicer_area.sigDataChanged.connect(self.update_content)
        self.slicer_area.sigShapeChanged.connect(self.update_content)
        self.slicer_area.sigCurrentCursorChanged.connect(self.cursorChangeEvent)
        self.slicer_area.sigCursorCountChanged.connect(self.update_cursor_count)
        self.slicer_area.sigIndexChanged.connect(self.update_spins)
        self.slicer_area.sigBinChanged.connect(self.update_spins)

    def disconnect_signals(self) -> None:
        super().disconnect_signals()
        self.slicer_area.sigDataChanged.disconnect(self.update_content)
        self.slicer_area.sigShapeChanged.disconnect(self.update_content)
        self.slicer_area.sigCurrentCursorChanged.disconnect(self.cursorChangeEvent)
        self.slicer_area.sigCursorCountChanged.disconnect(self.update_cursor_count)
        self.slicer_area.sigIndexChanged.disconnect(self.update_spins)
        self.slicer_area.sigBinChanged.disconnect(self.update_spins)

    @QtCore.Slot()
    def update_content(self) -> None:
        super().update_content()
        if len(self.label_dim) != self.data.ndim:
            # number of required cursors changed, resetting
            clear_layout(self.layout())
            self.initialize_widgets()

        for i in range(self.data.ndim):
            self.values_groups[i].blockSignals(True)
            self.spin_idx[i].blockSignals(True)
            self.spin_val[i].blockSignals(True)

            if i in self.array_slicer._nonuniform_axes:
                self.label_dim[i].setText(str(self.data.dims[i]).removesuffix("_idx"))
            else:
                self.label_dim[i].setText(str(self.data.dims[i]))

            lw = (
                self.label_dim[i]
                .fontMetrics()
                .boundingRect(self.label_dim[i].text())
                .width()
            )
            self.label_dim[i].setMaximumWidth(lw + 15)

            # update spinbox properties to match new data
            self.spin_idx[i].setRange(0, self.data.shape[i] - 1)
            self.spin_idx[i].setValue(self.slicer_area.get_current_index(i))

            self.spin_val[i].setRange(*self.array_slicer.lims[i])
            self.spin_val[i].setSingleStep(self.array_slicer.incs[i])
            self.spin_val[i].setValue(self.slicer_area.get_current_value(i))
            self.spin_val[i].setDecimals(self.array_slicer.get_significant(i))

            self.label_dim[i].blockSignals(False)
            self.spin_idx[i].blockSignals(False)
            self.spin_val[i].blockSignals(False)
        try:
            with np.errstate(divide="ignore"):
                self.spin_dat.setDecimals(
                    round(abs(np.log10(self.array_slicer.absnanmax)) + 1)
                )
        except OverflowError:
            self.spin_dat.setDecimals(4)
        self.spin_dat.setValue(
            self.array_slicer.point_value(self.current_cursor, binned=True)
        )

    def update_spins(self, *, axes=None) -> None:
        if axes is None:
            axes = range(self.data.ndim)
        if len(axes) != len(self.spin_idx):
            # called from sigIndexChanged before update was called
            return

        for i in axes:
            self.spin_idx[i].blockSignals(True)
            self.spin_val[i].blockSignals(True)
            self.spin_idx[i].setValue(self.slicer_area.current_indices[i])
            self.spin_val[i].setValue(self.slicer_area.current_values[i])
            self.spin_idx[i].blockSignals(False)
            self.spin_val[i].blockSignals(False)

        self.spin_dat.setValue(
            self.array_slicer.point_value(self.current_cursor, binned=True)
        )

    @QtCore.Slot(int)
    def update_cursor_count(self, count: int) -> None:
        if count == self.cb_cursors.count():
            return
        if count > self.cb_cursors.count():
            self.addCursor()
        else:
            self.remCursor()

    def addCursor(self) -> None:
        self.cb_cursors.setDisabled(False)
        # self.slicer_area.add_cursor()
        self.cb_cursors.addItem(
            self.slicer_area._cursor_icon(self.current_cursor),
            self.slicer_area._cursor_name(self.current_cursor),
        )
        self.cb_cursors.setCurrentIndex(self.current_cursor)
        self.btn_rem.setDisabled(False)

    def remCursor(self) -> None:
        # self.slicer_area.remove_cursor(self.cb_cursors.currentIndex())
        self.cb_cursors.removeItem(self.cb_cursors.currentIndex())
        for i in range(self.cb_cursors.count()):
            self.cb_cursors.setItemText(i, self.slicer_area._cursor_name(i))
            self.cb_cursors.setItemIcon(i, self.slicer_area._cursor_icon(i))
        self.cb_cursors.setCurrentText(
            self.slicer_area._cursor_name(self.current_cursor)
        )
        if i == 0:
            self.cb_cursors.setDisabled(True)
            self.btn_rem.setDisabled(True)

    @QtCore.Slot(int)
    def cursorChangeEvent(self, idx: int) -> None:
        self.cb_cursors.setCurrentIndex(idx)
        self.update_spins()

    @QtCore.Slot(str)
    def setActiveCursor(self, value: str) -> None:
        self.slicer_area.set_current_cursor(self.cb_cursors.findText(value))


class ItoolColorControls(ItoolControlsBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def initialize_widgets(self) -> None:
        self.btn_reverse = erlab.interactive.utils.IconActionButton(
            self.slicer_area.reverse_act,
            on=_ICON_ALIASES["invert"],
            off=_ICON_ALIASES["invert_off"],
        )
        self.btn_contrast = erlab.interactive.utils.IconActionButton(
            self.slicer_area.high_contrast_act, on=_ICON_ALIASES["contrast"]
        )
        self.btn_zero = erlab.interactive.utils.IconActionButton(
            self.slicer_area.zero_centered_act, on=_ICON_ALIASES["zero_center"]
        )
        self.btn_lock = erlab.interactive.utils.IconActionButton(
            self.slicer_area.lock_levels_act,
            on=_ICON_ALIASES["bright_auto"],
            off=_ICON_ALIASES["bright_percent"],
        )

        layout = typing.cast("QtWidgets.QLayout", self.layout())
        layout.addWidget(self.btn_reverse)
        layout.addWidget(self.btn_contrast)
        layout.addWidget(self.btn_zero)
        layout.addWidget(self.btn_lock)


class ItoolColormapControls(ItoolControlsBase):
    def __init__(
        self, *args, orientation=QtCore.Qt.Orientation.Vertical, **kwargs
    ) -> None:
        if isinstance(orientation, QtCore.Qt.Orientation):
            self.orientation = orientation
        elif orientation == "vertical":
            self.orientation = QtCore.Qt.Orientation.Vertical
        elif orientation == "horizontal":
            self.orientation = QtCore.Qt.Orientation.Horizontal
        super().__init__(*args, **kwargs)

    def initialize_layout(self) -> None:
        if self.orientation == QtCore.Qt.Orientation.Vertical:
            self.setLayout(QtWidgets.QVBoxLayout(self))
        else:
            self.setLayout(QtWidgets.QHBoxLayout(self))

        layout = typing.cast("QtWidgets.QBoxLayout", self.layout())
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

    def initialize_widgets(self) -> None:
        super().initialize_widgets()
        self.cb_colormap = erlab.interactive.colors.ColorMapComboBox(
            self, maximumWidth=175
        )
        self.cb_colormap.textActivated.connect(self.slicer_area.set_colormap)

        self.gamma_widget = erlab.interactive.colors.ColorMapGammaWidget(
            spin_cls=erlab.interactive.utils.BetterSpinBox
        )
        self.gamma_widget.valueChanged.connect(
            lambda g: self.slicer_area.set_colormap(gamma=g)
        )
        self.gamma_widget.slider.sliderPressed.connect(
            self.slicer_area.sigWriteHistory.emit
        )
        self.gamma_widget.spin.editingStarted.connect(
            self.slicer_area.sigWriteHistory.emit
        )
        self.gamma_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )

        self.misc_controls = self.add_control(ItoolColorControls(self))

        layout = typing.cast("QtWidgets.QBoxLayout", self.layout())
        layout.addWidget(self.cb_colormap)
        layout.addWidget(self.gamma_widget)
        layout.addWidget(self.misc_controls)

    def update_content(self) -> None:
        super().update_content()
        if isinstance(self.slicer_area.colormap, str):
            self.cb_colormap.setDefaultCmap(self.slicer_area.colormap)
        self.gamma_widget.blockSignals(True)
        self.gamma_widget.setValue(self.slicer_area.colormap_properties["gamma"])
        self.gamma_widget.blockSignals(False)

    def connect_signals(self) -> None:
        super().connect_signals()
        self.slicer_area.sigViewOptionChanged.connect(self.update_content)

    def disconnect_signals(self) -> None:
        super().disconnect_signals()
        self.slicer_area.sigViewOptionChanged.disconnect(self.update_content)


class ItoolBinningControls(ItoolControlsBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def initialize_widgets(self) -> None:
        super().initialize_widgets()
        self.gridlayout = QtWidgets.QGridLayout()
        self.gridlayout.setContentsMargins(0, 0, 0, 0)
        self.gridlayout.setSpacing(3)

        self.buttonslayout = QtWidgets.QVBoxLayout()
        self.buttonslayout.setContentsMargins(0, 0, 0, 0)
        self.buttonslayout.setSpacing(3)

        layout = typing.cast("QtWidgets.QBoxLayout", self.layout())
        layout.addLayout(self.gridlayout)
        layout.addLayout(self.buttonslayout)

        self.labels = tuple(QtWidgets.QLabel() for _ in range(self.data.ndim))
        self.val_labels = tuple(QtWidgets.QLabel() for _ in range(self.data.ndim))
        self.spins = tuple(
            erlab.interactive.utils.BetterSpinBox(
                self,
                integer=True,
                singleStep=2,
                # minimumWidth=60,
                value=1,
                minimum=1,
                maximum=self.data.shape[i],
                keyboardTracking=False,
            )
            for i in range(self.data.ndim)
        )
        for i, spin in enumerate(self.spins):
            spin.valueChanged.connect(lambda n, axis=i: self._update_bin(axis, n))

        self.reset_btn = erlab.interactive.utils.IconButton(
            _ICON_ALIASES["reset"], toolTip="Reset bins"
        )
        self.reset_btn.clicked.connect(self.reset)

        self.all_btn = erlab.interactive.utils.IconButton(
            on=_ICON_ALIASES["all_cursors"],
            checkable=True,
            toolTip="When checked, apply bins to all cursors upon change",
        )

        height = QtGui.QFontMetrics(self.labels[0].font()).height() + 3

        for i in range(self.data.ndim):
            self.gridlayout.addWidget(self.labels[i], 0, i, 1, 1)
            self.gridlayout.addWidget(self.spins[i], 1, i, 1, 1)
            self.gridlayout.addWidget(self.val_labels[i], 2, i, 1, 1)
            self.val_labels[i].setMaximumHeight(height)
            self.spins[i].setToolTip("Number of bins")
            self.val_labels[i].setToolTip("Value corresponding to number of bins")

        self.reset_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.all_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.buttonslayout.addWidget(self.reset_btn)
        self.buttonslayout.addWidget(self.all_btn)
        # for spin in self.spins:
        # spin.setMinimumWidth(60)

    def _update_bin(self, axis, n) -> None:
        if self.all_btn.isChecked():
            self.slicer_area.set_bin_all(axis, n)
        else:
            self.slicer_area.set_bin(axis, n)

    def connect_signals(self) -> None:
        super().connect_signals()
        self.slicer_area.sigCurrentCursorChanged.connect(self.update_content)
        self.slicer_area.sigBinChanged.connect(self.update_content)
        self.slicer_area.sigDataChanged.connect(self.update_content)
        self.slicer_area.sigShapeChanged.connect(self.update_content)

    def disconnect_signals(self) -> None:
        super().disconnect_signals()
        self.slicer_area.sigCurrentCursorChanged.disconnect(self.update_content)
        self.slicer_area.sigBinChanged.disconnect(self.update_content)
        self.slicer_area.sigDataChanged.disconnect(self.update_content)
        self.slicer_area.sigShapeChanged.disconnect(self.update_content)

    def update_content(self) -> None:
        super().update_content()

        if len(self.val_labels) != self.data.ndim:
            clear_layout(self.layout())
            self.initialize_widgets()

        bin_numbers = self.array_slicer.get_bins(self.current_cursor)
        bin_values = self.array_slicer.get_bin_values(self.current_cursor)

        for i in range(self.data.ndim):
            self.spins[i].blockSignals(True)
            self.labels[i].setText(str(self.data.dims[i]))
            self.spins[i].setRange(1, self.data.shape[i])
            self.spins[i].setValue(bin_numbers[i])
            if bin_values[i] is None:
                self.val_labels[i].setText("")
            else:
                self.val_labels[i].setText(f"{bin_values[i]:.3g}")
            self.spins[i].blockSignals(False)

    def reset(self) -> None:
        for spin in self.spins:
            spin.setValue(1)
