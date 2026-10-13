"""The Angle calculator window and its linked momentum and angle views."""

from __future__ import annotations

import dataclasses
import functools
import typing

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._momentum_overlays import _geometry
from erlab.interactive._momentum_overlays._editor import MomentumOverlayEditor
from erlab.interactive._qt_state import qt_window_state, restore_qt_window_state
from erlab.interactive._stateful_window import StatefulWindow
from erlab.interactive.angle_calculator import _calculation as calc
from erlab.interactive.angle_calculator._state import (
    AngleCalculatorSettings,
    AngleCalculatorState,
)
from erlab.interactive.angle_calculator._widgets import (
    DraggableCurveItem,
    NumericRangeControl,
    RichTextCheckBox,
)
from erlab.interactive.utils import qt_is_valid

if typing.TYPE_CHECKING:
    from erlab.interactive._options.schema import AppOptions


class AngleCalculatorWindow(StatefulWindow[AngleCalculatorSettings]):
    """Edit momentum contours and inspect their analyzer-angle coverage."""

    sigStateChanged = QtCore.Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        options: AppOptions | None = None,
    ) -> None:
        super().__init__(parent)
        self._qapp = QtWidgets.QApplication.instance()
        self.setWindowTitle("Angle calculator")
        self.resize(1380, 820)
        if options is None:
            options = erlab.interactive.options.model
        self._show_laser_arpes_controls = (
            options.angle_calculator.show_laser_arpes_controls
        )
        lattice_options = options.ktool.bz
        self._state = AngleCalculatorSettings(
            geometry=calc.Geometry(
                configuration=options.angle_calculator.configuration,
                analyzer_work_function=options.angle_calculator.analyzer_work_function,
                photon_energy=options.angle_calculator.photon_energy,
            ),
            lattice_a=lattice_options.default_a,
            lattice_b=lattice_options.default_b,
            lattice_c=lattice_options.default_c,
            lattice_alpha=lattice_options.default_alpha,
            lattice_beta=lattice_options.default_beta,
            lattice_gamma=lattice_options.default_gamma,
            centering=lattice_options.default_centering,
            lattice_rotation=lattice_options.default_rot,
        )
        self._restoring = True
        self._closed = False
        self._bz_cache: tuple | None = None
        self._out_of_plane_bz_cache: tuple | None = None
        self._pocket_translation_cache: tuple | None = None
        self._out_of_plane_view_initialized = False
        self._curve_drag: tuple[str, float] | None = None
        self._marker_drag = False
        self._range_drag: tuple[NumericRangeControl, AngleCalculatorSettings] | None = (
            None
        )
        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(24)
        self._redraw_timer.timeout.connect(self._redraw)
        self._pocket_redraw_timer = QtCore.QTimer(self)
        self._pocket_redraw_timer.setSingleShot(True)
        self._pocket_redraw_timer.setInterval(24)
        self._pocket_redraw_timer.timeout.connect(self._redraw_pockets)
        self.inputs: dict[
            str, QtWidgets.QDoubleSpinBox | erlab.interactive.utils.BetterSpinBox
        ] = {}
        self._build_ui()
        menu_bar = typing.cast("QtWidgets.QMenuBar", self.menuBar())
        self._edit_menu = typing.cast("QtWidgets.QMenu", menu_bar.addMenu("&Edit"))
        self._edit_menu.addActions((self.undo_action, self.redo_action))
        self._register_history_spinboxes(self.findChildren(QtWidgets.QAbstractSpinBox))
        self.restore_workspace_state(AngleCalculatorState(**self._state.model_dump()))
        self._out_of_plane_view_initialized = False
        for plot, curve in zip(
            (self.momentum_plot, self.angle_plot), self.scan_curves, strict=True
        ):
            plot.getViewBox().autoRange(items=[curve], padding=0.1)

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
        spin: QtWidgets.QDoubleSpinBox | erlab.interactive.utils.BetterSpinBox
        if significant_digits is None:
            spin = QtWidgets.QDoubleSpinBox(self)
            spin.setDecimals(2 if suffix == "°" else decimals)
        else:
            spin = erlab.interactive.utils.BetterSpinBox(
                self, compact=False, decimals=significant_digits, significant=True
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
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.West)
        self.tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.tabs)
        geometry_page = QtWidgets.QWidget()
        self.geometry_layout = geometry_layout = QtWidgets.QFormLayout(geometry_page)
        self.angle_labels: dict[str, QtWidgets.QLabel] = {}
        for name, label, lower, upper, suffix in (
            ("photon_energy", "<i>hν</i>", 0.001, 1000, " eV"),
            ("binding_energy", "<i>E</i><sub>B</sub>", -100000, 0, " eV"),
            ("analyzer_work_function", "𝜙<sub>A</sub>", 0, 100, " eV"),
            ("inner_potential", "V₀", 0, 50, " eV"),
            ("azimuth", "Azimuth", -360, 360, "°"),
            ("polar", "Polar", -90, 90, "°"),
            ("tilt", "Tilt", -90, 90, "°"),
        ):
            spin = self._spin(
                name,
                lower,
                upper,
                getattr(self._state.geometry, name),
                suffix,
                step=1.0
                if name == "photon_energy"
                else 0.01
                if suffix == " eV"
                else 0.1,
                significant_digits=4 if name == "photon_energy" else None,
            )
            if name in ("azimuth", "polar", "tilt"):
                self.angle_labels[name] = QtWidgets.QLabel(label)
                geometry_layout.addRow(self.angle_labels[name], spin)
            else:
                geometry_layout.addRow(label, spin)
            spin.valueChanged.connect(self._geometry_changed)
        for name, tooltip in (
            ("photon_energy", "Photon energy (eV)."),
            (
                "inner_potential",
                "Inner potential for out-of-plane momentum conversion",
            ),
            (
                "binding_energy",
                "Binding energy relative to the Fermi level (eV).\n"
                "Use negative values for occupied states.",
            ),
            (
                "analyzer_work_function",
                "Analyzer work function (eV).\n"
                "Energy from the analyzer Fermi level to vacuum.",
            ),
        ):
            spin = self.inputs[name]
            spin.setToolTip(tooltip)
            geometry_layout.labelForField(spin).setToolTip(tooltip)
        self.da_check = QtWidgets.QCheckBox("Deflectors")
        self.slit_combo = QtWidgets.QComboBox()
        self.slit_combo.addItems(["Vertical", "Horizontal"])
        angle_row, _ = geometry_layout.getWidgetPosition(self.inputs["azimuth"])
        geometry_layout.insertRow(angle_row, self.da_check)
        geometry_layout.insertRow(angle_row + 1, "Slit", self.slit_combo)
        self.da_check.toggled.connect(self._mode_changed)
        self.slit_combo.currentIndexChanged.connect(self._mode_changed)
        self.alpha_control = NumericRangeControl()
        self.scan_control = NumericRangeControl()
        geometry_layout.insertRow(angle_row + 2, "Slit 𝛼", self.alpha_control)
        self.scan_label = QtWidgets.QLabel("Deflector 𝛽")
        geometry_layout.addRow(self.scan_label, self.scan_control)
        self.current_spin = self._spin("current_cut", -90, 90, 0, "°")
        geometry_layout.addRow("Current 𝛽", self.current_spin)
        self.alpha_control.sigValueChanged.connect(self._alpha_changed)
        self.scan_control.sigValueChanged.connect(self._scan_changed)
        self.current_spin.valueChanged.connect(self._current_changed)
        self.out_of_plane_check = RichTextCheckBox(
            "Show α–kz plot",  # noqa: RUF001
            "Show <i>α</i>–<i>k<sub>z</sub></i> plot",  # noqa: RUF001
        )
        self.out_of_plane_check.setObjectName("show_out_of_plane")
        self.out_of_plane_check.setToolTip(
            "Show the photon-energy scan at the current β, with detector α and kz axes."
        )
        geometry_layout.addRow(self.out_of_plane_check)
        self.photon_energy_control = NumericRangeControl(
            bounds=(5, 1000),
            values=self._state.hv_range,
            suffix=" eV",
            decimals=3,
            significant_digits=4,
            step=1.0,
            logarithmic=True,
            snap_to_integer=True,
        )
        geometry_layout.addRow("<i>hν</i> range", self.photon_energy_control)
        self.out_of_plane_check.toggled.connect(self._out_of_plane_toggled)
        self.photon_energy_control.sigValueChanged.connect(
            self._photon_energy_range_changed
        )
        for control in (
            self.alpha_control,
            self.scan_control,
            self.photon_energy_control,
        ):
            control.sigDragStarted.connect(
                functools.partial(self._start_range_drag, control)
            )
            control.sigDragFinished.connect(self._finish_range_drag)
            control.sigDragCanceled.connect(self._cancel_range_drag)
        self.bz_status = QtWidgets.QLabel()
        self.bz_status.setWordWrap(True)
        geometry_layout.addRow(self.bz_status)
        self.photon_energy_control.slider.setToolTip("Logarithmic photon-energy scale")
        self.laser_arpes_separator = QtWidgets.QFrame()
        self.laser_arpes_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        geometry_layout.addRow(self.laser_arpes_separator)
        sample_work_function = self._spin(
            "sample_work_function",
            0,
            100,
            self._state.geometry.sample_work_function,
            " eV",
            step=0.01,
        )
        sample_work_function.setToolTip("Sample surface work function (eV).")
        sample_work_function.valueChanged.connect(self._geometry_changed)
        geometry_layout.addRow("Sample 𝜙ₛ", sample_work_function)
        self.cutoff_check = QtWidgets.QCheckBox("Show sample escape limit")
        self.cutoff_check.setObjectName("sample_escape_limit")
        self.cutoff_check.setToolTip(
            "Show the maximum in-plane momentum that can leave the sample.\n"
            "The angle view shows only the part that can reach the analyzer."
        )
        self.cutoff_check.toggled.connect(self._display_changed)
        geometry_layout.addRow(self.cutoff_check)
        self.correction_check = QtWidgets.QCheckBox("Parallel-plate correction")
        self.correction_check.setObjectName("parallel_plate_correction")
        self.correction_check.setToolTip(
            "Correct the sample-analyzer contact potential at zero applied bias.\n"
            "Assumes parallel planes and a lens response set by electron entry "
            "position.\n"
            "Requires normal sample alignment. Mechanical scans must be fixed at zero."
        )
        self.correction_check.toggled.connect(self._geometry_changed)
        geometry_layout.addRow(self.correction_check)
        self._laser_arpes_summary = QtWidgets.QLabel()
        self._laser_arpes_summary.setObjectName("laser_arpes_summary")
        self._laser_arpes_summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._laser_arpes_summary.setWordWrap(True)
        self._laser_arpes_summary.setToolTip(
            "Enable Laser ARPES mode in Preferences to change these settings."
        )
        geometry_layout.addRow(self._laser_arpes_summary)
        self._add_tab(geometry_page, "Geometry")

        lattice_page = QtWidgets.QWidget()
        lattice_layout = QtWidgets.QFormLayout(lattice_page)
        for name, label, lower, upper, value, suffix in (
            ("lattice_a", "a", 0, 1000, 6.59, " Å"),
            ("lattice_b", "b", 0, 1000, 6.59, " Å"),
            ("lattice_c", "c", 0, 1000, 6.01, " Å"),
            ("lattice_alpha", "α", 0, 180, 90, "°"),
            ("lattice_beta", "β", 0, 180, 90, "°"),
            ("lattice_gamma", "𝛾", 0, 180, 120, "°"),
            ("lattice_rotation", "Lattice rotation", -360, 360, 0, "°"),
        ):
            spin = self._spin(
                name,
                lower,
                upper,
                value,
                suffix,
                5,
                step=0.01 if suffix == " Å" else 0.1,
            )
            spin.valueChanged.connect(
                functools.partial(self._lattice_parameters_changed, name)
            )
            lattice_layout.addRow(label, spin)
        for name, tooltip in (
            ("lattice_a", "Length of the first conventional-cell vector (Å)."),
            ("lattice_b", "Length of the second conventional-cell vector (Å)."),
            (
                "lattice_gamma",
                "Angle between the first two conventional-cell vectors (degrees).",
            ),
        ):
            spin = self.inputs[name]
            spin.setToolTip(tooltip)
            lattice_layout.labelForField(spin).setToolTip(tooltip)
        self.centering_combo = QtWidgets.QComboBox()
        self.centering_combo.setObjectName("lattice_centering")
        self.centering_combo.addItems(["P", "A", "B", "C", "F", "I", "R"])
        self.centering_combo.setCurrentText(self._state.centering)
        self.centering_combo.currentTextChanged.connect(self._centering_changed)
        lattice_layout.addRow("Centering", self.centering_combo)
        self.lattice_status = QtWidgets.QLabel()
        self.lattice_status.setWordWrap(True)
        lattice_layout.addRow(self.lattice_status)
        self.bz_check = QtWidgets.QCheckBox("Show Brillouin zones")
        self.bz_check.setChecked(True)
        self.bz_check.toggled.connect(self._display_changed)
        lattice_layout.addRow(self.bz_check)
        self._add_tab(lattice_page, "Lattice")

        for form, following_widget, name in (
            (geometry_layout, self.da_check, "energy_geometry_separator"),
            (geometry_layout, self.inputs["azimuth"], "slit_sample_separator"),
            (lattice_layout, self.bz_check, "lattice_display_separator"),
        ):
            separator = QtWidgets.QFrame()
            separator.setObjectName(name)
            separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            row, _ = form.getWidgetPosition(following_widget)
            form.insertRow(row, separator)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout, 1)
        self.momentum_plot = self.graphics_layout.addPlot(row=0, col=0)
        self.angle_plot = self.graphics_layout.addPlot(row=0, col=1)
        for plot, xlabel, ylabel in (
            (self.momentum_plot, "kx (Å⁻¹)", "ky (Å⁻¹)"),
            (self.angle_plot, "𝛼 (°)", "𝛽 (°)"),
        ):
            # Keep the outer plot columns equally wide as tick labels change.
            plot.setMinimumWidth(0)
            plot.setPreferredWidth(0)
            plot.setLabel("bottom", xlabel)
            plot.setLabel("left", ylabel)
            plot.getAxis("bottom").enableAutoSIPrefix(False)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.showGrid(x=True, y=True, alpha=0.16)
            plot.setAspectLocked(True)
            plot.disableAutoRange()
            plot.getViewBox().sigRangeChanged.connect(self._view_changed)
        self.out_of_plane_plot = self.graphics_layout.addPlot(row=0, col=2)
        self.out_of_plane_plot.setMinimumWidth(0)
        self.out_of_plane_plot.setPreferredWidth(0)
        self.out_of_plane_plot.setLabel("bottom", "𝛼 (°)")
        self.out_of_plane_plot.setLabel("left", "kz (Å⁻¹)")
        self.out_of_plane_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.out_of_plane_plot.getAxis("left").enableAutoSIPrefix(False)
        self.out_of_plane_plot.showGrid(x=True, y=True, alpha=0.16)
        self.out_of_plane_plot.disableAutoRange()
        self.out_of_plane_plot.getViewBox().sigRangeChanged.connect(self._view_changed)
        self.out_of_plane_bz_curve = self.out_of_plane_plot.plot(
            pen=pg.mkPen("#888888", width=1)
        )
        self._hv_edge_curves = tuple(
            self.out_of_plane_plot.plot(pen=pg.mkPen("#56b4e9", width=1.5))
            for _ in range(2)
        )
        self._hv_labels = tuple(
            pg.TextItem(color="#56b4e9", anchor=(1, 1)) for _ in range(2)
        )
        for label in self._hv_labels:
            self.out_of_plane_plot.addItem(label, ignoreBounds=True)
            label.setZValue(5)
        self._hv_fill = pg.ImageItem(axisOrder="row-major")
        self._hv_fill.setZValue(-5)
        self.out_of_plane_plot.addItem(self._hv_fill, ignoreBounds=True)
        self.out_of_plane_scan_curve = self.out_of_plane_plot.plot(
            pen=pg.mkPen("#56b4e9", width=2)
        )
        self.out_of_plane_current_curve = DraggableCurveItem()
        self.out_of_plane_plot.addItem(self.out_of_plane_current_curve)
        self.overlays = MomentumOverlayEditor(
            self.momentum_plot, projected_plot=self.angle_plot
        )
        self.overlays.sigStateChanged.connect(self._pockets_changed)
        self.overlays.sigDragStarted.connect(self._flush_pending_history_changes)
        self.overlays.sigDragFinished.connect(self._finish_overlay_drag)
        self._add_tab(self.overlays, "Overlays")
        self.scan_curves = tuple(
            plot.plot(pen=pg.mkPen("#56b4e9", width=2))
            for plot in (self.momentum_plot, self.angle_plot)
        )
        self.cut_curve = DraggableCurveItem()
        self.momentum_plot.addItem(self.cut_curve)
        for curve, parameter, tooltip in (
            (self.cut_curve, "beta", "Drag to change the current β."),
            (
                self.out_of_plane_current_curve,
                "photon_energy",
                "Drag to change the photon energy.",
            ),
        ):
            curve.setZValue(10)
            curve.setToolTip(tooltip)
            curve.sigDragStarted.connect(
                functools.partial(self._start_curve_drag, parameter)
            )
            curve.sigDragged.connect(self._drag_curve)
            curve.sigDragFinished.connect(self._finish_curve_drag)
        self.cut_marker = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen("#d55e00", width=2),
            hoverPen=pg.mkPen("#ff8050", width=3),
        )
        self.angle_plot.addItem(self.cut_marker)
        self.cut_marker.sigPositionChanged.connect(self._marker_changed)
        self.cut_marker.sigPositionChangeFinished.connect(self._finish_marker_drag)
        self.cutoff_curves = tuple(
            plot.plot(
                pen=pg.mkPen("#999999", width=2, style=QtCore.Qt.PenStyle.DashLine)
            )
            for plot in (self.momentum_plot, self.angle_plot)
        )
        self.bz_curves = tuple(
            plot.plot(pen=pg.mkPen("#888888", width=1))
            for plot in (self.momentum_plot, self.angle_plot)
        )

    def _add_tab(self, widget, title):
        widget.layout().setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        widget.layout().setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(container)
        self.tabs.addTab(scroll, title)

    def _scan_name(self) -> typing.Literal["deflector", "polar", "tilt"]:
        if self.da_check.isChecked():
            return "deflector"
        return "polar" if self.slit_combo.currentIndex() == 0 else "tilt"

    def _geometry_changed(self, *_args):
        if self._restoring:
            return
        values = {
            field.name: self.inputs[field.name].value()
            for field in dataclasses.fields(calc.Geometry)
            if field.name in self.inputs
        }
        values["configuration"] = (
            1 + self.slit_combo.currentIndex() + (2 if self.da_check.isChecked() else 0)
        )
        values["low_energy_correction"] = self.correction_check.isChecked()
        self._state.geometry = calc.Geometry(**values)
        self._sync_laser_arpes_controls()
        self._changed()

    def _mode_changed(self, *_args):
        if self._restoring:
            return
        self._sync_scan_controls()
        self._geometry_changed()

    def _sync_scan_controls(self):
        name = self._scan_name()
        self.angle_labels["azimuth"].setText("Azimuth 𝛿")
        polar_symbol = {"deflector": "𝜒", "polar": "𝛽", "tilt": "𝜉"}[name]
        self.angle_labels["polar"].setText(f"Polar {polar_symbol}")
        self.angle_labels["tilt"].setText("Tilt 𝛽" if name == "tilt" else "Tilt 𝜉")
        self.inputs["polar"].setEnabled(name != "polar")
        self.inputs["tilt"].setEnabled(name != "tilt")
        self.scan_label.setText(f"{name.capitalize()} 𝛽")
        target_label = (
            self.scan_label if name == "deflector" else self.angle_labels[name]
        )
        target_row, _ = self.geometry_layout.getWidgetPosition(target_label)
        current_row, _ = self.geometry_layout.getWidgetPosition(self.scan_control)
        if current_row != target_row:
            self.geometry_layout.removeWidget(self.scan_control)
            for angle in ("polar", "tilt"):
                spin = self.inputs[angle]
                row, _ = self.geometry_layout.getWidgetPosition(
                    self.angle_labels[angle]
                )
                if angle == name:
                    self.geometry_layout.removeWidget(spin)
                    spin.hide()
                elif self.geometry_layout.indexOf(spin) < 0:
                    self.geometry_layout.setWidget(
                        row, QtWidgets.QFormLayout.ItemRole.FieldRole, spin
                    )
                    spin.show()
            self.geometry_layout.setWidget(
                target_row, QtWidgets.QFormLayout.ItemRole.FieldRole, self.scan_control
            )
        deflector_row, _ = self.geometry_layout.getWidgetPosition(self.scan_label)
        self.geometry_layout.setRowVisible(deflector_row, name == "deflector")
        self.scan_control.show()
        with (
            QtCore.QSignalBlocker(self.scan_control),
            QtCore.QSignalBlocker(self.current_spin),
            QtCore.QSignalBlocker(self.cut_marker),
        ):
            bounds = getattr(self._state, f"{name}_range")
            self.scan_control.setBounds((-45, 45) if name == "deflector" else (-90, 90))
            self.scan_control.setValues(bounds)
            self.current_spin.setRange(*bounds)
            self.current_spin.setValue(self._state.scan_positions[name])
            self.cut_marker.setBounds(bounds)
            self.cut_marker.setValue(self._state.scan_positions[name])

    def _sync_laser_arpes_controls(self) -> None:
        geometry = self._state.geometry
        if geometry.configuration == 1:
            normal = geometry.tilt == 0
        elif geometry.configuration == 2:
            normal = geometry.polar == 0
        else:
            normal = geometry.polar == 0 and geometry.tilt == 0
        if geometry.configuration in (1, 2):
            normal &= self.scan_control.values() == (0.0, 0.0)
        self.correction_check.setEnabled(normal)
        if not normal and geometry.low_energy_correction:
            with QtCore.QSignalBlocker(self.correction_check):
                self.correction_check.setChecked(False)
            self._state.geometry = geometry = dataclasses.replace(
                geometry, low_energy_correction=False
            )
        for widget in (
            self.laser_arpes_separator,
            self.inputs["sample_work_function"],
            self.cutoff_check,
            self.correction_check,
        ):
            self.geometry_layout.setRowVisible(widget, self._show_laser_arpes_controls)
        settings = []
        if geometry.low_energy_correction:
            settings.append("parallel-plate correction")
        if self._state.cutoff_visible:
            settings.append("sample escape limit")
        self._laser_arpes_summary.setText(
            f"Laser ARPES: φS = {geometry.sample_work_function:g} eV; "
            + ", ".join(settings)
        )
        self.geometry_layout.setRowVisible(
            self._laser_arpes_summary,
            not self._show_laser_arpes_controls and bool(settings),
        )

    def _alpha_changed(self, values):
        if not self._restoring:
            self._state.alpha_range = values
            self._changed()

    def _scan_changed(self, values):
        if self._restoring:
            return
        name = self._scan_name()
        setattr(self._state, f"{name}_range", values)
        self._state.scan_positions[name] = float(
            np.clip(self._state.scan_positions[name], *values)
        )
        self._sync_scan_controls()
        self._sync_laser_arpes_controls()
        self._changed()

    def _current_changed(self, value):
        if not self._restoring:
            self._state.scan_positions[self._scan_name()] = value
            with QtCore.QSignalBlocker(self.cut_marker):
                self.cut_marker.setValue(value)
            self._changed()

    def _marker_changed(self, marker):
        if not self._restoring:
            if marker.moving and not self._marker_drag:
                self._flush_pending_history_changes()
                self._marker_drag = True
            self.current_spin.setValue(marker.value())

    def _finish_marker_drag(self, *_args) -> None:
        if self._marker_drag:
            self._marker_drag = False
            self.cut_marker.moving = False
            self._write_state()
            self._flush_pending_history_write()
            self._update_history_actions()

    def _curve_parameter(self, parameter: str, point: QtCore.QPointF) -> float:
        if parameter == "beta":
            return float(
                calc.momentum_to_angles(point.x(), point.y(), self._state.geometry)[1]
            )
        _, hv = calc.alpha_kz_surface(
            [point.x()], [point.y()], self.current_spin.value(), self._state.geometry
        )
        return float(hv[0, 0])

    def _start_curve_drag(self, parameter: str, point: QtCore.QPointF) -> None:
        self._flush_pending_history_changes()
        value = self._curve_parameter(parameter, point)
        if np.isfinite(value):
            spin = self.current_spin if parameter == "beta" else self.inputs[parameter]
            # Keep the parameter unchanged when the press is slightly off the line.
            self._curve_drag = (parameter, spin.value() - value)

    def _drag_curve(self, point: QtCore.QPointF) -> None:
        if self._curve_drag is None:
            return
        parameter, offset = self._curve_drag
        value = self._curve_parameter(parameter, point)
        if np.isfinite(value):
            spin = self.current_spin if parameter == "beta" else self.inputs[parameter]
            spin.setValue(
                float(np.clip(value + offset, spin.minimum(), spin.maximum()))
            )

    def _finish_curve_drag(self) -> None:
        if self._curve_drag is not None:
            self._curve_drag = None
            for curve in (self.cut_curve, self.out_of_plane_current_curve):
                curve.cancelDrag()
            self._write_state()
            self._flush_pending_history_write()
            self._update_history_actions()

    def _start_range_drag(self, control: NumericRangeControl) -> None:
        self._flush_pending_history_changes()
        self._range_drag = control, self._history_state

    def _finish_range_drag(self) -> None:
        if self._range_drag is not None:
            self._range_drag[0].slider.cancelDrag()
            self._range_drag = None
            self._write_state()
            self._flush_pending_history_write()
            self._update_history_actions()

    def _cancel_range_drag(self) -> None:
        if self._range_drag is not None:
            control, state = self._range_drag
            control.slider.cancelDrag()
            # History suppression flushes gestures, so discard this one first.
            self._range_drag = None
            with self._history_suppressed():
                self._history_state = state
            self._update_history_actions()

    def _flush_pending_history_changes(self) -> None:
        self.overlays.finish_drag()
        self._finish_marker_drag()
        self._finish_range_drag()
        self._finish_curve_drag()
        super()._flush_pending_history_changes()

    def _record_state_change(self) -> None:
        if self.overlays.dragging:
            self.undo_action.setEnabled(True)
            self.redo_action.setEnabled(False)
        else:
            super()._record_state_change()

    def _finish_overlay_drag(self) -> None:
        self._write_state()
        self._flush_pending_history_write()
        self._update_history_actions()

    def _display_changed(self, *_args):
        if self._restoring:
            return
        self._state.cutoff_visible = self.cutoff_check.isChecked()
        self._state.bz_visible = self.bz_check.isChecked()
        self._sync_laser_arpes_controls()
        self._changed()

    def _centering_changed(self, value):
        if not self._restoring:
            self._state.centering = value
            self._changed()

    def _out_of_plane_toggled(self, visible):
        if self._restoring:
            return
        self._state.out_of_plane_visible = visible
        self._sync_out_of_plane_controls()
        if visible and not self._out_of_plane_view_initialized:
            self._fit_out_of_plane_range()
            self._out_of_plane_view_initialized = True
        self._changed()

    def _photon_energy_range_changed(self, values):
        if not self._restoring:
            self._state.hv_range = values
            self._changed()

    def _sync_out_of_plane_controls(self):
        visible = self._state.out_of_plane_visible
        self.geometry_layout.setRowVisible(self.photon_energy_control, visible)
        self.out_of_plane_plot.setVisible(visible)
        self.out_of_plane_plot.setMaximumWidth(16777215 if visible else 0)

    def _fit_out_of_plane_range(self):
        alpha = np.linspace(*self._state.alpha_range, 181)
        _, _, kz = calc.angles_to_momentum3d(
            alpha[None, :],
            self.current_spin.value(),
            self._state.geometry,
            photon_energy=np.asarray(self._state.hv_range)[:, None],
        )
        finite = kz[np.isfinite(kz)]
        if finite.size:
            lower, upper = float(finite.min()), float(finite.max())
            if not np.isfinite(kz[0]).all():
                # A scan can start below threshold and still contain valid energies.
                lower = float(
                    erlab.analysis.kspace.kz_func(
                        0.0, self._state.geometry.inner_potential, 0.0, 0.0
                    )
                )
            pad = max((upper - lower) * 0.08, 0.1)
            amin, amax = self._state.alpha_range
            self.out_of_plane_plot.setRange(
                xRange=(amin - 1, amax + 1),
                yRange=(max(0, lower - pad), upper + pad),
                padding=0,
            )

    def _lattice_parameters(self) -> dict[str, float]:
        return {
            name: getattr(self._state, name)
            for name in (
                "lattice_a",
                "lattice_b",
                "lattice_c",
                "lattice_alpha",
                "lattice_beta",
                "lattice_gamma",
                "lattice_rotation",
            )
        }

    def _lattice_parameters_changed(self, name: str, value: float) -> None:
        if self._restoring:
            return
        if name == "lattice_rotation":
            self._state.lattice_rotation = value
            self._changed()
            return
        parameters = self._lattice_parameters()
        parameters.pop("lattice_rotation")
        parameters[name] = value
        lengths = tuple(parameters.values())[:3]
        angles = tuple(parameters.values())[3:]
        if min(lengths) <= 0 or not all(0 < angle < 180 for angle in angles):
            self.lattice_status.setText(
                "Cell lengths must be positive. "
                "Cell angles must be between 0° and 180°."
            )
            return
        vectors = erlab.lattice.abc2avec(*parameters.values())
        if not np.isfinite(vectors).all() or abs(np.linalg.det(vectors)) < 1e-8:
            self.lattice_status.setText(
                "The cell must span a volume. The last valid lattice is shown."
            )
            return
        setattr(self._state, name, value)
        for field, accepted in self._lattice_parameters().items():
            with QtCore.QSignalBlocker(self.inputs[field]):
                self.inputs[field].setValue(accepted)
        self.lattice_status.clear()
        self._changed()

    def _changed(self):
        if self._restoring or self._closed:
            return
        if (
            self._curve_drag is None
            and self._range_drag is None
            and not self._marker_drag
        ):
            self._write_state()
        else:
            # Undo can finish the active gesture before restoring its starting state.
            self.undo_action.setEnabled(True)
            self.redo_action.setEnabled(False)
        self._view_changed()

    def _pockets_changed(self) -> None:
        self._write_state()
        self.sigStateChanged.emit()
        if not self._restoring and not self._closed:
            self._pocket_redraw_timer.start()

    def _view_changed(self, *_args):
        if self._restoring or self._closed or not qt_is_valid(self):
            return
        self._update_cut_marker_span()
        self.sigStateChanged.emit()
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _update_cut_marker_span(self):
        view_alpha = self.angle_plot.viewRange()[0]
        self.cut_marker.setSpan(
            *np.clip(
                (np.array(self._state.alpha_range) - view_alpha[0])
                / (view_alpha[1] - view_alpha[0]),
                0,
                1,
            )
        )
        # InfiniteLine.setSpan() does not invalidate its cached drawing bounds.
        self.cut_marker.viewTransformChanged()

    @staticmethod
    def _set_curve(curve, points, *, angular=False):
        points = np.array(points, dtype=float, copy=True).reshape(-1, 2)
        if angular and len(points) > 1:
            jumps = np.any(np.abs(np.diff(points, axis=0)) > 90, axis=1)
            points[np.flatnonzero(jumps) + 1] = np.nan
        curve.setData(points[:, 0], points[:, 1], connect="finite")

    def _mapped_curve(self, curves, points, geometry):
        self._set_curve(curves[0], points)
        alpha, beta = calc.momentum_to_angles(points[:, 0], points[:, 1], geometry)
        self._set_curve(curves[1], np.column_stack((alpha, beta)), angular=True)

    def _reciprocal_basis(self):
        parameters = self._lattice_parameters()
        parameters.pop("lattice_rotation")
        primitive = erlab.lattice.to_primitive(
            erlab.lattice.abc2avec(*parameters.values()), self._state.centering
        )
        theta = np.deg2rad(self._state.lattice_rotation)
        return erlab.lattice.to_reciprocal(primitive) @ np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def _redraw(self):
        if self._closed or not qt_is_valid(self):
            return
        geometry = self._state.geometry
        alpha = np.linspace(*self._state.alpha_range, 181)
        beta = np.linspace(*self.scan_control.values(), 181)
        boundary = np.vstack(
            (
                np.column_stack((alpha, np.full_like(alpha, beta[0]))),
                np.column_stack((np.full_like(beta, alpha[-1]), beta)),
                np.column_stack((alpha[::-1], np.full_like(alpha, beta[-1]))),
                np.column_stack((np.full_like(beta, alpha[0]), beta[::-1])),
            )
        )
        self._set_curve(self.scan_curves[1], boundary)
        kx, ky = calc.angles_to_momentum(boundary[:, 0], boundary[:, 1], geometry)
        self._set_curve(self.scan_curves[0], np.column_stack((kx, ky)))
        kx, ky = calc.angles_to_momentum(alpha, self.current_spin.value(), geometry)
        self._set_curve(self.cut_curve, np.column_stack((kx, ky)))
        self._update_cut_marker_span()
        if self._state.cutoff_visible:
            radius = calc.emission_radius(geometry)
            if np.isfinite(radius) and radius > 0:
                self._mapped_curve(
                    self.cutoff_curves,
                    _geometry.ellipse_points(
                        (0, 0), (radius * (1 - 1e-14),) * 2, npoints=721
                    ),
                    geometry,
                )
            else:
                for curve in self.cutoff_curves:
                    curve.clear()
        for curve in self.cutoff_curves:
            curve.setVisible(self._state.cutoff_visible)
        basis = self._reciprocal_basis()
        limits = tuple(tuple(axis) for axis in self.momentum_plot.viewRange())
        bz_key = (
            tuple(self._lattice_parameters().values()),
            self._state.centering,
            geometry,
            limits,
        )
        if self._state.bz_visible and (
            self._bz_cache is None or self._bz_cache[0] != bz_key
        ):
            axes = [np.linspace(*bounds, 121) for bounds in limits]
            surface = calc.momentum_surface(*axes, geometry)
            self._bz_cache = (bz_key, self._surface_bz_points(basis, *axes, surface))
        if self._state.bz_visible:
            if self._bz_cache[1] is None:
                for curve in self.bz_curves:
                    curve.clear()
            else:
                self._mapped_curve(self.bz_curves, self._bz_cache[1], geometry)
        for curve in self.bz_curves:
            curve.setVisible(self._state.bz_visible)
        if self._state.out_of_plane_visible:
            self._redraw_out_of_plane(basis, alpha)
        self.overlays.set_projection(
            functools.partial(calc.momentum_to_angles, geometry=geometry),
            None
            if geometry.low_energy_correction and geometry.configuration in (1, 2)
            else functools.partial(calc.angles_to_momentum, geometry=geometry),
            token=geometry,
            max_step=(90.0, 90.0),
        )
        self._redraw_pockets(basis)

    def _redraw_pockets(self, basis=None):
        if self._closed or not qt_is_valid(self):
            return
        self._pocket_redraw_timer.stop()
        translations = self._pocket_translations(
            self._reciprocal_basis() if basis is None else basis
        )
        self.overlays.set_translations(
            np.zeros((1, 2)) if translations is None else translations
        )
        self.overlays.redraw()
        dense = self._state.bz_visible and (
            (self._bz_cache is not None and self._bz_cache[1] is None)
            or (
                self._state.out_of_plane_visible
                and self._out_of_plane_bz_cache is not None
                and self._out_of_plane_bz_cache[1] is None
            )
        )
        self.bz_status.setText(
            "Too many Brillouin zones in view. Zoom in or use smaller cell lengths."
            if dense
            else "Too many pocket copies in view. Zoom in to show periodic copies."
            if translations is None
            else ""
        )

    @staticmethod
    def _surface_within_budget(basis, surface):
        lower, upper = erlab.lattice._surface_bz_translation_bounds(
            basis, surface, pad_cells=1
        )
        cells = upper.astype(float) - lower.astype(float) + 1
        return np.prod(cells) <= 250_000

    @classmethod
    def _surface_bz_points(cls, basis, x, y, surface):
        if not np.isfinite(surface).all(axis=-1).any():
            return np.empty((0, 2))
        # Bound rendering work without changing the physical surface.
        if not cls._surface_within_budget(basis, surface):
            return None
        lines, _ = erlab.lattice.get_surface_bz(basis, x, y, surface)
        return (
            np.vstack([np.vstack((line, [np.nan, np.nan])) for line in lines])
            if lines
            else np.empty((0, 2))
        )

    def _pocket_translations(self, basis):
        pockets = tuple(p for p in self.overlays.state.pockets if p.visible)
        if not pockets:
            return np.zeros((1, 2))
        views = tuple(
            tuple(tuple(axis) for axis in plot.viewRange())
            for plot in (self.momentum_plot, self.angle_plot)
        )
        key = (
            tuple(basis.ravel()),
            self._state.geometry,
            views,
            tuple(
                (p.center, p.radii, p.symmetry, p.point_group, p.symmetry_angle)
                for p in pockets
            ),
        )
        if (
            self._pocket_translation_cache is not None
            and self._pocket_translation_cache[0] == key
        ):
            return self._pocket_translation_cache[1]
        translations = self._visible_pocket_translations(basis, pockets, views)
        self._pocket_translation_cache = key, translations
        return translations

    def _visible_pocket_translations(self, basis, pockets, views):
        geometry = self._state.geometry
        bounds = np.array(views[0])
        # Cover both linked plots, including extrema inside a curved angular view.
        alpha = np.linspace(*views[1][0], 41)
        beta = (
            np.array([0.0])
            if geometry.low_energy_correction and geometry.configuration in (1, 2)
            else np.linspace(*views[1][1], 41)
        )
        kx, ky = calc.angles_to_momentum(alpha[None, :], beta[:, None], geometry)
        valid = np.isfinite(kx) & np.isfinite(ky)
        if valid.any():
            for axis, values in enumerate((kx[valid], ky[valid])):
                bounds[axis, 0] = min(bounds[axis, 0], values.min())
                bounds[axis, 1] = max(bounds[axis, 1], values.max())
        radius = max(max(p.radii) for p in pockets)
        bounds += np.array([-radius, radius])
        work_function = (
            geometry.sample_work_function
            if geometry.low_energy_correction
            else geometry.analyzer_work_function
        )
        energy = geometry.photon_energy + geometry.binding_energy - work_function
        with np.errstate(invalid="ignore"):
            normal_kz = erlab.analysis.kspace.kz_func(
                energy, geometry.inner_potential, 0.0, 0.0
            )
        if not np.isfinite(normal_kz):
            return np.zeros((1, 2))
        axes = [np.linspace(*axis, 121) for axis in bounds]
        surface = calc.momentum_surface(*axes, geometry)
        # Planar guides outside the emission surface use its normal-emission
        # layer. This extends the guides without modelling pocket kz dispersion.
        surface[..., :2] = np.stack(np.meshgrid(*axes), axis=-1)
        surface[..., 2] = np.where(
            np.isfinite(surface[..., 2]), surface[..., 2], normal_kz
        )
        if not self._surface_within_budget(basis, surface):
            return None
        cells, owners = erlab.lattice._surface_bz_cells(basis, surface)
        visible = cells[np.unique(owners[owners >= 0])]
        if not len(visible):
            return np.zeros((1, 2))
        centers = np.array(
            [
                matrix @ p.center
                for p in pockets
                for matrix in _geometry.symmetry_matrices(
                    p.point_group if p.symmetry else "1", p.symmetry_angle
                )
            ]
        )
        with np.errstate(invalid="ignore"):
            kz = erlab.analysis.kspace.kz_func(
                energy, geometry.inner_potential, centers[:, 0], centers[:, 1]
            )
        references = np.column_stack(
            (centers, np.where(np.isfinite(kz), kz, normal_kz))
        )[None, :, :]
        if not self._surface_within_budget(basis, references):
            return None
        cells, owners = erlab.lattice._surface_bz_cells(basis, references)
        reference_cells = cells[np.unique(owners[owners >= 0])]
        if len(visible) * len(reference_cells) > 50_000:
            return None
        shifts = (visible[:, None, :2] - reference_cells[None, :, :2]).reshape(-1, 2)
        shifts = np.vstack((np.zeros((1, 2)), shifts))
        _, unique = np.unique(np.round(shifts, 10), axis=0, return_index=True)
        shifts = shifts[unique]
        if len(shifts) * len(centers) > 5000:
            return None
        return shifts

    def _redraw_out_of_plane(self, basis, alpha):
        geometry = self._state.geometry
        beta = self.current_spin.value()
        _, _, edge_kz = calc.angles_to_momentum3d(
            alpha[None, :],
            beta,
            geometry,
            photon_energy=np.asarray(self._state.hv_range)[:, None],
        )
        for curve, label, energy, kz in zip(
            self._hv_edge_curves,
            self._hv_labels,
            self._state.hv_range,
            edge_kz,
            strict=True,
        ):
            curve.setData(alpha, kz, connect="finite")
            finite = np.flatnonzero(np.isfinite(kz))
            label.setVisible(bool(finite.size))
            if finite.size:
                label.setText(f"{energy:g} eV")
                label.setPos(alpha[finite[-1]], kz[finite[-1]])
        boundary = np.column_stack(
            (
                np.r_[alpha, alpha[::-1], alpha[0]],
                np.r_[edge_kz[0], edge_kz[1, ::-1], edge_kz[0, 0]],
            )
        )
        self._set_curve(self.out_of_plane_scan_curve, boundary)
        _, _, current_kz = calc.angles_to_momentum3d(alpha, beta, geometry)
        self.out_of_plane_current_curve.setData(alpha, current_kz, connect="finite")
        limits = tuple(tuple(axis) for axis in self.out_of_plane_plot.viewRange())
        key = (tuple(basis.ravel()), geometry, beta, limits, self._state.hv_range)
        axes = [np.linspace(*bounds, 121) for bounds in limits]
        surface, hv = calc.alpha_kz_surface(*axes, beta, geometry)
        lower, upper = self._state.hv_range
        coverage = (
            np.isfinite(hv)
            & (hv >= lower)
            & (hv <= upper)
            & (axes[0][None, :] >= self._state.alpha_range[0])
            & (axes[0][None, :] <= self._state.alpha_range[1])
        )
        self._hv_fill.setImage(
            np.where(coverage[..., None], (86, 180, 233, 55), (0, 0, 0, 0)).astype(
                np.uint8
            ),
            autoLevels=False,
        )
        (x0, x1), (y0, y1) = limits
        dx, dy = (x1 - x0) / 120, (y1 - y0) / 120
        self._hv_fill.setRect(
            QtCore.QRectF(x0 - dx / 2, y0 - dy / 2, x1 - x0 + dx, y1 - y0 + dy)
        )
        if self._state.bz_visible and (
            self._out_of_plane_bz_cache is None or self._out_of_plane_bz_cache[0] != key
        ):
            margin = max((upper - lower) * 0.1, 1.0)
            # Grazing rays can require unbounded photon energies outside this scan.
            surface[(hv < max(0.001, lower - margin)) | (hv > upper + margin)] = np.nan
            self._out_of_plane_bz_cache = (
                key,
                self._surface_bz_points(basis, *axes, surface),
            )
        if self._state.bz_visible:
            points = self._out_of_plane_bz_cache[1]
            if points is None:
                self.out_of_plane_bz_curve.clear()
            else:
                self._set_curve(self.out_of_plane_bz_curve, points)
        self.out_of_plane_bz_curve.setVisible(self._state.bz_visible)

    @property
    def _history_state(self) -> AngleCalculatorSettings:
        state = self._state.model_copy(deep=True)
        state.overlays = self.overlays.state.model_copy(deep=True)
        return state

    @_history_state.setter
    def _history_state(self, state: AngleCalculatorSettings) -> None:
        self._restore_settings(state)
        self.sigStateChanged.emit()

    def workspace_state_payload(self) -> dict[str, typing.Any]:
        return AngleCalculatorState(
            **self._history_state.model_dump(),
            momentum_range=self.momentum_plot.viewRange(),
            angle_range=self.angle_plot.viewRange(),
            out_of_plane_range=self.out_of_plane_plot.viewRange(),
            window_state=qt_window_state(self),
        ).model_dump(mode="json")

    def restore_workspace_state(
        self, state: AngleCalculatorState | dict[str, typing.Any]
    ) -> None:
        state = AngleCalculatorState.model_validate(state).model_copy(deep=True)
        self._restore_settings(state)
        self._restoring = True
        try:
            self.momentum_plot.setRange(
                xRange=state.momentum_range[0],
                yRange=state.momentum_range[1],
                padding=0,
            )
            self.angle_plot.setRange(
                xRange=state.angle_range[0], yRange=state.angle_range[1], padding=0
            )
            self.out_of_plane_plot.setRange(
                xRange=state.out_of_plane_range[0],
                yRange=state.out_of_plane_range[1],
                padding=0,
            )
            restore_qt_window_state(self, state.window_state)
        finally:
            self._restoring = False
        self._out_of_plane_view_initialized = True
        self._reset_history_stack()
        self._redraw()

    def _restore_settings(self, state: AngleCalculatorSettings) -> None:
        self.overlays.finish_drag(notify=False)
        self._marker_drag = False
        self.cut_marker.moving = False
        if self._range_drag is not None:
            self._range_drag[0].slider.cancelDrag()
            self._range_drag = None
        self._curve_drag = None
        for curve in (self.cut_curve, self.out_of_plane_current_curve):
            curve.cancelDrag()
        self._restoring = True
        try:
            self._state = AngleCalculatorSettings.model_validate(state.model_dump())
            for field in dataclasses.fields(calc.Geometry):
                if field.name in self.inputs:
                    self.inputs[field.name].setValue(
                        getattr(state.geometry, field.name)
                    )
            self.da_check.setChecked(state.geometry.configuration > 2)
            self.slit_combo.setCurrentIndex((state.geometry.configuration - 1) % 2)
            self.correction_check.setChecked(state.geometry.low_energy_correction)
            self.alpha_control.setValues(state.alpha_range)
            self.centering_combo.setCurrentText(state.centering)
            self.out_of_plane_check.setChecked(state.out_of_plane_visible)
            self.photon_energy_control.setValues(state.hv_range)
            self._sync_out_of_plane_controls()
            self._sync_scan_controls()
            self.cutoff_check.setChecked(state.cutoff_visible)
            self._sync_laser_arpes_controls()
            self.bz_check.setChecked(state.bz_visible)
            for name, value in self._lattice_parameters().items():
                self.inputs[name].setValue(value)
            self.lattice_status.clear()
            self.overlays.restore_state(state.overlays)
        finally:
            self._restoring = False
        self._redraw()

    def closeEvent(self, event):
        super().closeEvent(event)
        for curve in (self.cut_curve, self.out_of_plane_current_curve):
            curve.cancelDrag()
        self._closed = True
        self._redraw_timer.stop()
        self._pocket_redraw_timer.stop()
        self.overlays.set_active(False)

    def showEvent(self, event):
        self._closed = False
        self.overlays.set_active(True)
        super().showEvent(event)
        self._view_changed()
