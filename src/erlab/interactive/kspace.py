"""Interactive momentum conversion tool.

.. image:: ../images/ktool_1_light.png
    :align: center
    :alt: Momentum conversion tool screenshot
    :class: only-light

.. only:: format_html

    .. image:: ../images/ktool_1_dark.png
        :align: center
        :alt: Momentum conversion tool screenshot
        :class: only-dark
"""

from __future__ import annotations

__all__ = ["ktool"]

import functools
import importlib.resources
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pydantic
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets, uic

import erlab
from erlab.accessors.kspace import MomentumAccessor

if typing.TYPE_CHECKING:
    import matplotlib
    import varname
    import xarray as xr
else:
    import lazy_loader as _lazy

    matplotlib = _lazy.load("matplotlib")
    varname = _lazy.load("varname")


class _CircleROIControlWidget(QtWidgets.QWidget):
    def __init__(self, roi: _MovableCircleROI) -> None:
        super().__init__()
        self.setGeometry(QtCore.QRect(0, 640, 242, 182))
        self._roi = roi

        layout = QtWidgets.QFormLayout(self)
        self.setLayout(layout)

        self.x_spin = pg.SpinBox(dec=True, compactHeight=False)
        self.y_spin = pg.SpinBox(dec=True, compactHeight=False)
        self.r_spin = pg.SpinBox(dec=True, compactHeight=False)
        self.x_spin.sigValueChanged.connect(self.update_roi)
        self.y_spin.sigValueChanged.connect(self.update_roi)
        self.r_spin.sigValueChanged.connect(self.update_roi)

        layout.addRow("X", self.x_spin)
        layout.addRow("Y", self.y_spin)
        layout.addRow("Radius", self.r_spin)

        self._roi.sigRegionChanged.connect(self.update_spins)

    @QtCore.Slot()
    def update_roi(self) -> None:
        self._roi.blockSignals(True)
        self._roi.set_position(
            (self.x_spin.value(), self.y_spin.value()), self.r_spin.value()
        )
        self._roi.blockSignals(False)

    @QtCore.Slot()
    def update_spins(self) -> None:
        x, y, r = self._roi.get_position()
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.r_spin.blockSignals(True)
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self.r_spin.setValue(r)
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.r_spin.blockSignals(False)

    def setVisible(self, visible: bool) -> None:
        super().setVisible(visible)
        if visible:
            self.update_spins()


class _MovableCircleROI(pg.CircleROI):
    """Circle ROI with a menu to control position and radius."""

    def __init__(self, pos, size=None, radius=None, **args):
        args.setdefault("removable", True)
        super().__init__(pos, size, radius, **args)

    def getMenu(self):
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle("ROI")
            if self.removable:
                remAct = QtGui.QAction("Remove Circle", self.menu)
                remAct.triggered.connect(self.removeClicked)
                self.menu.addAction(remAct)
                self.menu.remAct = remAct
            self._pos_menu = self.menu.addMenu("Edit Circle")
            ctrlAct = QtWidgets.QWidgetAction(self._pos_menu)
            ctrlAct.setDefaultWidget(_CircleROIControlWidget(self))
            self._pos_menu.addAction(ctrlAct)

        return self.menu

    def radius(self) -> float:
        """Radius of the circle."""
        return float(self.size()[0] / 2)

    def center(self) -> tuple[float, float]:
        """Center of the circle."""
        x, y = self.pos()
        r = self.radius()
        return x + r, y + r

    def get_position(self) -> tuple[float, float, float]:
        """Return the center and radius of the circle."""
        return (*self.center(), self.radius())

    def set_position(self, center, radius: float | None = None) -> None:
        """Set the center and radius of the circle."""
        if radius is None:
            radius = self.radius()
        else:
            diameter = 2 * radius
            self.setSize((diameter, diameter), update=False)
        self.setPos(center[0] - radius, center[1] - radius)


class KspaceToolGUI(erlab.interactive.utils.ToolWindow):
    def __init__(
        self,
        avec: npt.NDArray | None = None,
        rotate_bz: float | None = None,
        centering: typing.Literal["P", "A", "B", "C", "F", "I", "R"] | None = None,
        cmap: str | None = None,
        gamma: float | None = None,
    ) -> None:
        # Initialize UI
        super().__init__()
        uic.loadUi(
            str(importlib.resources.files(erlab.interactive).joinpath("ktool.ui")), self
        )
        self.setWindowTitle("Momentum Conversion")

        self.plotitems: tuple[pg.PlotItem, pg.PlotItem] = (pg.PlotItem(), pg.PlotItem())
        self.images: tuple[
            erlab.interactive.utils.xImageItem, erlab.interactive.utils.xImageItem
        ] = (
            erlab.interactive.utils.xImageItem(axisOrder="row-major"),
            erlab.interactive.utils.xImageItem(axisOrder="row-major"),
        )

        for i, plot in enumerate(self.plotitems):
            self.graphics_layout.addItem(plot, i, 0)
            plot.addItem(self.images[i])
            plot.showGrid(x=True, y=True, alpha=0.5)

        opts = erlab.interactive.options.model

        if cmap is None:
            cmap = opts.colors.cmap.name
            if opts.colors.cmap.reverse:
                cmap = f"{cmap}_r"

        if gamma is None:
            gamma = opts.colors.cmap.gamma

        if cmap.endswith("_r"):
            cmap = cmap.removesuffix("_r")
            self.invert_check.setChecked(True)

        # Set up colormap controls
        self.cmap_combo.setDefaultCmap(cmap)
        self.cmap_combo.textActivated.connect(self.update_cmap)
        self.gamma_widget.setValue(gamma)
        self.gamma_widget.valueChanged.connect(self.update_cmap)
        self.invert_check.stateChanged.connect(self.update_cmap)
        self.contrast_check.stateChanged.connect(self.update_cmap)
        self.update_cmap()

        # Set up BZ controls
        self.bz_group.toggled.connect(self.update_bz)
        self.a_spin.valueChanged.connect(self.update_bz)
        self.b_spin.valueChanged.connect(self.update_bz)
        self.c_spin.valueChanged.connect(self.update_bz)
        self.alpha_spin.valueChanged.connect(self.update_bz)
        self.beta_spin.valueChanged.connect(self.update_bz)
        self.gamma_spin.valueChanged.connect(self.update_bz)
        self.rot_spin.valueChanged.connect(self.update_bz)
        self.kz_spin.valueChanged.connect(self.update_bz)
        self.centering_combo.currentIndexChanged.connect(self.update_bz)
        self.points_check.stateChanged.connect(self.update_bz)

        self.plotitems[0].setVisible(False)
        self.angle_plot_check.stateChanged.connect(
            lambda: self.plotitems[0].setVisible(self.angle_plot_check.isChecked())
        )

        self._roi_list: list[_MovableCircleROI] = []
        self.add_circle_btn.clicked.connect(self._add_circle)

        if rotate_bz is None:
            rotate_bz = opts.ktool.bz.default_rot

        with QtCore.QSignalBlocker(self.rot_spin):
            self.rot_spin.setValue(rotate_bz)

        if centering is None:
            centering = opts.ktool.bz.default_centering

        with QtCore.QSignalBlocker(self.centering_combo):
            self.centering_combo.setCurrentText(centering)

        if avec is None:
            avec = erlab.lattice.abc2avec(
                opts.ktool.bz.default_a,
                opts.ktool.bz.default_b,
                opts.ktool.bz.default_c,
                opts.ktool.bz.default_alpha,
                opts.ktool.bz.default_beta,
                opts.ktool.bz.default_gamma,
            )
        self._avec = avec

    @property
    def _avec(self) -> npt.NDArray:
        return erlab.lattice.abc2avec(
            self.a_spin.value(),
            self.b_spin.value(),
            self.c_spin.value(),
            self.alpha_spin.value(),
            self.beta_spin.value(),
            self.gamma_spin.value(),
        )

    @_avec.setter
    def _avec(self, avec: npt.NDArray) -> None:
        if avec.shape == (2, 2):
            avec = np.pad(avec, ((0, 1), (0, 1)))
            avec[2, 2] = 1.0e-15
        a, b, c, alpha, beta, gamma = erlab.lattice.avec2abc(avec)
        with (
            QtCore.QSignalBlocker(self.a_spin),
            QtCore.QSignalBlocker(self.b_spin),
            QtCore.QSignalBlocker(self.c_spin),
            QtCore.QSignalBlocker(self.alpha_spin),
            QtCore.QSignalBlocker(self.beta_spin),
            QtCore.QSignalBlocker(self.gamma_spin),
        ):
            self.a_spin.setValue(a)
            self.b_spin.setValue(b)
            self.c_spin.setValue(c)
            self.alpha_spin.setValue(alpha)
            self.beta_spin.setValue(beta)
            self.gamma_spin.setValue(gamma)

    @QtCore.Slot()
    def _add_circle(self) -> None:
        roi = _MovableCircleROI(
            [-0.3, -0.3], radius=0.3, removable=True, pen=pg.mkPen("m", width=2)
        )
        self.plotitems[1].addItem(roi)
        self._roi_list.append(roi)

        def _remove_roi():
            self.plotitems[1].removeItem(roi)
            self._roi_list.remove(roi)

        roi.sigRemoveRequested.connect(_remove_roi)

    def update_cmap(self) -> None:
        name = self.cmap_combo.currentText()
        for im in self.images:
            im.set_colormap(
                name,
                self.gamma_widget.value(),
                reverse=self.invert_check.isChecked(),
                high_contrast=self.contrast_check.isChecked(),
                update=True,
            )

    def get_bz_lines(self):
        raise NotImplementedError

    def update_bz(self) -> None:
        self.plotitems[1].clearPlots()
        if not self.bz_group.isChecked():
            return

        lines, vertices, midpoints = self.get_bz_lines()
        for line in lines:
            self.plotitems[1].plot(line[:, 0], line[:, 1], pen=pg.mkPen("m", width=2))

        if self.points_check.isChecked():
            self.plotitems[1].plot(
                vertices[:, 0],
                vertices[:, 1],
                symbol="o",
                pen=pg.mkColor(255, 255, 255, 0),
                symbolPen=pg.mkColor(255, 255, 255, 0),
                symbolBrush=pg.mkColor("m"),
                symbolSize=6,
            )
            self.plotitems[1].plot(
                midpoints[:, 0],
                midpoints[:, 1],
                symbol="x",
                pen=pg.mkColor(255, 255, 255, 0),
                symbolPen=pg.mkColor(255, 255, 255, 0),
                symbolBrush=pg.mkColor("m"),
                symbolSize=8,
            )


class KspaceTool(KspaceToolGUI):
    tool_name = "ktool"

    @property
    def preview_imageitem(self) -> pg.ImageItem:
        return self.images[1]

    class StateModel(pydantic.BaseModel):
        data_name: str
        center: float
        width: int
        offsets: dict[str, float]
        bounds_enabled: bool
        bounds: dict[str, float]
        resolution_enabled: bool
        resolution: dict[str, float]
        bz_enabled: bool
        lattice_params: tuple[float, float, float, float, float, float]
        rot: float
        kz: float
        centering: typing.Literal["P", "A", "B", "C", "F", "I", "R"]
        points: bool
        circle_rois: list[tuple[float, float, float]]
        cmap_name: str
        cmap_gamma: float
        cmap_invert: bool
        cmap_highcontrast: bool
        show_angle_plot: bool

    @property
    def info_text(self) -> str:
        from erlab.utils.formatting import (
            format_darr_shape_html,
            format_html_accent,
            format_html_table,
        )

        status = self.tool_status
        info: str = f"<b>{self.tool_name}</b>" + format_darr_shape_html(
            self.tool_data.T
        )
        info += "<b>" + self.config_label.text() + "</b><br>"
        info += "<br><b>Angle Offsets:</b>"

        offsets = self.offset_dict.copy()
        offsets["φ"] = self._work_function
        if self.data.kspace._has_hv:
            offsets["V₀"] = self._inner_potential

        info += format_html_table(
            [
                [
                    format_html_accent(self._OFFSET_LABELS[k], em_space=True),
                    f"{v}{self._OFFSET_UNITS[k]}",
                ]
                for k, v in status.offsets.items()
            ]
        )
        angstrom: str = " Å<sup>−1</sup>"

        bounds = self.bounds
        if bounds:
            info += "<br><b>Bounds:</b>"
            info += format_html_table(
                [
                    [
                        format_html_accent(k, em_space=True),
                        f"{mn}{angstrom}&emsp;",
                        "to&emsp;",
                        f"{mx}{angstrom}",
                    ]
                    for k, (mn, mx) in bounds.items()
                ]
            )

        resolution = self.resolution
        if resolution:
            info += "<br><b>Resolution:</b>"
            info += format_html_table(
                [
                    [format_html_accent(k, em_space=True), f"{v}{angstrom}"]
                    for k, v in resolution.items()
                ]
            )
        return info

    @property
    def tool_status(self) -> StateModel:
        return self.StateModel(
            data_name=self._argnames["data"],
            center=self.center_spin.value(),
            width=self.width_spin.value(),
            offsets={
                k: round(spin.value(), spin.decimals())
                for k, spin in self._offset_spins.items()
            },
            bounds_enabled=self.bounds_supergroup.isChecked(),
            bounds={k: spin.value() for k, spin in self._bound_spins.items()},
            resolution_enabled=self.resolution_supergroup.isChecked(),
            resolution={k: spin.value() for k, spin in self._resolution_spins.items()},
            bz_enabled=self.bz_group.isChecked(),
            lattice_params=erlab.lattice.avec2abc(self._avec),
            rot=self.rot_spin.value(),
            kz=self.kz_spin.value(),
            centering=self.centering_combo.currentText(),
            points=self.points_check.isChecked(),
            circle_rois=[roi.get_position() for roi in self._roi_list],
            cmap_name=self.cmap_combo.currentText(),
            cmap_gamma=self.gamma_widget.value(),
            cmap_invert=self.invert_check.isChecked(),
            cmap_highcontrast=self.contrast_check.isChecked(),
            show_angle_plot=self.angle_plot_check.isChecked(),
        )

    @tool_status.setter
    def tool_status(self, status: StateModel) -> None:
        self._argnames["data"] = status.data_name

        self.center_spin.blockSignals(True)
        self.center_spin.setValue(status.center)
        self.center_spin.blockSignals(False)

        self.width_spin.blockSignals(True)
        self.width_spin.setValue(status.width)
        self.width_spin.blockSignals(False)

        for k, v in status.offsets.items():
            self._offset_spins[k].blockSignals(True)
            self._offset_spins[k].setValue(v)
            self._offset_spins[k].blockSignals(False)

        self.bounds_supergroup.blockSignals(True)
        self.bounds_supergroup.setChecked(status.bounds_enabled)
        self.bounds_supergroup.blockSignals(False)
        for k, v in status.bounds.items():
            self._bound_spins[k].blockSignals(True)
            self._bound_spins[k].setValue(v)
            self._bound_spins[k].blockSignals(False)

        self.resolution_supergroup.blockSignals(True)
        self.resolution_supergroup.setChecked(status.resolution_enabled)
        self.resolution_supergroup.blockSignals(False)
        for k, v in status.resolution.items():
            self._resolution_spins[k].blockSignals(True)
            self._resolution_spins[k].setValue(v)
            self._resolution_spins[k].blockSignals(False)

        # Restore BZ parameters
        with (
            QtCore.QSignalBlocker(self.bz_group),
            QtCore.QSignalBlocker(self.a_spin),
            QtCore.QSignalBlocker(self.b_spin),
            QtCore.QSignalBlocker(self.c_spin),
            QtCore.QSignalBlocker(self.alpha_spin),
            QtCore.QSignalBlocker(self.beta_spin),
            QtCore.QSignalBlocker(self.gamma_spin),
            QtCore.QSignalBlocker(self.rot_spin),
            QtCore.QSignalBlocker(self.kz_spin),
            QtCore.QSignalBlocker(self.centering_combo),
            QtCore.QSignalBlocker(self.points_check),
        ):
            self.bz_group.setChecked(status.bz_enabled)
            self._avec = erlab.lattice.abc2avec(*status.lattice_params)
            self.rot_spin.setValue(status.rot)
            self.kz_spin.setValue(status.kz)
            self.centering_combo.setCurrentText(status.centering)
            self.points_check.setChecked(status.points)

        # Restore circle ROIs
        for x0, y0, radius in status.circle_rois:
            self._add_circle()
            self._roi_list[-1].set_position((x0, y0), radius)

        # Restore colormap
        self.cmap_combo.blockSignals(True)
        self.gamma_widget.blockSignals(True)
        self.invert_check.blockSignals(True)
        self.contrast_check.blockSignals(True)
        self.cmap_combo.setCurrentText(status.cmap_name)
        self.gamma_widget.setValue(status.cmap_gamma)
        self.invert_check.setChecked(status.cmap_invert)
        self.contrast_check.setChecked(status.cmap_highcontrast)
        self.cmap_combo.blockSignals(False)
        self.gamma_widget.blockSignals(False)
        self.invert_check.blockSignals(False)
        self.contrast_check.blockSignals(False)

        self.update()
        self.update_bz()
        self.update_cmap()

        self.angle_plot_check.setChecked(status.show_angle_plot)

    @property
    def tool_data(self) -> xr.DataArray:
        return self.data

    _OFFSET_LABELS: typing.ClassVar[dict[str, str]] = {
        "delta": "𝛿",
        "chi": "𝜒₀",
        "xi": "𝜉₀",
        "beta": "𝛽₀",
        "V0": "V₀",
        "wf": "𝜙",
    }

    _OFFSET_UNITS: typing.ClassVar[dict[str, str]] = {
        "delta": "°",
        "chi": "°",
        "xi": "°",
        "beta": "°",
        "V0": " eV",
        "wf": " eV",
    }

    def __init__(
        self,
        data: xr.DataArray,
        avec: npt.NDArray | None = None,
        rotate_bz: float | None = None,
        centering: typing.Literal["P", "A", "B", "C", "F", "I", "R"] | None = None,
        *,
        cmap: str | None = None,
        gamma: float | None = None,
        data_name: str | None = None,
    ) -> None:
        super().__init__(
            avec=avec, rotate_bz=rotate_bz, centering=centering, cmap=cmap, gamma=gamma
        )

        self._argnames: dict[str, str] = {}

        self._itool: QtWidgets.QWidget | None = None

        if data_name is None:
            try:
                self._argnames["data"] = typing.cast(
                    "str",
                    varname.argname(
                        "data",
                        func=self.__init__,  # type: ignore[misc]
                        vars_only=False,
                    ),
                )
            except varname.VarnameRetrievingError:
                self._argnames["data"] = "data"
        else:
            self._argnames["data"] = data_name

        self.data: xr.DataArray = data.copy(deep=True)

        self.config_label.setText(
            f"Configuration {int(self.data.kspace.configuration)} "
            f"({self.data.kspace.configuration.name})"
        )

        if self.data.kspace._has_eV:
            self.center_spin.setRange(self.data.eV[0], self.data.eV[-1])
            eV_step = self.data.eV.values[1] - self.data.eV.values[0]
            self.center_spin.setDecimals(erlab.utils.array.effective_decimals(eV_step))
            self.center_spin.setSingleStep(eV_step)
            self.width_spin.setRange(1, len(self.data.eV))
            self.center_spin.valueChanged.connect(self.update)
            self.width_spin.valueChanged.connect(self.update)
        else:
            if "eV" in self.data.coords and self.data["eV"].size == 1:
                fixed_energy = float(self.data.eV)
                # Although spinbox will be disabled, setting range and value for
                # displaying the fixed energy value.
                self.center_spin.setRange(fixed_energy - 0.1, fixed_energy + 0.1)
                self.center_spin.setValue(fixed_energy)

            self.energy_group.setDisabled(True)

        self.bounds_supergroup.toggled.connect(self.update)
        self.resolution_supergroup.toggled.connect(self.update)

        self._offset_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}

        for k in self.data.kspace._valid_offset_keys:
            self._offset_spins[k] = QtWidgets.QDoubleSpinBox()
            self._offset_spins[k].setRange(-360, 360)
            self._offset_spins[k].setSingleStep(0.05 if k == "delta" else 0.01)
            self._offset_spins[k].setDecimals(3)
            self._offset_spins[k].setValue(self.data.kspace.offsets[k])
            if (
                k != "delta"
                and k in self.data.coords
                and f"{k}_offset" not in self.data.attrs
            ):
                self._offset_spins[k].setValue(float(self.data[k].mean()))

            self._offset_spins[k].valueChanged.connect(self.update)
            self._offset_spins[k].setSuffix(self._OFFSET_UNITS[k])
            self.offsets_group.layout().addRow(
                self._OFFSET_LABELS[k], self._offset_spins[k]
            )

        if self.data.kspace._has_hv:
            self._offset_spins["V0"] = QtWidgets.QDoubleSpinBox()
            self._offset_spins["V0"].setRange(0, 100)
            self._offset_spins["V0"].setSingleStep(1)
            self._offset_spins["V0"].setDecimals(1)
            self._offset_spins["V0"].setSuffix(self._OFFSET_UNITS[k])
            self._offset_spins["V0"].setToolTip("Inner potential of the sample.")
            with warnings.catch_warnings(action="ignore", category=UserWarning):
                self._offset_spins["V0"].setValue(self.data.kspace.inner_potential)
            self._offset_spins["V0"].valueChanged.connect(self.update)
            self.offsets_group.layout().addRow(
                self._OFFSET_LABELS["V0"], self._offset_spins["V0"]
            )
            self.bz_form.setRowVisible(7, False)
        else:
            self.bz_form.setRowVisible(7, True)

        # Work function spinbox
        self._offset_spins["wf"] = QtWidgets.QDoubleSpinBox()
        self._offset_spins["wf"].setRange(0.0, 9.999)
        self._offset_spins["wf"].setSingleStep(0.01)
        self._offset_spins["wf"].setDecimals(4)
        self._offset_spins["wf"].setSuffix(self._OFFSET_UNITS["wf"])
        self._offset_spins["wf"].setToolTip("Work function of the system.")
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            self._offset_spins["wf"].setValue(self.data.kspace.work_function)
        self._offset_spins["wf"].valueChanged.connect(self.update)

        self.offsets_group.layout().addRow(
            self._OFFSET_LABELS["wf"], self._offset_spins["wf"]
        )

        self._bound_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._resolution_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for k in self.data.kspace.momentum_axes:
            for j in range(2):
                name = f"{k}{j}"
                self._bound_spins[name] = QtWidgets.QDoubleSpinBox()
                if k == "kz":
                    self._bound_spins[name].setRange(0, 100)
                else:
                    self._bound_spins[name].setRange(-10, 10)
                self._bound_spins[name].setSingleStep(0.01)
                self._bound_spins[name].setDecimals(4)
                self._bound_spins[name].valueChanged.connect(self.update)
                self._bound_spins[name].setSuffix(" Å⁻¹")
                self.bounds_group.layout().addRow(name, self._bound_spins[name])

            self._resolution_spins[k] = QtWidgets.QDoubleSpinBox()
            self._resolution_spins[k].setRange(0.0001, 10)
            self._resolution_spins[k].setSingleStep(0.001)
            self._resolution_spins[k].setDecimals(5)
            self._resolution_spins[k].valueChanged.connect(self.update)
            self._resolution_spins[k].setSuffix(" Å⁻¹")
            self.resolution_group.layout().addRow(k, self._resolution_spins[k])

        # Temporary customization for beta scaling
        # self._beta_scale_spin = QtWidgets.QDoubleSpinBox()
        # self._beta_scale_spin.setValue(1.0)
        # self._beta_scale_spin.setDecimals(2)
        # self._beta_scale_spin.setSingleStep(0.01)
        # self._beta_scale_spin.setRange(0.01, 10)
        # self.offsets_group.layout().addRow("scale", self._beta_scale_spin)
        # self._beta_scale_spin.valueChanged.connect(self.update)

        # Populate bounds and resolution
        self.calculate_bounds()
        self.calculate_resolution()

        self.bounds_btn.clicked.connect(self.calculate_bounds)
        self.res_btn.clicked.connect(self.calculate_resolution)
        self.res_npts_check.toggled.connect(self.calculate_resolution)

        if self.data.kspace._has_beta and not self.data.kspace._has_hv:
            for pi in self.plotitems:
                pi.vb.setAspectLocked(lock=True, ratio=1)
        self.open_btn.clicked.connect(self.show_converted)
        self.copy_btn.clicked.connect(self.copy_code)
        self.update()

        if avec is not None:
            self.bz_group.setChecked(True)

    @functools.cached_property
    def _binding_energy(self) -> npt.NDArray:
        return self.data.kspace._binding_energy.values

    @QtCore.Slot()
    def calculate_bounds(self) -> None:
        data = self.data.copy()
        data.kspace.offsets = self.offset_dict
        bounds = data.kspace.estimate_bounds()
        for k in data.kspace.momentum_axes:
            for j in range(2):
                name = f"{k}{j}"
                self._bound_spins[name].blockSignals(True)
                self._bound_spins[name].setValue(bounds[k][j])
                self._bound_spins[name].blockSignals(False)
        self.update()

    @QtCore.Slot()
    def calculate_resolution(self) -> None:
        for k, spin in self._resolution_spins.items():
            spin.setValue(
                self.data.kspace.estimate_resolution(
                    k, from_numpoints=self.res_npts_check.isChecked()
                )
            )

    @property
    def _work_function(self) -> float:
        return float(
            np.round(
                self._offset_spins["wf"].value(), self._offset_spins["wf"].decimals()
            )
        )

    @property
    def _inner_potential(self) -> float:
        return float(
            np.round(
                self._offset_spins["V0"].value(), self._offset_spins["V0"].decimals()
            )
        )

    @QtCore.Slot()
    def show_converted(self) -> None:
        with erlab.interactive.utils.wait_dialog(self, "Converting..."):
            data_kconv = self._assign_params(self.data.copy()).kspace.convert(
                bounds=self.bounds, resolution=self.resolution
            )

        tool = erlab.interactive.itool(data_kconv, execute=False)
        if isinstance(tool, QtWidgets.QWidget):
            if self._itool is not None:
                self._itool.close()
                self._itool.deleteLater()
            self._itool = tool
            self._itool.show()

    @QtCore.Slot()
    def copy_code(self) -> str:
        arg_dict: dict[str, typing.Any] = {}
        if self.bounds is not None:
            arg_dict["bounds"] = self.bounds
        if self.resolution is not None:
            arg_dict["resolution"] = self.resolution

        # Detected input name must be single identifier.
        # Otherwise the generated code will not apply offsets correctly.
        input_name: str = str(self._argnames["data"])
        if not input_name.isidentifier():
            input_name = "data"

        out_lines: list[str] = []

        if self.data.kspace._has_hv:
            v0: float = self._inner_potential
            with warnings.catch_warnings(action="ignore", category=UserWarning):
                if not np.isclose(v0, self.data.kspace.inner_potential):
                    out_lines.append(f"{input_name}.kspace.inner_potential = {v0}")

        wf: float = self._work_function
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            if not np.isclose(wf, self.data.kspace.work_function):
                out_lines.append(f"{input_name}.kspace.work_function = {wf}")

        offset_dict_repr = str(self.offset_dict).replace("'", '"')

        out_lines.extend(
            (
                f"{input_name}.kspace.offsets = {offset_dict_repr}",
                erlab.interactive.utils.generate_code(
                    MomentumAccessor.convert,
                    [],
                    arg_dict,
                    module=f"{input_name}.kspace",
                    assign=f"{input_name}_kconv",
                ),
            )
        )

        return erlab.interactive.utils.copy_to_clipboard(out_lines)

    @property
    def bounds(self) -> dict[str, tuple[float, float]] | None:
        if self.bounds_supergroup.isChecked():
            return {
                k: (
                    float(np.round(self._bound_spins[f"{k}0"].value(), 5)),
                    float(np.round(self._bound_spins[f"{k}1"].value(), 5)),
                )
                for k in self.data.kspace.momentum_axes
            }
        return None

    @property
    def resolution(self) -> dict[str, float] | None:
        if self.resolution_supergroup.isChecked():
            return {
                k: float(np.round(self._resolution_spins[k].value(), 5))
                for k in self.data.kspace.momentum_axes
            }
        return None

    @property
    def offset_dict(self) -> dict[str, float]:
        return {
            k: float(np.round(self._offset_spins[k].value(), 5))
            for k in self.data.kspace._valid_offset_keys
        }

    def _angle_data(self) -> xr.DataArray:
        if self.data.kspace._has_eV:
            data_binding = self.data.copy().assign_coords(eV=self._binding_energy)

            center, width = self.center_spin.value(), self.width_spin.value()
            if width == 1:
                return data_binding.isel(
                    eV=np.argmin(np.abs(self.data.eV.values - center))
                )

            arr = self.data.eV.values
            idx = np.searchsorted((arr[:-1] + arr[1:]) / 2, center)
            return (
                data_binding.isel(
                    eV=slice(idx - width // 2, idx + (width - 1) // 2 + 1)
                )
                .mean("eV", skipna=True, keep_attrs=True)
                .assign_coords(eV=center)
            )
        return self.data.copy()

    def _assign_params(self, data: xr.DataArray) -> xr.DataArray:
        data.kspace.offsets = self.offset_dict

        if self.data.kspace._has_hv:
            v0: float = self._inner_potential
            with warnings.catch_warnings(action="ignore", category=UserWarning):
                if not np.isclose(v0, self.data.kspace.inner_potential):
                    data.kspace.inner_potential = v0

        wf: float = self._work_function
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            if not np.isclose(wf, self.data.kspace.work_function):
                data.kspace.work_function = wf
        return data

    def get_data(self) -> tuple[xr.DataArray, xr.DataArray]:
        # Set angle offsets
        data_ang = self._assign_params(self._angle_data())
        # if "beta" in data_ang.dims:
        #     data_ang = data_ang.assign_coords(
        #         beta=data_ang.beta * self._beta_scale_spin.value()
        #     )

        # Convert to kspace
        data_k = data_ang.kspace.convert(
            bounds=self.bounds, resolution=self.resolution, silent=True
        )
        return data_ang, data_k

    @QtCore.Slot()
    def update(self) -> None:
        ang, k = self.get_data()
        self.images[0].setDataArray(ang.T)
        self.images[1].setDataArray(k.T)
        self.sigInfoChanged.emit()
        if self.bz_group.isChecked():
            self.update_bz()

    def get_bz_lines(
        self,
    ) -> tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]:
        avec_primitive = erlab.lattice.to_primitive(
            self._avec, centering_type=self.centering_combo.currentText()
        )
        bvec = erlab.lattice.to_reciprocal(avec_primitive)
        converted_slice: xr.DataArray | None = self.images[1].data_array
        if converted_slice is None:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))

        rot: float = self.rot_spin.value()

        if self.data.kspace._has_hv:
            kp_dim = next(d for d in converted_slice.dims if d != "kz")
            other = "kx" if kp_dim == "ky" else "ky"

            kp_vals = converted_slice[kp_dim].values
            kz_vals = converted_slice["kz"].values
            lines, vertices, midpoints = erlab.lattice.get_out_of_plane_bz(
                bvec,
                k_parallel=float(converted_slice[other]),
                angle=rot,
                bounds=(kp_vals.min(), kp_vals.max(), kz_vals.min(), kz_vals.max()),
                return_midpoints=True,
            )
        else:
            kx_vals = converted_slice["kx"].values
            ky_vals = converted_slice["ky"].values
            lines, vertices, midpoints = erlab.lattice.get_in_plane_bz(
                bvec,
                kz=self.kz_spin.value() * np.pi / self.c_spin.value(),
                angle=rot,
                bounds=(kx_vals.min(), kx_vals.max(), ky_vals.min(), ky_vals.max()),
                return_midpoints=True,
            )

        return lines, vertices, midpoints


def ktool(
    data: xr.DataArray,
    avec: npt.NDArray | None = None,
    rotate_bz: float | None = None,
    centering: typing.Literal["P", "A", "B", "C", "F", "I", "R"] | None = None,
    *,
    cmap: str | None = None,
    gamma: float | None = None,
    data_name: str | None = None,
    execute: bool | None = None,
) -> KspaceTool:
    """Interactive momentum conversion tool.

    This tool can also be accessed with :meth:`DataArray.kspace.interactive`, or from
    the ``View`` menu of an ImageTool window.

    Parameters
    ----------
    data
        Data to convert. Currently supports constant energy slices (2D data with alpha
        and beta dimensions) and all 3D data that has eV and alpha dimensions, including
        maps and photon energy dependent data.
    avec : array-like, optional
        Real-space lattice vectors as a 2x2 or 3x3 numpy array. If provided, the
        Brillouin zone boundary overlay will be calculated based on these vectors. If
        given as a 2x2 array, the third row and column will be assumed to be all 0. You
        can use utilities from :mod:`erlab.lattice` to construct these vectors.
    rotate_bz
        Rotation angle for the Brillouin zone boundary overlay.
    centering
        Optional centering type to convert the conventional unit cell into a primitive
        one. Must be one of ``"P"`` (primitive), ``"A"``, ``"B"``, ``"C"``, ``"I"``
        (body-centered), ``"F"`` (face-centered), and ``"R"`` (rhombohedral).
    cmap : str, optional
        Name of the colormap to use.
    gamma
        Initial gamma value for the colormap.
    data_name
        Name of the data variable in the generated code. If not provided, the name is
        automatically determined.

    """
    if data_name is None:
        try:
            data_name = str(varname.argname("data", func=ktool, vars_only=False))
        except varname.VarnameRetrievingError:
            data_name = "data"

    with erlab.interactive.utils.setup_qapp(execute):
        win = KspaceTool(
            data,
            avec=avec,
            rotate_bz=rotate_bz,
            centering=centering,
            cmap=cmap,
            gamma=gamma,
            data_name=data_name,
        )
        win.show()
        win.raise_()
        win.activateWindow()

    return win
