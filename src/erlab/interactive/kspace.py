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

import hashlib
import importlib.resources
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pydantic
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

import erlab
from erlab.accessors.kspace import IncompleteDataError, MomentumAccessor
from erlab.constants import AxesConfiguration

if typing.TYPE_CHECKING:
    from collections.abc import Hashable

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
    _PREVIEW_SYMMETRY_ABS_TOL = 0.05
    _PREVIEW_SYMMETRY_REL_TOL = 0.02
    _PREVIEW_SYMMETRY_ANGLE_TOL = 5.0

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
        erlab.interactive.utils.load_ui(
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
        self.cmap_combo._populate()
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
        self.preview_symmetry_group.setEnabled(False)

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
        with QtCore.QSignalBlocker(self.preview_symmetry_fold_spin):
            self.preview_symmetry_fold_spin.setValue(
                self._default_preview_symmetry_fold()
            )

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

    def _default_preview_symmetry_fold(self) -> int:
        a, b, _, _, _, gamma = erlab.lattice.avec2abc(self._avec)
        same_in_plane = np.isclose(
            a,
            b,
            rtol=self._PREVIEW_SYMMETRY_REL_TOL,
            atol=self._PREVIEW_SYMMETRY_ABS_TOL,
        )
        if same_in_plane and np.isclose(
            gamma, 120.0, atol=self._PREVIEW_SYMMETRY_ANGLE_TOL
        ):
            return 6
        if same_in_plane and np.isclose(
            gamma, 90.0, atol=self._PREVIEW_SYMMETRY_ANGLE_TOL
        ):
            return 4
        return 2

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
    _sigTriggerUpdate = QtCore.Signal()
    _UPDATE_LIMIT_HZ = 10.0

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
        preview_symmetry_enabled: bool = False
        preview_symmetry_fold: int | None = None
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
            preview_symmetry_enabled=self.preview_symmetry_group.isChecked(),
            preview_symmetry_fold=self.preview_symmetry_fold_spin.value(),
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
        self._sync_normal_emission_spins()

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
        for roi in list(self._roi_list):
            self.plotitems[1].removeItem(roi)
            roi.deleteLater()
        self._roi_list.clear()
        for x0, y0, radius in status.circle_rois:
            self._add_circle()
            self._roi_list[-1].set_position((x0, y0), radius)

        # Restore colormap
        with (
            QtCore.QSignalBlocker(self.cmap_combo),
            QtCore.QSignalBlocker(self.gamma_widget),
            QtCore.QSignalBlocker(self.invert_check),
            QtCore.QSignalBlocker(self.contrast_check),
        ):
            self.cmap_combo.setCurrentText(status.cmap_name)
            self.gamma_widget.setValue(status.cmap_gamma)
            self.invert_check.setChecked(status.cmap_invert)
            self.contrast_check.setChecked(status.cmap_highcontrast)

        with (
            QtCore.QSignalBlocker(self.preview_symmetry_group),
            QtCore.QSignalBlocker(self.preview_symmetry_fold_spin),
        ):
            if status.preview_symmetry_fold is not None:
                self.preview_symmetry_fold_spin.setValue(status.preview_symmetry_fold)
            self.preview_symmetry_group.setChecked(status.preview_symmetry_enabled)

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
    _NORMAL_EMISSION_LABELS: typing.ClassVar[dict[str, str]] = {
        "alpha": "𝛼",
        "beta": "𝛽",
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
        initial_normal_emission: tuple[float, float] | None = None,
        initial_delta: float | None = None,
    ) -> None:
        super().__init__(
            avec=avec, rotate_bz=rotate_bz, centering=centering, cmap=cmap, gamma=gamma
        )

        self._argnames: dict[str, str] = {}

        self._itool: QtWidgets.QWidget | None = None
        self._bz_cache_key: typing.Any = None
        self._bz_cache_value: (
            tuple[
                list[npt.NDArray[np.floating]],
                npt.NDArray[np.floating],
                npt.NDArray[np.floating],
            ]
            | None
        ) = None

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
        self._update_proxy = pg.SignalProxy(
            self._sigTriggerUpdate,
            delay=1 / self._UPDATE_LIMIT_HZ,
            rateLimit=self._UPDATE_LIMIT_HZ,
            slot=self._flush_debounced_update,
        )
        self._energy_controls_connected: bool = False

        if self.data.kspace._has_eV and self.data.eV.size > 1:
            self._update_energy_controls()
            self.width_spin.setRange(1, len(self.data.eV))
            self._ensure_energy_control_connections()
        else:
            if "eV" in self.data.coords and self.data["eV"].size == 1:
                fixed_energy = float(self.data.eV)
                # Although spinbox will be disabled, setting range and value for
                # displaying the fixed energy value.
                self.center_spin.setRange(fixed_energy - 0.1, fixed_energy + 0.1)
                self.center_spin.setValue(fixed_energy)

            self.energy_group.setDisabled(True)

        self.bounds_supergroup.toggled.connect(self.queue_update)
        self.resolution_supergroup.toggled.connect(self.queue_update)
        self.preview_symmetry_group.toggled.connect(self.queue_update)
        self.preview_symmetry_fold_spin.valueChanged.connect(self.queue_update)

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

            self._offset_spins[k].valueChanged.connect(self.queue_update)
            self._offset_spins[k].setSuffix(self._OFFSET_UNITS[k])
            self.offsets_group.layout().addRow(
                self._OFFSET_LABELS[k], self._offset_spins[k]
            )

        if self.data.kspace._has_hv:
            self._offset_spins["V0"] = QtWidgets.QDoubleSpinBox()
            self._offset_spins["V0"].setRange(0, 100)
            self._offset_spins["V0"].setSingleStep(1)
            self._offset_spins["V0"].setDecimals(1)
            self._offset_spins["V0"].setSuffix(self._OFFSET_UNITS["V0"])
            self._offset_spins["V0"].setToolTip("Inner potential of the sample.")
            with warnings.catch_warnings(action="ignore", category=UserWarning):
                self._offset_spins["V0"].setValue(self.data.kspace.inner_potential)
            self._offset_spins["V0"].valueChanged.connect(self.queue_update)
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
        self._offset_spins["wf"].valueChanged.connect(self._update_energy_controls)
        self._offset_spins["wf"].valueChanged.connect(self.queue_update)

        self.offsets_group.layout().addRow(
            self._OFFSET_LABELS["wf"], self._offset_spins["wf"]
        )

        self._normal_emission_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for axis, label in self._NORMAL_EMISSION_LABELS.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-360, 360)
            spin.setSingleStep(0.01)
            spin.setDecimals(3)
            spin.setSuffix("°")
            spin.setKeyboardTracking(False)
            spin.setToolTip("Angle corresponding to sample normal emission.")
            spin.valueChanged.connect(self._update_offsets_from_normal_emission)
            self._normal_emission_spins[axis] = spin
            self.normal_emission_group.layout().addRow(label, spin)

        for k in self.data.kspace._valid_offset_keys:
            self._offset_spins[k].valueChanged.connect(self._sync_normal_emission_spins)

        self._sync_normal_emission_spins()
        if initial_normal_emission is not None:
            self.data.kspace.set_normal(
                initial_normal_emission[0],
                initial_normal_emission[1],
                delta=initial_delta,
            )
            for key in self.data.kspace._valid_offset_keys:
                with QtCore.QSignalBlocker(self._offset_spins[key]):
                    self._offset_spins[key].setValue(self.data.kspace.offsets[key])
            self._sync_normal_emission_spins()

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
                self._bound_spins[name].valueChanged.connect(self.queue_update)
                self._bound_spins[name].setSuffix(" Å⁻¹")
                self.bounds_group.layout().addRow(name, self._bound_spins[name])

            self._resolution_spins[k] = QtWidgets.QDoubleSpinBox()
            self._resolution_spins[k].setRange(0.0001, 10)
            self._resolution_spins[k].setSingleStep(0.001)
            self._resolution_spins[k].setDecimals(5)
            self._resolution_spins[k].valueChanged.connect(self.queue_update)
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

    def _ensure_energy_control_connections(self) -> None:
        if self._energy_controls_connected:
            return
        self.center_spin.valueChanged.connect(self.queue_update)
        self.width_spin.valueChanged.connect(self.queue_update)
        self._energy_controls_connected = True

    def update_data(self, new_data: xr.DataArray) -> None:
        status = self.tool_status
        new_data = self.validate_update_data(new_data)

        self.data = new_data.copy(deep=True)
        self._bz_cache_key = None
        self._bz_cache_value = None
        self.config_label.setText(
            f"Configuration {int(self.data.kspace.configuration)} "
            f"({self.data.kspace.configuration.name})"
        )

        if self.data.kspace._has_eV and self.data.eV.size > 1:
            self.energy_group.setDisabled(False)
            self._update_energy_controls()
            self.width_spin.setRange(1, len(self.data.eV))
            self._ensure_energy_control_connections()
        else:
            if "eV" in self.data.coords and self.data["eV"].size == 1:
                fixed_energy = float(self.data.eV)
                self.center_spin.setRange(fixed_energy - 0.1, fixed_energy + 0.1)
                self.center_spin.setValue(fixed_energy)
            self.energy_group.setDisabled(True)

        self.tool_status = status
        self._notify_data_changed()

    def validate_update_data(self, new_data: xr.DataArray) -> xr.DataArray:
        data = erlab.interactive.utils.parse_data(new_data)
        current_offset_keys = tuple(self.data.kspace._valid_offset_keys)
        current_momentum_axes = tuple(self.data.kspace.momentum_axes)
        current_configuration = int(self.data.kspace.configuration)
        current_has_hv = self.data.kspace._has_hv

        if tuple(data.kspace._valid_offset_keys) != current_offset_keys:
            raise ValueError("Updated data has incompatible offset coordinates.")
        if tuple(data.kspace.momentum_axes) != current_momentum_axes:
            raise ValueError("Updated data has incompatible momentum axes.")
        if int(data.kspace.configuration) != current_configuration:
            raise ValueError("Updated data has incompatible analyzer configuration.")
        if data.kspace._has_hv != current_has_hv:
            raise ValueError("Updated data has incompatible photon-energy dimensions.")
        return data

    def _binding_energy(self) -> npt.NDArray[np.floating]:
        if hasattr(self, "_offset_spins") and "wf" in self._offset_spins:
            work_function = self._work_function
        else:
            work_function = self.data.kspace.work_function

        if self.data.kspace._is_energy_kinetic:
            if self.data.kspace._has_hv:
                raise ValueError(
                    "Energy axis of photon energy dependent data must be in "
                    "binding energy."
                )
            return np.asarray(
                self.data.eV.values - float(self.data.hv.values) + work_function
            )
        return np.asarray(self.data.eV.values)

    @QtCore.Slot()
    def _update_energy_controls(self) -> None:
        if not self.data.kspace._has_eV or self.data.eV.size <= 1:
            return

        energy_axis = self._binding_energy()
        e_min = float(np.nanmin(energy_axis))
        e_max = float(np.nanmax(energy_axis))
        e_step = float(np.nanmedian(np.abs(np.diff(energy_axis))))

        self.center_spin.blockSignals(True)
        self.center_spin.setRange(e_min, e_max)
        self.center_spin.setDecimals(erlab.utils.array.effective_decimals(e_step))
        self.center_spin.setSingleStep(e_step)
        self.center_spin.blockSignals(False)

    @QtCore.Slot()
    def calculate_bounds(self) -> None:
        data = self._assign_params(self.data.copy(deep=False))
        self._validate_kinetic_energy(
            data, context="estimating momentum bounds in ktool"
        )
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
        data = self._assign_params(self.data.copy(deep=False))
        self._validate_kinetic_energy(
            data, context="estimating momentum resolution in ktool"
        )
        for k, spin in self._resolution_spins.items():
            with QtCore.QSignalBlocker(spin):
                spin.setValue(
                    data.kspace.estimate_resolution(
                        k, from_numpoints=self.res_npts_check.isChecked()
                    )
                )
        self.update()

    @QtCore.Slot()
    def queue_update(self) -> None:
        self._sigTriggerUpdate.emit()

    @QtCore.Slot(object)
    def _flush_debounced_update(self, _args: object) -> None:
        self.update()

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

    def _converted_output(self) -> xr.DataArray:
        data = self._assign_params(self.data.copy(deep=False))
        self._validate_kinetic_energy(data, context="opening converted data from ktool")
        return data.kspace.convert(bounds=self.bounds, resolution=self.resolution)

    @QtCore.Slot()
    def show_converted(self) -> None:
        with erlab.interactive.utils.wait_dialog(self, "Converting..."):
            data_kconv = self._converted_output()

        tool = self._launch_output_imagetool(
            data_kconv, slot_key="ktool.converted_output"
        )
        if tool is not None:
            self._itool = tool

    def _build_copy_code(self, *, input_name: str | None = None) -> str:
        arg_dict: dict[str, typing.Any] = {}
        if self.bounds is not None:
            arg_dict["bounds"] = self.bounds
        if self.resolution is not None:
            arg_dict["resolution"] = self.resolution

        if input_name is None:
            # Detected input name must be single identifier.
            # Otherwise the generated code will not apply offsets correctly.
            input_name = str(self._argnames["data"])
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

        alpha_normal, beta_normal = self._current_normal_emission_angles()
        delta_offset = self.offset_dict["delta"]

        out_lines.extend(
            (
                f"{input_name}.kspace.set_normal("
                f"alpha={alpha_normal!r}, beta={beta_normal!r}, delta={delta_offset!r}"
                f")",
                erlab.interactive.utils.generate_code(
                    MomentumAccessor.convert,
                    [],
                    arg_dict,
                    module=f"{input_name}.kspace",
                    assign=f"{input_name}_kconv",
                ),
            )
        )

        return "\n".join(out_lines)

    def _copy_output_name(self, input_name: str | None) -> str:
        if input_name is None:
            input_name = str(self._argnames["data"])
            if not input_name.isidentifier():
                input_name = "data"
        return f"{input_name}_kconv"

    def current_provenance_spec(
        self,
    ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
        return self._compose_with_input_provenance(
            lambda input_name: erlab.interactive.imagetool.provenance.script(
                erlab.interactive.imagetool.provenance.ScriptCodeOperation(
                    label="Convert to momentum space",
                    code=self._build_copy_code(input_name=input_name),
                ),
                start_label="Start from current ktool input data",
                active_name=self._copy_output_name(input_name),
            )
        )

    def output_imagetool_data(self, slot_key: Hashable) -> xr.DataArray | None:
        if slot_key != "ktool.converted_output":
            return None
        return self._converted_output()

    def output_imagetool_provenance(
        self, slot_key: Hashable, data: xr.DataArray
    ) -> erlab.interactive.imagetool.provenance.ToolProvenanceSpec | None:
        if slot_key != "ktool.converted_output":
            return None
        return self._compose_with_input_provenance(
            lambda input_name: erlab.interactive.imagetool.provenance.script(
                erlab.interactive.imagetool.provenance.ScriptCodeOperation(
                    label="Convert current data to momentum space",
                    code=self._build_copy_code(input_name=input_name),
                ),
                start_label="Start from current ktool input data",
                active_name=self._copy_output_name(input_name),
            )
        )

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

    def _current_normal_emission_angles(self) -> tuple[float, float]:
        if "xi" not in self.data.coords:
            raise IncompleteDataError("coord", "xi")

        offsets = self.offset_dict
        angle_params = {
            "delta": offsets["delta"],
            "xi": float(self.data["xi"].values),
            "xi0": offsets["xi"],
        }

        match self.data.kspace.configuration:
            case AxesConfiguration.Type1 | AxesConfiguration.Type2:
                angle_params["beta0"] = offsets["beta"]
            case _:
                if "chi" not in self.data.coords:
                    raise IncompleteDataError("coord", "chi")
                angle_params["chi"] = float(self.data["chi"].values)
                angle_params["chi0"] = offsets["chi"]

        alpha, beta = erlab.analysis.kspace._normal_emission_from_angle_params(
            self.data.kspace.configuration, angle_params
        )
        return float(np.round(alpha, 5)), float(np.round(beta, 5))

    @QtCore.Slot()
    def _sync_normal_emission_spins(self) -> None:
        alpha_normal, beta_normal = self._current_normal_emission_angles()

        for axis, value in {"alpha": alpha_normal, "beta": beta_normal}.items():
            spin = self._normal_emission_spins[axis]
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    @QtCore.Slot()
    def _update_offsets_from_normal_emission(self) -> None:
        data = self._assign_params(self.data.copy(deep=False))
        data.kspace.set_normal(
            self._normal_emission_spins["alpha"].value(),
            self._normal_emission_spins["beta"].value(),
        )

        for key in data.kspace._valid_offset_keys:
            spin = self._offset_spins[key]
            spin.blockSignals(True)
            spin.setValue(data.kspace.offsets[key])
            spin.blockSignals(False)

        self._sync_normal_emission_spins()
        self.queue_update()

    def _angle_data(self) -> xr.DataArray:
        if self.data.kspace._has_eV:
            binding_energy = self._binding_energy()
            data_binding = self.data.copy(deep=False).assign_coords(eV=binding_energy)

            center, width = self.center_spin.value(), self.width_spin.value()
            arr = binding_energy
            idx = int(np.argmin(np.abs(arr - center)))
            if width == 1:
                return data_binding.isel(eV=idx)

            start = max(0, idx - width // 2)
            stop = min(arr.size, idx + (width - 1) // 2 + 1)
            if start >= stop:
                start = int(np.clip(idx, 0, arr.size - 1))
                stop = int(np.clip(idx + 1, 1, arr.size))
            return (
                data_binding.isel(eV=slice(start, stop))
                .mean("eV", skipna=True, keep_attrs=True)
                .assign_coords(eV=center)
            )
        return self.data.copy(deep=False)

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

    def _validate_kinetic_energy(self, data: xr.DataArray, *, context: str) -> None:
        data.kspace._check_kinetic_energy(context=context)

    @staticmethod
    def _preview_supports_symmetry(preview_data: xr.DataArray) -> bool:
        return preview_data.ndim == 2 and set(preview_data.dims) == {"kx", "ky"}

    def _set_preview_symmetry_available(self, available: bool) -> None:
        if not available:
            with QtCore.QSignalBlocker(self.preview_symmetry_group):
                self.preview_symmetry_group.setChecked(False)
        self.preview_symmetry_group.setEnabled(available)

    def _symmetrized_preview(self, preview_data: xr.DataArray) -> xr.DataArray:
        preview_data = preview_data.transpose("ky", "kx")
        return erlab.analysis.transform.symmetrize_nfold(
            preview_data,
            self.preview_symmetry_fold_spin.value(),
            axes=("ky", "kx"),
            center={"ky": 0.0, "kx": 0.0},
            reshape=False,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )

    def get_data(self) -> tuple[xr.DataArray, xr.DataArray]:
        # Set angle offsets
        data_ang = self._assign_params(self._angle_data())
        self._validate_kinetic_energy(data_ang, context="updating ktool preview")
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
        k_preview = k.T
        preview_symmetry_available = self._preview_supports_symmetry(k_preview)
        self._set_preview_symmetry_available(preview_symmetry_available)
        if preview_symmetry_available and self.preview_symmetry_group.isChecked():
            k_preview = self._symmetrized_preview(k_preview)
        self.images[0].setDataArray(ang.T)
        self.images[1].setDataArray(k_preview)
        self._notify_data_changed()
        if self.bz_group.isChecked():
            self.update_bz()

    @staticmethod
    def _bz_digest_array(values: npt.ArrayLike) -> bytes:
        """Return a compact digest for cached BZ geometry inputs."""
        arr = np.ascontiguousarray(np.asarray(values, dtype=float))
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
        return digest.digest()

    def _bz_cache_token(self, converted_slice: xr.DataArray) -> tuple[typing.Any, ...]:
        """Build a cache key for the currently displayed BZ overlay geometry."""
        lattice_token = tuple(np.round(self._avec.ravel(), 8))
        base_token: tuple[typing.Any, ...] = (
            lattice_token,
            self.centering_combo.currentText(),
            round(self.rot_spin.value(), 6),
            tuple(converted_slice.dims),
        )

        if self.data.kspace._has_hv:
            kp_dim = next(d for d in converted_slice.dims if d != "kz")
            other = "kx" if kp_dim == "ky" else "ky"
            return (
                *base_token,
                kp_dim,
                self._bz_digest_array(converted_slice[kp_dim].values),
                self._bz_digest_array(converted_slice["kz"].values),
                self._bz_digest_array(converted_slice[other].values),
            )

        return (
            *base_token,
            round(self.kz_spin.value(), 6),
            self._bz_digest_array(converted_slice["kx"].values),
            self._bz_digest_array(converted_slice["ky"].values),
        )

    def _exact_hv_bz_surface(
        self, converted_slice: xr.DataArray
    ) -> tuple[
        str,
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        """Build the sampled 3D momentum surface used for exact ``hv`` BZ overlays."""
        kp_dim = next(d for d in converted_slice.dims if d != "kz")
        plot_dims = ("kz", kp_dim)
        kx_vals = np.asarray(
            converted_slice["kx"]
            .broadcast_like(converted_slice)
            .reset_coords(drop=True)
            .transpose(*plot_dims),
            dtype=float,
        )
        ky_vals = np.asarray(
            converted_slice["ky"]
            .broadcast_like(converted_slice)
            .reset_coords(drop=True)
            .transpose(*plot_dims),
            dtype=float,
        )
        kz_vals = np.asarray(
            converted_slice["kz"]
            .broadcast_like(converted_slice)
            .reset_coords(drop=True)
            .transpose(*plot_dims),
            dtype=float,
        )
        surface = np.stack([kx_vals, ky_vals, kz_vals], axis=-1)
        return (
            typing.cast("str", kp_dim),
            np.asarray(converted_slice[kp_dim].values, dtype=float),
            np.asarray(converted_slice["kz"].values, dtype=float),
            surface,
        )

    def get_bz_lines(
        self,
    ) -> tuple[
        list[npt.NDArray[np.floating]],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]:
        avec_primitive = erlab.lattice.to_primitive(
            self._avec, centering_type=self.centering_combo.currentText()
        )
        bvec = erlab.lattice.to_reciprocal(avec_primitive)
        converted_slice: xr.DataArray | None = self.images[1].data_array
        if converted_slice is None:
            return [], np.zeros((0, 2)), np.zeros((0, 2))

        cache_token = self._bz_cache_token(converted_slice)
        if cache_token == self._bz_cache_key and self._bz_cache_value is not None:
            return self._bz_cache_value

        rot: float = self.rot_spin.value()

        if self.data.kspace._has_hv:
            kp_dim = next(d for d in converted_slice.dims if d != "kz")
            other = "kx" if kp_dim == "ky" else "ky"
            other_values = np.asarray(converted_slice[other].values, dtype=float)
            if other_values.ndim == 0 or other_values.size == 1:
                kp_vals = np.asarray(converted_slice[kp_dim].values, dtype=float)
                kz_vals = np.asarray(converted_slice["kz"].values, dtype=float)
                legacy_lines, vertices, midpoints = erlab.lattice.get_out_of_plane_bz(
                    bvec,
                    k_parallel=float(other_values.reshape(-1)[0]),
                    angle=rot,
                    bounds=(
                        float(np.nanmin(kp_vals)),
                        float(np.nanmax(kp_vals)),
                        float(np.nanmin(kz_vals)),
                        float(np.nanmax(kz_vals)),
                    ),
                    return_midpoints=True,
                )
                lines = [np.asarray(line, dtype=float) for line in legacy_lines]
            else:
                theta = np.deg2rad(rot)
                bvec = bvec @ np.array(
                    [
                        [np.cos(theta), np.sin(theta), 0.0],
                        [-np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                _, plot_x, plot_y, surface = self._exact_hv_bz_surface(converted_slice)
                lines, vertices, midpoints = typing.cast(
                    "tuple[list[npt.NDArray[np.floating]],"
                    " npt.NDArray[np.floating], npt.NDArray[np.floating]]",
                    erlab.lattice.get_surface_bz(
                        bvec,
                        plot_x,
                        plot_y,
                        surface,
                        return_midpoints=True,
                    ),
                )
        else:
            kx_vals = converted_slice["kx"].values
            ky_vals = converted_slice["ky"].values
            legacy_lines, vertices, midpoints = erlab.lattice.get_in_plane_bz(
                bvec,
                kz=self.kz_spin.value() * np.pi / self.c_spin.value(),
                angle=rot,
                bounds=(kx_vals.min(), kx_vals.max(), ky_vals.min(), ky_vals.max()),
                return_midpoints=True,
            )
            lines = [np.asarray(line, dtype=float) for line in legacy_lines]

        self._bz_cache_key = cache_token
        self._bz_cache_value = (lines, vertices, midpoints)
        return self._bz_cache_value


def ktool(
    data: xr.DataArray,
    avec: npt.NDArray | None = None,
    rotate_bz: float | None = None,
    centering: typing.Literal["P", "A", "B", "C", "F", "I", "R"] | None = None,
    *,
    cmap: str | None = None,
    gamma: float | None = None,
    data_name: str | None = None,
    initial_normal_emission: tuple[float, float] | None = None,
    initial_delta: float | None = None,
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
    initial_normal_emission
        Optional pair of ``(alpha, beta)`` values used once during initialization to
        seed the normal emission controls and derived angle offsets.
    initial_delta
        Optional delta value to apply alongside ``initial_normal_emission``.

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
            initial_normal_emission=initial_normal_emission,
            initial_delta=initial_delta,
        )
        win.show()
        win.raise_()
        win.activateWindow()

    return win
