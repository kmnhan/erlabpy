"""Explicit-export settings for Figure Composer."""

from __future__ import annotations

from qtpy import QtCore, QtWidgets

from erlab.interactive._figurecomposer._model._state import FigureExportState
from erlab.interactive._options.parameters import SavefigNumberWidget


class FigureExportPanel(QtWidgets.QWidget):
    """Edit per-figure savefig overrides without requesting a redraw."""

    state_requested = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("figureComposerExportPage")
        self._syncing = False
        self._build_ui()
        self.set_export(FigureExportState())

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        description = QtWidgets.QLabel(self)
        description.setWordWrap(True)
        description.setText(
            "These settings apply only when you export this figure. "
            "Inherited values come from the workspace or user settings."
        )
        layout.addWidget(description)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.dpi_control = SavefigNumberWidget(
            special_values=(
                ("Use defaults", "inherit"),
                ("Use figure DPI", "figure"),
            ),
            value="inherit",
            minimum=1.0,
            maximum=10000.0,
            decimals=1,
            step=10.0,
            custom_value=100.0,
            parent=self,
        )
        self.dpi_control.setObjectName("figureComposerExportDpiControl")
        self.dpi_control.setToolTip(
            "Resolution passed to savefig for explicit exports."
        )
        form.addRow("DPI", self.dpi_control)

        self.transparent_combo = QtWidgets.QComboBox(self)
        self.transparent_combo.setObjectName("figureComposerExportTransparentCombo")
        self.transparent_combo.addItem("Use defaults", "inherit")
        self.transparent_combo.addItem("Enabled", True)
        self.transparent_combo.addItem("Disabled", False)
        self.transparent_combo.setToolTip(
            "Background transparency passed to savefig for explicit exports."
        )
        form.addRow("Transparent background", self.transparent_combo)

        self.bbox_combo = QtWidgets.QComboBox(self)
        self.bbox_combo.setObjectName("figureComposerExportBboxCombo")
        self.bbox_combo.addItem("Use defaults", "inherit")
        self.bbox_combo.addItem("Standard", "standard")
        self.bbox_combo.addItem("Tight", "tight")
        self.bbox_combo.setToolTip(
            "Bounding box passed to savefig. Tight crops to the figure contents."
        )
        form.addRow("Bounding box", self.bbox_combo)

        self.padding_control = SavefigNumberWidget(
            special_values=(
                ("Use defaults", "inherit"),
                ("Use layout", "layout"),
            ),
            value="inherit",
            minimum=0.0,
            maximum=100.0,
            decimals=3,
            step=0.05,
            custom_value=0.1,
            suffix="in",
            parent=self,
        )
        self.padding_control.setObjectName("figureComposerExportPaddingControl")
        self.padding_control.setToolTip(
            "Padding passed to savefig when the bounding box is tight."
        )
        form.addRow("Padding", self.padding_control)
        layout.addLayout(form)

        action_layout = QtWidgets.QHBoxLayout()
        self.use_defaults_button = QtWidgets.QPushButton("Use Defaults", self)
        self.use_defaults_button.setObjectName("figureComposerExportUseDefaultsButton")
        self.use_defaults_button.setToolTip("Remove all per-figure export overrides.")
        action_layout.addWidget(self.use_defaults_button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)
        layout.addStretch(1)

        self.dpi_control.sigValueChanged.connect(self._control_changed)
        self.transparent_combo.currentIndexChanged.connect(self._control_changed)
        self.bbox_combo.currentIndexChanged.connect(self._control_changed)
        self.padding_control.sigValueChanged.connect(self._control_changed)
        self.use_defaults_button.clicked.connect(self._use_defaults)

    def set_export(self, export: FigureExportState) -> None:
        """Project one immutable export snapshot into the controls."""
        self._syncing = True
        try:
            self.dpi_control.set_value(export.dpi)
            self._set_combo_data(self.transparent_combo, export.transparent)
            self._set_combo_data(self.bbox_combo, export.bbox_inches)
            self.padding_control.set_value(export.pad_inches)
        finally:
            self._syncing = False

    @staticmethod
    def _set_combo_data(combo: QtWidgets.QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index < 0:
            raise ValueError(f"Unsupported export control value {value!r}")
        combo.setCurrentIndex(index)

    def export_state(self) -> FigureExportState:
        """Return the validated per-figure export state shown by the controls."""
        return FigureExportState.model_validate(
            {
                "dpi": self.dpi_control.get_value(),
                "transparent": self.transparent_combo.currentData(),
                "bbox_inches": self.bbox_combo.currentData(),
                "pad_inches": self.padding_control.get_value(),
            }
        )

    @QtCore.Slot()
    @QtCore.Slot(int)
    @QtCore.Slot(object)
    def _control_changed(self, _value: object = None) -> None:
        if not self._syncing:
            self.state_requested.emit(self.export_state())

    @QtCore.Slot()
    def _use_defaults(self) -> None:
        export = FigureExportState()
        self.set_export(export)
        self.state_requested.emit(export)
