import pyqtgraph.parametertree
from qtpy import QtCore, QtWidgets

import erlab
from erlab.interactive._options.defaults import (
    DEFAULT_OPTIONS,
    make_parameter,
    parameter_to_dict,
)


class _CustomParameterTree(pyqtgraph.parametertree.ParameterTree):
    """Custom ParameterTree that keeps a handle to the parameter it is displaying."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parameter: pyqtgraph.parametertree.Parameter | None = None

    def setParameters(self, *args, **kwargs):
        self._parameter = args[0] if args else None
        super().setParameters(*args, **kwargs)

    @property
    def parameter(self) -> pyqtgraph.parametertree.Parameter | None:
        return self._parameter


class OptionDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        self.tree = _CustomParameterTree()
        self.setup_ui()
        self.update()

    def setup_ui(self):
        """Set up the UI for the settings dialog."""
        buttons = QtWidgets.QDialogButtonBox()
        self.btn_ok = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Ok,
        )
        self.btn_cancel = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        )
        self.btn_apply = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Apply,
        )
        self.btn_restore = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults,
        )

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self.apply)
        self.btn_restore.clicked.connect(self.restore)

        self.btn_apply.setEnabled(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tree)
        layout.addWidget(buttons)
        self.setLayout(layout)

    @property
    def current_options(self) -> dict:
        """Get the currently displayed settings from the parameter tree."""
        if self.tree.parameter is None:
            return {}
        return parameter_to_dict(self.tree.parameter)

    @property
    def modified(self) -> bool:
        """Check if the current settings differ from the saved settings."""
        return self.current_options != erlab.interactive.options.option_dict

    @property
    def is_default(self) -> bool:
        """Check if the current settings are the default settings."""
        return self.current_options == DEFAULT_OPTIONS

    @QtCore.Slot()
    def apply(self):
        erlab.interactive.options.option_dict = dict(self.current_options)
        self.update()

    def _set_parameters(self, d: dict) -> None:
        """Update the parameter tree with the given options dictionary."""
        if self.tree.parameter is not None:
            self.tree.parameter.sigTreeStateChanged.disconnect(self._tree_changed)

        parameter: pyqtgraph.parametertree.Parameter = make_parameter(d)
        parameter.sigTreeStateChanged.connect(self._tree_changed)
        self.tree.setParameters(parameter, showTop=False)
        self._tree_changed()

    @QtCore.Slot()
    def update(self):
        """Update the parameter tree with current settings."""
        self._set_parameters(erlab.interactive.options.option_dict)

    @QtCore.Slot()
    def _tree_changed(self):
        """Update button states when the tree changes."""
        self.btn_restore.setDisabled(self.is_default)
        if self.modified:
            self.setWindowTitle("Settings (Unsaved Changes)")
            self.btn_apply.setEnabled(True)
        else:
            self.setWindowTitle("Settings")
            self.btn_apply.setEnabled(False)

    @QtCore.Slot()
    def restore(self):
        """Populate the parameter tree with default settings."""
        self._set_parameters(DEFAULT_OPTIONS)

    def accept(self):
        """Apply changes and close the dialog."""
        self.apply()
        super().accept()

    def reject(self):
        """Ask the user if they want to save changes before closing."""
        if self.modified:
            match QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before closing?",
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            ):
                case QtWidgets.QMessageBox.StandardButton.Cancel:
                    return
                case QtWidgets.QMessageBox.StandardButton.Ok:
                    self.apply()
        super().reject()
