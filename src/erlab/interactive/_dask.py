"""Dask cluster and client management with Qt menus and dialogs."""

import typing
import weakref
import webbrowser

from qtpy import QtCore, QtWidgets

import erlab

if typing.TYPE_CHECKING:
    import dask.distributed


class _IntOrNoneWidget(QtWidgets.QWidget):
    """A widget that allows selecting an integer or None."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._auto_check = QtWidgets.QCheckBox("Auto", self)
        self._auto_check.setChecked(True)
        self._spin = QtWidgets.QSpinBox(self)
        self._spin.setRange(1, 1024)
        self._spin.setDisabled(True)
        self._auto_check.toggled.connect(self.toggle_visibility)

        layout.addWidget(self._auto_check)
        layout.addWidget(self._spin)
        layout.addStretch()

    @QtCore.Slot(bool)
    def toggle_visibility(self, checked: bool) -> None:
        self._spin.setDisabled(checked)

    def value(self) -> int | None:
        if self._auto_check.isChecked():
            return None
        return self._spin.value()


class LocalClusterSetupDialog(QtWidgets.QDialog):
    """Dialog for configuring a local Dask cluster."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Setup Local Dask Cluster")

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        groupbox = QtWidgets.QGroupBox(self)
        form_layout = QtWidgets.QFormLayout(groupbox)
        groupbox.setLayout(form_layout)
        layout.addWidget(groupbox)

        self._n_workers_widget = _IntOrNoneWidget()
        self._threads_per_worker_widget = _IntOrNoneWidget()
        self._processes_checkbox = QtWidgets.QCheckBox("Use Processes")
        self._processes_checkbox.setChecked(True)

        form_layout.addRow("Number of Workers:", self._n_workers_widget)
        form_layout.addRow("Threads per Worker:", self._threads_per_worker_widget)
        form_layout.addRow(self._processes_checkbox)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def n_workers(self) -> int | None:
        return self._n_workers_widget.value()

    def threads_per_worker(self) -> int | None:
        return self._threads_per_worker_widget.value()


class ClientSetupDialog(QtWidgets.QDialog):
    """Dialog for configuring a Dask client connection."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Setup Dask Client Connection")

        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        groupbox = QtWidgets.QGroupBox(self)
        form_layout = QtWidgets.QFormLayout(groupbox)
        groupbox.setLayout(form_layout)
        layout.addWidget(groupbox)

        self._scheduler_address_edit = QtWidgets.QLineEdit(self)
        self._timeout_spin = QtWidgets.QSpinBox(self)
        self._timeout_spin.setRange(1, 99)
        self._timeout_spin.setValue(5)
        self._timeout_spin.setSuffix(" seconds")

        form_layout.addRow("Address:", self._scheduler_address_edit)
        form_layout.addRow("Connection Timeout:", self._timeout_spin)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def scheduler_address(self) -> str:
        return self._scheduler_address_edit.text().strip()

    def timeout(self) -> int:
        return self._timeout_spin.value()


class DaskMenu(QtWidgets.QMenu):
    """Menu for managing Dask clusters and clients.

    Add this menu into an existing menu bar to provide options for creating local Dask
    clusters and connecting to existing clusters.

    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        The parent main window, used for dialog parenting.

    """

    def __init__(self, main_window: QtWidgets.QMainWindow, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._main_window = weakref.ref(main_window)

        self.setup_actions()
        self.aboutToShow.connect(self.update_actions_visibility)

    @property
    def main_window(self) -> QtWidgets.QMainWindow:
        """Get the parent main window."""
        main_window = self._main_window()
        if main_window:
            return main_window
        raise LookupError("Parent was destroyed")

    @property
    def default_client(self) -> "dask.distributed.Client | None":
        """Get the default Dask client, if any."""
        import dask.distributed

        try:
            return dask.distributed.default_client()
        except ValueError:
            return None

    def setup_actions(self) -> None:
        # Initialize actions
        self.create_local_action = QtWidgets.QAction("Create Local Cluster…")
        self.create_local_action.triggered.connect(self.create_local_cluster)

        self.connect_existing_action = QtWidgets.QAction("Connect to Existing Cluster…")
        self.connect_existing_action.triggered.connect(self.connect_existing_cluster)

        self.close_client_action = QtWidgets.QAction("Close Client")
        self.close_client_action.triggered.connect(self.close_client)

        self.about_client_action = QtWidgets.QAction("Client Info")
        self.about_client_action.triggered.connect(self.show_client_info)

        self.open_dashboard_action = QtWidgets.QAction("Open Dashboard")
        self.open_dashboard_action.triggered.connect(self.open_dashboard)

        # Populate menu
        self.addAction(self.create_local_action)
        self.addAction(self.connect_existing_action)

        self.addSeparator()

        self.addAction(self.close_client_action)
        self.addAction(self.about_client_action)
        self.addAction(self.open_dashboard_action)

    @QtCore.Slot()
    def create_local_cluster(self) -> None:
        dialog = LocalClusterSetupDialog(self.main_window)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            import dask.distributed

            try:
                with erlab.interactive.utils.wait_dialog(
                    self.main_window, "Creating Local Dask Cluster…"
                ):
                    dask.distributed.Client(
                        n_workers=dialog.n_workers(),
                        threads_per_worker=dialog.threads_per_worker(),
                        processes=dialog._processes_checkbox.isChecked(),
                        set_as_default=True,
                    )
            except Exception:  # pragma: no cover
                erlab.interactive.utils.MessageDialog.critical(
                    self.main_window,
                    "Cluster Creation Failed",
                    "Failed to create local Dask cluster.",
                )
            else:
                self.show_client_info()

    @QtCore.Slot()
    def connect_existing_cluster(self) -> None:
        dialog = ClientSetupDialog(self.main_window)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            with erlab.interactive.utils.wait_dialog(
                self.main_window, "Connecting to Dask Scheduler…"
            ):
                address, timeout = dialog.scheduler_address(), dialog.timeout()

                import dask.distributed

                try:
                    dask.distributed.Client(
                        address=address, timeout=timeout, set_as_default=True
                    )
                except Exception:
                    erlab.interactive.utils.MessageDialog.critical(
                        self.main_window,
                        "Connection Failed",
                        f"Failed to connect to Dask scheduler at '{address}'",
                    )

    @QtCore.Slot()
    def close_client(self) -> None:
        client = self.default_client
        if client:  # pragma: no branch
            client.close()

    @QtCore.Slot()
    def show_client_info(self) -> None:
        client = self.default_client
        if client:  # pragma: no branch
            info = client.scheduler_info()
            n_threads = sum(w["nthreads"] for w in info["workers"].values())
            client_info = {
                "Address": info["address"],
                "Workers": str(info["n_workers"]),
                "Total threads": str(n_threads),
            }

            memory = [w["memory_limit"] for w in info["workers"].values()]
            if all(memory):
                from dask.utils import format_bytes

                client_info["Total memory"] = format_bytes(sum(memory))

            msg_box = QtWidgets.QMessageBox(self.main_window)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
            msg_box.setText("Dask Client Information")
            msg_box.setInformativeText(
                "\n".join(f"{k}: {v}" for k, v in client_info.items())
            )
            msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
            dash_btn = msg_box.addButton(
                "Open Dashboard", QtWidgets.QMessageBox.ButtonRole.AcceptRole
            )
            msg_box.exec()

            if msg_box.clickedButton() == dash_btn:
                self.open_dashboard()

    @QtCore.Slot()
    def open_dashboard(self) -> None:
        client = self.default_client
        if client:  # pragma: no branch
            webbrowser.open(client.dashboard_link)

    @QtCore.Slot()
    def update_actions_visibility(self) -> None:
        has_client: bool = self.default_client is not None
        self.connect_existing_action.setDisabled(has_client)
        self.create_local_action.setDisabled(has_client)
        self.close_client_action.setEnabled(has_client)
        self.open_dashboard_action.setEnabled(has_client)
        self.about_client_action.setEnabled(has_client)
