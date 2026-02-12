import dask.distributed
import pytest
from qtpy import QtWidgets

from erlab.interactive._dask import ClientSetupDialog, DaskMenu, _IntOrNoneWidget


@pytest.fixture
def main_window(qtbot):
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    win.show()
    return win


@pytest.fixture
def cleanup_dask_client():
    try:
        yield
    finally:
        try:
            client = dask.distributed.default_client()
        except ValueError:
            pass
        else:
            client.close()


def test_actions_visibility(main_window, monkeypatch):
    menu = DaskMenu(main_window)

    # No default client
    menu.update_actions_visibility()
    assert menu.connect_existing_action.isEnabled()
    assert menu.create_local_action.isEnabled()
    assert not menu.close_client_action.isEnabled()
    assert not menu.open_dashboard_action.isEnabled()
    assert not menu.about_client_action.isEnabled()

    # Simulate a default client without launching a real distributed cluster.
    monkeypatch.setattr(dask.distributed, "default_client", lambda: object())
    menu.update_actions_visibility()
    assert not menu.connect_existing_action.isEnabled()
    assert not menu.create_local_action.isEnabled()
    assert menu.close_client_action.isEnabled()
    assert menu.open_dashboard_action.isEnabled()
    assert menu.about_client_action.isEnabled()


def test_create_local_cluster(main_window, accept_dialog, cleanup_dask_client):
    menu = DaskMenu(main_window)

    def _configure_local_cluster(dialog):
        if hasattr(dialog, "_processes_checkbox"):
            dialog._processes_checkbox.setChecked(False)
            if dialog._n_workers_widget._auto_check.isChecked():
                dialog._n_workers_widget._auto_check.setChecked(False)
            dialog._n_workers_widget._spin.setValue(1)
            if dialog._threads_per_worker_widget._auto_check.isChecked():
                dialog._threads_per_worker_widget._auto_check.setChecked(False)
            dialog._threads_per_worker_widget._spin.setValue(1)

    accept_dialog(
        menu.create_local_cluster, chained_dialogs=2, pre_call=_configure_local_cluster
    )
    assert dask.distributed.default_client() == menu.default_client


def test_connect_existing_cluster(main_window, accept_dialog, cleanup_dask_client):
    menu = DaskMenu(main_window)

    with dask.distributed.LocalCluster(
        processes=False, n_workers=1, threads_per_worker=1
    ) as cluster:

        def populate_dialog(dialog: ClientSetupDialog):
            dialog._scheduler_address_edit.setText(cluster.scheduler_address)

        accept_dialog(menu.connect_existing_cluster, pre_call=populate_dialog)
        assert menu.default_client.scheduler.address == cluster.scheduler_address

        # Close the client for next part of the test
        menu.close_client()

    def populate_dialog_fail(dialog: ClientSetupDialog):
        dialog._scheduler_address_edit.setText("nonexistent:8786")
        dialog._timeout_spin.setValue(1)

    accept_dialog(
        menu.connect_existing_cluster, chained_dialogs=2, pre_call=populate_dialog_fail
    )
    assert menu.default_client is None


def test_int_or_none_widget(qapp):
    w = _IntOrNoneWidget()
    # Default is Auto -> None
    assert w.value() is None

    # Uncheck auto to enable spin
    w._auto_check.setChecked(False)
    w._spin.setValue(7)
    assert w.value() == 7

    # Re-enable auto
    w._auto_check.setChecked(True)
    assert w.value() is None


def test_client_setup_dialog_getters(qapp):
    dlg = ClientSetupDialog()
    dlg._scheduler_address_edit.setText("  tcp://host:8786  ")
    dlg._timeout_spin.setValue(12)
    assert dlg.scheduler_address() == "tcp://host:8786"
    assert dlg.timeout() == 12
