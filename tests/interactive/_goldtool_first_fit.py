"""Run a first GoldTool fit without imports or JIT caches from the test runner."""

import sys

import xarray as xr
from qtpy import QtWidgets

from erlab.interactive.fermiedge import EdgeFitTask, GoldTool

app = QtWidgets.QApplication([])
win = GoldTool(xr.load_dataarray(sys.argv[1], engine="h5netcdf"), data_name="gold")
win.params_edge.widgets["# CPU"].setValue(1)
win.params_edge.widgets["T (K)"].setValue(100.0)
win.params_roi.modify_roi(x0=-15.0, x1=15.0, y0=-0.2, y1=0.2)
task = EdgeFitTask(
    win.data, win._along_dim, *win.roi_limits_ordered, win.params_edge.values
)
results: list[tuple[xr.DataArray, xr.DataArray]] = []
failures: list[str] = []
task.signals.sigFinished.connect(
    lambda center, stderr: results.append((center, stderr))
)
task.signals.sigFailed.connect(failures.append)

try:
    # Compiling on the GUI thread first would hide the worker-stack regression.
    assert "erlab.analysis.fit.models" not in sys.modules
    win._threadpool.start(task)
    assert win._threadpool.waitForDone(120_000), "GoldTool fit timed out"
    app.processEvents()
    assert not failures, failures
    assert len(results) == 1
    edge_center, _edge_stderr = results[0]
    edge_center.to_netcdf(sys.argv[2], engine="h5netcdf")
finally:
    win.close()
