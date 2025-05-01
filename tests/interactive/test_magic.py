import numpy as np
import xarray as xr

import erlab


def test_itool_magic(qtbot, ipapp) -> None:
    erlab.interactive.imagetool.manager.main(execute=False)
    manager = erlab.interactive.imagetool.manager._manager_instance

    qtbot.addWidget(manager)
    qtbot.wait_until(erlab.interactive.imagetool.manager.is_running)

    ipapp.run_line_magic("load_ext", line="erlab.interactive")

    ipapp.user_global_ns["example_data"] = xr.DataArray(
        np.arange(25).reshape((5, 5)),
        dims=["alpha", "eV"],
        coords={"alpha": np.arange(5), "eV": np.arange(5)},
    )
    ipapp.run_line_magic("itool", line="-m example_data --cmap viridis")

    qtbot.wait_until(lambda: manager.ntools == 1, timeout=5000)

    assert manager.get_tool(0).array_slicer.point_value(0) == 12.0

    manager.remove_all_tools()
    qtbot.wait_until(lambda: manager.ntools == 0)
    manager.close()
