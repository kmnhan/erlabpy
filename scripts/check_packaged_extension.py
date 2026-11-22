"""Run a recorded scientific extension through the packaged ImageTool worker."""

import sys
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

import erlab
from erlab.extensions import routine, run_routine

KEEP_ATTRS_AT_IMPORT = xr.get_options()["keep_attrs"]


def calibrate_block(counts):
    # Check the interpreter and options again when Dask evaluates each chunk.
    assert getattr(sys, "frozen", False)
    assert xr.get_options()["keep_attrs"] is True
    return 2 * counts + 1


@routine()
def calibrate(data: xr.DataArray) -> xr.DataArray:
    """Apply a linear calibration to detector counts."""
    assert KEEP_ATTRS_AT_IMPORT is True
    bundle = Path(getattr(sys, "_MEIPASS", ""))
    for module in (erlab, xr, np, da):
        assert module.__file__ is not None
        assert Path(module.__file__).is_relative_to(bundle), module.__file__
    data = data.chunk({"energy": 2})
    return data.copy(
        data=data.data.map_blocks(
            calibrate_block, dtype=data.dtype, meta=np.empty((), dtype=data.dtype)
        )
    )


if __name__ == "__main__":
    from types import SimpleNamespace
    from unittest.mock import patch

    from erlab.extensions import _subprocess

    executable = Path(sys.argv[1]).resolve(strict=True)
    counts = xr.DataArray([2, 3, 5, 7], dims="energy", name="spectrum")
    # Only the production command builder targets the packaged executable.
    launcher = SimpleNamespace(executable=str(executable), frozen=True)
    with (
        patch.object(_subprocess, "sys", launcher),
        xr.set_options(keep_attrs=False),
    ):
        result = run_routine(
            counts,
            script=__file__,
            routine_id="calibrate",
            execution_options={"keep_attrs": True},
        )
        assert result.chunks == ((2, 2),)
        xr.testing.assert_identical(result.compute(), counts.copy(data=[5, 7, 11, 15]))
