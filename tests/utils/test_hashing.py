import numpy as np
import xarray as xr

from erlab.utils.hashing import fingerprint_dataarray


def test_fingerprint_dataarray() -> None:
    rng = np.random.default_rng(0)

    # Base DataArray
    data = rng.normal(size=(20, 30))
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 10, 30)
    base = xr.DataArray(data, dims=("x", "y"), coords={"x": x, "y": y}, name="A")

    # Identical copies have same fingerprint
    fp0 = fingerprint_dataarray(base)
    assert fingerprint_dataarray(base.copy(deep=True)) == fp0

    # Changing data changes fingerprint
    d1 = base.copy(deep=True)
    d1.data[0, 0] += 1.0
    assert fingerprint_dataarray(d1) != fp0

    # Changing dtype changes fingerprint (via meta signature and bytes)
    d2 = base.astype(np.float32)
    assert fingerprint_dataarray(d2) != fp0

    # Changing coordinate values changes fingerprint
    d3 = base.assign_coords(x=base.x + 0.1)
    assert fingerprint_dataarray(d3) != fp0

    # Renaming a coordinate changes fingerprint (meta includes coord keys)
    d4 = base.rename({"x": "xnew"})
    assert fingerprint_dataarray(d4) != fp0

    # Changing name changes fingerprint
    d5 = base.copy()
    d5.name = "B"
    assert fingerprint_dataarray(d5) != fp0

    # Changing attributes changes fingerprint
    d6 = base.copy()
    d6.attrs["note"] = "hello"
    assert fingerprint_dataarray(d6) != fp0

    # Non-contiguous vs contiguous arrays with identical values hash the same
    arr = np.arange(60).reshape(6, 10)
    da_c = xr.DataArray(
        arr, dims=("a", "b"), coords={"a": np.arange(6), "b": np.arange(10)}
    )
    da_f = xr.DataArray(
        np.asfortranarray(arr),
        dims=("a", "b"),
        coords={"a": np.arange(6), "b": np.arange(10)},
    )
    assert fingerprint_dataarray(da_c) == fingerprint_dataarray(da_f)

    # Object dtype arrays: only the head up to min(128, sample_max) is hashed
    obj = np.array([f"s{i}" for i in range(200)], dtype=object).reshape(20, 10)
    obj_da = xr.DataArray(obj, dims=("p", "q"), name="obj")
    fp_obj_head16 = fingerprint_dataarray(obj_da, sample_max=16)
    # Change outside head: fingerprint should remain same
    obj_da_tail = obj_da.copy(deep=True)
    obj_da_tail.data[19, 9] = "changed"
    assert fingerprint_dataarray(obj_da_tail, sample_max=16) == fp_obj_head16
    # Change inside head: fingerprint should change
    obj_da_head = obj_da.copy(deep=True)
    obj_da_head.data[0, 0] = "changedHead"
    assert fingerprint_dataarray(obj_da_head, sample_max=16) != fp_obj_head16

    # String dtype arrays (uses object hashing path): change should affect fingerprint
    str_arr = np.array([["a", "b"], ["c", "d"]], dtype="U1")
    da_str = xr.DataArray(str_arr, dims=("m", "n"))
    fp_s1 = fingerprint_dataarray(da_str)
    da_str2 = da_str.copy(deep=True)
    da_str2.data[0, 0] = "z"
    assert fingerprint_dataarray(da_str2) != fp_s1

    # Dask-backed array: fingerprint should start with "dask:" and reflect data changes
    try:
        import dask.array as da
    except Exception:
        da = None
    if da is not None:
        d = rng.normal(size=(10, 10))
        dask_data = da.from_array(d, chunks=(5, 5))
        d_da = xr.DataArray(dask_data, dims=("x", "y"))
        dfp1 = fingerprint_dataarray(d_da)
        assert dfp1.startswith("dask:")
        # Same data/chunks -> same token
        d_da2 = xr.DataArray(da.from_array(d, chunks=(5, 5)), dims=("x", "y"))
        assert fingerprint_dataarray(d_da2) == dfp1
        # Different data -> different token
        d2 = d.copy()
        d2[0, 0] += 1
        d_da3 = xr.DataArray(da.from_array(d2, chunks=(5, 5)), dims=("x", "y"))
        assert fingerprint_dataarray(d_da3) != dfp1


def test_fingerprint_dataarray_large() -> None:
    rng = np.random.default_rng(0)
    xvals = np.linspace(0, 1, 1000)
    yvals = np.linspace(0, 10, 2000)
    darr = xr.DataArray(
        rng.normal(size=(1000, 2000)).astype(np.float64),
        dims=("x", "y"),
        coords={"x": xvals, "y": yvals},
    )

    # Fingerprint for large array
    fp = fingerprint_dataarray(darr)

    # Fingerprint for different values
    fp_new = fingerprint_dataarray(darr.copy(data=darr.values * 2))

    assert fp != fp_new


def test_datetime_timedelta_hashing() -> None:
    times = np.array(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"], dtype="datetime64"
    )
    deltas = np.array([1, 2, 3, 4], dtype="timedelta64[D]")

    darr_time = xr.DataArray(
        times,
        dims=("t",),
        coords={"t": np.arange(len(times))},
        name="times",
    )
    darr_delta = xr.DataArray(
        deltas,
        dims=("t",),
        coords={"t": np.arange(len(deltas))},
        name="deltas",
    )

    fp_time = fingerprint_dataarray(darr_time)
    fp_delta = fingerprint_dataarray(darr_delta)

    # Changing a value should change fingerprint
    darr_time2 = darr_time.copy(deep=True)
    darr_time2.data[0] += np.timedelta64(1, "D")
    assert fingerprint_dataarray(darr_time2) != fp_time

    darr_delta2 = darr_delta.copy(deep=True)
    darr_delta2.data[0] += np.timedelta64(1, "D")
    assert fingerprint_dataarray(darr_delta2) != fp_delta
