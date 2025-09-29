"""Utilities for hashing xarray DataArrays."""

__all__ = ["fingerprint_dataarray"]

import pickle

import numpy as np
import numpy.typing as npt
import xarray as xr


def _digest_bytes(b: memoryview | bytes | bytearray) -> int:
    try:
        import xxhash
    except ImportError:
        import zlib

        return zlib.adler32(b)
    else:
        h = xxhash.xxh64()
        h.update(b)
        return h.intdigest() & 0xFFFFFFFFFFFFFFFF


def _meta_signature(darr: xr.DataArray) -> int:
    return (
        hash(
            (
                darr.name,
                darr.shape,
                tuple(darr.dims),
                tuple(darr.coords.keys()),
                str(darr.dtype),
            )
        )
        & 0xFFFFFFFFFFFFFFFF
    )


def _attrs_signature(darr: xr.DataArray) -> int:
    return _digest_bytes(pickle.dumps(darr.attrs, protocol=5))


def _sample_ndarray(arr: npt.NDArray, *, sample_max: int, blocks: int) -> npt.NDArray:
    size = arr.size
    r = arr.ravel()
    if size <= sample_max:
        sample: npt.NDArray = r
    else:
        blocks = max(1, blocks)
        blk_len = sample_max // blocks
        if blk_len == 0:
            blk_len = 1
            blocks = min(sample_max, size)
        step = (size - blk_len) / (blocks - 1) if blocks > 1 else 0
        parts = []
        for i in range(blocks):
            start = round(i * step)
            end = start + blk_len
            parts.append(r[start:end])
        sample = np.concatenate(parts)
    if not sample.flags.c_contiguous:
        sample = np.ascontiguousarray(sample)
    return sample


def _hash_numeric_array(arr: npt.NDArray, *, sample_max: int, blocks: int) -> int:
    sample = _sample_ndarray(arr, sample_max=sample_max, blocks=blocks)
    if sample.dtype.kind in ("M", "m"):
        # datetime, timedelta
        return _digest_bytes(memoryview(sample.tobytes()).cast("B"))
    return _digest_bytes(memoryview(sample).cast("B"))


def _hash_object_array(arr: npt.NDArray, *, sample_max: int) -> int:
    head = arr.ravel()[: min(arr.size, min(128, sample_max))]
    try:
        b = repr(head.tolist()).encode()
    except Exception:
        b = repr(head).encode()
    return _digest_bytes(memoryview(b))


def _hash_array(arr: npt.NDArray, *, sample_max: int, blocks: int) -> int:
    if arr.dtype.kind in ("O", "U", "S", "V"):
        # object, unicode, bytes, void
        return _hash_object_array(arr, sample_max=sample_max)
    return _hash_numeric_array(arr, sample_max=sample_max, blocks=blocks)


# Coordinate hashing (values, not just keys)
def _coords_signature(
    darr: xr.DataArray,
    *,
    coord_sample_max: int = 1024,
    blocks: int = 2,
) -> int:
    pieces: list[int] = []
    for cname, coord in darr.coords.items():
        cnp = coord.values
        pieces.append(
            hash(
                (
                    (cname, cnp.shape, str(cnp.dtype)),
                    _hash_array(cnp, sample_max=coord_sample_max, blocks=blocks),
                )
            )
            & 0xFFFFFFFFFFFFFFFF
        )
    # Combine pieces deterministically
    # (xor fold then hash to reduce order sensitivity)
    accum = 0
    for p in pieces:
        accum ^= p
        accum &= 0xFFFFFFFFFFFFFFFF
    return hash((accum, len(pieces))) & 0xFFFFFFFFFFFFFFFF


def fingerprint_dataarray(
    darr: xr.DataArray,
    *,
    sample_max: int = 4096,
    blocks: int = 3,
    coord_sample_max: int = 1024,
) -> str:
    """Fast, approximate hash including data, coordinates, and attributes.

    This function computes a hash string for an xarray DataArray that incorporates its
    data, coordinates, and attributes. The hash is designed to change if any of these
    components change (most of the time). It uses sampling for large arrays to balance
    speed and accuracy.

    Parameters
    ----------
    darr : DataArray
        The :class:`xarray.DataArray` to calculate the fingerprint for.
    sample_max : int, optional
        Maximum number of data elements to sample for hashing. If the total number of
        elements in the DataArray exceeds this value, a subset of the data will be
        sampled. Default is 4096.
    blocks : int, optional
        Number of blocks to sample from the data when sampling is needed. More blocks
        increase the chance of detecting changes in the data but also increase
        computation time. Default is 3, which provides a good balance for most cases.
    coord_sample_max : int, optional
        Maximum number of elements to sample from each coordinate array for hashing.
        Default is 1024.

    Returns
    -------
    str
        A string representing the fingerprint of the DataArray.

    Note
    ----
    - Different python processes will produce different fingerprints for the same data
      due to the use of the built-in :func:`hash`. Use only for comparisons within a
      single process.
    - This function is not cryptographically secure and should not be used for security
      purposes.

    """
    if darr.chunks is not None:
        # For dask arrays, use dask's built-in tokenization
        import dask.base

        return f"dask:{dask.base.tokenize(darr)}"

    # Cast to numpy
    if isinstance(darr.data, np.ndarray):
        arr: npt.NDArray = darr.data
    else:
        arr = darr.values

    size: int = arr.size
    meta_sig: int = _meta_signature(darr)
    coord_sig: int = _coords_signature(darr, coord_sample_max=coord_sample_max)
    attr_sig: int = _attrs_signature(darr)

    # Data hash
    data_hash = _hash_array(arr, sample_max=sample_max, blocks=blocks)

    return f"np:{size}:{data_hash:x}:{meta_sig:x}:{coord_sig:x}:{attr_sig:x}"
