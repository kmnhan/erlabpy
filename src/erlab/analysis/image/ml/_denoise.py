"""Denoise angle-energy slices of ARPES data with the bundled ONNX model."""

__all__ = ["denoise"]
import copy
import functools
import os
import pathlib
import threading
import typing
import warnings

import numpy as np
import scipy
import xarray as xr

if typing.TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import numpy.typing as npt
    import onnxruntime as ort

DEFAULT_MODEL_NAME = "arpes-denoiser.onnx"

# The model expects input images that sum to this value.
NORMALIZED_TOTAL = 10_000

# Coordinate attributes recording the physical pixel-edge extent of a resized
# axis. A single-pixel axis can only be resized if it carries both.
EDGE_START_ATTR = "resampling_pixel_edge_start"
EDGE_STOP_ATTR = "resampling_pixel_edge_stop"

# Prevents two threads from building the same cached ONNX session at once.
_SESSION_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_bool(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_int(value: object) -> bool:
    """Check for a Python or NumPy integer, excluding booleans."""
    return isinstance(value, (int, np.integer)) and not _is_bool(value)


def _parse_shape(resize: typing.Any) -> tuple[int, int]:
    """Turn ``resize`` (an int or a pair of ints) into a ``(rows, cols)`` tuple."""
    if _is_int(resize):
        resize = (int(resize), int(resize))
    try:
        shape = tuple(resize)
    except TypeError as exc:
        raise ValueError(
            "resize must be a positive integer or two positive integers."
        ) from exc
    if len(shape) != 2 or any(not _is_int(n) or n < 1 for n in shape):
        raise ValueError("resize must contain two positive integers.")
    return int(shape[0]), int(shape[1])


# ---------------------------------------------------------------------------
# Area rebinning
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _rebin_matrix(n_in: int, n_out: int) -> "scipy.sparse.csr_matrix":
    """Sparse ``(n_out, n_in)`` matrix that rebins a 1D signal by area overlap.

    Both grids span the same extent. On a common axis of ``n_in * n_out``
    units, input pixel ``i`` covers ``[i * n_out, (i + 1) * n_out)`` and output
    pixel ``j`` covers ``[j * n_in, (j + 1) * n_in)``. Entry ``(j, i)`` is the
    fraction of input pixel ``i`` inside output pixel ``j``, so every column
    sums to 1 and the total intensity is preserved.
    """
    values, indices, indptr = [], [], [0]
    for j in range(n_out):
        out_start, out_stop = j * n_in, (j + 1) * n_in
        # Input pixels that can overlap output pixel j (stop is exclusive).
        first = out_start // n_out
        stop = min((out_stop + n_out - 1) // n_out, n_in)
        for i in range(first, stop):
            overlap = min(out_stop, (i + 1) * n_out) - max(out_start, i * n_out)
            if overlap > 0:
                indices.append(i)
                values.append(overlap / n_out)
        indptr.append(len(values))
    return scipy.sparse.csr_matrix(
        (values, indices, indptr), shape=(n_out, n_in), dtype=np.float64
    )


def _area_rebin(
    image: "npt.ArrayLike", shape: tuple[int, ...]
) -> "npt.NDArray[np.float64]":
    """Rebin a 2D image to ``shape``, preserving its total (summed) intensity."""
    result = np.asarray(image, dtype=np.float64)
    for axis in (0, 1):
        if result.shape[axis] == shape[axis]:
            continue
        matrix = _rebin_matrix(result.shape[axis], shape[axis])
        result = matrix @ result if axis == 0 else (matrix @ result.T).T
    return result


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def _check_uniform_spacing(values: "npt.NDArray[np.number]", dim: "Hashable") -> None:
    """Raise unless the 1D numeric ``values`` are evenly spaced (and not constant)."""
    # Allow a few ULPs of rounding error from how the axis was generated/stored.
    eps = (
        np.finfo(typing.cast("np.dtype[np.floating]", values.dtype)).eps
        if values.dtype.kind == "f"
        else np.finfo(float).eps
    )
    values = values.astype(np.float64)
    step = (values[-1] - values[0]) / (values.size - 1)
    tolerance = max(abs(step) * 1e-7, eps * max(abs(step), np.max(np.abs(values))) * 4)
    expected = np.linspace(values[0], values[-1], values.size)
    if step == 0 or not np.allclose(values, expected, rtol=0, atol=tolerance):
        raise ValueError(f"{dim}: only uniformly spaced coordinates are supported.")


def _pixel_edges(
    values: "npt.NDArray[np.number]",
    attrs: "Mapping[Hashable, typing.Any]",
    dim: "Hashable",
) -> tuple[float | np.number, float | np.number]:
    """Return the outer pixel edges ``(start, stop)`` of an axis.

    For two or more pixel centers the edges lie half a step outside the first
    and last centers. A single center has no step, so the edges must be stored
    in the coordinate attributes.
    """
    if values.size == 1:
        start, stop = attrs.get(EDGE_START_ATTR), attrs.get(EDGE_STOP_ATTR)
        if (
            start is None
            or stop is None
            or not np.isfinite([start, stop]).all()
            or start == stop
        ):
            raise ValueError(
                f"{dim}: a singleton axis needs pixel-edge attributes to resize."
            )
        return start, stop
    values = values.astype(np.float64)
    step = (values[-1] - values[0]) / (values.size - 1)
    return values[0] - step / 2, values[-1] + step / 2


def _dimension_coordinate(
    coord: xr.DataArray, dim: "Hashable", size: int, resized: bool
) -> xr.DataArray:
    """Validate a dimension coordinate and, if its axis is resized, resample it.

    The resampled coordinate has ``size`` pixel centers spread over the same
    pixel-edge extent, in the same direction, and records that extent in its
    attributes.
    """
    if coord.dims != (dim,):
        raise ValueError(f"{dim}: expected one-dimensional pixel centers.")
    values = np.asarray(coord.values)

    if values.dtype.kind not in "iuf":
        # Non-numeric labels can be kept but not interpolated.
        if resized:
            raise ValueError(f"{dim}: resizing requires numeric pixel centers.")
        return coord.copy(deep=True)

    if not np.isfinite(values).all():
        raise ValueError(f"{dim}: coordinates must be finite.")
    if values.size > 1:
        _check_uniform_spacing(values, dim)
    if not resized:
        return coord.copy(deep=True)

    start, stop = _pixel_edges(values, coord.attrs, dim)
    centers = start + (np.arange(size) + 0.5) * (stop - start) / size
    attrs = copy.deepcopy(coord.attrs)
    attrs.update({EDGE_START_ATTR: float(start), EDGE_STOP_ATTR: float(stop)})
    return xr.DataArray(centers, dims=(dim,), attrs=attrs)


def _output_coordinates(
    data: xr.DataArray, sizes: "Mapping[str, int]", restore_size: bool
) -> dict["Hashable", xr.DataArray]:
    """Build the coordinates of the denoised array.

    The selected dimension coordinates are always validated, even when
    ``restore_size`` is true and the original coordinates are returned unchanged.
    Otherwise they are resampled to ``sizes``, and auxiliary coordinates that
    depend on a resized axis are dropped with a warning.
    """
    resized_dims = {dim for dim, size in sizes.items() if data.sizes[dim] != size}

    coords: dict[Hashable, xr.DataArray] = {}
    for dim, size in sizes.items():
        if dim in data.coords:
            coords[dim] = _dimension_coordinate(
                data.coords[dim], dim, size, dim in resized_dims
            )

    if restore_size:
        return {name: coord.copy(deep=True) for name, coord in data.coords.items()}

    dropped = []
    for name, coord in data.coords.items():
        if name in sizes:
            continue
        if resized_dims.intersection(coord.dims):
            dropped.append(str(name))
        else:
            coords[name] = coord.copy(deep=True)
    if dropped:
        # stacklevel=3 points at the caller of denoise().
        warnings.warn(
            "Dropped auxiliary coordinates on resized axes: " + ", ".join(dropped),
            stacklevel=3,
        )
    return coords


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _cached_session(
    path: str, mtime_ns: int, size: int, num_threads: int
) -> "ort.InferenceSession":
    """Create a CPU ONNX Runtime session.

    ``mtime_ns`` and ``size`` are unused here; they are part of the cache key
    so that a modified model file is reloaded.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "The `onnxruntime` package is required for denoising. "
            'Install it with `pip install "erlab[ml]"`.'
        ) from e

    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        path, sess_options=options, providers=["CPUExecutionProvider"]
    )


def _load_session(
    model_path: str | os.PathLike[str] | None, num_threads: int
) -> tuple[pathlib.Path, "ort.InferenceSession"]:
    """Resolve the model file and return ``(path, session)``."""
    if model_path is None:
        path = pathlib.Path(__file__).with_name("_data") / DEFAULT_MODEL_NAME
    else:
        path = pathlib.Path(
            model_path
        ).expanduser()  # relative paths use the working directory
    path = path.resolve(strict=True)
    stat = path.stat()
    with _SESSION_LOCK:
        session = _cached_session(
            str(path), stat.st_mtime_ns, stat.st_size, int(num_threads)
        )
    return path, session


def _check_model_io(
    session: "ort.InferenceSession", shape: tuple[int, ...], tta: bool
) -> tuple[str, str]:
    """Check the model's input/output signature and return their names.

    The model must take one float32 NCHW tensor and return one tensor. Any
    fixed input dimensions must match ``(1, 1, rows, cols)``, and also the
    transposed image when TTA is on.
    """
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or inputs[0].type != "tensor(float)"
        or len(inputs[0].shape) != 4
    ):
        raise ValueError("Expected one float32 NCHW input and one image output.")

    model_shape = inputs[0].shape  # dynamic dimensions are strings or None
    image_shapes = [shape, shape[::-1]] if tta else [shape]
    for rows, cols in image_shapes:
        required = (1, 1, rows, cols)
        if any(
            isinstance(n, int) and n > 0 and n != r
            for n, r in zip(model_shape, required, strict=True)
        ):
            raise ValueError(f"Model input {model_shape} does not accept {required}.")
    return inputs[0].name, outputs[0].name


def _tta_transforms(tta: bool) -> list[tuple[bool, tuple[int, ...]]]:
    """List the ``(transpose, flip_axes)`` augmentations to run."""
    if not tta:
        return [(False, ())]
    return [
        (transpose, flip_axes)
        for transpose in (False, True)
        for flip_axes in ((), (0,), (1,), (0, 1))
    ]


def _predict(
    session: "ort.InferenceSession", image: "npt.NDArray[np.floating]", tta: bool
) -> "npt.NDArray[np.float64]":
    """Run the model on ``image`` and return the TTA-averaged prediction.

    The prediction is on the model's native output grid, which may be coarser
    than ``image`` (e.g. a stride-2 network).
    """
    input_name, output_name = _check_model_io(session, image.shape, tta)

    aligned = []
    for transpose, flip_axes in _tta_transforms(tta):
        augmented = np.flip(image.T if transpose else image, axis=flip_axes)
        tensor = np.ascontiguousarray(augmented[None, None], dtype=np.float32)
        prediction = np.asarray(session.run([output_name], {input_name: tensor})[0])
        if (
            prediction.ndim != 4
            or prediction.shape[:2] != (1, 1)
            or min(prediction.shape[2:]) < 1
        ):
            raise ValueError(f"Expected output [1, 1, h, w], got {prediction.shape}.")
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Model returned nonfinite values.")

        # Undo the augmentation so every prediction lines up with the input.
        prediction = np.flip(prediction[0, 0].astype(np.float64), axis=flip_axes)
        aligned.append(prediction.T if transpose else prediction)
    return np.stack(aligned).mean(axis=0)


def _denoise_2d(
    image: "npt.NDArray[np.floating]",
    *,
    model_path: pathlib.Path | None,
    num_threads: int,
    shape: tuple[int, int],
    tta: bool,
    restore_scale: bool,
    restore_size: bool,
) -> "npt.NDArray[np.float64]":
    """Denoise one slice in energy-first order, with its own normalization."""
    image = np.asarray(image, dtype=np.float64)
    if not np.isfinite(image).all():
        raise ValueError("data must be finite.")
    if np.any(image < 0):
        warnings.warn(
            "Negative intensities are retained; model suitability needs evaluation.",
            stacklevel=2,
        )
    original_shape = image.shape
    image = _area_rebin(image, shape)

    total = float(image.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Each slice must have finite positive total intensity.")
    normalized = np.asarray((image / total) * NORMALIZED_TOTAL, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("Normalized data cannot be represented as float32.")

    # Load in the executing process; ONNX sessions cannot be sent to Dask workers.
    _, session = _load_session(model_path, num_threads)
    native = _predict(session, normalized, tta)
    # Scale the area rebin by the pixel-count ratio to keep the intensity level.
    result = _area_rebin(native, shape) * (np.prod(shape) / native.size)
    if restore_scale:
        result = (result / NORMALIZED_TOTAL) * total
    if restore_size:
        # Keep the predicted total rather than matching it to the input total.
        result = _area_rebin(result, original_shape)
    if not np.isfinite(result).all():
        raise FloatingPointError("Scaled output contains nonfinite values.")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def denoise(
    data: xr.DataArray,
    resize: int | tuple[int, int] = (300, 300),
    *,
    along: str = "alpha",
    model_path: str | os.PathLike[str] | None = None,
    tta: bool = True,
    restore_scale: bool = True,
    restore_size: bool = True,
    num_threads: int = 0,
) -> xr.DataArray:
    """Denoise angle-energy slices of ARPES data.

    The energy dimension is ``"eV"`` and the angular dimension is selected by ``along``.
    Each slice along the remaining dimensions is denoised independently. For example,
    data with dimensions ``("alpha", "eV", "beta")`` is processed one ``alpha``-``eV``
    slice at each ``beta`` value.

    The model receives each slice in ``("eV", along)`` order on the ``resize`` grid. The
    result keeps the input dimension order.

    The denoising procedure is as follows:

    1. Rebin the input onto the processing grid (``resize``) by area overlap, which
       preserves total intensity.
    2. Normalize the image so it sums to ``NORMALIZED_TOTAL``.
    3. Run the model, optionally averaging over the eight flips/transposes of the image
       (test-time augmentation, TTA).
    4. Map the prediction back onto the processing grid, then optionally undo the
       normalization and rebin to the original grid.

    Parameters
    ----------
    data : xarray.DataArray
        Floating-point data containing ``"eV"`` and ``along`` dimensions. Values must be
        finite, and each slice must have a positive total intensity. Numeric coordinates
        on the two selected dimensions must be uniformly spaced; non-numeric ones are
        allowed only on axes that are not resized. Coordinates on other dimensions are
        kept unchanged. Dask-backed arrays stay lazy. Each selected dimension must be
        contained in one chunk; other dimensions can be split across chunks. For
        example, use ``data.chunk({"eV": -1, "alpha": -1, "beta": 1})`` for one slice
        per chunk.
    resize : int or (int, int), default (300, 300)
        Processing grid as ``(energy_pixels, along_pixels)``, independent of the input
        dimension order. An int means a square grid.
    along
        Angular dimension to denoise together with ``"eV"``. Defaults to ``"alpha"``.
    model_path : str or path-like, optional
        Compatible ONNX model file. Defaults to the model bundled with ERLabPy. Relative
        paths are resolved against the working directory. Custom model files must be
        accessible at the resolved path on each Dask worker.
    tta
        Average predictions over the eight flips/transposes of the image. ``False`` runs
        a single forward pass.
    restore_scale
        Undo each slice's normalization. This rescales by that slice's input total but
        does not force its output total to equal it.
    restore_size
        Rebin each slice to its original grid by area sums and reuse the original
        coordinates (including auxiliary ones). The predicted total is kept, so pixel
        values scale with the change in pixel area. This does not recover detail lost at
        ``resize``. If ``False``, only ``"eV"`` and ``along`` are resized, with
        coordinates resampled over the same pixel-edge extent.
    num_threads
        ONNX Runtime intra-op thread count for unchunked data. The default, ``0``, lets
        ONNX Runtime choose the thread count. This parameter is ignored for chunked
        input since chunked data always uses one thread to avoid nested parallelism with
        Dask.

    Returns
    -------
    xarray.DataArray
        The denoised image, named ``"<name>_denoised"`` (or ``"denoised"``), with the
        input attributes plus ``denoising_*`` attributes recording the settings used.
        Chunked input produces a lazy result. Computation and scheduling remain under
        the caller's control.

    Notes
    -----
    This function uses the pretrained denoising model from :cite:t:`kim2021denoise`.

    ONNX sessions are cached within each process by model path, modification time, file
    size and thread count. For chunked input, intensity validation and model loading
    occur when the result is computed.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray.")
    if data.ndim < 2 or data.dtype.kind != "f":
        raise TypeError("data must be a floating-point xarray.DataArray with >=2 dims.")
    if not isinstance(along, str):
        raise TypeError("along must be a dimension name.")
    if along == "eV":
        raise ValueError("along must be different from the energy dimension 'eV'.")
    core_dims = ("eV", along)
    if any(dim not in data.dims for dim in core_dims):
        raise ValueError(f"data must contain the 'eV' and {along!r} dimensions.")
    shape = _parse_shape(resize)
    if not all(_is_bool(flag) for flag in (tta, restore_scale, restore_size)):
        raise TypeError("tta, restore_scale, and restore_size must be booleans.")
    if not _is_int(num_threads) or num_threads < 0:
        raise ValueError("num_threads must be an integer >= 0.")

    if min(data.shape) == 0:
        raise ValueError("data must be nonempty.")

    sizes = dict(zip(core_dims, shape, strict=True))
    coords = _output_coordinates(data, sizes, restore_size=bool(restore_size))
    model_file = (
        None if model_path is None else pathlib.Path(model_path).expanduser().resolve()
    )
    resized_dims = {
        dim
        for dim, size in sizes.items()
        if not restore_size and data.sizes[dim] != size
    }
    result = xr.apply_ufunc(
        _denoise_2d,
        xr.Variable(data.dims, data.data),
        input_core_dims=[core_dims],
        output_core_dims=[core_dims],
        exclude_dims=resized_dims,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={
            "output_sizes": {
                dim: data.sizes[dim] if restore_size else size
                for dim, size in sizes.items()
            }
        },
        kwargs={
            "model_path": model_file,
            "num_threads": 1 if data.chunks is not None else int(num_threads),
            "shape": shape,
            "tta": tta,
            "restore_scale": restore_scale,
            "restore_size": restore_size,
        },
    )

    attrs = copy.deepcopy(data.attrs)
    attrs.update(
        denoising_model=DEFAULT_MODEL_NAME if model_file is None else model_file.name,
        denoising_resize=shape,
        denoising_along=along,
        denoising_tta=int(tta),
        denoising_restore_scale=int(restore_scale),
        denoising_restore_size=int(restore_size),
    )
    return xr.DataArray(
        result,
        coords=coords,
        attrs=attrs,
        name="denoised" if data.name is None else f"{data.name}_denoised",
    ).transpose(*data.dims)
