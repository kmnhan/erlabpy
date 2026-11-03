"""Data loader for the Fermiologics FeSuMa analyzer.

This is a base class for implementing loaders that can load FeSuMa data. Assign
appropriate radial calibration coefficients to the class attributes in subclasses to
auto-convert Fourier-mode images into angles.
"""

from __future__ import annotations

__all__ = ["FeSuMaLoader"]

import pathlib
import typing
from collections.abc import Mapping

import h5py
import numpy as np
import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase

if typing.TYPE_CHECKING:
    import os


def _decode(value: typing.Any) -> typing.Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode(value.item())
        if value.dtype.kind in "OSU":
            return np.asarray([_decode(v) for v in value.flat]).reshape(value.shape)
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _wave_units(source: h5py.Dataset) -> list[str]:
    units = []
    for value in np.atleast_1d(source.attrs.get("IGORWaveUnits", ())).flat:
        unit = str(_decode(value))
        if unit.startswith("[") and unit.endswith("]"):
            unit = unit[1:-1]
        units.append(unit)
    return units


def _attributes(source: h5py.Dataset) -> dict[str, typing.Any]:
    attrs = {
        key: _decode(value)
        for key, value in source.attrs.items()
        if key not in ("IGORWaveType", "IGORWaveUnits", "IGORWaveScaling")
    }
    units = _wave_units(source)
    if units and units[0]:
        attrs["units"] = units[0]
    return attrs


def _voltage_coordinates(dataset: h5py.Dataset) -> dict[str, typing.Any]:
    values = np.asarray(dataset[()])
    labels = [
        str(_decode(label)) for label in dataset.attrs["IGORWaveDimensionLabels"][1:, 0]
    ]
    name = dataset.name.rsplit("/", 1)[-1]
    return {
        f"{name}_{label}": value.item()
        for label, value in zip(labels, values, strict=True)
    }


def _scales(image: h5py.Dataset) -> tuple[tuple[float, float, str], ...]:
    return tuple(
        (
            float(_decode(image.attrs[f"DimOffset{label}"])),
            float(_decode(image.attrs[f"DimDelta{label}"])),
            str(_decode(image.attrs[f"ScaleUnits{label}"])),
        )
        for label in ("X", "Y")
    )


def _read_images(
    file_path: str,
    dataset_paths: tuple[str, ...],
    image_shape: tuple[int, int],
    dtype: np.dtype,
) -> np.ndarray:
    values = np.empty((len(dataset_paths), *image_shape), dtype=dtype)
    with h5py.File(file_path, "r") as source:
        for index, dataset_path in enumerate(dataset_paths):
            source[dataset_path].read_direct(values[index])
    return values


def _resample_image(
    image: np.ndarray,
    coordinates: np.ndarray,
    valid: np.ndarray,
    *,
    serial: bool = False,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    result = erlab.analysis.interpolate.interpn(
        tuple(np.arange(size, dtype=float) for size in image.shape),
        np.asarray(image, dtype=np.float64),
        tuple(coordinates),
        serial=serial,
    )
    if weights is not None:
        result *= weights
    result[~valid] = np.nan
    return result


class FeSuMaLoader(LoaderBase):
    """Load Fermiologics FeSuMa HDF5 exports.

    Native axes are ``detector_x`` and ``detector_y``. Subclasses can set the
    ``fourier*_coefficients`` attributes to enable automatic conversion of detector
    images into angles. Angular X/Y axes become ``alpha``/``beta`` for configuration 4
    and ``beta``/``alpha`` for configuration 3, set through :attr:`additional_attrs`.

    Calibration coefficients ``(k0, k1, ...)`` must start at zero and increase over the
    mode's angular range. They can be found in ``Fermiologics_ImageWarp.ipf``.

    Conversion uses linear interpolation and Jacobian intensity correction
    ``abs(r'(theta) * r(theta) / theta)``, normalized by its maximum over the mode's
    angular range. Set ``jacobian=False`` for interpolation without intensity
    correction.
    """

    name = "fesuma"
    description = "Fermiologics FeSuMa"
    extensions: typing.ClassVar[set[str]] = {".h5"}
    always_single = True
    skip_validate = True
    name_map: typing.ClassVar[dict] = {"hv": "MonochromatorEnergy"}
    coordinate_attrs = ("hv",)
    additional_attrs: typing.ClassVar[dict] = {"configuration": 4}
    # Assumes X along the horizontal axis in the lab frame for vertical cryostat
    # configurations, modify for subclasses if needed.

    fourier16_coefficients: typing.ClassVar[tuple[float, ...] | None] = None
    """Fourier 16° calibration coefficients for angles from 0 to 8°."""

    fourier28_coefficients: typing.ClassVar[tuple[float, ...] | None] = None
    """Fourier 28° calibration coefficients for angles from 0 to 15°."""

    fourier32_coefficients: typing.ClassVar[tuple[float, ...] | None] = None
    """Fourier 32° calibration coefficients for angles from 0 to 17°."""

    def _angle_dimension_names(self, data: xr.DataArray) -> dict[str, str]:
        if not all(
            data[dim].attrs["units"] == "degree" for dim in ("detector_x", "detector_y")
        ):
            return {}
        configuration = data.attrs["configuration"]
        if configuration == 4:
            return {"detector_x": "alpha", "detector_y": "beta"}
        if configuration == 3:
            return {"detector_x": "beta", "detector_y": "alpha"}
        raise ValueError("FeSuMa angular data require configuration 3 or 4")

    def post_process(self, darr: xr.DataArray) -> xr.DataArray:
        chunks = darr.encoding.pop("_fesuma_chunks", None)
        darr = super().post_process(darr)
        angle_names = self._angle_dimension_names(darr)
        darr = darr.rename(angle_names)
        if chunks is not None:
            chunk_sizes = {
                angle_names.get(key, key): value for key, value in chunks.items()
            }
            if len(chunk_sizes) != len(chunks):
                raise ValueError("Duplicate FeSuMa chunk dimensions")
            darr = darr.chunk(chunk_sizes)
        return darr

    def _convert_angles(
        self, data: xr.DataArray, *, without_values: bool, jacobian: bool
    ) -> xr.DataArray:
        dims = ("detector_x", "detector_y")
        if all(data[dim].attrs["units"] == "degree" for dim in dims):
            return data
        calibrations = {
            "Fourier 16°": (self.fourier16_coefficients, 8.0),
            "Fourier 28°": (self.fourier28_coefficients, 15.0),
            "Fourier 32°": (self.fourier32_coefficients, 17.0),
        }
        if "AnalyzerMode" in data.coords:
            modes = data["AnalyzerMode"].values
            if any(
                coefficients is not None and mode in modes
                for mode, (coefficients, _) in calibrations.items()
            ):
                raise ValueError(
                    "FeSuMa angle conversion requires one lens mode per file"
                )
            return data
        mode = data.attrs["AnalyzerMode"]
        if mode in ("Direct", "Direct Low"):
            return data
        coefficients, max_angle = calibrations[mode]
        if coefficients is None:
            return data
        source = [np.asarray(data[dim].values, dtype=float) for dim in dims]
        coefficients_array = np.asarray(coefficients, dtype=float)
        if coefficients_array[0] != 0:
            raise ValueError("Invalid FeSuMa radial calibration coefficients")
        angles = np.linspace(0, max_angle, 8193)
        radii = np.polynomial.polynomial.polyval(angles, coefficients_array)
        if not np.isfinite(radii).all() or not np.all(np.diff(radii) > 0):
            raise ValueError(
                "FeSuMa radial calibration must increase over its angular range"
            )
        x, y = source
        radius = np.hypot(x[:, None], y[None, :])
        angle = np.interp(radius, radii, angles, left=np.nan, right=np.nan)
        derivative_coefficients = np.polynomial.polynomial.polyder(coefficients_array)
        for _ in range(2):
            residual = (
                np.polynomial.polynomial.polyval(angle, coefficients_array) - radius
            )
            derivative = np.polynomial.polynomial.polyval(
                angle, derivative_coefficients
            )
            angle -= np.divide(
                residual,
                derivative,
                out=np.zeros_like(angle),
                where=np.isfinite(angle) & (radius != 0) & (derivative != 0),
            )
        factor = np.divide(angle, radius, out=np.zeros_like(angle), where=radius != 0)
        angular_source = (factor * x[:, None], factor * y[None, :])
        if not np.isfinite(angular_source[0]).any():
            raise ValueError(
                "No detector pixels lie within the FeSuMa angular calibration"
            )
        target = []
        for values, original in zip(angular_source, source, strict=True):
            axis = np.linspace(np.nanmin(values), np.nanmax(values), original.size)
            target.append(axis[::-1] if original[-1] < original[0] else axis)
        radius = np.hypot(target[0][:, None], target[1][None, :])
        factor = np.polynomial.polynomial.polyval(radius, coefficients_array[1:])
        sampling = (factor * target[0][:, None], factor * target[1][None, :])
        valid = radius <= max_angle
        indices = []
        for values, original in zip(sampling, source, strict=True):
            low, high = min(original[0], original[-1]), max(original[0], original[-1])
            tolerance = 1e-10 * max(abs(low), abs(high), 1)
            valid &= (values >= low - tolerance) & (values <= high + tolerance)
            delta = original[1] - original[0]
            indices.append(
                np.clip((values - original[0]) / delta, 0, original.size - 1)
            )
        coordinates = np.where(valid[None, :, :], np.stack(indices), 0)
        weights = None
        if jacobian:
            jacobian_polynomial = np.polynomial.Polynomial(
                coefficients_array[1:]
            ) * np.polynomial.Polynomial(derivative_coefficients)
            roots = jacobian_polynomial.deriv().roots()
            extrema = np.asarray(roots[np.isreal(roots)].real, dtype=float)
            extrema = extrema[(extrema > 0) & (extrema < max_angle)]
            max_jacobian = np.max(
                np.abs(jacobian_polynomial(np.r_[0, max_angle, extrema]))
            )
            weights = (
                np.abs(
                    factor
                    * np.polynomial.polynomial.polyval(radius, derivative_coefficients)
                )
                / max_jacobian
            )
        if without_values:
            data = data.copy(
                data=np.broadcast_to(np.zeros((), dtype=float), data.shape)
            )
        else:
            data = xr.apply_ufunc(
                _resample_image,
                data,
                input_core_dims=[list(dims)],
                output_core_dims=[list(dims)],
                exclude_dims=set(dims),
                kwargs={
                    "coordinates": coordinates,
                    "valid": valid,
                    "serial": data.chunks is not None,
                    "weights": weights,
                },
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={
                    "output_sizes": dict(
                        zip(dims, (len(axis) for axis in target), strict=True)
                    )
                },
                keep_attrs=True,
            )
        return data.assign_coords(
            {
                dim: xr.DataArray(values, dims=dim, attrs={"units": "degree"})
                for dim, values in zip(dims, target, strict=True)
            }
        )

    @property
    def file_dialog_methods(self):
        return {"FeSuMa Raw Data (*.h5)": (self.load, {})}

    def identify(self, num: int, data_dir: str | os.PathLike):
        path = pathlib.Path(data_dir) / f"Data_{num:05d}.h5"
        return ([path] if path.is_file() else []), {}

    def load_single(
        self,
        file_path: str | os.PathLike,
        *,
        without_values: bool = False,
        convert_angles: bool = True,
        jacobian: bool = True,
        chunks: int | Mapping[str, int] | typing.Literal["auto"] | None = None,
    ) -> xr.DataArray:
        path = str(pathlib.Path(file_path).resolve())
        attrs: dict[str, typing.Any] = {}
        coordinate_units: dict[str, str] = {}
        voltage_coordinates: set[str] = set()
        vector_labels: dict[str, list[str]] = {}
        records: list[dict[str, typing.Any]] = []
        with h5py.File(path, "r") as source:
            for key, dataset in source.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                units = _wave_units(dataset)
                if units and units[0]:
                    coordinate_units[key] = units[0]
                if key == "IntensityConservvationWeights":
                    attrs["intensity_conservation_weights_path"] = dataset.name
                    continue
                value = _decode(dataset[()])
                if key == "CommonData":
                    attrs.update({str(k): _decode(v) for k, v in value})
                else:
                    attrs[key] = value
            groups = sorted(
                (key for key in source if key.isdigit()), key=lambda key: int(key)
            )
            if not groups:
                raise ValueError("The file contains no FeSuMa analysisImage groups")
            first_image = source[groups[0]]["analysisImage"]
            image_shape = typing.cast("tuple[int, int]", first_image.shape)
            dtype, scales = first_image.dtype, _scales(first_image)
            for group_name in groups:
                group = source[group_name]
                image = group["analysisImage"]
                if image.shape != image_shape:
                    raise ValueError(f"FeSuMa image shape changes in {group_name}")
                if _scales(image) != scales:
                    raise ValueError(f"FeSuMa detector scale changes in {group_name}")
                record = _attributes(image)
                for key, dataset in group.items():
                    if key == "analysisImage":
                        continue
                    units = _wave_units(dataset)
                    if key in (
                        "AnalyzerCurrentVoltages",
                        "AnalyzerUserSetVoltages",
                        "AnalyzerUserSetAmplitudes",
                    ):
                        channels = _voltage_coordinates(dataset)
                        record.update(channels)
                        voltage_coordinates.update(channels)
                        coordinate_units.update(dict.fromkeys(channels, units[0]))
                    else:
                        if dataset.ndim == 1 and (
                            dataset.size > 1
                            or "IGORWaveDimensionLabels" in dataset.attrs
                        ):
                            labels = dataset.attrs.get("IGORWaveDimensionLabels")
                            vector_labels[key] = (
                                [str(_decode(label)) for label in labels[1:, 0]]
                                if labels is not None
                                else [str(index) for index in range(dataset.size)]
                            )
                            record[key] = dataset[()]
                        else:
                            record[key] = _decode(dataset[()])
                        if units and units[0]:
                            coordinate_units[key] = units[0]
                records.append(record)
        count = len(records)
        coords: dict[str, typing.Any] = {}
        energy_values = None
        if attrs["wSaveType"] == "Adjustment":
            scan_dims: tuple[str, ...] = ()
            scan_shape: tuple[int, ...] = ()
            array_name = "analysisImage"
        else:
            array_name = attrs["AcquisitionMode"]
            step_values = np.asarray(
                [int(record.pop("AcquisitionCurrentStep")) for record in records]
            )
            sweep_values = np.asarray(
                [int(record.pop("AcquisitionCurrentSweep")) for record in records]
            )
            unique_sweeps = np.unique(sweep_values)
            nsteps = int(np.count_nonzero(sweep_values == unique_sweeps[0]))
            regular = False
            if attrs["AcquisitionCoordinate"] == "EKIN":
                start, delta = (
                    float(attrs["AcquisitionEkin" + key]) for key in ("Start", "Step")
                )
                energy_values = start + delta * step_values
                regular = np.array_equal(
                    sweep_values, np.repeat(unique_sweeps, nsteps)
                ) and np.array_equal(
                    step_values, np.tile(np.arange(nsteps), len(unique_sweeps))
                )
            if regular:
                if len(unique_sweeps) == 1:
                    scan_dims, scan_shape = ("eV",), (nsteps,)
                    coords["sweep"] = int(unique_sweeps[0])
                else:
                    scan_dims, scan_shape = (
                        ("sweep", "eV"),
                        (
                            len(unique_sweeps),
                            nsteps,
                        ),
                    )
                    coords["sweep"] = unique_sweeps
                coords["eV"] = typing.cast("np.ndarray", energy_values)[:nsteps]
            else:
                scan_dims, scan_shape = ("frame",), (count,)
                coords.update(frame=np.arange(count))
                coords["sweep"] = ("frame", sweep_values)
                if energy_values is not None:
                    coords["eV"] = ("frame", energy_values)
        for index, (offset, delta, unit) in enumerate(scales):
            name = ("detector_x", "detector_y")[index]
            coords[name] = xr.DataArray(
                offset + delta * np.arange(image_shape[index]),
                dims=name,
                attrs={"units": unit},
            )
        for key in sorted(set().union(*(record.keys() for record in records))):
            metadata_values: list[typing.Any] = [record.get(key) for record in records]
            if key in voltage_coordinates:
                voltage_values = np.asarray(
                    [record.get(key, np.nan) for record in records]
                )
                if np.all(voltage_values == voltage_values[0]):
                    coords[key] = voltage_values[0]
                else:
                    coords[key] = (scan_dims, voltage_values.reshape(scan_shape))
                continue
            present_values = [value for value in metadata_values if value is not None]
            first_value = present_values[0]
            has_missing = len(present_values) != count
            if not has_missing and all(
                np.array_equal(metadata_values[0], value, equal_nan=False)
                for value in metadata_values
            ):
                attrs[key] = first_value
            elif np.ndim(first_value) == 0:
                missing_strings = isinstance(first_value, str) and has_missing
                values = np.asarray(
                    [np.nan if value is None else value for value in metadata_values],
                    dtype=object if missing_strings else None,
                )
                coords[key] = xr.Variable(scan_dims, values.reshape(scan_shape))
                if missing_strings:
                    coords[key].encoding["_FillValue"] = ""
            else:
                values = np.stack(
                    [
                        np.full(np.shape(first_value), np.nan)
                        if value is None
                        else value
                        for value in metadata_values
                    ]
                )
                for label, component in zip(vector_labels[key], values.T, strict=True):
                    name = f"{key}_{label}"
                    coords[name] = (scan_dims, component.reshape(scan_shape))
                    if key in coordinate_units:
                        coordinate_units[name] = coordinate_units[key]
        for key, value in attrs.items():
            if isinstance(value, np.ndarray) and value.dtype.kind in "OSU":
                attrs[key] = value.tolist()
        dataset_paths = tuple(f"/{group}/analysisImage" for group in groups)
        dims = (*scan_dims, "detector_x", "detector_y")
        shape = (*scan_shape, *image_shape)
        if without_values:
            values = np.broadcast_to(np.zeros((), dtype=dtype), shape)
        elif chunks is None:
            values = _read_images(path, dataset_paths, image_shape, dtype).reshape(
                shape
            )
        else:
            import dask.array as da
            from dask import delayed

            block_size: typing.Any = 1
            if isinstance(chunks, Mapping):
                if scan_dims:
                    block_size = chunks.get(scan_dims[-1], 1)
            elif isinstance(chunks, int):
                block_size = chunks
            elif chunks != "auto":
                raise ValueError(
                    "FeSuMa chunks must be an integer, mapping, 'auto', or None"
                )
            if block_size == -1:
                block_size = count
            if (
                not isinstance(block_size, int)
                or isinstance(block_size, bool)
                or block_size < 1
            ):
                raise ValueError(
                    "FeSuMa source chunk size must be a positive integer or -1"
                )
            blocks = []
            for start in range(0, count, block_size):
                block_paths = dataset_paths[start : start + block_size]
                block = delayed(_read_images)(path, block_paths, image_shape, dtype)
                blocks.append(
                    da.from_delayed(
                        block, shape=(len(block_paths), *image_shape), dtype=dtype
                    )
                )
            values = da.concatenate(blocks, axis=0).reshape(shape)
        result = xr.DataArray(
            values,
            dims=dims,
            coords=coords,
            attrs=attrs,
            name=array_name,
        )
        key_mapping = self._name_map_reversed
        for name, unit in coordinate_units.items():
            if (
                key_mapping.get(name, name) in self.coordinate_attrs
                and name in result.attrs
            ):
                result = result.assign_coords({name: result.attrs.pop(name)})
            if name in result.coords:
                result[name].attrs["units"] = unit
        if energy_values is not None:
            result["eV"].attrs["units"] = "eV"
        if convert_angles:
            result = self._convert_angles(
                result, without_values=without_values, jacobian=jacobian
            )
        if isinstance(chunks, Mapping) and not without_values:
            result.encoding["_fesuma_chunks"] = dict(chunks)
        return result
