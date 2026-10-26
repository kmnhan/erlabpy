"""Data loader for MB Scientific analyzers.

Provides a base class for implementing loaders that can load ``.txt`` and ``.krx``
files acquired with MB Scientific analyzers. Subclass to implement the actual loading.
"""

__all__ = ["MBSLoader", "load_krax", "load_text"]

import itertools
import os
import pathlib
import re
import struct
import typing

import numpy as np
import numpy.typing as npt
import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase


def _parse_attr(val: str) -> float | bool | str:
    try:
        out: float | bool | str = float(val)
    except ValueError:
        if val == "Yes":
            out = True
        elif val == "No":
            out = False
        else:
            out = val

    if isinstance(out, float) and out.is_integer():
        out = int(out)

    return out


def _parse_header(header_str: str) -> dict[str, str]:
    header: dict[str, str] = {}
    keys: set[str] = set()
    for line in header_str.partition("DATA:")[0].splitlines():
        match = re.fullmatch(r"([^\t:]*)([\t:])(.*)", line)
        if match is None:
            continue
        name, separator, value = match.groups()
        name = name.strip()
        if separator == ":":
            value = value.lstrip(" \t")
            if name.casefold() == "timestamp":
                name = "TIMESTAMP:"
        if (not name and not value.strip()) or name.casefold() in keys:
            continue
        # Igor's NumberByKey and StringByKey use the first case-insensitive match.
        keys.add(name.casefold())
        header[name] = value
    return header


def _infer_axis(
    header: dict[str, typing.Any],
    size: int,
    *keys: tuple[str, str | None, str],
) -> npt.NDArray[np.float64] | None:
    header = {key.casefold(): value for key, value in header.items()}
    for start_key, step_key, end_key in keys:
        values = []
        for key in (start_key, step_key, end_key):
            try:
                values.append(
                    float(header.get(key.casefold(), np.nan)) if key else np.nan
                )
            except (TypeError, ValueError):
                values.append(np.nan)
        start, step, end = values
        if not np.isfinite(start):
            continue
        if size == 1:
            return np.array([start], dtype=np.float64)
        if np.isfinite(step) and step != 0:
            return start + step * np.arange(size, dtype=np.float64)
        if np.isfinite(end) and end != start:
            return np.linspace(start, end, size)
    return None


def _default_axis(size: int, name: str) -> npt.NDArray[np.float64]:
    erlab.utils.misc.emit_user_level_warning(
        f"Cannot infer the MBS {name} scale; using zero-based channel indices."
    )
    return np.arange(size, dtype=np.float64)


def _energy_axis(header: dict[str, typing.Any], size: int) -> npt.NDArray[np.float64]:
    values = _infer_axis(header, size, ("Start K.E.", "Step Size", "End K.E."))
    if values is None:
        values = _default_axis(size, "energy")
    return values


def _angle_axis(
    header: dict[str, typing.Any], size: int, *, spin: bool = False
) -> npt.NDArray[np.float64]:
    keys = [("ScaleMin", "ScaleMult", "ScaleMax")]
    if spin:
        keys.append(("S0ScaleMin", "S0ScaleMult", "S0ScaleMax"))
    keys.append(("XScaleMin", "XScaleMult", "XScaleMax"))
    values = _infer_axis(header, size, *keys)
    if values is None:
        values = _infer_axis(
            header, size, ("SY", None, "EndY"), ("Start Line", None, "End Line")
        )
    if values is None:
        values = _default_axis(size, "angle")
    return values


def _deflector_axis(
    header: dict[str, typing.Any], size: int, axis: str = "X"
) -> npt.NDArray[np.float64]:
    keys: list[tuple[str, str | None, str]] = [
        (f"MapStart{axis}", None, f"MapEnd{axis}")
    ]
    if axis == "X":
        keys.append(("YScaleMin", "YScaleMult", "YScaleMax"))
    values = _infer_axis(header, size, *keys)
    if values is None:
        values = _default_axis(size, f"deflector {axis}")
    return values


def load_text(filename: str | os.PathLike) -> xr.DataArray:
    """Load data from MB Scientific analyzers in ``.txt`` format.

    Adapted to Python from the ``MBS_FileLoader_v4.ipf`` Igor procedure with some code
    from `mueslo/mbs <https://github.com/mueslo/mbs>`_.

    Files with one intensity column return a one-dimensional array. Files with two
    columns and no ``NoS`` entry are read as energy and intensity. If energy metadata
    are missing, use the energy column when it is present. Missing scales otherwise
    use zero-based channel indices and issue a warning.
    """
    header_lines = []
    data_start: int | None = None
    # Prefer DATA: so numerical header notes are not mistaken for data.
    with open(filename) as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith("DATA:"):
                data_start = i + 1
                break
        f.seek(0)
        data_lines: typing.Iterable[str] = f
        if data_start is not None:
            header_lines = list(itertools.islice(f, data_start))
        else:
            for line in f:
                if values := line.split():
                    try:
                        for value in values:
                            float(value)
                    except ValueError:
                        pass
                    else:
                        data_lines = itertools.chain((line,), f)
                        break
                header_lines.append(line)
            else:
                raise ValueError("The MBS text file does not contain data.")

        data = np.loadtxt(data_lines, dtype=np.float64, ndmin=2)

    header_dict = {
        key or "Unknown": _parse_attr(value.strip())
        for key, value in _parse_header("".join(header_lines)).items()
    }

    if data.size == 0:
        raise ValueError("The MBS text file does not contain data.")

    energies = _infer_axis(
        header_dict, data.shape[0], ("Start K.E.", "Step Size", "End K.E.")
    )
    has_energy = True
    has_energy_column = False
    n_columns = data.shape[1]
    n_slices = next(
        (value for key, value in header_dict.items() if key.casefold() == "nos"), None
    )
    if n_columns > 1 and n_slices != n_columns:
        has_energy_column = (
            n_slices == n_columns - 1
            or n_columns == 2
            or (energies is not None and np.allclose(energies, data[:, 0]))
        )
        if not has_energy_column and n_slices == data.shape[0]:
            data = data.T  # Constant energy map (defl.)
            has_energy = False

    if has_energy_column:
        if energies is None:
            energies = data[:, 0]
        data = data[:, 1:]

    # MBS_FileLoader_v4.ipf normalizes data with 'ActScans' attr, but we skip that here

    if has_energy:
        if energies is None:
            energies = _energy_axis(header_dict, data.shape[0])
        if data.shape[1] == 1:
            return xr.DataArray(
                data[:, 0].astype(np.int32),
                dims=("energies",),
                coords={"energies": energies},
                attrs=header_dict,
                name=pathlib.Path(filename).stem[:-6],
            )

    coords = {"angles": _angle_axis(header_dict, data.shape[1])}
    if has_energy:
        coords["energies"] = typing.cast("npt.NDArray[np.float64]", energies)
    else:
        coords["defl_angles"] = _deflector_axis(header_dict, data.shape[0])

    return xr.DataArray(
        data.T.astype(np.int32),
        dims=("angles", "energies" if has_energy else "defl_angles"),
        coords=coords,
        attrs=header_dict,
        name=pathlib.Path(filename).stem[:-6],
    )


def load_krax(
    filename: str | os.PathLike, *, without_values: bool = False
) -> xr.DataArray:
    """Load MBS spectra, deflector maps, and spin channels in ``.krx`` format.

    Adapted to Python from the ``load_krax_FS.ipf`` Igor procedure by Felix Baumberger,
    University of Geneva, and ``MBSIkrx.ipf`` by Cheewin Pisanupoj, MB Scientific AB.

    Spin channels have a ``spin`` dimension with zero-based channel indices. Channels
    stored in consecutive row blocks of one image are split along this dimension.
    Deflector scans use ``defl_angles`` for X and ``defl_angles_y`` for Y. X varies
    fastest in a map. Separate spin maps are grouped by channel.
    Missing scales use zero-based channel indices and issue a warning.
    """

    def read_exact(size):
        buffer = file.read(size)
        if len(buffer) != size:
            raise ValueError("Unexpected end of the MBS file.")
        return buffer

    def read_header(pos, end):
        file.seek(pos)
        size = struct.unpack("<I", read_exact(4))[0]
        if pos + 4 + size > end:
            raise ValueError("MBS header length exceeds its image block.")
        header = {}
        for key, value in _parse_header(read_exact(size).decode("utf-8")).items():
            try:
                number = float(value)
            except ValueError:
                header[key] = value
            else:
                header[key] = int(number) if number.is_integer() else number
        return header

    with open(filename, "rb") as file:
        file_size = os.fstat(file.fileno()).st_size
        file.seek(4)
        is_64bit = struct.unpack("<I", read_exact(4))[0] == 0

        if is_64bit:
            fmt = "<Q"
            num_bytes = 8
        else:
            fmt = "<I"
            num_bytes = 4

        file.seek(0)
        n_entries = struct.unpack(fmt, read_exact(num_bytes))[0]
        if n_entries == 0 or n_entries % 3 or (n_entries + 1) * num_bytes > file_size:
            raise ValueError("Invalid MBS image index size.")
        n_images = n_entries // 3
        index = np.frombuffer(
            read_exact(n_entries * num_bytes), dtype="<u8" if is_64bit else "<u4"
        ).reshape(n_images, 3)
        image_size_y, image_size_x = (int(v) for v in index[0, 1:])
        if (
            image_size_x == 0
            or image_size_y == 0
            or np.any(index[:, 1:] != index[0, 1:])
        ):
            raise ValueError("MBS images must have the same nonzero dimensions.")

        positions = [int(v) * 4 for v in index[:, 0]]
        ends = [*positions[1:], file_size]
        image_nbytes = image_size_x * image_size_y * 4
        index_end = file.tell()
        if positions[0] < index_end or any(
            pos + image_nbytes + 4 > end
            for pos, end in zip(positions, ends, strict=True)
        ):
            raise ValueError("MBS image data exceeds its image block.")

        map_size: tuple[int, ...] = ()
        if index_end < positions[0]:
            if positions[0] - index_end < 2 * num_bytes:
                raise ValueError("Incomplete MBS map dimensions.")
            ndim = struct.unpack(fmt, read_exact(num_bytes))[0]
            # Igor skips the second size word and reads image sizes from the index.
            file.seek(num_bytes, os.SEEK_CUR)
            if file.tell() + ndim * num_bytes > positions[0]:
                raise ValueError("MBS map metadata overlaps the first image.")
            map_size = struct.unpack(
                f"<{ndim}{'Q' if is_64bit else 'I'}", read_exact(ndim * num_bytes)
            )
        header_dict = read_header(positions[0] + image_nbytes, ends[0])
        shape = (n_images, image_size_y, image_size_x)
        data = (
            np.broadcast_to(np.zeros((), dtype=np.int32), shape)
            if without_values
            else np.empty(shape, dtype=np.int32)
        )
        if not without_values:
            for ii, pos in enumerate(positions):
                file.seek(pos)
                data[ii] = np.frombuffer(read_exact(image_nbytes), dtype="<i4").reshape(
                    image_size_y, image_size_x
                )

    return _reshape_krax(data, header_dict, map_size)


def _reshape_krax(
    data: npt.NDArray[np.int32],
    header: dict[str, typing.Any],
    map_size: tuple[int, ...],
) -> xr.DataArray:
    n_images, size_y, size_x = data.shape
    map_x, map_y = map_size[2:4] if len(map_size) >= 4 else (n_images, 1)
    n_spin = map_size[4] if len(map_size) == 5 else 0
    dims = ["defl_angles_y", "defl_angles", "angles", "energies"]

    if n_spin and n_images == map_x * map_y * n_spin:
        data = data.reshape(n_spin, map_y, map_x, size_y, size_x)
        dims.insert(0, "spin")
    elif n_spin and n_images == map_x * map_y and size_y % n_spin == 0:
        size_y //= n_spin
        data = data.reshape(map_y, map_x, n_spin, size_y, size_x)
        data = data.transpose(2, 0, 1, 3, 4)
        dims.insert(0, "spin")
    elif n_spin and n_images == n_spin:
        dims = ["spin", "angles", "energies"]
    elif n_spin and n_images == 1 and size_y % n_spin == 0:
        size_y //= n_spin
        data = data.reshape(n_spin, size_y, size_x)
        dims = ["spin", "angles", "energies"]
    elif n_spin:
        raise ValueError("MBS spin channels do not match the map dimensions.")
    else:
        if n_images != map_x * map_y:
            raise ValueError("MBS image count does not match the map dimensions.")
        data = data.reshape(map_y, map_x, size_y, size_x)

    coords: dict[str, npt.NDArray] = {
        "angles": _angle_axis(header, size_y, spin=bool(n_spin)),
        "energies": _energy_axis(header, size_x),
    }
    for dim, size, axis in (
        ("defl_angles", map_x, "X"),
        ("defl_angles_y", map_y, "Y"),
    ):
        if dim in dims and size > 1:
            coords[dim] = _deflector_axis(header, size, axis)
    if n_spin:
        coords["spin"] = np.arange(n_spin)

    out = xr.DataArray(data, dims=dims, coords=coords, attrs=header)
    return out.squeeze(
        dim=[dim for dim in dims if dim != "spin" and out.sizes[dim] == 1]
    )


class MBSLoader(LoaderBase):
    """Load data acquired with MB Scientific analyzers.

    The loader reads angle-resolved or angle-integrated data and constant-energy
    deflector maps from ``.txt`` files. It also reads spectra, deflector maps, and spin
    channels from ``.krx`` files.
    """

    name = "mbs"
    description = "MB Scientific .txt and .krx files"

    extensions: typing.ClassVar[set[str]] = {".txt", ".krx"}

    name_map: typing.ClassVar[dict] = {
        "eV": "energies",
        "alpha": "angles",
        "beta": "defl_angles",
    }
    additional_attrs: typing.ClassVar[dict] = {}
    always_single = True
    skip_validate = True

    @property
    def file_dialog_methods(self):
        return {"MBS Raw Data (*.txt *.krx)": (self.load, {})}

    def load_single(
        self, file_path: str | os.PathLike, without_values: bool = False
    ) -> xr.DataArray | xr.DataTree:
        file_path = pathlib.Path(file_path)

        match file_path.suffix:
            case ".txt":
                return load_text(file_path)
            case ".krx":
                return load_krax(file_path, without_values=without_values)
            case _:
                raise ValueError(f"Unsupported file extension {file_path.suffix}")

    def identify(
        self, num: int, data_dir: str | os.PathLike, prefix: str | None = None
    ) -> tuple[list[pathlib.Path], dict[str, npt.NDArray]] | None:
        prefix_pattern = r".*" if prefix is None else re.escape(prefix)

        all_files: set[pathlib.Path] = erlab.io.utils.get_files(
            data_dir, extensions=".txt"
        )  # Exclude .krx files

        pattern = re.compile(prefix_pattern + str(num).zfill(4) + r"_\d{5}.txt")
        files: list[pathlib.Path] = [f for f in all_files if pattern.match(f.name)]

        if len(files) > 1:
            erlab.utils.misc.emit_user_level_warning(
                f"Multiple files found for scan {num}, using {files[0]}. "
                "Try providing the `prefix` argument to specify."
            )
            files = files[:1]

        return files, {}
