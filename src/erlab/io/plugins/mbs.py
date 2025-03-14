__all__ = ["MBSLoader", "load_krax", "load_text"]

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


def _parse_header(header_str_list: list[str]) -> dict[str, float | bool | str]:
    header: dict[str, float | bool | str] = {}
    for line in header_str_list:
        name, val = line.split("\t", 1)
        val = val.strip()
        if not name and not val:
            continue

        if name == "":
            name = "Unknown"

        if name in header:
            erlab.utils.misc.emit_user_level_warning(
                f"Duplicate header key '{name}' found, overwriting."
            )

        header[name] = _parse_attr(val)

    return header


def load_text(filename: str | os.PathLike) -> xr.DataArray:
    """Load data from MB Scientific analyzers in ``.txt`` format.

    Adapted to Python from the ``MBS_FileLoader_v4.ipf`` Igor procedure with some code
    from `mueslo/mbs <https://githubhttps://github.com/mueslo/mbs.com/mueslo/mbs>`_.

    """
    header_lines = []
    data_start: int | None = None
    # Read the header (all lines before the "Data" line)
    with open(filename) as f:
        for i, line in enumerate(f):
            if line.startswith("DATA:"):
                data_start = i
                break
            # fix for old files
            if line.startswith("TIMESTAMP:") and not line.startswith("TIMESTAMP:\t"):
                line = line.replace("TIMESTAMP:", "TIMESTAMP:\t", 1)
            header_lines.append(line)
    if data_start is None:
        raise ValueError("Not a valid MBS data file.")

    header_dict = _parse_header(header_lines)

    # Load the numerical data after the header.
    data = np.loadtxt(filename, skiprows=data_start + 1, dtype=np.float64)

    # Match step size and center energy
    e0 = float(header_dict["Start K.E."]) + 0.5 * float(header_dict["Step Size"])
    e1 = float(header_dict["End K.E."]) - 0.5 * float(header_dict["Step Size"])

    if "ScaleMin" in header_dict:
        x0 = float(header_dict["ScaleMin"])
        x1 = float(header_dict["ScaleMax"])
        y0 = float(header_dict.get("MapStartX", np.nan))
        y1 = float(header_dict.get("MapEndX", np.nan))
    else:
        # Old header format
        x0 = float(header_dict["XScaleMin"])
        x1 = float(header_dict["XScaleMax"])
        y0 = float(header_dict.get("YScaleMin", np.nan))
        y1 = float(header_dict.get("YScaleMax", np.nan))

    has_energy: bool = True

    if int(header_dict["NoS"]) != int(data.shape[1]):
        if (
            header_dict["NoS"] == data.shape[1] - 1  # angle-resolved
            or data.shape[1] == 2  # angle-integrated
        ):
            if not np.allclose(
                np.linspace(
                    float(header_dict["Start K.E."]),
                    float(header_dict["End K.E."]) - float(header_dict["Step Size"]),
                    data.shape[0],
                ),
                data[:, 0],
            ):
                raise ValueError("Energy scale mismatch; data file may be corrupt.")

            data = data[:, 1:]

        elif header_dict["NoS"] == data.shape[0]:  # Constant energy map (defl.)
            data = data.T
            has_energy = False

        else:
            raise ValueError(
                f"Number of slices are {header_dict['NoS']} in header, "
                f"but data has {data.shape[1]} columns."
            )

    # MBS_FileLoader_v4.ipf normalizes data with 'ActScans' attr, but we skip that here

    coords = {"angles": np.linspace(x0, x1, data.shape[1])}
    if has_energy:
        coords["energies"] = np.linspace(e0, e1, data.shape[0])
    else:
        coords["defl_angles"] = np.linspace(y0, y1, data.shape[0])

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
    """Load MBS deflector maps in ``.krx`` format.

    Adapted to Python from the ``load_krax_FS.ipf`` Igor procedure by Felix Baumberger,
    University of Geneva.
    """

    def read_header(file, pos, size):
        file.seek(pos)
        return file.read(size).decode("utf-8")

    def header_to_dict(header_str):
        out = {}
        for row in header_str.strip().split("\r\n"):
            if row.startswith("DATA:"):
                break
            k, v = row.split("\t", 1)
            try:
                v = float(v)
            except ValueError:
                pass
            else:
                if v.is_integer():
                    v = int(v)
            out[k] = v
        return out

    with open(filename, "rb") as file:
        file.seek(4)
        v0 = struct.unpack("<I", file.read(4))[0]
        is_64bit = v0 == 0

        if is_64bit:
            fmt = "<Q"
            num_bytes = 8
        else:
            fmt = "<I"
            num_bytes = 4

        file.seek(0)
        v1 = struct.unpack(fmt, file.read(num_bytes))[0]
        n_images = v1 // 3

        # File-position of first image
        file.seek(num_bytes)
        image_pos = struct.unpack(fmt, file.read(num_bytes))[0]

        # Parallel detection angle
        file.seek(2 * num_bytes)
        image_sizeY = struct.unpack(fmt, file.read(num_bytes))[0]

        # Energy coordinate
        file.seek(3 * num_bytes)
        image_sizeX = struct.unpack(fmt, file.read(num_bytes))[0]

        # Autodetect header format and get wave scaling from first header
        header_pos = (image_pos + image_sizeX * image_sizeY + 1) * 4
        header = read_header(file, header_pos, 1200)
        header = header[: header.find("DATA:")]
        header_dict = header_to_dict(header)

        e0 = header_dict["Start K.E."] + 0.5 * header_dict["Step Size"]
        e1 = header_dict["End K.E."] - 0.5 * header_dict["Step Size"]
        if header.startswith("Lines\t"):
            x0 = header_dict["ScaleMin"]
            x1 = header_dict["ScaleMax"]
            y0 = header_dict.get("MapStartX", np.nan)
            y1 = header_dict.get("MapEndX", np.nan)
        else:
            # Old header format
            x0 = header_dict["XScaleMin"]
            x1 = header_dict["XScaleMax"]
            y0 = header_dict.get("YScaleMin", np.nan)
            y1 = header_dict.get("YScaleMax", np.nan)

        data = []
        for ii in range(n_images):
            file.seek((ii * 3 + 1) * num_bytes)
            image_pos = struct.unpack(fmt, file.read(num_bytes))[0]

            file.seek(image_pos * 4)

            if without_values:
                buffer = np.zeros((image_sizeY, image_sizeX), dtype=np.int32)
            else:
                buffer = np.frombuffer(
                    file.read(image_sizeX * image_sizeY * 4), dtype=np.int32
                ).reshape((image_sizeY, image_sizeX))

            header_pos = (image_pos + image_sizeX * image_sizeY + 1) * 4
            file.seek(header_pos)
            header = file.read(1200).decode("utf-8")
            header = header[: header.find("DATA:")]
            header_dict = header_to_dict(header)

            data.append(
                xr.DataArray(
                    buffer,
                    dims=["angles", "energies"],
                    coords={
                        "angles": np.linspace(x0, x1, image_sizeY),
                        "energies": np.linspace(e0, e1, image_sizeX),
                    },
                    attrs=header_dict,
                )
            )

        if n_images == 1:
            return data[0]

        return xr.concat(data, dim="defl_angles").assign_coords(
            defl_angles=np.linspace(y0, y1, n_images)
        )


class MBSLoader(LoaderBase):
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
