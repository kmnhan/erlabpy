"""Data loader for the Scienta Omicron DA30 analyzer.

Provides a base class for implementing loaders that can load data acquired with Scienta
Omicron's DA30 analyzer using ``SES.exe``. Subclass to implement the actual loading.
"""

from __future__ import annotations

__all__ = [
    "CasePreservingConfigParser",
    "DA30Loader",
    "load_txt",
    "load_zip",
    "parse_ini",
]

import configparser
import itertools
import os
import pathlib
import re
import typing
import zipfile

import numpy as np
import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


class InvalidDA30ZipError(Exception):
    """Raised when the file is not a valid DA30 zip file."""

    def __init__(self, filename: str | os.PathLike):
        super().__init__(f"{filename} does not appear to be a valid DA30 zip file.")


class CasePreservingConfigParser(configparser.ConfigParser):
    """ConfigParser that preserves the case of the keys."""

    def optionxform(self, optionstr):
        return str(optionstr)


class DA30Loader(LoaderBase):
    """Load Scienta Omicron DA30 data acquired through SES.

    This loader reads Igor Binary Wave (``.ibw``), packed Igor (``.pxt``), SES text
    (``.txt``), and DA30 ZIP data. A file with multiple regions returns an
    :class:`xarray.DataTree`.
    """

    name = "da30"
    description = "Scienta Omicron DA30 with SES"

    extensions: typing.ClassVar[set[str]] = {".ibw", ".pxt", ".txt", ".zip", ""}

    name_map: typing.ClassVar[dict] = {
        "eV": ["Kinetic Energy [eV]", "Energy [eV]"],
        "alpha": ["Y-Scale [deg]", "Thetax [deg]"],
        "beta": ["Thetay [deg]"],
        "hv": ["BL Energy", "Excitation Energy"],
    }
    additional_attrs: typing.ClassVar[dict] = {}
    always_single = True
    skip_validate = True

    @property
    def file_dialog_methods(self):
        return {"DA30 Raw Data (*.ibw *.pxt *.txt *.zip)": (self.load, {})}

    def load_single(
        self,
        file_path: str | os.PathLike,
        *,
        without_values: bool = False,
    ) -> xr.DataArray | xr.DataTree:
        file_path = pathlib.Path(file_path)

        match file_path.suffix:
            case ".ibw":
                data: xr.DataArray | xr.DataTree = xr.load_dataarray(
                    file_path, engine="erlab-igor"
                )

            case ".pxt":
                data = xr.open_datatree(file_path, engine="erlab-igor")

                if len(data.children) == 1:
                    # Get DataArray if single region
                    data = next(iter(next(iter(data.children.values())).values()))
                    data.load()  # Ensure repr shows data

            case ".zip" | "":
                data = load_zip(file_path, without_values=without_values)

            case ".txt":
                data = load_txt(
                    file_path, without_values=without_values, name_map=self.name_map
                )

            case _:
                raise ValueError(f"Unsupported file extension {file_path.suffix}")

        return data

    def identify(self, num: int, data_dir: str | os.PathLike):
        matches = []

        pattern = re.compile(r"(.*?)" + str(num).zfill(4))
        pattern_ibw = re.compile(
            r"(.*?)" + str(num).zfill(4) + ".*" + str(num).zfill(3)
        )
        for file in sorted(
            erlab.io.utils.get_files(
                data_dir, extensions=(".ibw", ".pxt", ".txt", ".zip")
            )
        ):
            match file.suffix:
                case ".ibw":
                    m = pattern_ibw.match(file.stem)
                case _:
                    m = pattern.match(file.stem)

            if m is not None:
                matches.append(file)

        return matches, {}

    def post_process(self, data: xr.DataArray) -> xr.DataArray:
        data = super().post_process(data)

        if ("beta" not in data.coords) or not (
            data.attrs.get("Lens Mode", "").startswith("DA")
        ):
            # Zero DA offset if not using DA lens mode
            data = data.assign_coords(beta=0.0)

        return data

    def files_for_summary(self, data_dir: str | os.PathLike):
        return sorted(
            erlab.io.utils.get_files(
                data_dir, extensions=(".pxt", ".ibw", ".txt", ".zip")
            )
        )


def load_txt(
    filename: str | os.PathLike,
    without_values: bool = False,
    *,
    name_map: Mapping[str, str | Iterable[str]] | None = None,
) -> xr.DataArray | xr.DataTree:
    """Load data from an SES ``.txt`` file from a Scienta Omicron analyzer.

    Signals with matching coordinates are attached to the spectrum as coordinates.
    Multiple regions or signals with separate coordinates are returned in a DataTree.

    Parameters
    ----------
    filename : str or os.PathLike
        The path to the SES ``.txt`` file.
    without_values : bool, optional
        If True, the values are not loaded, only the coordinates and attributes.
    name_map : mapping, optional
        Loader name mapping. Signal coordinate names cannot use its keys or aliases.

    Returns
    -------
    xr.DataArray or xr.DataTree
        A DataArray for one spectrum, or a DataTree containing the separate arrays.
    """
    filename = pathlib.Path(filename)
    sections: dict[str, list[str]] = {}
    data_locations: dict[str, tuple[int, int]] = {}
    section = ""
    with filename.open("rb") as stream:
        for lineno, line in enumerate(stream):
            if lineno == 0:
                line = line.removeprefix(b"\xef\xbb\xbf")
            line = line.strip()
            if line.startswith(b"[") and line.endswith(b"]"):
                section = line[1:-1].decode("utf-8")
                if section in sections:
                    raise ValueError(f"{filename}: duplicate section [{section}]")
                sections[section] = []
                if section.startswith(("Data ", "Signal ")):
                    # Store the byte offset and physical line count, not the data text.
                    data_locations[section] = (stream.tell(), 0)
            elif section in data_locations:
                offset, count = data_locations[section]
                data_locations[section] = (offset, count + 1)
            elif line:
                if not section:
                    raise ValueError(
                        f"{filename}: content before the first SES section"
                    )
                sections[section].append(line.decode("utf-8").strip())

    metadata: dict[str, dict[str, typing.Any]] = {}
    for name, lines in sections.items():
        if name.startswith(("Data ", "Signal ")):
            continue
        metadata[name] = {}
        for line in lines:
            key, sep, value = line.partition("=")
            if sep:
                metadata[name][key.strip()] = _parse_value(value.strip())

    try:
        num_regions = int(metadata["Info"]["Number of Regions"])
        with filename.open("rb") as stream:
            out = [
                _load_txt_region(
                    sections,
                    metadata,
                    region,
                    stream=stream,
                    data_locations=data_locations,
                    without_values=without_values,
                )
                for region in range(1, num_regions + 1)
            ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{filename}: invalid SES text file: {exc}") from exc

    if not out:
        raise ValueError(f"{filename}: no regions declared")
    reserved_names: set[str] = set()
    if name_map is not None:
        reserved_names.update(name_map)
        for aliases in name_map.values():
            reserved_names.update((aliases,) if isinstance(aliases, str) else aliases)
    for payloads in out:
        if "Data" not in payloads or "Signal" not in payloads:
            continue
        spectrum, signal = payloads["Data"], payloads["Signal"]
        signal_name = str(signal.attrs.get("Signal names", "Signal"))
        if (
            signal_name
            and signal_name not in spectrum.coords
            and signal_name != spectrum.name
            and signal_name not in reserved_names
            and all(
                dim in spectrum.dims
                and signal.sizes[dim] == spectrum.sizes[dim]
                and np.allclose(signal[dim], spectrum[dim], rtol=1e-12, atol=1e-12)
                for dim in signal.dims
            )
        ):
            payloads["Data"] = spectrum.assign_coords(
                {signal_name: (signal.dims, signal.data)}
            )
            del payloads["Signal"]

    if len(out) == 1 and len(out[0]) == 1:
        return next(iter(out[0].values()))
    # Region names are normally unique. Use numbered nodes if they cannot be paths.
    names = [str(next(iter(payloads.values())).name) for payloads in out]
    use_names = len(set(names)) == len(names) and all(
        name and "/" not in name and name not in {".", ".."} for name in names
    )
    nodes = {}
    for region, (name, payloads) in enumerate(zip(names, out, strict=True), start=1):
        region_name = name if use_names else f"Region {region}"
        for kind, data in payloads.items():
            path = region_name if len(payloads) == 1 else f"{region_name}/{kind}"
            nodes[path] = data.to_dataset(promote_attrs=True)
    return xr.DataTree.from_dict(nodes)


def _load_txt_region(
    sections: dict[str, list[str]],
    metadata: dict[str, dict[str, typing.Any]],
    region: int,
    *,
    stream: typing.BinaryIO,
    data_locations: dict[str, tuple[int, int]],
    without_values: bool,
) -> dict[str, xr.DataArray]:
    header = metadata[f"Region {region}"]
    attrs = dict(metadata["Info"])
    attrs.update(
        (key, value)
        for key, value in header.items()
        if not key.startswith("Dimension ")
    )
    for prefix in (
        "Info",
        "Run Mode Information",
        "User Interface Information",
    ):
        attrs.update(metadata.get(f"{prefix} {region}", {}))
    run_mode = sections.get(f"Run Mode Information {region}", [])
    if any("=" not in line for line in run_mode):
        attrs["Run Mode Information"] = "\n".join(run_mode)

    out = {}
    for kind in ("Data", "Signal"):
        data_sections = [
            name
            for name in sections
            if re.fullmatch(rf"{kind} {region}(?::[^\]]+)?", name)
        ]
        if not data_sections:
            continue
        if kind == "Data":
            axis_keys = []
            for dim in range(1, 4):
                prefix = f"Dimension {dim}"
                if f"{prefix} size" not in header:
                    break
                axis_keys.append(
                    (f"{prefix} name", f"{prefix} size", f"{prefix} scale")
                )
            if not axis_keys or "Dimension 4 size" in header:
                raise ValueError(f"[{region}]: expected one to three dimensions")
            stacked = len(axis_keys) == 3
        else:
            if header.get("Number of signals") != 1:
                raise ValueError(
                    f"[{region}]: multiple external signals are not supported"
                )
            axis_keys = [("Signal unit", "Signal steps", "Signal scale [0]")]
            stacked = any(":" in name for name in data_sections)
            if stacked:
                axis_keys.append(
                    ("Dimension 3 name", "Dimension 3 size", "Dimension 3 scale")
                )

        sizes = []
        dims = []
        coords = {}
        for name_key, size_key, scale_key in axis_keys:
            size = int(header[size_key])
            if size < 1:
                raise ValueError(f"[{region}]: invalid {size_key}: {size}")
            name = str(header.get(name_key, "")) or size_key.rsplit(" ", 1)[0]
            scale = str(header.get(scale_key, ""))
            if scale:
                coord = np.asarray(scale.split(), dtype=np.float64)
            elif scale_key == "Dimension 3 scale":
                coord = np.arange(size, dtype=np.float64)
            else:
                raise ValueError(f"[{region}]: missing {scale_key}")
            if coord.size != size:
                raise ValueError(f"[{region}]: {scale_key} does not match its size")
            if name in coords:
                raise ValueError(f"[{region}]: duplicate dimension name {name!r}")
            dims.append(name)
            sizes.append(size)
            coords[name] = coord

        nframes = sizes[-1] if stacked else 1
        if len(data_sections) != nframes:
            raise ValueError(
                f"[{region}]: expected {nframes} {kind} sections, "
                f"found {len(data_sections)}"
            )

        if without_values:
            values = np.zeros(tuple(reversed(sizes)), dtype=np.float64)
        else:
            values = np.empty(tuple(reversed(sizes)), dtype=np.float64)
            ncolumns = sizes[1] if kind == "Data" and len(sizes) > 1 else 1
            for frame, name in enumerate(data_sections):
                offset, count = data_locations[name]
                stream.seek(offset)
                table = np.loadtxt(
                    itertools.islice(stream, count), ndmin=2, encoding="utf-8"
                )
                if table.shape != (sizes[0], ncolumns + 1):
                    raise ValueError(
                        f"[{name}]: expected a {sizes[0]} by {ncolumns + 1} table, "
                        f"found {table.shape}"
                    )
                if not np.allclose(table[:, 0], coords[dims[0]]):
                    raise ValueError(
                        f"[{name}]: data coordinates do not match the scale"
                    )
                if frame == 0:
                    coords[dims[0]] = table[:, 0].copy()
                target = values[frame] if stacked else values
                target[...] = table[:, 1:].T.reshape(target.shape)
                del table

        out[kind] = xr.DataArray(
            values,
            dims=dims[::-1],
            coords=coords,
            name=str(header["Region Name"]),
            attrs=attrs,
        )

    if not out:
        raise ValueError(f"[{region}]: no Data or Signal sections found")
    return out


def load_zip(
    filename: str | os.PathLike,
    without_values: bool = False,
) -> xr.DataArray | xr.DataTree:
    """Load data from a ``.zip`` file from a Scienta Omicron DA30 analyzer.

    If the file contains a single region, a DataArray is returned. Otherwise, a DataTree
    containing all regions is returned.

    Parameters
    ----------
    filename : str or os.PathLike
        The path to the ``.zip`` file or the directory containing the unzipped files.
    without_values : bool, optional
        If True, the values are not loaded, only the coordinates and attributes.

    Returns
    -------
    xr.DataArray or xr.DataTree
        The loaded data. If the file contains a single region, a DataArray is returned.
        Otherwise, a DataTree containing all regions is returned.

    """
    filename = pathlib.Path(filename)
    zipped: bool = not filename.is_dir()

    if zipped:
        with zipfile.ZipFile(filename, mode="r", allowZip64=False) as zf:
            f_names = zf.namelist()
            regions = _get_zip_regions(f_names)
            if len(regions) == 0:
                raise InvalidDA30ZipError(filename)
            out = _load_zip_content(
                regions,
                read_ini=lambda name: _parse_ini_string(zf.read(name).decode("utf-8")),
                read_bin=lambda name: _read_zip_array(zf, name),
                without_values=without_values,
            )
    else:
        f_names = os.listdir(filename)
        regions = _get_zip_regions(f_names)
        if len(regions) == 0:
            raise InvalidDA30ZipError(filename)
        out = _load_zip_content(
            regions,
            read_ini=lambda name: parse_ini(filename / name),
            read_bin=lambda name: np.fromfile(filename / name, dtype=np.float32),
            without_values=without_values,
        )

    if len(out) == 1:
        return out[0]

    return xr.DataTree.from_dict(
        {str(da.name): da.to_dataset(promote_attrs=True) for da in out}
    )


def _parse_value(value):
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _parse_ini_string(content: str) -> dict:
    parser = CasePreservingConfigParser(strict=False)
    out = {}
    parser.read_string(content)
    for section in parser.sections():
        out[section] = {k: _parse_value(v) for k, v in parser.items(section)}
    return out


def _read_zip_array(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    return np.frombuffer(zf.read(member), dtype=np.float32)


def _get_zip_regions(f_names: list[str]) -> list[str]:
    return [
        fn[9:-4] for fn in f_names if fn.startswith("Spectrum_") and fn.endswith(".bin")
    ]


def _load_zip_content(
    regions: list[str],
    *,
    read_ini: Callable[[str], dict],
    read_bin: Callable[[str], np.ndarray],
    without_values: bool,
) -> list[xr.DataArray]:
    out: list[xr.DataArray] = []

    for region in regions:
        region_info = read_ini(f"Spectrum_{region}.ini")["spectrum"]
        attrs = {}
        for d in read_ini(f"{region}.ini").values():
            attrs.update(d)

        shape = []
        coords = {}
        for d in ("depth", "height", "width"):
            n = int(region_info[d])
            offset = float(region_info[f"{d}offset"])
            delta = float(region_info[f"{d}delta"])
            shape.append(n)
            coords[region_info[f"{d}label"]] = np.linspace(
                offset, offset + (n - 1) * delta, n
            )

        if without_values:
            arr = np.zeros(shape, dtype=np.float32)
        else:
            arr = read_bin(f"Spectrum_{region}.bin").reshape(shape)

        out.append(
            xr.DataArray(arr, coords=coords, name=region_info["name"], attrs=attrs)
        )

    return out


def parse_ini(filename: str | os.PathLike) -> dict:
    """Parse an ``.ini`` file into a dictionary."""
    with open(filename, encoding="utf-8") as f:
        return _parse_ini_string(f.read())
