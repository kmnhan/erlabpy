"""Data loader for the Scienta Omicron DA30 analyzer.

Provides a base class for implementing loaders that can load data acquired with Scienta
Omicron's DA30 analyzer using ``SES.exe``. Subclass to implement the actual loading.
"""

__all__ = ["CasePreservingConfigParser", "DA30Loader", "load_zip", "parse_ini"]

import configparser
import importlib
import os
import pathlib
import re
import tempfile
import typing
import zipfile

import numpy as np
import xarray as xr

import erlab
from erlab.io.dataloader import LoaderBase


class InvalidDA30ZipError(Exception):
    """Raised when the file is not a valid DA30 zip file."""

    def __init__(self, filename: str | os.PathLike):
        super().__init__(f"{filename} does not appear to be a valid DA30 zip file.")


class CasePreservingConfigParser(configparser.ConfigParser):
    """ConfigParser that preserves the case of the keys."""

    def optionxform(self, optionstr):
        return str(optionstr)


class DA30Loader(LoaderBase):
    name = "da30"
    description = "Scienta Omicron DA30 with SES"

    extensions: typing.ClassVar[set[str]] = {".ibw", ".pxt", ".zip", ""}

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
        return {"DA30 Raw Data (*.ibw *.pxt *.zip)": (self.load, {})}

    def load_single(
        self,
        file_path: str | os.PathLike,
        without_values: bool = False,
        use_libarchive: bool = True,
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
                data = load_zip(
                    file_path,
                    without_values=without_values,
                    use_libarchive=use_libarchive,
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
            erlab.io.utils.get_files(data_dir, extensions=(".ibw", ".pxt", ".zip"))
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
            erlab.io.utils.get_files(data_dir, extensions=(".pxt", ".ibw", ".zip"))
        )


def load_zip(
    filename: str | os.PathLike,
    without_values: bool = False,
    use_libarchive: bool = True,
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
    use_libarchive : bool, optional
        If True, tries to use `libarchive
        <https://github.com/Changaco/python-libarchive-c>`_ to extract the files.
        Install it with ``pip install libarchive-c`` or ``conda install -c conda-forge
        python-libarchive-c``. If False, uses the built-in ``zipfile``. This does not
        have drastic performance improvements when loading a single file, but improves
        multithreaded performance since the underlying C library bypasses the GIL. If
        the C libarchive library is not available, it will fall back to the built-in
        `zipfile` module even if ``libarchive-c`` is installed.

    Returns
    -------
    xr.DataArray or xr.DataTree
        The loaded data. If the file contains a single region, a DataArray is returned.
        Otherwise, a DataTree containing all regions is returned.

    """
    zipped: bool = not os.path.isdir(filename)

    if zipped:
        if use_libarchive:
            use_libarchive = importlib.util.find_spec("libarchive") is not None
            # Check if libarchive is available
            if use_libarchive:
                try:
                    import libarchive
                except (TypeError, ImportError):
                    # TypeError is raised if underlying C libarchive is not available
                    use_libarchive = False

        zf = zipfile.ZipFile(filename, mode="r", allowZip64=False)
        f_names = zf.namelist()
        if use_libarchive:
            zf.close()

    else:
        f_names = os.listdir(filename)

    regions: list[str] = [
        fn[9:-4] for fn in f_names if fn.startswith("Spectrum_") and fn.endswith(".bin")
    ]

    if len(regions) == 0:
        raise InvalidDA30ZipError(filename)

    out: list[xr.DataArray] = []

    for region in regions:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if zipped:
                required_files: set[str] = {
                    f"Spectrum_{region}.ini",
                    f"{region}.ini",
                }
                if not without_values:
                    required_files.add(f"Spectrum_{region}.bin")

                if use_libarchive:
                    # Extract required files using libarchive
                    extracted = 0

                    with libarchive.file_reader(str(filename)) as archive:
                        for entry in archive:
                            name = entry.pathname
                            if name in required_files:
                                with (pathlib.Path(tmp_dir) / name).open("wb") as f:
                                    for block in entry.get_blocks():
                                        f.write(block)
                                extracted += 1
                                if extracted == len(required_files):
                                    break
                else:
                    for f in required_files:
                        zf.extract(f, tmp_dir)

                unzipped = pathlib.Path(tmp_dir)
            else:
                unzipped = pathlib.Path(filename)

            region_info = parse_ini(unzipped / f"Spectrum_{region}.ini")["spectrum"]
            attrs = {}
            for d in parse_ini(unzipped / f"{region}.ini").values():
                attrs.update(d)

            if not without_values:
                arr = np.fromfile(unzipped / f"Spectrum_{region}.bin", dtype=np.float32)

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

        if not without_values:
            arr = arr.reshape(shape)
        else:
            arr = np.zeros(shape, dtype=np.float32)

        out.append(
            xr.DataArray(arr, coords=coords, name=region_info["name"], attrs=attrs)
        )

    if zipped and not use_libarchive:
        zf.close()

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


def parse_ini(filename: str | os.PathLike) -> dict:
    """Parse an ``.ini`` file into a dictionary."""
    parser = CasePreservingConfigParser(strict=False)
    out = {}
    with open(filename, encoding="utf-8") as f:
        parser.read_file(f)
        for section in parser.sections():
            out[section] = {k: _parse_value(v) for k, v in parser.items(section)}
    return out
