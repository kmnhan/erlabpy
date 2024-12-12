"""Data loader for beamline 20 LOREA at ALBA."""

import pathlib
import re
import struct
from typing import ClassVar

import numpy as np
import xarray as xr

import erlab
import erlab.io.nexusutils
from erlab.io.dataloader import LoaderBase


def _get_data(group):
    return group.get_default()[group.get_default().signal]


class LOREALoader(LoaderBase):
    name = "lorea"

    aliases = ("alba_bl20",)

    name_map: ClassVar[dict] = {
        "eV": ["instrument.analyser.energies", "energies"],
        "alpha": ["instrument.analyser.angles", "angles"],
        "beta": ["instrument.analyser.defl_angles", "defl_angles"],
        "delta": "instrument.manipulator.saazimuth",  # azi
        "chi": "instrument.manipulator.sapolar",  # polar
        "xi": "instrument.manipulator.satilt",  # tilt
        "x": "instrument.manipulator.sax",
        "y": "instrument.manipulator.say",
        "z": "instrument.manipulator.saz",
        "hv": "instrument.monochromator.energy",
        "sample_temp": "sample.temperature",
    }

    coordinate_attrs = ("beta", "delta", "chi", "xi", "hv", "x", "y", "z")
    additional_attrs: ClassVar[dict] = {"configuration": 3}

    skip_validate: bool = True
    always_single: bool = True

    @property
    def file_dialog_methods(self):
        return {"ALBA BL20 LOREA Raw Data (*.nxs *.krx)": (self.load, {})}

    def load_single(self, file_path, without_values: bool = False) -> xr.DataArray:
        if pathlib.Path(file_path).suffix == ".krx":
            return self._load_krx(file_path)

        return erlab.io.nexusutils.nxgroup_to_xarray(
            erlab.io.nexusutils.get_entry(file_path), _get_data, without_values
        )

    def identify(self, num, data_dir, krax=False):
        if krax:
            target_files = erlab.io.utils.get_files(data_dir, ".krx")
            pattern = re.compile(rf".+-\d-{str(num).zfill(5)}_\d.krx")
        else:
            target_files = erlab.io.utils.get_files(data_dir, ".nxs")
            pattern = re.compile(rf"{str(num).zfill(3)}_.+.nxs")

        matches = [path for path in target_files if pattern.match(path.name)]

        return matches, {}

    def _load_krx(self, file_path):
        # Adapted from `load_krax_FS.ipf` Igor procedure by Felix Baumberger

        def read_header(file, pos, size):
            file.seek(pos)
            return file.read(size).decode("utf-8")

        def header_to_dict(header_str):
            out = {}
            for row in header_str.strip().split("\r\n"):
                if row.startswith("DATA:"):
                    break
                k, v = row.split("\t")
                try:
                    v = float(v)
                except ValueError:
                    pass
                else:
                    if v.is_integer():
                        v = int(v)
                out[k] = v
            return out

        with open(file_path, "rb") as file:
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

            if header.startswith("Lines\t"):
                e0 = header_dict.get("Start K.E.", None)
                e1 = header_dict.get("End K.E.", None)
                x0 = header_dict.get("ScaleMin", None)
                x1 = header_dict.get("ScaleMax", None)
                y0 = header_dict.get("MapStartX", None)
                y1 = header_dict.get("MapEndX", None)
            else:
                # Old header format
                e0 = header_dict.get("Start K.E.", None)
                e1 = header_dict.get("End K.E.", None)
                x0 = header_dict.get("XScaleMin", None)
                x1 = header_dict.get("XScaleMax", None)
                y0 = header_dict.get("YScaleMin", None)
                y1 = header_dict.get("YScaleMax", None)

            data = []
            for ii in range(n_images):
                file.seek((ii * 3 + 1) * num_bytes)
                image_pos = struct.unpack(fmt, file.read(num_bytes))[0]

                file.seek(image_pos * 4)
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
