import arpes
import numpy as np

from erlab.io.igor import load_ibw


def parse_livepolar(wave, normalize=False):
    wave = wave.rename({"W": "eV", "X": "phi", "Y": "theta"})
    new_coords = {}
    new_coords["alpha"] = np.pi / 2
    new_coords["beta"] = np.deg2rad(wave.attrs["tilt"])
    new_coords["phi"] = np.deg2rad(wave["phi"])
    new_coords["theta"] = np.deg2rad(wave["theta"])
    new_coords["chi"] = np.deg2rad(wave.attrs["azimuth"])
    new_coords["hv"] = wave.attrs["hv"]
    new_coords["psi"] = 0.0
    new_coords["eV"] = wave["eV"] - wave.attrs["hv"]
    wave = wave.assign_coords(new_coords)
    wave = wave / wave.attrs["mesh_current"]
    if normalize:
        wave = arpes.preparation.normalize_dim(wave, "theta")
    return wave


def parse_livexy(wave):
    wave = wave.rename({"W": "eV", "X": "y", "Y": "x"})
    new_coords = {}
    new_coords["alpha"] = np.pi / 2
    new_coords["beta"] = np.deg2rad(wave.attrs["tilt"])
    # new_coords["phi"] = np.deg2rad(wave["phi"])
    new_coords["theta"] = np.deg2rad(wave.attrs["polar"])
    new_coords["chi"] = np.deg2rad(wave.attrs["azimuth"])
    new_coords["hv"] = wave.attrs["hv"]
    new_coords["psi"] = 0.0
    new_coords["eV"] = wave["eV"] - wave.attrs["hv"]
    wave = wave.assign_coords(new_coords)
    wave = wave / wave.attrs["mesh_current"]
    return wave


def load_livexy(filename, data_dir=None):
    dat = load_ibw(filename, data_dir)
    return parse_livexy(dat)


def load_livepolar(filename, data_dir=None):
    dat = load_ibw(filename, data_dir)
    return parse_livepolar(dat)


def load(filename, data_dir=None, **kwargs):
    return arpes.io.load_data(filename, location="BL4", data_dir=data_dir, **kwargs)
