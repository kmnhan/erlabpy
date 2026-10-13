"""Validated workspace state for the Angle calculator."""

from __future__ import annotations

import dataclasses
import math
import typing

import numpy as np
import pydantic

from erlab.interactive._momentum_overlays._state import MomentumOverlayState
from erlab.interactive._qt_state import QtWindowState
from erlab.interactive.angle_calculator._calculation import Geometry
from erlab.lattice import abc2avec


class AngleCalculatorSettings(pydantic.BaseModel):
    """Calculator settings recorded in Undo and Redo history."""

    model_config = pydantic.ConfigDict(allow_inf_nan=False)

    geometry: Geometry = pydantic.Field(default_factory=Geometry)
    alpha_range: tuple[float, float] = (-15.0, 15.0)
    polar_range: tuple[float, float] = (-15.0, 15.0)
    tilt_range: tuple[float, float] = (-15.0, 15.0)
    deflector_range: tuple[float, float] = (-15.0, 15.0)
    scan_positions: dict[str, float] = pydantic.Field(
        default_factory=lambda: {"polar": 0.0, "tilt": 0.0, "deflector": 0.0}
    )
    lattice_a: float = pydantic.Field(default=6.59, gt=0, le=1000)
    lattice_b: float = pydantic.Field(default=6.59, gt=0, le=1000)
    lattice_c: float = pydantic.Field(default=6.01, gt=0, le=1000)
    lattice_alpha: float = pydantic.Field(default=90.0, gt=0, lt=180)
    lattice_beta: float = pydantic.Field(default=90.0, gt=0, lt=180)
    lattice_gamma: float = pydantic.Field(default=120.0, gt=0, lt=180)
    centering: typing.Literal["P", "A", "B", "C", "I", "F", "R"] = "P"
    lattice_rotation: float = 0.0
    hv_range: tuple[float, float] = (20.0, 120.0)
    out_of_plane_visible: bool = False
    overlays: MomentumOverlayState = pydantic.Field(
        default_factory=MomentumOverlayState
    )
    cutoff_visible: bool = False
    bz_visible: bool = True

    @pydantic.model_validator(mode="after")
    def valid_state(self):
        geometry = dataclasses.asdict(self.geometry)
        if not all(math.isfinite(value) for value in geometry.values()):
            raise ValueError("Geometry values must be finite.")
        if self.geometry.configuration not in (1, 2, 3, 4):
            raise ValueError("Unknown analyzer configuration.")
        if (
            not 0.001 <= self.geometry.photon_energy <= 1000
            or not -100000 <= self.geometry.binding_energy <= 0
        ):
            raise ValueError(
                "Use positive photon energy and nonpositive binding energy."
            )
        if not all(
            0 <= value <= 100
            for value in (
                self.geometry.analyzer_work_function,
                self.geometry.sample_work_function,
            )
        ):
            raise ValueError("Work functions must be between 0 and 100 eV.")
        if not -360 <= self.geometry.azimuth <= 360 or not all(
            -90 <= value <= 90 for value in (self.geometry.polar, self.geometry.tilt)
        ):
            raise ValueError("Sample angles exceed the supported range.")
        for name in ("alpha", "polar", "tilt", "deflector"):
            lower, upper = getattr(self, f"{name}_range")
            limit = 45 if name == "deflector" else 90
            if not -limit <= lower <= upper <= limit:
                raise ValueError(
                    f"{name.capitalize()} intervals must be ordered within ±{limit}°."
                )
        if set(self.scan_positions) != {"polar", "tilt", "deflector"}:
            raise ValueError("All three scan positions are required.")
        for name, position in self.scan_positions.items():
            lower, upper = getattr(self, f"{name}_range")
            if not lower <= position <= upper:
                raise ValueError("The current cut must be within its scan interval.")
        if not 0.0 <= self.geometry.inner_potential <= 50.0:
            raise ValueError("Inner potential must be between 0 and 50 eV.")
        if not 5 <= self.hv_range[0] <= self.hv_range[1] <= 1000:
            raise ValueError(
                "Photon energy limits must be ordered between 5 and 1000 eV."
            )
        vectors = abc2avec(
            self.lattice_a,
            self.lattice_b,
            self.lattice_c,
            self.lattice_alpha,
            self.lattice_beta,
            self.lattice_gamma,
        )
        if not np.isfinite(vectors).all() or abs(np.linalg.det(vectors)) < 1e-8:
            raise ValueError("The cell must span a volume.")
        return self


class AngleCalculatorState(AngleCalculatorSettings):
    """Workspace settings with plot ranges and native window state."""

    momentum_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.2, 0.2),
        (-0.2, 0.2),
    )
    angle_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-35.0, 35.0),
        (-35.0, 35.0),
    )
    out_of_plane_range: tuple[tuple[float, float], tuple[float, float]] = (
        (-20.0, 20.0),
        (1.0, 7.0),
    )
    window_state: QtWindowState = pydantic.Field(default_factory=QtWindowState)

    @pydantic.model_validator(mode="after")
    def valid_view_state(self):
        for limits in (self.momentum_range, self.angle_range, self.out_of_plane_range):
            if any(lower >= upper for lower, upper in limits):
                raise ValueError("Plot limits must increase.")
        return self
