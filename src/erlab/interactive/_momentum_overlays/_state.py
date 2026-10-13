"""Validated state for momentum-image and pocket overlays."""

from __future__ import annotations

import base64
import typing
import uuid

import pydantic


class PocketState(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(allow_inf_nan=False)

    uid: str = pydantic.Field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    center: tuple[float, float] = (0.0, 0.0)
    radii: tuple[float, float] = (0.05, 0.05)
    rotation: float = pydantic.Field(default=0.0, ge=-360, le=360)
    visible: bool = True
    symmetry: bool = True
    point_group: str = "1"
    symmetry_angle: float = pydantic.Field(default=0.0, ge=-360, le=360)
    color: str = "#e69f00"

    @pydantic.field_validator("point_group")
    @classmethod
    def valid_point_group(cls, value):
        if value not in (
            "1",
            "2",
            "3",
            "4",
            "6",
            "m",
            "2mm",
            "3m",
            "4mm",
            "6mm",
        ):
            raise ValueError("Unknown planar point group.")
        return value

    @pydantic.field_validator("radii")
    @classmethod
    def positive_radii(cls, value):
        if not all(0.00001 <= radius <= 100 for radius in value):
            raise ValueError("Pocket radii must be between 0.00001 and 100 Å⁻¹.")
        return value

    @pydantic.field_validator("center")
    @classmethod
    def bounded_center(cls, value):
        if not all(-100 <= coordinate <= 100 for coordinate in value):
            raise ValueError("Pocket centers must be within ±100 Å⁻¹.")
        return value


class ImageState(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(allow_inf_nan=False)

    png_base64: str
    center: tuple[float, float] = (0.0, 0.0)
    width: float = pydantic.Field(default=0.3, ge=0.00001, le=1000)
    rotation: float = pydantic.Field(default=0.0, ge=-360, le=360)
    opacity: float = pydantic.Field(default=0.65, ge=0, le=1)
    visible: bool = True
    locked: bool = False
    name: str = "Image"
    invert: bool = False
    black_transparent: bool = False
    black_threshold: int = pydantic.Field(default=0, ge=0, le=255)

    @pydantic.field_validator("center")
    @classmethod
    def bounded_center(cls, value):
        if not all(-100 <= coordinate <= 100 for coordinate in value):
            raise ValueError("Image centers must be within ±100 Å⁻¹.")
        return value

    @pydantic.field_validator("png_base64")
    @classmethod
    def valid_png(cls, value):
        if not base64.b64decode(value, validate=True).startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("The embedded image must be a PNG.")
        return value


class MomentumOverlayState(pydantic.BaseModel):
    pockets: list[PocketState] = pydantic.Field(default_factory=list)
    image: ImageState | None = None

    @pydantic.model_validator(mode="after")
    def unique_pockets(self):
        if len({pocket.uid for pocket in self.pockets}) != len(self.pockets):
            raise ValueError("Pocket identifiers must be unique.")
        return self


class PocketClipboard(pydantic.BaseModel):
    """Versioned pocket data shared by the momentum overlay editors."""

    version: typing.Literal[1]
    pockets: list[PocketState] = pydantic.Field(min_length=1)
