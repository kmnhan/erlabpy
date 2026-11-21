"""Xarray options that affect a recorded calculation."""

from __future__ import annotations

import contextlib
import textwrap
import typing
from dataclasses import dataclass

import xarray as xr

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

# Xarray adds options between releases. Record only options supported by the
# installed version, and reject a replay that requires an unavailable option.
_CALCULATION_OPTIONS = (
    "arithmetic_broadcast",
    "arithmetic_compat",
    "arithmetic_join",
    "keep_attrs",
    "use_bottleneck",
    "use_flox",
    "use_new_combine_kwarg_defaults",
    "use_numbagg",
    "use_opt_einsum",
)


@dataclass(frozen=True)
class ExecutionSettings:
    """Effective xarray calculation options in a stable order."""

    options: tuple[tuple[str, bool | str], ...] = ()

    @classmethod
    def from_options(cls, options: Mapping[str, bool | str]) -> typing.Self:
        if unknown := options.keys() - set(_CALCULATION_OPTIONS):
            raise ValueError(f"Unknown calculation options: {sorted(unknown)}")
        return cls(
            tuple((key, options[key]) for key in _CALCULATION_OPTIONS if key in options)
        )

    @classmethod
    def capture(cls) -> typing.Self:
        current = xr.get_options()
        return cls(
            tuple((key, current[key]) for key in _CALCULATION_OPTIONS if key in current)
        )

    def as_dict(self) -> dict[str, bool | str]:
        return dict(self.options)

    def wrap_code(self, statement: str) -> str:
        """Use the recorded settings while the notebook evaluates this step."""
        if not self.options:
            return statement
        options = ",\n".join(f"    {name}={value!r}" for name, value in self.options)
        return f"with xr.set_options(\n{options},\n):\n" + textwrap.indent(
            statement, "    "
        )

    @contextlib.contextmanager
    def scope(self) -> Iterator[None]:
        """Install the options in an interpreter owned by the calculation."""
        if not self.options:
            yield
            return
        with xr.set_options(**self.as_dict()):
            yield
