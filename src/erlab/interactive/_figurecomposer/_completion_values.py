"""Curated completion values and their text conversion helpers."""

from __future__ import annotations

import typing

from erlab.interactive._figurecomposer._text import _code_args, _literal_from_text

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

LABEL_COMPLETIONS: tuple[str, ...] = (
    r"$E-E_F$ (eV)",
    r"$E$ (eV)",
    r"$E_{\mathrm{kin}}$ (eV)",
    r"$k_x$ (Å${}^{-1}$)",
    r"$k_y$ (Å${}^{-1}$)",
    r"$k_z$ (Å${}^{-1}$)",
    r"$k_{||}$ (Å${}^{-1}$)",
    r"$h\nu$ (eV)",
    r"$\alpha$ (deg)",
    r"$\beta$ (deg)",
    r"$\theta$ (deg)",
    r"$\phi$ (deg)",
    r"$T$ (K)",
    "Temperature (K)",
    "Intensity (arb. units)",
)

FONT_SIZE_COMPLETIONS: tuple[str, ...] = (
    "xx-small",
    "x-small",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
    "smaller",
    "larger",
)


def _format_completion_literal(value: typing.Any, *, completions: Sequence[str]) -> str:
    """Format a literal while showing curated string values without quotes."""
    if value is None:
        return ""
    if isinstance(value, str) and value in completions:
        return value
    return _code_args((value,))


def _completion_literal_from_text(
    text: str, *, completions: Sequence[str]
) -> typing.Any:
    """Parse curated strings directly and other text as a Python literal."""
    stripped = text.strip()
    if not stripped:
        return None
    if stripped in completions:
        return stripped
    return _literal_from_text(stripped)
