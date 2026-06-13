"""Matplotlib style and rcParams defaults for Figure Composer."""

from __future__ import annotations

import contextlib
import typing
import warnings

import matplotlib as mpl
from matplotlib import style as mpl_style

import erlab.interactive._stylesheets

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure

_MM_PER_INCH = 25.4
_LAYOUT_COLLAPSED_WARNING = (
    r"constrained_layout not applied because axes sizes collapsed to zero\."
)


def _configured_stylesheets() -> tuple[str, ...]:
    with contextlib.suppress(Exception):
        from erlab.interactive._options import options

        return tuple(options.model.figure.stylesheets)
    return ()


def _available_configured_stylesheets() -> tuple[str, ...]:
    configured = _configured_stylesheets()
    available = erlab.interactive._stylesheets.available_stylesheets(configured)
    return tuple(name for name in configured if name in available)


def _unavailable_configured_stylesheets() -> tuple[str, ...]:
    configured = _configured_stylesheets()
    available = erlab.interactive._stylesheets.available_stylesheets(configured)
    return tuple(name for name in configured if name not in available)


@contextlib.contextmanager
def _figure_style_context() -> Iterator[None]:
    stylesheets = _available_configured_stylesheets()
    if not stylesheets:
        yield
        return
    with mpl_style.context(list(stylesheets)):
        yield


def _styled_rcparams_value(key: str) -> typing.Any:
    with _figure_style_context():
        return typing.cast("typing.Any", mpl.rcParams)[key]


def _default_figsize() -> tuple[float, float]:
    width, height = _styled_rcparams_value("figure.figsize")
    return (float(width), float(height))


def _default_figure_dpi() -> float:
    return float(_styled_rcparams_value("figure.dpi"))


def _apply_figure_dpi(figure: Figure, dpi: float) -> None:
    figure_any = typing.cast("typing.Any", figure)
    if getattr(figure_any, "_original_dpi", dpi) == dpi:
        return
    figure_any._original_dpi = dpi
    figure_any._set_dpi(
        dpi * getattr(figure_any.canvas, "device_pixel_ratio", 1),
        forward=False,
    )


def _default_layout() -> typing.Literal["constrained", "tight"] | None:
    if bool(_styled_rcparams_value("figure.constrained_layout.use")):
        return "constrained"
    if bool(_styled_rcparams_value("figure.autolayout")):
        return "tight"
    return None


def _default_export_dpi() -> float | typing.Literal["figure"]:
    value = _styled_rcparams_value("savefig.dpi")
    if value == "figure":
        return "figure"
    return float(value)


def _default_export_transparent() -> bool:
    return bool(_styled_rcparams_value("savefig.transparent"))


def _default_export_bbox_inches() -> str | None:
    value = _styled_rcparams_value("savefig.bbox")
    return None if value is None else str(value)


@contextlib.contextmanager
def _figure_draw_context() -> Iterator[None]:
    with _figure_style_context(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_LAYOUT_COLLAPSED_WARNING,
            category=UserWarning,
        )
        yield


def _style_code_lines() -> list[str]:
    lines: list[str] = []
    available = _available_configured_stylesheets()
    unavailable = _unavailable_configured_stylesheets()
    if available:
        lines.append(f"plt.style.use({list(available)!r})")
    if unavailable:
        lines.append(
            "# Skipped unavailable stylesheets: "
            + ", ".join(repr(name) for name in unavailable)
        )
    return lines


def _style_required_imports() -> tuple[str, ...]:
    if erlab.interactive._stylesheets.stylesheets_require_erlab_plotting(
        _configured_stylesheets()
    ):
        return ("import erlab.plotting  # registers ERLab matplotlib stylesheets",)
    return ()
