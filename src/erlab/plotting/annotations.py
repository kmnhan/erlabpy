"""Plot annotations."""

from __future__ import annotations

__all__ = [
    "PRETTY_NAMES",
    "PRETTY_UNITS",
    "copy_mathtext",
    "fancy_labels",
    "integer_ticks",
    "label_subplot_properties",
    "label_subplots",
    "mark_points",
    "mark_points_outside",
    "property_labels",
    "scale_units",
    "set_titles",
    "set_xlabels",
    "set_ylabels",
    "sizebar",
]

import contextlib
import io
import re
import typing
from collections.abc import Iterable

import matplotlib
import matplotlib.figure
import matplotlib.font_manager
import matplotlib.mathtext
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms as mtransforms
import numpy as np

from erlab.accessors.utils import either_dict_or_kwargs
from erlab.plotting.colors import axes_textcolor

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import pyperclip
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
else:
    import lazy_loader as _lazy

    pyperclip = _lazy.load("pyperclip")

PRETTY_NAMES: dict[str, tuple[str, str]] = {
    "temperature": ("Temperature", "Temperature"),
    "T": (r"\ensuremath{T}", r"$T$"),
    "sample_temp": (r"\ensuremath{T}", r"$T$"),
    "t": (r"\ensuremath{t}", r"$t$"),
    "beta": (r"\ensuremath{\beta}", r"$\beta$"),
    "theta": (r"\ensuremath{\theta}", r"$\theta$"),
    "chi": (r"\ensuremath{\chi}", r"$\chi$"),
    "alpha": (r"\ensuremath{\alpha}", r"$\alpha$"),
    "psi": (r"\ensuremath{\psi}", r"$\psi$"),
    "phi": (r"\ensuremath{\phi}", r"$\phi$"),
    "xi": (r"\ensuremath{\xi}", r"$\xi$"),
    "Eb": (r"\ensuremath{E}", r"$E$"),
    "Ek": (r"\ensuremath{E_{\text{kin}}}", r"$E_{\text{kin}}$"),
    "eV": (r"\ensuremath{E-E_F}", r"$E-E_F$"),
    "kx": (r"\ensuremath{k_{x}}", r"$k_x$"),
    "ky": (r"\ensuremath{k_{y}}", r"$k_y$"),
    "kz": (r"\ensuremath{k_{z}}", r"$k_z$"),
    "kp": (r"\ensuremath{k_{||}}", r"$k_{||}$"),
    "hv": (r"\ensuremath{h\nu}", r"$h\nu$"),
}
"""Pretty names for automated labeling of plots.

Contains a mapping from dimension names to tuples of strings. The first element of the
tuple is used when matplotlib is configured to use LaTeX, and the second is used for
`Mathtext <https://matplotlib.org/stable/users/explain/text/mathtext.html>`_. Unless you
have explicitly set the ``text.usetex`` parameter in your matplotlib configuration, the
second element will be used.

"""

PRETTY_UNITS: dict[str, tuple[str, str]] = {
    "temperature": (r"K", r"K"),
    "T": (r"K", r"K"),
    "sample_temp": (r"K", r"K"),
    "t": (r"s", r"s"),
    "theta": (r"deg", r"deg"),
    "beta": (r"deg", r"deg"),
    "psi": (r"deg", r"deg"),
    "chi": (r"deg", r"deg"),
    "alpha": (r"deg", r"deg"),
    "phi": (r"deg", r"deg"),
    "xi": (r"deg", r"deg"),
    "Eb": (r"eV", r"eV"),
    "Ek": (r"eV", r"eV"),
    "eV": (r"eV", r"eV"),
    "hv": (r"eV", r"eV"),
    "kx": (r"Å\ensuremath{{}^{-1}}", r"Å${}^{-1}$"),
    "ky": (r"Å\ensuremath{{}^{-1}}", r"Å${}^{-1}$"),
    "kz": (r"Å\ensuremath{{}^{-1}}", r"Å${}^{-1}$"),
    "kp": (r"Å\ensuremath{{}^{-1}}", r"Å${}^{-1}$"),
}
"""Pretty units for automated labeling of plots.

Contains a mapping from dimension names to tuples of strings. The first element of the
tuple is used when matplotlib is configured to use LaTeX, and the second is used for
`Mathtext <https://matplotlib.org/stable/users/explain/text/mathtext.html>`_. Unless you
have explicitly set the ``text.usetex`` parameter in your matplotlib configuration, the
second element will be used.

"""

SI_PREFIXES: dict[int, str] = {
    24: "Y",
    21: "Z",
    18: "E",
    15: "P",
    12: "T",
    9: "G",
    6: "M",
    3: "k",
    2: "h",
    1: "da",
    0: "",
    -1: "d",
    -2: "c",
    -3: "m",
    -6: "μ",
    -9: "n",
    -12: "p",
    -15: "f",
    -18: "a",
    -21: "z",
    -24: "y",
}  #: Maps powers of 10 to valid SI prefix strings.

SI_PREFIX_NAMES: tuple[str, ...] = (
    "yotta",
    "zetta",
    "exa",
    "peta",
    "tera",
    "giga",
    "mega",
    "kilo",
    "hecto",
    "deca",
    "",
    "deci",
    "centi",
    "milli",
    "micro",
    "nano",
    "pico",
    "femto",
    "atto",
    "zepto",
    "yocto",
)  #: Names of the SI prefixes.


def _alph_label(val, prefix, suffix, numeric, capital):
    """Generate labels from string or integer."""
    if isinstance(val, int | np.integer) or val.isdigit():
        if numeric:
            val = str(val)
        else:
            ref_char = "A" if capital else "a"
            val = chr(int(val) + ord(ref_char) - 1)
    elif not isinstance(val, str):
        raise TypeError("Input values must be integers or strings.")
    return prefix + val + suffix


def _unit_from_label(label: str) -> str | None:
    """Try to determine the unit from a given label.

    Returns None if it fails to determine the unit.

    """
    m = re.match(r".*\((.*)\)\s*?$", label)
    if m is None:
        return None
    try:
        return m.group(1)
    except IndexError:
        return None


def get_si_str(si: int) -> str:
    """Return the SI prefix string to be plotted by :mod:`matplotlib`.

    Parameters
    ----------
    si : int
        Exponent of 10 corresponding to a SI prefix.

    Returns
    -------
    str
        SI prefix corresponding to ``si``.

    """
    if plt.rcParams["text.usetex"] and si == -6:
        return "\\ensuremath{\\mu}"
    try:
        return SI_PREFIXES[si]
    except KeyError as e:
        raise ValueError("Invalid SI prefix.") from e


def name_for_dim(dim_name: str, escaped: bool = True) -> str:
    """Return the pretty name for a given dimension."""
    names: tuple[str, str] | None = PRETTY_NAMES.get(dim_name)

    if names is None:
        name = dim_name
    else:
        name = names[0] if plt.rcParams["text.usetex"] else names[1]

    if not escaped:
        name = name.replace("$", "")
    return name


def unit_for_dim(dim_name: str, radians: bool = False) -> str:
    """Return the pretty unit for a given dimension."""
    units: tuple[str, str] | None = PRETTY_UNITS.get(dim_name)

    if units is None:
        unit = ""
    else:
        unit = units[0] if plt.rcParams["text.usetex"] else units[1]

    if radians:
        unit = unit.replace("deg", "rad")
    return unit


def label_for_dim(dim_name: str, radians: bool = False, escaped: bool = True) -> str:
    """Return the pretty label for a given dimension."""
    name = name_for_dim(dim_name, escaped=escaped)
    unit = unit_for_dim(dim_name, radians=radians)
    if unit == "":
        return name
    return f"{name} ({unit})"


def parse_special_point(name: str) -> str:
    special_points = {"G": r"\Gamma", "D": r"\Delta"}

    if name in special_points:
        return special_points[name]

    return name


def parse_point_labels(name: str, roman: bool = True, bar: bool = False) -> str:
    name = parse_special_point(name)

    if name.endswith("*"):
        name = name[:-1]
        format_str = "\\mathdefault{{{}}}^*" if roman else "{}^*"
    elif name.endswith("'"):
        name = name[:-1]
        format_str = "\\mathdefault{{{}}}\\prime" if roman else "{}\\prime"
    elif roman:
        format_str = r"\mathdefault{{{}}}"
    else:
        format_str = r"{}"

    name = format_str.format(parse_special_point(name))

    return f"$\\overline{{{name}}}$" if bar else f"${name}$"


def copy_mathtext(
    s: str,
    fontsize: float
    | typing.Literal[
        "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"
    ]
    | None = None,
    fontproperties: matplotlib.font_manager.FontProperties | None = None,
    outline: bool = False,
    rcparams: dict | None = None,
    **mathtext_rc,
) -> str:
    """Copy math equations to the clipboard as SVG.

    Parameters
    ----------
    s
        The math text to convert to SVG.
    fontsize
        The font size to use for the math text. If None, the default font size is used.
    fontproperties
        The font properties to use for the math text. If None, the default font
        properties are used.
    outline
        If True, the SVG will contain outlined paths instead of text.
    rcparams
        Additional rc parameters to use for the math text rendering. If None, the
        default rc parameters are used.

    Returns
    -------
    str
        The SVG string representation of the math text.

    Example
    -------
    >>> import erlab.plotting as eplt
    >>> svg_str = eplt.copy_mathtext(r"$E = mc^2$", fontsize=12)

    """
    from matplotlib.backends.backend_svg import FigureCanvasSVG

    if fontproperties is None:
        fontproperties = matplotlib.font_manager.FontProperties(size=fontsize)
    else:
        fontproperties.set_size(fontsize)
    if rcparams is None:
        rcparams = {}
    parser = matplotlib.mathtext.MathTextParser("path")
    width, height, depth, _, _ = parser.parse(s, dpi=72, prop=fontproperties)

    fig = matplotlib.figure.Figure(figsize=(width / 72, height / 72))
    fig.patch.set_facecolor("none")
    fig.text(0, depth / height, s, fontproperties=fontproperties)

    FigureCanvasSVG(fig)

    for k, v in mathtext_rc.items():
        if k in ["bf", "cal", "it", "rm", "sf", "tt"] and isinstance(
            v, matplotlib.font_manager.FontProperties
        ):
            v = v.get_fontconfig_pattern()
        rcparams[f"mathtext.{k}"] = v

    with io.BytesIO() as buffer:
        rcparams.setdefault("svg.fonttype", "path" if outline else "none")
        rcparams.setdefault("svg.image_inline", True)
        with plt.rc_context(rcparams):
            fig.canvas.print_svg(buffer)  # type: ignore[attr-defined]

        buffer_str = buffer.getvalue().decode("utf-8")

    with contextlib.suppress(pyperclip.PyperclipException):
        # Try to copy to clipboard, if available
        pyperclip.copy(buffer_str)
    return buffer_str


def fancy_labels(
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    *,
    radians: bool = False,
) -> None:
    """Apply pretty labels to Matplotlib axes based on their current label text.

    This function converts plain dimension names already set on the axes into nicely
    formatted labels (e.g., symbols and units). The labels are determined based on
    :attr:`PRETTY_NAMES <erlab.plotting.annotations.PRETTY_NAMES>` and
    :attr:`PRETTY_UNITS <erlab.plotting.annotations.PRETTY_UNITS>`.

    Parameters
    ----------
    ax
        The target axes or an iterable of axes. If None (default), the current axes is
        used.
    radians : bool, optional
        If `True`, angle units will be displayed in radians instead of degrees.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import erlab.plotting as eplt
    >>> fig, ax = plt.subplots()
    >>> ax.set_xlabel("kx")
    >>> ax.set_ylabel("eV")
    >>> fancy_labels(ax)
    >>> ax.get_xlabel()
    '$k_x$ (Å${}^{-1}$)'
    >>> ax.get_ylabel()
    '$E-E_F$ (eV)'

    """
    if ax is None:
        ax = plt.gca()
    if np.iterable(ax):
        for axi in ax:
            fancy_labels(axi, radians=radians)
        return

    ax.set_xlabel(label_for_dim(dim_name=ax.get_xlabel(), radians=radians))
    ax.set_ylabel(label_for_dim(dim_name=ax.get_ylabel(), radians=radians))
    if hasattr(ax, "get_zlabel") and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(label_for_dim(dim_name=ax.get_zlabel(), radians=radians))


def property_labels(
    values: dict | None = None,
    decimals: int | None = None,
    si: int = 0,
    name: str | None = None,
    unit: str | None = None,
    order: typing.Literal["C", "F", "A", "K"] = "C",
    **values_kwargs,
) -> list[str]:
    """Generate labels from a dictionary of values.

    Given a dictionary of values, this function generates a list of label strings  with
    the key as the dimension name and the value as the dimension value. Multiple
    key-value pairs will be separated by a newline character in each label. The name and
    unit will be automatically determined from the key.

    Parameters
    ----------
    values
        Key-value pair of annotations. The values may be a single value or an array-like
        of values. If a single value, it will be converted to a list of length 1. If an
        array of 2 or more dimensions is given, it will be flattened with the order
        given by `order`. All values must be of the same length when flattened.
    decimals
        Number of decimal places to round to. If decimals is None, no rounding is
        performed. If decimals is negative, it specifies the number of positions to the
        left of the decimal point.
    si
        Powers of 10 for automatic SI prefix setting.
    name
        When set, overrides automatic dimension name setting.
    unit
        When set, overrides automatic unit setting.
    order
        Order in which to flatten `ax`. 'C' means to flatten in row-major (C-style)
        order. 'F' means to flatten in column-major (Fortran-style) order. The default
        is 'C'.

    """
    values = either_dict_or_kwargs(values, values_kwargs, "property_labels")
    strlist: typing.Any = []
    for k, v in values.items():
        if not isinstance(v, tuple | list | np.ndarray):
            v = [v]
        else:
            v = np.array(v).flatten(order=order)
        strlist.append(
            [
                property_label(
                    k, val, decimals=decimals, si=si, name=name, unit=unit
                ).strip()
                for val in v
            ]
        )
    strlist = list(zip(*strlist, strict=True))
    return ["\n".join(strlist[i]) for i in range(len(strlist))]


def label_subplot_properties(
    axes: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes],
    values: dict,
    decimals: int | None = None,
    si: int = 0,
    name: str | None = None,
    unit: str | None = None,
    order: typing.Literal["C", "F", "A", "K"] = "C",
    **kwargs,
) -> None:
    r"""Labels subplots with automatically generated labels.

    Parameters
    ----------
    axes
        `matplotlib.axes.Axes` to label. If an array is given, the order will be
        determined by the flattening method given by `order`.
    values
        Key-value pair of annotations. The values may be a single value or an array-like
        of values. If a single value, it will be converted to a list of length 1. If an
        array of 2 or more dimensions is given, it will be flattened with the order
        given by `order`. All values must be of the same length when flattened.
    decimals
        Number of decimal places to round to. If decimals is None, no rounding is
        performed. If decimals is negative, it specifies the number of positions to the
        left of the decimal point.
    si
        Powers of 10 for automatic SI prefix setting.
    name
        When set, overrides automatic dimension name setting.
    unit
        When set, overrides automatic unit setting.
    order
        Order in which to flatten `ax`. 'C' means to flatten in row-major (C-style)
        order. 'F' means to flatten in column-major (Fortran-style) order. The default
        is 'C'.
    **kwargs
        Extra arguments to `erlab.plotting.annotations.label_subplots`.

    """
    kwargs.setdefault("fontweight", plt.rcParams["font.weight"])
    kwargs.setdefault("prefix", "")
    kwargs.setdefault("suffix", "")
    kwargs.setdefault("loc", "upper right")

    label_subplots(
        axes,
        property_labels(
            values, decimals=decimals, si=si, name=name, unit=unit, order=order
        ),
        order=order,
        **kwargs,
    )


def label_subplots(
    axes: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes],
    values: Iterable[int | str] | None = None,
    startfrom: int = 1,
    order: typing.Literal["C", "F", "A", "K"] = "C",
    loc: typing.Literal[
        "upper left",
        "upper center",
        "upper right",
        "center left",
        "center",
        "center right",
        "lower left",
        "lower center",
        "lower right",
    ] = "upper left",
    offset: tuple[float, float] = (0.0, 0.0),
    prefix: str = "",
    suffix: str = "",
    numeric: bool = False,
    capital: bool = False,
    fontweight: typing.Literal[
        "ultralight",
        "light",
        "normal",
        "regular",
        "book",
        "medium",
        "roman",
        "semibold",
        "demibold",
        "demi",
        "bold",
        "heavy",
        "extra bold",
        "black",
    ] = "normal",
    fontsize: (
        float
        | typing.Literal[
            "xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"
        ]
        | None
    ) = None,
    **kwargs,
) -> None:
    r"""Labels subplots with automatically generated labels.

    Parameters
    ----------
    axes
        `matplotlib.axes.Axes` to label. If an array is given, the order will be
        determined by the flattening method given by `order`.
    values
        Integer or string labels corresponding to each Axes in `axes` for
        manual labels.
    startfrom
        Start from this number when creating automatic labels. Has no
        effect when `values` is not `None`.
    order
        Order in which to flatten `ax`. 'C' means to flatten in
        row-major (C-style) order. 'F' means to flatten in column-major
        (Fortran-style) order. 'A' means to flatten in column-major
        order if a is Fortran contiguous in memory, row-major order
        otherwise. 'K' means to flatten a in the order the elements
        occur in memory. The default is 'C'.
    loc
        The box location. The default is ``'upper left'``.
    offset
        Values that are used to position the legend in conjunction with
        `loc`, given in display units.
    prefix
        String to prepend to the alphabet label.
    suffix
        String to append to the alphabet label.
    numeric
        Use integer labels instead of alphabets.
    capital
        Capitalize automatically generated alphabetical labels.
    fontweight
        Set the font weight. The default is ``'normal'``.
    fontsize
        Set the font size. The default is ``'medium'`` for axes, and ``'large'`` for
        figures.
    **kwargs
        Extra arguments to `matplotlib.text.Text`: refer to the `matplotlib`
        documentation for a list of all possible arguments.

    """
    kwargs["fontweight"] = fontweight
    if plt.rcParams["text.usetex"] & (fontweight == "bold"):
        prefix = "\\textbf{" + prefix
        suffix = suffix + "}"
        kwargs.pop("fontweight")

    axlist = np.array(axes, dtype=object).flatten(order=order)
    if values is None:
        value_arr = np.array(
            [i + startfrom for i in range(len(axlist))], dtype=np.int64
        )
    else:
        value_arr = np.array(values).flatten(order=order)
        if not (axlist.size == value_arr.size):
            raise IndexError(
                "The number of given values must match the number of given axes."
            )

    for i, ax in enumerate(axlist):
        if fontsize is None:
            fontsize = "large" if isinstance(ax, matplotlib.figure.Figure) else "medium"

        label_str = _alph_label(value_arr[i], prefix, suffix, numeric, capital)
        with plt.rc_context({"text.color": axes_textcolor(ax)}):
            at = matplotlib.offsetbox.AnchoredText(
                label_str,
                loc=loc,
                frameon=False,
                pad=0,
                borderpad=0.5,
                prop=dict(fontsize=fontsize, **kwargs),
                bbox_to_anchor=ax.bbox,
                bbox_transform=matplotlib.transforms.ScaledTranslation(
                    offset[0] / 72, offset[1] / 72, ax.get_figure().dpi_scale_trans
                ),
                clip_on=False,
            )
        ax.add_artist(at)


def mark_points(
    points: Sequence[float],
    labels: Sequence[str],
    *,
    y: float | Sequence[float] = 0.0,
    pad: tuple[float, float] = (0, 1.75),
    literal: bool = False,
    roman: bool = True,
    bar: bool = False,
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    **kwargs,
) -> None:
    """Mark points above the horizontal axis.

    Useful when annotating high symmetry points along a cut.

    Parameters
    ----------
    points
        Floats indicating the position of each label.
    labels
        Sequence of label strings indicating a high symmetry point. Must be the same
        length as ``points``.
    y
        Position of the label in data coordinates. If a single float is given, it will
        be used for all points. If a sequence is given, it must be the same length as
        ``points``.
    pad
        Offset of the text in points.
    literal
        If `True`, take the input string literally.
    roman
        If `False`, itallic fonts are used.
    bar
        If `True`, prints a bar over the label.
    ax
        `matplotlib.axes.Axes` to annotate. If `None`, the current axes is used.
    **kwargs
        Extra arguments to `matplotlib.axes.Axes.text`. By default, the horizontal
        alignment is set to ``'center'`` and the vertical alignment is set to
        ``'baseline'``. The text is not clipped to the axes limits.

    """
    if ax is None:
        ax = plt.gca()

    if np.iterable(ax):
        for a in np.asarray(ax, dtype=object).flatten():
            mark_points(
                points,
                labels,
                y=y,
                pad=pad,
                literal=literal,
                roman=roman,
                bar=bar,
                ax=a,
                **kwargs,
            )
    else:
        fig = ax.get_figure()

        if fig is None:
            raise ValueError("Given axes does not belong to a figure")
        kwargs.setdefault("ha", kwargs.pop("horizontalalignment", "center"))
        kwargs.setdefault("va", kwargs.pop("verticalalignment", "baseline"))
        kwargs.setdefault("clip_on", False)

        if not np.iterable(y):
            y = [y] * len(points)

        default_color = kwargs.pop("c", kwargs.pop("color", None))

        for xi, yi, label in zip(points, y, labels, strict=True):
            if default_color is None:
                color = "k" if ax.get_ylim()[1] < yi else axes_textcolor(ax)
            else:
                color = default_color
            ax.text(
                xi,
                yi,
                label if literal else parse_point_labels(label, roman, bar),
                transform=ax.transData
                + mtransforms.ScaledTranslation(
                    pad[0] / 72, pad[1] / 72, fig.dpi_scale_trans
                ),
                color=color,
                **kwargs,
            )


def mark_points_outside(
    points: Sequence[float],
    labels: Sequence[str],
    axis: typing.Literal["x", "y"] = "x",
    *,
    literal: bool = False,
    roman: bool = True,
    bar: bool = False,
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes] | None = None,
    **kwargs,
) -> None:
    """Mark points above the horizontal axis.

    Useful when annotating high symmetry points along a cut.

    Parameters
    ----------
    points
        Floats indicating the position of each label.
    labels
        Sequence of label strings indicating a high symmetry point. Must be the same
        length as `points`.
    axis
        If ``'x'``, marks points along the horizontal axis. If ``'y'``, marks points
        along the vertical axis.
    literal
        If `True`, take the input string literally.
    roman
        If ``False``, itallic fonts are used.
    bar
        If ``True``, prints a bar over the label.
    ax
        `matplotlib.axes.Axes` to annotate.
    **kwargs
        Extra arguments to `matplotlib.text.Text`: refer to the `matplotlib`
        documentation for a list of all possible arguments.
    """
    if ax is None:
        ax = plt.gca()
    if np.iterable(ax):
        for ax_i in np.asarray(ax, dtype=object).flat:
            mark_points_outside(
                points,
                labels=labels,
                axis=axis,
                literal=literal,
                roman=roman,
                bar=bar,
                ax=ax_i,
            )
    else:
        if axis == "x":
            label_ax = ax.twiny()
            label_ax.set_xlim(ax.get_xlim())
            label_ax.set_xticks(
                points,
                labels=[
                    lab if literal else parse_point_labels(lab, roman, bar)
                    for lab in labels
                ],
                **kwargs,
            )
        else:
            label_ax = ax.twinx()
            label_ax.set_ylim(ax.get_ylim())
            label_ax.set_yticks(
                points,
                labels=[
                    lab if literal else parse_point_labels(lab, roman, bar)
                    for lab in labels
                ],
                **kwargs,
            )
        label_ax.set_frame_on(False)


def property_label(key, value, decimals=None, si=0, name=None, unit=None) -> str:
    delim = "" if name == "" else " = "
    if name is None:
        name = name_for_dim(key, escaped=False)
        if name is None:
            name = ""
    if unit is None:
        unit = unit_for_dim(key)
        if unit is None:
            unit = ""

    unit = get_si_str(si) + unit
    value /= 10**si
    if decimals is not None:
        value = np.around(value, decimals=decimals)
    if int(value) == value:
        value = int(value)

    if key == "Eb":
        if value == 0:
            if delim == "":
                return "$E_F$"
            return "$E = E_F$"
        if delim == "":
            name = "E_F"
        else:
            delim += "E_F"
        if value > 0:
            delim += "+"

    base = "${}" + delim + "{}$ {}"
    return str(base.format(name, value, unit))


class _SIFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, si: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._si_exponent = int(si)

    def __call__(self, x, pos=None):
        self.orderOfMagnitude += self._si_exponent

        match_format = re.match(r".*%1.(\d+)f", self.format)
        if match_format is None:
            # Match failed, may be due to changes in matplotlib
            raise RuntimeError("Failed to match format string. Please report this bug")

        sigfigs = int(match_format.group(1))
        self.format = self.format.replace(
            f"%1.{sigfigs}f", f"%1.{max(0, sigfigs + self._si_exponent)}f"
        )
        val = super().__call__(x, pos)

        self.orderOfMagnitude -= self._si_exponent
        return val


def scale_units(
    ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes],
    axis: typing.Literal["x", "y", "z"],
    si: int = 0,
    *,
    prefix: bool = True,
    power: bool = False,
) -> None:
    """Rescales ticks and adds an SI prefix to the axis label.

    Useful when you want to rescale the ticks without actually rescaling the data. For
    example, when plotting a cut from a low pass energy scan, you might want to convert
    the energy units from eV to meV.

    Using this function on an axis where the major locator is not the default formatter
    `matplotlib.ticker.ScalarFormatter` will result in undefined behavior.

    Parameters
    ----------
    ax
        Axes or iterable of Axes to rescale.
    axis
        The axis you wish to rescale.
    si
        Exponent of 10 corresponding to a SI prefix.
    prefix
        If True, tries to detect the unit from the axis label and scales it accordingly.
        The scaling behaviour is controlled by the `power` argument. If no units are
        found in the axis label, it is silently ignored.
    power
        If False, prefixes the detected unit on the axis label with a SI prefix
        corresponding to `si`. If True, the unit is prefixed with a scientific notation
        instead.

    """
    if np.iterable(ax):
        for a in np.asarray(ax, dtype=object).flat:
            scale_units(a, axis, si, prefix=prefix, power=power)
        return

    getlabel = getattr(ax, f"get_{axis}label")
    setlabel = getattr(ax, f"set_{axis}label")

    label = getlabel()
    unit = _unit_from_label(label)

    getattr(ax, f"{axis}axis").set_major_formatter(_SIFormatter(si))

    if prefix and (unit is not None):
        if power:
            setlabel(label.replace(f"({unit})", f"($\\times{{{10}}}^{{{si}}}$ {unit})"))
        else:
            setlabel(label.replace(f"({unit})", f"({get_si_str(si)}{unit})"))


def integer_ticks(ax: matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes]) -> None:
    """Set the ticks on the x and y axes to only display integer values.

    Modifies the x and y ticks of the given axes to only show integer values that are
    within the current limits of the axes.

    Parameters
    ----------
    ax
        Axes or iterable of Axes.

    """
    if isinstance(ax, Iterable):
        for a in np.asarray(ax, dtype=object).flat:
            integer_ticks(a)
        return
    ax.set_xticks(
        [
            t
            for t in ax.get_xticks()
            if t.is_integer() and t >= ax.get_xlim()[0] and t <= ax.get_xlim()[1]
        ]
    )
    ax.set_yticks(
        [
            t
            for t in ax.get_yticks()
            if t.is_integer() and t >= ax.get_ylim()[0] and t <= ax.get_ylim()[1]
        ]
    )


def set_titles(axes, labels: Iterable[str] | str, order="C", **kwargs) -> None:
    """Set titles for multiple axes.

    Parameters
    ----------
    axes
        Axes to set titles for.
    labels
        Title strings to set. If a single string is given, it will be used for all axes.
    order
        Order in which to flatten `ax`. 'C' means to flatten in row-major (C-style)
        order. 'F' means to flatten in column-major (Fortran-style) order. The default
        is 'C'.
    **kwargs
        Extra arguments to `matplotlib.axes.Axes.set_title`: refer to the `matplotlib`
        documentation for a list of all possible arguments.

    """
    axlist = np.array(axes, dtype=object).flatten(order=order)
    if isinstance(labels, str):
        labels = [labels] * len(axlist)
    labels = np.asarray(labels)
    for ax, label in zip(axlist.flat, labels.flat, strict=True):
        ax.set_title(label, **kwargs)


def set_xlabels(axes, labels: Iterable[str] | str, order="C", **kwargs) -> None:
    """Set x-axis labels for multiple axes.

    Parameters
    ----------
    axes
        Axes to set x-axis labels for.
    labels
        Label strings to set. If a single string is given, it will be used for all axes.
    order
        Order in which to flatten `ax`. 'C' means to flatten in row-major (C-style)
        order. 'F' means to flatten in column-major (Fortran-style) order. The default
        is 'C'.
    **kwargs
        Extra arguments to `matplotlib.axes.Axes.set_xlabel`: refer to the `matplotlib`
        documentation for a list of all possible arguments.
    """
    axlist = np.array(axes, dtype=object).flatten(order=order)
    if isinstance(labels, str):
        labels = [labels] * len(axlist)
    labels = np.asarray(labels)
    for ax, label in zip(axlist.flat, labels.flat, strict=True):
        ax.set_xlabel(label, **kwargs)


def set_ylabels(axes, labels: Iterable[str] | str, order="C", **kwargs) -> None:
    """Set y-axis labels for multiple axes.

    Parameters
    ----------
    axes
        Axes to set y-axis labels for.
    labels
        Label strings to set. If a single string is given, it will be used for all axes.
    order
        Order in which to flatten `ax`. 'C' means to flatten in row-major (C-style)
        order. 'F' means to flatten in column-major (Fortran-style) order. The default
        is 'C'.
    **kwargs
        Extra arguments to `matplotlib.axes.Axes.set_ylabel`: refer to the `matplotlib`

    """
    axlist = np.array(axes, dtype=object).flatten(order=order)
    if isinstance(labels, str):
        labels = [labels] * len(axlist)
    labels = np.asarray(labels)
    for ax, label in zip(axlist.flat, labels.flat, strict=True):
        ax.set_ylabel(label, **kwargs)


def sizebar(
    ax: matplotlib.axes.Axes,
    value: float,
    unit: str,
    si: int = 0,
    resolution: float = 1.0,
    decimals: int = 0,
    label: str | None = None,
    loc: typing.Literal[
        "upper left",
        "upper center",
        "upper right",
        "center left",
        "center",
        "center right",
        "lower left",
        "lower center",
        "lower right",
    ] = "lower right",
    pad: float = 0.1,
    borderpad: float = 0.5,
    sep: float = 3.0,
    frameon: bool = False,
    **kwargs,
) -> AnchoredSizeBar:
    """Add a size bar to an axes.

    Parameters
    ----------
    ax
        The `matplotlib.axes.Axes` instance to place the size bar in.
    value
        Length of the size bar in terms of `unit`.
    unit
        An SI unit string without any prefixes. For example, 'm' for meters.
    si
        Exponents that have a corresponding SI prefix. This determines the unit of the
        label that is shown next to the size bar. For example, if the ``value`` is
        ``200e-6`` and the unit is ``"m"``, setting ``si=-6`` will result in a label of
        ``"200 μm"``, and ``si=-3`` will result in ``"0.2 mm"`` (assuming ``decimals``
        is set appropriately).
    resolution
        Scale of the current axes coordinates. For example, if the plot axes values are
        given in μm, the resolution should be 1e-6.
    decimals
        Number of decimals displayed on the size bar label.
    label
        When provided, overrides the automatically generated label string.
    loc
        Location of the size bar.
    pad
        Padding around the label and size bar, in fraction of the font size.
    borderpad
        Border padding, in fraction of the font size.
    sep
        Separation between the label and the size bar, in points.
    frameon
        If True, draw a box around the horizontal bar and label.
    **kwargs
        Keyword arguments forwarded to
        `mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`.

    Example
    -------
    >>> # Plot a size bar with a length of 200 μm on an axes given in mm
    >>> eplt.sizebar(ax, value=200e-6, unit="m", si=-6, resolution=1e-3)

    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    size = value / resolution
    unit = get_si_str(si) + unit
    value = np.around(value / 10**si, decimals=decimals)
    if int(value) == value:
        value = int(value)
    if label is None:
        label = f"{value} {unit}"

    kwargs.setdefault("color", axes_textcolor(ax))

    asb = AnchoredSizeBar(
        ax.transData,
        size,
        label,
        loc=loc,
        pad=pad,
        borderpad=borderpad,
        sep=sep,
        frameon=frameon,
        **kwargs,
    )
    ax.add_artist(asb)
    return asb
