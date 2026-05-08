import argparse
import shlex

from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

import erlab


def _normalize_manager_target_args(line: str) -> str:
    parts = shlex.split(line)
    normalized: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in {"-m", "--manager"}:
            if i + 2 < len(parts) and parts[i + 1].lstrip("+-").isdigit():
                normalized.extend([part, "--manager-index", parts[i + 1]])
                i += 2
                continue
            normalized.append(part)
            i += 1
            continue
        if part.startswith("--manager="):
            value = part.split("=", maxsplit=1)[1]
            if value.lstrip("+-").isdigit():
                normalized.extend(["--manager", "--manager-index", value])
                i += 1
                continue
        normalized.append(part)
        i += 1
    return " ".join(shlex.quote(part) for part in normalized)


@magics_class
class ImageToolMagics(Magics):
    @magic_arguments()
    @argument(
        "--link",
        "-l",
        action="store_true",
        help="Link ImageTool windows for multiple data.",
    )
    @argument(
        "--manager",
        "-m",
        action="store_true",
        default=None,
        help="Use the ImageTool manager.",
    )
    @argument(
        "--manager-index",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    @argument(
        "--centered",
        "-c",
        action="store_true",
        help="Center the colormap normalization.",
    )
    @argument(
        "--cmap",
        "-cm",
        default="magma",
        type=str,
        help="Colormap to be used. By default, the colormap is set to 'magma'.",
    )
    @argument(
        "data",
        help="Data to be displayed. Can be a DataArray, Dataset, DataTree, "
        "numpy array, or a list of these.",
    )
    @line_magic
    def itool(self, args):
        args = _normalize_manager_target_args(args)
        args = parse_argstring(self.itool, args)

        data = args.data
        if isinstance(data, str):  # pragma: no branch
            # data is a python expression
            data = f"_data_obj = {data}"
            parsed = self.shell.transform_ast(
                self.shell.compile.ast_parse(self.shell.transform_cell(data))
            )
            code = self.shell.compile(parsed, "<ipython-itool-data>", "exec")

            local_ns = {}
            exec(code, self.shell.user_ns, local_ns)  # noqa: S102
            data = local_ns.get("_data_obj")

        manager = args.manager_index if args.manager_index is not None else args.manager
        return erlab.interactive.itool(
            data=data,
            link=args.link,
            manager=manager,
            zero_centered=args.centered,
            cmap=args.cmap,
        )
