from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

import erlab


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
        args = parse_argstring(self.itool, args)

        data = args.data
        if isinstance(data, str):
            # data is a python expression
            data = f"_data_obj = {data}"
            parsed = self.shell.transform_ast(
                self.shell.compile.ast_parse(self.shell.transform_cell(data))
            )
            code = self.shell.compile(parsed, "<ipython-itool-data>", "exec")

            local_ns = {}
            exec(code, self.shell.user_ns, local_ns)  # noqa: S102
            data = local_ns.get("_data_obj")

        return erlab.interactive.itool(
            data=data,
            link=args.link,
            manager=args.manager,
            zero_centered=args.centered,
            cmap=args.cmap,
        )
