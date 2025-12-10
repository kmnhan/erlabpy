from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

import erlab


@magics_class
class InteractiveToolMagics(Magics):
    def parse_expression(self, data_str):
        data = f"_data_obj = {data_str}"
        parsed = self.shell.transform_ast(
            self.shell.compile.ast_parse(self.shell.transform_cell(data))
        )
        code = self.shell.compile(parsed, "<ipython-erlab-interactive-data>", "exec")

        local_ns = {}
        exec(code, self.shell.user_ns, local_ns)  # noqa: S102
        return local_ns.get("_data_obj")

    @magic_arguments()
    @argument(
        "--cmap",
        "-cm",
        default="magma",
        type=str,
        help="Colormap to be used. By default, the colormap is set to 'magma'.",
    )
    @argument("data", help="Data to convert to k-space.")
    @line_magic
    def ktool(self, args):
        args = parse_argstring(self.ktool, args)
        data = args.data
        return erlab.interactive.ktool(
            data=self.parse_expression(data), cmap=args.cmap, data_name=data
        )

    @magic_arguments()
    @argument("data", help="Data to visualize.")
    @line_magic
    def dtool(self, args):
        args = parse_argstring(self.dtool, args)
        data = args.data
        return erlab.interactive.dtool(data=self.parse_expression(data), data_name=data)

    @magic_arguments()
    @argument("data", help="Data to extract the Fermi edge from.")
    @line_magic
    def goldtool(self, args):
        args = parse_argstring(self.goldtool, args)
        data = args.data
        return erlab.interactive.goldtool(
            data=self.parse_expression(data), data_name=data
        )

    @magic_arguments()
    @argument("data", help="Data to fit.")
    @line_magic
    def restool(self, args):
        args = parse_argstring(self.restool, args)
        data = args.data
        return erlab.interactive.restool(
            data=self.parse_expression(data), data_name=data
        )

    @magic_arguments()
    @argument("data", help="Data containing the mesh to be removed.")
    @line_magic
    def meshtool(self, args):
        args = parse_argstring(self.meshtool, args)
        data = args.data
        return erlab.interactive.meshtool(
            data=self.parse_expression(data), data_name=data
        )
