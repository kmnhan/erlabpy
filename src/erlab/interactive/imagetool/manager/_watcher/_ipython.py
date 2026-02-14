"""IPython-specific watcher integration (magics and shell hooks)."""

from __future__ import annotations

import contextlib
import typing

from erlab.interactive.imagetool.manager._watcher._core import (
    _display_message,
    _get_or_create_watcher,
    _register_post_run_cell_callback,
    _ShellProtocol,
    _Watcher,
    watch,
    watched_variables,
)

if typing.TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

try:
    import IPython
    from IPython.core.magic import Magics, line_magic, magics_class
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
except ImportError as e:
    raise ImportError(
        "The `IPython` package is required for IPython integration"
    ) from e


def _safe_get_ipython_shell() -> _ShellProtocol | None:
    with contextlib.suppress(Exception):
        shell = IPython.get_ipython()
        if shell is not None and hasattr(shell, "user_ns"):
            return shell
    return None


def enable_ipython_auto_push(shell: _ShellProtocol | None = None) -> _Watcher:
    """Enable post-cell synchronization for an IPython shell."""
    shell_obj = shell or _safe_get_ipython_shell()
    if shell_obj is None:
        raise RuntimeError("No active IPython shell found")

    watcher, key = _get_or_create_watcher(shell=shell_obj)
    _register_post_run_cell_callback(shell_obj, watcher, key)
    return watcher


@magics_class
class WatcherMagics(Magics):
    def __init__(self, shell) -> None:
        # You must call the parent constructor
        super().__init__(shell)
        self._watcher, _ = _get_or_create_watcher(shell=self._typed_shell)

    @property
    def _typed_shell(self) -> InteractiveShell:
        shell = self.shell
        if shell is None:
            raise RuntimeError("IPython shell is not available")
        return shell

    @magic_arguments()
    @argument(
        "-d", action="store_true", help="Stop watching the specified variable(s)."
    )
    @argument("-x", action="store_true", help="Remove from manager.")
    @argument("-z", action="store_true", help="Stop watching all variables.")
    @argument("darr", nargs="*", help="DataArray variable(s) to be watched.")
    @line_magic
    def watch(self, line) -> None:
        """Watch DataArray variable(s) and show in ImageTool manager.

        This magic command allows you to watch one or more xarray DataArray variables in
        your IPython environment. When a watched variable is modified, the changes are
        applied automatically to the data shown in the ImageTool manager.

        Usage:

        * ``%watch``          - Show list of all watched variables

        * ``%watch spam bar`` - Open the DataArray variables spam and bar and keep
                                watching for changes

        * ``%watch -d spam``  - Stop watching the variable spam

        * ``%watch -x spam``  - Completely remove the variable spam from the manager

        * ``%watch -z``       - Stop watching all variables

        """
        args = parse_argstring(self.watch, line)
        shell = self._typed_shell

        if not args.darr and not args.z:
            watched = watched_variables(shell=shell)
            if len(watched) == 0:
                _display_message("No variables are being watched.")
                return
            _display_message(
                "Currently watched variables:\n"
                + "\n".join([f" - {v}" for v in watched]),
                "Currently watched variables:\n"
                + " ".join([f"<code>{varname}</code>" for varname in watched]),
            )

            return

        if args.z:
            watch(shell=shell, stop_all=True, remove=args.x)
            _display_message("Stopped watching all variables.")
            return

        watch(
            *args.darr,
            shell=shell,
            stop=args.d or args.x,
            remove=args.x,
        )

        if args.d or args.x:
            _display_message(
                f"Stopped watching {', '.join(args.darr)}",
                "⏹️ Stopped watching "
                + " ".join([f"<code>{varname}</code>" for varname in args.darr]),
            )
        else:
            _display_message(
                f"Watching {', '.join(args.darr)}",
                "🔄 Watching "
                + " ".join([f"<code>{varname}</code>" for varname in args.darr]),
            )

        return
