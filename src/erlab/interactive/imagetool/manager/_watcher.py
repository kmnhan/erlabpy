"""Watcher for DataArray changes.

``%watch`` magic command that tracks changes to DataArrays and update ImageTools
automatically.

"""

__all__ = ["WatcherMagics"]

import logging
import threading
import time
import typing
import uuid

import IPython
import xarray as xr
import zmq
from IPython.core.magic import Magics, line_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

import erlab

logger = logging.getLogger(__name__)


class _MessageObject:
    def __init__(self, message: str, html: str | None = None) -> None:
        if html is None:
            html = message.replace("\n", "<br>")
        self.message = message
        self.html = html

    def __repr__(self) -> str:
        return self.message

    def _repr_html_(self) -> str:
        return self.html


def _display_message(message: str, html: str | None = None) -> None:
    msg_obj = _MessageObject(message, html)
    try:
        from IPython.display import display

        display(msg_obj)
    except Exception:
        print(msg_obj.__repr__())


def _stopped_watching_message(varname: str, reason: str) -> None:
    _display_message(
        f"Stopped watching '{varname}' ({reason})",
        f"⏹️ Stopped watching <code>{varname}</code> ({reason})",
    )


class _Watcher:
    def __init__(self, shell: IPython.InteractiveShell) -> None:
        self._shell: IPython.InteractiveShell = shell

        self.watched_vars: dict[str, dict[str, str]] = {}  # varname -> state
        self._rate_limit_s: float = 0.15
        self._last_send: float = 0.0
        self._lock: threading.RLock = threading.RLock()

        # Unique ID for this watcher instance
        self._uid = str(uuid.uuid4())

        # Flags for thread receiving updates from GUI
        self._stop = threading.Event()
        self._thread_started: bool = False

    def watch(self, varname: str) -> None:
        """Watch a DataArray variable and show in ImageTool manager."""
        ns = self._shell.user_ns
        if varname not in ns:
            raise NameError(f"{varname!r} not found")
        obj = ns[varname]
        if not isinstance(obj, xr.DataArray):
            raise TypeError(f"{varname!r} is not an xarray.DataArray")

        show: bool = False

        with self._lock:
            if varname in self.watched_vars:
                # Already linked, trigger refresh and show
                uid = self.watched_vars[varname]["uid"]
                show = True
            else:
                uid = f"{varname} {self._uid}"

        fingerprint = erlab.utils.hashing.fingerprint_dataarray(obj)
        with self._lock:
            # Store fingerprint to detect changes and assign a unique ID
            self.watched_vars[varname] = {"fingerprint": fingerprint, "uid": uid}
            self._last_send = time.time()

        try:
            self._push_to_gui(varname, obj, show=show)  # Initial push
        except Exception:
            with self._lock:
                self.watched_vars.pop(varname, None)
            raise

        if not self._thread_started:
            self.start_thread()

    def stop_watching(self, varname: str, remove: bool = False) -> None:
        with self._lock:
            state = self.watched_vars.pop(varname, None)
        if state:  # pragma: no branch
            erlab.interactive.imagetool.manager.unwatch_data(
                state["uid"], remove=remove
            )

    def stop_watching_all(self, remove: bool = False) -> None:
        with self._lock:
            names = list(self.watched_vars.keys())
        for name in names:
            self.stop_watching(name, remove=remove)

    def _maybe_push(self, *__) -> None:
        if not self.watched_vars:
            return
        now = time.time()
        if now - self._last_send < self._rate_limit_s:
            return
        with self._lock:
            snapshot = list(self.watched_vars.items())

        changed: list[tuple[str, xr.DataArray, str]] = []
        for name, state in snapshot:
            obj = self._shell.user_ns.get(name, None)
            if obj is None or not isinstance(obj, xr.DataArray):
                # Variable deleted or changed type, stop watching
                self.stop_watching(name)
                _stopped_watching_message(name, "variable deleted or changed type")
                continue
            new_fp = erlab.utils.hashing.fingerprint_dataarray(obj)
            if new_fp != state["fingerprint"]:  # pragma: no branch
                changed.append((name, obj, new_fp))

        if len(changed) == 0:
            return

        with self._lock:
            for name, _, new_fp in changed:
                if name in self.watched_vars:  # pragma: no branch
                    self.watched_vars[name]["fingerprint"] = new_fp
            self._last_send = now

        for name, obj, _ in changed:
            self._push_to_gui(name, obj)

    def _push_to_gui(
        self, varname: str, darr: xr.DataArray, show: bool = False
    ) -> None:
        with self._lock:
            state = self.watched_vars.get(varname)
            if not state:
                return
            uid = state["uid"]
        erlab.interactive.imagetool.manager.watch_data(varname, uid, darr, show=show)

    def start_thread(self) -> None:
        self._thread_started = True
        self._watcher_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._watcher_thread.start()

    def _recv_loop(self) -> None:
        self._stop.clear()
        context = zmq.Context.instance()
        sock: zmq.Socket = context.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        try:
            logger.debug("Starting watcher recv loop...")
            sock.connect(
                f"tcp://{erlab.interactive.imagetool.manager.HOST_IP}:"
                f"{erlab.interactive.imagetool.manager.PORT_WATCH}"
            )
            logger.debug("Watcher connected to server.")

            while not self._stop.is_set():
                logger.debug("Watcher waiting for messages...")
                info = typing.cast("dict[str, str]", sock.recv_json())
                logger.debug("Watcher received message: %s", info)
                varname, uid, event = info["varname"], info["uid"], info["event"]

                if not uid.endswith(self._uid):
                    continue  # Not for this watcher instance

                with self._lock:
                    if varname not in self.watched_vars:
                        continue  # Not watching
                match event:
                    case "updated":
                        # Schedule update on kernel thread
                        if hasattr(self._shell, "kernel") and hasattr(
                            self._shell.kernel, "io_loop"
                        ):  # pragma: no branch
                            logger.debug("Scheduling update on kernel io_loop")
                            self._shell.kernel.io_loop.add_callback(
                                self._apply_update_now, varname, uid
                            )
                        else:
                            logger.debug("Applying update directly")
                            self._apply_update_now(varname, uid)
                    case "removed":  # pragma: no branch
                        with self._lock:
                            self.watched_vars.pop(varname, None)
                        _stopped_watching_message(varname, "removed from ImageTool")
        except Exception:
            logger.exception("Error in watcher recv loop")
        finally:
            logger.debug("Watcher recv loop exiting")
            sock.close()

    def _apply_update_now(self, varname: str, uid: str) -> None:
        darr = erlab.interactive.imagetool.manager.fetch(uid)
        if darr is None:  # Data not found, ignore
            return

        new_hash: str = erlab.utils.hashing.fingerprint_dataarray(darr)

        with self._lock:
            state = self.watched_vars.get(varname)
            if not state or state["uid"] != uid:
                # Not watching anymore or from different source, ignore
                return
            state["fingerprint"] = new_hash

        # Update variable in user namespace
        self._shell.user_ns[varname] = darr

        # Show message in output
        _display_message(
            f"Updated '{varname}' from ImageTool",
            f"↩️ Updated <code>{varname}</code> from ImageTool",
        )

    def shutdown(self) -> None:
        self._stop.set()
        if hasattr(self, "_watcher_thread") and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=0.5)
            self._thread_started = False

    def __del__(self):
        self.shutdown()


@magics_class
class WatcherMagics(Magics):
    def __init__(self, shell: IPython.InteractiveShell) -> None:
        # You must call the parent constructor
        super().__init__(shell)
        self._watcher = _Watcher(shell)

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

        if not args.darr and not args.z:
            if not self._watcher.watched_vars:
                _display_message("No variables are being watched.")
                return
            _display_message(
                "Currently watched variables:\n"
                + "\n".join(
                    [f" - {varname}" for varname in self._watcher.watched_vars]
                ),
                "Currently watched variables:\n"
                + " ".join(
                    [
                        f"<code>{varname}</code>"
                        for varname in self._watcher.watched_vars
                    ]
                ),
            )

            return

        if args.z:
            self._watcher.stop_watching_all(remove=args.x)
            _display_message("Stopped watching all variables.")
            return

        for var in args.darr:
            if args.d or args.x:
                self._watcher.stop_watching(var, remove=args.x)
            else:
                self._watcher.watch(var)

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
