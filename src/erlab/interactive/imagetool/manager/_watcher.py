"""Watch DataArray changes and synchronize them with ImageTool manager."""

from __future__ import annotations

__all__ = [
    "WatcherMagics",
    "enable_ipython_auto_push",
    "maybe_push",
    "shutdown",
    "watch",
    "watched_variables",
]

import contextlib
import inspect
import logging
import threading
import time
import typing
import uuid
from collections.abc import MutableMapping

import xarray as xr
import zmq

import erlab

try:
    import IPython
    from IPython.core.magic import Magics, line_magic, magics_class
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
except Exception:  # pragma: no cover - fallback for non-IPython environments
    IPython = None

    class Magics:
        def __init__(self, shell) -> None:
            self.shell = shell

    def magics_class(cls):
        return cls

    def line_magic(func):
        return func

    def magic_arguments(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator

    def argument(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator

    def parse_argstring(*args, **kwargs):
        raise RuntimeError("IPython is required for %watch magic support")


logger = logging.getLogger(__name__)

NamespaceType = MutableMapping[str, typing.Any]


class _ShellProtocol(typing.Protocol):
    user_ns: NamespaceType


class _NamespaceShell:
    def __init__(self, user_ns: NamespaceType) -> None:
        self.user_ns = user_ns


_STATE_LOCK: threading.RLock = threading.RLock()
_WATCHERS: dict[int, _Watcher] = {}
_POST_RUN_CELL_CALLBACKS: dict[int, tuple[typing.Any, typing.Callable[..., None]]] = {}


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
    def __init__(self, shell: _ShellProtocol) -> None:
        self._shell: _ShellProtocol = shell

        self.watched_vars: dict[str, dict[str, str]] = {}  # varname -> state
        self._rate_limit_s: float = 0.15
        self._last_send: float = 0.0
        self._lock: threading.RLock = threading.RLock()

        # Unique ID for this watcher instance
        self._uid = str(uuid.uuid4())

        # Flags for thread receiving updates from GUI
        self._stop = threading.Event()
        self._thread_started: bool = False
        self._watcher_thread: threading.Thread | None = None

        # Polling thread for environments without post_run_cell hook
        self._poll_stop = threading.Event()
        self._poll_interval_s: float = 0.25
        self._poll_thread: threading.Thread | None = None

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
            erlab.interactive.imagetool.manager._unwatch_data(
                state["uid"], remove=remove
            )

    def stop_watching_all(self, remove: bool = False) -> None:
        with self._lock:
            names = list(self.watched_vars.keys())
        for name in names:
            self.stop_watching(name, remove=remove)

    def _maybe_push(self, *__) -> None:
        with self._lock:
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
        erlab.interactive.imagetool.manager._watch_data(varname, uid, darr, show=show)

    def start_thread(self) -> None:
        with self._lock:
            if self._watcher_thread is not None and self._watcher_thread.is_alive():
                self._thread_started = True
                return

            self._thread_started = True
            self._watcher_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._watcher_thread.start()

    def start_polling(self, interval_s: float = 0.25) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self._poll_interval_s = interval_s
        with self._lock:
            if self._poll_thread is not None and self._poll_thread.is_alive():
                return
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def _poll_loop(self) -> None:
        while not self._poll_stop.wait(self._poll_interval_s):
            try:
                self._maybe_push()
            except Exception:  # pragma: no cover - background thread safeguard
                logger.exception("Error in watcher poll loop")

    def _recv_loop(self) -> None:
        self._stop.clear()
        context = zmq.Context.instance()
        sock: zmq.Socket = context.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 100)
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
                try:
                    info = typing.cast("dict[str, str]", sock.recv_json())
                except zmq.Again:
                    continue
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
        self._poll_stop.set()

        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=2.0)
            if self._watcher_thread.is_alive():
                logger.warning("Watcher thread did not stop within timeout")
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
            if self._poll_thread.is_alive():
                logger.warning("Watcher poll thread did not stop within timeout")
        self._thread_started = False

    def __del__(self):
        self.shutdown()


def _infer_caller_namespace() -> NamespaceType:
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame and frame.f_back else None
        if caller is None:
            return typing.cast("NamespaceType", globals())
        return typing.cast("NamespaceType", caller.f_globals)
    finally:
        del frame


def _safe_get_ipython_shell() -> _ShellProtocol | None:
    if IPython is None:
        return None

    with contextlib.suppress(Exception):
        shell = IPython.get_ipython()
        if shell is not None and hasattr(shell, "user_ns"):
            return typing.cast("_ShellProtocol", shell)
    return None


def _resolve_target(
    shell: _ShellProtocol | None = None,
    namespace: NamespaceType | None = None,
) -> tuple[_ShellProtocol | None, NamespaceType]:
    if shell is not None:
        return shell, shell.user_ns
    if namespace is not None:
        return None, namespace
    raise ValueError("Either shell or namespace must be provided")


def _get_or_create_watcher(
    shell: _ShellProtocol | None = None,
    namespace: NamespaceType | None = None,
) -> tuple[_Watcher, int]:
    shell_obj, namespace_obj = _resolve_target(shell=shell, namespace=namespace)
    key = id(namespace_obj)

    with _STATE_LOCK:
        watcher = _WATCHERS.get(key)
        if watcher is None:
            watcher_shell = (
                shell_obj if shell_obj is not None else _NamespaceShell(namespace_obj)
            )
            watcher = _Watcher(watcher_shell)
            _WATCHERS[key] = watcher
        elif shell_obj is not None:
            watcher._shell = shell_obj
    return watcher, key


def _get_watcher_if_exists(
    shell: _ShellProtocol | None = None,
    namespace: NamespaceType | None = None,
) -> tuple[_Watcher | None, int]:
    _, namespace_obj = _resolve_target(shell=shell, namespace=namespace)
    key = id(namespace_obj)
    with _STATE_LOCK:
        return _WATCHERS.get(key), key


def _register_post_run_cell_callback(
    shell: _ShellProtocol, watcher: _Watcher, key: int
) -> bool:
    events = getattr(shell, "events", None)
    if events is None or not hasattr(events, "register"):
        return False

    with _STATE_LOCK:
        if key in _POST_RUN_CELL_CALLBACKS:
            return True
        callback = watcher._maybe_push
        _POST_RUN_CELL_CALLBACKS[key] = (shell, callback)

    try:
        events.register("post_run_cell", callback)
    except Exception:
        with _STATE_LOCK:
            _POST_RUN_CELL_CALLBACKS.pop(key, None)
        logger.exception("Failed to register post_run_cell callback")
        return False
    else:
        return True


def _unregister_post_run_cell_callback(key: int) -> None:
    with _STATE_LOCK:
        callback_state = _POST_RUN_CELL_CALLBACKS.pop(key, None)

    if callback_state is None:
        return

    shell, callback = callback_state
    events = getattr(shell, "events", None)
    if events is not None and hasattr(events, "unregister"):
        with contextlib.suppress(Exception):
            events.unregister("post_run_cell", callback)


def enable_ipython_auto_push(shell: _ShellProtocol | None = None) -> _Watcher:
    """Enable post-cell synchronization for an IPython shell."""
    shell_obj = shell or _safe_get_ipython_shell()
    if shell_obj is None:
        raise RuntimeError("No active IPython shell found")

    watcher, key = _get_or_create_watcher(shell=shell_obj)
    _register_post_run_cell_callback(shell_obj, watcher, key)
    return watcher


def watched_variables(
    *,
    shell: _ShellProtocol | None = None,
    namespace: NamespaceType | None = None,
) -> tuple[str, ...]:
    """Return currently watched variable names for the selected namespace."""
    if shell is None and namespace is None:
        ip_shell = _safe_get_ipython_shell()
        if ip_shell is not None:
            shell = ip_shell
        else:
            namespace = _infer_caller_namespace()

    watcher, _ = _get_watcher_if_exists(shell=shell, namespace=namespace)
    if watcher is None:
        return ()
    with watcher._lock:
        return tuple(watcher.watched_vars.keys())


def watch(
    *varnames: str,
    shell: _ShellProtocol | None = None,
    namespace: NamespaceType | None = None,
    stop: bool = False,
    remove: bool = False,
    stop_all: bool = False,
    poll_interval_s: float = 0.25,
) -> tuple[str, ...]:
    """Watch namespace variables and synchronize them with ImageTool manager.

    Parameters
    ----------
    *varnames
        Variable names to watch or stop watching.
    shell
        Shell-like object exposing ``user_ns``. If omitted, the active IPython shell
        is used when available.
    namespace
        Namespace mapping used when no shell is available (e.g. ``globals()`` in
        marimo notebooks).
    stop
        If ``True``, stop watching specified variables.
    remove
        If ``True``, remove watched variables from manager while stopping.
    stop_all
        If ``True``, stop watching all variables in the namespace.
    poll_interval_s
        Polling interval used when post-run hooks are unavailable.

    Returns
    -------
    tuple of str
        Currently watched variable names after applying the operation.
    """
    if shell is None and namespace is None:
        ip_shell = _safe_get_ipython_shell()
        if ip_shell is not None:
            shell = ip_shell
        else:
            namespace = _infer_caller_namespace()

    if not varnames and not stop_all and not stop and not remove:
        return watched_variables(shell=shell, namespace=namespace)

    watcher, key = _get_or_create_watcher(shell=shell, namespace=namespace)
    shell_obj = watcher._shell

    if stop_all:
        watcher.stop_watching_all(remove=remove)
    elif stop or remove:
        for varname in varnames:
            watcher.stop_watching(varname, remove=remove)
    else:
        for varname in varnames:
            watcher.watch(varname)

        if not _register_post_run_cell_callback(shell_obj, watcher, key):
            watcher.start_polling(poll_interval_s)

    return watched_variables(shell=shell, namespace=namespace)


def maybe_push(
    *,
    shell: _ShellProtocol | None = None,
    namespace: NamespaceType | None = None,
) -> None:
    """Push changed variables to manager for a given shell or namespace."""
    if shell is None and namespace is None:
        ip_shell = _safe_get_ipython_shell()
        if ip_shell is not None:
            shell = ip_shell
        else:
            namespace = _infer_caller_namespace()

    watcher, _ = _get_watcher_if_exists(shell=shell, namespace=namespace)
    if watcher is not None:
        watcher._maybe_push()


def shutdown(
    *,
    shell: _ShellProtocol | None = None,
    namespace: NamespaceType | None = None,
    remove: bool = False,
) -> None:
    """Shutdown watcher for a namespace and unregister associated hooks."""
    if shell is None and namespace is None:
        ip_shell = _safe_get_ipython_shell()
        if ip_shell is not None:
            shell = ip_shell
        else:
            namespace = _infer_caller_namespace()

    _, namespace_obj = _resolve_target(shell=shell, namespace=namespace)
    key = id(namespace_obj)

    with _STATE_LOCK:
        watcher = _WATCHERS.pop(key, None)

    _unregister_post_run_cell_callback(key)

    if watcher is None:
        return

    watcher.stop_watching_all(remove=remove)
    watcher.shutdown()


@magics_class
class WatcherMagics(Magics):
    def __init__(self, shell) -> None:
        # You must call the parent constructor
        super().__init__(shell)
        self._watcher, _ = _get_or_create_watcher(shell=self.shell)

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
            watched = watched_variables(shell=self.shell)
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
            watch(shell=self.shell, stop_all=True, remove=args.x)
            _display_message("Stopped watching all variables.")
            return

        watch(
            *args.darr,
            shell=self.shell,
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
