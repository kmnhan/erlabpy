# Agent instructions for Qt changes

Qt-facing code and tests must behave consistently under PyQt6 and PySide6. Import
bindings through `qtpy` and use explicit Qt6 enums. Follow the existing
[Qt import and UI conventions](../docs/source/contributing/development.md#code-standards).
Validate changed lifetime or binding-sensitive paths under both bindings.

Avoid binding-specific signal or metaobject assumptions. Isolate unavoidable low-level
differences in a compatibility helper.

- Guard deferred or queued Qt access with `qt_is_valid`.
- Keep wrappers such as `QMenu` and `QAction` strongly referenced while they are used.
- In `destroyed` handlers, check object identity rather than a reusable ID.
- Disconnect exactly the signals that were connected.
- When consolidating behavior in a base class, remove subclass overrides that only
  delegate to `super()` and check nearby overrides for the same case.

Use a trailing Unicode ellipsis (`…`) on an action only when it requires more input,
selection, or confirmation before completion. Immediate commands, toggles, submenus,
informational windows, and commands that only sometimes prompt have no ellipsis.

Actions that open ImageTool from a managed tool must use Manager-aware launch paths.
If a path already displays a dialog, suppress a duplicate Manager alert by logging with
`extra={"suppress_ui_alert": True}`. Watcher changes must preserve IPython post-run-cell
behavior and cover the polling fallback used by plain namespaces or marimo.

## Tests

Directly cover changed interactive lines, including warnings and early returns, unless
the branch is impractical to test. Prefer `accept_dialog` for normal dialog interactions
and stubs for hard-to-reach branches. Context-menu and file-dialog features need both
accepted and cancelled paths. For test contracts and Manager diagnostics, read
[testing.md](testing.md).

For Manager integration branches, patch
`erlab.interactive.imagetool.manager.is_running` and `show_in_manager`. Use
`manager_context` when the behavior requires a live Manager.
