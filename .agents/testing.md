# Agent instructions for tests

When a new or expanded test fails, inspect the runtime behavior before changing the
expected result. Change the expectation only when it is demonstrably incorrect.
Tests and stubs must follow the current runtime contract. Update outdated mocks instead
of adding production fallbacks, obsolete private shims, or pytest-dependent behavior.

Assert behavior or stable metadata such as action data, roles, shortcuts, and enabled
state. Avoid assertions on user-facing labels. Use multidimensional cases where they
can expose a regression. Coverage exclusions need a brief reason why the path is
impractical to test; prefer direct tests when feasible.

Manager tests can interpret `sys.argv[1:]` as file paths. Prefer explicit test node IDs
or patch `sys.argv` rather than passing long `-k` expressions. `manager_context` uses
isolated registries and ephemeral ports; the normal Manager port is `45555`. For port
conflicts or timeouts, check for stale Manager test processes and stop them before
retrying. Run parallel Manager jobs only with isolated ports.

If module-specific `--cov=<module>` causes Qt import problems, use `--cov=erlab` and
filter the resulting report. Plain `uv run pytest` does not collect coverage.
