# Repository Guidelines

## Project layout

Place runtime code in `src/erlab/`: analysis routines, interactive Qt tools, IO plugins,
and visualization helpers. Mirror that tree under `tests/`, with plugin fixtures in
`tests/io/plugins/`. Keep build outputs (`build/`, `dist/`), the PyInstaller recipe
(`manager.spec`), shared assets (`resources/`, `PythonInterface.ipf`), and documentation
(`docs/`) separate from runtime code so releases remain lean.

## Commands and required checks

Run these commands from the repository root. The command list describes each entry
point; the requirements below specify when checks must run.

- `uv sync --all-extras --dev --group pyqt6` — install the editable project, all package
  extras, development dependencies, and PyQt6 for GUI tests.
- `uv run pytest` — run the whole suite. Coverage requires an explicit option, such as
  `uv run pytest --cov=erlab`; its configuration is in `pyproject.toml`.
- `uv run python -m scripts.ci_test_groups --check-partition` — verify that the fast CI
  coverage shards still cover every test file exactly once.
- `uv run python -m scripts.ci_test_groups compat` — print the compatibility smoke
  targets used in non-primary CI lanes.
- `uv run mypy src` — static typing pass.
- `uv run pyinstaller manager.spec` — bundle the ImageTool Manager app. This requires
  the separate `pyinstaller` dependency group. Before packaging, read
  [.github/workflows/build.yml](.github/workflows/build.yml) for platform setup and the
  CI build sequence: `uv sync --no-editable --all-extras --group pyqt6 --group pyinstaller`,
  then `uv run --no-editable pyinstaller manager.spec`.
- `uv build` — produce wheels and sdists for release.

Required checks:

- After every change, run `uv run ruff format .` and `uv run ruff check --fix .`.
  Prefer automatic fixes over manual lint cleanup.
- After changing the test tree or CI grouping rules, run
  `uv run python -m scripts.ci_test_groups --check-partition`.
- When changing CI grouping or adding new top-level test modules, run all of:
  `uv run ruff check`, `uv run ruff format --check`, `uv run mypy src`, `uv run pytest`,
  and `uv run python -m scripts.ci_test_groups --check-partition`.

Documentation and Qt changes also have checks in their sections below.

## Writing style

Use ASD-STE100 Simplified Technical English for agent responses, commit messages,
documentation, and pull request titles and descriptions. Use short and direct
sentences. Put one idea or instruction in each sentence. Use precise words and
consistent terms. Do not use idioms, unnecessary jargon, or ambiguous wording.
Keep a necessary technical term when an approved alternative is not accurate.
Define the term when the reader might not know it.
Use concise noun phrases or gerund phrases for headings and titles. Do not use complete
sentences as headings. Prefer established scientific and ARPES terminology.

## Documentation

Sources live in `docs/source/` and use MyST and Sphinx. Install dependencies with
`uv sync --all-extras --dev --group docs`, then build with
`uv run --directory docs make html`.

| Content | Directory |
| --- | --- |
| Tutorials | `docs/source/tutorials/` |
| Task guides | `docs/source/how-to/` |
| Explanations | `docs/source/explanation/` |
| Product descriptions | `docs/source/reference/` |
| Images | `docs/source/images/` |

Before pushing large documentation edits, run `uv run --directory docs make linkcheck`
to check links. When moving or renaming pages, add redirects to `rediraffe_redirects`
in `docs/source/conf.py`. The site uses `sphinxext-rediraffe` for redirects.

### Documentation types

Use the Diátaxis compass before adding a How-to guide. Confirm that all four statements
are true:

- The guide informs action during the application of an existing skill.
- The guide addresses a concrete user goal or problem, not the operation of a tool.
- A competent user can apply the guide to their own work without recreating
  tutorial-specific state.
- The guide gives a focused, executable sequence with only the decisions, checks, and
  recovery guidance required by the task.

When a capability taught in a tutorial is also a valid work task, illustrate it in a
How-to guide. Remove the tutorial's teaching and controlled demonstration, but do not
remove a guide only because its code also appears in a tutorial. Link to Explanation and
Reference instead of adding digressions or exhaustive option lists.

Teach required xarray concepts and ERLabPy data conventions in the controlled tutorial
path. Explanation pages can assume that knowledge. Use Explanation for ERLabPy design
choices and scientific workflow decisions that users must understand. Do not make each
Explanation page a self-contained introduction. Prefer tables, lists, diagrams, and
focused figures to long prose. Put figures that demonstrate a procedure in the relevant
How-to guide.

### API notes and consistency

When changing public API behavior, add a `.. versionchanged::` note in the relevant
docstring or documentation page so it appears in the generated docs. The version
should refer to the next release, likely a minor bump under semver. Verify the planned
version before finalizing the note.
Use `.. versionadded::` sparingly, for user-facing features or APIs where running the
same code on older versions would be confusing or yield ambiguous errors.

When changing documentation content or URLs, read
[skills/arpes-analysis/SKILL.md](skills/arpes-analysis/SKILL.md), check that its guidance
and documentation links still match, and update it if needed.
Write in concrete user-facing terms. Prefer naming the visible object or action over
abstract implementation phrasing. Avoid vague category labels and compressed prose
that does not describe what the user will see.

## Code style and imports

- Use 4-space indentation, Ruff's 88-character limit, and double quotes.
  Use snake_case for modules and functions, and CapWords for classes.
- Use NumPy-style docstrings. PEP 484 type hints are recommended for all public APIs.
- Install `prek` and run `uv run prek install` to enable the configured Git hooks.
  `.pre-commit-config.yaml` includes Ruff and Commitizen hooks, but no mypy hook.
  Mypy checks use the separate command and conditions listed above.
- Prefer top-level `erlab` imports in modules that already use `lazy_loader`, even if
  a narrower import is possible. Prefer absolute imports over relative imports.
- Keep small implementation paths direct. Do not add dataclasses, wrapper helpers, or
  renamed import aliases for simple values or one-off calls unless they remove real
  complexity, define a meaningful boundary, or match an established local pattern.
- Use modern typing syntax by default. Avoid deprecated `typing` aliases.
  Use built-in containers such as `list[str]`
  and `dict[str, int]`, and import abstract collection types such as `Callable` from
  `collections.abc`. Put those imports inside `if typing.TYPE_CHECKING:` when they are
  used only for annotations. Do not use the aliases deprecated since Python 3.9:
  `typing.Iterable`, `typing.Iterator`, `typing.Mapping`, `typing.Sequence`,
  `typing.Callable`, `typing.Dict`, `typing.List`, or `typing.Tuple`.
- Do not use `assert` in runtime code under `src/`. Use explicit `if`/`raise` checks
  for runtime invariants. Use `typing.cast` for type narrowing that should not affect
  runtime behavior.
- Avoid injecting symbols into library module scope through `globals()` mutation,
  such as `globals().update`. Prefer explicit imports and direct references.

## Tests and CI

Name test files `test_<feature>.py` in the matching directory under `tests/`.
Add loader regression tests in `tests/io/plugins/test_<plugin>.py` with regression data.
Set `ERLAB_TEST_DATA_DIR` to a local clone of `erlabpy-data` so fixtures resolve.
Pytest uses strict markers and `strict_xfail = true` in `pyproject.toml`.
Coverage skips legacy updater code. Aim for branch coverage elsewhere and parametrize
datasets to catch multidimensional regressions.

### Test behavior

- If newly added or expanded tests fail after an initial implementation, re-examine the
  runtime code before assuming the tests are wrong. Do not modify tests only to make
  them pass unless you can clearly justify that the asserted behavior is incorrect;
  otherwise you may mask a real defect in the implementation.
- Tests and monkeypatched stubs must implement the current runtime contract. Do not add
  production fallbacks or compatibility branches only to accommodate outdated fake
  objects in tests.
- Do not assert user-facing label text, including Qt action labels. Prefer stable
  metadata, object names, action data, roles, shortcuts, enabled state, or behavior.
- For generated or copied code tests, follow [Generated code and
  provenance](#generated-code-and-provenance).

### CI grouping

The fast PR workflow runs one fully covered, sharded `3.13 + pyqt6` lane plus smaller
compatibility smoke jobs. The weekly compatibility workflow keeps the full upgraded
`3.11-3.14 x pyqt6/pyside6` matrix.

Test grouping is centralized in `scripts/_ci_test_groups.py`. When adding a new
top-level test module under `tests/analysis/`, `tests/interactive/`, `tests/io/`, or
`tests/`, update that file. Assign the new test to exactly one coverage shard and, if
appropriate, to the compatibility smoke set. Run the checks in
[Commands and required checks](#commands-and-required-checks).

`tests/conftest.py` assigns the `compat`, `gui`, and `serial` markers during collection
from those grouping rules. Keep the markers semantically meaningful. Do not scatter
ad hoc CI-only marker assignments across unrelated test files.

## Generated code and provenance

User-facing generated, copied, and replay code must be as clean and direct as possible.
Treat readability as product behavior. Copied code also teaches users who move between
scripts and the GUI. Prefer public APIs, semantic variable names, complete expressions,
and best-practice examples.

Avoid meaningless relay assignments, repeated scratch variables, reused generic names
such as `derived` across unrelated subexpressions, leaked internal manager helpers, and
temporaries with no semantic value. Prefer names from watched variables, console
assignments, provenance inputs, or visible actions. Inline one-use aliases when safe.

Copied code is notebook-facing representative code, not an exact serialized replay
program. Use direct public expressions. Omit routine framework imports for one
provenance step. Import each required module at most once in a composed workflow.
Do not add defensive `.copy()` calls, private restoration helpers, or other internal
machinery only to preserve replay isolation. If a normal public operation updates input
metadata, copied code should show that update directly. Keep exact source isolation and
non-mutating behavior in the separate structured replay and execution paths.

When changing provenance or code-generation paths, scan the emitted code for redundant
reassignments and internal implementation details. Add property-style tests for
cleanliness when it is the behavior under test.

When testing generated or copied code, execute it with `exec()` in an explicit
namespace and assert that the resulting object or value exactly matches the expected
result. Do not compare copied code strings or formatting unless the exact text is the
behavior under test. Cleanliness checks must still execute the generated code and
assert the result exactly.

## Interactive Qt code

### Bindings and lifetime

Import bindings through `qtpy` and use explicit enums such as
`QtCore.Qt.CheckState.Checked`. For Qt imports, prefer
`from qtpy import QtWidgets, QtCore, QtGui`. Some widgets have co-located `.ui` files.

- All Qt-facing runtime code and tests must behave the same under both PyQt6 and
  PySide6. Prefer `qtpy` APIs that are binding-neutral, avoid binding-specific
  signal/metaobject assumptions, and when lower-level Qt behavior is unavoidable, add a
  compatibility helper and validate the touched path under both bindings.
- Write Qt code with explicit lifetime ownership across bindings from the start. Guard
  queued or deferred Qt access with `qt_is_valid`, keep needed `QMenu`/`QAction` and
  other Qt wrappers strongly referenced for as long as they are queried, make
  `destroyed` handlers verify object identity instead of acting only by reusable IDs,
  and disconnect exactly the signals that were connected. Validate touched
  lifetime-sensitive paths under both PySide6 and PyQt6.
- If a subclass override only delegates to `super()`, remove it. Audit nearby overrides
  when consolidating logic into a base class.

### Actions and Manager integration

- For Qt action and menu labels, use a trailing Unicode ellipsis (`…`) only when the
  command requires additional user input, a required selection, or required confirmation
  before it can complete. Do not use an ellipsis for immediate commands, toggles,
  submenus, informational windows/panels, or commands that only sometimes prompt. Use
  `…`, not ASCII `...`, for user-facing Qt action and menu labels.
- For tools launched from ImageTool, new actions that open/show data in ImageTool should
  be manager-aware (use manager flow when the parent tool is managed).
- If a code path already shows an explicit UI dialog (`MessageDialog`/`QMessageBox`),
  avoid duplicate manager alert popups by logging with
  `extra={"suppress_ui_alert": True}`.
- Keep watcher semantics stable for IPython users when adding non-IPython support.
  Validate both post-run-cell (IPython) and polling fallback (for example, marimo or a
  plain namespace) paths in tests.

### Qt tests and diagnostics

- For new context-menu or file-dialog features, add tests for both accept and cancel
  dialog paths.
- Prefer `accept_dialog` for real dialog interactions; use monkeypatch stubs to target
  hard-to-hit branches.
- New or modified lines in touched interactive modules should be directly covered by
  tests, including warning/early-return branches, unless there is a clear reason a
  branch is untestable.
- `erlab.interactive.imagetool.manager.main()` inspects `sys.argv[1:]` as potential
  file paths. For manager tests, prefer explicit test node IDs over long `-k`
  expressions, or patch `sys.argv` in tests, to avoid accidental file-path parsing.
- The default Manager request port is `45555`. The `manager_context` fixture uses
  ephemeral ports and an isolated registry. If tests fail with `Address already in
  use` or timeout, check and terminate stale manager processes before rerunning.
- Avoid running multiple manager test jobs in parallel on the same machine unless ports
  are isolated.
- If a test only needs to cover manager integration branches, prefer patching
  `erlab.interactive.imagetool.manager.is_running` and `show_in_manager` over launching
  the real manager. Reserve `manager_context` for behavior that genuinely depends on a
  live manager instance.
- For coverage runs, `--cov` is generally stable; if `--cov=<module path>` triggers
  local Qt import issues, use broader `--cov=erlab` and filter coverage output to target
  files.
- Do not make runtime code behave differently under pytest (e.g., `PYTEST_VERSION`,
  `sys.modules["pytest"]`, or test-only env checks in `src/`). Keep production behavior
  explicit via function arguments/state, and implement test-specific behavior in
  tests/fixtures/monkeypatching instead.
- `# pragma: no cover` / `# pragma: no branch` is allowed for edge cases that are hard
  or impractical to exercise in CI; prefer tests when feasible and add a brief comment
  explaining why the pragma is needed.
- Do not export private compatibility shims only for monkeypatched tests; update tests
  to patch the real module/function location instead.
- When manager internals change, update test doubles and helper methods to the new
  interface instead of preserving obsolete paths in runtime code with `hasattr` guards
  or wrapper-scanning fallbacks.

## Commits and pull requests

Follow Conventional Commits with scopes, for example,
`feat(analysis.gold): support multi-angle Fermi fits`. Reference issues with `(#123)`
when relevant.

PRs should summarize behavior changes, list the commands you ran, and include
screenshots or GIFs for GUI changes. Mention dependent data or documentation PRs in the
description. The required check commands and their conditions are in
[Commands and required checks](#commands-and-required-checks).

When a user asks for a commit message, provide a Conventional Commit subject. Include
a longer, user-facing description paragraph if there are user-visible changes.
