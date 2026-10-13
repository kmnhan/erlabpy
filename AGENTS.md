# Repository guidelines

## Checks

Run commands from the repository root. After changes, run `uv run ruff format .`
and `uv run ruff check --fix .`. Formatting, lint, and typing conventions are defined
in `pyproject.toml`.

Choose tests for the changed behavior and reuse passing checks while that code is
unchanged. Test moves, target-path updates, and small test additions need affected
tests and the partition check, not the full suite or typing pass.

- After test-tree or CI grouping changes, run
  `uv run python -m scripts.ci_test_groups --check-partition`.
- Add new top-level test modules to `scripts/_ci_test_groups.py`. Keep collection
  markers centralized there and in `tests/conftest.py`.
- For shared collection, marker, or grouping changes that can affect unrelated tests,
  or when requested, run the full validation set: `uv run ruff check`,
  `uv run ruff format --check`, `uv run mypy src`, `uv run pytest`, and the partition
  check.
- After documentation source or public docstring changes, run
  `uv run --directory docs make html`. Before pushing large documentation edits,
  also run `uv run --directory docs make linkcheck`.
- Validate touched Qt lifetime and compatibility paths under both PyQt6 and PySide6.

## Task guidance

Read the matching agent instructions. Use contributor guides for setup and APIs.

| Task | Guidance |
| --- | --- |
| Environment setup | [Development installation](docs/source/contributing/development.md#editable-installation-from-source); install the configured hooks with `uv run prek install` |
| Test changes or failures | [Test instructions](.agents/testing.md) |
| Qt widgets, actions, or Manager behavior | [Qt instructions](.agents/qt.md) |
| New interactive tools or Manager outputs | [Tool interface and integration](docs/source/contributing/interactive-tools.md) |
| Copied code or provenance | [Generated-code instructions](.agents/generated-code.md) |
| Documentation or public API behavior changes | [Documentation instructions](.agents/documentation.md) |
| Loader changes | [Loader guide](docs/source/contributing/loaders.ipynb); set `ERLAB_TEST_DATA_DIR` to a local `erlabpy-data` clone for fixture tests |
| ImageTool Manager packaging | [.github/workflows/build.yml](.github/workflows/build.yml) for dependencies, platform setup, and the non-editable build sequence |

## Code contracts

- Keep small implementation paths direct. Add helpers or abstractions only when they
  remove real complexity, define a useful boundary, or follow an established pattern.
- Prefer absolute imports. In modules that use `lazy_loader`, prefer top-level
  `erlab` imports. Use explicit imports rather than `globals()` mutation.
- In `src/`, use explicit checks and exceptions for runtime invariants, not `assert`.
  Use `typing.cast` for narrowing that should not affect runtime behavior.
- Qt-facing code and tests must behave consistently under PyQt6 and PySide6.
- Tests must follow the current runtime contract. Fix outdated test doubles instead
  of adding production compatibility shims or pytest-dependent behavior.

## Writing and contributions

Use ASD-STE100 Simplified Technical English in responses, documentation, commits, and
PRs. Use short, connected sentences and established scientific terms. Define unfamiliar
terms. Use concise noun or gerund phrases for headings. Avoid excessive hyphenation.

For commit and changelog text, read [Writing instructions](.agents/writing.md).

PRs must explain the behavior change, list checks run, include screenshots or GIFs for
GUI changes, and identify dependent data or documentation PRs.
