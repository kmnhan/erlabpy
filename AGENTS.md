# Repository Guidelines

## Project Structure & Module Organization

Place runtime code in `src/erlab/` (analysis routines, interactive Qt tools, IO plugins, visualization helpers). Tests mirror that tree under `tests/`, with plugin fixtures in `tests/io/plugins/`. Build outputs (`build/`, `dist/`, `manager.spec`), shared assets (`resources/`, `PythonInterface.ipf`), and docs (`docs/`) stay isolated so releases remain lean.

## Build, Test, and Development Commands

- `uv sync --all-extras --dev --group pyqt6` — editable env with every optional feature and Qt bindings installed for GUI tests.
- `uv run pytest` — whole suite with coverage configuration from `pyproject.toml`.
- `uv run ruff format .` and `uv run ruff check --fix .` after every change to keep style consistent. Prefer automatic fixes instead of manual lint cleanups.
- `uv run mypy src` — static typing pass.
- `uv run pyinstaller manager.spec` — bundle the ImageTool manager app.
- `uv build` — produce wheels and sdists for release.

## Documentation Workflow

Sources live in `docs/source/` (MyST + Sphinx). Install extras using `uv sync --all-extras --dev --group docs`, then render locally via `uv run --directory docs make html`. Put tutorials in `docs/source/user-guide/`, guides in `docs/source/contributing.md`, and images in `docs/source/images/`. Run `make linkcheck` before pushing large doc edits to guard cross-references. The `sphinxext-rediraffe` extension is used for maintaining redirects. Add to the `rediraffe_redirects` dict in `docs/source/conf.py` when moving or renaming pages.

When changing docs content or URLs, verify that `skills/arpes-analysis/SKILL.md` still matches current docs/links and update it if needed.

## Coding Style & Naming Conventions

Use 4-space indentation, Ruff’s 88-character limit, and double quotes. Modules/functions stay snake_case, classes use CapWords. Some Qt widgets keep co-located `.ui` files, import bindings through `qtpy`, and rely on explicit enums such as `QtCore.Qt.CheckState.Checked`. In case of Qt imports, prefer `from qtpy import QtWidgets, QtCore, QtGui`. Install `prek` so Ruff, mypy, and commitizen hooks run automatically. Docstrings use NumPy style. It is recommended to follow PEP 484 type hinting for all public APIs.

## Testing Guidelines

Pytest enforces strict markers and `xfail_strict`; name files `test_<feature>.py` beside the code they cover. Loader plugins need regression data in `tests/io/plugins/test_<plugin>.py`; set `ERLAB_TEST_DATA_DIR` to a local clone of `erlabpy-data` so fixtures resolve. Coverage already skips legacy updater code, so aim for branch coverage elsewhere and parametrize datasets to catch multidimensional regressions.

## Commit & Pull Request Guidelines

Follow Conventional Commits with scopes (e.g., `feat(analysis.gold): support multi-angle Fermi fits`) and reference issues via `(#123)` when relevant. PRs should summarize behavior changes, list the commands you ran, and attach screenshots or GIFs for GUI tweaks. Run `uv run ruff check`, `uv run ruff format --check`, `uv run mypy src`, and `uv run pytest` before requesting review, and mention dependent data/doc PRs in the description.
