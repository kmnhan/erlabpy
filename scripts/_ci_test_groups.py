from __future__ import annotations

import ast
import pathlib
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"


def _test_function_names(file_target: str) -> tuple[str, ...]:
    file_path = REPO_ROOT / file_target
    tree = ast.parse(file_path.read_text())
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def _split_test_file_targets(
    file_target: str, *, index: int, total: int
) -> tuple[str, ...]:
    if not 0 <= index < total:
        raise ValueError("split index must be in range(total)")
    nodeids = tuple(
        f"{file_target}::{name}" for name in _test_function_names(file_target)
    )
    if not nodeids:
        raise ValueError(f"No top-level test functions found in {file_target}")
    return nodeids[index::total]


COMPAT_TARGETS: tuple[str, ...] = (
    "tests/accessors/test_fit.py",
    "tests/accessors/test_general.py",
    "tests/analysis/test_transform.py",
    "tests/analysis/test_xps.py",
    "tests/io/test_igor.py",
    "tests/io/test_io_utils.py",
    "tests/plotting/test_atoms.py",
    "tests/plotting/test_general.py",
    "tests/interactive/test_colors.py",
    "tests/interactive/test_options.py",
    "tests/interactive/test_ptable.py",
    "tests/interactive/imagetool/test_module_split.py",
    "tests/interactive/imagetool/test_server_multipart.py",
    "tests/interactive/imagetool/test_manager_warnings.py",
    "tests/io/test_dataloader.py::test_loader",
    "tests/utils/test_array.py",
)

COVERAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "cov-core-a": (
        "tests/accessors",
        "tests/analysis/test_kspace.py",
    ),
    "cov-core-b": (
        "tests/analysis/fit",
        "tests/analysis/test_gold.py",
        "tests/analysis/test_correlation.py",
        "tests/analysis/test_image.py",
        "tests/analysis/test_image_savgol.py",
        "tests/analysis/test_interpolate.py",
        "tests/analysis/test_transform.py",
        "tests/analysis/test_mask.py",
        "tests/analysis/test_mesh.py",
        "tests/analysis/test_xps.py",
        "tests/plotting",
        "tests/utils",
        "tests/test_conftest.py",
        "tests/test_constants.py",
        "tests/test_lattice.py",
    ),
    "cov-io": ("tests/io",),
    "cov-qt-imagetool": (
        "tests/interactive/imagetool/test_history.py",
        "tests/interactive/imagetool/test_imagetool.py",
        "tests/interactive/imagetool/test_imagetool_coordinate_widget.py",
        "tests/interactive/imagetool/test_manager_warnings.py",
        "tests/interactive/imagetool/test_module_split.py",
        "tests/interactive/imagetool/test_provenance.py",
        "tests/interactive/imagetool/test_replay_graph.py",
        "tests/interactive/imagetool/test_server_multipart.py",
        "tests/interactive/imagetool/test_slicer.py",
        "tests/interactive/imagetool/test_watcher.py",
    ),
    "cov-qt-manager-figurecomposer": (
        "tests/interactive/imagetool/manager/figurecomposer",
    ),
    "cov-qt-manager-mainwindow": (
        "tests/interactive/imagetool/manager/test_mainwindow.py",
        "tests/interactive/imagetool/manager/test_modelview.py",
        "tests/interactive/imagetool/manager/test_wrapper.py",
    ),
    "cov-qt-manager-workspace-a": _split_test_file_targets(
        "tests/interactive/imagetool/manager/test_workspace.py",
        index=0,
        total=2,
    ),
    "cov-qt-manager-workspace-b": _split_test_file_targets(
        "tests/interactive/imagetool/manager/test_workspace.py",
        index=1,
        total=2,
    ),
    "cov-qt-manager-provenance-console": (
        "tests/interactive/imagetool/manager/test_provenance.py",
        "tests/interactive/imagetool/manager/test_console.py",
        "tests/interactive/imagetool/manager/test_app.py",
        "tests/interactive/imagetool/manager/test_dialogs.py",
        "tests/interactive/imagetool/manager/test_registry_server.py",
        "tests/interactive/imagetool/manager/test_warnings.py",
    ),
    "cov-qt-tools": (
        "tests/interactive/test_bzplot.py",
        "tests/interactive/test_dask.py",
        "tests/interactive/test_derivative.py",
        "tests/interactive/test_explorer.py",
        "tests/interactive/test_fastbinning.py",
        "tests/interactive/test_fermiedge.py",
        "tests/interactive/test_fit1d.py",
        "tests/interactive/test_fit2d.py",
        "tests/interactive/test_kspace.py",
        "tests/interactive/test_magic.py",
        "tests/interactive/test_mesh.py",
    ),
    "cov-qt-ux": (
        "tests/interactive/test_colors.py",
        "tests/interactive/test_options.py",
        "tests/interactive/test_options_parameters.py",
        "tests/interactive/test_options_tree.py",
        "tests/interactive/test_ptable.py",
        "tests/interactive/test_utils.py",
    ),
}

GUI_PREFIXES: tuple[str, ...] = ("tests/interactive/",)
GUI_TARGETS: tuple[str, ...] = ()

SERIAL_PREFIX_GROUPS: tuple[tuple[str, str | None], ...] = (
    ("tests/interactive/", None),
)
SERIAL_TARGET_GROUPS: dict[str, str] = {
    "tests/io/plugins/test_erpes.py": "hdf5-cache",
    "tests/io/plugins/test_maestro.py": "hdf5-cache",
}
SERIAL_NODEID_GROUPS: dict[str, str] = {
    "tests/interactive/test_explorer.py::test_explorer_general": (
        "qt-tests-interactive-test_explorer"
    ),
    "tests/interactive/imagetool/test_watcher.py::test_watcher_real": (
        "qt-tests-interactive-imagetool-test_watcher"
    ),
    (
        "tests/interactive/test_utils.py::"
        "test_tool_window_managed_detached_output_preserves_provenance"
    ): "qt-tests-interactive-test_utils",
}
SERIAL_PREFIXES: tuple[str, ...] = tuple(
    prefix for prefix, _group in SERIAL_PREFIX_GROUPS
)
SERIAL_TARGETS: tuple[str, ...] = tuple(SERIAL_TARGET_GROUPS)
SERIAL_NODEIDS: frozenset[str] = frozenset(SERIAL_NODEID_GROUPS)

COMPAT_NODEIDS: frozenset[str] = frozenset(
    target for target in COMPAT_TARGETS if "::" in target
)
COMPAT_FILES: frozenset[str] = frozenset(
    target for target in COMPAT_TARGETS if "::" not in target
)


def _normalize_relative(path: pathlib.Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def iter_all_test_files() -> list[str]:
    return sorted(_normalize_relative(path) for path in TESTS_ROOT.rglob("test_*.py"))


def expand_targets(targets: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for target in targets:
        if "::" in target:
            file_target, _ = target.split("::", maxsplit=1)
            file_path = REPO_ROOT / file_target
            if not file_path.is_file():
                raise ValueError(f"Unknown test node target: {target}")
            expanded.append(target)
            continue

        path = REPO_ROOT / target
        if path.is_dir():
            expanded.extend(
                sorted(
                    _normalize_relative(file_path)
                    for file_path in path.rglob("test_*.py")
                )
            )
        elif path.is_file():
            expanded.append(target)
        else:
            raise ValueError(f"Unknown test target: {target}")
    return expanded


def expand_file_targets(targets: Sequence[str]) -> list[str]:
    return [target for target in expand_targets(targets) if "::" not in target]


def get_group_targets(group: str) -> tuple[str, ...]:
    if group == "compat":
        return COMPAT_TARGETS
    try:
        return COVERAGE_GROUPS[group]
    except KeyError as exc:
        raise KeyError(f"Unknown test group: {group}") from exc


def get_group_file_targets(group: str) -> list[str]:
    return expand_file_targets(get_group_targets(group))


def coverage_membership() -> dict[str, list[str]]:
    membership: dict[str, list[str]] = defaultdict(list)
    for group_name, targets in COVERAGE_GROUPS.items():
        for file_target in expand_file_targets(targets):
            membership[file_target].append(group_name)
    return dict(membership)


def coverage_nodeid_membership() -> dict[str, dict[str, list[str]]]:
    membership: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for group_name, targets in COVERAGE_GROUPS.items():
        for target in expand_targets(targets):
            if "::" not in target:
                continue
            file_target, nodeid = target.split("::", maxsplit=1)
            membership[file_target][nodeid].append(group_name)
    return {file_target: dict(nodeids) for file_target, nodeids in membership.items()}


def check_coverage_partition() -> list[str]:
    all_files = set(iter_all_test_files())
    membership = coverage_membership()
    nodeid_membership = coverage_nodeid_membership()
    counts = Counter(
        {file_target: len(groups) for file_target, groups in membership.items()}
    )

    missing = sorted(all_files - counts.keys() - nodeid_membership.keys())
    duplicate_members = {
        file_target: groups
        for file_target, groups in sorted(membership.items())
        if len(groups) > 1
    }
    split_and_file_members = sorted(counts.keys() & nodeid_membership.keys())
    extras = sorted((counts.keys() | nodeid_membership.keys()) - all_files)

    errors: list[str] = []
    if missing:
        errors.append("Missing from coverage partition:")
        errors.extend(f"  - {file_target}" for file_target in missing)
    if duplicate_members:
        errors.append("Covered by multiple groups:")
        errors.extend(
            f"  - {file_target}: {', '.join(groups)}"
            for file_target, groups in duplicate_members.items()
        )
    if split_and_file_members:
        errors.append("Covered both as a full file and split nodeids:")
        errors.extend(f"  - {file_target}" for file_target in split_and_file_members)
    for file_target, nodeids in sorted(nodeid_membership.items()):
        expected_nodeids = set(_test_function_names(file_target))
        missing_nodeids = sorted(expected_nodeids - nodeids.keys())
        extra_nodeids = sorted(nodeids.keys() - expected_nodeids)
        duplicate_nodeids = {
            nodeid: groups
            for nodeid, groups in sorted(nodeids.items())
            if len(groups) > 1
        }
        if missing_nodeids:
            errors.append(
                f"Missing split nodeids from coverage partition: {file_target}"
            )
            errors.extend(f"  - {nodeid}" for nodeid in missing_nodeids)
        if duplicate_nodeids:
            errors.append(f"Split nodeids covered by multiple groups: {file_target}")
            errors.extend(
                f"  - {nodeid}: {', '.join(groups)}"
                for nodeid, groups in duplicate_nodeids.items()
            )
        if extra_nodeids:
            errors.append(f"Unknown split nodeids in coverage partition: {file_target}")
            errors.extend(f"  - {nodeid}" for nodeid in extra_nodeids)
    if extras:
        errors.append("Unknown files referenced by coverage groups:")
        errors.extend(f"  - {file_target}" for file_target in extras)
    return errors


def is_gui_path(rel_path: str) -> bool:
    return rel_path in GUI_TARGETS or rel_path.startswith(GUI_PREFIXES)


def is_serial_path(rel_path: str) -> bool:
    return rel_path in SERIAL_TARGETS or rel_path.startswith(SERIAL_PREFIXES)


def is_serial_nodeid(nodeid: str) -> bool:
    return nodeid in SERIAL_NODEIDS


def serial_xdist_group(rel_path: str, nodeid: str) -> str | None:
    if nodeid in SERIAL_NODEID_GROUPS:
        return SERIAL_NODEID_GROUPS[nodeid]
    if rel_path in SERIAL_TARGET_GROUPS:
        return SERIAL_TARGET_GROUPS[rel_path]
    for prefix, group_name in SERIAL_PREFIX_GROUPS:
        if rel_path.startswith(prefix):
            if group_name is not None:
                return group_name
            return f"qt-{rel_path.removesuffix('.py').replace('/', '-')}"
    return None


def is_compat_path(rel_path: str) -> bool:
    return rel_path in COMPAT_FILES


def is_compat_nodeid(nodeid: str) -> bool:
    return nodeid in COMPAT_NODEIDS


def iter_known_groups() -> Iterable[str]:
    yield from COVERAGE_GROUPS
    yield "compat"
