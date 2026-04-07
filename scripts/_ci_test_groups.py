from __future__ import annotations

import pathlib
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"

COMPAT_TARGETS: tuple[str, ...] = (
    "tests/accessors/test_fit.py",
    "tests/accessors/test_general.py",
    "tests/analysis/test_transform.py",
    "tests/analysis/test_xps.py",
    "tests/io/test_igor.py",
    "tests/io/test_io_utils.py",
    "tests/plotting/test_atoms.py",
    "tests/plotting/test_general.py",
    "tests/interactive/test_options.py",
    "tests/interactive/test_colors.py",
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
    "cov-qt-imagetool": ("tests/interactive/imagetool",),
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

SERIAL_PREFIXES: tuple[str, ...] = ("tests/interactive/imagetool/",)
SERIAL_TARGETS: tuple[str, ...] = ()

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


def check_coverage_partition() -> list[str]:
    all_files = set(iter_all_test_files())
    membership = coverage_membership()
    counts = Counter(
        {file_target: len(groups) for file_target, groups in membership.items()}
    )

    missing = sorted(all_files - counts.keys())
    duplicate_members = {
        file_target: groups
        for file_target, groups in sorted(membership.items())
        if len(groups) > 1
    }
    extras = sorted(counts.keys() - all_files)

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
    if extras:
        errors.append("Unknown files referenced by coverage groups:")
        errors.extend(f"  - {file_target}" for file_target in extras)
    return errors


def is_gui_path(rel_path: str) -> bool:
    return rel_path in GUI_TARGETS or rel_path.startswith(GUI_PREFIXES)


def is_serial_path(rel_path: str) -> bool:
    return rel_path in SERIAL_TARGETS or rel_path.startswith(SERIAL_PREFIXES)


def is_compat_path(rel_path: str) -> bool:
    return rel_path in COMPAT_FILES


def is_compat_nodeid(nodeid: str) -> bool:
    return nodeid in COMPAT_NODEIDS


def iter_known_groups() -> Iterable[str]:
    yield from COVERAGE_GROUPS
    yield "compat"
