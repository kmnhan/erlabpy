# Agent instruction audit

Audit date: 2026-09-07. Baseline: `510c33136` (`main`). The working tree was clean
before the audit. This document records the audit; it does not add agent policy.

## Diagnosis

The root instructions contain useful project knowledge. The main problems were
scattered requirements, dense paragraphs, repeated rules, ambiguous check scope, and
several stale factual claims. The Qt lifetime, generated-code, test-double, and
scientific calibration rules address distinct risks. They were retained.

The refactor groups related requirements and separates command descriptions from
required checks. It keeps the root file as the source of repository-wide guidance.
No policy was moved into a nested file with narrower applicability. No performance or
token-saving claim follows from the word counts.

## Instruction inventory and loading

| Source | Inspection and loading status |
| --- | --- |
| Root `AGENTS.md` | Loaded in the task context and read from disk. The supplied repository text matched the original file. |
| Nested `AGENTS.md` and `AGENTS.override.md` | No additional project-owned files found. The search included hidden and ignored files. No override file was found. |
| `.venv/lib/python3.14/site-packages/marimo/AGENTS.md` and `.venv-build-homebrew/lib/python3.14/site-packages/marimo/AGENTS.md` | Present and read for inventory. These are dependency instructions, not loaded repository-wide rules. Neither was edited. |
| Ancestors from `/` through `/Users/khan/Source/python` | Checked both instruction filenames at each level. None were present. |
| `/Users/khan/.codex/AGENTS.md` | Accessible global file. Its delegation paragraph was also supplied in the task context. Read-only scouts used the specified low effort, no inherited conversation, separate ownership, and no further delegation. No global file was edited. |
| `/Users/khan/.codex/config.toml` | Inspected instruction-related settings. No custom instruction-file or project-document discovery setting was found in the searched fields. Configuration presence is not evidence that arbitrary files are loaded. |
| Session and app instructions | Visible task instructions govern autonomy, authorization, check scope, communication, and tool use. They were considered without copying them into repository policy. This audit cannot inspect hidden provider instructions or prove another client's future loading behavior. |
| `/Users/khan/.agents/skills/writing-for-agents/SKILL.md` | Read and applied for organization and explicit context pointers. Its suggestions to prune presumed defaults or use model-specific wording did not override the user's preservation rules. |
| `skills/arpes-analysis/SKILL.md` | Read for the audit, with all five reference documents and `agents/openai.yaml`. Its scientific operating contract was inspected, not activated as a request to analyze data. |
| `/Users/khan/.codex/skills/arpes-analysis/SKILL.md` | Listed in the available skill catalog, then inspected and compared. Its full body was not initially loaded merely by being listed. It differs from the repository copy. See unresolved issues below. |
| Contributor documentation | Read `docs/source/contributing.md`, `contributing/development.md`, `contributing/documentation.md`, `contributing/interactive-tools.md`, and the Markdown cells of `contributing/loaders.ipynb`. These were evidence and relevant guidance, not automatically loaded files. |
| Prior task notes | Consulted for repeated wasted work and the reasons behind unusual rules. Historical observations were not treated as current configuration or permission to remove policy. |

No requested project instruction file was inaccessible. The scope did not include
unrelated global skills, every third-party instruction document, or a forensic check
of all possible client configuration sources.

## Edits and word counts

Counts use whitespace-separated words, including headings, code, and link text, as
with `wc -w`. Original counts come from `git show HEAD:<path>`.

| File | Before | After | Change |
| --- | ---: | ---: | ---: |
| `AGENTS.md` | 2,217 | 2,360 | +143 |
| `skills/arpes-analysis/references/docs-links.md` | 219 | 215 | -4 |
| `skills/arpes-analysis/references/publication-plotting.md` | 855 | 855 | 0 |
| Edited instruction files | 3,291 | 3,430 | +139 |
| `skills/arpes-analysis/SKILL.md` | 1,804 | 1,804 | Unchanged |
| This audit report | 0 | 3,390 | New review artifact |

The complete original instruction corpus above, including the main skill and all five
references, contained 9,457 words. The revised corpus contains 9,596 words. This audit
report is a separate review artifact and is excluded from those totals.

The working-tree diff contains the exact edits. The root file now has short paragraphs,
task subsections, a documentation directory table, and explicit internal references.
The skill edits remove one exact duplicate link and correct `explitly` to `explicitly`.

## Requirement preservation

Original line numbers refer to `AGENTS.md` at the baseline above. Destinations refer to
the revised root file. Rows cover every substantive original paragraph or bullet;
exactly overlapping requirements share a row. No distinct behavioral requirement was
removed. Changes to factual descriptions are explained in the next section.

| Original lines and requirements | Disposition and destination |
| --- | --- |
| 5: runtime areas, mirrored tests, plugin fixtures, separate outputs/assets/docs | Retained in Project layout. Corrected the classification of `manager.spec`. |
| 9-12, 14-16: development setup, pytest, partition, compatibility targets, mypy, packaging, release build | Retained in Commands and required checks. Clarified package extras, explicit coverage, and packaging prerequisites. |
| 13: whole-repository Ruff after every change; prefer automatic fixes | Retained at Required checks with the same breadth and preference. |
| 20-27: ASD-STE100 scope, short direct sentences, one idea, precise terms, no idioms/jargon/ambiguity, technical-term exception, definitions, heading forms, scientific terminology | Retained in Writing style. |
| 31: MyST/Sphinx setup and HTML build; all five content locations | Retained in Documentation; locations moved into a table in the same section. |
| 31: linkcheck before large-doc pushes; redirects when moving/renaming pages | Retained in Documentation. Corrected linkcheck's working directory. |
| 33-41: all four How-to qualification tests | Retained verbatim in Documentation types. |
| 43-46: task guide for tutorial capabilities; remove teaching/demo framing; do not remove shared-code guides; link Explanation/Reference | Retained verbatim in Documentation types. |
| 48-53: tutorial xarray/data conventions; Explanation assumptions and purpose; no repeated introductions; prefer focused visuals; procedure figures in How-to | Retained verbatim in Documentation types. |
| 55-56: required behavior-change note; next-release recommendation and version check; sparse feature/API additions | Retained in API notes and consistency, including advisory strength and version uncertainty. |
| 58: check local skill after content/URL changes; update if needed; concrete user-facing prose and preferences | Retained in API notes and consistency. The skill pointer now explicitly says when to read it. |
| 62: indentation, line length, quotes, naming, NumPy docstrings, recommended public type hints | Retained in Code style and imports. |
| 62: Qt imports, explicit enums, preferred module imports, co-located UI files | Moved within the root file to Bindings and lifetime. |
| 62: install prek for automatic hooks | Retained in Code style and imports. Added the actual installation command; corrected the false mypy-hook claim without adding an unconditional mypy check. |
| 63-64: prefer top-level erlab with lazy_loader, even if narrower imports are possible; prefer absolute imports | Merged into one bullet with both preferences and the original condition. |
| 65: direct small paths; no one-off helpers/dataclasses/aliases unless complexity, boundary, or local-pattern exception applies | Retained in Code style and imports with all three exceptions. |
| 66-67: modern typing default; avoid deprecated aliases; explicit eight-name ban; built-ins and collections.abc; TYPE_CHECKING condition | Merged into one bullet. Both the general preference and specific prohibition remain. |
| 68-69: no runtime asserts in src; explicit invariant checks; casts for narrowing; avoid globals mutation | Retained in Code style and imports with the original scope and alternatives. |
| 73: test naming/location, plugin regression data, local-data environment variable, strict pytest, updater exclusions, branch/multidimensional coverage | Retained in Tests and CI. Corrected contradictory test placement and the current pytest option spelling. Local-data instruction remains mandatory. |
| 75: re-examine runtime after new-test failures; change assertions only with justification | Retained in Test behavior. |
| 76-78: fast and weekly matrices; centralized shard definitions; exact-one-shard assignment; conditional smoke coverage; centralized semantic markers | Retained in CI grouping with all directory and marker names. |
| 79 and 115: partition on test-tree/grouping changes; full five-command checklist on CI grouping/new top-level modules | Moved to Required checks. CI and PR sections link back. The broader partition trigger remains separate. |
| 80: current test-double contract; no production fallback for obsolete fakes | Retained in Test behavior. Qt-specific extensions below remain scoped to Qt. |
| 81 and 95: no user-facing/action-label assertions; use stable metadata/state | Merged in Test behavior. Retained action data, roles, shortcuts, enabled state, object names, and behavior as alternatives. |
| 81 and 89: exec in explicit namespace; exact result checks; text-comparison exception; conditional cleanliness tests; emitted-code review | Merged in Generated code and provenance. Test behavior explicitly points there. |
| 85: clean generated/copied/replay code as product and teaching material; public complete expressions; semantic names and sources; no relay/scratch/internal temporaries; safe alias inlining | Retained in Generated code and provenance. Split dense prose without removing naming constraints. |
| 87: representative copied code; single-step/composed imports; no copies/private machinery solely for replay isolation; visible metadata updates; separate exact non-mutating replay | Retained in Generated code and provenance with the different execution-path contracts intact. |
| 93: equal PyQt6/PySide6 behavior; neutral APIs; avoid binding assumptions; helper and dual validation for unavoidable low-level behavior | Retained in Bindings and lifetime. |
| 94: explicit ownership; queued/deferred validity guards; strong wrapper references; destroyed identity; exact disconnection; dual-binding lifetime checks | Retained in Bindings and lifetime. |
| 95: Unicode ellipsis only for required input/selection/confirmation; all exclusions; no ASCII ellipsis | Retained in Actions and Manager integration. Only the duplicate label-test sentence moved. |
| 96: manager-aware new open/show actions from managed parent tools | Retained with advisory strength in Actions and Manager integration. |
| 97-99: accept/cancel tests; prefer accept_dialog; stubs for difficult branches; changed-line coverage including early returns, with untestable exception | Retained in Qt tests and diagnostics. |
| 100: argv pitfall; explicit test node IDs or argv patch | Retained in Qt tests and diagnostics. Corrected the entry-point name. |
| 101-102: port warning; inspect/terminate stale managers on bind failures/timeouts; no parallel manager jobs unless ports isolated | Corrected the fixture-port fact. Preserved both diagnostic action and parallelism condition. |
| 103-104: patch dispatch-only integration; reserve real manager for dependent behavior; conditional broad coverage fallback and filtering | Retained in Qt tests and diagnostics. |
| 105: no pytest-specific runtime behavior; explicit production inputs/state; test-side fixtures/patching | Retained in Qt tests and diagnostics, including concrete examples. |
| 106: suppress duplicate Manager alerts when code already shows a dialog | Moved within the root file to Actions and Manager integration; exact logging option retained. |
| 107: coverage pragmas allowed for difficult CI edges; prefer tests; explain pragma | Retained in Qt tests and diagnostics with all qualifications. |
| 108-109: no exported shims for monkeypatches; patch actual location; update Manager test interfaces; no obsolete hasattr/wrapper fallbacks | Retained in Qt tests and diagnostics. |
| 110: remove super-only override; audit nearby overrides during consolidation | Moved within the Qt section to Bindings and lifetime. |
| 111: preserve IPython watchers; test post-run-cell and polling fallback | Moved to Actions and Manager integration with both paths retained. |
| 115-116: scoped Conventional Commits, relevant issue notation, PR behavior/commands/GUI media/dependencies, longer user-facing commit body when applicable | Retained in Commits and pull requests. Conditional checks moved as recorded above. |

The main ARPES skill and its Fermi, momentum, and curve-analysis references are
unchanged. All notebook acceptance gates, raw-data preservation rules, physical
calibration conditions, explicit approval boundaries, clean-kernel execution, source
priority, and conditional reference-loading instructions remain. The plotting
reference changed only one misspelled word. One of two identical Fermi-guide links
was removed from the documentation index; the destination is still listed.

## Evidence for factual corrections

| Correction | Repository evidence |
| --- | --- |
| `manager.spec` is a recipe, not generated output | Tracked file; imports and `Analysis(...)` build definition in `manager.spec:1-8,116`. |
| Development setup installs package extras and PyQt6, not every optional dependency group | `pyproject.toml:72-138` distinguishes optional dependencies from `docs`, `pyinstaller`, and binding groups. |
| Plain pytest does not enable coverage | `pyproject.toml:226-231` has no `--cov`; CI supplies coverage explicitly at `.github/workflows/ci.yml:179`. Coverage settings remain in `pyproject.toml:166-169`. |
| Packaging needs the separate dependency group | `pyproject.toml:122-132`; imports at `manager.spec:7-8`; actual non-editable build sequence at `.github/workflows/build.yml:80-95`. |
| Root-invoked linkcheck needs the docs directory | `docs/Makefile:19-20` routes targets through Sphinx. The root has no Makefile. |
| Installing prek alone does not enable hooks; no mypy hook is configured | `.pre-commit-config.yaml:34-53`; setup at `docs/source/contributing/development.md:192-194`. Mypy remains a separate required command under the preserved conditions. |
| Tests belong under the mirrored tests tree, not beside runtime source | Original root layout rule; `pyproject.toml:245` sets `testpaths = ["tests"]`; representative `tests/analysis/`, `tests/interactive/`, and `tests/io/plugins/` trees. |
| Current pytest spelling is `strict_xfail` | `pyproject.toml:244`. Strict failure behavior is retained. |
| Manager main is a module function | `src/erlab/interactive/imagetool/manager/__init__.py:398-405`. |
| Manager fixture ports are ephemeral | `tests/conftest.py:553-564` isolates the registry and sets ports to zero. Runtime defaults remain at `manager/_server.py:104-115`; multi-manager behavior is documented in `manager/__init__.py:21-24`. |
| Link deletion and spelling correction remove no rule | Original `references/docs-links.md:19,22` had identical targets and labels. Original `references/publication-plotting.md:20` contained the typo. |

## Unresolved issues and unapplied policy proposals

These are review items, not new requirements.

1. **Whole-repository formatting after every change.** This includes instruction-only
   edits and can change unrelated files. The required formatter did exactly that in
   this audit: it reformatted a Python example in the untouched contributor guide.
   The unrelated edit was removed. Consider defining changed-file formatting as the
   normal edit loop and reserving repository-wide formatting for a stated gate.
   The existing command and breadth were retained.
2. **Full-check scope.** Original line 115 attaches its condition to the entire
   five-command list. The refactor preserves that literal scope. It does not establish
   whether full pytest/mypy are required for every other code change. Contributor
   documentation says to run applicable checks and shows a broader workflow. Define
   ordinary local checks, pre-PR checks, and CI-delegated checks explicitly before
   changing the requirement. CI checks mypy on `.`; the root command remains `src`.
3. **Environment synchronization.** Exact `uv sync` can remove groups from an existing
   environment. The audit's documented docs sync removed the installed PySide6 and
   packaging groups. Consider a documented separate environment or group-preserving
   setup for mixed tasks. No environment policy or command was silently substituted.
4. **Local test-data clone.** The instruction to set `ERLAB_TEST_DATA_DIR` was retained.
   `test_data_dir` can download pinned data when the variable is absent
   (`tests/conftest.py:403-420`). The loader contribution notebook explicitly requires
   a clone when adding regression data. Consider distinguishing data contribution
   from running existing tests; runtime fallback alone does not invalidate policy.
5. **Hook strength and release assumptions.** Root guidance requires prek installation;
   contributor guidance recommends it. Keep the stronger root rule pending a decision.
   Version `3.27.2` is current, but the planned next version was not established.
   The existing "likely a minor bump" wording remains an assumption requiring a check.
6. **Manager timeout recovery.** Ephemeral ports reduce collisions but do not make every
   timeout a stale-process failure. Consider requiring endpoint and process ownership
   evidence before termination, and distinguishing GUI teardown failures. The original
   diagnostic requirement and parallelism exception were retained.
7. **Global skill drift.** The installed ARPES skill lacks the repository's notebook
   reproducibility acceptance gate and `adaptive=True` guidance. It recommends
   `fast=True`, which `src/erlab/analysis/gold.py:99-112` deprecates in favor of
   `use_step_edge`. Updating the installed copy is outside this task's edit boundary.
8. **Scientific approval and copying boundaries.** The skill's calibration handoffs are
   conditional scientific requirements, not general permission gates. Its raw-data
   protection and independent candidate copies serve a different purpose from GUI
   copied code. `tests/interactive/imagetool/test_provenance.py:1782-1803` verifies
   visible copied-code metadata updates and separate replay isolation. Neither policy
   was relaxed. Consider unifying repeated calibration wording only after mapping
   each distinct trigger and the unavailable-GUI fallback.
9. **Inherited guidance and historical notes.** The writing skill recommends pruning
   presumed defaults and using model-specific shortcuts. Those suggestions conflict
   with this audit's preservation rules and were not applied. Historical notes about
   read-only investigations followed by approval must not become a general extra
   approval gate for already authorized implementation. The global delegation text
   was not imported into repository policy or changed.
10. **Low-impact tests and Qt coverage.** Session guidance discourages tests that merely
    mirror reversible low-impact edits. Repository Qt guidance expects direct coverage
    of touched lines, with explicit exceptions. Preserve both scopes and favor useful
    behavioral tests. Do not use the general advice to drop binding or lifetime checks.

## Historical evidence about wasted work

Prior task notes identify avoidable work beyond instruction length:

- A previous pyqtgraph audit ran broad comparisons with little useful output. Focused
  method comparisons were more useful. Its full pytest run was interrupted after the
  user requested PR publication; focused dual-binding regressions had already passed.
- An ARPES planning task overemphasized validation machinery and repeatedly asked the
  user to choose architecture. The user wanted a small useful execution path and no
  token-consuming model evaluations in CI. That workbench remained a proposal; it is
  not a repository feature or permission to implement one in this audit.
- Prior Qt and provenance notes document actual failures behind wrapper ownership,
  deferred validity checks, current test doubles, semantic generated code, and separate
  replay isolation. These are not candidates for deletion merely because they look
  obvious.

These are historical observations, not a current performance benchmark. They support
reviewing unnecessary repetition and check scope. They do not justify removing the
current scientific acceptance gates or compatibility promises. No model-based
evaluation, new workbench, memory update, or new policy was added.

## Verification record

| Check | Result |
| --- | --- |
| Requirement comparison | Compared every original paragraph/bullet with the revision. A separate read-only review found no substantive omissions. Follow-up review preserved advisory version wording, the general deprecated-alias rule, and conditional mypy scope. |
| Local instruction links | Checked 39 relative file and heading links across the root instructions and local skill documents. All resolve. Internal moves remain in the root file; no narrower automatic scope was introduced. |
| Documentation URLs | Inventoried 54 external links. ERLabPy page paths map to local `.md`, `.rst`, or `.ipynb` sources, except generated search and LLM exports. Export configuration is in `docs/source/conf.py:64,235-272`; search is a Sphinx output. Named tutorial and watcher anchors were checked. Published URL availability and every generated API fragment were not checked live. |
| Documented commands | Statically checked dependency groups, pytest/Ruff/mypy configuration, CI group CLI, docs Makefile, hook config, packaging workflow, and release workflow. This verifies definitions and stated prerequisites, not every command's successful execution on all platforms. |
| `uv run ruff format .` | Executed. It reformatted one unrelated contributor example. That exact formatter-only edit was removed to keep the requested scope. |
| `uv run ruff check --fix .` and `uv run ruff check` | Executed and passed. No application-code or test edits were retained. |
| Ruff format check on the four edited Markdown files | Executed and passed. Whole-repository format checking reports the pre-existing contributor-example formatting difference after its restoration. |
| `uv run python -m scripts.ci_test_groups --check-partition` | Executed and passed: each test file belongs to exactly one coverage group. |
| `uv run python -m scripts.ci_test_groups compat` | Executed and printed the compatibility smoke targets. This did not execute those tests. |
| `uv sync --all-extras --dev --group docs` | Executed successfully after an initial HTML attempt failed because `sphinxcontrib.mermaid` was missing. This synchronized the local `.venv`; no dependency configuration or lockfile change was retained. |
| `uv run --directory docs make html` | Retried after setup. It finished with problems and 16 warnings, including existing-output redirect collisions under `docs/build/html`. Exit status was nonzero. No clean-output build was run and no site/configuration repair was attempted. |
| `git diff --check` and final scope review | Passed. Only the root instructions, two skill reference documents, and this report are changed. |

The full pytest suite, dual-binding GUI tests, mypy, linkcheck, scientific notebook
execution, and packaging/release/deployment procedures were not run. No runtime or test
code, public API, CI grouping, top-level test module, or published documentation source
was changed. The conditional full-suite checklist did not apply. The linkcheck gate
applies before pushing large documentation changes; no push was requested or made.

Remaining limits: the site build did not pass; published links were checked against
local definitions rather than the live site; historical notes are not current
benchmarks; and the global installed skill remains out of date. No commit or push was
made. No instruction file outside this repository was edited.
