# ImageTool provenance redesign

## First milestone

Define and test the Qt-free contract for a provenance operation across the whole
catalog. A contributor should be able to read one operation definition and answer
three questions. Which inputs does it consume? What Python calculation does it
perform? Which DataArray does it return? Adding an operation should require no
changes to Refresh, Copy Full Code, window placement, or the graph traversal.
The same user workflows must stay as fast as the pre-redesign implementation
or get faster, with equal or lower peak and retained memory. Apply the
[performance criteria](#performance-and-large-workspaces) from the first
milestone onward.

Stage 1 (`7d3f7541`) and Stage 2 (`d7c4329b`) provide the revision graph on
main `9b62b1e3`. The first Manager integration covers **Interpolate → From
ImageTool**. Other producer paths still use the existing provenance system. The
Stage 3 prototype is preserved at `71ee41ee`. Its tests and console behavior
matrix can be recovered from that commit.

## The model

| Concept | What it means |
| --- | --- |
| Operation | A reusable calculation with named input slots and validated parameters. It knows how to describe the calculation without access to a window, revision ID, or notebook variable name. |
| Application | One use of an operation with its effective parameters, execution settings, and exact revision in each ordered input slot. The result revision points to this application as its producer. Each slot is one incoming edge. Two slots can use the same revision. |
| Recipe | The operation and the bindings for its **next** evaluation. A slot can follow a live entry, pin a revision, or rebuild an earlier producer. Evaluating the recipe produces a new application. |
| ImageTool entry | A stable window identity with a displayed revision, pending recipe, Auto setting, and tree placement. A revision becomes virtual when no entry displays it. It can still supply data through its producer or a checkpoint. |

This is the existing distinction between `Recipe`, `Application`, and
`ImageToolEntry`. Extend these records in place. For ported operations, `Application.apply()` and `Application.code()` use the
same call description and the settings recorded when the recipe was created.
[Graph records](../../src/erlab/_provenance/_records.py) ·
[Graph commands](../../src/erlab/_provenance/_graph.py)

## How to define an operation

For a built-in transform, the definition constructs **one call** to a public
function or method. Its arguments are ordinary typed values and explicit input
references. References can occur inside lists, tuples, and mappings. Captured
literals retain their types. A string parameter stays a string even if its text
matches a slot or notebook name. The operation type registered by installed
code determines the callable. The saved record contains its stable type and
fields. The call description is derived from those fields.

The concatenation operation builds its call from the ordered slots. Constructing
the call performs no calculation. [Call definitions](../../src/erlab/_provenance/_calls.py)

```python
def invocation(self) -> Call:
    slots = ("data", *self.input_slots())
    arrays = [InputRef(slot) for slot in slots]
    return Call.function(xr.concat, args=(arrays,), kwargs=self.kwargs)
```

If the bindings are A₀, B₀, A₀, execution passes those three arrays to
`xr.concat` in order. Given notebook names `a` and `b`, Copy Full Code emits
`xr.concat([a, b, a], ...)` with the same keyword arguments. The ellipsis here
stands for the captured options. Name allocation, imports, and literal formatting
belong to the exporter. Validation, editor controls, and labels can continue to
use the existing typed fields. [Existing concat implementation](../../src/erlab/_provenance/_operations/_analysis.py)

Keep the call description small. For a calculation that requires branching,
numerical guards, multiple statements, or selection from a tuple or Dataset,
use an ordinary Qt-free Python function that returns the intended DataArray.
Reuse an existing public API when it already owns the behavior. For example,
normalization's near-zero denominator rule belongs in one numerical function
called by both live execution and exported Python. For an existing public
setter, describe its receiver, attribute, and value once. The executor applies
it to an isolated result and the exporter can show the same assignment directly.
Other compound mutations can return the updated array from a public function.
Use a sequence of graph applications when intermediate results
are themselves user-visible or editable steps. An editor may group those steps.

**Porting an existing transform.** Keep its stable type name, settings schema,
input roles, authoring behavior, and any code-trust requirements. Replace its
numerical `apply` and handwritten Python emitter with the call definition. Port
its existing public-behavior examples to the shared test described below. A
future built-in of the same form supplies only its definition and fixtures.
[Operation catalog](../../src/erlab/_provenance/_operations) ·
[Operation registration and serialization](../../src/erlab/_provenance/_model.py)

### Python readability is an acceptance criterion

Centralize the meaning of inputs, execution, and publication in
`erlab._provenance`. Keep numerical algorithms in their natural domain modules
and UI choices in ordinary Qt code. Use explicit parameters, named functions,
normal imports, and small typed records where they clarify ownership. A few
similar lines of UI wiring are acceptable. Each abstraction must remove a real
decision that callers would otherwise have to repeat.

For built-in operations, reject expression strings passed as arguments, overloaded
operators used to construct a hidden language, caller-frame inspection, runtime
monkey-patching, and generating Python text to execute the live calculation.
Stored user programs have their own explicit execution and trust boundary.
If a port needs branching, write a normal Python function with that branching.
If a backend has a genuinely different calling convention, use a small named
adapter with a direct call.

Review a representative port and its emitted notebook Python next to the
equivalent handwritten calculation. A reviewer must be able to follow the inputs,
the actual callable, its result, and its publication using ordinary Python
control flow. The exported code should be something a user could write with the
public API. A port that needs a private language or obscures these steps fails
review even if the tests pass. [Repository code contracts](../../AGENTS.md#code-contracts) ·
[Generated-code conventions](../generated-code.md)

**Imports.** Use explicit absolute imports in implementation and test `.py`
files touched by the rewrite. Prefer `import erlab` for public APIs exposed
through `lazy_loader`, and resolve the qualified callable when the operation
needs it. For private dependencies, import from the owning module by its full
name. Keep type-only dependencies under `typing.TYPE_CHECKING` where appropriate.
Qt adapters import top-level modules from `qtpy`, such as
`from qtpy import QtCore, QtWidgets`. The provenance core must import and run
with Qt blocked. Preserve Ruff's import sorting and the cold-import tests.
[Repository import conventions](../../AGENTS.md#code-contracts) ·
[Qt code standards](../../docs/source/contributing/development.md#code-standards)

Run `prek run absolute-package-imports --all-files` for package code. Run
`uv run ruff check --select TID252 --config 'lint.flake8-tidy-imports.ban-relative-imports="all"'`
with the affected rewrite test file paths as arguments. This also catches
single-dot sibling imports. The `__init__.pyi` files read by
`lazy_loader.attach_stub` use its required relative export declarations.

## Where inputs and execution come from

- **Arrays and files.** An incoming array establishes a source revision. A file
  node and load edge record the file identity, loader, arguments, and selected
  output, including a selection from a Dataset, DataTree, or sequence. Reload
  can reread the origin. Historical replay uses the recorded file version or an
  identified checkpoint. A value with incomplete history remains usable with its
  known origin visible.
- **Stored programs and extensions.** The recorded program text or identified
  routine is already the calculation. Give it explicit input bindings and an
  output. Retain authorization for stored code, fit expressions, and extension
  loaders. Exporting describes the program without executing it. Use the same
  program or versioned routine when the calculation actually runs.
  Bind the execution settings before module initialization, on re-use of a
  worker, and when a deferred task runs. Surface an unavailable dependency and
  the available recovery action. [Execution](../../src/erlab/interactive/imagetool/_provenance/_execution.py) · [Code trust](../../src/erlab/interactive/imagetool/_provenance/_trust.py)
- **Ownership and publication.** Protect a mutable source before its producer
  can reuse the buffer. Preserve stable lazy backing and validate recorded file
  versions when chunks are read. Read-only operations can share owned inputs.
  Isolate every operand a calculation can mutate, including its mutable
  metadata. Preserve repeated references to the same operand within that call.
  Capture operands and settings
  before dispatch. Evaluate once, verify the affected revisions at commit, then
  install the result and its application together. A failed, cancelled, or
  superseded installation preserves the previous result, history, and followers.
  These rules apply equally to a synchronous method and a worker result.

## Framework acceptance

1. **Account for the catalog.** On the rebased foundation there are 50 built-in
   operation types plus the internal `source_view` marker in the loaded registry.
   Check the registry and the separate source, setup, script, and replay dispatch
   paths. For each entry, identify the public call, existing program or source
   boundary, or presentation role that preserves its behavior. Record any case
   that needs another mechanism and why. Derive this coverage check from the
   registry so a future operation makes an omission visible.
2. **Exercise the same interface across different behaviors.** Build a small
   parameterized suite with real representatives of a method, a function with
   ordered or repeated inputs, a numerical helper, a mutating operation, and a
   fit or other calculation with a selected output. Cover file ingress and an
   authorized program or extension through their own execution boundaries. For
   each replayable case, start from the saved typed record, run the calculation,
   and execute its emitted Python on independent inputs without Qt or a Manager.
   Compare both results with an independently specified expectation, including
   dtype, dimensions, coordinates, values, name, and attributes.
3. **Test the boundaries that previously disagreed.** Include float32 cancelling
   offsets, guarded normalization, an extension that reads settings at import,
   reuse under a different setting, and deferred computation after ambient
   settings change. Check unchanged source buffers and metadata. Count reads and
   copies with a lazy input when only its coordinates are required. Stage 2's
   eager source checkpoint and per-operand copies are known starting points.
4. **Demonstrate the cost of a port.** Add a further existing operation using
   only its binding to the public calculation and its fixture. Review its diff.
   Its executor, notebook exporter, graph, and Manager command paths must require
   zero operation-specific branches. Show which former execution and code
   generation methods the port deletes. Complete this milestone when each
   execution form has evidence and every catalog entry has a porting route.
   Verify the affected workloads below before closing the milestone. Track
   the remaining production ports separately.

### Implemented paths

- The catalog check covers 50 operations and `source_view`. The [call
  definition](../../src/erlab/_provenance/_calls.py) supplies live execution
  and notebook export for the ported built-ins. Transpose adds one method
  binding to the existing operation. It removes its separate `apply` and
  `expression_code` implementations.
- The [headless parity tests](../../tests/test_provenance_graph.py) restore
  typed records, execute copied Python, and compare both results with an
  independent expectation. They cover rename, transpose, ordered concat,
  affine coordinates, guarded normalization, an accessor setter, both outputs
  of mesh removal, and a selected variable from a file. Mesh removal preserves
  the public tuple unpacking and chooses the recorded array as the result.
- The [Manager integration](../../src/erlab/interactive/imagetool/manager/_operation_graph.py)
  resolves both interpolation inputs from their selected entries. Child,
  top-level, and replacement displays use the same calculation. The historical
  editor and Copy Full Code use the displayed application. Refresh uses the
  current bindings. The Manager tests also exercise a rejected installation
  and release of the last owner.
- Copied Python for a versioned file calls the public
  [file reader](../../src/erlab/io/utils.py) with the recorded loader, selected
  array and local filesystem identity. Replacing the file after copying the
  code raises an error on the next read. A separate machine or a multi-file
  source without a complete version needs an independent named checkpoint.
- A mutable callback receives one isolated array per distinct input revision.
  It can edit object-valued index members and nested coordinate attributes or
  encoding without changing the source or an existing sibling result.
- The extension tests exercise module initialization, a second submission with
  different settings, deferred computation, and notebook export. The worker
  implementation is on the local `dev-provenance-extension-candidate` branch.
  Its measured time and memory costs keep this milestone open.

### Verification of the integration

The full PyQt6 suite passed 11,778 tests with 14 skips on the integrated
code at `e5ae4297`. This includes the corrected legacy Dataset selection,
historical file reads, callback ownership and deferred interpolation chooser.
The chooser and related workspace checks passed 73 tests under each binding.
The file-copy and extension-loader regressions also passed under PySide6.

The experimental branch passed 919 extension, graph, replay and clipboard
tests under each Qt binding. A minimal frozen macOS arm64 executable used the
Manager startup sequence and reproduced both an eager result and three deferred
chunks in a child process. The full standalone GUI bundle and other operating
systems still need separate checks.

After the file and chooser changes, 860 experimental extension, graph, replay
and I/O tests passed under PyQt6. All 81 graph tests then passed with the
final numeric ownership optimization. The resulting combined selection passed
861 tests under PySide6. Both branches passed Ruff, mypy, import checks, the
test partition check and the HTML documentation build.

### Operation and replay inventory

The [porting inventory](../../tests/interactive/imagetool/provenance_ports.csv)
identifies the calculation or boundary for each registered type. The
`test_porting_routes_cover_the_registered_operations` test checks the inventory
against the loaded registry. A porting route describes work to do. Its numerical,
ownership, and replay tests establish when it works.

The registry excludes some decisions made by the [replay compiler](../../src/erlab/interactive/imagetool/_provenance/_graph.py)
and [replay executor](../../src/erlab/interactive/imagetool/_provenance/_execution.py).

| Existing replay input or node | Destination and evidence required |
| --- | --- |
| `full_data`, `public_data`, `selection` | Resolve the selected source view and its preparation steps when capturing the input. Test a filtered, transposed selection against its expected coordinates. |
| `file_load` | Preserve the loader kind, file version, arguments, selection, and optional cast. Exercise the ERLab loader, local callable, and extension loader. Cover Dataset and DataTree variables, sequence items, and old positional selections. |
| `caller_input`, `external_input`, `live_input` | Supply or request an operand at its declared boundary. Capture the revision before evaluation. Check an unavailable input, two names for one operand, and an editable caller buffer. |
| `operation`, `script` | Use the catalog binding or the authorized stored program, with the ordered inputs and named result. Check the recorded settings, selected output, and denied execution. |
| `setup`, `relay`, `source_view` | Preserve setup information and pass-through relationships during import and export. Check the notebook prologue and the visible source without creating an extra numerical result. |
| Schema-v2 parent context | Recover the saved parent revision or its checkpoint for operations that need it. Test a legacy coordinate operation with distinct parent and current data. |

## Performance and large workspaces

**Baseline.** Compare each migrated path with the same user action on
pre-redesign main (`9b62b1e3` for this branch). Use the same hardware, dependency
versions, backend, arrays, chunking, display settings, and retained history.
Verify the final arrays and completed updates in both runs. For new graph
behavior without a previous equivalent, measure scaling and operation counts.
Record both commits, the workload and the command so the comparison can be rerun.

| Workload | Acceptance |
| --- | --- |
| Many entries, small arrays | Grow a workspace from 1,000 to 10,000 entries, including pending rows, long chains, wide fan-out, and branches with shared ancestors. Hold the affected branch fixed while adding unrelated rows. A local edit, stale-state query, or Refresh should do work proportional to the dependencies it needs. Count entries and edges visited, numerical calls, and rows materialized. Each required shared producer runs once per request. Inspecting one row must leave unrelated pending windows unloaded. |
| Few entries, large arrays | Exercise eager volumes and chunked, file-backed data whose logical size exceeds RAM. Inspect metadata, preview a command, take a plane, and publish a derived result. Count chunks read and full-buffer copies, including coordinates. Metadata queries read zero data chunks. A slice or coordinate-only operand reads only the chunks it needs. Charge viewer rendering separately. Read-only paths reuse owned buffers. Include the cost of protecting an incoming mutable buffer that the producer immediately overwrites. |
| Interactive updates | During a growing acquisition grid, move the cursor while new frames arrive, with filtering and Auto both enabled and disabled. Measure throughput, end-to-end command latency, and cursor-event gaps through capture, calculation, and publication. Report the median, tail, and worst observed gap as well as the final displayed result. Coalesce superseded work and bound the queued frames and worker payloads. |
| Persistence and lifetime | Measure save and reopen time and peak memory with shared data and pending rows. Whole-workspace work should scale with records, edges, and unique payloads processed. Count shared buffers and stored payloads once. Record reachable stored bytes and physical file size after compaction separately. |

**Leak check.** In one long-lived process, hold the surviving windows, undo
history, pins, and active readers fixed. Repeat create → replace → refresh →
close and save → reopen cycles, including failure, cancellation, and superseded
worker results. After pending work drains, count live revisions, unique payload
bytes, queued jobs, open handles, and caches. They must settle with a fixed
workload. Then release the last owner and verify that representative arrays,
windows, and workers can be collected and file handles close. This catches a
leak even when a small test leaves process memory apparently flat.

**Comparison and sign-off.** Keep deterministic visit, copy, read, and lifetime
checks in the test suite. For timing and process memory, alternate repeated
baseline and candidate runs in fresh processes with the same warm-up. Measure
peak and settled process memory across the Manager and its workers as well as
live payload bytes. Record the process-memory metric and how shared pages were
counted. Repeat an inconclusive measurement and report the variation.
A reproducible slowdown or increase in peak or retained memory keeps the
affected milestone open. A gain in one metric cannot offset a regression in
another. Use the affected workloads for each port and the complete set before
switching the default implementation.

The foundation already tests a 1,100-entry chain and diamond, lazy status queries,
Auto coalescing, and release of retained revisions. Extend these checks through
the Manager and workspace so they also catch hidden work in the UI and storage.
On macOS arm64, eight alternating pairs on a 1,024 × 1,024 interpolation
gave 177.63 ms on main and 178.28 ms on the integration snapshot. Paired changes
ranged from −13.4% to +9.2%. Eight pairs on a 2,048 × 512 input gave 111.82 ms
and 114.52 ms. Both stores together retained two unique buffers, totaling
10 MiB, after each result window closed in the latter workload. Timing varied
between processes. A fixed one-edge refresh took about 0.17 ms with either
1,000 or 10,000 unrelated entries. This measurement covers graph evaluation.
A further eight pairs measured the macOS physical footprint, including compressed
pages. The median paired differences were +0.20 MiB for both peak and settled
memory and −0.03% for latency. The two versions retained the same six arrays
after window close in that workload.

The live Rename action measured 0.958 ms on main and 0.995 ms on the candidate
across eight process pairs. Paired changes ranged from −13.6% to +14.1%.
Direct replay of the small method had a measurable cost, 1.37 µs on main and
1.52 µs with the general positional method binding. Transpose measured 32.12 µs
and 31.76 µs. These direct measurements exclude window publication.

The extension's isolated interpreter measured higher first-use latency and
peak memory on the same Manager workload. Keep its performance work open.
[Graph tests](../../tests/test_provenance_graph.py) ·
[Pending previews](../../tests/interactive/imagetool/manager/workspace/test_pending_preview.py) ·
[Workspace storage](../../tests/interactive/imagetool/manager/workspace/test_storage.py)

### Pending windows and live inputs

A real workspace test on the ordinary integration used 1,000 and 10,000 pending
rows and one active interpolation. Each of four Refresh commands ran the
interpolation once, produced the expected data, and opened zero unrelated
datasets. Opening details for a pending row read its coordinates and its
16 × 16 preview. The other rows stayed unloaded.

The interpolation chooser reads coordinates on demand. Rows that reference the
same saved payload and equivalent displayed coordinates share one eligibility
check per pass. With 10,000 differently named, unfiltered rows referencing one
payload, opening the popup took about 0.18 s and read one coordinate pair.
The original chooser exceeded 70 s for 10,000 shared rows. With 1,000 distinct
files, constructing the dialog took 24 ms and opening the popup took 6.42 s.
The popup read each coordinate pair, skipped the signal arrays and retained
zero datasets after the pass. The selected row is checked again before use.

Replacing both inputs during cursor activity exposed a second conversion to
public array layout during graph capture. The Manager now reuses the array
from the source-change signal and applies the same ownership and metadata
rules. Across 24 replacements it removed 24 conversions under each Qt binding.
Both versions still made the 24 copies needed to protect those input revisions.

Historical file replay now checks the version during notebook execution.
Against the unguarded export at `d86aa722`, six paired runs gave a median
elapsed difference of −0.02 ms for eager interpolation and +0.13 ms for
coordinate-only interpolation. Reading one 8 KiB tile cost 0.47 ms more.
Reading one cell cost 0.56 ms more and requested its 8 KiB tile from the
backend. The old code requested eight logical bytes. The largest increase
in traced peak allocation across the four workloads was 28 KiB. After closing
the result, the paired difference was at most 0.3 KiB. The full test suite
was active during these measurements. Two additional pairs used the actual
copied code from pre-redesign main at `9b62b1e3`. Both versions produced the
same result bytes. The main-branch eager export needed the notebook grid bound
as `target`. The median tile read took 6.36 ms on main and 7.31 ms on the
integration. Traced peak memory fell by about 1 MiB for both interpolation
workloads and rose by about 16 KiB for the selected tile.

The callback ownership benchmark compared mutable and numeric indexes at
`d86aa722` and `e5ae4297`.
With 32 mutable Python index labels, a correct derivation took 309 µs and
peaked at 31,914 traced bytes. The earlier code took 137 µs and 13,192 bytes
but changed the source and a sibling. Copying the 32 label objects alone
peaked at 14,026 bytes in the same Python runtime. Rebuilding the axis with
the public xarray constructor used more time in all four paired processes.
For three plain numeric workloads, peak, published and post-close allocation
counts matched the earlier code. The measured CPU time ranges overlapped.
Pre-redesign main has no corresponding graph command. Against the Stage 2
graph at `d7c4329b`, the integrated numeric derivations took about 9–13 µs
more CPU and 0.9–1.0 KiB more peak allocation. The two implementations each
retained 64 traced bytes after closing the complete graph. Stage 2 also let
the mutable callback change the caller's original index.

### Extension measurements

On macOS arm64 with Python 3.14.3 and PyQt6, the benchmark queued one registered
routine that took the mean of a 256 × 256 float64 array. It checked each result.
Three fresh-process pairs ran one cold job and 25 warm jobs each. Run
`PYTHONPATH=<checkout>/src python bench_extension_worker.py <label>` for baseline
`9b62b1e3` and the candidate based on `0771e1d3` (extension source digest
`4c2336a8f473`). The table gives the median across the three process runs.
Memory is sampled RSS, summed across the caller and its children every 20 ms.
Shared resident pages can count once per process.

| Worker measurement | Baseline | Candidate |
| --- | ---: | ---: |
| First job | 4.07 ms | 944 ms |
| Warm median | 1.67 ms | 2.43 ms |
| Warm 90th percentile | 2.01 ms | 3.28 ms |
| Peak RSS | 304 MiB | 518 MiB |
| RSS increase after last owner closes | 0.43 MiB | 0.45 MiB |

Warm medians ranged from 1.48 to 1.70 ms on the baseline and 2.22 to 2.64 ms
on the candidate. One slice read one source chunk. A shared reduction read all four source chunks
once each on both paths. The private interpreter preserves xarray settings
through script imports and deferred callbacks. The worker culls its most recent
graph before sending the selected tasks. After worker replacement, the first
request sends the saved graph to the new interpreter. Changing the settings
starts a new interpreter because imported helpers can cache them. The time and
peak memory regressions keep the extension gate open. A read-only memory map
reduced the warm transfer time for a 32 MiB operand from 12.3 ms to 0.71 ms,
but required a 32 MiB file and 282 ms on the first transfer. It does not resolve
the interpreter cost.

The experimental worker now starts one idle timer across 26 consecutive jobs.
Five alternating runs measured warm median times of 1.60 ms on main and
2.31 ms with this worker. The first job took 3.72 ms and 928 ms, respectively.
macOS also measured the physical footprint of the actual Manager worker group.
The active worker added about 153 MiB. The baseline added about 0.3 MiB.
Both released that memory after the last owner closed.

A separately spawned, single-threaded scientific interpreter on macOS could
fork a clean child before importing a user routine. This reduced the median
time to change settings from 906 ms to 242 ms. The first result took 870 ms,
and the template retained about 140 MiB after its child exited. NumPy could
not start in an isolated CPython subinterpreter on this installation.

## Finding and retiring the old paths

The operation registry covers *what* can run. To find *who else decides*,
trace the code in both directions during the framework milestone. Use that
inventory to check the command and query interfaces before migrating callers.

1. Start at user and API entry points. Follow actions, Qt signals, notebook and
   server ingress, ToolWindow and worker callbacks, workspace restore, and
   clipboard and code export through to the displayed or returned result.
   Trace read paths too, including enabled states, badges, and historical editors.
2. Work backward from where managed data is installed or replaced, dependencies
   are chosen, specs are read or written, and code is replayed or emitted.
   Search the repository beyond the provenance packages. Follow delegates and
   plugin callbacks to their callers. This direction also catches an action that
   changes an array without mentioning provenance.
3. Reconcile the two lists. For each distinct route, record the entry point,
   the decision and its current owner, its destination in the new design, and
   the public-behavior test. Group aliases only after verifying that they reach
   the same decision. Keep this short migration map in the plan or change under
   review. It is a work list and has no role in runtime dispatch.

The rebased foundation contains these two examples of independent decisions.

| Route | Current decision | Evidence needed to retire it |
| --- | --- | --- |
| Reload from an ImageTool or from the Manager | The viewer also checks availability and can choose a direct loader or its local provenance. | For a managed row, exercise both actions with a viable but conflicting local loader. Check that each action uses the graph-selected origin once and explains unavailability consistently. Exercise standalone Reload separately. [Viewer reload paths](../../src/erlab/interactive/imagetool/viewer.py) |
| Paste Provenance Steps | The edit controller branches on tree parent and displayed source spec to choose how to apply the pasted step. | Paste the same metadata edit into stale child and top-level results. Both must use the old displayed values, copy those values, and refresh from the new binding. [Paste controller](../../src/erlab/interactive/imagetool/manager/_provenance_edit/_controller.py) |

The Manager tree's Rename editor also calculates outside the operation record.
The [window wrapper](../../src/erlab/interactive/imagetool/manager/_wrapper.py)
renames the displayed, slicer, and accepted filter arrays, then records the new
name. A future port must keep those arrays and the copied result in agreement
when a filter is active. The existing editor remains in use for this milestone.

The following additional routes connect user entry points to the writers and
readers on the rebased foundation. The last column gives the observable test for
moving each decision. When a path has both managed and standalone callers, keep
the standalone case in the test.

| Entry point and present owner | Destination | Public behavior to test |
| --- | --- | --- |
| [Single transform](../../src/erlab/interactive/imagetool/dialogs.py), `DataTransformDialog.accept`, calls `process_data` and composes separate nested and detached specs. The [concat dialog](../../src/erlab/interactive/imagetool/manager/_dialogs.py) selects multiple rows and also builds a spec at publication. | Submit the operation and ordered operands before evaluating. Place its result through an entry command. | Apply the same transform in three placements. Concatenate A, B, A and execute the copied Python. |
| [Plot selection and associated coordinates](../../src/erlab/interactive/imagetool/plot_items.py), `open_in_new_window` passes current plot data and a separately built source spec. `open_associated_coord` applies a spec to the parent. | Submit the cursor selection, crop, binning, and view operations with the selected revision. | Open a transposed, binned profile and its coordinate. Move the cursor and execute each historical result’s copied code. |
| [Batch transform and filter](../../src/erlab/interactive/imagetool/manager/_actions.py), `apply_batch_transform_dialog` and `apply_batch_filter_dialog`, preflight each row and stage results. | Resolve each target's operands and validate the group before committing. | An invalid last target leaves the earlier targets and their history intact. Count evaluations on success. |
| [File Open in a viewer](../../src/erlab/interactive/imagetool/_mainwindow.py), `_open_file`, loads and selects an array before replacing the source. [Manager ingress](../../src/erlab/interactive/imagetool/manager/_io.py), `receive_data`, combines loader selections and preparation operations. | Register a file source and its selected output, or capture an incoming array as a source revision. | Open a multi-variable file through both actions. Replay the chosen variable after another variable changes order. |
| [Server and watched replacements](../../src/erlab/interactive/imagetool/manager/_actions.py), `_data_replace` and `_data_watched_update`, choose which metadata assignments to keep. The [viewer](../../src/erlab/interactive/imagetool/viewer.py) applies a display filter to replacement data. | Capture the incoming revision, applicable metadata edits, and filter separately. | Replace a watched array with a growing scan while slicing. Verify the displayed plane, filter output, old source, and accepted update count. |
| [Acquisition context](../../src/erlab/interactive/imagetool/manager/_acquisition_context.py), `apply_to_file_data`, applies accepted metadata assignments at file ingress. | Save the effective assignments with the selected file revision. | Load a file with an existing scalar coordinate. Check the keep and replace choices, their order, and the replayed result. |
| [Console](../../src/erlab/interactive/imagetool/manager/_console.py) handles both recorded calls and direct `slicer_area.replace_source_data` on assignment. | Bind a console calculation to its input revisions. Publish an array assignment as a source and an item assignment as an edit. | Overwrite a caller buffer, then inspect, show, assign, and copy a previously calculated result. |
| [Metadata table](../../src/erlab/interactive/imagetool/manager/_metadata_editor.py), `_plan_target_edit`, chooses a direct patch or validates a historical edit. [Historical editor](../../src/erlab/interactive/imagetool/manager/_provenance_edit/_controller.py), `_apply_validated_edit`, installs the result according to scope. | Edit the selected application on its branch. Commit the new data and history together. | Edit a stale child and a top-level result. Check the original coordinates, the next refresh, an unchanged sibling, and a rejected edit. |
| [ToolWindow output](../../src/erlab/interactive/utils.py), `_open_output_imagetool`, accepts calculated data and a spec, then resolves a named or detached result. [Figure Composer](../../src/erlab/interactive/imagetool/manager/_figurecomposer/_controller.py) refreshes a referenced source separately. | A tool authors a durable recipe. A figure consumes a reference for display. | Reuse a named output, close its editor, then refresh it. Reopen a figure that references an ftool parameter directly. |
| [Extension completion](../../src/erlab/interactive/imagetool/manager/_extensions/_execution.py), `_insert_if_current`, constructs the operation after the worker returns. | Capture inputs, version, settings, and parameters at submission. Verify the result is still current before publication. | Change settings across two jobs on one worker. Compute a deferred output later. Supersede a running job and check that it publishes nothing. |
| [Manager lineage](../../src/erlab/interactive/imagetool/manager/_lineage.py) plans live or recorded input resolution. The [dependency index](../../src/erlab/interactive/imagetool/manager/_dependency.py) reads specs and tool inputs to derive stale status and schedule updates. | Graph queries read applications for displayed data and recipes for future updates. | Stop Auto at a stale prerequisite, resume it, and count shared ancestors evaluated in a diamond. Check the badge and enabled actions at each step. |
| [Details and Copy Full Code](../../src/erlab/interactive/imagetool/manager/_details_panel.py), `_copy_full_derivation_code`, can walk tree parents to find a watched name. | Resolve the displayed application's historical operands. Assign notebook names at export. | Change the source with Auto off. Execute copied code on independent inputs for both a child and a top-level result. |
| [Duplicate](../../src/erlab/interactive/imagetool/manager/_actions.py), `_duplicate_subtree`, copies a managed subtree. [Workspace save](../../src/erlab/interactive/imagetool/manager/_workspace/_saving.py) writes per-row specs and [restore](../../src/erlab/interactive/imagetool/manager/_workspace/_loading.py) rebuilds source and displayed state for loaded and pending rows. | Duplicate entries with explicit bindings. Restore shared revisions before attaching windows. Import older specs once. | Duplicate a stale branch and refresh only the copy. Save and reopen with a pending row, then inspect both histories without loading unrelated arrays. |

The reverse trace starts at the wrapper's [data and history writers](../../src/erlab/interactive/imagetool/manager/_wrapper.py),
especially `_replace_imagetool_data`, `replace_with_detached_data`, and
`set_displayed_provenance`. Follow the [legacy revision store](../../src/erlab/interactive/imagetool/manager/_revisions.py)
when a wrapper binds a checkpoint, stores a metadata edit, or releases unreachable
revisions. Compare retained history and checkpoint lifetime across close and reopen.
Also trace `replace_source_data` and `set_data`
directly on the viewer. These reach console assignment and viewer File Open,
which can bypass a wrapper writer. Repeat both searches when switching each
route. Existing [Manager tests](../../tests/interactive/imagetool/manager)
cover the legacy behavior and supply fixtures for these comparisons.

For each migrated route, drive its real entry point with the legacy decision
disabled or contradicted *before* the action. Assert the expected data, recorded
inputs, and next bindings. Delete the old decision or name the remaining
standalone, import, or export consumer. Extend the existing dependency tests
with a narrow check for retired legacy entry points so a new caller cannot
quietly reintroduce them. Before switching the default, repeat the reverse
search and account for every remaining legacy decision and managed writer.
[Existing module-boundary tests](../../tests/interactive/imagetool/test_module_split.py)

## Manager integration

Dialogs, console calls, ToolWindows, batch actions, and paste submit the same
operation and operand choices. The Manager resolves entries to revisions once.
The graph supplies the evaluated result to each placement. Child and Top-Level
create identical lineage with different display parents. Replace retains the
entry and its earlier revision. Graph queries drive the inspector, editors, and
availability of actions. A migrated writer may export a projection for a legacy
reader. An existing row crosses through explicit import once. A saved spec or
tree parent cannot independently choose the input of a migrated command.

Use the original three-entry example as the integration test. Set A₀ = [10, 20],
B₀ = [1, 2], and C₀ = 2 × B₀. Turn Auto off for C.

| Action | B displays | C displays | Inputs used by C | Input for next update |
| --- | --- | --- | --- | --- |
| Create C | [1, 2] | [2, 4] | B₀ | B₀ |
| Replace B with B + A | [11, 22] | [2, 4] | B₀ | B₁ |
| Refresh C | [11, 22] | [22, 44] | B₁ | B₁ |

Drive actual Manager actions. While C is stale, its editor and executed Copy Full
Code use B₀. The separate **Inputs for next update** control shows B₁ and Auto.
Overwriting the caller's B₀ buffer leaves C₀ reproducible. Refresh evaluates
the current bindings. Reload rebuilds their origins and respects pins. Shared
producers run once per request. Auto waits at a stale prerequisite with Auto off.
A failed update leaves C's displayed revision available. Include a conflicting
legacy spec before registration, and verify the selected inputs in every placement.
Interpolation's existing two-input tests are one integration fixture. Apply the
same assertions to the other producer paths. [Foundation acceptance tests](../../tests/test_provenance_graph.py)

## Remaining work

| Milestone | Observable completion |
| --- | --- |
| Port the existing writers and readers | Convert operations by common call form and route producers through the shared submission path. Retire their independent source selection, precomputation, and emitters. Preserve file and watched ingress, display filters and transpose, rejected updates, batch rollback, metadata edits, and clipboard behavior. Grouped historical edits restore the complete group from any member and affect only the selected branch. |
| Console and ToolWindows | A console window handle is live, a calculated result retains its inputs, and `.data` is an independent editable snapshot. Item assignment records an edit. Assigning a plain array starts a checkpoint. Assigning a recorded result retains provenance. Preserve direct `xr` and `era` calls, repeated operands, and `tools.apply`. Use the console matrix in the prototype plan, lines 233–289. A ToolWindow edits durable output recipes. In ktool, a changed conversion offset can make an output stale and restoring it can clear the cause. Display overlays leave it current. Exercise editor close and out-of-order fit completion. Figure Composer can reference ftool directly without adding a data edge. |
| Native workspace | Save shared graph records once and restore the graph before the windows. Reopen a workspace while C is stale and verify both bindings, Auto, copied Python, and Refresh. Repeat a grouped edit and run the persistence and lifetime comparisons above. |
| Legacy migration | After the native format round-trips, import representative old workspaces. Recover evidenced file origins, operations, bindings, placement, and checkpoints. Preserve available descriptions where only a value survives. Verify trust and legacy fixtures before switching the default reader and writer. |

Use one Qt binding locally and exercise binding-sensitive paths on both bindings
in CI. The rebased foundation passed 767 selected tests. The
[integration results](#verification-of-the-integration) give the current checks.
