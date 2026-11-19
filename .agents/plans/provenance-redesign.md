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
main `9b62b1e3`. The Manager still uses the existing provenance system. The
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
`ImageToolEntry`. Extend these records in place. At present `Application.apply()`
asks the operation to calculate, and `Application.code()` asks it to write a
second description of the calculation. This is where the two paths can diverge.
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

For example, this is the proposed binding for the existing concatenation
operation. In this sketch, `Call` and `InputRef` are plain records. Constructing
them performs no calculation.

```python
def invocation(self):
    names = ("data", *self.input_slots())
    arrays = [InputRef(name) for name in names]
    return Call(function=xr.concat, args=(arrays,), kwargs=self.kwargs)
```

If the bindings are A₀, B₀, A₀, execution passes those three arrays to
`xr.concat` in order. Given notebook names `a` and `b`, Copy Full Code emits
`xr.concat([a, b, a], ...)` with the same keyword arguments. The ellipsis here
stands for the captured options. Name allocation, imports, and literal formatting
belong to the exporter. Validation, editor controls, and labels can continue to
use the existing typed fields. [Existing concat implementation](../../src/erlab/interactive/imagetool/_provenance/_operations/_analysis.py)

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
[Operation catalog](../../src/erlab/interactive/imagetool/_provenance/_operations) ·
[Operation registration and serialization](../../src/erlab/interactive/imagetool/_provenance/_model.py)

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
Paired timing and peak-memory measurements are pending for the new framework.
[Graph tests](../../tests/test_provenance_graph.py) ·
[Pending previews](../../tests/interactive/imagetool/manager/workspace/test_pending_preview.py) ·
[Workspace storage](../../tests/interactive/imagetool/manager/workspace/test_storage.py)

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
in CI. The rebased foundation passed 767 selected tests. Framework implementation
and conformance testing begin with the catalog and execution forms above.
