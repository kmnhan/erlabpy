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

For example, this is the entire proposed numerical definition for the existing
concatenation operation. `call` and `slot` illustrate the interface.

```python
def invocation(self):
    arrays = [slot("data"), *(slot(name) for name in self.input_slots())]
    return call(xr.concat, arrays, **self.kwargs)
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
