# ImageTool provenance redesign

## Starting point

The next implementation is **Interpolate → From ImageTool** through the Manager.
One operation definition must suffice for all three result placements, historical
replay, Copy Full Code, and Refresh. Review that workflow before migrating a
second action.

The foundation on this branch is Stage 1 (`7d3f7541`) and Stage 2 (`d7c4329b`),
rebased onto main `9b62b1e3`. It includes a Qt-free revision graph, ordered input
bindings, Replace, Refresh, Reload, Auto, and a historical code adapter. The
live Manager on this base still uses the existing provenance system.
[Graph and commands](../../src/erlab/_provenance/_graph.py) ·
[Foundation tests](../../tests/test_provenance_graph.py)

The preliminary Stage 3 commit `71ee41ee` is preserved on
`archive/provenance-stage3-71ee41ee`. Consult its tests
for historical edits, console semantics, and rollback as each workflow is
migrated. Its plan and validation log are available with
`git show 71ee41ee:.agents/plans/provenance-redesign.md`. The implementation
milestones below supersede that log.

## Required behavior

A data node identifies one revision of a DataArray. An ImageTool entry identifies
one window across replacements. Each named input slot of an applied operation is
one directed edge into the output revision. Slots have an order, and two slots
can refer to the same revision. A file node identifies a path. Its outgoing load
edge records the loader, arguments, and selected data.

- **File load and Replace.** Loading F and replacing its result gives
  `F → A₀ → A₁`. One window displays A₁. A₀ becomes virtual and its load step
  remains visible in the history. A virtual notebook source retains the original
  data when the caller can overwrite its buffer.
- **Two inputs and Replace.** For `B₁ = B₀ + A₀`, the same B entry moves to B₁.
  One application records both B₀ and A₀. Opening a child or top-level result
  instead creates a new entry with those same inputs.
- **Stale descendant.** Start with A₀ = [10, 20], B₀ = [1, 2], and C₀ = 2 × B₀.
  Turn Auto off for C. The following is the acceptance example for graph queries,
  the Manager UI, and later the native workspace format.

| Action | B displays | C displays | Inputs used by C | Input for next update |
| --- | --- | --- | --- | --- |
| Create C | [1, 2] | [2, 4] | B₀ | B₀ |
| Replace B with B + A | [11, 22] | [2, 4] | B₀ | B₁ |
| Refresh C | [11, 22] | [22, 44] | B₁ | B₁ |

**Inputs used** and Copy Full Code describe the selected, displayed revision.
A stale result shows **Inputs for next update** separately, plus its Auto state.
Refresh follows the entry's current bindings. Reload follows those bindings back
to their origins, rereads requested files, and honors pinned values. A shared
producer runs once per request. Auto waits at a stale prerequisite with Auto off.
An explicit Refresh can update those prerequisites. A missing input or file
identifies the affected source and the action available to the user.

## Implementation decisions

| Owner | Decision |
| --- | --- |
| Operation definition | Produce one description of a public numerical call with ordered operand roles, effective arguments, and execution settings. Runtime resolves its operands to arrays and invokes the call. Notebook generation renders the same call with bound Python names. |
| Applied operation | Bind each operand to the exact revision consumed. Retain those bindings for the lifetime of the result. Historical replay and Copy Full Code traverse these edges. |
| ImageTool entry | Own the displayed revision, pending recipe, Auto, and placement. A pending input follows an entry, pins a revision, or rebuilds an earlier producer. Replacing a derived B with smooth(B) can therefore rebuild its original crop when the crop's source changes. |
| Manager and Qt | Capture the user's operand and parameter choices once. Ask the graph to evaluate and publish. Render graph queries for rows, input roles, badges, editors, and command availability. Display parent and order come from entry placement. Window and ToolWindow registries provide widget identity and lifetime. |

Start with the smallest call representation needed by interpolation. Use an
existing public xarray or ERLab operation when it implements the full semantics.
Where validation, sorting, or numerical ordering needs extra logic, put that
logic in one Qt-free public numerical function and use the same function from
both paths. Notebook output must run on named inputs in a plain Python process.
The present `Application.apply()` and `Application.code()` delegate to separately
written implementations. Reusing that pair as-is leaves the original source of
disagreement in place. [Applied operation](../../src/erlab/_provenance/_records.py)

A live calculation accepts the selected input revisions, their next bindings,
and the operation. It produces the value that the window installs. A source
command accepts a file or external array. An explicit import can accept an
existing value with its saved history. Keep those interfaces distinct. Capture
before asynchronous work. Recheck the affected revisions and destination before
commit. A rejected, cancelled, or superseded installation preserves the last
successful graph and window state, including dependent states and undo history.

The existing format can serve an unmigrated reader through an explicit projection
at its boundary. For a migrated workflow, trace each remaining legacy call to its
specific consumer. Its saved recipe cannot choose an operand or recreate history
for a live command. A closure or rename must preserve access to a recorded input
through its revision, a verified file, or an identified checkpoint.

During the transition, register each source entry once. A public replacement
or watched update advances that same entry. When an existing result first
crosses from the old system, import its evidenced lineage once. If only the
value survives, identify that checkpoint in the inspector. Subsequent graph-owned
updates advance the entry directly. Reconstructing it from a window spec on
selection or Refresh would give the same result two owners.

## First milestone — Interpolate from another ImageTool

1. **Characterize the current action.** Use a pair of named 2-D arrays with
   different coordinates and an analytic source, for example
   `B(y, x) = 10*y + 2*x` on x = [0, 1, 2], y = [0, 1]. Give A the same y
   and x = [0.5, 1.5]. Linear interpolation yields [[1, 3], [11, 13]].
   Drive the actual dialog. Keep the existing tests for all matching dimensions,
   selected dimensions, linear/nearest, filtered input, coordinate validation,
   and the sorted-source memory optimization. Run these assertions on the base
   before switching the writer. Record any pre-existing failures.
   [Manager cases](../../tests/interactive/imagetool/manager/test_mainwindow.py) ·
   [Interpolation kernels and code tests](../../tests/interactive/imagetool/test_provenance.py)
2. **Define and bind the calculation once.** B supplies the displayed `data`
   slot. A supplies the displayed `target` slot. The target grid comes from A's
   coordinates. Store the choice of all or selected dimensions, the method,
   and effective settings in the operation. Both execution and exported code
   must preserve the behavior of the public calculation, including axis order,
   labeled coordinates, the result name, and lazy data. Already sorted inputs
   avoid a full sorting copy. Exercise a descending source and a dimension whose
   name needs mapping syntax. The Qt-free test must execute the emitted Python
   on independent arrays. Keep sorting and coordinate selection in the shared
   numerical call. The Python renderer substitutes names and renders arguments.
3. **Replace the live publication path.** Register the two source entries at
   ingress. Capture their revisions at submission. If a consumed view is
   filtered or transposed, include that transformation in its history. Route
   Child, Top-Level, and Replace through the same evaluation and installation
   transaction. Child and Top-Level differ only in placement. Replace keeps B's
   entry and gives the result B's prior revision as its recorded input. Reject
   a cycle in the next bindings before modifying a window. Stage 2's
   `add_source()` takes an eager checkpoint and its evaluator copies operands.
   Test through the Manager with a versioned lazy target whose coordinates are
   available and whose intensity payload counts reads. Account separately for
   chunks the target viewer needs to draw its own slice. Registration must
   preserve laziness. Inspecting provenance and interpolating onto the target
   grid must add zero target intensity reads.
   Check for redundant full copies of already-owned eager inputs, and verify
   that the producer can reuse a mutable source buffer after acceptance.
   [Current dialog](../../src/erlab/interactive/imagetool/dialogs.py)
4. **Run the complete history.** Use the public Manager or viewer action for
   source changes and the actual Copy Full Code and Refresh actions. With Auto
   off, publish C and change only A's x
   to [0.25, 1.75]. C and its executed Copy Full Code must still give
   [[1, 3], [11, 13]] on A₀'s grid. The inspector identifies A₀ as used and A₁
   for the next update. Refresh yields [[0.5, 3.5], [10.5, 13.5]] on A₁'s grid.
   Repeat with only B changed to catch swapped roles. Compare values, dimensions,
   coordinates, dtype, name, and attributes against independently constructed
   DataArrays. While C is stale and A's window now displays A₁, reopen C's
   recorded step. The editor must restore B₀ and A₀'s grid. Unchanged Apply
   preserves C₀ and its next bindings. Repeat after closing the target window.
   Exercise a superseded submission and a failed window install. Check the
   original A/B/C example for a follower of a replaced B.
5. **Prove the old decisions are gone.** Remove the precomputation, separate
   `_multi_input_operation_provenance()` call, and per-placement publication
   from `InterpolationDialog._accept_manager_target()`. Make the migrated editor
   read its recorded target and primary input from the application. Retire the
   live use of `_interpolation_named_input_spec()`, which reconstructs the input
   from row scope and tree parent. Preserve Custom Grid, Uniform Interpolation,
   standalone use, and the shared helpers their callers still need.
   [Current editor](../../src/erlab/interactive/imagetool/manager/_provenance_edit/_controller.py)

**Completion evidence.** Show the one operation definition, the common submission
and publication path, the removed decisions, and the end-to-end test results.
Make the old spec unavailable or deliberately misleading *before registration
and submission*. Creation, inspection, copied Python, and Refresh must still
use the selected inputs. Verify a single numerical evaluation per publication.
If a second interpolation-specific algorithm or source-selection rule is still
needed for one of those actions, resolve that design issue before the next
workflow.

## Subsequent milestones

| Work | Evidence for completion |
| --- | --- |
| Remaining Manager commands | Migrate file and notebook ingress, watched updates, transforms and batch actions, display filters, metadata, and paste one workflow at a time. Show the removed decision and exercise the normal UI. Include rejected watched updates, transpose → duplicate → filter, shared historical edits, and multi-window rollback. Grouped edits restore the full group from any member. Unchanged Apply preserves its revision. A changed group updates only the selected branch. |
| Console | A window handle is live. A calculated result retains its inputs. The public `.data` is an independent editable snapshot. Item assignment records an edit. Assigning a plain DataArray creates an owned checkpoint. Assigning a recorded result preserves its provenance. Check ordinary functions, saved methods, repeated operands, the supplied `xr.concat` and `era.transform.rotate` aliases, and `tools.apply` for separately imported callables. Unsupported writes explain the snapshot/edit/assign route. Use the approved console behavior matrix from the prototype plan, lines 233–289. |
| Operation and worker execution | Require one numerical definition for each migrated operation. For affine coordinates, cancelling ±21.2 offsets must preserve four distinct float32 coordinates in copied Python. For extensions, bind the script version, input revision and effective settings at submission. Exercise import-time `keep_attrs` in copied code and across consecutive submissions with different settings. Change broadcasting after a lazy result is published and compute its chunks. These four cases failed on the prototype. Preserve chunking and keep the caller's options isolated. |
| ToolWindows and Figure Composer | Model a ToolWindow as an editor attached to durable output recipes. In ktool, changing a conversion offset makes its output stale, restoring the offset clears that cause, and toggling the zone overlay leaves the output current. Replay after editor close. Check several fit outputs and out-of-order completion. A figure may reference ftool before ftool publishes an ImageTool. Figure composition creates no data edge. |
| Native workspace and resource use | Save shared records once by ID and restore the graph before constructing windows. Save and reopen the stale C example, then check displayed data, both input revisions, Auto, copied code, and Refresh. Repeat a grouped historical edit. Protect mutable input buffers, coordinates, and metadata before the producer can reuse them. Keep file-backed lazy inputs sliceable and verify their source version on use. Measure reachable bytes across refresh/save/reopen and file size after compaction. Measure cursor-event gaps during growing acquisition, filtering, Auto, and publication against the same workload on main. |
| One-way migration | After the native round trip passes, import representative older workspaces. Translate evidenced file origins, inputs, steps, placement, and checkpoints. Keep a value with incomplete history as an identified checkpoint and preserve recoverable descriptions. Preserve stored-code trust checks. Switch the default reader and writer after verifying the fixtures and remove superseded live compatibility paths. |

Throughout the migration, preserve lazy inspection and the indexed propagation
tests for chains, shared producers, and unrelated entries. Run a representative
catalog operation and its generated code with Qt imports blocked. Use one Qt
binding locally and cover binding-sensitive Manager and ToolWindow behavior
under both PyQt6 and PySide6 in CI. At each handoff, record the tested revision,
the workflow it covers, and the first unresolved case.

On the rebased foundation, 767 tests passed under PyQt6 across the graph,
ImageTool replay, workspace revision/checkpoint, and interpolation selections.
The CI partition, Ruff, and mypy checks passed. The next implementation starts
with the interpolation fixture above.
