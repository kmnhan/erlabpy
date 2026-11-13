# ImageTool provenance graph redesign

When a source changes, a user needs two answers. Which inputs produced the data on
screen? Which inputs will Refresh use? Those answers can differ. The graph must
support both, including after the workspace is saved and reopened.

## User requirements

### Data and operations

Each real node represents one ImageTool and its DataArray. A provenance operation
labels a directed edge from each input to the result. An operation with two inputs
has two edges leading to the same result. The Manager tree, provenance panel, and
execution all use this graph.

There are two special cases. A **file node** represents a file path. A load
operation connects it to the data that it produced. A **virtual node** represents
a data state without a corresponding ImageTool. Replay can reconstruct it from
its incoming operation or read a stored array. For example, after replacing data
supplied through `qshow()`, the original notebook array must remain available
to the operations that used it.

**Open Child Window** and **Open Top-Level Window** both create a real node with
the same provenance. Tree placement is presentation metadata. **Replace Current**
keeps the ImageTool and changes the data it displays. The graph records the
operation and preserves the earlier state as a virtual node.

### Replacement examples

1. **File origin.** Load A from file F, then replace A with A′.

   ```text
   On load          F → A
   After replace    F → A → A′
   ```

   There are three graph nodes and one ImageTool. A is now virtual. The provenance
   panel for A′ folds the load and replacement steps into its history.

2. **Two independent ImageTools.** Replace B with the result of an operation that
   reads both B and A.

   ```text
   Before           A     B
   After replace    B → B′ ← A
   ```

   The Manager still has two ImageTools. They display A and B′. The two arrows
   describe one calculation, and B is a virtual input to it.

3. **A derived result with Auto off.** Start with B → C and an independent A.
   Replace B as in the second example. A solid arrow below records an input that
   was used. A dotted arrow shows a source for the next refresh.

   ```text
   Before           B → C          A

   After replace    B → C          C still displays the result made from B
                    B → B′ ← A     B′ is the new value in B's ImageTool
                    B′ ⇢ C         Refresh will use B′

   After refresh    B′ → C′        C's ImageTool now displays C′
   ```

   Auto controls when C updates. Until a refresh succeeds, C must still explain
   the result made from B. With Auto enabled, the same calculation can run
   automatically.

Reload walks upstream to a file, a virtual node with stored source data, or a
real node without an ancestor. It then evaluates the required operations.
Refresh re-evaluates affected operations from updated inputs.

### ToolWindows and Figure Composer

A ToolWindow such as dtool, goldtool, or ktool is a persistent editor attached
to its sources. Publishing an ImageTool creates a data node connected by edges
to the ImageTools used in the calculation. The editor itself stays outside the
data graph. It can remain associated with one or more output operations. A change
in its effective parameters can make an output stale even when the input array is
current.

Figure Composer has its own tab and figure recipes. A figure can use an ImageTool
or a ToolWindow such as ftool directly as a source. Figure composition creates no
data provenance edge.

### Visible behavior and resource use

Users must be able to predict an action from what the GUI shows. For a stale
result, show what produced it, what has changed, what Refresh will use, and
whether Auto is enabled. If a file or input is unavailable, identify it and show
how the user can proceed. A user should never have to infer an input policy from
tree placement, the lifetime of a window, or the presence of a cache.

Graph operations must stay fast in large workspaces and during frequent data
updates. Preserve data needed for replay and restoration. Bound the memory and
disk used by superseded results, temporary calculations, and duplicate payloads.

### Console contract

Approved on 2026-11-06. Arbitrary array edits use the explicit
`.data` → edit → assign boundary. This section defines the target behavior and
supersedes conflicting console expectations in the repair history below.
Implementation followed the plan review. The verification evidence below covers
this contract. The [Manager migration record](#manager-migration-record) gives the Stage 3 acceptance results.

#### Objects and ownership

| Object | Meaning | Lifetime and mutation |
| --- | --- | --- |
| `h = tools[0]` | A live reference to one Manager window | Reads and supported writes use that window's current data. Passing or assigning the handle preserves its identity. |
| `r = h * 2` | A calculated result with recorded inputs and operations | Values, coordinates, metadata, and recorded inputs remain fixed after calculation. Replacing the source does not change `r`. |
| `a = h.data` or `a = r.data` | An independent, editable DataArray | Normal xarray mutation applies. The Manager does not track the array, its views, or variables that refer to it. |

A handle is a Manager interface with a bounded set of array operations. It is
not a substitute for every DataArray API. `.data` is the explicit boundary to
ordinary xarray and NumPy use. Eager snapshots own their mutable values,
coordinates, and nested metadata. Lazy snapshots retain independent expressions
over stable backing data under the [input ownership contract](#input-ownership-contract).
Requesting a lazy snapshot does not materialize its full array.

Saved live methods resolve both data and provenance when called. For example,
`select = h.isel` follows the current window when `select(...)` runs. An operation
that already started, including a constructed coarsening operation, retains its
captured input. A closed window makes its live handle unavailable with a clear
error. Already calculated results retain their inputs after that window closes.

#### Editing boundary

The supported window writes are item assignment through `h[key] = value` and
replacement through `h.data = value`. A direct item assignment records the
operation. Assigning a plain DataArray records a new checkpoint: the Manager
knows its values but not the Python steps that produced them. That checkpoint
does not inherit the target window's previous reload recipe. Assigning a handle
or a recorded result retains its captured provenance instead.

Use the snapshot boundary for coordinates, attributes, encoding, NumPy views,
and arbitrary xarray edits:

```python
def transform(h):
    a = h.data
    a.coords["gain"].values[:] = 7
    a.attrs["sample"] = "edited"
    h.data = a
    return h * 2
```

Inspection through a handle must expose read-only objects or reject the access
with a message that directs the user to `.data`. This applies through nested
objects as well. Expressions such as `h.coords["gain"].values[:] = 7`,
`h.attrs["sample"] = "edited"`, and `h.values[:] = 0` must raise before changing
the window. They must not succeed on unnoticed writable copies. Raw array
conversion and mutable exports require the explicit snapshot, for example
`np.asarray(h.data)`. Scalar conversion is allowed and produces a fixed Python
value, as described below. Implementing general mutable proxies is outside this
contract.

Each supported write prepares its data and provenance, then commits them
together. Failure leaves the window, graph, navigation history, and dependent
states unchanged. A successful write preserves earlier recorded inputs and
marks affected followers stale. Later edits to an assigned DataArray cannot
change the accepted revision. Calculated results expose no mutation path;
edit `r.data` and assign it to a window to publish a changed value.

Python assignment rules still apply. `h.data[...] = value` edits only the
temporary snapshot. `h.data += value` reads a snapshot and calls the property
setter, so it publishes a checkpoint. `h += value` rebinds the Python variable
to a calculated result; it does not edit the original window.

#### Functions and recorded operations

`f(h)` receives the same live handle in ordinary Python and IPython, including
imported functions, nested calls, and repeated positional or keyword arguments.
Each supported write commits when executed. A later exception does not undo
earlier successful writes. `f(h.data)` receives an ordinary snapshot, and its
mutations affect the window only after explicit assignment.

Supported arithmetic, comparisons, indexing, methods, and accessors each need a
defined recording rule. Capture each operation's input values and revisions at
the same point. Preserve separate revisions when a calculation uses a value
from before an edit and another from after it. Unsupported handle operations or
mutation protocols fail with a resolution that uses `.data`; they must not
silently discard provenance or substitute a mutable internal DataArray. This
check applies when an operation reaches the handle, not by intercepting every
function call.

User functions and installed library modules keep their ordinary Python identity.
The supplied console aliases for `xr`, `era`, and `erlab` dispatch supported library
calls with handles to the recorded-call interface. This preserves direct expressions
such as `xr.concat([tools[0], tools[1]], dim="scan")` and
`era.transform.rotate(tools[0], ...)`. Calls with ordinary DataArrays use the
original function. Functions or modules imported separately remain ordinary Python
objects; `tools.apply` is available for tracked calls through those imports.

This bounded alias dispatch supersedes the earlier requirement to call
`tools.apply` explicitly for every library operation. The user corrected that
restriction on 2026-11-06. It does not infer arbitrary edits or function recipes.
Use no user-function substitution, console syntax transform, or cell-boundary
content scan. Opaque callbacks require snapshots and do not gain a replay promise
from inspected function source.

An ordinary function call records the supported array operations that actually
ran. It does not record the entire Python program. For example:

```python
def scale(h):
    factor = 2 if h.mean().item() > 0 else 3
    return h * factor
```

The resulting multiplication records the chosen factor. Refresh of a published
result replays the recorded operations with the graph's next input bindings;
it does not rerun this conditional. Converted Python scalars are fixed settings.
Copy Full Code uses the inputs that produced the displayed result. Any future
feature that reruns a complete custom function must declare its inputs and
dependencies explicitly. It is outside this console repair.

#### Publication and resource use

New windows, child windows, replacement, and assignment use the same preparation
and commit rules. `qshow(replace=...)` replaces the specified window, preserves
its identity, and retains its earlier revision for existing results. It must not
create an extra window. Publishing `r` after its source changes still publishes
the values and inputs captured by `r`.

Shape inspection, supported read operations, and small selections must avoid
copies of the full source. Snapshot export and acceptance of mutable data have
explicit ownership costs. Preserve lazy execution and recorded file guards.
Keep revisions needed by retained results and release temporary references when
their owners are gone. An unrelated cell must do no work proportional to retained
array sizes. These rules supplement the existing resource acceptance conditions.

#### Behavior matrix

These are acceptance requirements. The verification evidence below records their
audit against the revised contract. Use multidimensional arrays
with an auxiliary coordinate and nested attributes where they expose differences.

| ID | Action | Required outcome |
| --- | --- | --- |
| E1 | Obtain `a = h.data`; edit values, coordinates, nested metadata, or a NumPy view | Only `a` changes. The window revision and follower states stay unchanged. Repeat for `r.data` and lazy data. |
| E2 | Execute `h.data[key] = value` | Only the temporary snapshot changes. No window edit or cell-boundary publication occurs. |
| E3 | Assign an edited plain DataArray with `h.data = a` | One owned checkpoint replaces the window data. Previous results remain reproducible; followers become stale. Later edits to `a` have no effect. |
| E4 | Execute `h[key] = value` | One recorded item edit updates the window. The recorded operation consumes the exact prior displayed input, including an active filter or axis swap. |
| E5 | Write through `h.coords`, `h.attrs`, `h.values`, `.loc`, or another unapproved nested path | Raise with the snapshot editing resolution. Values, metadata, revisions, and follower states remain unchanged. No writable-copy fallback. |
| E6 | Execute `h.data += value`, then separately `h += value` | The first publishes a checkpoint through the setter. The second rebinds the variable to a recorded result and leaves the window unchanged. |
| E7 | Attempt an invalid item edit or replacement that fails preparation or installation | Preserve the previous window, graph, navigation history, and follower states. No partial revision is published. |
| E8 | Edit a calculated result directly, or edit its exported snapshot | Direct mutation is rejected. Snapshot mutation is independent; opening the original result still uses its original values and provenance. |

| ID | Action | Required outcome |
| --- | --- | --- |
| F1 | Call `f(h, again=h)` in the same cell, a later cell, an imported helper, and ordinary Python | Preserve argument identity and the same read/write behavior in every context. User function objects remain unchanged. |
| F2 | Apply a supported write in a helper, then raise | Keep that committed write and its stale-state changes. A failed write itself has no effect. An exception before snapshot assignment leaves the window unchanged. |
| F3 | Call `f(h.data)` and mutate the input | The window stays unchanged, including on failure. Only explicit later assignment publishes the snapshot. |
| F4 | Save `select = h.isel`, replace the source, then call `select(...)` | Use current data and matching current provenance. A result or coarsening operation constructed before replacement retains its earlier input. |
| F5 | Combine calculated values from before and after a handle edit | Display and copied code use both exact revisions. Repeat after replacing or closing the source before publication. |
| F6 | Take a Python conditional branch based on data, then Refresh a published result after a source change | Replay the recorded branch operations and fixed settings. Do not rerun the Python conditional. Historical copied code still reproduces the old displayed result before Refresh. |
| F7 | Request an unsupported handle operation from a library function, callback, or mutation protocol such as an unapproved `out=` target | Reject that operation before its mutation and explain the snapshot route. Earlier committed writes follow F2. Calling the ordinary function with snapshots uses normal library behavior. |
| F8 | Inspect supported attributes or request raw conversion such as `np.asarray(h)` | Supported inspection is read-only. Raw conversion requires `.data`; no internal buffer or unnoticed writable export escapes. |

| ID | Action | Required outcome |
| --- | --- | --- |
| P1 | Publish a live handle or recorded result into a new root or child window | Capture the correct revision and use the same lineage rules regardless of tree placement. |
| P2 | Use `qshow(manager=True, replace=target)` with a live handle or recorded result | Replace the target without adding a window. Preserve target identity and old inputs. Failure leaves the target unchanged. |
| P3 | Assign `target.data = h` or `target.data = r` | Preserve captured provenance through the same replacement path. Contrast with the new checkpoint required for `target.data = h.data`. |
| P4 | Replace a source while a published follower has Auto off, then Refresh | Keep the old result and recorded inputs until Refresh succeeds. Mark it stale and show its next inputs. Successful Refresh publishes matching data and provenance. Repeat with Auto enabled. |
| P5 | Publish a retained lazy result after source replacement or closure | Retain its computation over stable backing data. Preserve file-version checks before the first Refresh; an unavailable backing file reports the reason and recovery path. |
| P6 | Repeatedly read shape, select a small plane, and run unrelated cells with snapshots retained | Shape and plane selection make no full-source copies. Unrelated cells do not scan retained arrays. Lazy operations compute only when required. |
| P7 | Release temporary snapshots and results; retain an older result across source changes | Release unused captures without retaining the Manager. Keep every revision needed to reproduce the retained result. |

#### Verification gate

Before implementation, inventory the supported handle operations and map all
console read, write, conversion, function, and publication paths to the matrix.
Identify paths that will be removed or will require explicit snapshots. This
inventory must include both live handles and calculated results.

After implementation is authorized, first reproduce mismatches against the
matrix and label intended behavior changes. For every applicable row, check
displayed values, coordinates, metadata, recorded inputs, follower state, and
executed Copy Full Code together. Check Refresh separately from historical
replay. Checkpoint-based copied code must receive the exact recorded checkpoint
through the normal input-resolution path; a description of the edit is not a
replay recipe. Exercise binding-sensitive publication and failure paths under
both PyQt6 and PySide6. Measure copy counts and lazy execution as well as values.

Record test evidence or a specific unresolved reason for every row. Stage 3
remains open while any required behavior is unresolved. Existing passing tests
that expect writable detached handle attributes do not satisfy this gate.

#### Implementation inventory

The initial audit maps the current paths as follows. This inventory selects the
implementation boundaries; the behavior matrix remains the acceptance source.

| Path | Matrix rows | Implementation decision |
| --- | --- | --- |
| Window and result `.data`; snapshot assignment | E1–E3, E6, E8, F3 | Keep independent snapshots and checkpoint assignment. Test augmented assignment separately from temporary snapshot writes. |
| Handle arithmetic, comparisons, indexing, and item assignment | E4, E7, F5 | Keep recorded operations. Validate operands before execution and use one rollbackable window commit for all supported writes. |
| Generic attributes, setters, NumPy conversion, and fallback results | E5, E8, F7–F8 | Replace open delegation with bounded scalar inspection, coordinate results, and known methods. Require `.data` for mutable exports and unsupported operations. Reject attribute shadowing and unapproved NumPy output targets. |
| Saved methods, qsel and kspace accessors, and coarsening | F4, F7, P6 | Resolve live receivers at call time. Keep already constructed coarsening inputs fixed. Support recorded accessor operations and named reducers; require snapshots for opaque callbacks and other accessors. |
| Library call dispatch and callable-source capture | F1, F3, F6–F7 | Supplied aliases route direct calls through the same bounded recording path as `tools.apply`: registered library operations, multi-input `era.transform.shift`, and `xr.concat`. Keep installed modules, user functions, and ordinary DataArray calls unchanged. Do not infer function recipes. Retain normal NumPy ufunc dispatch. Reject unsupported tracked calls before invoking the function. |
| Cell hooks | F1, F6, P6–P7 | Retain result naming and display only. User functions keep their identity, and array contents are never scanned. |
| Assignment, `tools.itool`, and handle `qshow` | P1–P3, E7 | Share handle publication and replacement paths. Honor replacement targets and preserve provenance without falling back to snapshot publication. |
| Graph bindings, Refresh, Auto, lazy guards, and capture ownership | F5–F6, P4–P7 | Audit existing ownership and scheduling behavior through the matrix. Preserve exact consumed revisions and measure copies and capture release. |

The initial bounded read interface keeps array shape, dimensions, dtype, size,
chunk information, coordinate selection, standard arithmetic and comparisons,
known DataArray selection and reduction methods, qsel operations, registered
kspace operations, and named coarsening reducers. Mutable attributes such as
`attrs`, `coords`, `values`, `encoding`, `variable`, and `.loc` require `.data`.
Scalar conversion remains available. A library call can use supported handle
operations directly. Supported library calls through supplied aliases retain
provenance; use `tools.apply` for separately imported functions that require raw
DataArrays. Whole-function replay and general mutable proxies remain out of scope.

#### Console verification evidence

The initial 53 contract cases produced 46 failures before implementation. They
exposed unrestricted exports, opaque callbacks, and ignored
replacement targets. One invalid test assumed that a one-dimensional ImageTool
was unsupported; it was corrected to use an empty array after checking the runtime.
Older tests that expected implicit writable exports now use the explicit snapshot
boundary. Native function execution and copied-code replay remain covered.

The following evidence is in `test_console.py` and `test_console_contract.py`
under `tests/interactive/imagetool/manager/`. Tests execute copied code with the
recorded checkpoint inputs and compare the resulting DataArrays, including
coordinates and attributes. Refresh is checked separately from historical replay.

| Rows | Evidence |
| --- | --- |
| E1–E2 | Snapshot export cases cover eager and lazy data, live handles and calculated results, temporary expressions, coordinates, nested attributes, encoding, variables, and NumPy views. Window values and revisions remain unchanged. |
| E3 | `snapshot_edits_require_explicit_assignment`, `snapshot_assignment_starts_new_history`, and `assigned_data_is_an_independent_snapshot` check owned checkpoints, stale followers, and old copied results. |
| E4 | Item-edit cases cover filtered displays, transposed axes, accepted revisions, and navigation state. Earlier results keep their inputs. |
| E5, F7–F8 | Contract cases reject implicit mutable exports, attribute writes, callbacks, raw conversion, and NumPy mutation protocols before mutation. Shape, dimensions, signatures, coordinate results, and scalar reads remain available. |
| E6 | `augmented_assignment` distinguishes property assignment from variable rebinding and checks follower state and historical code. |
| E7 | `failed_edit_rolls_back` injects preparation and installation failures for snapshot assignment, direct item edits, and `qshow` replacement. Window data, graph revision, navigation, and followers are preserved. |
| E8 | `result_mutation_requires_snapshot` rejects direct mutation and verifies that snapshot edits do not alter publication or copied code. |
| F1–F3 | Helpers run in the same cell, later cells, ordinary Python, and an imported module. Repeated arguments keep identity. Successful writes survive a later exception; snapshot mutations require assignment. Installed modules and user functions remain unchanged. Supplied library aliases dispatch supported tracked calls. |
| F4 | Saved selection, qsel, and reduction methods use current data. Constructed coarsening operations, including their wrapped methods, retain the original inputs. |
| F5 | Before/after edit cases use arithmetic, ufuncs, methods, concat, and assignment. Retained results publish their original inputs after replacement or closure. Multi-input Shift retains a nested one-dimensional source. |
| F6, P4 | `refresh_keeps_chosen_python_branch` checks fixed branch settings, historical code before Refresh, new code after Refresh, and both manual and Auto updates. Child publication also checks both Auto states. |
| P1–P3 | Publication cases cover live and calculated inputs, root and child windows, assignment, `qshow`, and `tools.itool`. Replacement accepts an index, negative index, one-item list, child UID, or next unused index. Invalid targets preserve existing windows. |
| P5 | `retained_lazy_result_keeps_file_guard` publishes after source replacement or closure. Lazy data stays chunked. File replacement and removal remain detectable before the first Refresh. |
| P6 | Repeated shape and small-plane reads make no full-source DataArray copies. An explicit snapshot makes one full-source copy. Lazy inspection computes no tasks, and unrelated cells do not tokenize retained arrays. |
| P7 | Temporary exports and input holders are released. Named results retain their source captures after the original result is collected. Retained results do not keep the Manager alive. The core graph test verifies release of an unused captured revision. |

Two integration repairs were needed during this audit. Statement operations such
as `kspace.set_normal` now remain in replay when followed by arithmetic or a
method. Fresh child results bind their explicit recipes to the graph and preserve
their requested Auto setting. Both paths execute copied code in regression tests.

Validation passed: 474 console, dependency, and core graph cases under PyQt6;
403 console and dependency cases under PySide6; mypy on 338 source files; Ruff;
the test partition check; and the HTML documentation build. The child Auto cases
were added after the broad PyQt6 run and also checked separately under PyQt6.
The changes remain uncommitted.

The direct-library-call correction adds regression cases for concat and rotation
through both `era` and `erlab`, positional and keyword data arguments, historical
code after source replacement, and subsequent Refresh. Tests also check plain
DataArray calls, separately imported functions, optional `tools.apply`, unchanged
installed functions, and rejection before an unsupported tracked call executes.
All 285 console cases pass under PyQt6 and PySide6. Ruff, mypy, the test partition
check, `git diff --check`, and the HTML documentation build also pass.

## Proposed implementation choices and six-stage plan

### Implementation and acceptance rules

1. Preserve the data contract through creation, Refresh, Reload, and restoration.
   Trace validation, calculation, and window installation together. Check both a
   small selection and a large lazy result with the default compute threshold.
2. Account for each calculation or copy whose cost grows with the dataset. Record
   its size, frequency, and execution thread. Define ownership before moving work
   to a worker. Historical inputs must remain stable through publication.
3. Give each queued request a defined outcome. Run it, keep it pending with a
   guaranteed resume path, or cancel it and clear its scheduling state. Check
   early returns, cancelled close prompts, supersession, failures, and destruction.
4. Measure responsiveness during the complete workflow. Include acquisition,
   filtering, Auto, history capture, and publication while cursor events arrive.
   Record event-loop delays and worst observed values alongside operation times.
5. Test observable requirements. Lazy tests cover the refreshed output as well as
   source capture. Lifecycle tests check that work resumes after cancellation.
   Permitting full-frame copies requires evidence that interaction stays usable.
6. Close a stage only when each acceptance condition has evidence. Record
   remaining uncertainty and keep the stage open while a required case fails.

The ownership repairs and their verification records below describe earlier
work. They do not establish compliance with the revised
[console contract](#console-contract), close Stage 3, or establish a new
performance baseline. The pauses already present on main are outside this
change's acceptance conditions.

### Replay migration test gate

Before migrating each remaining replay path to the graph:

1. Adapt its tests to check user-facing behavior through the normal GUI or public
   API. Run these tests on the old path before changing production code. Record
   any existing failure or intended behavior change separately; a passing old
   implementation is not proof that its behavior is correct.
2. Check the relevant workflow from input to displayed result and replay. Include
   values, coordinates, metadata, recorded input access, executed copied code,
   stale state, and Auto behavior where applicable. Check initial publication
   before Refresh as well as later updates. Tests must detect disagreement
   between the window and the values that produced its recorded result.
   Include operation order within one console cell, the lifetime of editable
   aliases, and GUI interaction after a file read becomes unavailable.
3. Migrate the path with the same behavior assertions. Change fixture setup only
   where needed to reach the implementation. Keep graph IDs, private helper calls,
   and storage structures in separate implementation tests. Resolve a failing
   behavior assertion against the user contract before changing its expectation.
4. Complete the migration only after the affected behavior tests pass on the new
   path. Retain the old-path baseline and record any approved behavior changes.
   Check binding-sensitive workflows under both PyQt6 and PySide6.

The Stage 3 repair tests first reproduced these failures before production edits.
The console `.data` edit behavior described in this repair history was later
superseded by the independent snapshot contract below. The final verification
section records that change.

- After `tools[0].data[0, 0] = 100`, existing followers must become stale and a
  newly calculated result must replay the edited values. The reproduction showed
  an Aggregate result of `26.5` whose recorded input replayed to `1.5`. The
  earlier console test checked the edited array but did not check its followers
  or replay. The new test checks both with executed copied code.
- A newly published lazy result must enforce file validity before its first
  Refresh. Replacing its NetCDF file and closing the source made the actual
  slicer change from `1` to `9`, while the recorded revision rejected the read.
  The earlier retained-lazy test used stable Dask values and refreshed before
  checking. The new file test checks the actual slicer before any Refresh.
- Same-process `fetch()` must return independent, editable data for ordinary and
  filtered windows. The ordinary result rejected assignment; editing a filtered
  result changed the window to `42` while its recorded revision remained `1`.
  The new test edits fetched values, coordinates, and nested attributes and checks
  that the displayed data stays unchanged. Lazy results retain their chunks.

### Input ownership contract

- Submitting eager data lends it to the Manager for the duration of the call.
  The producer must not write concurrently with submission. Before returning,
  the Manager takes an isolated snapshot of mutable values, coordinates, and
  metadata. The producer can then reuse its buffer, including before the next
  notification. Managed source displays and recorded revisions use owned data.
- A NumPy read-only flag is not ownership: another view can still write to its
  backing allocation. Only buffers backed by immutable bytes can be shared
  without copying values. Manager item edits detach before writing. Raw internal
  arrays are not a mutable producer interface.
- Console objects and writes follow the
  [console contract](#console-contract). Explicit snapshots use the same eager
  isolation and stable lazy backing requirements as other accepted data.
- Workers receive owned revisions. Revision checks reject superseded requests;
  they do not substitute for memory ownership. Mutable eager input therefore
  requires one copy at acceptance. Coalescing cannot remove that copy safely.
- A lazy input is a retained computation over stable backing values. Its tasks
  and backing resources must produce the same values for the lifetime of its
  retained results. A producer that reads reusable buffers or mutable external
  state must first supply a snapshot or a versioned resource. The Manager copies
  the expression without computing the full array. Replacing or closing a source
  window changes future bindings, not the validity of existing lazy results.
- Recorded file reads retain version checks. Changing an external file is a
  change to backing values, unlike closing its source window. A stale lazy
  result over unchanged backing data must remain sliceable with Auto off.
  Initial publication installs these checks in the window's lazy computation,
  including after operations discard file encoding. Materialized checkpoints
  end the dependency on the file. Applying guards does not replay the operation.

The acceptance tests must reproduce source replacement and closure with an open
320 MB lazy result, and buffer reuse while a worker is paused before capture,
without sending another source notification. These supplement the notified
supersession and worker lifetime tests.

The implementation enforces the eager contract when a window joins the Manager,
when its source is replaced, and when the graph accepts an eager input. Window
edits detach shared immutable values. The graph retains lazy expressions without
a source-window lifetime flag; actual file changes still fail the file-version
guard. Explicit legacy checkpoint creation continues to materialize its input.

The examples above define the required behavior. Stages 1 and 2 implement the
graph and its commands. Stage 3 connects them to the live Manager.

### 1. Graph model outside Qt

| Record | Responsibility |
| --- | --- |
| Data revision | Identifies one immutable DataArray value and records its producing application, if any. An optional checkpoint holds the data. The revision is virtual when no ImageTool entry displays it. |
| File source | Identifies a file. The load edge owns the loader, arguments, and selected array or region. |
| Operation application | Records the operation, the parameters actually used, and the exact input node for each ordered, named slot. Each slot gives one edge. |
| Refresh recipe | Describes the operation and input bindings to use on the next evaluation. An editor can change the recipe while the applied record remains available. |
| ImageTool entry | Keeps a stable ID across replacement and points to the current revision. Owns the Auto setting and tree placement. |

Reuse the typed operations for execution and Python generation. Preserve input
roles in operations such as `B.interp_like(A)` and input order in concatenation,
including when two slots use the same node. Validate IDs, slots, and acyclic
recorded dependencies without starting a Manager or importing Qt into the model.

### 2. Replace, reload, and refresh

Give each refresh input one of three bindings. It can follow an ImageTool entry,
use a pinned revision, or re-evaluate an earlier producing step. The third case
matters when B was already derived. If `B = crop(A)` and B is replaced with
`smooth(B)`, a later change to A should run the crop and then smooth its result.
Validate live bindings for cycles separately. Using a pinned C to replace B is
valid even if C follows B. Making both ImageTools follow each other would give
Refresh a cycle.

| Command | Graph behavior |
| --- | --- |
| Replace | Protect irreplaceable input values, evaluate the operation, then publish its record and move the existing entry to the new revision together. Invalidate dependents that follow the entry. |
| Reload | For a live row, use its refresh bindings and rebuild the required inputs from their origins. Honor fixed inputs. Perform each requested file read even when a cached result exists. |
| Refresh | Use the current bound inputs and effective settings. Process stale prerequisites in dependency order. Auto schedules eligible results. |
| Historical replay and Copy Full Code | Use the applied parameters and recorded input revisions of the selected result. Exact restoration requires a checkpoint or a verifiable original source. |

For stale C, both commands follow the input shown for its next update. Refresh
uses the available B′. Refresh can read a file when a step it must run is a
file load. Reload also rebuilds current followed entries, so it rebuilds B′
from its origins and then recalculates C. Each pinned input keeps its exact
revision. The command preview lists the entries and earlier steps to rebuild, the
file loads to run, and the current and pinned revisions to use. It also identifies
other entries affected by publication. Previewing a command does not read a file.
When a file is missing, Reload identifies the path and the displayed revision
that is still available.

When a followed entry and an earlier step request the same producer, the planner
shares one calculation if both bindings use the same stored recipe snapshot or
it can establish equivalence from the settings. Both slots then record the same
revision. A shared file load appears once in the command preview. Each new
Reload reads the requested files again.

For example, submitting the same captured recipe for `isel(x=indices)` through
`set_recipe()` shares one selection and one result revision. This also applies
when the recipe comes from `graph.entry()`. Recipe construction captures the
operation and its bindings. Accessing an editable operation or file call detaches
its settings so that a caller can change them without changing a stored revision.
Submitting that editable copy or constructing an independent array indexer can
produce a separate evaluation. The planner does not scan or hash array contents.

Auto waits for stale prerequisites. If B has Auto off and C has Auto on, C waits
until B is refreshed. Explicitly refreshing C updates stale prerequisites in the
same request. After a successful publication, Auto schedules eligible followers
as separate refresh requests. A host can defer these requests to coalesce updates.
Graph commands return the revision displayed when the command finishes.
If an input changes during an Auto refresh, the graph discards the superseded
attempt and queues the latest inputs for the next pass. With a deferred scheduler
each pass yields to the host. A failure on current inputs leaves the entry stale
with an error until another change or an explicit retry.

Stage data and its provenance together. Check that the input revisions and
requested settings are still current before publishing. A failed, cancelled, or
superseded run leaves the last successful result available.

#### First Qt-free acceptance test

Use `test_replace_preserves_recorded_input_until_refresh` as the first test.
A plain Python graph fixture supplies the DataArrays that would normally enter
through `qshow()`.

Start with A₀ = [10, 20] and B₀ = [1, 2]. Create C₀ = 2 × B₀ and turn Auto off
for C. Replace B with B₁ = B₀ + A₀, then explicitly refresh C.

| Point | B displays | C displays | Input used for displayed C | Input for next refresh | C state |
| --- | --- | --- | --- | --- | --- |
| Before replace | B₀ [1, 2] | C₀ [2, 4] | B₀ | B₀ | Current |
| After replace | B₁ [11, 22] | C₀ [2, 4] | B₀ | B₁ | Stale |
| After refresh | B₁ [11, 22] | C₁ [22, 44] | B₁ | B₁ | Current |

Assert the values and revision IDs at each point. The three ImageTool entries keep
their IDs. B₁ has one producing application with inputs B₀ and A₀. After Replace,
release or overwrite the caller's B₀ buffer and verify that the virtual B₀ still
resolves to [1, 2]. After Refresh, C₁ records B₁ as its sole input. Add separate
cases for the derived B above, a failed refresh, and shared file input evaluated
once per request.

### 3. Manager tree and commands

Make the tree a projection with one row per ImageTool entry and stable references
for display parent and order. Show every input in the provenance panel, including
sources elsewhere in the tree. Fold virtual steps for reading while keeping
shared ancestors accessible. Drive badges, action availability, and the recorded
and next inputs from graph queries.

Show **Inputs used** above the operations that produced the displayed result.
For stale results, show **Inputs for next update** below the provenance list.
Expanding this control shows the planned inputs and their roles. The Auto state
tells the user whether the Manager will request the update automatically.

Route file and notebook ingress, watched updates, dialogs, paste, and console
results through the same graph commands. Record a selection or filter as a step
when a calculation consumes the displayed subset. Editing a shared historical step
creates a revised branch for the selected result. Test that child and top-level
placement give identical lineage, that stale C shows both inputs, and that its
copied Python reproduces the displayed result.

Console integration also requires the
[behavior matrix](#behavior-matrix) and its verification gate. The Manager
migration record below includes the publication checks for this gate.

#### Ownership of provenance for new Manager results

Record one application per supported operation when its operands are captured.
Each operand position has its own named slot and exact revision, even when two
slots use the same array. The applied record describes the displayed data. The
entry's recipe describes the next update. A window, tree row, dialog, or Python
variable can identify a result. None of these owns another editable history.

An inspector row identifies both the recorded application and the selected
result. The first identifies what to edit. The second identifies which branch
to update.

The following inventory tracks where the Manager accepts or reads provenance.
It includes actions that a user can take after initial publication.

| Path | Stage 3 requirement |
| --- | --- |
| File, notebook, watched, and restored inputs (`_io`, `_mainwindow`, `_data_graph`) | Capture the origin, file preparation, or imported recipe at ingress. Keep the recorded input distinct from a newer live value. |
| Dialogs, batch actions, extensions, and metadata edits (`dialogs`, `_actions`, `_metadata_editor`, `_extensions`) | Publish the actual ordered operations and consumed inputs in the graph transaction that installs the result. Cover all three placements, including replacement with a display filter active. |
| Console and clipboard (`_console`, `_details_panel`, `_provenance_edit`) | Bind input occurrences at capture. Derive portable clipboard and Python output from the selected recorded steps. Verify repeated inputs, named expressions, and paste into multiple targets. |
| Tree, inspector, and commands (`_wrapper`, `_details_panel`, `_dialogs`, `_modelview`, `_dependency`, `_lineage`) | Read displayed lineage, input roles, source states, and next bindings from graph queries. Tree location supplies placement. Moving or closing a window must preserve recorded lineage. |
| Step actions (`_provenance_edit`) | Resolve a revision once for availability, edit, delete, revert, reorder, copy, and multi-selection. Use the recorded settings and inputs to restore the editor. Accepting equivalent settings and bindings preserves the revision. Check each applicable action from the result and its input history. |
| Publication and rollback (`_data_graph`, `_wrapper`, `_revisions`) | Install the graph revision and roll back failed installations. Refresh, Reload, and Auto must not rebuild provenance from the current parent window. |
| Persistence, transfer, and ToolWindows (`_workspace`, `_server`, `_figurecomposer`, `_extensions`) | Read older data at import. Export a projection where a consumer requires the existing format. Keep open-window queries independent of that export. ToolWindow editor ownership and native graph persistence remain stages 4 and 5. |

Before closure, exercise each live path with the old window spec absent or
deliberately different. Check the displayed value, ordered input revisions,
the editor's settings, executed copied code, and the source followed after an
update. Repeat binding-sensitive cases under both Qt bindings. Keep a reader
only when it has a demonstrated compatibility or transfer caller.

#### Acceptance for live publication

A live command captures the exact operand revision for each slot and reads its
operation settings once. The calculation uses that capture. The window installs
the resulting revision. Child, top-level, and replacement placement use the same
calculation. A worker returns the result of its captured request. A console call
supplies its operand roles directly. Only the compatibility reader may interpret
an older recipe to recover a role.

Stage 3 remains open until the publication API enforces this contract. A live
publisher supplies ordered revision IDs, the bindings for the next update, and
the operation with its effective execution settings. The evaluator resolves the
recorded revisions and produces the value. Window placement consumes that
calculation. File reads and raw arrays enter as sources. The compatibility
reader can import a saved value with its historical recipe. A worker must
return the result of the task bound to its input before dispatch.

The viewer may project an evaluated revision into slicer coordinates. Preserve
the source axis order and the display order independently across duplication
and workspace restore. A managed filter consumes its graph source revision.
Copy Full Code and Refresh must use the same effective xarray settings as
initial execution, including any reduction backend policy imposed by the
Manager. A change in the caller's options must preserve the recorded result.

Verify a first filter on a duplicated transposed array, a two-input Shift
across a change in attribute policy, and a float32 reduction across a change
in the reduction backend. Compare values, dimensions, coordinates and metadata
with executed Copy Full Code. Inject a conflicting value at the command
boundary and verify that only the captured revision supplies the calculation.
Review the remaining calls that accept saved data with a recipe and identify
their import or external source contract before closing the stage.

A raw input starts at the array the Manager received. Record each selection,
dtype or angle conversion, rename, and metadata assignment that the Manager
applies. Keep the revision supplying a watched notebook name explicit. Retire
that name on a newer notification, even when the new value cannot be displayed.
Edit source metadata below the layout and display filter in the same transaction
that installs the value. On rejection restore the window, revision, and follower
states together.

Before marking Stage 3 complete, trace each live publisher in the inventory above
from input capture through calculation and installation. Check each reader of
the result against its recorded revision. Exercise these sequences through the
Manager and execute Copy Full Code against the actual inputs:

- Transpose a nonsquare array, rename it, and transpose it back. Compare dimensions,
  coordinates, values, and name. Fail a rename after widget installation and check
  the previous revision and its followers.
- Reject a watched value with five non-singleton dimensions. Copy the displayed
  result while the notebook name refers to the rejected array. Then accept a
  valid update and check the name and revision used for each result. Repeat
  after reopening a workspace, with a second window that consumes the watched
  input. Copy that result before Refresh and check its new value after Refresh.
- Replace an input with integer data and angle conversion, and reduce a 5D input
  in the viewer. Inspect the original checkpoint and each preparation operation.
  Repeat with retained metadata and a pending refresh recipe. Open the same 5D
  calculation in a new window and assign it to an existing window. Check the
  selected slice after publication and Refresh. For a plain array assignment,
  check that the slice starts at the received array.
- Apply a filter after transposing a nonsquare array. Use both named and
  positional Gaussian widths. Check the exact copied values, the unfiltered
  source revision, Reload, and lazy and materialized workspace reads.
- Change the xarray attribute policy after a console calculation. Check the
  displayed array, Copy Full Code, and Refresh, including array attributes.
- Give a console handle a conflicting operand value while retaining its captured
  revision. Arithmetic, a typed operation, and item assignment must each execute
  once on the graph value. Compare the published array and Copy Full Code.
- Change a batch control during the first target's preflight. Every target must
  use the settings accepted for the command. Each batch filter runs once per
  source. A newer source write during installation must retain its own revision.
- Change a transform control during preflight. Compare the installed result,
  recorded settings, and executed code. Exercise child, top-level, and replacement
  placement. Change the source and check Refresh.
- Exercise named and repeated console operands with a conflicting window history.
  Check their ordered revisions before and after publication. A typed console
  call must still run and replay if its portable recipe exporter is unavailable.

Record the results and any remaining gaps here. Passing an earlier selection
alone does not close a later failure.

#### Manager migration record

The graph now has separate entry points for calculation, source ingress, and
import. `prepare_evaluation()` takes producing steps and resolves their ordered
input revisions. It returns the evaluated value and a commit that checks the
observed entries. The live command has no output array argument. Placement
can install a retained calculation without running the operation again.
`prepare_source()` accepts an external array or the selected file output.
`prepare_import()` accepts a saved or transferred value and its history.
Pasting a workflow supplies its steps and asks the graph to evaluate them.
The version 3 reader and external ToolWindow transfers still need the import
path because the Manager did not perform their original calculation.

An extension job binds the routine, script version, input revision, and settings
at dispatch. The Manager accepts the completed task. A caller can inspect the
input and returned array without changing the value accepted by the graph.
The Manager executes registered routines in a dedicated process. The process
sets the captured xarray options before it imports the script and calls the
routine. The Manager still checks the script, source revision, and destination
before publication. Dialogs capture the calculation policy before preflight
and use it for every target. Widget preflight can prepare a preview.
When source preparation has recorded steps, the graph evaluates those steps
and the widget installs the graph value.

The publication checks compare the first Gaussian filter on a transposed
duplicate with executed Copy Full Code. They also check a two-input Shift
after a change to `keep_attrs`, and a float32 mean after a change to
`use_numbagg`. A conflicting widget operand cannot replace the recorded
input. Swapping the display axes reuses the accepted filter result. A raw
replacement with a metadata edit executes the assignment once. Divide by
Coordinate preserves the input name in both the window and copied Python.

On 2026-11-13 the calculation settings changes passed 7,109 PyQt6 tests in
the selection below, with six platform skips. The worker race also passed
with executed Copy Full Code and Refresh. A spawn-process check returned a
lazy array with its original chunking and executed its script-defined task
after the worker exited. Ruff, mypy, test partition, HTML, and diff checks
passed. The packaged executable and PySide6 remain for CI.

On 2026-11-12 the publication API changes in the working tree at `4df92098`
passed 7,104 PyQt6 tests. Six platform-specific cases were skipped. The run
covered the complete ImageTool test directory, the provenance graph, momentum
conversion, interactive utilities, and image analysis. Ruff lint and format,
mypy on 338 source files, the test partition, the HTML build, and
`git diff --check` passed. The PySide6 run remains with CI.

Live transforms now evaluate the captured settings on the graph inputs.
Registration accepts the resulting revision and creates the window from it.
Replacement records any reduction accepted by the destination window. Rename
edits the source revision before layout and filtering. Console calls record
the operand for each slot when they run. A typed call executes its recorded
operation. A Python expression keeps its code and the names of its captured
operands. An extension worker uses the operation and input captured at dispatch.

A watched notification retires the previous notebook name before input
validation. Only successful publication binds the name to the new raw
revision. Copy Full Code resolves a saved calculation's recorded inputs when
the user requests the code. A saved result can therefore refer to its
checkpoint after the notebook variable changes. Version 3 imports can still
use their saved program when the input revisions cannot be reconstructed.

A managed filter now reads the original source dimensions, applies the captured
filter, and installs its result in the selected display order. The graph retains
the unfiltered source revision separately. A workspace records the input order
needed to restore positional filters. The Normalize filter and its copied code
use the same public normalization function, including its denominator mask.

A batch captures the controls before its first preflight. Each target uses that
capture. A filter runs once per target. Installation checks the source revision
again so that a newer write keeps its value and history.

The console graph command accepts an operation and ordered references to its
operands. It reads those revisions, executes the operation, and retains the
output. Arithmetic and item assignment execute their recorded Python. Typed
calls execute the recorded operation directly. A calculated handle reads the
retained output. Copy Full Code and Refresh use the captured xarray settings,
including the attribute policy.

On 2026-11-12, the filter, batch, and console repairs passed 7,093 PyQt6
tests across ImageTool, the provenance graph, momentum conversion, interactive
utilities, and image analysis. Six platform-specific cases were skipped. After
the final console handle lifetime change, all 454 cases in the console, console
contract, and pending workspace modules passed. These selections overlap.
They include a direct handle and a handle that outlives its original namespace.
Both publish and refresh from the graph while allowing the Manager to close.
Ruff, mypy, the test partition, the HTML build, and the diff check passed.
PySide6 validation remains assigned to CI.

On 2026-11-12, 6,974 PyQt6 tests passed in the complete ImageTool test
directory and the provenance graph, momentum conversion, and interactive
utility modules. Six platform-specific cases were skipped. Another 675 tests
passed for the derivative tool, fitting tools, Dask, code trust, and IPython
integration. The checks include Copy Full Code after a rejected watched update
and workspace restore, Rename rollback after widget installation, and Refresh
after selecting a 5D slice in a replacement window. Ruff, mypy, the test
partition, the HTML build, and the diff check passed. PySide6 validation is
assigned to CI.

The 2026-11-11 audit covered calculated results. It passed 3,866 Manager,
console, provenance, workspace, viewer, and replay cases under each Qt binding.
Another 629 workspace storage, dialog, and momentum conversion tests and the
13 original publication cases passed under each binding. A later audit
reopened Stage 3 when file and notebook registration used the window's history.

The live registration API now takes the accepted input or the calculated
revision. A registration callback cannot fetch or consume a top-level or
child window until its revision is published. A notebook input starts from the array received by the Manager. If the
sender already reduced the array, that selection is the starting value. If
the Manager converts integer data or angle coordinates, the graph keeps the
received array and records the conversion.

File registration captures the path, loader, selected array, and preparation.
File → Open in a managed ImageTool also replaces its input recipe. Reload
reads the new file, and changes to the former source no longer mark the window
stale. If registration fails, the window recovers its data and bindings.
A session-only loader stays with its live file revision. A portable copy can
start from its recorded array.

Saved workspaces, standalone transfers, and external ToolWindow outputs enter
through the import API. Only an explicitly imported ImageTool can reconstruct
missing graph records from its saved history on first use. The registration
tests substitute or clear the window history before registration. They check
the source revision, execute copied code, and update the files or windows
independently. The workspace tests also reopen and replay an older saved recipe.

The 2026-11-12 PyQt6 selection passed 4,900 distinct cases, including the
six registration failures and legacy workspace replay. A further audit found
four file cases outside that selection. The regression tests now reload two
different callables with the same Python name, reject a file after the window
has accepted its data, apply acquisition metadata to a file opened in place,
and reload a prepared file after an external data replacement. A failed file
installation must preserve the previous revision and the state of its followers.
After these fixes, 2,384 PyQt6 tests passed across the graph, Manager, console,
workspace loading, provenance, and replay. PySide6 validation is assigned to CI.

Add Data Files and File → Open in a managed ImageTool use the same file
ingress controller. It applies acquisition metadata and passes the selected
array, loader, and preparation to the graph. The managed window uses one
replacement transaction for file opens, dialog and batch results, console
assignments, and direct source writes. Reload in the window and Reload in the
Manager ask the graph whether it owns the command. A saved program that pins
older workspace checkpoints can still need the import replay path.

Workspace storage changes use a separate widget installation method. They
preserve the published revision. An external xarray reader of an immutable
workspace payload acquires an object lease when the Manager accepts its data.
A later save can append a generation while the recorded array stays readable.

The PyQt6 run passed 6,149 tests across the Manager, workspace, viewer,
provenance graph, and replay. Six platform-specific cases were skipped on
macOS. A follow-up selection passed 266 overlapping tests after cleanup.
Ruff, mypy, the partition check, the HTML build, and the diff check passed.

A live publisher passes either the ordered input revisions read by the
calculation or the revision of a completed calculation. The Manager uses the
same inputs for a top-level result, a child, and a replacement. When opening
a window changes coordinates, dimensions, or dtype, it records that
preparation after the calculation. A second window can display the same
calculation revision if it needs no preparation.

The publication tests substitute a recipe that points to a different window.
They exercise Concat, Shift, Open Input, console calculations, and notebook
inputs at both tree levels. They check the recorded revisions, execute copied
code, and change the selected and substituted windows independently.

The Manager console records a supported array calculation when it runs. Its
handle retains the output revision and its ordered inputs before it has a window.
Opening a window or assigning the result to an ImageTool uses those recorded
edges. A plain window handle opens a graph-owned result even when the user has
entered no Python calculation.
A local Python program can keep several statements in one application. Its rows
identify the application and the position of the statement. A plain DataArray
assignment starts a checkpoint with no producing operation.

| Consumer | Recorded data and checks |
| --- | --- |
| Console, dialogs, batch transforms, and extensions | Capture the ordered operands and the view each operand used. Keep named intermediate results in the graph. The concat tests select the same input twice, including a calculated input, then execute copied code before and after a source replacement. |
| Inspector and clipboard | Query the displayed revision and its edges. Expand an input to inspect its recorded producer. Copy the selected statements or applications with their connections. Direct and historical Shift rows open the same editor. An unchanged selection keeps its followed source. |
| Edit, delete, revert, and reorder | Use the selected window and revision together. Keep a complete operation group together. Validate a batch before installing its results. A failed second edit leaves both windows unchanged. The program tests copy, remove, and revert individual statements. |
| Display filters, Refresh, Reload, and Auto | The recipe identifies display filters and layout steps. Install the raw value and its recorded filter together. Keep the previous checkpoint for workspace history. A materialized selection can supply its filter without reading the source chunk again. |
| Window placement and file inputs | Promotion preserves the entry ID and its bindings. Duplication copies internal bindings after all destination IDs are known. A stale copy retains its recorded values. File edits use the recorded load and selected file revision. When the sender provides the raw selection, eager and lazy preparation steps get their own revisions. |
| Workspace and ToolWindow boundaries | Export a version 3 recipe for saved documents, standalone ImageTools, and ToolWindows that require one. Bind a compatibility checkpoint through the workspace's ownership check. On import, preserve the saved executor when the payload cannot identify the inputs of an application. Older pending windows can still show and edit their saved history. |

Metadata edits identify each assignment by its application revision and, for a
recorded program, its statement position. The graph starts from the displayed
input of each edited step. A stale nested result and a stale top-level result
therefore keep the same data when the user edits metadata or pastes steps.
Refresh follows the source specified by the recipe. A bulk metadata edit
validates all results and installs them in one transaction. If a later window
rejects its data, the earlier windows recover their data and revision IDs.

Manager `replace_data()`, viewer `set_data()` and `replace_source_data()`,
and notebook watch updates use the same raw replacement plan. It takes
the metadata assignments from the published revision. When it reapplies
an assignment, it records the incoming array before dtype or angle conversion.
A manual write to a watched window uses the supplied array as its checkpoint.
A watch notification identifies the notebook variable. If a manual write
replaces a followed result, Refresh still uses the pending recipe, including
any unapplied metadata edit. A write from an installation callback uses the
last published revision and replaces the interrupted result.

On `dev-provenance-redesign` at `4df92098` with these working tree changes,
6,217 PyQt6 tests passed across the Manager, workspace, ImageTool, slicer,
module boundaries, graph, and replay. Six platform-specific tests were skipped.
The three public replacement cases also passed with conflicting history set on
the window. Ruff, mypy, the partition check, the HTML build, and the diff check
passed.

The workspace compatibility store keeps explicit checkpoints for legacy inputs.
Graph calculations can retain a lazy input expression until they need its values.

Two consumers of `partial = A + B` use the same revision of `partial`. A
regression test changes xarray's coordinate alignment from `inner` to `outer`
before publication. It checks the recorded input coordinates, the displayed
values, executed copied code, and Refresh for both consumers. A `qsel` call
with a calculated receiver and indexer records both revisions in order. Other
tests replace an input with the calculated result, close an input before publication,
and reopen a saved calculation with a coordinate that follows another window.
The momentum conversion test checks that the conversion consumes the original
setup revision and that the two steps form one editable group on that branch.
Accepting the setup editor without changes preserves the result revision.
The plain handle test opens a window with `tools[0].qshow(manager=True)` and
substitutes a window spec that points to another tool. The inspector, copied
code, and Refresh still use the source recorded in the graph.

The final console, graph, and related dialog selection passed 409 cases under
each Qt binding. The Manager provenance selection passed another 606 under
each binding.

The regressions are in `tests/test_provenance_graph.py`,
`tests/interactive/imagetool/manager/test_console_contract.py`, and the
Manager provenance and workspace tests. They include results with the old
window spec cleared or replaced, multi-window failure, copied-code execution,
source closure, workspace reload, and partial approval of stored Python.
The 167 cases selected for the metadata and Paste changes passed under both
PyQt6 and PySide6. They include copied code from a stale result, replacement
and revert of an assignment inside a saved program, rollback after the second
window rejects an edit, and a pending recipe that needs a display conversion.

The Manager and provenance selection passed all 1,545 cases under PyQt6 and
all 1,545 under PySide6. The workspace loading and storage selection passed
304 cases. The related wrapper, extension, and replay selection passed 1,156.
Ruff formatting and lint, mypy on 338 source files, the test partition check,
the HTML documentation build, and `git diff --check` passed. These were
targeted test runs.

The compatibility store retains data and code approvals for the version 3
format. The graph supplies a projection when that store needs a recipe. A
file received with only its prepared array retains that array as its checkpoint
and the load settings as its replay information. ToolWindow editor ownership
and native graph serialization remain in stages 4 and 5.

#### Grouped operation editing

One dialog can edit several recorded operations together. For example, the
momentum conversion dialog can set the work function and normal emission before
it converts the data. Keep each application and its input revisions in the graph.
The group marker identifies which applications share an editor. Replay and
Copy Full Code still use the individual operations in their recorded order.

Selecting any member on the result or under **Inputs used** must open the
same editor. In recorded history, start at the selected member and follow the
input edges on that result's branch. Resolve the complete sequence using the
marker kind, group ID, member index, and size. Validate the sequence with the
editor's existing group matcher. A group ID can also occur on another branch.
If the selected path has an incomplete or ambiguous group, keep its steps
available for inspection and explain why the group cannot be edited.

Restore the dialog with the recorded input to the first member and the settings
of every member. Use the selected member's focus marker to select its control.
Accepting any historical editor without changes must preserve the branch and its
input bindings. A group must also preserve its operation order, exact values, and
omitted arguments. For example, a momentum axis with automatic bounds must
remain automatic when the user edits the work function.

For a historical edit, replace the complete sequence on the selected branch.
The edited sequence can add or remove a setup step. Preserve the group's input
bindings, connect its new output to the downstream steps, and publish the
rebuilt result in one graph command. Other branches keep their recorded
revisions. Cancellation, a failed calculation, or a superseded edit publishes
no part of the replacement.

**Stage 3 acceptance case.** Convert an angle scan to momentum and derive two
results from the converted ImageTool. Under **Inputs used** in one result,
open the group from both a setup row and the conversion row. Check the focused
control and apply once without edits. Then change the work function. Compare
the selected result and its executed **Copy Full Code** with a direct
calculation. Check that the converted ImageTool and the sibling still display
their original revisions. Change the scan and Refresh the selected result to
verify that it uses the edited settings. Also exercise an edit that adds or
removes a setup step, a cancelled edit, and an incomplete group. Run the
Manager cases under PyQt6 and PySide6.

### 4. ToolWindow editors and figure sources

Bind the editor to durable output recipes by output ID. Compare effective settings
per output. In ktool, changing a conversion offset makes the converted output
stale. Restoring the offset before refresh clears that cause. Toggling the
Brillouin-zone overlay leaves the converted output current. Closing the editor
must still allow replay of its published operation.

Next cover a fitting tool with several outputs and asynchronous completion. Publish
the completed fit with the inputs and settings it used. Verify that an older fit
cannot overwrite a newer result. Exercise a figure that references ftool before
ftool has published an ImageTool. Test the Manager and ToolWindow paths under
both PyQt6 and PySide6.

### 5. Workspace storage and performance

Save shared graph records once and refer to them by ID from entries, editor
recipes, and figures. Save entry placement and needed checkpoints. Restore the
graph before constructing windows. Repeat the first acceptance test with an
actual `.itws` save and reopen **while C is stale**.
Check C's displayed data, recorded input, next input, Auto setting, and copied
Python before refreshing. Preserve stored-code trust checks when replaying.

Save the group markers with their operation applications. After reopening the
workspace, repeat the [grouped historical edit](#grouped-operation-editing)
from a child's recorded inputs and verify that it resolves the same members.

Begin measuring graph cost in stage 1. Extend the checks to the GUI and workspace
as those stages become available.

- Index dependents by source. Count node visits on a long chain and on a diamond
  in which two paths share an ancestor. Add an unrelated branch and verify that
  an update visits only its prerequisites and affected descendants.
  Evaluate a shared producer once per request. Reading provenance and status must
  leave lazy arrays uncomputed.
- Measure slice response time during repeated updates to a growing acquisition
  grid, including filtering and publication. Record a baseline on the same data
  and hardware. Coalesce superseded work and check that interaction remains usable
  while updates run. Set timing thresholds from measurements.
- Protect an authoritative copy when the last reproducible input would otherwise
  be lost. Keep required checkpoints reachable from live entries, retained history,
  stale results, pins, and in-flight work. Release temporary references on success,
  failure, and cancellation. Distinguish a stored source from an evictable cache.
- Hold the retained history fixed and repeat refresh, save, and reopen. Measure
  live array bytes and reachable workspace payloads. Share identical stored
  payloads where safe. Check physical file size after compaction separately from
  reachable payload size. Growth should correspond to retained results and their
  metadata.

### 6. Migration

Build the one-way importer last, after the new format passes its own round trip.
Translate explicit operations, file origins, input and output bindings, placement,
and checkpoints. If only a value survives, keep it as a checkpointed root with
any descriptive history that can be recovered. Infer data edges only from evidence
of a calculation. Check legacy fixtures and representative workspaces before the
default reader and writer switch.

## Handoff for the next chat

- **Current console decision.** The 2026-11-06
  [console contract](#console-contract) is authoritative and implemented. The
  verification evidence above maps each matrix row to tests. See the Manager
  migration record for the separate file and notebook registration checks.
  The implementation and test reports below are historical evidence and can
  contain superseded console expectations.
- **Starting point.** The main checkout inspected for this note was at `93d75f30`.
  The separate branch `dev-provenance-recipe-foundation` at `a1576d8e` contains
  typed graph code and an earlier four-stage plan in `.agents/plans/provenance.md`.
  That plan uses operation nodes. Compare candidate code from it against the
  requirements here for data nodes and operation edges.
- **Stage 1.** Commit `cc603569` on `dev-provenance-redesign` contains the
  Qt-free graph. Its acceptance test passes with Qt imports blocked. The
  tests also cover a replaced derived input, a failed refresh, a file load shared
  by two branches, and code generation for a concatenation with repeated inputs.
  The graph executes synchronous requests on its owner thread. The caller supplies
  the file reader. [Graph](../../src/erlab/_provenance/_graph.py) ·
  [Tests](../../tests/test_provenance_graph.py)
- **Graph cost.** Replacing the source of a 1,100-step chain visited each affected
  entry once, including both branches and their shared result. Refresh evaluated
  each stale step once. An independent branch received no visits. Repeated file
  refreshes kept one checkpoint for the current file value and one for an
  explicitly retained earlier result until its owner released it.
- **Pending producer edits.** Replace now records the displayed input and keeps
  its edited recipe for Refresh. The result stays stale until Refresh runs that
  recipe. Regression tests cover repeated replacement, redirected sources, and
  pins on virtual inputs.
- **Stage 2.** Commit `4df92098` implements Reload, command previews, Auto
  scheduling, and cancellation. In the
  file-backed acceptance case, Refresh of stale C uses B′ on screen. Reload reads
  the changed file once, rebuilds B, and records the rebuilt revision as C's input.
  A failed or superseded request publishes none of its prepared revisions.
- **Review fixes.** Auto preserves newer work through consecutive supersessions.
  The request planner tracks a producing step across successive revisions and
  shares its result across equivalent bindings. Workspace storage in stage 5
  will need to retain that producer identity.
- **Historical Python.** Copy Full Code walks the recorded revisions of the
  selected result. Notebook names identify exact checkpoint inputs. To emit a
  file read, the caller must verify the recorded version of the file source.
  Generated extension calls check the recorded script hash. The code adapter uses
  the existing ImageTool emitter. [Recorded code](../../src/erlab/interactive/imagetool/_provenance/_recorded_code.py)
- **Stage 3.** The working tree on `dev-provenance-redesign`, based on
  `4df92098`, connects the graph to the live Manager. Open Child Window and Open
  Top-Level Window record the same source revision. Replace Current records the
  previous result as a virtual input. Dialogs, batch transforms, named console
  inputs, and pasted steps record the values used during calculation. The tree
  supplies placement. The graph supplies the recorded and next inputs, Auto
  state, Refresh, Reload, and Copy Full Code.
- **Manager cases.** While C is stale, its recorded input opens in a separate
  ImageTool and its copied code reproduces C. Editing a historical aggregation
  updates only the selected branch. Selected rows share one Reload
  request. If a window rejects its replacement, the other windows and graph
  entries keep their previous values. [Manager adapter](../../src/erlab/interactive/imagetool/manager/_data_graph.py)
  · [Manager tests](../../tests/interactive/imagetool/manager/test_mainwindow.py)
- **Grouped historical edits.** On 2026-11-09, the
  [grouped operation editing acceptance case](#grouped-operation-editing)
  reproduced a disabled Edit action under a child's **Inputs used**. The
  Manager now resolves the group along the selected input path and opens the
  same dialog as the conversion result. Both Qt bindings passed the Manager
  cases for adding a work-function step, executing copied code, refreshing the
  edited child, and preserving its siblings. Two groups sharing a setup
  revision and group ID open with their own settings. An incomplete group is
  unavailable even when another branch contains its missing steps.
  [Manager tests](../../tests/interactive/imagetool/manager/test_mainwindow.py)
- **Ingress and compatibility.** The Manager can retain the actual variable
  selected by its file loader before applying a preparation step. Its graph
  records the display conversion to float64 and the requested angle conversion
  to degrees. A sender can supply only the prepared result. Its recorded file
  selection and preparation then form a single load with the displayed array as
  its checkpoint. New transforms of restored rows follow the same source in
  both placements. A stale restored child retains its displayed checkpoint
  and the binding for its next update. Any earlier history that cannot be
  bound stops at an explicitly identified checkpoint.
- **Restored displays.** A new result of a stale, filtered child retains the
  displayed selection and follows the child's pending update. Changing or
  clearing its filter and transpose preserves that relationship. A restored
  integer coordinate selected as data replays the conversion to the displayed
  dtype. The file recipe also retains angle conversion after a selection, so
  reloading a reopened file reproduces its coordinates in degrees.
- **Review fixes.** Window installation stays rollbackable until the graph
  rechecks its inputs. Console item edits preserve navigation history, and a
  failed multi-window update restores each window's undo and redo stacks.
  Superseded acquisition frames can be released after their last reference. A lazy
  selection reads the chunks it needs. File-backed chunks check their recorded
  file version. Evaluated selections remain available if the lazy source
  changes. ToolWindow output ownership remains in stage 4. Graph IDs and stale
  history across save and reopen belong to stage 5.
- **Initial acquisition check.** On the local PyQt6 system, six float64 frames grew
  from 77 to 82 planes of 256 × 256 samples (40–43 MB). Replacing a frame and
  moving the cursor incurred no graph payload copy on arrival. The median
  replacement and cursor calls took 22 ms and 1 ms. With Auto enabled, the
  subsequent event copied the consumed mutable frame once. For a freshly
  filtered view it copied the source and the filtered result. Immutable
  bytes-backed source frames required no source payload copy. These are
  measurements on one machine. These separate call timings did not establish
  cursor responsiveness during Auto. The review repair below measures cursor
  events across the complete update instead.
- **Existing performance limits.** The Gaussian filtering pauses measured on
  main are outside this rewrite's acceptance conditions. The earlier timing
  measurements below precede the ownership fix and do not measure its required
  mutable-input acceptance copy.
- **After stage 3.** In stage 4, bind the ktool conversion output to its input and
  effective conversion settings. Change an offset, restore it, and toggle the
  zone overlay. Check the output state after each action and after editor close.
- **Later in the rewrite.** Move the provenance operation implementations and
  their Python generation helpers out of the Qt package. Verify a catalog
  operation can execute and emit Python with Qt imports blocked.

The initial stage 1 checks passed from the repository root with the `ml` extra installed
and `ERLAB_TEST_DATA_DIR` set to a local `erlabpy-data` checkout. Both Qt runs
passed 178 tests.

```sh
uv run ruff format .
uv run ruff check --fix .
uv run mypy src
uv run python -m scripts.ci_test_groups --check-partition
for binding in pyqt6 pyside6; do
    QT_API="$binding" PYTEST_QT_API="$binding" uv run pytest -q \
        tests/test_provenance_graph.py \
        tests/interactive/imagetool/manager/workspace/test_store.py \
        tests/interactive/imagetool/test_module_split.py::test_provenance_layers_load_in_one_direction
done
```

After the pending recipe fix, Ruff, mypy, and the partition check passed again.
Running `uv run pytest -q tests/test_provenance_graph.py` with the same two
binding settings passed all 19 graph tests in each run.

Stage 2 and its review fixes passed the same Ruff, mypy, and partition checks.
The local PyQt6 run passed 1,115 tests after the recipe sharing fix. Use one
binding locally for graph changes and leave the binding matrix to CI.

```sh
QT_API=pyqt6 PYTEST_QT_API=pyqt6 uv run pytest -q \
    tests/test_provenance_graph.py \
    tests/interactive/imagetool/test_replay_graph.py \
    tests/interactive/imagetool/test_provenance.py \
    tests/interactive/imagetool/manager/workspace/test_store.py \
    tests/interactive/imagetool/test_module_split.py::test_provenance_layers_load_in_one_direction
```

At each handoff, record the branch, revision, exact checks run, and first remaining
case.

Stage 3 checks use PyQt6 locally. From the repository root, with
`ERLAB_TEST_DATA_DIR` set to the data checkout, the graph and Manager checks are:

```sh
QT_API=pyqt6 PYTEST_QT_API=pyqt6 uv run pytest -q \
    tests/test_provenance_graph.py \
    tests/interactive/imagetool/test_replay_graph.py \
    tests/interactive/imagetool/test_provenance.py \
    tests/interactive/imagetool/manager/test_mainwindow.py \
    tests/interactive/imagetool/manager/test_console.py \
    tests/interactive/imagetool/manager/test_console_contract.py \
    tests/interactive/imagetool/manager/provenance \
    tests/interactive/imagetool/manager/test_modelview.py \
    tests/interactive/imagetool/manager/test_wrapper.py \
    tests/interactive/imagetool/manager/test_extensions.py \
    tests/interactive/imagetool/manager/workspace/test_store.py \
    tests/interactive/imagetool/manager/workspace/test_trust.py \
    tests/interactive/imagetool/test_module_split.py::test_provenance_layers_load_in_one_direction
QT_API=pyqt6 PYTEST_QT_API=pyqt6 uv run pytest -q \
    tests/interactive/imagetool/test_imagetool.py \
    tests/interactive/imagetool/test_slicer.py
workspace=tests/interactive/imagetool/manager/workspace
QT_API=pyqt6 PYTEST_QT_API=pyqt6 uv run pytest -q \
    $workspace/test_loading.py::test_workspace_import_deduplicates_owned_revision_sources \
    $workspace/test_loading.py::test_workspace_import_keeps_closed_sources_detached \
    $workspace/test_loading.py::test_workspace_imported_history_keeps_each_source_binding \
    $workspace/test_loading.py::test_workspace_replacement_rehomes_matching_revision_checkpoints \
    $workspace/test_saving.py::test_background_save_owns_removed_revision_sources \
    $workspace/test_saving.py::test_background_save_owns_removed_transformed_dask_source \
    $workspace/test_saving.py::test_manager_workspace_save_as_rebinds_lazy_data_to_new_document \
    $workspace/test_saving.py::test_workspace_source_ownership_follows_promoted_lazy_window
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run python -m scripts.ci_test_groups --check-partition
uv run --directory docs make html
```

The file-code checks cover a file rewritten with equal values but a different
integer width. Copy Full Code then requires the recorded checkpoint.
For a custom loader, Copy Load Code emits the loading call and Copy Full Code
uses the recorded array. Chunking an originally unchunked backend can change the
last digits of a floating-point reduction. The displayed result remains available.

Before the latest review, Stage 3 selections passed 3,243 distinct PyQt6 cases. Ruff formatting
and lint, mypy on 338 source files, the test partition, the HTML build, and
`git diff --check` passed. Those checks omitted the three cases that reopened Stage 3.

### Stage 3 review repairs

- Refresh validation checks array metadata. Installation uses ImageTool's normal
  compute threshold. The 320 MB regression keeps both the source and result
  chunked, with and without an accepted filter. A small selection computes once
  and retains its exact result after the source changes.
- Closing pauses Auto without discarding pending work. Cancelling the unsaved
  workspace prompt resumes it. Discarding the workspace removes the pending
  entries. The regression uses the close prompt and checks a later source update.
- Auto seals already owned eager inputs of at least 8 MiB in a worker. Mutable
  producer buffers are copied at acceptance under the ownership contract above;
  immutable numeric values are then shared without another payload copy. Before
  accepting a worker result, the
  graph checks the input and target revisions. A relevant change discards the
  attempt; an unrelated row change does not restart it. Tests cover source
  replacement, removal, copy failure, and recovery. Workers release their source,
  result, and error payloads after completion or an interrupted wait. Dedicated
  lifetime tests disable garbage collection while checking those releases.
- Compatibility history reuses an already protected graph input through a
  retained reader. This removes the duplicate full-source copies from window
  publication. Explicit history reads and workspace saving still get isolated
  data. The regression mutates the sender's arrays and saves and reopens both
  current and historical displayed results.

Before the ownership repair, the local timing sweep used four preallocated float32 updates growing from
129 to 132 planes of 512 × 512 samples (135–138 MB). A 5 ms timer moved the source
cursor while a 450 ms timer replaced the frame. Each case ran three times, with
and without a Gaussian filter along `x` with sigma 0.5. The table gives the range
of the largest cursor-event gap in each run, including frame arrival, filtering,
Auto, history capture, publication, and inspector updates.

| Input | Auto off | Auto with synchronous capture | Auto with worker capture |
| --- | --- | --- | --- |
| Plain | 36–65 ms | 46–63 ms | 34–39 ms |
| Gaussian filtered | 266–299 ms | 303–344 ms | 242–296 ms |

All 72 updates and 48 Auto publications passed. Mutating the original acquisition
buffers left the complete recorded source arrays unchanged. Each plain update
copied one full source on a worker; a filtered update copied the source and its
filtered result there. No full-size checkpoint copy ran on the GUI thread. Its
remaining 1.05 MB checkpoint copies took at most 0.155 ms. Installation took
13–14 ms. Worker capture and publication intervals, excluding source arrival,
still included cursor-event gaps of 26–30 ms. The eager Gaussian calculation
itself took 222–283 ms with Auto off or worker capture. These local measurements
support removing the checkpoint-copy regression; they do not establish
uninterrupted filtered acquisition.

After these repairs, the broad PyQt6 graph and Manager selection passed 2,657
tests, the standalone viewer selection passed 616 tests, and the affected graph,
wrapper, history, and workspace selection passed 595 tests. These selections
overlap. Both saved-history variants passed after the final reader change.
The dedicated success, failure, interrupted-wait, and unrelated-update checks
passed under PyQt6 and PySide6. Ruff formatting and lint, mypy on 338 source
files, the partition check, `git diff --check`, and the HTML build passed.
The changes remain uncommitted on `dev-provenance-redesign`, based on `4df92098`.

### Comparison with main

On 2026-11-06, the same acquisition harness ran against main `93d75f30` and this
working tree. Both used the same interpreter, dependencies, PyQt6 offscreen
platform, preallocated 135–138 MB frames, 5 ms cursor timer, 450 ms update timer,
and warmup. Three runs per case alternated between main and this branch. Each
run completed four updates and checked the final child array.

The shared harness used a nonuniform spatial pattern and committed each filter
with `emit_edited=True`. The earlier uniform frames could hide main's failure to
update its child after a filter applied without that signal. This correction
ensures both versions do the requested work and produce the same result.

| Largest cursor-event gap in each run | Main | Current working tree |
| --- | --- | --- |
| Plain, Auto off | 36.7–38.8 ms | 34.5–38.2 ms |
| Plain, Auto on | 58.5–71.2 ms | 37.8–38.4 ms |
| Gaussian filtered, Auto off | 241.7–280.5 ms | 246.9–259.5 ms |
| Gaussian filtered, Auto on | 754.6–906.5 ms | 245.0, 760.4, 254.6 ms |

All 24 trials and 96 updates passed the final-data checks. Main published twice
per filtered update; this branch combined those changes into one publication.
The remaining roughly 250 ms filtered arrival pause is present on main. These
measurements do not identify it as a regression from the rewrite. Auto improves
the complete workflow in this comparison.

The 760.4 ms outlier on this branch must not be omitted. Its filter-and-acceptance
callback took 681.7 ms. The cursor gap ended before the child publication started;
that later publication took 15.3 ms. The precise cause within the filter callback
has not been isolated. Thus the baseline comparison corrects the regression
classification, but does not establish uninterrupted filtering or a 300 ms upper
bound. This existing performance limit alone is not evidence that stage 3 made
acquisition slower.

### Ownership repair verification

The exact failure cases failed before the ownership change and pass afterward:

| Case | Required result |
| --- | --- |
| Replace the source of an Auto-off 320 MB lazy result | The old result stays chunked and its actual slicer reads the original values. |
| Close that source while its separate result window stays open | Both the slicer and recorded revision remain readable. |
| Pause worker capture, then reuse the producer's buffer before any notification | The source window, recorded input ID, and refreshed result retain the accepted values. Tested through the viewer and same-process `replace_data`. |
| Reuse values, coordinates, and nested metadata immediately after acceptance | The recorded revision retains all original content. |
| Edit console data or a separately opened historical input | Window edits still work; the recorded input remains unchanged. |

The lazy tests no longer expect source replacement to make retained data
unreadable. Existing file-version guards and explicit checkpoint materialization
tests still pass. The console retains both `tools[i].data += value` and
`tools[i].data[...] = value`; mutable access first detaches the window buffer.

Validation on 2026-11-06:

- The broad PyQt6 graph, Manager, workspace, and viewer selection passed 3,263
  cases and exposed two mutable-access failures. Both were corrected and checked
  in the final console and history run: 159 cases passed. An earlier copy-count
  assertion was replaced with the observable requirement that the edit consumes
  the exact accepted input revision and preserves its values.
- The latest viewer and focused ownership selection passed 629 PyQt6 cases.
  The graph and Manager follow-up passed its 477 other cases; its sole obsolete
  console copy-count assertion is included in the corrected final run above.
- Sixteen focused ownership, lazy-output, and worker lifetime cases passed under
  PySide6. The final console, history, and exact failure-case selection passed
  another nine PySide6 cases. These selections overlap.
- Ruff formatting and lint, mypy on 338 files, the partition check,
  `git diff --check`, and the HTML build passed.

The earlier timing tables describe the superseded borrowed-buffer strategy.
The current contract requires a values copy when accepting a mutable eager
buffer, on the calling thread, before returning to its producer. Immutable
bytes-backed values avoid that copy. The fix does not claim that this mandatory
copy is free or that arbitrary lazy callbacks can snapshot changing external
state. Changes remain uncommitted on `dev-provenance-redesign` at `4df92098`.

### Console, lazy publication, and fetch verification

The regression assertions were established before these production fixes. They
use console execution, Manager publication, the actual slicer, public `fetch()`,
and executed Copy Full Code. They do not select a replay implementation or assert
graph IDs. A separate graph test checks that an existing coordinate guard cannot
leave values unguarded.

- Console item, `.loc`, and `.values` edits invalidate existing followers. A new
  Aggregate result and its copied code use the edited values. The old result's
  copied code retains the old values. This also passes for filtered inputs, a
  retained alias, an edit followed by a failed statement, and a calculation in
  the same cell. Metadata edits are visible to a fetch in that cell. An alias of
  replaced data cannot overwrite the new source. Read-only access leaves
  followers current.
- A newly published lazy Aggregate result remains sliceable after source closure
  when its file is unchanged. Replacing that file makes its slicer reject the
  read before the first Refresh. The same check passes when the operation drops
  file encoding. It also covers the review's separate top-level window case.
- Same-process fetch returns an independent, editable DataArray for eager and
  lazy sources, with and without an accepted filter. Values, coordinates, and
  nested attributes are isolated. Fetching lazy data retains its chunks.

Validation on 2026-11-06:

- The PyQt6 Manager and viewer modules passed 815 cases. The graph and console
  modules passed 198 cases. Five focused cases also checked the later metadata
  alias and lazy-fetch additions; these overlap the module runs.
- The focused PySide6 ownership, console, publication, and retrieval selection
  passed 25 cases. Eight further cases checked metadata aliases, eager and lazy
  fetch, file validity, and coordinate guards. These selections overlap.
- Ruff formatting and lint, mypy on 338 source files, the test partition check,
  and `git diff --check` passed. The broad console run first exposed an outdated
  Manager test double; it was updated to implement the current runtime contract.

These checks address the three review findings. They do not add a claim about
the cost of the mutable-input acceptance copy or console content hashing.

### Console boundaries and unavailable files

The next review found four gaps in those tests. The fixes now enforce these
behavioral requirements:

- Commit a console edit before a filter, filter removal, or axis swap consumes
  it. The resulting Aggregate and executed copied code agree for ordinary and
  filtered inputs. Existing results retain their original values.
- Install recorded file guards on slicer coordinates as well as its values.
  The new root-window cases fail with the old installation code. The promoted
  child cases also pass, but their later slicer rebuild had hidden the defect.
- Show one unavailable message with the file path, retain the last readable
  cursor state, and keep expected file errors out of the global alert handler.
  Tests replace the file both before navigation and during the chunk read.
  Reload recovers through an open source. A closed source becomes a fixed input;
  the message explains that the current file must be opened in a new ImageTool
  and the calculation repeated. Unrelated read errors still propagate.
- Release a temporary console DataArray after its cell. Later unrelated cells
  do not hash it. Retained DataArray, NumPy, sliced NumPy, and coordinate aliases
  remain editable. A new temporary DataArray obtained after a NumPy alias also
  publishes metadata edits.

The 135,266,304-byte reproduction retained only `tools[0].data.shape`. Tracking
was empty afterward. Across eight unrelated cells, the median remained 0.1 ms
before and after that read, compared with the reproduced 56.5 ms afterward before
the fix. This measures temporary-view lifetime only. Exposing editable data still
requires a copy, and retained mutable aliases still require content checks.

The PyQt6 graph, console, Manager, viewer, and history selection passed 1,140
cases. The focused PySide6 selection passed 35 cases. Four subsequent focused
cases check unrelated read errors and metadata edits after retaining NumPy
aliases and passed under both bindings; three overlap the earlier selections.
Ruff formatting and lint, mypy on 338 source files, the partition check,
`git diff --check`, and the HTML documentation build passed. The changes remain
uncommitted.

### Independent console snapshots

The user selected an explicit snapshot contract on 2026-11-06. Reading
`tools[i].data` returns an independent, editable DataArray. Changes to its values,
NumPy views, coordinates, or nested attributes do not affect the window. This
also applies to temporary expressions such as `tools[i].data[...] = value`.
Lazy snapshots retain their chunks and recorded file guards.

Use `tools[i][...] = value` to edit the current window and record the operation.
These edits immediately mark followers stale and preserve earlier recorded
inputs. Assigning a plain DataArray through `tools[i].data = value` starts a new
checkpoint. It does not retain the window's previous reload recipe. Assignments
from live handles and recorded expressions retain their provenance behavior.

The former alias tracker and its cell-boundary content checks were removed.
The Manager does not retain a returned snapshot. An unrelated console cell does
not inspect snapshots retained by user variables. The baseline tests reproduced
window changes from aliases and old reload recipes after plain reassignment
before this contract was implemented.

Validation: all 144 console cases passed under PyQt6 and PySide6. These include
eager, lazy, filtered, temporary, and retained snapshots; direct edits before
filtering or axis swaps; source replacement and closure; and executed copied code
for earlier and newly calculated results. Ruff formatting and lint, mypy on 338
source files, the test partition check, `git diff --check`, and the HTML
documentation build passed. The changes remain uncommitted.

### Console operand and snapshot separation

The follow-up review found two older provenance defects and a copy regression:
derived handles exposed mutable arrays with unchanged recipes, saved methods
calculated from old data while recording current inputs, and ordinary handle
reads called the new public snapshot getter. The first 36 regression cases
reproduced 17 failures before the production changes.

Calculations and publication now use an internal operand that pairs data with
its recorded inputs. Public `.data` access returns an independent snapshot for
both window handles and calculated results. Mutable attribute access is also
isolated. Saved methods and accessors resolve the current operand when called;
an already constructed coarsening operation retains the operand used to construct
it. Console help and wrapped-call signatures retain the same public parameters.

Derived results own their values, coordinates, and metadata. Eager mutable output
values are copied once when accepted into a result handle; values backed by
immutable bytes can be shared. Lazy results retain their computation without
materializing it. Ordinary read-only operations use the owned input directly.
Unknown methods and calls with callbacks receive a detached receiver. This
preserves source data even when a callback edits values or nested metadata.

The final selection contains 185 console cases. PySide6 passed the complete
module. PyQt6 passed 184 cases in the module run and the remaining case after
correcting its fixture: standard xarray rejects mutation of indexed coordinates,
so the callback test now uses a writable auxiliary coordinate. The test checks
repeated calculations as well as unchanged source coordinates. The other new
cases cover snapshot exports, retained methods and accessors, executed copied
code, Refresh, callbacks, lazy loading, and copy counts.

For the same 135,266,304-byte volume used in the review, eight repeated shape
reads and eight plane selections each made zero full-source copies. Median
times were 0.049 ms for shape and 0.222 ms for `isel(scan=0)`. These measurements
cover those reads, not explicit snapshots or the acceptance of new mutable
calculation results.

Ruff formatting and lint, mypy on 338 source files, the test partition check,
`git diff --check`, and the HTML documentation build passed. Changes remain
uncommitted.
