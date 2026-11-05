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

## Proposed implementation choices and six-stage plan

The choices below are proposals. The examples above define the required behavior.
In particular, the precise Reload policy for an already stale row is still open.

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

| Command | Proposed behavior |
| --- | --- |
| Replace | Protect irreplaceable input values, evaluate the operation, then publish its record and move the existing entry to the new revision together. Invalidate dependents that follow the entry. |
| Reload | For a live row, use its refresh bindings and rebuild the required inputs from their origins. Honor fixed inputs. Perform each requested file read even when a cached result exists. |
| Refresh | Use the current bound inputs and effective settings. Process stale prerequisites in dependency order. Auto schedules eligible results. |
| Historical replay and Copy Full Code | Use the applied parameters and recorded input revisions of the selected result. Exact restoration requires a checkpoint or a verifiable original source. |

Under the proposed Reload policy, Refresh of stale C uses the available B′.
Reload of C first rebuilds B′ from its origins. Reading a changed file produces
a new revision. If the original file is gone, show whether the recorded result
can still be restored from a checkpoint.

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

Route file and notebook ingress, watched updates, dialogs, paste, and console
results through the same graph commands. Record a selection or filter as a step
when a calculation consumes the displayed subset. Editing a shared historical step
creates a revised branch for the selected result. Test that child and top-level
placement give identical lineage, that stale C shows both inputs, and that its
copied Python reproduces the displayed result.

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

- **Starting point.** The main checkout inspected for this note was at `93d75f30`.
  The separate branch `dev-provenance-recipe-foundation` at `a1576d8e` contains
  typed graph code and an earlier four-stage plan in `.agents/plans/provenance.md`.
  That plan uses operation nodes. Compare candidate code from it against the
  requirements here for data nodes and operation edges.
- **Stage 1.** The working tree on `dev-provenance-redesign`, based on `e3315fbb`,
  contains the Qt-free graph. Its acceptance test passes with Qt imports blocked. The
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
- **Next case.** Make B file-backed in the first acceptance test, replace B, and
  change the file while C is stale. Settle which origins Reload of C must read,
  then test Reload alongside Refresh. Auto scheduling, Manager integration, and
  the workspace round trip are still pending.
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

At each handoff, record the branch, revision, exact checks run, and first remaining
case.
