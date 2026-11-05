# Provenance contract and implementation plan

## Contract

Users must be able to understand an action from the visible data, recorded operations,
and source choices. Window ownership, open windows, caches, and retained checkpoints
must not silently change its meaning.

Preserve one complete source-to-result derivation graph for each result. The graph
records the source definitions, operation parameters, and every primary and auxiliary
input. A window displays a graph output. Displaying the output as a child, a top-level
window, or a replacement does not select a different source policy.

Equivalent operations on equivalent sources must produce equivalent recipes and
behavior whether they come from a dialog, a pasted step, the console, or a tool output.
Copied code, dependency status, provenance display, and execution must use this same
derivation.

## Source and execution rules

- An external source records how to load or update a file, watched variable, or
  acquisition source. Reload resolves the source and repeats its recorded operations.
- A memory source owns its available data independently of the window that displays
  it. It is a valid terminal input and has no external read to repeat.
- A fixed snapshot explicitly selects one historical revision. Its fixed state must
  be visible. A snapshot stays fixed during update and reload.
- Update uses current observed source revisions and recomputes affected operations.
  It does not require an external read.
- Reload refreshes external sources and evaluates the recorded graph. It resolves a
  shared source once per request. Closing a window does not select a different recipe.
- Historical replay uses the exact recorded input revisions. Checkpoints preserve
  history; their availability does not define ordinary reload policy.
- A failed or unavailable external source reports the failure. It does not silently
  substitute an old checkpoint and report a successful reload.
- Replacement keeps the previous output's producer as its upstream input before the
  window displays the new output. Removing a self window binding does not freeze the
  producer. Actual dependency cycles remain invalid.
- Evaluation records the recipe and input revisions used. Publish a result only when
  its required inputs have completed and its recipe and source versions are current.
  A failed or cancelled evaluation preserves the previous result.
- Acquisition updates must keep slicing responsive as the scan grid grows. Filters,
  deferred fits, and result publication must preserve compatible cursor and view state.

## Implementation boundaries

Use the existing typed operation catalog, replay compiler, saved recipe model, revision
store, code-trust checks, and Qt update hooks. Keep one saved semantic representation.
Extend or normalize that representation where needed; do not maintain a second saved
graph alongside the recipes.

Separate source and producer identity from Manager window identity. Window bindings
can locate an available materialized result. They cannot change its producer recipe
or select a checkpoint instead. Keep historical observations separate from source
policy. Every source-view transform must be applied exactly once.

Import old saved data at the parser boundary. Existing records do not reliably
distinguish intentional fixed inputs from automatic captures made by replacement.
Preserve those records as visible fixed inputs. Provide an explicit conversion to the
recorded producer where possible. Do not infer intent from labels or tree placement.
Keep unused legacy fallbacks lazy while the old execution paths remain active.

## Four stages

Use `dev-provenance-redesign` as the integration branch. Each stage has its own PR
against that branch. After a stage is reviewed and merged there, start the next stage
from the updated integration branch. Keep stage PRs out of `main`.

Stages 1 and 2 prepare the shared machinery. Stage 3 switches the affected user actions
together. Stage 4 removes the old paths. After all four stages pass their checks, open
one final PR from the integration branch to `main`. Squash that final PR to keep one
provenance redesign entry on `main`.

### Stage 1: Recipe construction

Status: implemented; awaiting review and merge.

Add pure authoring primitives that use the existing `ToolProvenanceSpec`, `ScriptInput`,
and `ReplayStep` models. Construct a producer input separately from an explicit fixed
snapshot. Append typed operations and attach selected fragments through the same
primary-input recipe constructor. Preserve auxiliary edges and explicit input policies.
Construction must be independent of Manager nodes, array capture, and result placement.

Keep all current UI, reload, and workspace import paths unchanged. Schema 3 remains the
saved representation in this stage. The new primitives are internal foundations for
the evaluator and integration stages, not a new public API.

Completion criteria:

- Direct construction and fragment attachment give equal recipes for unary and
  named-input operations.
- The existing compiler executes these recipes with the required revision resolver.
  Copied code executes correctly for followed producers and explicit caller inputs.
- Producer inputs retain their source recipe without a window binding. An explicit
  snapshot resolves its exact recorded data even if its historical producer changes.
- JSON round trips preserve steps, source choices, and dependencies.
- Construction preserves its input values and rejects ambiguous or missing inputs.
- Existing dialog, paste, replacement, and reload regression tests retain their
  current results under both Qt bindings.

### Stage 2: Shared evaluation

Status: pending.

Implement one evaluation request with update, reload, and historical-replay modes.
Normalize external, memory, and fixed sources at the graph boundary. Resolve durable
producer identities independently of windows. Pass the request mode through nested
ImageTool and ToolWindow recipes. Reuse the compiler and revision store, and share
source reads within a request.

Prepare candidate data and execution records before publication. Check source
generations, support cancellation, and integrate deferred tool completion. Tool
updates that mutate widgets or start fits need an explicit preparation and commit
boundary; a context manager alone does not make them transactional.

Completion criteria: internal evaluation tests cover all three modes, mixed file and
memory inputs, fixed snapshots, shared sources, missing inputs, nested and deferred
results, cancellation, and source changes during evaluation. Existing user actions
retain their behavior until stage 3.

### Stage 3: User action integration

Status: pending.

Route dialogs, paste, replacement, console results, child creation, and tool outputs
through the shared builders and evaluator. Derive dependency status, displayed
provenance, and copied code from the same recipe. Show explicit source choices and
migrate old captures visibly. Update the user documentation when the contract changes.

Completion criteria: equivalent action paths have equivalent graph meaning and reload
results. Verify disk-only updates, changed auxiliary inputs, closed producer windows,
workspace round trips, growing scan dimensions, filters, and uninterrupted slicing.
Reload does not accumulate a transformation on its previous output. Historical
restoration remains exact. Full copied code must preserve fixed input choices instead
of rebuilding their historical producers. Validate Qt-facing paths under PyQt6 and
PySide6.

### Stage 4: Legacy removal

Status: pending.

Remove superseded authoring and execution decisions after all runtime callers use the
shared path. Retain compatibility import at the parser boundary for supported saved
formats. Remove duplicate source-policy flags and caches only when their replacements
cover every caller.

Completion criteria: one runtime source-resolution and recipe-construction path
remains. Compatibility fixtures, the framework regression matrix, and the required
repository checks pass.

## Validation and progress

Follow the repository checks in `AGENTS.md` and the task instructions it links. Reuse
passing checks while their code is unchanged. Record completed checks and material
decisions here before opening each stage PR. Keep performance claims tied to measured
evidence and preserve failures or unresolved limitations in the handoff.

Stage 1 decisions:

- Keep schema 3 and input version 2. Add internal builders without changing existing
  parsers or call sites.
- Use a producer recipe for a followed input and an explicit revision for a fixed
  snapshot. Keep optional producer history on a snapshot for explanation and restore.
- Attach only fragments with a declared, unbound primary input. Extract the selected
  steps first; attachment must not silently discard an existing producer or transform.
- Shared evaluation and independent ownership of live memory sources belong to stage
  2. Stage 1 can represent recorded memory data as an explicit snapshot.
- The current full-code emitter cannot emit a graph containing resolved live values.
  Without a revision resolver it can instead expand a fixed snapshot's historical
  producer. Stage 1 leaves that existing behavior unchanged. Stages 2 and 3 must give
  runtime replay and full-code emission a common interpretation of fixed sources.

Stage 1 validation:

- Ruff formatting and lint checks passed.
- The 25 recipe-builder tests passed under PyQt6 and PySide6 on Python 3.14.3.
- The existing provenance and replay-graph modules, bound-shift clipboard tests, and
  Shift dialog reload and placement tests passed: 902 tests under each Qt binding.
- After rebasing onto the Figure Composer update in PR #668, the 25 builder and five
  affected Manager tests passed again under each binding. Unchanged provenance and
  replay-graph checks were reused.
- `uv run mypy src` passed for all 333 source files using a Python 3.13 environment
  with the optional ONNX dependency installed. The initial Python 3.14 environment
  lacked that dependency; no source change was needed for the typing pass.
- The CI partition check and `git diff --check` passed.
- Independent review found no remaining stage 1 issues. Existing user-action call
  sites, execution, code emission, and workspace parsing remain unchanged.
