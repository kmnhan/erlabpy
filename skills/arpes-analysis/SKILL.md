---
name: arpes-analysis
description: Conduct, teach, and automate reproducible ARPES analysis with ERLabPy in executable Jupyter notebooks. Use when an agent must guide a user or analyze data by loading and inspecting ARPES datasets, calibrating the Fermi level, determining normal emission, converting angles to momentum, extracting or fitting EDCs and MDCs, tracing dispersions, creating constant-energy maps, producing publication figures, using ERLabPy interactive tools, or troubleshooting ERLabPy and xarray-lmfit.
---

# ERLabPy ARPES Analysis

## Operating contract

- Create or edit an executable Jupyter notebook unless the user requests another
  artifact.
- Work autonomously when the data and physical assumptions support a decision. Ask for
  the minimum missing physical input when they do not.
- Inspect data, execute the notebook from a clean kernel, inspect the rendered results,
  and correct invalid analysis before delivery.
- Keep the loaded data unchanged. Assign each calibration or transformation to a new,
  semantic variable.
- Put all input paths, physical parameters, fit choices, and calibration values in
  visible cells. Do not rely on hidden kernel or GUI state.
- Use public ERLabPy and xarray-lmfit APIs in copied or generated code.
- Treat a static normal-emission estimate as a candidate, regardless of agent
  confidence. For publication or quantitative momentum conversion, require an approved
  calibration. If none exists, open ImageTool or KTool for the user and wait for the
  accepted values before momentum conversion.

## Coordinate compatible skills

Treat this skill as the coordinator for the scientific workflow and final notebook.
Select only the smallest useful set of available auxiliary skills. Use a notebook skill
for notebook mechanics, a visualization skill for rendered-output inspection, or a
file-inspection skill for its supported formats. Do not load overlapping general
workflow guidance or create separate analysis pipelines.

- Keep one notebook as the primary reproducible artifact unless the user requests a
  different split.
- Use one visible configuration and provenance section for all analysis stages.
- Load each raw input once. Pass semantic notebook variables between calibration,
  conversion, fitting, and plotting stages.
- Apply the calibration and physical decision rules in this skill to ARPES decisions.
  Use auxiliary skills only for their specialized mechanics. Follow user and
  higher-priority instructions when they require a different choice.
- Execute and inspect the combined notebook as one document. Do not validate isolated
  fragments and then assume that the assembled notebook works.
- If no compatible auxiliary skill is available, create, execute, and inspect the
  notebook directly with the tools that the agent environment provides.

Use these imports in complete examples:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import erlab
import erlab.analysis as era
import erlab.plotting as eplt
```

Import `erlab.interactive as eri` only in a cell that uses a GUI tool to keep a headless
analysis notebook executable when no optional Qt binding is installed.

## Data intake

- Reuse a notebook variable only when an earlier visible cell creates it from a
  reproducible input.
- Load files with `erlab.io.load` and an explicit `Path`. Select a loader explicitly
  when automatic loader selection is ambiguous.
- Keep loader settings, scan identifiers, and concatenation choices in the notebook.
- Build an alignment table before combining scans. Include the sample or cleavage,
  acquisition order, motor coordinates, analyzer mode, photon energy, and log notes.
  Put files in one alignment group only when the sample geometry did not change.
- Use the user-specified experimental log as the authority when it conflicts with
  embedded analyzer metadata. Compare scalar angle coordinates with raw attributes and
  the log. Restore a missing or incorrect scalar coordinate before conversion and
  record the source value.
- Use a semantic raw variable such as `raw_data` or `gold_reference`. Do not overwrite it
  with corrected or converted data.
- Stop and report missing files, unsupported layouts, or incomplete acquisition
  metadata. Do not create synthetic replacements unless the user requests simulation.

## Analysis order

1. State the objective, requested outputs, assumptions, and success criteria.
2. Record package versions, input paths, loader settings, and physical parameters.
3. Load the data once and keep an unchanged raw-data variable.
4. Inspect `dims`, `coords`, `attrs`, units, coordinate order, monotonicity, finite
   values, and sampling. Plot representative raw slices. For weak or anisotropic data,
   also prepare energy-averaged, coordinate-aware smoothed, normalized, and derivative
   views. Keep the raw view beside them.
5. Establish the Fermi level. Read
   [references/fermi-calibration.md](references/fermi-calibration.md) only when the
   energy zero is unverified, a reference must be fit, residual curvature must be
   assessed, or a photon-energy-dependent correction is needed. Otherwise record the
   accepted calibration provenance and skip the fit.
6. Determine normal emission and convert to momentum only after accepting the energy
   scale. Read [references/momentum-conversion.md](references/momentum-conversion.md)
   only for angle-space data that needs momentum coordinates or when an existing
   conversion must be validated.
7. Read [references/curve-analysis.md](references/curve-analysis.md) only when the task
   includes EDC or MDC extraction, curve fitting, or dispersion tracing.
8. Read
   [references/publication-plotting.md](references/publication-plotting.md) only when
   the task includes constant-energy maps, figure construction, fit overlays, or
   figure export.
9. Summarize results, uncertainties, calibration provenance, failed checks, and
   limitations next to the relevant outputs.

Skip an inapplicable stage explicitly. Do not silently assume that the energy or
momentum axes are calibrated.

## Notebook requirements

- Keep cells small and runnable in top-to-bottom order.
- Put selections and fit parameters in a short configuration cell before their use.
- Display compact dataset summaries instead of full arrays.
- Show raw data beside processed data when a transformation can change interpretation.
- Show fit data, best fit, and residuals. Report failed fits and unconstrained
  parameters instead of hiding them.
- Save final figures from explicit figure objects. Record the output paths.
- Execute once from a clean kernel. Repeat only when randomness, in-place mutation,
  calibration-file writes, or other rerun-sensitive behavior is present.

### Reproducibility acceptance gate

Treat the delivered `.ipynb` as the primary executable artifact, not as a report that
points at a separate analysis. Before delivery, inspect its code cells and reject the
notebook unless all of the following are true:

- The notebook visibly loads the raw experimental files and authoritative metadata,
  defines every physical calibration, performs every substantive transformation and
  fit, constructs each final figure from explicit figure objects, and saves the stated
  numerical products.
- Paths, scan groups, accepted GUI-derived values, fit windows, smoothing widths,
  normalization rules, masks, and uncertainty choices appear in executable cells.
- `display(Image(...))`, saved PNG or PDF files, and cached NetCDF files are only
  optional summaries or acceleration paths. They must not replace the code that
  generates the displayed result from raw data.
- Do not use `%run`, `runpy`, `exec(open(...))`, subprocess calls, or imports of local
  build scripts as the primary analysis path. A helper script may generate notebook
  cells, but its substantive source must be materialized in those cells.
- A `REBUILD_FROM_RAW = False` switch does not make a report notebook reproducible.
  The raw-to-result path must be present and runnable without changing hidden source
  files or relying on outputs created by a previous kernel or external pipeline.
- Execute the notebook top to bottom in a clean kernel with pre-existing derived
  figures hidden or absent when practical. Confirm that the execution regenerates the
  numerical products and inline figures, then inspect the rendered results.

Fail the handoff rather than call a notebook reproducible when any substantive step is
available only in an external script or pre-rendered artifact.

## Decision rules

- Prefer an approved reference calibration over inference from sample data.
- Reuse one `(alpha_normal, beta_normal)` pair for all photon energies in one unchanged
  sample alignment. Do not fit independent normal-emission offsets by photon energy.
- Do not transfer offsets across an alignment change, even when the sample name and
  motor readbacks appear similar.
- Treat implicit zero offsets as defaults, not measured normal-emission values.
- Do not infer an angle-dependent Fermi correction from sample bands.
- For `gold.poly`, use reliable `sample_temp` metadata. Use `fast=True` when that
  temperature is missing or unreliable, and supply a unit-checked resolution estimate.
- Do not use symmetrized data to justify a normal-emission calibration.
- Do not use intensity maxima or apparent brightness symmetry to locate normal
  emission. Matrix elements can suppress one side of the same contour.
- Exclude analyzer-acceptance boundaries and detector masks from derivatives,
  correlations, and center estimates. The center of a circular acceptance mask is not
  the sample normal-emission angle.
- Compare photon energies only through identified corresponding features. Bulk bands
  can change with out-of-plane momentum, and matrix elements can reveal different
  bands. Do not align complete images or all detected ridges as if they were one band.
- Do not interpret optimizer success alone as a valid physical fit.
- Use Voigt spectral peaks by default. Tie their Gaussian sigma parameters to one
  instrumental-width parameter with expressions.
- Do not track a band through missing intensity only because the optimizer returned a
  peak center.
- Treat a constant-energy contour as an intensity map unless the user explicitly asks
  for line contours.
- Preserve descending coordinates when APIs support them. Sort explicitly when an API
  requires monotonic increasing coordinates, and record the change.
- Confirm energy units before supplying temperature, resolution, or fit windows.

## Interactive tools

- Use optional GUI tools when they are available and visual interaction will reduce
  uncertainty. Do not install Qt only to use a diagnostic GUI unless the user requests
  it.
- Use ImageTool when exploration or region selection benefits from linked views.
- Use `goldtool` when Fermi-edge windows or model choices need visual adjustment.
- Use `ktool` when normal emission, geometry, or momentum-conversion candidates need
  interactive comparison.
- Use `ftool` when 1D or sequence fits need visual initialization or propagation.
- Use Figure Composer when visual figure construction is useful.
- Make KTool or ImageTool a required user handoff for publication or quantitative
  momentum conversion when no approved normal-emission calibration exists. Do this
  even when the agent considers one static-image candidate obvious. Open the tool, tell
  the user what to select, and wait for the displayed angles or copied code.
- Reproduce every accepted GUI operation with explicit notebook code. Do not deliver a
  notebook that depends on an open GUI or clipboard state.

## Source priority

1. Inspect the installed ERLabPy version and its public API.
2. Use the local documentation that matches the environment when available.
3. Read [references/docs-links.md](references/docs-links.md) only when exact public
   documentation or a stable link is needed.
4. Use the stable `llms.txt` and `llms-full-no-changelog.txt` exports only when the
   focused pages do not contain enough detail.
5. Use changelog content only for version-specific or unreleased behavior.

State what was verified and what remains uncertain. Ask for a reference spectrum,
geometry value, fit model choice, or physical interpretation only when the notebook
cannot determine it safely.
