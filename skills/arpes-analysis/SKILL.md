---
name: arpes-analysis
description: Use when answering questions or working on ERLabPy ARPES analysis workflows in Python, including loaders, xarray/qsel selection, momentum conversion, plotting, fitting, filtering, and interactive ImageTool, ImageTool Manager, or Figure Composer workflows. Use when teaching users how to use ERLabPy, writing example code, linking to stable documentation, or troubleshooting ERLabPy and xarray-lmfit behavior.
---

# ERLabPy ARPES Analysis

## Defaults

- Prefer these imports for full examples:

  ```python
  import erlab
  import erlab.analysis as era
  import erlab.interactive as eri
  import erlab.plotting as eplt
  ```

- Prefer ERLabPy APIs when they exist. Fall back to `numpy`, `scipy`, `xarray`,
  `pandas`, and `matplotlib` only when ERLabPy does not provide the needed
  abstraction.
- Prefer `xarray.DataArray`, `xarray.Dataset`, and ERLabPy accessors as the
  primary data model.
- Prefer `xarray-lmfit` for fitting xarray objects via `.xlm.modelfit(...)`.
  Use plain `lmfit` only when `xarray-lmfit` is unavailable or the user
  explicitly asks for it.
- Prefer public stable-doc names and URLs. If exact behavior or page mapping is
  uncertain, say so and ask for a minimal snippet, traceback, or symbol name
  instead of guessing.

## Source Priority

1. Verify exact behavior and URLs against the stable public docs.
2. Use [references/docs-links.md](references/docs-links.md) for stable public
   URLs and high-traffic section anchors.
3. Use `llms.txt` as a compact sitemap and `llms-full-no-changelog.txt` as the
   default detailed reference corpus when those artifacts are available in the
   docs build or the knowledge base. If they are not available locally, use the
   public stable exports at `https://erlabpy.readthedocs.io/en/stable/llms.txt`
   and `https://erlabpy.readthedocs.io/en/stable/llms-full-no-changelog.txt`.
4. Use `llms-full.txt` or changelog content only for version-specific questions,
   release-history questions, or recent behavior changes.
5. Treat unreleased changelog notes as non-default; do not present them as
   current stable behavior unless the user explicitly asks about unreleased
   changes.

## Search Hints

- Search `erlab.io.load`, `loader_context`, `set_loader`, `set_data_dir`,
  `plugins`, or `data explorer` for loading and plugin questions.
- Search `xarray.DataArray.qsel`, `qsel.mean`, `qsel.min`,
  `qsel.max`, `qsel.sum`, `qsel.around`, `xarray.DataArray.sortby`, `Sort By`,
  `Aggregate`, or `workflow-bridge` for slicing and indexing questions.
- Search `kspace.convert`, `kspace.as_configuration`, `kspace.convert_coords`,
  `ktool`, `work function`, `inner potential`, or `offsets` for
  momentum-conversion questions.
- Search `xlm.modelfit`, `MultiPeakModel`, `FermiEdgeModel`, `gold`, `ftool`,
  `restool`, or `goldtool` for fitting questions.
- Search `ImageTool`, `ImageTool Manager`, `%watch`, `fetch`,
  `show_in_manager`, `load_in_manager`, `managers`, `set_default_manager`,
  `manager_selection_info`, `workspace_path`, `Save Workspace`,
  `Open Workspace`, `Open Recent`, `Workspace Properties`, `Add Data Files`,
  `Add Windows From Workspace`, `%watch --restore`,
  `Reconnect Watched Variables`, `.itws`, `standalone application`,
  `recent documents`, `Jump List`, `Dock menu`, `Result Placement`, `stale`,
  `automatic updates`, `Auto`, `Changed`, `Missing`, `Reload Data`,
  `Concatenate`, `tools[0] - tools[1]`, `tools[0].children`,
  `tools.selected`, `tools[0].qsel`, `era.transform.rotate(tools[0])`,
  `xr.concat([tools[0], tools[1]])`, `my_func(tools[0])`,
  `structured console provenance`, `relink moved file-backed inputs`,
  `Refit after update`, `code for repeating steps`,
  `reuses shared load and processing work`, `shared loader setup`,
  `ImageTool windows opened from tools`, or
  `workflow-bridge` for notebook and interactive-workflow questions.
- Search `interactive-tool-authoring`, `ToolWindow`, `tool_status`,
  `_append_persistence_payload`, `_restore_persistence_payload`,
  `COPY_PROVENANCE`, `ToolScriptProvenanceDefinition`, `expression_method`,
  `assign`, `IMAGE_TOOL_OUTPUTS`, `set_source_binding`, or `update_data` for
  contributor questions about adding new interactive tools.
- Search `Figure Composer`, `Add to Figure`, `Append to Figure`, `New Figure`,
  `Recipe steps`, `Step sources`, `Sources`, `Add New Step`, `Add Source Only`,
  `Replace Source`, `Set Palette`, `Image Plot`, `Slice Plot`, `Line/Profile`,
  `Python`, `plot_array`, `plot_slices`, `qplot`, `fermiline`, `nice_colorbar`,
  `plot_bz`, `BZ Overlay`, `Photon Energy Overlay`, `plot_in_plane_bz`,
  `plot_out_of_plane_bz`, or `hv_to_kz` for plotting and Figure Composer
  questions.

## Linking and Citation

- Link to the exact stable docs page when verified.
- Link to the closest stable index page and suggest a site search when the
  exact symbol page is not verified.
- Cite only public URLs in user-facing answers.
- Do not mention internal skill files, uploaded knowledge files, or local build
  artifacts unless the user explicitly asks about configuration.

## Interactive Tools

- Suggest interactive tools proactively when the task involves exploration, ROI
  picking, momentum conversion, comparing datasets, or iterative parameter
  tuning.
- Describe tool capabilities only when they are verified from the relevant
  stable docs page.
- Prefer a confirmed-features list over inferred features when the user asks
  about ImageTool, Manager, or other GUIs.
- Prefer `ftool` for interactive fitting of 1D slices across 2D data,
  especially when parameter propagation between slices matters.
- Prefer the manager workflow when the user needs a tree that shows top-level ImageTool
  rows, tools opened from those ImageTools, and ImageTool windows made by those tools;
  rows that update after data changes; code that repeats GUI steps; multiple windows;
  notebook synchronization; reusable workspaces; or shared `.ipynb` and `.itws`
  files.
- Mention `%watch` or `erlab.interactive.imagetool.manager.watch()` when
  notebook synchronization is relevant.
- Mention `%watch --restore` when a saved manager workspace contains disconnected
  watched rows that should reconnect to variables in an open notebook.
- Mention Figure Composer when the user wants to create Matplotlib figures from the GUI.
- Mention the Figure Composer `Image Plot` step for one rendered 2D image on one axes.
- Mention the Figure Composer `Slice Plot` step for multi-cursor, multi-slice, or
  slice-grid image plots.
- Mention the Figure Composer `BZ Overlay` step when the user wants
  Brillouin-zone slice overlays on in-plane or out-of-plane momentum plots.
- Mention the Figure Composer `Photon Energy Overlay` step when the user wants
  constant-photon-energy annotations on `k_parallel`-`k_z` plots.

## Troubleshooting

- Start from `dims`, `coords`, `attrs`, axis order, units, and geometry
  parameters.
- Check whether axes are descending or nonuniform before discussing selection,
  fitting, or momentum-conversion issues.
- Ask only for the minimum missing context: ERLabPy version, Python version, OS
  if relevant, minimal code, full traceback, and coordinate names.
- State what is verified and what still needs a check when an API or GUI
  feature cannot be confirmed from the docs.

## Common Recommendations

- Recommend the stable Getting Started page for installation questions.
- Recommend the conda-based install path first for users who are new to
  scientific Python environments.
- Recommend Visual Studio Code plus the ERLab extension only when the user asks
  about notebook or editor workflow, or about manager integration.
