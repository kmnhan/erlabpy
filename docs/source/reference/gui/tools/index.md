(interactive-misc-tools)=

# Interactive tools

These applications provide specialized analysis, plotting, and data-browsing
interfaces. Most can be opened from ImageTool or from Python.

| Tool | Purpose |
| --- | --- |
| {doc}`ktool` | Configure, preview, and apply momentum conversion to ARPES data |
| {doc}`dtool` | Visualize dispersive data with derivative-based methods |
| {doc}`goldtool` | Fit Fermi edges in reference spectra and create energy and angle corrections |
| {doc}`ftool` | Fit one- and two-dimensional data interactively with lmfit models |
| {doc}`restool` | Fit a resolution-broadened Fermi–Dirac EDC to estimate energy resolution |
| {doc}`meshtool` | Detect and remove grid-like mesh artifacts from fixed-mode ARPES data |
| {doc}`data-explorer` | Browse files, preview data and metadata, and open supported datasets |
| {doc}`periodic-table` | Inspect element data, absorption edges, masses, and photoionization cross sections |
| {doc}`bzplotter` | Create and export three-dimensional Brillouin-zone plots |
| {doc}`notebook-shortcuts` | Launch interactive tools with IPython and Jupyter line magics |

When a tool is opened from a managed ImageTool, ImageTool Manager can record the source
relationship and place its results below the tool. See
{ref}`imagetool-manager-nested-results` and {ref}`imagetool-manager-refresh`.

```{toctree}
:hidden: true
:maxdepth: 1

ktool
dtool
goldtool
ftool
restool
meshtool
data-explorer
periodic-table
bzplotter
notebook-shortcuts
```
