(how-to-gui-fermi-edge-correction)=

# Fermi edge correction

Use these procedures to derive a straight slit correction from a measured metal
reference and apply it in ImageTool Manager. Use the correction only for measurements
for which the same detector correction is valid. This normally requires the same
analyzer configuration, lens mode, pass energy, and detector coordinate definition.

Fermi edge correction in the GUI is performed with {ref}`guide-goldtool`.

For the corresponding Python workflow, see {doc}`../python/fermi-edge-correction`.

## Fitting a Fermi edge from a reference measurement

1. Open the reference spectrum in Manager. Double-click its row to show its ImageTool
   window.
2. Right-click the main image and select {guilabel}`goldtool`.
3. Resize the ROI to choose a suitable angle and energy window around the Fermi edge.
4. Enter the measured temperature in {guilabel}`T (K)` and the energy-resolution FWHM in
   {guilabel}`Resolution`. Different toggles control the input parameters to
   {func}`erlab.analysis.gold.edge`.
5. Select {guilabel}`Go` to start the fitting procedure. Badly conditioned fits are
   automatically excluded from the results.
6. Select {guilabel}`Polynomial` or {guilabel}`Spline` and inspect the fitted edge.
7. You can view the fitted edge using {guilabel}`Open edge in ImageTool`. To use this
   edge to correct another measurement, see
   {ref}`how-to-gui-fit-fermi-edge-reuse`.

(how-to-gui-fit-fermi-edge-reuse)=

## Correction of another measurement

1. In goldtool, select the {guilabel}`Polynomial` or {guilabel}`Spline` tab that contains
   the accepted edge model.
2. Select {guilabel}`Open edge in ImageTool`. Manager adds the evaluated edge below
   goldtool.
3. In the ImageTool of the measurement you wish to correct, select {menuselection}`Edit --> Shift…`.
4. Set {guilabel}`Dimension` to `eV`.
5. Set {guilabel}`Shift Source` to {guilabel}`Existing ImageTool` and select the edge
   ImageTool created by goldtool.
6. Keep {guilabel}`Negate shift` selected. Select {guilabel}`Shift coordinates` to
   retain the complete shifted energy range.
7. Set {guilabel}`Result Placement` to {guilabel}`Open Child Window` and select
   {guilabel}`OK`.

The corrected child records both the measured data and the fitted edge as inputs. If
either input changes, Manager marks the result as stale.
