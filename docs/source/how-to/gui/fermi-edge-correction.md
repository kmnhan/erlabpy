(how-to-gui-fermi-edge-correction)=

# Fermi edge correction

Use these procedures to derive an energy correction from a measured metal reference and
apply it in ImageTool Manager. Use the correction only for measurements for which the
same detector correction is valid. This normally requires the same analyzer
configuration, lens mode, pass energy, and detector-coordinate definition.

Manager checks whether the arrays are structurally compatible. It cannot determine
whether their acquisition conditions match.

For the corresponding Python workflow, see
{doc}`../python/fermi-edge-correction`.

## Reference spectrum correction

1. Open the reference spectrum in Manager. Double-click its row to show its ImageTool
   window.
2. Right-click the main image and select {guilabel}`goldtool`.
3. Draw the ROI around the detector-coordinate range to fit. Set its energy bounds to
   include the edge and usable background.
4. Enter the measured temperature in {guilabel}`T (K)` and the energy-resolution FWHM
   in {guilabel}`Resolution`. Select {guilabel}`Step edge` when the temperature is
   missing or unreliable.
5. Select {guilabel}`Go`.
6. Select {guilabel}`Polynomial` or {guilabel}`Spline`. Inspect the fitted edge centers,
   error bars, model, and residuals. Adjust the model only when the fit diagnostics
   require it.
7. Select {guilabel}`Corrected` to inspect the corrected reference.
8. Keep {guilabel}`Shift coords` selected and select
   {guilabel}`Open corrected in ImageTool`.

The corrected ImageTool is a child of goldtool in the Manager tree. Its non-energy
dimensions keep their original order. Its `eV` coordinate and length can change so that
the shifted spectra retain the original energy range.

## Correction of another measurement

1. In goldtool, select the {guilabel}`Polynomial` or {guilabel}`Spline` tab that contains
   the accepted edge model.
2. Select {guilabel}`Open edge in ImageTool`. Manager adds the evaluated edge below
   goldtool.
3. Double-click the measurement to correct in the Manager tree.
4. Select {menuselection}`Edit --> Shift…` in ImageTool.
5. Set {guilabel}`Dimension` to `eV`.
6. Set {guilabel}`Shift Source` to {guilabel}`Existing ImageTool`. Select the edge
   ImageTool created by goldtool.
7. Keep {guilabel}`Negate shift` selected. Select {guilabel}`Shift coordinates` to
   retain the complete shifted energy range.
8. Set {guilabel}`Result Placement` to {guilabel}`Open Child Window` and select
   {guilabel}`OK`.

The corrected child records both the measured data and the fitted edge as inputs. If
either input changes, Manager marks the result as stale.

If {guilabel}`Existing ImageTool` is unavailable, confirm that the measured data and
edge are in the same Manager, that the edge is current, and that their shared detector
coordinates have the same values and lengths.

(how-to-gui-fit-fermi-edge-separate-ranges)=

## Separate EDC fit ranges

Use this procedure when the edge position changes substantially across a measured
reference or when other spectral features make one fixed range unreliable.

1. Open the two-dimensional reference data in goldtool.
2. Draw the ROI around the detector-coordinate range to fit. Set its energy bounds to
   one outer range that contains the edge and usable background for every EDC.
3. Enter the measured temperature in {guilabel}`T (K)` and the energy-resolution
   estimate in {guilabel}`Resolution`.
4. Select {guilabel}`Adaptive`.
5. Select {guilabel}`Step edge` when the temperature is missing or unreliable.
6. Select {guilabel}`Go`.
7. Inspect the fitted edge centers, error bars, and the polynomial or spline before you
   use the correction.

Adaptive range selection detects a falling edge. If no valid edge is found in one EDC,
that EDC uses the complete outer energy range. An outlying center, a large uncertainty,
or unsupported structure in the polynomial or spline indicates that the fit needs
inspection. Change the outer ROI only when the new bounds still contain the edge and
suitable background for every EDC.

Use {ref}`how-to-python-fit-fermi-edge-separate-ranges` to inspect the estimated range
for one difficult EDC. See {ref}`guide-goldtool-edge-controls` for the control
definitions.
