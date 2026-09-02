(guide-goldtool)=

# goldtool

Interactive tool for obtaining the shape of the Fermi edge from data such as a gold
reference spectrum.

`goldtool` can be started with {func}`erlab.interactive.goldtool`:

```python
import erlab.interactive as eri

eri.goldtool(data)
```

It can also be opened from the right-click context menu of any image plot in ImageTool.

Use the `%goldtool` magic (see {ref}`interactive-misc-magics`) to launch it directly from IPython.

When `goldtool` is opened from an ImageTool in the manager, it remembers the selected
spectrum or slice that opened it. If that ImageTool changes, the manager can mark the
tool and its corrected ImageTool window as {guilabel}`Stale`. Enable
{guilabel}`Refit after update` when you want the edge fit to rerun automatically
after compatible updates.

(guide-goldtool-edge-controls)=

## Fermi edge fit controls

The ROI defines the detector-coordinate range and the outer energy range used for the
fit.

- {guilabel}`T (K)` sets the sample temperature. {guilabel}`Fix T` controls whether the
  parameter is fixed in the fitting process.
- {guilabel}`Resolution` sets the initial energy-resolution FWHM in electronvolts.
- {guilabel}`Step edge` uses a Gaussian-broadened step instead of the Fermi-Dirac
  model. It disables {guilabel}`Fix T`.
- {guilabel}`Adaptive` estimates a separate energy range for each EDC inside the outer
  ROI range. An EDC uses the complete outer range if no valid falling edge is found.
- {guilabel}`Bin x` and {guilabel}`Bin y` average adjacent detector and energy points
  before the fit.
- {guilabel}`Linear` includes a linear background above the Fermi level.
- {guilabel}`Method` selects the lmfit minimizer. {guilabel}`Scale cov` controls
  covariance scaling.
- {guilabel}`# CPU` sets the number of parallel workers.
- {guilabel}`Go` starts the fits.

The {guilabel}`Polynomial` and {guilabel}`Spline` tabs control the model fitted to the
edge centers:

- {guilabel}`Residuals` shows the difference between the fitted model and edge centers.
- {guilabel}`Corrected` shows the corrected reference data.
- {guilabel}`Shift coords` lets the corrected output change its energy coordinate and
  length so that it retains the complete shifted range.
- {guilabel}`Open corrected in ImageTool` opens the corrected reference data.
- {guilabel}`Open edge in ImageTool` evaluates the selected model on the input detector
  coordinates and opens the resulting edge positions. In Manager, this output can be
  used as the shift source for another compatible ImageTool.

See {ref}`how-to-gui-fermi-edge-correction` for more information on using the Fermi edge
fit controls and the corrected output.
