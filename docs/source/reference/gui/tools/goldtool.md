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

## Fermi-edge fit controls

The ROI defines the detector-coordinate range and the outer energy range used for the
fit.

- {guilabel}`T (K)` sets the sample temperature. {guilabel}`Fix T` controls whether the
  full Fermi-edge model can vary it.
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

Use {ref}`how-to-gui-fit-fermi-edge-separate-ranges` when one fixed energy range is not
reliable for all EDCs.
