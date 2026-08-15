# Curve fitting

Use these guides to fit curves and inspect saved fits in the GUI. Read
{doc}`the curve-fitting explanation <../../explanation/fitting>` before you select a
model or interpret fit diagnostics.

(how-to-gui-fit-data)=

## Fitting one curve

1. Open the curve in ftool. See {ref}`guide-ftool` for the available entry points.
2. Select the model and model options in {guilabel}`Setup`.
3. Set the fit window with {guilabel}`X range` or the dashed bounds in the plot.
4. Open {guilabel}`Fit` and choose {guilabel}`Guess` when the model provides suitable
   initial estimates.
5. Inspect and adjust parameter values, bounds, and expressions.
6. Choose {guilabel}`Fit`.
7. Inspect the data, best fit, components, residual behavior, parameter uncertainties,
   and fit statistics.

If the fit reaches {guilabel}`Max nfev`, increase the limit only after checking the
model and initial parameters. Use {guilabel}`Fit ×20` when repeated optimization from
the previous result is appropriate for the model. Do not treat convergence alone as a
physically valid fit.

Use {guilabel}`Copy code` to reproduce the fit in Python. Use {guilabel}`Save fit` to
store the result with {func}`xarray_lmfit.save_fit`.

Use {ref}`how-to-gui-fit-stack-with-ftool` for a stack of curves and
{ref}`how-to-gui-reopen-saved-fit` to continue work from a saved result.

(how-to-gui-fit-fermi-edge-separate-ranges)=

## Fermi edge fitting with separate EDC ranges

Use this procedure when the edge position changes substantially across a measured
reference or when other spectral features make one fixed range unreliable.

1. Open the two-dimensional reference data in goldtool.
2. Draw the ROI around the detector-coordinate range to fit. Set its energy bounds to
   one outer range that contains the edge and usable background for every EDC.
3. Enter the measured temperature in {guilabel}`T (K)` and the energy-resolution
   estimate in {guilabel}`Resolution`.
4. Select {guilabel}`Adaptive`.
5. Select {guilabel}`Step edge` when the temperature is missing or unreliable.
6. Choose {guilabel}`Go`.
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

(how-to-gui-fit-stack-with-ftool)=

## Fitting a stack of curves

1. Open the two-dimensional data in ftool.
2. Confirm that the fitted coordinate is horizontal. Choose {guilabel}`Transpose` if
   the image has the wrong orientation.
3. Set the curve fit window with {guilabel}`X range`.
4. Set the sequence range with {guilabel}`Y range`.
5. Select a representative curve with {guilabel}`Index` or the yellow cursor.
6. Fit that curve and confirm that its model and parameters are suitable.
7. Select a {guilabel}`Fill mode`:

   - Use {guilabel}`Previous` to initialize from the last good fit.
   - Use {guilabel}`Extrapolate` to project parameters from the previous two fits.
   - Use {guilabel}`None` when each curve already has suitable initial values.

8. Choose {guilabel}`Fit ⤒` or {guilabel}`Fit ⤓` for the required sequence direction.
9. Inspect the parameter plot and fitted curve at each suspicious or failed index.
10. Correct failed fits before choosing {guilabel}`Save fit` or {guilabel}`Copy code`.

When ftool is managed, parameter maps opened in ImageTool appear below the ftool row.
Use {guilabel}`Refit after update` only when compatible input changes should repeat the
same fit sequence.

(how-to-gui-reopen-saved-fit)=

## Reopening a saved fit

Load the saved fit dataset and pass it directly to ftool:

```python
import erlab.interactive as eri
from xarray_lmfit import load_fit

fit_result = load_fit("fit-result.h5")
eri.ftool(fit_result)
```

Confirm that the saved model and fitted parameter values are present. For a
two-dimensional fit, every saved curve must use the same model definition.

The saved data is limited to the fit range stored in the result. Use a Manager workspace
instead when the full source data, ftool state, fit result, and child ImageTools must be
restored together.
