(guide-ftool)=

# ftool

Interactive curve-fitting tool for 1D and 2D data. By default uses {class}`erlab.analysis.fit.models.MultiPeakModel`, but you can pass any 1D lmfit model.

There are three ways to start `ftool`.

1. {func}`erlab.interactive.ftool`

   ```python
   import erlab.interactive as eri

   eri.ftool(data)
   ```

   To supply a custom model:

   ```python
   eri.ftool(data, model=my_model)
   ```

2. From the ImageTool context menu

   Right-click an image plot or line plot and choose {guilabel}`ftool`.

3. From IPython using the `%ftool` magic described in
   {ref}`interactive-misc-magics`.

   ```ipython
   %ftool data
   %ftool --model my_model data
   ```

When `ftool` is opened from an ImageTool in the manager, it remembers the slice or line
cut that opened it. If that ImageTool changes, the manager can update the tool from the
latest compatible data. Enable {guilabel}`Refit after update` when the same fit
should rerun after updates. For 2D fits, parameter maps opened in ImageTool appear as
child rows under `ftool`.

## Overview

When you first open {guilabel}`ftool`, you will see a stack of controls on the left and
a plot on the right, as shown below. The controls have two tabs: {guilabel}`Setup` and
{guilabel}`Fit`.

:::::{tab-set}

::::{tab-item} Setup

```{image} ../../../images/ftool_1d_setup_light.png
:align: center
:class: only-light
```

:::{only} format_html

```{image} ../../../images/ftool_1d_setup_dark.png
:align: center
:class: only-dark
```

:::
::::

::::{tab-item} Fit

```{image} ../../../images/ftool_1d_fit_light.png
:align: center
:class: only-light
```

:::{only} format_html

```{image} ../../../images/ftool_1d_fit_dark.png
:align: center
:class: only-dark
```

:::
::::

:::::

- The main plot shows the data with the fit overlay, plus dashed vertical lines that
  define the current fit window.

  - Check {guilabel}`Plot components` to show individual model components (if any). This
    also adds a legend for each curve. You can show/hide a component by clicking its
    legend entry.

- The left panel contains controls for setting up and performing the fit. The
  {guilabel}`Setup` tab is for choosing the model and preprocessing options, while the
  {guilabel}`Fit` tab contains parameter settings and options related to the fitting
  process.

## Models and options

First, use the {guilabel}`Model` drop-down to choose a predefined model, a user-provided model, or a model loaded from disk.

- Built-in options are:

  - {class}`erlab.analysis.fit.models.MultiPeakModel`
  - {class}`erlab.analysis.fit.models.FermiEdgeModel`
  - {class}`erlab.analysis.fit.models.StepEdgeModel`
  - {class}`erlab.analysis.fit.models.PolynomialModel`
  - {class}`erlab.analysis.fit.models.TLLModel`
  - {class}`erlab.analysis.fit.models.SymmetrizedGapModel`
  - {class}`lmfit.models.ExpressionModel`

- {guilabel}`From file` loads a lmfit model saved with {func}`lmfit.model.save_model`.

Some models have additional options that appear below the model selector that are used to initialize the model:

- {class}`MultiPeakModel <erlab.analysis.fit.models.MultiPeakModel>`:

  - {guilabel}`# Peaks` and {guilabel}`Peak shape` define how many components are fit and whether they are Lorentzian or Gaussian.
  - {guilabel}`Background` and {guilabel}`Degree` add a constant, linear, or polynomial background.
  - {guilabel}`Fermi-Dirac` multiplies the peaks by a Fermi-Dirac distribution.
  - {guilabel}`Convolve` applies instrumental broadening; {guilabel}`Oversample` controls the internal sampling density used for the convolution.

- {class}`ExpressionModel <lmfit.models.ExpressionModel>`:

  - Edit the independent variable name in the `f(...)` header. Then type your formula,
    such as `a * x + b`, in the expression box.
  - Click {guilabel}`Apply` to rebuild the model from the current expression.
  - Use {guilabel}`Edit init script…` to define helper functions or constants used in the expression.
  - For more information, see the documentation for {class}`lmfit.models.ExpressionModel`.

For complete fitting procedures, see {ref}`how-to-gui-fit-data`.

{guilabel}`Fit ×20` performs 20 consecutive fits on the same curve. Each fit supplies
its fitted parameters as the starting values for the next fit. This can help a
nonlinear or highly correlated model converge after one fit stops near a solution.

## Two-dimensional input

For two-dimensional input, ftool shows an image, one selected curve, and a
parameter-versus-coordinate plot. {guilabel}`Transpose` controls which dimension is the
fitted coordinate. {guilabel}`Index` and the yellow cursor select the curve displayed
in the fit panel.

:::::{tab-set}

::::{tab-item} Setup

```{image} ../../../images/ftool_2d_setup_light.png
:align: center
:class: only-light
```

:::{only} format_html

```{image} ../../../images/ftool_2d_setup_dark.png
:align: center
:class: only-dark
```

:::
::::

::::{tab-item} Fit

```{image} ../../../images/ftool_2d_fit_light.png
:align: center
:class: only-light
```

:::{only} format_html

```{image} ../../../images/ftool_2d_fit_dark.png
:align: center
:class: only-dark
```

:::
::::

:::::

{guilabel}`Fill mode` controls parameter initialization during a fit sequence.
{guilabel}`Previous` uses the last good fit, {guilabel}`Extrapolate` projects from the
previous two fits, and {guilabel}`None` keeps the existing initial values.

For the work procedure, see {ref}`how-to-gui-fit-stack-with-ftool`.
