# Other interactive tools

In addition to ImageTool and the ImageTool manager, other interactive tools for specific tasks are available in the {mod}`erlab.interactive` module.

Most of these tools can be opened as auxiliary windows from within ImageTool, and are also integrated into the ImageTool manager, allowing you to manage them alongside your ImageTool windows.

Here are some of the miscellaneous interactive tools provided:

(guide-ktool)=

## ktool

Interactive conversion from angles to momentum space.

There are three ways to invoke the GUI.

1. {meth}`xarray.DataArray.kspace.interactive`

   ```python
   data.kspace.interactive()
   ```

2. {func}`erlab.interactive.ktool`

   This option is recommended because the name of the input data will be automatically detected and applied to the generated code that is copied to the clipboard.

   ```python
   import erlab.interactive as eri

   eri.ktool(data)
   ```

3. From within ImageTool

   Click "Open in ktool" from the "View" menu in the menu bar.

   The button will be disabled if the data is not compatible with {func}`ktool <erlab.interactive.ktool>`.

The GUI is divided into two tabs.

```{image} ../../images/ktool_1_light.png
:align: center
:alt: KspaceTool 1
:class: only-light
```

:::{only} format_html

```{image} ../../images/ktool_1_dark.png
:align: center
:alt: KspaceTool 1
:class: only-dark
```

:::

The first tab is for setting momentum conversion parameters. The image is updated in real time as you change the parameters.

Clicking the "Copy code" button will copy the code for conversion to the clipboard.

The "Open in ImageTool" button performs a full three-dimensional conversion and opens the result in the ImageTool.

```{image} ../../images/ktool_2_light.png
:align: center
:alt: KspaceTool 2
:class: only-light
```

:::{only} format_html

```{image} ../../images/ktool_2_dark.png
:align: center
:alt: KspaceTool 2
:class: only-dark
```

:::

The second tab provides visualization options. You can overlay Brillouin zones and high symmetry points on the result, adjust colors, apply binning, and more.

The "Add Circle ROI" button allows you to add a circular region of interest to the image, which can be edited by dragging or right-clicking on it.

You can pass some parameters to customize the GUI. For example, you can set the Brillouin zone size/orientation and the colormap like this:

```python
data.kspace.interactive(
    avec=np.array([[-3.485, 6.03], [6.97, 0.0]]), rotate_bz=30.0, cmap="viridis"
)
```

For all available parameters, see the documentation for {func}`erlab.interactive.ktool`.

(guide-dtool)=

## dtool

Interactive tool for visualizing dispersive data using derivative-based methods.

The GUI can be invoked with {func}`erlab.interactive.dtool`:

```python
import erlab.interactive as eri

eri.dtool(data)
```

It can also be opened from within ImageTool from the right-click context menu of any image plot.

```{image} ../../images/dtool_light.png
:align: center
:alt: DerivativeTool window in light mode
:class: only-light
```

:::{only} format_html

```{image} ../../images/dtool_dark.png
:align: center
:alt: DerivativeTool window in dark mode
:class: only-dark
```

:::

- The first section interpolates the data to a grid prior to smoothing.

- The second section applies smoothing prior to differentiation.

- In the third section, selecting different tabs will apply different methods.
  Each tab contains parameters relevant to the selected method.

- Clicking the copy button will copy the code for differentiation to the clipboard.

- Both the smoothed data and the result can be opened in ImageTool from the right-click menu of each plot, where it can be analyzed further or saved to disk.

(guide-goldtool)=

## goldtool

Interactive tool for obtaining the shape of the Fermi edge (e.g., from a gold reference spectrum).

The GUI can be invoked with {func}`erlab.interactive.goldtool`:

```python
import erlab.interactive as eri

eri.goldtool(data)
```

It can also be opened from within ImageTool from the right-click context menu of any image plot.

(guide-restool)=

## restool

Interactive tool for fitting a single resolution-broadened Fermi-Dirac distribution to an energy distribution curve (EDC). The momentum range to be integrated over can be adjusted interactively. This is useful for quickly determining the energy resolution of the current experiment.

The GUI can be invoked with {func}`erlab.interactive.restool`:

```python
import erlab.interactive as eri

eri.restool(data)
```

It can also be opened from within ImageTool from the right-click context menu of any image plot that contains an energy axis.

## Data explorer

```{image} ../../images/explorer_light.png
:align: center
:alt: Data explorer window in light mode
:class: only-light
```

:::{only} format_html

```{image} ../../images/explorer_dark.png
:align: center
:alt: Data explorer window in dark mode
:class: only-dark
```

:::

Provides a file-browser-like interface for exploring and visualizing ARPES data stored on your disk. See {func}`erlab.interactive.data_explorer` for more information.
