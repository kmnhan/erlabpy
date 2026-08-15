(guide-dtool)=

# dtool

Interactive tool for visualizing dispersive data using derivative-based methods.

`dtool` can be started with {func}`erlab.interactive.dtool`:

```python
import erlab.interactive as eri

eri.dtool(data)
```

It can also be opened from the right-click context menu of any image plot in ImageTool.

The `%dtool` line magic (see {ref}`interactive-misc-magics`) provides the same entry point from notebooks.

```{image} ../../../images/dtool_light.png
:align: center
:alt: DerivativeTool window in light mode
:class: only-light
```

:::{only} format_html

```{image} ../../../images/dtool_dark.png
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

- Both the smoothed data and the result can be opened in ImageTool from the right-click
  menu of each plot, where it can be analyzed further or saved to disk. When `dtool`
  was opened from an ImageTool in the manager, these ImageTool windows stay under
  `dtool` in the manager tree and can be updated when the ImageTool that opened
  `dtool` changes.
