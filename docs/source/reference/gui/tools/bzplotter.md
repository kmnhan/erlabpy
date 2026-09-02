# BZPlotter

This tool is not an analysis tool, but rather a utility for creating plots of three-dimensional Brillouin zones and exporting them as vector graphics.

The GUI can be invoked with {class}`erlab.interactive.bzplot.BZPlotter`:

```python
import erlab.interactive as eri

eri.bzplot.BZPlotter()
```

Once opened, a matplotlib figure window will appear alongside a control panel for adjusting the lattice parameters. The figure can be rotated interactively using the mouse, and the plot can be exported as any of the formats supported by matplotlib via the standard matplotlib save button.
