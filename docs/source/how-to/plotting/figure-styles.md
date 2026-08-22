(how-to-plotting-figure-styles)=
(how-to-python-apply-plot-style)=

# Figure styles

## Python

Use a Matplotlib style context to change one figure without changing later figures:

```python
import matplotlib.pyplot as plt
import erlab.plotting as eplt

with plt.style.context(["erlab.general", "erlab.nature", "erlab.arial"]):
    eplt.plot_array(data, cmap="Greys", gamma=0.5)
```

See the {ref}`bundled style-sheet table <reference-plotting-styles>` to select a style
or font combination.

Confirm that every requested font is installed. Save and inspect the figure in its
final output format because font substitution and line widths can differ from the
notebook display.

See the [Matplotlib style-sheet guide](https://matplotlib.org/stable/users/explain/customizing.html)
for style composition and {data}`rcParams <matplotlib.rcParams>`.

## Figure Composer

Figure Composer styles are shared settings. They apply to Figure Composer figures that
use those settings. A Python style context applies only to the figure created inside
the context.

To apply the same bundled styles:

1. Open the shared Settings window.
2. Select {guilabel}`Figure Composer`.
3. In {guilabel}`Stylesheets`, select `erlab.general`, then choose {guilabel}`Add`.
4. Add `erlab.nature` and `erlab.arial` in the same way.
5. Use {guilabel}`Up` and {guilabel}`Down` to keep the styles in this order:
   `erlab.general`, `erlab.nature`, `erlab.arial`.
6. Open or refresh a figure. Check its fonts, line widths, dimensions, and export
   settings.

For custom stylesheets, see {ref}`how-to-gui-install-figure-style`. See
{ref}`figure-composer-styles` for style and export precedence.
