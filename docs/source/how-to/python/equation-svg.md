(how-to-python-copy-mathtext)=

# Exporting equations as SVG

Convert Matplotlib MathText to SVG before you paste it into a vector graphics editor:

```python
import erlab.plotting as eplt

svg = eplt.copy_mathtext(r"$e^{i\pi} + 1 = 0$")
```

The function returns the SVG text and tries to copy it to the system clipboard. Paste
the result into the target application. If clipboard access is not available, save the
returned text:

```python
from pathlib import Path

Path("equation.svg").write_text(svg, encoding="utf-8")
```

Set `outline=True` when the target system does not have the required fonts. Outlined
text preserves its appearance, but it is no longer editable as text. See
{func}`erlab.plotting.copy_mathtext` for font and MathText settings.
