(guide-restool)=

# restool

Interactive tool for fitting a single resolution-broadened Fermi-Dirac distribution to an energy distribution curve (EDC). The momentum range to be integrated over can be adjusted interactively. This is useful for quickly determining the energy resolution of the current experiment.

The GUI can be invoked with {func}`erlab.interactive.restool`:

```python
import erlab.interactive as eri

eri.restool(data)
```

It can also be opened from the ImageTool image-plot context menu when the data contains an energy axis.

The `%restool` magic (see {ref}`interactive-misc-magics`) provides a quick way to launch it from IPython.

When `restool` is launched from an ImageTool in the manager, it remembers the selected
EDC that opened it. If that ImageTool is replaced with compatible data, the manager can
update the tool and optionally rerun the fit when {guilabel}`Refit after update` is
enabled.
