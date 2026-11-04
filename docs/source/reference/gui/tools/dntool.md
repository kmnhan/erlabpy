(guide-dntool)=

# dntool

`dntool` denoises a two-dimensional angle-energy image with
{func}`erlab.analysis.image.ml.denoise`, which uses the neural network described in
{cite:t}`kim2021denoise`.

```python
import erlab.interactive as eri

eri.dntool(data)
```

You can also open `dntool` from the right-click menu of an ImageTool image that includes
an `eV` dimension.

Select **Run denoise** to start processing. The model and processing options are
described in {func}`erlab.analysis.image.ml.denoise`.

Saved tools include the last completed result, so reopening does not run the model
again. A custom model file is required only when you run denoise or refresh the input.
