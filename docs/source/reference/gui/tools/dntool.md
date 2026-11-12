(guide-dntool)=

# dntool

`dntool` denoises a two-dimensional angle-energy image with
{func}`erlab.analysis.image.ml.denoise`, which uses the pretrained model from
{cite:t}`kim2021denoise`.

```python
import erlab.interactive as eri

eri.dntool(data)
```

You can also open `dntool` from the right-click menu of an ImageTool image that includes
an `eV` dimension.

Select **Run denoise** to start processing. The model and processing options are
described in {func}`erlab.analysis.image.ml.denoise`.

**Execution provider** defaults to **Automatic**, which uses CUDA, then CoreML, then CPU
according to availability. You can also select a provider directly. If the selected
accelerator is unavailable, use **CPU** or **Automatic**. CoreML can take longer on the
first run while it prepares the model. The provider choice is saved with the tool and
included in copied code and result history.

Saved tools include the last completed result, so reopening does not run the model
again. A custom model file is required only when you run denoise or refresh the input.
