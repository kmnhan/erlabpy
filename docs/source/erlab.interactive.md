# Interactive ({mod}`erlab.interactive`)

```{eval-rst}
.. automodule:: erlab.interactive
```

## Interactive-tool authoring API

The following protected members are extension points for interactive tools. Use them
only when you implement a subclass of {class}`erlab.interactive.utils.ToolWindow`.

```{eval-rst}
.. autoattribute:: erlab.interactive.utils.ToolWindow.sigInfoChanged
.. autoattribute:: erlab.interactive.utils.ToolWindow.sigDataChanged
.. automethod:: erlab.interactive.utils.ToolWindow._append_persistence_payload
.. automethod:: erlab.interactive.utils.ToolWindow._restore_persistence_payload
.. automethod:: erlab.interactive.utils.ToolWindow._cancel_background_work
.. automethod:: erlab.interactive.utils.ToolWindow._run_or_defer_restore_work
.. automethod:: erlab.interactive.utils.ToolWindow._flush_restore_work
.. automethod:: erlab.interactive.utils.ToolWindow._discard_restore_work
.. automethod:: erlab.interactive.utils.ToolWindow._write_state
.. automethod:: erlab.interactive.utils.ToolWindow._reset_history_stack
.. automethod:: erlab.interactive.utils.ToolWindow._replace_last_state
.. automethod:: erlab.interactive.utils.ToolWindow._notify_data_changed
.. automethod:: erlab.interactive.utils.ToolWindow.validate_update_inputs
   :no-index:
.. automethod:: erlab.interactive.utils.ToolWindow.update_inputs
   :no-index:
.. automethod:: erlab.interactive.utils.ToolWindow._launch_output_imagetool
.. automethod:: erlab.interactive.utils.ToolWindow._launch_detached_output_imagetool
```

These implementation classes provide examples for tool authors. Their public launcher
functions remain the supported user interface.

```{eval-rst}
.. autoclass:: erlab.interactive.derivative.DerivativeTool
   :no-members:
.. autoattribute:: erlab.interactive.derivative.DerivativeTool.result
.. autoclass:: erlab.interactive.kspace.KspaceToolGUI
   :no-members:
.. autoclass:: erlab.interactive.kspace.KspaceTool
   :no-members:
.. autoclass:: erlab.interactive.fermiedge.GoldTool
   :no-members:
.. automethod:: erlab.interactive.fermiedge.GoldTool.validate_update_inputs
.. automethod:: erlab.interactive.fermiedge.GoldTool.update_inputs
.. autoclass:: erlab.interactive.fermiedge.ResolutionTool
   :no-members:
.. autoclass:: erlab.interactive._mesh.MeshTool
   :no-members:
.. autoclass:: erlab.interactive._fit1d.Fit1DTool
   :no-members:
.. autoclass:: erlab.interactive._fit2d.Fit2DTool
   :no-members:
.. autoclass:: erlab.interactive._fit2d.Fit2DTool.Output
   :members:
.. automethod:: erlab.interactive._fit2d.Fit2DTool._show_dataarray_in_itool
.. autoclass:: erlab.interactive.imagetool.plot_items.ItoolPlotItem
   :no-members:
.. automethod:: erlab.interactive.imagetool.plot_items.ItoolPlotItem.make_tool_source_spec
```
