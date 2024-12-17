Using ImageTool
===============

.. image:: ../images/imagetool_light.png
    :align: center
    :alt: Imagetool
    :class: only-light

.. only:: format_html

    .. image:: ../images/imagetool_dark.png
        :align: center
        :alt: Imagetool
        :class: only-dark

Inspired by *Image Tool* for Igor Pro written by the Advanced Light Source at Lawrence
Berkeley National Laboratory, :class:`ImageTool <erlab.interactive.imagetool.ImageTool>`
is a simple tool exploring images interactively.

Features:

- Zoom and pan

- Real-time slicing & binning

- Multiple cursors

- Easy size adjustment

- Advanced colormap control

- Interactive editing like rotation, normalization, cropping, momentum conversion, and
  more

ImageTool displays *image-like* :class:`xarray.DataArray`\ s from 2 to 4 dimensions. Non-uniform coordinates are converted to index arrays automatically.

Invoke ImageTool by calling :func:`itool <erlab.interactive.imagetool.itool>`: ::

    import erlab.interactive as eri
    eri.itool(data)

Or use the :class:`qshow <erlab.accessors.general.InteractiveDataArrayAccessor>` accessor: ::

    data.qshow()

Tips
----

- Hover over buttons for tooltips.

- Right-click plots for context menus.

  - ``Copy selection code``: copies code to reproduce the data.

- Toggle the ``Lock color limits`` button to lock color range and display a colorbar.

  - When toggled on, the color range is locked to the minimum and maximum of the entire
    data.

  - The color range can be manually set by dragging or right-clicking on the colorbar.

- Rotate and normalize data via the edit and view menus.

- ImageTool is extensible. At our home lab, we use a modified version of ImageTool to
  plot data as it is being collected in real-time!

Keyboard shortcuts
------------------

Some shortcuts not in the menu bar. Mac users replace :kbd:`Ctrl` with :kbd:`⌘` and
:kbd:`Alt` with :kbd:`⌥`.

.. list-table::
    :header-rows: 1

    * - Shortcut
      - Description
    * - :kbd:`LMB` Drag
      - Pan
    * - :kbd:`RMB` Drag
      - Zoom and scale
    * - :kbd:`Ctrl+LMB` Drag
      - Move active cursor
    * - :kbd:`Ctrl+Alt+LMB` Drag
      - Move all cursors
    * - :kbd:`Alt` while dragging a cursor line
      - Move all cursor lines

Rule of thumb: hold :kbd:`Alt` to apply actions to all cursors. Shortcuts for 'shifting'
a cursor involves the :kbd:`Shift` key.

.. _imagetool-manager-guide:

Using the ImageTool manager
---------------------------
Interactive tools in Jupyter notebooks block other code. Use :class:`ImageToolManager
<erlab.interactive.imagetool.ImageToolManager>` as a workaround.

Run ``itool-manager`` in a terminal to start the manager. Calls to :func:`itool
<erlab.interactive.imagetool.itool>` and :meth:`DataArray.qshow
<erlab.accessors.general.InteractiveDataArrayAccessor.__call__>` will open ImageTool in
the manager.

Or run: ::

    python -m erlab.interactive.imagetool.manager

.. note::

  - Only one manager can run per machine.

  - Sending data to the manager has slight overhead, noticeable for large data. Use
    `use_manager=False` to :func:`itool <erlab.interactive.imagetool.itool>` and
    :meth:`DataArray.qshow
    <erlab.accessors.general.InteractiveDataArrayAccessor.__call__>` to open data
    directly.

The manager shows a list of opened ImageTools and buttons to manage them. Hover over
buttons for tooltips.

Open data in the manager by:

- Invoking ImageTool from :func:`itool <erlab.interactive.imagetool.itool>` or
  :meth:`qshow <erlab.accessors.general.InteractiveDataArrayAccessor.__call__>` from any
  script or notebook.

- Opening files through the ``File`` menu in the manager.

- Dragging and dropping ARPES data into the manager window.

Save and load ImageTool windows to a HDF5 file via the ``Save Workspace As...`` and ``Open Workspace...`` menu items.

The manager has a Python console to manipulate ImageTool windows and data, and run Python code. Toggle the console with :kbd:`⌃+`` (Mac) or :kbd:`Ctrl+`` (Windows/Linux) or through the ``View`` menu.

Explore the menubar for more options!
