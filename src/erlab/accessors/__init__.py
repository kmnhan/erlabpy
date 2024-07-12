"""
Some `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ for convenient
data analysis and visualization.

.. currentmodule:: erlab.accessors

Modules
=======

.. autosummary::
   :toctree: generated

   utils
   general
   kspace
   fit

ERLabPy provides a collection of `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ for convenient
data analysis and visualization. The following table lists the available accessors.

.. list-table::
   :header-rows: 1

   * - Accessor
     - Description
   * - :class:`da.qshow <erlab.accessors.general.InteractiveDataArrayAccessor>`,
       :class:`ds.qshow <erlab.accessors.general.InteractiveDatasetAccessor>`
     - Interactive data visualization
   * - :class:`da.qplot <erlab.accessors.general.PlotAccessor>`
     - Plotting data
   * - :class:`da.modelfit <erlab.accessors.fit.ModelFitDataArrayAccessor>`,
       :class:`ds.modelfit <erlab.accessors.fit.ModelFitDatasetAccessor>`,
       :class:`da.parallel_fit <erlab.accessors.fit.ParallelFitDataArrayAccessor>`
     - Curve fitting
   * - :class:`da.kspace <erlab.accessors.kspace.MomentumAccessor>`
     - Momentum conversion

"""  # noqa: D205
