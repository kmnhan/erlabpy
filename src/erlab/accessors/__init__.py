"""
Some `xarray accessors
<https://docs.xarray.dev/en/stable/internals/extending-xarray.html>`_ for convenient
data analysis and visualization.

ERLabPy provides a collection of accessors for convenient data analysis and
visualization. The following table lists all accessors provided by ERLabPy:

.. list-table::
   :header-rows: 1

   * - Accessor
     - Description
   * - :class:`da.qshow <erlab.accessors.general.InteractiveDataArrayAccessor>`,
       :class:`ds.qshow <erlab.accessors.general.InteractiveDatasetAccessor>`
     - Interactive data visualization
   * - :class:`da.qplot <erlab.accessors.general.PlotAccessor>`
     - Plotting data
   * - :class:`da.qsel <erlab.accessors.general.SelectionAccessor>`
     - Convenient data selection
   * - :class:`da.qinfo <erlab.accessors.general.InfoDataArrayAccessor>`
     - Quickly check data details
   * - :class:`da.modelfit <erlab.accessors.fit.ModelFitDataArrayAccessor>`,
       :class:`ds.modelfit <erlab.accessors.fit.ModelFitDatasetAccessor>`,
       :class:`da.parallel_fit <erlab.accessors.fit.ParallelFitDataArrayAccessor>`
     - Curve fitting
   * - :class:`da.kspace <erlab.accessors.kspace.MomentumAccessor>`
     - Momentum conversion

.. currentmodule:: erlab.accessors

Modules
=======

.. autosummary::
   :toctree: generated

   utils
   general
   kspace
   fit

"""  # noqa: D205
