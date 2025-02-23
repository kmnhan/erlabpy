#####################
ERLabPy documentation
#####################

.. only:: format_html

   **Date**: |today|

   .. image:: https://img.shields.io/pypi/pyversions/erlab?style=flat-square&logo=python&logoColor=white
       :target: https://pypi.org/project/erlab/
       :alt: Supported Python Versions
   .. image:: https://img.shields.io/pypi/v/erlab?style=flat-square&logo=pypi&logoColor=white
       :target: https://pypi.org/project/erlab/
       :alt: PyPi
   .. image:: https://img.shields.io/conda/vn/conda-forge/erlab?style=flat-square&logo=condaforge&logoColor=white
       :target: https://anaconda.org/conda-forge/erlab
       :alt: Conda Version
   .. image:: https://img.shields.io/github/last-commit/kmnhan/erlabpy?style=flat-square&logo=github&color=lightseagreen
       :target: https://github.com/kmnhan/erlabpy.git
       :alt: Last Commit


The ``erlab`` package provides a complete python workflow for ARPES (Angle-Resolved
Photoemission Spectroscopy) experiments. It provides a wide range of tools for
processing, analyzing, and visualizing ARPES data.

*ERLabPy* is built on top of the popular scientific python libraries `numpy
<https://numpy.org>`_, `scipy <https://scipy.org>`_, and `xarray
<https://xarray.pydata.org>`_, and is designed to be easy to use and integrate with
existing scientific Python workflows so that you can quickly get started with your data
analysis.

.. only:: format_html

   .. grid:: 1 1 2 2
       :gutter: 2

       .. grid-item-card:: Getting started
           :link: getting-started
           :link-type: doc

            The getting started guide provides installation instructions and an
            overview on the dependencies.

       .. grid-item-card::  User guide
           :link: user-guide/index
           :link-type: doc

            The user guide provides some tutorials and examples on how to use
            ERLabPy.

       .. grid-item-card::  API reference
           :link: reference
           :link-type: doc

            The reference guide provides detailed information of the API, including
            descriptions of most available methods and parameters.

       .. grid-item-card::  Contributing guide
           :link: contributing
           :link-type: doc

            The contributing guide contains information on how to contribute to the
            project.


.. image:: images/imagetool_light.png
    :align: center
    :alt: ImageTool window in light mode
    :class: only-light

.. only:: format_html

    .. image:: images/imagetool_dark.png
        :align: center
        :alt: ImageTool window in dark mode
        :class: only-dark

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Contents

   getting-started
   user-guide/index
   reference
   contributing
   bibliography
   changelog
