=================
Development Guide
=================

Code style
==========

Code is formatted using `black <https://black.readthedocs.io/en/stable/>`_. Imports are sorted using `isort <https://pycqa.github.io/isort/>`_. These are also offered as VS Code extensions `ms-python.black-formatter <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`_ and `ms-python.isort <https://marketplace.visualstudio.com/items?itemName=ms-python.isort>`_.

ERLabPy uses numpy style docstrings. However, the attribute, parameter, and return types are annotated according to `PEP 484 <https://peps.python.org/pep-0484/>`_. See `numpydoc style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and the `sphinx examples <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html>`_ for more information.

Documentation
=============

Documentation is automatically generated using `Sphinx <https://www.sphinx-doc.org/en/master/>`_.

--------------------------
Building the documentation
--------------------------

Install requirements
--------------------

.. code-block:: bash

   conda activate envname
   conda install sphinx, sphinx-autodoc-typehints, nbsphinx furo -y
   pip install sphinx-qt-documentation

Build
-----

.. code-block:: bash

   cd my/directory/erlabpy

.. code-block:: bash

   cd docs
   make clean
   make html && make latexpdf
   cd ..