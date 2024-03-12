*****************
Development Guide
*****************

Code style
==========

Code is formatted using `black <https://black.readthedocs.io/en/stable/>`_. Imports are sorted using `isort <https://pycqa.github.io/isort/>`_. These are also offered as VS Code extensions `ms-python.black-formatter <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`_ and `ms-python.isort <https://marketplace.visualstudio.com/items?itemName=ms-python.isort>`_.

ERLabPy uses numpy style docstrings. However, the attribute, parameter, and return types are annotated according to :pep:`484`. See `numpydoc style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and the `sphinx examples <https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html>`_ for more information.

Documentation
=============

Documentation is generated using `Sphinx <https://www.sphinx-doc.org/en/master/>`_. It is automatically updated by Read the Docs when a new commit is pushed to the repository.


Building the documentation locally
----------------------------------

* Check whether all documentation dependencies are installed.

  .. code-block:: bash

      pip install -r docs/requirements.txt

* Build html and pdf documentation.
  
  .. code-block:: bash

      cd my/directory/erlabpy/docs
      make clean
      make html && make latexpdf
