=================
Development Guide
=================

Building the documentation
==========================

--------------------
Install requirements
--------------------

.. code-block:: bash

    conda install sphinx, sphinx-autodoc-typehints, furo -y
    pip install sphinx-qt-documentation

-----
Build
-----

.. code-block:: bash

    cd docs
    make clean
    make html && make latexpdf
    cd ..




