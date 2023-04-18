

Development Guide
=================

Building the documentation
--------------------------

.. code-block:: bash

    sphinx-apidoc -f -o docs/source -d 3 -e -M -T erlab
    sphinx-build -b html -j auto docs/source/ docs/build/html
    sphinx-build -b latex -j auto docs/source/ docs/build/latex




