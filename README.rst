CFDverify
=========

Efficient solution verification tools for computational fluid dynamics (CFD).

CFDverify is a python package designed to make verifying computational fluid dynamics (CFD) simulations easier! Solution verification is essential for trustworthy CFD results, but you need to spend your time running simulations, not fiddling with error analysis. CFDverify provides tested methods, batteries included! Check out the tutorial notebooks, use the provided utilities for easy plotting and output, or peruse the documentation to catch up on the theory. CFDverify is in early stages of development, so make sure to open an `issue <https://github.com/ORNL/cfd-verify/issues>`_ if it doesn't have a feature you need; but, be warned, the interface may change unexpectedly until the first major release!

Installation
------------

To get the latest release of CFDverify, use pip :code:`pip install cfdverify`. The latest release is hosted on the Python Package Index (PyPI) `here <https://pypi.org/project/cfdverify/>`_.

If you prefer to install from the `source <https://github.com/ORNL/cfd-verify>`_, download this repository to a suitable location on your computer, (optionally) activate your environment, and install with pip.

.. code-block:: bash

    git clone git@github.com:ORNL/cfd-verify.git
    cd cfd-verify
    source /path/to/your/venv/bin/activate
    pip install .

To install dependencies for testing the code, install with the command :code:`pip install .[tests]`. Likewise, to install documentation dependencies use the command :code:`pip install .[docs]`. Alternatively, install all optional dependencies using the command :code:`pip install .[full]`.

Documentation
-------------

Documentation is available on `Read the Docs <https://cfd-verify.readthedocs.io>`_.

To build CFDverify's documentation locally, execute the command :code:`make html` in the docs directory of CFDverify.

.. code-block:: bash

    cd cfd-verify/docs
    make html
    # or make latexpdf to generate a PDF

The documentation can then be read using any web browser by opening the file cfd-verify/docs/build/html/index.html. Install CFDverify using :code:`pip install .[docs]` to ensure you have all the required dependencies to build the documentation. If you choose to build a PDF version, you will need to make sure your system has the necessary LaTeX executables.

Testing
-------

To run CFDverify's tests, execute the command :code:`pytest` in the top level of the CFDverify directory or in the tests sub-directory. Install CFDverify using :code:`pip install .[tests]` to ensure you have all the required dependencies to run tests.

Contributing
------------

More information coming soon! For now, feel free to open an `issue <https://github.com/ORNL/cfd-verify/issues>`_ or contact the author.

Author
------

Justin Weinmeister: <weinmeistejr@ornl.gov>
