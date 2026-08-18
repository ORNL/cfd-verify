Discretization
==============

The discretization module is designed to help you conduct solution verification of your CFD simulations. Below is the documentation for the available discretization objects included in CFDverify. If you are looking for the DiscretizationModel, ErrorModel, or UncertaintyModel class information, follow one of the links below.

.. toctree::
    :titlesonly:

    discretization_model
    discretization_error
    discretization_uncertainty

.. autoclass:: cfdverify.discretization.discretization_error.DiscretizationError
    :members:

.. autoclass:: cfdverify.discretization.Classic
    :members:
    :show-inheritance:

.. autoclass:: cfdverify.discretization.Average
    :members:
    :show-inheritance:

.. autoclass:: cfdverify.discretization.CustomDiscretizationError
    :members:
    :show-inheritance:
