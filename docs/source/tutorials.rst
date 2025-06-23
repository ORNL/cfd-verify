Tutorials
=========

.. toctree::

CFDverify has tutorials on both itself and solution verification in general! If you are familiar with solution verification, feel free to skip down to the CFDverify tutorials section.

Verification, Validation, and Uncertainty Quantification
--------------------------------------------------------

Verification, validation, and uncertainty quantification (VVUQ) is a set of activities which add credability to modeling and simualtion results. Broadly, verification is the activity of making sure the problem is solved right, validation is the activity of making sure the model represents reality, and uncertainty quantification is the activity of accounting for the natural uncertainties of systems. Often, the activities of verification and validation (V&V) are discussed separatly from uncertainty quantification (UQ). Within V&V, there are actually three activities, (1) code verification, (2) solution verification, and (3) validation. First, code verification ensures that the equations implemented in a solver are error free. Next, solution verification quantifies the error in a particular solution of a model. Finally, validation compares simulation results to experiments to quantify the model's error relative to the real world. 

Code Verification
^^^^^^^^^^^^^^^^^

CFDverify does not currently support code verification activities. Please refer to the following references for assistance. If you have ideas on how to contribute, please do!

Solution Verification
^^^^^^^^^^^^^^^^^^^^^

Solution verification is the impetus behind CFDverify. To learn more, keep reading below!

Validation
^^^^^^^^^^

CFDverify does not currently support validation activities. Please refer to the following references for assistance. If you have ideas on how to contribute, please do!

CFDverify
---------

The tutorials subdirectory contains scripts to help new users learn CFDverify. There are currently five tutorial scripts containing quick examples of how to run CFDverify for solution verification analysis. Additionally, the subdirectory contains the same scripts in Jupyter notebook form (.ipynb) if you prefer to conduct your analysis in Jupyter notebooks.

The tutorials are:

#. Quick start showing basic usage
#. Input options for discretization class construction
#. Pre-defined models available for discretization class
#. Discretization class attributes and methods of interest
#. How to build a custom discretization class with pre-defined models