******
Theory
******

This section includes a minimum theory to help users perform effective verification and validation. It is not a substitute for proper training on the topic. To start, several terms can be helpful to know. Verification activities evaluate the correctness of a model. Validation activities assess a model's representation of a real system. Together, these activities are used to assess the accuracy of a model and are often talked about together as verification and validation (V&V). V&V is often part of uncertainty quantification (UQ) activities, which may include uncertainty propagation and sensitivity analysis among many other types. These activities are often described together as VVUQ. 

.. note:: CFDverify makes a best effort attempt to follow the terminology defined in the *Verification, Validation, and Uncertainty Quantification Terminology in Computational Modeling and Simulation* standard published by the American Society of Mechanical Engineers (ASME) [ASME1]_. If you notice any discrepancies, please open an `issue <https://github.com/ORNL/cfd-verify/issues>`_.

.. _code_verification_theory:

Code Verification
=================

Code verification activities determine if a code is implemented correctly. There are multiple methods with varying levels of rigor that can be used to ensure a code is correct. Books which cover code verification in detail include those by Roache [Roache1998]_ or Oberkampf & Roy [Oberkampf2010]_. 

While code developers often conduct code verification to ensure their solver is bug-free, it is ultimately the responsibility of an analyst to verify their results. When available, analysts are encouraged to run verification suites on the same executable as their analysis. The test problems should also be reviewed to ensure they contain the same equations as the analysis problem, including boundary conditions, source terms, etc. when appropriate. 

Order of Accuracy
-----------------

Order of accuracy tests are the most rigorous code verification activity because they assess that the solver not only converges to the right value, but that it also converges at the formal order of accuracy. These tests often require solving the test problem on multiple meshes to observe the order of convergence. It is common to not recover the *precise* formal order, but it is often sufficient to show that it is being asymptotically approached with refinement. This can require solving more than the minimum two discretization levels to conduct this test. Order of accuracy tests should be conducted with exact solutions for their best assurance; see the following two subsections for more details on the differences between exact and approximate solutions.

Exact Solutions
^^^^^^^^^^^^^^^

Exact solutions are necessary for rigorous code verification to ensure that the code converges to the correct solution. Without an exact solution, the order of accuracy of a solver can be established, but their is no guarantee that it converges to the correct value. The partial differential equations modeled in scientific codes often do not have analytic solutions, but exact solutions can be obtained a few ways. Perhaps the most common is the method of manufactured solutions (MMS) [Roache1998]_ & [Oberkampf2010]_. 

Approximate Solutions
^^^^^^^^^^^^^^^^^^^^^

Approximate solutions are often used in code verification rather than exact solutions due to the difficulty in obtaining exact solutions. Their ubiquity should not be an endorsement, though, as approximate solutions do not provide the rigor of exact solutions. Approximate solutions can be used in CFDverify through the :py:class:`OrderOfAccuracy` class when the `relative_error` parameter is set to `True`.

Oberkampf and Roy identify truncated infinite series solutions, reduction of partial differential equations to ordinary differential equations, and benchmark solutions as three types of approximate solutions used in scientific computing [Oberkampf2010]_. When used, extra care should be taken during code verification to avoid subtle issues [Roache1998]_. In all cases analysts should ensure (1) the approximate solution is sufficiently far from the computed solutions used to compute convergence and (2) the approximate solution accurately represents the exact solution. If the approximate solution is too close to the simulation results, the computed orders will be inaccurate as the discretization distance, :math:`\Delta = h_{simulation} - f_{approximate}`, does not approximate the distance to zero discretization, :math:`\Delta = h_{simulation} - 0`. Additionally, the approximate result's accuracy must be evaluated. 

References
==========

.. [ASME1] *Verification, Validation, and Uncertainty Quantification Terminology in Computational Modeling and Simulation*, ASME VVUQ 1-2022, American Society of Mechanical Engineers, 2022.

.. [Oberkampf2010] William L. Oberkampf and Christopher J. Roy, *Verification and Validation in Scientific Computing*, Cambridge University Press, 2010.

.. [Roache1998] Patrick J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, 1998. 