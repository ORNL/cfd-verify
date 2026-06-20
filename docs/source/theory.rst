******
Theory
******

This section presents a brief theory to help users perform effective verification and validation. It is not a substitute for proper training, but it is helpful for first-time users. First, it is important to define verification, validation, and uncertainty quantification to delineate them. Verification activities evaluate the correctness of a model. Verification includes code verification, which evaluates that the models implemented in a code are done so without error, and solution verification, which evaluates the error of a simulation result obtained for analysis. Validation activities assess a model's representation of a real system. Together, these activities are used to assess the accuracy of a model and are often talked about together as verification and validation (V&V). V&V is often part of uncertainty quantification (UQ) activities, which may include uncertainty propagation and sensitivity analysis among many other types. Altogether, these activities are often referred to as VVUQ.

.. note:: CFDverify makes a best effort attempt to follow the terminology defined in the *Verification, Validation, and Uncertainty Quantification Terminology in Computational Modeling and Simulation* standard published by the American Society of Mechanical Engineers (ASME) [ASME1]_ except where noted. If you notice any discrepancies, please open an `issue <https://github.com/ORNL/cfd-verify/issues>`_.

.. note:: VVUQ activities are almost never exhaustively conducted for codes and simulations. Therefore, care should be taken to not use the lay terms verified or validated unless appropriate. For example, it would be inappropriate to state that a computational fluid dynamics (CFD) solver is *validated* after conducting a few validation exercises; however, it may be appropriate to say that a particular model is *validated* for a particular application if validation activities showed the model error was sufficiently small to make decisions based on the context of use. 

.. _code_verification_theory:

Code Verification
=================

Code verification activities determine if a code is implemented correctly. There are multiple methods with varying levels of rigor that can be used to ensure a code is correct. Books which cover code verification in detail include those by Roache [Roache1998]_ or Oberkampf & Roy [Oberkampf2010]_. 

While code developers often conduct code verification to ensure their solver is bug-free, it is ultimately the responsibility of an analyst to verify their results. Code verification must be run on the analysis executable to ensure test reliability, as differences between machines can mean that verification tests which pass on a developer's computer may not pass on an analysis machine. The test problems used in code verification must also contain the same equations as the analysis problem, including boundary conditions, source terms, etc. to ensure verification. Hence, please review the test problems in your code verification suite and make sure that they reflect your needs.

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


.. _solution_verification_theory:

Solution Verification
=====================

Solution verification activities identify errors in simulations. 
It is sometimes called calculation verification. 
According to [ASME]_ solution verification seeks to identify numerical errors alone; however, other sources, such as [Oberkampf2010]_ include other error sources such as mistyped values or post-processing errors in solution verification.
[ASME]_ does not have an alternative term for these error sources at this time.
As such, analysts are encouraged to be mindful of these errors sources and develop management strategies to identify, quantify, and eliminate them as needed regardless of whether they are reported as solution verification activities. 

Numerical Error
^^^^^^^^^^^^^^^

Numerical error is the error of a numerical solution in reference to the exact solution of the governing equations.
The exact solution is generally not known, so the numerical error is almost always an estimate.
Additionally, because the numerical error is only estimated, it is often converted into a numerical uncertainty when reporting solution results. 

Round-off Error
^^^^^^^^^^^^^^^

Round-off error occurs when a number is represented imprecisely by a computer. 
This most typically occurs in the representation of a decimal number using a floating point number. 
In modern computers, this is normally a double precision floating point number.

Iterative Error
^^^^^^^^^^^^^^^

Iterative error occurs when an iterative method is used to solve a problem rather than a direct method and the method is stopped before reaching machine precision (which would result in a round-off error).
In practice, it is often computationally inefficient to solve a problem to machine precision as the resulting increase in accuracy is not useful for any subsequent decision-making process that uses the results.
However, users should be cautioned that the iterative error may be much larger than they expect and care should be taken to ensure the remaining iterative error is tolerable for their application. 

.. note:: A common misconception is that the iterative residual output from a solver is the iterative error. It is not. Convergence is often asymptotic, so the relative error between two iterations is only a portion of the total iterative error between the current result and the solution.

.. warning:: Many solvers output normalized residuals (i.e., divided by the first, or one of the first, iteration's relative errors) which may be smaller than the true residual. Ensure you turn off any normalization when computing iterative errors!

Discretization Error
^^^^^^^^^^^^^^^^^^^^

Discretization error is the error due to solving a continuous equation using a discrete approximation method. 
Discretization errors occur in the dimensions of the governing equations and are most commonly the spatial and temporal dimensions of the problem.
Frequently, discretization error is the dominant numerical error term in a simulation result.
Discretization error arises in a solution  

Statistical Sampling Error
^^^^^^^^^^^^^^^^^^^^^^^^^^

Statistical sampling error is the error which occurs when a solution is computed from a finite sample of a population. 
It is commonly encountered when sampling a result from an unsteady problem or when using a stochastic solver (e.g., a Monte Carlo solver).

References
==========

.. [ASME1] *Verification, Validation, and Uncertainty Quantification Terminology in Computational Modeling and Simulation*, ASME VVUQ 1-2022, American Society of Mechanical Engineers, 2022.

.. [Oberkampf2010] William L. Oberkampf and Christopher J. Roy, *Verification and Validation in Scientific Computing*, Cambridge University Press, 2010.

.. [Roache1998] Patrick J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa Publishers, 1998. 